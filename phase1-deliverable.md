# Phase 1 Deliverable Specification: burn-inference Offload Stack

**Status:** Draft v0.1
**Date:** 2026-04-04
**Scope:** Phase 1 only — full offloading stack behind `BackendHandle`; Gemma 4 26B MoE on Intel iGPU with 16 GB RAM

---

## Table of Contents

1. [Overview](#1-overview)
2. [Scope](#2-scope)
3. [New Components](#3-new-components)
4. [Memory Architecture](#4-memory-architecture)
5. [Data Flow](#5-data-flow)
6. [Test Specification](#6-test-specification)
7. [Acceptance Criteria](#7-acceptance-criteria)
8. [Non-Goals](#8-non-goals)
9. [Dependencies](#9-dependencies)
10. [Open Questions](#10-open-questions)

---

## 1. Overview

Phase 1 adds the full offloading stack on top of the already-validated Phase 0 engine. Phase 0 proved that the scheduler, radix cache, HTTP API, and tokenizer service are correct. Phase 1 does not revisit those components — it only changes what is behind the `BackendHandle` trait.

### Platform

- **OS:** Linux
- **GPU:** Intel Iris Xe / Arc iGPU
- **RAM:** 16 GB
- **SSD:** 256 GB+ NVMe
- **Driver stack:** Mesa 22+ / Vulkan 1.3

### Model

- **Gemma 4 26B MoE** (GGUF Q4_K_M)
- 128 experts, 8 active + 1 shared per token
- 30 transformer layers
- 256K context window
- Hybrid attention: alternating local (sliding window 1024 tokens) and global (full context) layers

### Backend

- **`burn-wgpu`** with offload layer (not ggml — ggml arrives in Phase 2A)
- Custom WGSL compute shaders for MoE routing and batched expert GEMM

### Timeline

- Weeks 7–14 (Phase 0 covered Weeks 1–6)

### What Phase 1 proves

1. **A 26B MoE model can run on 16 GB RAM** by streaming expert weights on-demand from SSD with an LRU cache of 32 hot expert slots and a ping-pong double-buffer for global KV layers.
2. **The `BackendHandle` seam holds.** Replacing `WgpuBackendHandle` with `WgpuOffloadBackendHandle` requires zero changes to `inference-engine` or `inference-api`.
3. **Prefetch overlap is real and measurable.** SSD reads for the next set of expert weights overlap GPU compute for the current layer, confirmed via tracing spans.
4. **Expert cache hit rate exceeds 75%** on typical text after warmup, meaning most tokens never wait for an SSD read on their hot experts.
5. **The `PrefetchOps` default no-op does not regress Phase 0.** The new trait extension is purely additive.

### What is unchanged from Phase 0

| Component | Status |
|-----------|--------|
| `inference-engine` crate | Unchanged — no modifications |
| `inference-api` crate | Unchanged — no modifications |
| `TokenizerService` | Unchanged |
| `RadixCache` | Unchanged |
| Scheduler (`schedule_batch`, `run_overlapped_loop`) | Unchanged |
| `BackendHandle` trait (existing methods) | Unchanged — `prefetch` default no-op extended only |
| `StubBackend` | Unchanged |
| All Phase 0 cargo tests | Must continue to pass |

---

## 2. Scope

### In scope

- New crate `inference-backend-wgpu-offload`: contains `WgpuOffloadBackendHandle`, `WeightCache<K>`, `KvOffloadManager`, `GgufIndex`, `PrefetchOps` trait extension
- New crate `inference-model-gemma4`: `Gemma4MoeModel`, `GemmaRunner`, `Gemma4Config`
- WGSL shaders: `shaders/moe_router.wgsl`, `shaders/mul_mat_id.wgsl`
- `PrefetchOps` trait addition to `burn-backend` (new file `ops/prefetch.rs`; default no-op; does not alter existing trait objects)
- All 7 correctness/behaviour tests passing (see Section 6), plus all Phase 0 tests unmodified
- All performance targets met on target hardware (see Section 7)

### Explicitly out of scope

- Any change to `inference-engine`, `inference-api`, or `inference-backend` crates (the `BackendHandle` seam must not move)
- `burn-ggml` backend — Phase 2A
- Prefill-decode disaggregation — Phase 2
- Multi-GPU or distributed inference — not planned
- ANE / Intel NPU — not planned
- Models other than Gemma 4 26B MoE — deferred
- Quantization schemes other than Q4_K_M — deferred
- Prometheus metrics endpoint — Phase 1 stretch goal only (not a hard gate)
- Priority scheduling beyond FCFS — unchanged from Phase 0
- Windows or macOS support for Phase 1 — Linux only

---

## 3. New Components

### 3.1 `WgpuOffloadBackendHandle`

**Location:** `inference-backend-wgpu-offload/src/handle.rs`

**Purpose:** Replaces `WgpuBackendHandle` from Phase 0 as the concrete implementation of `BackendHandle` for the Gemma 4 26B MoE model. It owns all offload subsystems and delegates GPU execution to `GemmaRunner`.

```rust
pub struct WgpuOffloadBackendHandle {
    runner:       Arc<GemmaRunner>,
    expert_cache: Arc<ExpertWeightCache>,
    kv_manager:   Arc<KvOffloadManager>,
    kv_pool:      Arc<WgpuKvPool>,
    model_cfg:    ModelConfig,
}

impl BackendHandle for WgpuOffloadBackendHandle {
    fn forward(&self, batch: &Batch)
        -> impl Future<Output = Result<Logits, EngineError>> + Send;

    fn kv_pool(&self) -> &dyn KvPool;

    fn model_config(&self) -> &ModelConfig;

    /// Fires async prefetch for the expert weights the scheduler predicts
    /// will be needed in the next forward pass.  Called by the engine loop
    /// immediately after `schedule_batch` returns, before the GPU thread
    /// picks up the batch.  Default no-op from BackendHandle is overridden here.
    fn prefetch(&self, batch: &Batch);
}
```

**Lifecycle:**

1. `WgpuOffloadBackendHandle::new(model_path, cfg)` — opens GGUF file via `GgufIndex`, loads non-expert weights into GPU memory, pre-allocates N=32 expert slots in RAM, creates ping/pong KV buffers on SSD.
2. On each `forward` call: `GemmaRunner::decode_step` drives layer-by-layer execution. After routing at each MoE layer, `expert_cache.prefetch(top_k_keys)` fires immediately so SSD reads overlap the current layer's matmuls.
3. On `prefetch` call (from engine loop): fires `expert_cache.prefetch` for any experts likely needed in the next batch iteration based on routing statistics from the previous step.

---

### 3.2 `WeightCache<K: CacheKey>` — Unified Generic Cache

**Location:** `inference-backend-wgpu-offload/src/weight_cache.rs`

**Purpose:** Generic LRU cache of weight tensors resident in host RAM. On cache miss, reads from the GGUF mmap via `GgufIndex` using `pread`-style access and transfers into the slot. On eviction, the slot's memory is reused without deallocation.

#### Key types

```rust
pub trait CacheKey: Hash + Eq + Clone + Send + 'static {}

pub struct ExpertKey {
    pub layer_idx:  usize,
    pub expert_idx: usize,
}

pub struct LayerKey {
    pub layer_idx: usize,
}

impl CacheKey for ExpertKey {}
impl CacheKey for LayerKey  {}

pub type ExpertWeightCache = WeightCache<ExpertKey>;
pub type LayerWeightCache  = WeightCache<LayerKey>;
```

#### `WeightCache<K>`

```rust
pub struct WeightCache<K: CacheKey> {
    model_path: PathBuf,
    max_slots:  usize,
    /// LRU map: key → slot.  Protected by a Tokio mutex so async tasks can
    /// await a miss without blocking the executor thread.
    slots: tokio::sync::Mutex<LruCache<K, WeightSlot>>,
    /// Byte-offset lookup table parsed from GGUF at startup.
    index: Arc<dyn CacheIndex<K>>,
}

impl<K: CacheKey> WeightCache<K> {
    /// Async get.  Returns immediately on hit.  On miss, reads from GGUF
    /// and loads the slot before returning.  The returned `WeightGuard`
    /// pins the slot for the duration of the forward pass.
    pub async fn get(&self, key: K) -> Result<Arc<WeightGuard>, CacheError>;

    /// Fire-and-forget prefetch.  Spawns a Tokio task that calls `get`
    /// for each key.  Keys already resident are no-ops.  Keys already
    /// being fetched by another task deduplicate via the slot's
    /// `Notify`-based in-flight tracker.
    pub fn prefetch(&self, keys: &[K]);
}
```

#### `WeightSlot`

```rust
pub struct WeightSlot {
    /// Host-pinned buffer, `max_slots`-pre-allocated at startup.
    pub data:   Vec<u8>,
    pub state:  SlotState,
    pub notify: Arc<tokio::sync::Notify>,
}

pub enum SlotState {
    Empty,
    Loading,
    Ready { key: ExpertKey, generation: u64 },
}
```

#### `WeightGuard`

```rust
/// RAII guard: prevents the slot from being evicted while the GPU is
/// consuming the weight.  Drop signals the cache the slot is available
/// for eviction again.
pub struct WeightGuard {
    slot_ptr: Arc<WeightSlot>,
    _pin:     PhantomData<*mut ()>,  // !Send intentionally — must be used on one thread
}

impl WeightGuard {
    pub fn as_bytes(&self) -> &[u8];
}
```

#### `CacheIndex<K>` trait

```rust
pub trait CacheIndex<K: CacheKey>: Send + Sync + 'static {
    /// Returns the byte offset and length within the GGUF mmap for `key`.
    fn lookup(&self, key: &K) -> Option<ByteRange>;
}

pub struct ByteRange {
    pub offset: u64,
    pub len:    usize,
}
```

**Implementation notes:**

- `max_slots` is configurable; default N=32 for `ExpertWeightCache` (~14 MB per expert × 32 = ~448 MB).
- LRU eviction: when all slots are pinned, `get` blocks until a slot is unpinned. This is the back-pressure mechanism that prevents the GPU from racing too far ahead of the SSD.
- Prefetch deduplication: slot state `Loading` plus a `Notify` allows multiple callers to await the same in-flight load without spawning duplicate I/O.
- All I/O uses `tokio::task::spawn_blocking` wrapping `pread(2)` on the GGUF file descriptor.

---

### 3.3 `KvOffloadManager` — Ping-Pong SSD KV for Global Attention Layers

**Location:** `inference-backend-wgpu-offload/src/kv_offload.rs`

**Purpose:** Gemma 4 26B has 5 global attention layers (layers 0, 6, 12, 18, 24 — exact indices TBD from model config). Each global layer needs the full 256K-token KV cache, which is too large to keep in RAM for all layers simultaneously. `KvOffloadManager` keeps two RAM buffers (ping and pong) and streams each global layer's KV to/from SSD with double-buffering so compute and I/O overlap.

```rust
pub struct KvOffloadManager {
    /// Directory on NVMe where per-layer KV files are stored.
    cache_dir:         PathBuf,
    num_global_layers: usize,
    /// One file per global layer, pre-allocated at startup.
    files:             Mutex<Vec<KvLayerFile>>,
    /// The buffer currently in use by the GPU for the active layer.
    ping:              Arc<KvBuffer>,
    /// The buffer being filled by async SSD read for the next layer.
    pong:              Arc<KvBuffer>,
}

impl KvOffloadManager {
    /// Swap ping/pong and return the now-active buffer for `global_layer_idx`.
    /// If the pong prefetch for this layer has not completed, awaits it.
    pub async fn swap_and_get(
        &self,
        global_layer_idx: usize,
    ) -> Result<Arc<KvBuffer>, OffloadError>;

    /// Fire-and-forget: begin reading `next_global_layer_idx` KV data from SSD
    /// into the pong buffer.  Safe to call while ping is in use by the GPU.
    pub fn prefetch_next(&self, next_global_layer_idx: usize);

    /// Asynchronously write `buf` back to SSD for `global_layer_idx`.
    /// Called after the GPU has updated the KV cache for a layer.
    /// Uses `tokio::task::spawn_blocking` + `pwrite(2)`.
    pub fn writeback_async(&self, global_layer_idx: usize, buf: Arc<KvBuffer>);
}
```

#### `KvBuffer`

```rust
pub struct KvBuffer {
    /// Host-pinned RAM buffer, pre-allocated at startup.
    /// Size = num_heads × head_dim × max_seq_len × sizeof(INT8) × 2 (K+V).
    pub data:  Vec<u8>,
    pub state: KvBufferState,
    pub ready: Arc<tokio::sync::Notify>,
}

pub enum KvBufferState {
    Empty,
    Loading { layer_idx: usize },
    Ready   { layer_idx: usize },
    InUse   { layer_idx: usize },
}
```

#### `KvLayerFile`

```rust
pub struct KvLayerFile {
    pub path: PathBuf,
    pub fd:   std::fs::File,
    /// Pre-allocated file size = full KV footprint for this layer.
    pub size: u64,
}
```

**Ping-pong timeline:**

```
Time ──►

Layer N (global):  [GPU: matmul+attn] ──────────────► [GPU: writeback trigger]
                                       ↑
SSD read (layer N+1):                  [pread into pong] ──────►
                                                                 ↓
Swap:                                                           [swap ping/pong]
Layer N+1 (global): [GPU: matmul+attn] ──────────────►
...
```

The prefetch for layer N+1 is fired immediately after routing for layer N completes and the GPU begins the layer N attention matmul. This hides ~200–400 ms of SSD latency behind ~500–800 ms of GPU compute at 5 tok/s decode.

---

### 3.4 WGSL Compute Shaders

#### 3.4.1 `shaders/moe_router.wgsl`

**Purpose:** Given the router logit vector (128 values), compute softmax over all 128 experts, then select the top-8 expert indices and their normalized weights.

**Workgroup layout:** 128 threads × 1 × 1 (one thread per expert).

**Algorithm:**

1. Each thread loads its logit from `router_logits` input buffer into shared memory.
2. Parallel max reduction over 128 shared-memory values to find the global max (for numerically stable softmax).
3. Each thread computes `exp(logit[i] - max)` and stores to shared memory.
4. Parallel sum reduction to compute the partition function Z.
5. Each thread computes `softmax[i] = exp(logit[i] - max) / Z`.
6. Parallel top-8 selection via partial sort in shared memory (128 → 16 → 8 steps).
7. Write `top_k_indices: array<u32, 8>` and `top_k_weights: array<f32, 8>` to output buffer.

**Subgroup fast path:**

```wgsl
// Enabled when Features::SUBGROUP is advertised (Intel ANV driver, Mesa 22+).
// Uses subgroup_max and subgroup_add for steps 2 and 4 without shared memory.
// Falls back to shared-memory path unconditionally if subgroups unavailable.
#ifdef USE_SUBGROUPS
  var max_val = subgroupMax(logit);
  var sum_val = subgroupAdd(exp(logit - max_val));
#else
  // shared memory reduction
#endif
```

**Bindings:**

| Binding | Type | Description |
|---------|------|-------------|
| `@group(0) @binding(0)` | `array<f32, 128>` | Router logits input |
| `@group(0) @binding(1)` | `array<u32, 8>` | Top-K indices output |
| `@group(0) @binding(2)` | `array<f32, 8>` | Top-K weights output |

**Push constants:** none (all dimensions are compile-time constants: 128 experts, top-8).

#### 3.4.2 `shaders/mul_mat_id.wgsl`

**Purpose:** Expert-indexed batched GEMM. For each token in the batch, multiply the token's input vector by the weight matrix of the expert assigned to that token. Equivalent to `ggml`'s `GGML_OP_MUL_MAT_ID`.

**Signature (conceptual):**

```
out[token_idx] = weight[expert_id[token_idx]] @ input[token_idx]
```

where `weight` is a 3-D array `[num_experts, N, K]` and `input` is `[n_tokens, K]`.

**Tiling:** 16 × 16 workgroup tiles over the output matrix (N rows × M tokens).

**Bindings:**

| Binding | Type | Description |
|---------|------|-------------|
| `@group(0) @binding(0)` | `array<f32>` | Packed weight buffer `[num_experts × N × K]` |
| `@group(0) @binding(1)` | `array<f32>` | Input activations `[n_tokens × K]` |
| `@group(0) @binding(2)` | `array<u32>` | Expert ID per token `[n_tokens]` |
| `@group(0) @binding(3)` | `array<f32>` | Output `[n_tokens × N]` |

**Push constants:**

```wgsl
struct PushConstants {
    M:             u32,   // n_tokens
    N:             u32,   // output dim (expert hidden dim)
    K:             u32,   // input dim
    expert_stride: u32,   // N * K in elements
    n_tokens:      u32,   // same as M; kept separate for clarity
}
```

**Reference:** Port from `ggml-vulkan.cpp` `GGML_OP_MUL_MAT_ID` compute shader, translated to WGSL syntax and adapted for `burn-wgpu` dispatch conventions. The tiling strategy and shared-memory layout are identical to the ggml-vulkan reference implementation.

---

### 3.5 `Gemma4MoeModel` — Burn Module Definition

**Location:** `inference-model-gemma4/src/model.rs`

**Purpose:** Full Gemma 4 26B transformer as a Burn module. Consumes token IDs, paged KV cache handles, and a batch descriptor; produces logits.

#### Architecture summary

- 30 transformer layers, indexed 0–29
- Each layer contains:
  - Pre-attention RMSNorm
  - Hybrid attention (local or global depending on `layer_idx % 6 == 0`)
  - Post-attention RMSNorm
  - MoE FFN (128 experts, 8 active + 1 shared, GeGLU gating)
  - Post-FFN RMSNorm
- Token embedding table + final lm_head projection

#### Hybrid attention

```rust
pub enum AttentionKind {
    /// Sliding window of 1024 tokens; uses only local KV cache pages.
    Local { window: usize },
    /// Full causal attention over the entire context; uses SSD-backed KV.
    Global,
}

impl Gemma4Layer {
    pub fn attention_kind(layer_idx: usize) -> AttentionKind {
        if layer_idx % 6 == 0 {
            AttentionKind::Global
        } else {
            AttentionKind::Local { window: 1024 }
        }
    }
}
```

#### MoE FFN

```rust
pub struct MoeFfn<B: Backend> {
    pub router:        Linear<B>,          // hidden_dim → 128 logits
    pub shared_expert: FeedForward<B>,     // always active
    pub expert_gates:  Vec<Linear<B>>,     // 128 × (hidden → intermediate, gate proj)
    pub expert_downs:  Vec<Linear<B>>,     // 128 × (intermediate → hidden)
}

impl<B: Backend> MoeFfn<B> {
    pub fn forward(
        &self,
        x:            Tensor<B, 2>,        // [n_tokens, hidden_dim]
        expert_cache: &ExpertWeightCache,
        device:       &B::Device,
    ) -> Tensor<B, 2>;                     // [n_tokens, hidden_dim]
}
```

The `forward` method:
1. Calls `moe_router.wgsl` to get `top_k_indices` and `top_k_weights` for each token.
2. Calls `expert_cache.prefetch(top_8_keys)` immediately after routing (fire-and-forget for next layer).
3. Awaits `expert_cache.get(key)` for each of the 8 active experts.
4. Dispatches `mul_mat_id.wgsl` for the batched expert GEMM.
5. Adds shared expert output.
6. Combines with `top_k_weights` coefficients.

#### `GemmaRunner`

```rust
pub struct GemmaRunner {
    model:        Gemma4MoeModel<Wgpu>,
    expert_cache: Arc<ExpertWeightCache>,
    kv_manager:   Arc<KvOffloadManager>,
    device:       WgpuDevice,
}

impl GemmaRunner {
    /// Execute one forward pass for a batch.  Drives layer-by-layer execution
    /// with explicit prefetch scheduling interleaved with GPU compute.
    pub async fn decode_step(
        &self,
        batch:    &Batch,
        kv_pool:  &WgpuKvPool,
    ) -> Result<Logits, EngineError>;
}
```

**Prefetch schedule in `decode_step`:**

```
for layer_idx in 0..30:
    if layer is Global:
        kv_manager.prefetch_next(layer_idx + 1)   // overlap I/O with this layer's compute
        kv_buf = kv_manager.swap_and_get(layer_idx).await
        run_global_attention(layer_idx, kv_buf)
        kv_manager.writeback_async(layer_idx, kv_buf)
    else:
        run_local_attention(layer_idx)

    // After MoE routing for this layer, prefetch experts for same layer next token
    // (next call to decode_step) — routing distribution is autocorrelated
    expert_cache.prefetch(top_k_keys_for_next_step)
    await expert matmuls for current step
```

---

### 3.6 `GgufIndex` — GGUF File Parser

**Location:** `inference-backend-wgpu-offload/src/gguf_index.rs`

**Purpose:** Open a GGUF file, parse the header and tensor index, expose byte-range lookups for named tensors. Uses `memmap2` for zero-copy access to tensor data.

```rust
pub struct GgufIndex {
    pub metadata: HashMap<String, GgufValue>,
    pub tensors:  HashMap<String, TensorInfo>,
    mmap:         Mmap,  // memory-mapped GGUF file; kept alive for the process lifetime
}

impl GgufIndex {
    /// Open and parse a GGUF file.  Validates the magic bytes and version.
    /// Reads the full header into `metadata` and `tensors`.  Does not load
    /// any tensor data into RAM — data is accessed on-demand via `mmap`.
    pub fn open(path: &Path) -> Result<Self, GgufError>;

    /// Return a byte slice into the mmap for the named tensor.
    /// Returns `None` if the tensor name is not in the index.
    pub fn tensor_bytes(&self, name: &str) -> Option<&[u8]>;

    /// Return byte offset + length for each of the 128 expert weight pairs
    /// (gate_proj + up_proj + down_proj) in `layer`.
    /// Ordered by expert index 0–127.
    pub fn expert_offsets(&self, layer: usize) -> Vec<ExpertOffsetInfo>;

    /// Return byte offset + length for non-expert weights in each layer
    /// (attention projections, norms).
    pub fn layer_offsets(&self) -> Vec<LayerOffsetInfo>;
}
```

```rust
pub struct TensorInfo {
    pub name:   String,
    pub shape:  Vec<u64>,
    pub dtype:  GgufDtype,
    pub offset: u64,   // byte offset from start of mmap data region
    pub size:   u64,   // byte length
}

pub struct ExpertOffsetInfo {
    pub expert_idx: usize,
    pub gate_proj:  ByteRange,
    pub up_proj:    ByteRange,
    pub down_proj:  ByteRange,
}

pub struct LayerOffsetInfo {
    pub layer_idx:   usize,
    pub q_proj:      ByteRange,
    pub k_proj:      ByteRange,
    pub v_proj:      ByteRange,
    pub o_proj:      ByteRange,
    pub pre_norm:    ByteRange,
    pub post_norm:   ByteRange,
}

pub enum GgufValue {
    U8(u8), U16(u16), U32(u32), U64(u64),
    I8(i8), I16(i16), I32(i32), I64(i64),
    F32(f32), F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}
```

**Implementation notes:**

- Supports GGUF version 3 (the version used by llama.cpp for Gemma 4 exports). Version 2 support is a stretch goal.
- `tensor_bytes` is the fast path: it returns a `&[u8]` slice directly from the mmap. No copy is made. The caller is responsible for dequantization.
- `expert_offsets` uses the naming convention `blk.{layer}.ffn_gate_exps.{expert}`, `blk.{layer}.ffn_up_exps.{expert}`, `blk.{layer}.ffn_down_exps.{expert}` (llama.cpp convention for Gemma 4 MoE GGUF exports).

---

### 3.7 `PrefetchOps` Trait Addition to Burn

**Location:** `burn-backend/src/ops/prefetch.rs` (new file in upstream burn fork or vendored copy)

**Purpose:** A minimal extension point that lets a `burn-wgpu` backend implementation signal to the runtime that certain tensors should be prefetched from slow memory. The default implementation is a no-op so that all existing `Backend` implementations continue to compile and pass tests unchanged.

```rust
// burn-backend/src/ops/prefetch.rs

/// Describes a set of tensors that should be prefetched from slow storage
/// (SSD, HBM offload) into GPU-accessible memory before they are needed.
pub struct PrefetchPrimitive<B: Backend> {
    pub floats: Vec<FloatTensor<B>>,
    pub ints:   Vec<IntTensor<B>>,
}

/// Optional prefetch hint trait.  Default implementation is a no-op so that
/// every existing Backend implementation continues to work without changes.
pub trait PrefetchOps<B: Backend> {
    fn prefetch(primitive: PrefetchPrimitive<B>, device: &B::Device) {
        // Default: discard the hint.  No-op on all Phase 0 backends.
        let _ = (primitive, device);
    }
}
```

**Integration with `BackendHandle::prefetch`:**

The `BackendHandle::prefetch` method (already present in Phase 0 as a no-op) calls `PrefetchOps::prefetch` on the wgpu backend. Because the default `PrefetchOps` implementation is a no-op, `StubBackend` and `WgpuBackendHandle` continue to work exactly as before.

`WgpuOffloadBackendHandle` overrides `BackendHandle::prefetch` to call `expert_cache.prefetch(predicted_expert_keys)`, bypassing `PrefetchOps` entirely (the prefetch logic lives above the Burn abstraction layer).

---

## 4. Memory Architecture

### 4.1 16 GB RAM Budget

```
Component                           Size         Notes
─────────────────────────────────── ──────────── ──────────────────────────────────────
OS + runtime                        ~1.5 GB      kernel, Mesa, Vulkan loader, Tokio
Non-expert weights (in RAM)         ~1.5 GB      attention Q/K/V/O projections,
                                                 embed table, norms (all layers)
Expert hot cache (N=32 slots)       ~0.45 GB     32 × ~14 MB per expert (Q4_K_M)
Local attention KV (in RAM)         ~0.25 GB     25 local layers × ~10 MB each
                                                 (INT8, 1024-token window)
Ping buffer (global KV)             ~1.1 GB      1 global layer full KV, INT8,
                                                 256K context, 8 heads
Pong buffer (global KV)             ~1.1 GB      prefetch slot for next global layer
Activation / scratch                ~0.5 GB      wgpu intermediate buffers,
                                                 dequant workspace
Headroom                            ~1.0 GB      for OS file cache, burst spikes
─────────────────────────────────── ──────────── ──────────────────────────────────────
Total used                          ~7.45 GB     fits comfortably in 16 GB
```

**Note:** The wgpu iGPU shares the system RAM pool. The iGPU's VRAM is a sub-region of the 16 GB. The budget above accounts for both CPU-side and GPU-side usage combined, because on Intel iGPU there is no separate VRAM — all GPU allocations come from the same 16 GB pool.

### 4.2 SSD Layout

```
/path/to/model.gguf         ~67 GB    Q4_K_M weights (non-expert + expert)
  └─ expert weights          ~53 GB   128 experts × 30 MoE layers × ~14 MB
  └─ non-expert weights      ~14 GB   attention, embed, norms (all layers)

/path/to/kv_cache/          ~14 GB    SSD KV cache, pre-allocated at startup
  ├─ layer_000.kvcache       ~2.8 GB  global layer 0 (256K ctx × INT8)
  ├─ layer_006.kvcache       ~2.8 GB  global layer 6
  ├─ layer_012.kvcache       ~2.8 GB  global layer 12
  ├─ layer_018.kvcache       ~2.8 GB  global layer 18
  └─ layer_024.kvcache       ~2.8 GB  global layer 24
```

KV cache files are pre-allocated at startup via `fallocate(2)` so that later `pwrite` calls do not trigger file system allocation faults on the critical path.

### 4.3 Ping-Pong KV Timeline

The following diagram shows how one decode iteration processes the 5 global layers (G0, G6, G12, G18, G24) interleaved with SSD I/O:

```
Decode iteration timeline (one token, 30 layers):

Layer:  L1  L2  L3  L4  L5 [G0]  L7  L8  L9  L10 L11 [G6] ...

GPU:    ████████████████████[████]████████████████████[████]...
                            G0 attn                   G6 attn

SSD→RAM:                    [════G6 read═════]
                                              [═════G12 read════]

RAM→SSD:                              [G0 wb]
                                                        [G6 wb]

Legend:
  ████  GPU compute (local attention + MoE FFN)
  [██]  GPU compute for global attention layer
  [══]  SSD pread into pong buffer (async, overlaps GPU)
  [wb]  SSD pwrite writeback (async, fire-and-forget)
```

Key property: by the time the GPU finishes G0's attention and the subsequent local layers L1–L5, the pong buffer for G6 is already filled. The `swap_and_get` call for G6 returns immediately without blocking on I/O.

---

## 5. Data Flow

### 5.1 Request Lifecycle (Phase 1)

```
HTTP POST /v1/chat/completions
        │
        ▼
TokenizerService (unchanged from Phase 0)
  encode prompt → token IDs
        │
        ▼
engine_loop / scheduler (unchanged from Phase 0)
  RadixCache prefix match
  KvCachePool page allocation
  schedule_batch() → Batch
        │
        ├──► BackendHandle::prefetch(batch)
        │         │
        │         ▼
        │    WgpuOffloadBackendHandle::prefetch()
        │    expert_cache.prefetch(predicted_keys)
        │    [fire-and-forget Tokio tasks for pread]
        │
        ▼
BackendHandle::forward(batch)
        │
        ▼
GemmaRunner::decode_step(batch, kv_pool)
        │
        ├─ for each layer 0..29:
        │
        │   [if local layer]
        │   ├─ run_local_attention(layer_idx, kv_pool)   ← GPU, paged KV in RAM
        │   │
        │   [if global layer]
        │   ├─ kv_manager.prefetch_next(layer_idx + 1)   ← fire async SSD read
        │   ├─ kv_buf = kv_manager.swap_and_get(layer_idx).await
        │   ├─ run_global_attention(layer_idx, kv_buf)   ← GPU, KV from ping buffer
        │   └─ kv_manager.writeback_async(layer_idx, kv_buf)
        │
        │   [MoE FFN for every layer]
        │   ├─ moe_router.wgsl → top_k_indices, top_k_weights
        │   ├─ expert_cache.prefetch(next_step_keys)     ← fire-and-forget
        │   ├─ await expert_cache.get(key) × 8           ← hit or wait for pread
        │   ├─ mul_mat_id.wgsl                           ← GPU, batched expert GEMM
        │   └─ shared_expert forward + combine
        │
        ▼
Logits [n_output_rows, vocab_size]
        │
        ▼
engine_loop: sample next token (argmax / top-p / top-k)
        │
        ▼
TokenizerService: decode token ID → text piece
        │
        ▼
SSE stream → HTTP response
```

### 5.2 Expert Cache State Machine (per slot)

```
        ┌───────────────────────────────────────┐
        │                                       │
  prefetch(key)                           eviction (LRU)
        │                                       │
        ▼                                       │
  ┌─────────┐   pread done    ┌─────────┐       │
  │ Loading │────────────────►│  Ready  │───────┘
  └─────────┘                 └─────────┘
                                   │   ▲
                          get(key) │   │ WeightGuard::drop()
                                   │   │
                                   ▼   │
                               ┌────────┐
                               │ InUse  │
                               └────────┘
```

- A slot in `Loading` state is not eligible for eviction.
- A slot in `InUse` state (held by a `WeightGuard`) is not eligible for eviction.
- Only `Ready` slots can be evicted. If all slots are either `Loading` or `InUse`, `get` blocks.

---

## 6. Test Specification

All Phase 0 tests must continue to pass without modification. The following additional tests are required for Phase 1.

### 6.1 Semantic Correctness — Capital of China

**Test name:** `test_gemma4_capital_of_china`

**Preconditions:** Gemma 4 26B MoE Q4_K_M GGUF weights present at the path specified by the `GEMMA4_GGUF_PATH` environment variable. Test is skipped if the variable is not set.

**Hardware:** Linux, Intel Iris Xe / Arc, Mesa 22+ / Vulkan 1.3.

**Procedure:**
1. Construct `WgpuOffloadBackendHandle` from the GGUF path.
2. Send prompt `"What is the capital of China?"` through the full engine stack (tokenizer → scheduler → forward → sample → detokenize).
3. Collect the complete output string (stop on EOS or 64 tokens).

**Assertion:**
```rust
assert!(
    output.to_lowercase().contains("beijing"),
    "Expected 'beijing' in output, got: {:?}", output
);
```

---

### 6.2 Semantic Correctness — Arithmetic

**Test name:** `test_gemma4_one_plus_one`

**Preconditions:** Same as 6.1.

**Procedure:**
1. Send prompt `"What is 1+1? Answer with only a number."`.
2. Collect output (stop on EOS or 8 tokens).

**Assertion:**
```rust
assert_eq!(
    output.trim(),
    "2",
    "Expected exactly '2', got: {:?}", output
);
```

---

### 6.3 Expert Cache Correctness — Forced Eviction

**Test name:** `test_expert_cache_eviction_correctness`

**Preconditions:** Gemma 4 26B MoE GGUF weights. Does not require GPU — uses a mock GPU backend that passes expert weight bytes through a checksum function instead of running a real matmul.

**Procedure:**
1. Construct `ExpertWeightCache` with `max_slots = 4` (forces heavy eviction for a 30-layer, 128-expert model).
2. Construct `ExpertWeightCache` with `max_slots = 256` (all-resident baseline).
3. Run a fixed sequence of 20 `get(expert_key)` calls (same key sequence for both caches), recording the bytes returned by each call.
4. Compare the byte slices returned by the N=4 cache against those returned by the N=256 cache for each call.

**Assertion:**
```rust
for (i, (evict_bytes, reference_bytes)) in evict_results.iter().zip(reference_results.iter()).enumerate() {
    assert_eq!(
        evict_bytes, reference_bytes,
        "Cache eviction correctness failure at access {}: bytes differ", i
    );
}
```

---

### 6.4 KV Ping-Pong Correctness

**Test name:** `test_kv_pingpong_output_matches_ram_reference`

**Preconditions:** Gemma 4 26B MoE GGUF weights. Requires a GPU to run actual attention kernels.

**Procedure:**
1. Construct a `KvOffloadManager` backed by a temporary directory.
2. Run global attention for layers 0, 6, 12, 18, 24 using the ping-pong manager with a 128-token prompt.
3. Run the same global attention layers using `KvBuffers` kept entirely in RAM (no SSD involvement).
4. Compare the output activation tensors from both runs.

**Assertion:**
```rust
for (layer_idx, (offload_out, reference_out)) in offload_outputs.iter().zip(reference_outputs.iter()).enumerate() {
    let max_abs_diff = (offload_out - reference_out).abs().max();
    assert!(
        max_abs_diff < 1e-4,
        "KV ping-pong output mismatch at global layer {}: max_abs_diff = {}", layer_idx, max_abs_diff
    );
}
```

---

### 6.5 Prefetch Overlap — SSD Reads Overlap GPU Compute

**Test name:** `test_prefetch_overlap_confirmed`

**Preconditions:** Gemma 4 26B MoE GGUF weights. Requires Linux with a real NVMe SSD.

**Procedure:**
1. Instrument `GemmaRunner::decode_step` with `tracing` spans:
   - `span_gpu_matmul`: wraps each layer's GPU matmul
   - `span_ssd_read`: wraps each `pread` call in `KvOffloadManager` and `WeightCache`
2. Run a 512-token prompt decode. Collect all span start/end timestamps.
3. For each `span_ssd_read`, verify that at least one `span_gpu_matmul` started before the SSD read finished (i.e., the intervals overlap).

**Assertion:**
```rust
let overlapping_count = count_overlapping_spans(&gpu_spans, &ssd_spans);
assert!(
    overlapping_count > 0,
    "No SSD reads overlapped with GPU compute — prefetch is not working"
);
```

---

### 6.6 Expert Cache Hit Rate Benchmark

**Test name:** `bench_expert_cache_hit_rate`

**Preconditions:** Gemma 4 26B MoE GGUF weights.

**Procedure:**
1. Run 5 warmup decode iterations on a fixed prompt (`"The quick brown fox"` extended to 128 tokens).
2. Run 100 measurement decode iterations on the same prompt continuation.
3. Record `ExpertWeightCache` hit count and miss count over the measurement period.

**Assertion:**
```rust
let hit_rate = hit_count as f64 / (hit_count + miss_count) as f64;
assert!(
    hit_rate > 0.75,
    "Expert cache hit rate {:.1}% is below the 75% target", hit_rate * 100.0
);
```

---

### 6.7 MUL_MAT_ID Correctness — Against Reference Matmuls

**Test name:** `test_mul_mat_id_matches_reference`

**Preconditions:** GPU required for wgpu dispatch.

**Procedure:**
1. Construct a synthetic weight matrix `W` of shape `[8, 256, 512]` (8 experts, 256 output, 512 input) with random float values.
2. Construct a token input matrix `X` of shape `[16, 512]` (16 tokens).
3. Assign each token a random expert index in `[0, 8)`.
4. Run `mul_mat_id.wgsl` to produce `out_batched[16, 256]`.
5. Compute reference: for each token, call `out_reference[i] = W[expert_id[i]] @ X[i]` using a standard Burn `matmul` op.
6. Compare element-wise.

**Assertion:**
```rust
let max_abs_diff = (out_batched - out_reference).abs().max_element();
assert!(
    max_abs_diff < 1e-3,
    "mul_mat_id output deviates from reference by {}", max_abs_diff
);
```

---

### 6.8 PrefetchOps No-Op Regression

**Test name:** `test_prefetchops_noop_does_not_break_phase0`

**Preconditions:** No GPU or weights required. Uses `StubBackend`.

**Procedure:**
1. Call `PrefetchOps::prefetch` on `StubBackend` with an empty `PrefetchPrimitive`.
2. Verify the call returns without panic.
3. Run all Phase 0 scheduler unit tests with the `PrefetchOps` extension compiled in.

**Assertion:** All Phase 0 tests pass. `prefetch` call on `StubBackend` returns `()` without error.

---

## 7. Acceptance Criteria

All of the following must be satisfied before Phase 1 is considered complete. Each item is a hard gate — partial credit is not accepted.

### Correctness

- [ ] **Phase 0 regression-free.** `cargo test --workspace` passes on a clean Linux checkout without modification to `inference-engine` or `inference-api`.
- [ ] **Semantic test 6.1 passes.** Gemma 4 26B MoE produces output containing "beijing" for the capital-of-China prompt.
- [ ] **Semantic test 6.2 passes.** Gemma 4 26B MoE produces output `"2"` for the 1+1 prompt.
- [ ] **Expert cache correctness (6.3) passes.** N=4 eviction cache produces byte-identical output to N=256 all-resident cache for the same key sequence.
- [ ] **KV ping-pong correctness (6.4) passes.** SSD-backed global KV outputs match RAM-resident reference within tolerance 1e-4.
- [ ] **MUL_MAT_ID correctness (6.7) passes.** Batched expert GEMM matches 8 separate reference matmuls within tolerance 1e-3.
- [ ] **PrefetchOps no-op (6.8) passes.** All Phase 0 tests pass with `PrefetchOps` extension compiled in.

### Performance

| Metric | Target | Gate? |
|--------|--------|-------|
| TTFT, batch 1, cold cache, 512-token prompt | < 4 s | Hard |
| TTFT, batch 1, warm radix cache, 512-token prompt | < 0.5 s | Hard |
| Decode throughput, batch 1 | ≥ 5 tok/s | Hard |
| Decode throughput, batch 4 | ≥ 15 tok/s | Hard |
| Expert cache hit rate (typical text, post-warmup) | > 75% | Hard |
| Scheduler overhead per iteration | < 1 ms | Hard |
| RAM usage during inference | < 10 GB | Hard |
| SSD reads overlap GPU compute | confirmed (test 6.5) | Hard |

### Architecture invariants

- [ ] **`BackendHandle` seam holds.** `grep -r 'burn_wgpu\|WgpuBackend\|GemmaRunner' inference-engine/ inference-api/` returns zero results.
- [ ] **`PrefetchOps` default no-op compiles without warnings** on all existing Backend implementations.
- [ ] **No unsafe code** in `inference-backend-wgpu-offload` except inside `GgufIndex::tensor_bytes` (mmap slice cast), which must be annotated with a `// SAFETY:` comment explaining the lifetime invariant.

---

## 8. Non-Goals

The following are explicitly deferred to later phases. They must not be implemented in Phase 1, as premature implementation would obscure the correctness signal that Phase 1 is designed to produce.

| Deferred item | Target phase |
|---------------|-------------|
| `burn-ggml` backend (ggml + llama.cpp kernels, macOS Metal) | Phase 2A |
| Prefill-decode disaggregation (separate prefill/decode workers) | Phase 2 |
| FlashMoE kernel (fused routing + sparse GEMM in one shader) | Phase 2 |
| INT4 dequant inside the shader (currently dequant to FP32 host-side) | Phase 2 |
| Tensor parallelism (splitting attention heads across two iGPUs) | Not planned |
| Multi-node distributed inference | Not planned |
| ANE (Apple Neural Engine) integration | Phase 3+ |
| Intel NPU integration | Not planned |
| Prometheus `/metrics` endpoint | Phase 1 stretch goal (not a hard gate) |
| Priority scheduling beyond FCFS | Not planned |
| Speculative decoding | Phase 3 |
| Continuous batching with dynamic sequence length adjustment | Phase 2 |
| LoRA adapter loading | Not planned |
| Models other than Gemma 4 26B MoE | Phase 2+ |
| Quantization schemes other than Q4_K_M | Phase 2 |
| Windows or macOS support for the offload stack | Phase 2A (macOS via ggml) |
| Auto-tuning of `max_slots` / ping-pong buffer size | Not planned |

---

## 9. Dependencies

The following new crates are added to the workspace `Cargo.toml` in Phase 1. All other dependencies are inherited from Phase 0 unchanged.

### New crates added

| Crate | Version | Feature flags | Purpose |
|-------|---------|---------------|---------|
| `memmap2` | `0.9` | default | Memory-mapped GGUF file access in `GgufIndex`; zero-copy tensor byte slices |
| `lru` | `0.12` | default | LRU eviction policy in `WeightCache<K>` |
| `tokio` | `1` | `full` | Already present in Phase 0; ensure `full` features enabled for `spawn_blocking`, `Mutex`, `Notify`, `mpsc` |

### Existing crates with new feature flags

| Crate | New feature | Reason |
|-------|-------------|--------|
| `burn-wgpu` | `subgroup-ops` (if available) | Enable `Features::SUBGROUP` in `moe_router.wgsl` fast path on Intel ANV |
| `tracing` | `std` | Already enabled; used for prefetch overlap test spans (test 6.5) |
| `tracing-subscriber` | `env-filter` | Used in integration test harness for span collection |

### `Cargo.toml` additions

```toml
# workspace Cargo.toml — new members
[workspace]
members = [
    "inference-engine",
    "inference-api",
    "inference-backend",
    "inference-model-gemma",      # Phase 0 — unchanged
    "inference-backend-wgpu-offload",  # NEW in Phase 1
    "inference-model-gemma4",          # NEW in Phase 1
]

# inference-backend-wgpu-offload/Cargo.toml
[dependencies]
memmap2            = "0.9"
lru                = "0.12"
tokio              = { version = "1", features = ["full"] }
burn-wgpu          = { version = "0.14", features = ["subgroup-ops"] }
tracing            = "0.1"
inference-backend  = { path = "../inference-backend" }
inference-engine   = { path = "../inference-engine" }

# inference-model-gemma4/Cargo.toml
[dependencies]
burn               = { version = "0.14", features = ["wgpu"] }
burn-wgpu          = "0.14"
tokenizers         = "0.19"
serde              = { version = "1", features = ["derive"] }
serde_json         = "1"
inference-backend  = { path = "../inference-backend" }
```

---

## 10. Open Questions

The following questions must be resolved during Phase 1 implementation. Each has a suggested resolution; the implementer must confirm or revise before the relevant component is built.

### Q1: Exact global layer indices for Gemma 4 26B MoE

**Question:** The spec states "alternating local and global layers" with `layer_idx % 6 == 0` as global. Is this confirmed by the Gemma 4 26B architecture? The exact pattern must be verified against the GGUF metadata key `gemma4.attention.sliding_window_pattern` before `Gemma4Layer::attention_kind` is implemented.

**Suggested resolution:** Parse `gemma4.attention.sliding_window_pattern` from the GGUF header during `GgufIndex::open`. Assert the parsed pattern matches the hardcoded assumption. Fail fast with a clear error message if it does not.

**Blocking:** `Gemma4MoeModel`, `KvOffloadManager` (affects `num_global_layers` and KV file sizing).

---

### Q2: GGUF tensor naming convention for Gemma 4 MoE expert weights

**Question:** The spec uses `blk.{layer}.ffn_gate_exps.{expert}` as the tensor name pattern (llama.cpp convention). This must be verified against an actual Gemma 4 26B MoE GGUF export from `llama.cpp`. If the naming convention differs (e.g., Hugging Face safetensors-derived GGUF may use different names), `GgufIndex::expert_offsets` will return empty results silently.

**Suggested resolution:** Add a startup validation pass in `GgufIndex::open` that checks for the presence of at least one expert tensor from layer 0. Fail with a descriptive error listing the first 10 tensor names if the expected pattern is not found.

**Blocking:** `GgufIndex`, `ExpertWeightCache` initialization.

---

### Q3: `burn-wgpu` subgroup support on Mesa 22 / Intel ANV

**Question:** The `moe_router.wgsl` fast path uses `subgroupMax` and `subgroupAdd`. These require `wgpu::Features::SUBGROUP` to be supported and advertised by the Vulkan driver. Mesa 22 + Intel ANV is expected to support this, but it has not been confirmed on the target hardware.

**Suggested resolution:** At `WgpuOffloadBackendHandle::new`, query `adapter.features().contains(Features::SUBGROUP)`. If not available, compile `moe_router.wgsl` with `USE_SUBGROUPS` undefined (shared-memory fallback). Log a warning but do not fail. Add a test that verifies the subgroup path produces the same output as the fallback path.

**Blocking:** `moe_router.wgsl` shader compilation strategy.

---

### Q4: Ping-pong buffer sizing for 256K context global KV

**Question:** The memory budget estimates each global KV buffer at ~1.1 GB (INT8, 256K tokens, 8 heads, head_dim 256). This estimate must be confirmed against the actual Gemma 4 26B attention configuration. If the actual size is larger, the RAM budget is violated.

**Formula:** `buffer_size = num_kv_heads × head_dim × max_seq_len × 2 (K+V) × sizeof(INT8)`

**Suggested resolution:** Compute buffer size dynamically from `Gemma4Config` at startup. If `2 × buffer_size > available_ram - 4 GB`, reduce `max_seq_len` and log a warning. Phase 1 can accept a reduced max context if necessary; full 256K is a stretch target.

**Blocking:** `KvOffloadManager` initialization.

---

### Q5: Tokio blocking thread pool and SSD read latency

**Question:** Phase 1 uses `tokio::task::spawn_blocking` for all `pread` calls. On a cold NVMe SSD, a single 14 MB expert read may take 5–30 ms. If many expert cache misses occur simultaneously, the Tokio blocking thread pool could saturate, causing queuing latency that defeats the prefetch overlap.

**Suggested resolution:** Set `TOKIO_WORKER_THREADS` and configure a dedicated blocking thread pool with a minimum of 8 threads for SSD I/O (via `tokio::runtime::Builder::new_multi_thread().max_blocking_threads(16)`). Monitor blocking thread saturation in the prefetch overlap test (test 6.5).

**Blocking:** `WeightCache` and `KvOffloadManager` async I/O design.

---

### Q6: Expert weight format in Q4_K_M GGUF

**Question:** Q4_K_M weights use a sub-block quantization format where each 32-weight block has a shared scale and minimum value stored adjacent to the quantized nibbles. The `WeightCache` returns raw bytes and expects the consumer (`mul_mat_id.wgsl`) to dequantize. The shader must implement the Q4_K_M dequantization formula inline, or the weights must be dequantized to FP32 on CPU before uploading to the GPU buffer.

**Suggested resolution (Phase 1):** Dequantize to FP32 on CPU in `WeightGuard::as_bytes` before upload. This is ~10x slower than in-shader dequant but is correct and simple. In-shader Q4_K_M dequant is a Phase 2 optimization.

**Blocking:** `mul_mat_id.wgsl`, `WeightCache` upload path.

---

