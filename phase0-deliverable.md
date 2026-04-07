# Phase 0 Deliverable Specification: burn-inference

**Status:** Draft v0.1
**Date:** 2026-04-06
**Scope:** Phase 0 only — core engine orchestration with `burn-wgpu`, Gemma 3 1B–4B, no offloading

---

## Table of Contents

1. [Overview](#1-overview)
2. [Scope](#2-scope)
3. [Crate Structure](#3-crate-structure)
4. [Component Specifications](#4-component-specifications)
5. [Data Flow](#5-data-flow)
6. [Test Specification](#6-test-specification)
7. [Acceptance Criteria](#7-acceptance-criteria)
8. [Non-Goals](#8-non-goals)
9. [Dependencies](#9-dependencies)
10. [Open Questions](#10-open-questions)

---

## 1. Overview

Phase 0 is the foundation phase of `burn-inference`. Its sole purpose is to validate that the core engine orchestration logic is correct and performant, in isolation from the complexity of weight offloading, SSD KV streaming, and heterogeneous backends that arrive in later phases.

### What Phase 0 proves

1. **The scheduler is correct.** The `run_overlapped_loop`, `schedule_batch`, `PrefillAdder`, and `collect_decode_requests` components correctly manage request state transitions (`Waiting` → `Prefilling` → `Decoding` → `Done`), allocate and release KV cache pages without leaks, and handle chunked prefill for prompts that exceed the per-iteration token budget.

2. **The radix cache is correct.** The slab-based `RadixCache` correctly matches prefixes, handles edge splitting, maintains ref-count locking, evicts LRU leaves, and produces measurable cache hit rates on repeated prompts.

3. **The idle loop is correct.** When no requests are in flight, the engine task suspends completely via `sched_rx.recv().await` and consumes zero CPU. This is not a performance nicety — it is a correctness requirement for a long-running server process.

4. **The `BackendHandle` seam is the only backend coupling.** The scheduler and HTTP layer have no direct knowledge of `burn-wgpu`, GGUF loading, or tensor types. Swapping the backend in Phase 1 (from `WgpuBackendHandle` to `GgmlBackendHandle`) requires no changes to `inference-engine` or `inference-api`.

5. **End-to-end correctness on real hardware.** The system must produce semantically correct answers from Gemma 3 1B or 4B weights on both Linux/Vulkan (`burn-wgpu`) and macOS/Metal (`burn-wgpu` with Metal backend), using the standard `burn-wgpu` backend with no custom kernels.

### Why Phase 0 comes before offloading

The offloading machinery in Phase 1+ (`KvOffloadManager`, `WeightCache`, `ExpertWeightCache`, `PrefetchOps`) introduces asynchronous I/O, multi-buffer pipeline management, and complex interactions between the scheduler and the backend's memory manager. If those systems are built on a buggy scheduler or a leaky KV pool, debugging becomes exponentially harder.

Phase 0 deliberately constrains the model to one that fits entirely in GPU/iGPU memory (Gemma 3 1B–4B, 2–8 GB in BF16/F16), eliminating all I/O from the critical path. Every remaining correctness issue is a pure scheduling or model logic issue — the simplest possible context for diagnosis.

---

## 2. Scope

### In scope

- `inference-engine` crate: all components listed in Section 3 and fully specified in Section 4
- `inference-api` crate: axum HTTP server, SSE streaming, `TokenizerService`
- `inference-backend` crate: `BackendHandle` trait, `StubBackend`, `WgpuBackendHandle`
- `inference-model-gemma` crate: `Gemma3Config`, `Gemma3Model`, GGUF/safetensors loader
- All 6 correctness/behaviour tests passing (see Section 6)
- All performance targets met on the target hardware (see Section 7)
- Workspace `Cargo.toml`, CI configuration (GitHub Actions), and `cargo clippy` / `cargo fmt` passing

### Explicitly out of scope

- KV cache SSD offloading (`KvOffloadManager`) — Phase 1
- Weight streaming from disk (`WeightCache`) — Phase 1
- Expert weight streaming (`ExpertWeightCache`, FlashMoE) — Phase 2
- `burn-ggml` backend (macOS Metal via ggml) — Phase 1
- Prefill-decode disaggregation (separate prefill/decode workers) — Phase 2
- ANE (Apple Neural Engine) integration — Phase 3+
- Intel NPU — not planned
- Priority scheduling beyond FCFS — Phase 1
- Prometheus metrics endpoint — Phase 1
- Multi-node / distributed inference — not planned
- Models other than Gemma 3 1B/4B — Phase 1+
- Quantization beyond BF16/F16 — Phase 1 (INT8/INT4 via GGUF)

---

## 3. Crate Structure

```
burn-inference/
├── Cargo.toml                          # workspace manifest; members = all four crates
├── .github/
│   └── workflows/
│       └── ci.yml                      # cargo test + clippy + fmt on ubuntu-latest and macos-latest
├── inference-engine/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                      # pub re-exports; feature flags
│       ├── error.rs                    # EngineError, AbortReason
│       ├── request.rs                  # Request, RequestId, RequestState, SamplingParams,
│       │                               #   TokenEvent, FinishReason, PageIndex
│       ├── batch.rs                    # Batch, BatchPhase, ExpertRouting (stub for Phase 0)
│       ├── config.rs                   # EngineConfig, ModelConfig, EngineStats
│       ├── kv_pool.rs                  # KvCachePool, PageTable, PageState, KvBufferHandle
│       ├── radix_cache.rs              # RadixCache, RadixNode, NodeId
│       ├── prefill.rs                  # PrefillAdder (greedy chunked-prefill allocator)
│       ├── decode.rs                   # collect_decode_requests
│       ├── scheduler.rs                # schedule_batch, assemble_batch, process_and_transition
│       └── engine_loop.rs             # run_overlapped_loop, GpuWork, EngineContext
├── inference-api/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── types.rs                    # ChatCompletionRequest, ChatCompletionResponse,
│       │                               #   ChatMessage, Role, StreamChunk, UsageStats
│       ├── routes.rs                   # axum Router; POST /v1/chat/completions,
│       │                               #   GET /v1/models, GET /health
│       ├── sse.rs                      # SSE stream builder; maps TokenEvent → StreamChunk
│       └── tokenizer_service.rs        # TokenizerService, TokenizerHandle, encode/decode tasks
├── inference-backend/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── backend.rs                  # BackendHandle trait, Logits, ForwardOutput
│       ├── stub.rs                     # StubBackend: deterministic fake logits, no GPU required
│       └── wgpu.rs                     # WgpuBackendHandle: wraps burn-wgpu + Gemma3Model
└── inference-model-gemma/
    ├── Cargo.toml
    └── src/
        ├── lib.rs
        ├── config.rs                   # Gemma3Config (parsed from model JSON or GGUF header)
        ├── model.rs                    # Gemma3Model: Burn module; RMSNorm, RoPE, GQA,
        │                               #   paged attention, GeGLU FFN
        └── loader.rs                   # load_from_gguf(), load_from_safetensors();
                                        #   maps weight names to Burn parameter paths
```

### File-level descriptions

| File | Responsibility |
|------|---------------|
| `inference-engine/src/engine_loop.rs` | The main async loop: `run_overlapped_loop`. Owns `EngineContext`, drives the scheduler/GPU thread channel, implements all three idle states. |
| `inference-engine/src/scheduler.rs` | `schedule_batch`: assembles one `Batch` per call. Stateless relative to `EngineContext`. |
| `inference-engine/src/prefill.rs` | `PrefillAdder`: greedy token-budget allocator for prefill work. Handles KV page allocation and radix cache prefix matching. |
| `inference-engine/src/decode.rs` | `collect_decode_requests`: collects all `Decoding`-state requests; frees pages for `Done`/`Aborted` requests. |
| `inference-engine/src/radix_cache.rs` | `RadixCache`: slab-based prefix tree. `match_prefix`, `insert_prefix`, `split_edge`, `evict_pages`, `unlock_nodes`. |
| `inference-engine/src/kv_pool.rs` | `KvCachePool`: free-list page allocator. `PageTable`: (request_id, slot) → page_index map. |
| `inference-backend/src/backend.rs` | `BackendHandle` trait definition. This file must not import from `inference-engine`. |
| `inference-backend/src/stub.rs` | `StubBackend`: returns fixed logits from a pre-seeded RNG. Used in all unit/scheduler tests. |
| `inference-backend/src/wgpu.rs` | `WgpuBackendHandle`: wraps `Gemma3Model<Wgpu>` and the `KvCachePool` GPU buffer. |
| `inference-model-gemma/src/model.rs` | Full Gemma 3 transformer in Burn. Paged attention kernel (via `burn-wgpu` custom op or emulated). |
| `inference-model-gemma/src/loader.rs` | GGUF parser and safetensors loader. Maps Hugging Face weight names to Burn parameter keys. |
| `inference-api/src/tokenizer_service.rs` | Tokio task that owns the HF tokenizer. Async encode/decode via `mpsc` channels. |
| `inference-api/src/routes.rs` | axum route handlers. Each handler sends to `TokenizerService`, which sends to `engine_loop`. |

---

## 4. Component Specifications

### 4.1 `BackendHandle` Trait

**Location:** `inference-backend/src/backend.rs`

**Contract:** The only interface between `inference-engine` and any compute backend. The engine never directly imports burn, wgpu, or ggml types. All tensor operations are opaque from the engine's perspective.

```rust
use std::future::Future;

/// Raw logits output from a forward pass.
/// Shape: [num_output_rows, vocab_size].
/// For prefill batches: one row per request (last-position logits only).
/// For decode batches: one row per request.
/// For mixed batches: prefill requests first, then decode requests, in batch order.
pub struct Logits {
    pub data: Vec<f32>,
    pub num_rows: usize,
    pub vocab_size: usize,
}

/// The sole interface between the engine scheduler and the compute backend.
/// Implemented by: StubBackend, WgpuBackendHandle (Phase 0), GgmlBackendHandle (Phase 1+).
pub trait BackendHandle: Send + Sync + 'static {
    /// Execute one forward pass. Must be called from a context where it is safe
    /// to block (i.e., from a dedicated GPU thread, not from within the Tokio runtime).
    /// Returns logits as described above.
    fn forward(&self, batch: &Batch) -> impl Future<Output = Result<Logits, EngineError>> + Send;

    /// Return a reference to the KV cache pool.
    /// The engine uses this to check free page counts before scheduling.
    fn kv_pool(&self) -> &dyn KvPool;

    /// Return model configuration (num_layers, num_kv_heads, head_dim, vocab_size).
    fn model_config(&self) -> &ModelConfig;

    /// Optional: signal the backend to begin prefetching data for the given batch.
    /// Default implementation is a no-op (correct for burn-wgpu in Phase 0).
    /// burn-ggml overrides this in Phase 1 to issue pread() calls for weight pages.
    fn prefetch(&self, _batch: &Batch) {}
}

/// Trait object-safe interface for KV pool free-page queries.
pub trait KvPool: Send + Sync {
    fn free_pages(&self) -> usize;
    fn total_pages(&self) -> usize;
}
```

**Thread safety:** `BackendHandle` must be `Send + Sync + 'static`. The engine holds it as `Arc<dyn BackendHandle>`. In Phase 0, `WgpuBackendHandle` achieves this by sending work to an owned GPU thread via a `std::sync::mpsc` channel and returning results via `tokio::sync::oneshot`.

**Error handling:** `forward` returns `Result<Logits, EngineError>`. On error, the engine transitions all requests in the batch to `Aborted { reason: AbortReason::BackendError(msg) }` and returns their KV pages to the pool.

---

### 4.2 `StubBackend`

**Location:** `inference-backend/src/stub.rs`

**Contract:** A deterministic fake backend that returns plausible logits without requiring a GPU or model weights. Used in all unit tests and scheduler benchmarks.

```rust
pub struct StubBackend {
    config: ModelConfig,
    pool: StubKvPool,
    /// Seed for reproducible token sequences across test runs.
    rng_seed: u64,
    /// Optional override: if Some, always returns this token id as the argmax.
    forced_token: Option<u32>,
}

impl StubBackend {
    pub fn new(config: ModelConfig, num_pages: usize, rng_seed: u64) -> Self;

    /// Configure the backend to always produce `token_id` as the top logit.
    /// Useful for tests that need deterministic output strings.
    pub fn with_forced_token(mut self, token_id: u32) -> Self;
}
```

**Behaviour:**
- `forward` returns in < 100 microseconds (no GPU work, no I/O).
- Returns `Logits` with `vocab_size = config.vocab_size` rows, each row being zero except for one large positive value at a position derived from the batch's input tokens (deterministic function of `rng_seed` and batch content).
- `kv_pool()` returns a stub pool that tracks allocations in memory but performs no GPU operations.
- `prefetch` is a no-op.

**Use in tests:** All tests in `inference-engine` must compile and pass with `StubBackend`. No test in `inference-engine` or `inference-api` may depend on `WgpuBackendHandle` or `Gemma3Model` at compile time.

---

### 4.3 `WgpuBackendHandle`

**Location:** `inference-backend/src/wgpu.rs`

**Contract:** Wraps `Gemma3Model<Wgpu>` and a GPU-resident `KvCachePool` buffer. Executes forward passes on the `burn-wgpu` backend (Vulkan on Linux, Metal on macOS).

```rust
pub struct WgpuBackendHandle {
    /// Shared reference to the model; the model's parameters live on the GPU.
    model: Arc<Gemma3Model<Wgpu>>,
    /// GPU-resident KV cache buffer.
    kv_buffer: Arc<KvBuffer<Wgpu>>,
    /// Pool metadata (free list, page states). CPU-side only.
    pool_meta: Arc<Mutex<KvCachePoolMeta>>,
    config: ModelConfig,
    /// Channel for sending GPU work to the dedicated GPU thread.
    work_tx: std::sync::mpsc::SyncSender<GpuWorkItem>,
}

/// Sent from the engine task to the GPU thread.
struct GpuWorkItem {
    batch: Batch,
    result_tx: tokio::sync::oneshot::Sender<Result<Logits, EngineError>>,
}

impl WgpuBackendHandle {
    /// Construct and initialise. Spawns the GPU thread.
    /// `weight_path`: path to GGUF or safetensors file.
    /// `num_kv_pages`: number of KV cache pages to pre-allocate on the GPU.
    pub async fn new(
        weight_path: &Path,
        num_kv_pages: usize,
        device: WgpuDevice,
    ) -> Result<Self, EngineError>;
}
```

**GPU thread:** `WgpuBackendHandle::new` spawns a `std::thread` (not a Tokio task) that owns the `Wgpu` device context and loops on the `std::sync::mpsc::Receiver<GpuWorkItem>`. For each item, it executes the Burn forward pass synchronously and sends the result back via the `oneshot::Sender`.

**Paged attention:** Phase 0 implements paged attention as a standard dense attention pass that gathers KV entries from paged memory into a contiguous buffer before calling Burn's attention operation. This is correct but not peak-performance; a custom paged attention WGSL kernel is deferred to Phase 1.

**KV buffer layout:** `[num_layers, 2, num_pages, page_size, num_kv_heads, head_dim]` allocated as a single Burn tensor on the GPU device.

**Error handling:** GPU OOM is mapped to `EngineError::OutOfKvPages`. Burn panics are caught with `std::panic::catch_unwind` at the GPU thread boundary and mapped to `EngineError::Backend(msg)`.

---

### 4.4 `Request` and `RequestState`

**Location:** `inference-engine/src/request.rs`

```rust
pub type RequestId = uuid::Uuid;
pub type PageIndex = u32;
pub type NodeId = usize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestState {
    Waiting,
    Prefilling { processed_tokens: usize },
    Decoding,
    Done,
    Aborted { reason: AbortReason },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbortReason {
    ClientDisconnected,
    MaxTokensExceeded,
    BackendError(String),
    Preempted,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub max_new_tokens: usize,
    pub stop_sequences: Vec<String>,
    pub stream: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            max_new_tokens: 512,
            stop_sequences: vec![],
            stream: false,
        }
    }
}

pub struct Request {
    pub id: RequestId,
    pub input_ids: Vec<u32>,
    pub state: RequestState,

    /// Tokens already in KV cache from radix prefix match. Skip during prefill.
    pub cached_len: usize,
    /// Total tokens with KV entries on device (cached + computed this session).
    pub device_len: usize,
    /// Tokens to compute in the current scheduler iteration (set by PrefillAdder).
    pub extend_len: usize,

    /// KV cache page indices owned by this request.
    pub kv_pages: Vec<PageIndex>,

    /// Tokens generated so far in the decode phase.
    pub output_ids: Vec<u32>,

    pub params: SamplingParams,

    /// Channel to stream token events back to the HTTP handler.
    /// None if streaming is disabled (batch mode).
    pub token_tx: Option<tokio::sync::mpsc::Sender<TokenEvent>>,

    /// Radix cache node IDs locked by this request (must call unlock on completion).
    pub locked_nodes: Vec<NodeId>,

    pub arrival_time: std::time::Instant,
    pub first_token_time: Option<std::time::Instant>,
}

#[derive(Debug)]
pub enum TokenEvent {
    Token(u32),
    Done { finish_reason: FinishReason },
    Error(String),
}

#[derive(Debug, Clone)]
pub enum FinishReason {
    EosToken,
    StopString,
    MaxTokens,
    Aborted,
}
```

**State machine invariants (must be upheld by the scheduler):**
- A `Request` in `Waiting` has `kv_pages.is_empty()`, `device_len == 0`, `extend_len == 0`.
- A `Request` in `Prefilling` has `kv_pages.len() >= 1`. `device_len + extend_len <= input_ids.len()`.
- When `device_len == input_ids.len()`, the next iteration transitions to `Decoding`.
- A `Request` in `Decoding` has `extend_len == 0` (set by the scheduler each iteration).
- On `Done` or `Aborted`, `locked_nodes` must be unlocked and `kv_pages` freed before the request is dropped.

---

### 4.5 `KvCachePool`

**Location:** `inference-engine/src/kv_pool.rs`

```rust
pub struct KvCachePool {
    pub num_pages: usize,
    pub page_size: usize,
    free_pages: Vec<PageIndex>,
    page_states: Vec<PageState>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageState {
    Free,
    InUse(RequestId),
    // Offloaded variant added in Phase 1:
    // Offloaded { ssd_offset: u64, size_bytes: usize },
}

impl KvCachePool {
    pub fn new(num_pages: usize, page_size: usize) -> Self;

    /// Allocate one page. Returns None if the pool is exhausted.
    pub fn allocate(&mut self, owner: RequestId) -> Option<PageIndex>;

    /// Free a page and return it to the pool.
    /// Panics in debug builds if the page is not InUse.
    pub fn free(&mut self, page: PageIndex);

    pub fn free_pages(&self) -> usize;
    pub fn total_pages(&self) -> usize;

    /// Compute bytes required for a pool of the given dimensions.
    pub fn required_bytes(
        num_pages: usize,
        page_size: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype_bytes: usize,
    ) -> usize {
        2 * num_layers * num_pages * page_size * num_kv_heads * head_dim * dtype_bytes
    }
}

pub struct PageTable {
    inner: std::collections::HashMap<(RequestId, usize), PageIndex>,
}

impl PageTable {
    pub fn insert(&mut self, req_id: RequestId, slot: usize, page: PageIndex);
    pub fn lookup(&self, req_id: RequestId, slot: usize) -> Option<PageIndex>;
    /// Remove all entries for req_id and return the freed page indices.
    pub fn remove_request(&mut self, req_id: RequestId) -> Vec<PageIndex>;
}
```

**Invariants:**
- `free_pages.len() + in_use_count == num_pages` at all times.
- No page index appears in more than one request's `kv_pages` vector simultaneously.
- After every request completion (`Done` or `Aborted`), all its page indices are freed exactly once.

**Thread safety:** `KvCachePool` is not `Sync`. It is owned exclusively by `EngineContext`, which is accessed only from the engine task (single-writer). No external locking is required.

---

### 4.6 `RadixCache`

**Location:** `inference-engine/src/radix_cache.rs`

```rust
use slab::Slab;
use rustc_hash::FxHashMap;
use std::{cmp::Reverse, collections::BinaryHeap, time::Instant};

pub type NodeId = usize;

pub struct RadixNode {
    pub edge_tokens: Vec<u32>,
    pub kv_pages: Vec<PageIndex>,
    pub children: FxHashMap<u32, NodeId>,
    pub parent: Option<NodeId>,
    /// In-flight request ref count. Nodes with lock_ref > 0 cannot be evicted.
    pub lock_ref: i32,
    pub last_used: Instant,
}

pub struct RadixCache {
    nodes: Slab<RadixNode>,
    root: NodeId,
    /// LRU heap over (last_used, node_id). May contain stale entries.
    evictable: BinaryHeap<(Reverse<Instant>, NodeId)>,
    page_size: usize,
    total_pages: usize,
    max_pages: usize,
}

impl RadixCache {
    pub fn new(page_size: usize, max_pages: usize) -> Self;

    /// Walk the tree following `tokens`. Return (matched_len, kv_pages, locked_node_ids).
    /// `matched_len` is always a multiple of `page_size`.
    /// Increments lock_ref on each matched node. Caller MUST call unlock_nodes().
    pub fn match_prefix(
        &mut self,
        tokens: &[u32],
    ) -> (usize, Vec<PageIndex>, Vec<NodeId>);

    /// Insert `tokens[..N*page_size]` → `pages[..N]` into the tree.
    /// Handles full-edge matches (descend), partial-edge matches (split_edge), and
    /// no-match (new leaf). Only inserts complete pages (partial last page is ignored).
    /// Evicts LRU pages before inserting if total_pages would exceed max_pages.
    pub fn insert_prefix(&mut self, tokens: &[u32], pages: &[PageIndex]);

    /// Decrement lock_ref for each node id. Update last_used to now.
    /// Adds newly evictable nodes to the eviction heap.
    pub fn unlock_nodes(&mut self, node_ids: &[NodeId]);

    /// Evict LRU unlocked leaf nodes until `target_pages` pages are freed.
    /// Returns freed page indices (caller must return them to KvCachePool).
    /// If fewer than `target_pages` can be freed, returns all that were freed.
    pub fn evict_pages(&mut self, target_pages: usize) -> Vec<PageIndex>;

    /// Promote a completed request's KV pages into the cache.
    /// Called when a request transitions Prefilling → Decoding.
    /// Only full pages are inserted.
    pub fn promote_request(&mut self, req: &Request);

    pub fn total_cached_pages(&self) -> usize;
    pub fn num_nodes(&self) -> usize;
}
```

**Invariants:**
- The root node always exists with `NodeId = 0` and `edge_tokens.is_empty()`.
- Every `NodeId` stored as a child pointer exists in `nodes` (slab does not contain dangling entries).
- `total_pages` equals the sum of `kv_pages.len()` across all non-root nodes.
- A node with `lock_ref > 0` is never in the `evictable` heap as an active candidate.
- After `evict_pages(n)`, `total_pages` decreases by exactly `returned_vec.len()`.

**Thread safety:** `RadixCache` is `!Sync`. It is owned by `EngineContext` and accessed only from the engine task. No locking inside `RadixCache` itself.

---

### 4.7 `PrefillAdder`

**Location:** `inference-engine/src/prefill.rs`

```rust
/// Greedy algorithm to fill the per-iteration token budget with prefill work.
/// Operates on `EngineContext` in place; modifies `ctx.waiting`, `ctx.prefilling`,
/// request states, and KV page allocations.
pub struct PrefillAdder<'a> {
    ctx: &'a mut EngineContext,
    budget: usize,
    tokens_used: usize,
}

impl<'a> PrefillAdder<'a> {
    pub fn new(ctx: &'a mut EngineContext, budget: usize) -> Self;

    /// Run the greedy allocation. Returns the list of requests added to the
    /// current batch (a subset of ctx.prefilling).
    pub fn run(mut self) -> Vec<Arc<parking_lot::Mutex<Request>>>;
}
```

**Algorithm (normative):**

1. **Continue in-progress prefill requests.** Iterate `ctx.prefilling` in order. For each, assign `extend_len = min(remaining_tokens, budget - tokens_used)`. Stop when budget is exhausted.

2. **Admit new requests from waiting queue.** While `tokens_used < budget` and `ctx.waiting` is non-empty:
   a. Peek at the front request. Compute `pages_needed = ceil(input_ids.len() / page_size)`.
   b. If `ctx.kv_pool.free_pages() < pages_needed`, attempt `ctx.radix_cache.evict_pages(deficit)` and free returned pages to pool. If still insufficient, stop.
   c. Pop the request. Call `ctx.radix_cache.match_prefix(&req.input_ids)`. Set `req.cached_len`, `req.device_len = cached_len`, `req.locked_nodes`.
   d. Allocate KV pages for uncached portion. Push request to `ctx.prefilling`.
   e. Assign `extend_len = min(input_ids.len() - cached_len, budget - tokens_used)`.
   f. Set `req.state = RequestState::Prefilling { processed_tokens: 0 }`.

3. Return all requests that received `extend_len > 0` as the batch's prefill list.

**Invariant after `run()`:** `sum(req.extend_len for req in result) <= budget`.

---

### 4.8 `run_overlapped_loop`

**Location:** `inference-engine/src/engine_loop.rs`

```rust
pub struct EngineContext {
    pub backend: Arc<dyn BackendHandle>,
    pub radix_cache: RadixCache,
    pub kv_pool: KvCachePool,
    pub waiting: std::collections::VecDeque<Arc<parking_lot::Mutex<Request>>>,
    pub prefilling: Vec<Arc<parking_lot::Mutex<Request>>>,
    pub decoding: Vec<Arc<parking_lot::Mutex<Request>>>,
    pub config: EngineConfig,
    pub stats: EngineStats,
}

/// Message type for the engine-to-GPU-thread channel.
pub struct GpuWork {
    pub batch: Batch,
    pub result_tx: tokio::sync::oneshot::Sender<Result<Logits, EngineError>>,
}

/// The main engine loop. Must be spawned as a Tokio task.
/// `sched_rx`: receives new Request objects from the HTTP/tokenizer layer.
/// `gpu_tx`: sends GpuWork items to the dedicated GPU thread.
pub async fn run_overlapped_loop(
    mut ctx: EngineContext,
    mut sched_rx: tokio::sync::mpsc::Receiver<Arc<parking_lot::Mutex<Request>>>,
    gpu_tx: std::sync::mpsc::SyncSender<GpuWork>,
);
```

**Three-state idle protocol (normative):**

```
State          | Condition                          | Suspension point
---------------|------------------------------------|-----------------------------------------
Active         | batch in-flight, more work exists  | result_rx.await (GPU result)
GPU draining   | batch in-flight, nothing new to    | result_rx.await (GPU result)
               | schedule alongside it              |
Truly idle     | no in-flight work, no waiting reqs | sched_rx.recv().await (new request)
```

The loop body per iteration:

1. Drain `sched_rx` with `try_recv()` (non-blocking) into `ctx.waiting`.
2. If `in_flight` is `Some`, `await` the `oneshot::Receiver` to collect logits. Call `process_and_transition(&mut ctx, &batch, logits)`.
3. Call `schedule_batch(&mut ctx)`. If it returns `None`:
   - If `in_flight` is `Some` (GPU still running from step 1's non-await path — edge case): await it, transition, `continue`.
   - Else (truly idle): `sched_rx.recv().await`, push to waiting, drain remaining, `continue`.
4. Call `ctx.backend.prefetch(&batch)` (no-op in Phase 0).
5. Create `oneshot::channel()`. Send `GpuWork { batch, result_tx }` to `gpu_tx`. Store `(batch, result_rx)` in `in_flight`.
6. Loop back to step 1 immediately (overlap: GPU is running, CPU schedules next).

**Prohibition:** `tokio::task::yield_now()` MUST NOT appear anywhere in this function or any function it calls. `yield_now` re-queues the task for the next poll cycle without yielding to a blocking event source, causing 100% CPU spin when idle.

---

### 4.9 `TokenizerService`

**Location:** `inference-api/src/tokenizer_service.rs`

```rust
/// A cloneable handle for sending encode/decode requests to the TokenizerService task.
#[derive(Clone)]
pub struct TokenizerHandle {
    encode_tx: tokio::sync::mpsc::Sender<EncodeRequest>,
    decode_tx: tokio::sync::mpsc::Sender<DecodeRequest>,
}

struct EncodeRequest {
    messages: Vec<ChatMessage>,
    result_tx: tokio::sync::oneshot::Sender<Result<Vec<u32>, TokenizerError>>,
}

struct DecodeRequest {
    token_ids: Vec<u32>,
    result_tx: tokio::sync::oneshot::Sender<Result<String, TokenizerError>>,
}

impl TokenizerHandle {
    /// Encode a chat message list to a token ID sequence using the chat template.
    pub async fn encode(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Result<Vec<u32>, TokenizerError>;

    /// Decode a token ID sequence to a UTF-8 string.
    pub async fn decode(&self, token_ids: Vec<u32>) -> Result<String, TokenizerError>;
}

/// Spawns the TokenizerService Tokio task. Returns a handle for communication.
pub fn start_tokenizer_service(tokenizer_path: &Path) -> TokenizerHandle;
```

**Implementation:** The service task owns a `tokenizers::Tokenizer` (HuggingFace `tokenizers` crate). It loops on `tokio::select!` over the encode and decode channels. Encoding uses `tokenizer.apply_chat_template(messages, add_generation_prompt=true)`. Decoding uses `tokenizer.decode(token_ids, skip_special_tokens=true)`.

The tokenizer is CPU-bound and fast (microseconds per call). The async channel indirection ensures the HTTP handler does not share mutable state with the tokenizer object.

**Thread safety:** The `tokenizers::Tokenizer` is `!Send`. The service task is pinned to a single Tokio thread. Cloned `TokenizerHandle`s are `Send + Sync` and can be shared across all HTTP request handlers.

---

### 4.10 HTTP API Routes

**Location:** `inference-api/src/routes.rs`

**Base URL:** `http://0.0.0.0:8080`

#### POST `/v1/chat/completions`

**Request body (subset of OpenAI API):**

```json
{
  "model": "gemma-3-1b",
  "messages": [
    {"role": "user", "content": "What is the capital of China?"}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 256,
  "stream": false
}
```

**Non-streaming response:**

```json
{
  "id": "chatcmpl-<uuid>",
  "object": "chat.completion",
  "model": "gemma-3-1b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Beijing"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 1,
    "total_tokens": 15
  }
}
```

**Streaming response (stream: true):** Returns `Content-Type: text/event-stream`. Each SSE event carries a delta chunk in the same format as OpenAI's streaming API. The final event has `finish_reason` set and `data: [DONE]` as the terminating sentinel.

**Handler logic:**
1. Validate request body. Return `422` on schema violation.
2. Call `tokenizer_handle.encode(messages).await`. Return `500` on tokenizer error.
3. Construct `Request` with a new `RequestId` (UUID v4) and `SamplingParams` from the request body.
4. Create `mpsc::channel::<TokenEvent>()`. Store `token_tx` in the `Request`.
5. Send `Arc<Mutex<Request>>` to engine via `engine_tx.send(req).await`. Return `503` if the engine channel is full (back-pressure).
6. If `stream: false`: collect all `TokenEvent::Token` items until `TokenEvent::Done`, decode via tokenizer, return JSON response.
7. If `stream: true`: return an SSE stream that maps each `TokenEvent` to a delta chunk.

#### GET `/v1/models`

Returns a list with a single entry for the loaded model. Format matches OpenAI's `/v1/models` response.

#### GET `/health`

Returns `{"status": "ok", "engine_state": "idle"|"active"}` with HTTP 200.

**Error codes:**
- `400`: malformed JSON.
- `422`: schema validation failure (unknown role, negative max_tokens, etc.).
- `500`: internal tokenizer or engine error.
- `503`: engine queue full (apply back-pressure to caller).

---

## 5. Data Flow

The following ASCII diagram traces the complete lifecycle of a single streaming request from HTTP POST to final SSE event.

```
Client
  │
  │ POST /v1/chat/completions
  │ {"messages": [...], "stream": true, ...}
  ▼
┌──────────────────────────────────────────────────────────────┐
│  axum handler (inference-api/src/routes.rs)                  │
│                                                              │
│  1. Deserialize ChatCompletionRequest                        │
│  2. tokenizer_handle.encode(messages).await                  │
│     → Vec<u32>  (input_ids)                                  │
│  3. Request::new(id, input_ids, params)                      │
│  4. mpsc::channel::<TokenEvent>()                            │
│     req.token_tx = Some(token_tx)                            │
│  5. engine_tx.send(Arc::new(Mutex::new(req))).await          │
│  6. Return SSE stream backed by token_rx                     │
└───────────────────────────┬──────────────────────────────────┘
                            │ engine_tx (tokio mpsc)
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  run_overlapped_loop  (engine_loop.rs)                       │
│                                                              │
│  ── Iteration N (truly idle → woken by new request) ──       │
│  sched_rx.recv().await  ← suspends here until request arrives│
│  ctx.waiting.push_back(req)                                  │
│                                                              │
│  ── Iteration N+1 ──                                         │
│  try_recv() drains any additional requests                   │
│  schedule_batch(&mut ctx):                                   │
│    PrefillAdder::new(&mut ctx, budget).run():                │
│      radix_cache.match_prefix(&req.input_ids)                │
│        → (cached_len=0, pages=[], locked_nodes=[])           │
│      kv_pool.allocate() × ceil(seq_len / page_size)          │
│      req.state = Prefilling { processed_tokens: 0 }          │
│      req.extend_len = min(seq_len, budget)                   │
│    assemble_batch(prefill=[req], decode=[])                  │
│  backend.prefetch(&batch)   ← no-op (burn-wgpu)              │
│  oneshot::channel() → (result_tx, result_rx)                 │
│  gpu_tx.send(GpuWork { batch, result_tx })                   │
│  in_flight = Some((batch, result_rx))                        │
└───────────────────────────┬──────────────────────────────────┘
                            │ std::sync::mpsc (SyncSender)
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  GPU thread  (std::thread, owns WgpuDevice)                  │
│                                                              │
│  Gemma3Model::forward(batch):                                │
│    gather KV pages from kv_buffer                            │
│    embed input_ids → hidden states                           │
│    N × (RMSNorm, RoPE, GQA paged attention, GeGLU FFN)       │
│    final RMSNorm + lm_head → logits [1, vocab_size]          │
│                                                              │
│  result_tx.send(Ok(logits))                                  │
└───────────────────────────┬──────────────────────────────────┘
                            │ oneshot result
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  run_overlapped_loop  (continued)                            │
│                                                              │
│  result_rx.await → Logits                                    │
│  process_and_transition(&mut ctx, &batch, logits):           │
│    For prefill req:                                          │
│      req.device_len += extend_len                            │
│      if device_len == input_ids.len():                       │
│        sample token from logits[0] (temperature/top-p/top-k)│
│        req.output_ids.push(sampled_token)                    │
│        req.state = Decoding                                  │
│        ctx.prefilling.remove(req)                            │
│        ctx.decoding.push(req)                                │
│        radix_cache.promote_request(&req)                     │
│        token_tx.send(TokenEvent::Token(sampled_token))       │
│      else: req remains in prefilling for next chunk          │
└───────────────────────────┬──────────────────────────────────┘
                            │ (loop continues, now decoding)
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  Subsequent decode iterations (one token per iteration)      │
│                                                              │
│  collect_decode_requests(&mut ctx) → [req]                   │
│  batch: input_ids = [last_output_token], position = device_len│
│  GPU forward → logits [1, vocab_size]                        │
│  sample → new_token                                          │
│  req.output_ids.push(new_token)                              │
│  req.device_len += 1                                         │
│  token_tx.send(TokenEvent::Token(new_token))                 │
│                                                              │
│  Stop condition check:                                       │
│    EOS token → Done { finish_reason: EosToken }              │
│    stop_sequence match → Done { finish_reason: StopString }  │
│    len(output_ids) >= max_new_tokens → Done { MaxTokens }    │
│                                                              │
│  On Done:                                                    │
│    radix_cache.unlock_nodes(&req.locked_nodes)               │
│    kv_pool.free(page) for each page in req.kv_pages          │
│    ctx.decoding.remove(req)                                  │
│    token_tx.send(TokenEvent::Done { finish_reason })         │
└───────────────────────────┬──────────────────────────────────┘
                            │ TokenEvent::Token / Done
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  SSE stream in axum handler                                  │
│                                                              │
│  For each TokenEvent::Token(id):                             │
│    tokenizer_handle.decode(vec![id]).await → text fragment   │
│    Emit: data: {"choices": [{"delta": {"content": "..."}}]}  │
│                                                              │
│  For TokenEvent::Done:                                       │
│    Emit: data: {"choices":[{"finish_reason":"stop"}]}        │
│    Emit: data: [DONE]                                        │
└───────────────────────────┬──────────────────────────────────┘
                            │ SSE chunks
                            ▼
                          Client
```

---

## 6. Test Specification

All tests are in the standard Rust `#[test]` / `#[tokio::test]` framework. `cargo test` must complete without failures or warnings at the `deny(warnings)` lint level.

### 6.1 Unit Tests: `KvCachePool`

**File:** `inference-engine/src/kv_pool.rs` (inline tests)

| Test | Assertion |
|------|-----------|
| `test_allocate_and_free` | Allocate 16 pages one by one; `free_pages()` decrements. Free all; `free_pages() == num_pages`. |
| `test_allocate_exhaustion` | Allocate until `None`; `free_pages() == 0`. |
| `test_double_free_panics` | In debug build, freeing a `Free` page panics. |
| `test_required_bytes` | `KvCachePool::required_bytes(512, 16, 28, 4, 128, 2) == 512 * 16 * 28 * 4 * 128 * 2 * 2`. |
| `test_page_table_roundtrip` | Insert (id, 0) → 7; lookup (id, 0) == Some(7); remove_request returns [7]. |

### 6.2 Unit Tests: `RadixCache`

**File:** `inference-engine/src/radix_cache.rs` (inline tests)

| Test | Assertion |
|------|-----------|
| `test_empty_match` | `match_prefix(&[1,2,3])` on empty cache returns `(0, [], [])`. |
| `test_insert_and_match_full` | Insert `[1,2,3,4]` → `[p0]` (page_size=4). `match_prefix(&[1,2,3,4,5])` returns `(4, [p0], [node_id])`. |
| `test_insert_and_match_partial_edge` | Insert `[1,2,3,4]`; insert `[1,2,5,6]`. Match `[1,2,3,4]` still returns `(4, [p0])`. |
| `test_edge_split` | Insert `[1,2,3,4]`, then `[1,2,5,6]`. Cache has 3 nodes: root, shared prefix `[1,2]` (0 full pages), diverging leaves. |
| `test_lru_eviction` | Insert 3 nodes with 1 page each; max_pages=2. `evict_pages(1)` evicts the oldest. `total_cached_pages() == 2`. |
| `test_lock_prevents_eviction` | Insert node, call `match_prefix` (locks it). `evict_pages(10)` returns 0 freed pages. After `unlock_nodes`, eviction succeeds. |
| `test_promote_request` | Construct a `Request` with `device_len=8`, `input_ids=[1..8]`, `kv_pages=[p0]` (page_size=8). Call `promote_request`. `match_prefix(&[1..8])` returns `(8, [p0])`. |
| `test_page_alignment` | Insert tokens with non-page-aligned length; verify only full pages are stored. |

### 6.3 Unit Tests: `PrefillAdder`

**File:** `inference-engine/src/prefill.rs` (inline tests, using `StubBackend`)

| Test | Assertion |
|------|-----------|
| `test_single_request_within_budget` | Request with 64 tokens, budget=1024. `run()` returns 1 request; `req.extend_len == 64`. |
| `test_chunked_prefill` | Request with 4096 tokens, budget=1024. `run()` assigns `extend_len=1024`. Second iteration: `extend_len=1024`. Four iterations to complete. |
| `test_budget_split_across_requests` | Two requests, 600 tokens each, budget=1024. First gets 600, second gets 424. |
| `test_kv_exhaustion_blocks_admission` | Pool has 2 free pages; request needs 3 pages. `run()` returns empty (request stays in waiting). |
| `test_evict_to_admit` | Radix cache holds 2 evictable pages; pool has 0 free; new request needs 2 pages. `run()` evicts and admits. |
| `test_cache_hit_reduces_pages_needed` | Radix cache has prefix `[1..8]` cached (1 page). New request with `input_ids=[1..16]`. `cached_len=8`, pages needed = ceil((16-8)/page_size). |

### 6.4 Unit Tests: `run_overlapped_loop`

**File:** `inference-engine/src/engine_loop.rs` (tests using `StubBackend`)

| Test | Assertion |
|------|-----------|
| `test_request_completes` | Send one request; await `token_rx` until `TokenEvent::Done`. |
| `test_state_transitions` | Observe `RequestState` via shared `Arc<Mutex<Request>>`: Waiting → Prefilling → Decoding → Done. |
| `test_no_yield_now_in_idle` | Engine receives no requests for 2s; CPU usage of the engine task is effectively zero (verified by OS-level thread CPU accounting or a `std::sync::atomic` spin counter that must remain 0). |
| `test_concurrent_8_requests` | Send 8 requests simultaneously; all receive `TokenEvent::Done`. |
| `test_kv_pages_freed_on_done` | After `Done`, `ctx.kv_pool.free_pages() == ctx.kv_pool.total_pages()`. |

### 6.5 Integration Tests: Scheduler Overhead

**File:** `inference-engine/tests/scheduler_bench.rs`

Uses `StubBackend`. `StubBackend::forward` returns in < 100 µs.

| Test | Assertion |
|------|-----------|
| `test_scheduler_overhead` | 1000 scheduling iterations (alternating prefill/decode batches of 32 requests). Total elapsed time for scheduling calls (excluding `forward`) < 1000 ms (i.e., < 1 ms/iter). |

### 6.6 Correctness Tests (require real model weights)

These tests require `GEMMA_WEIGHTS_PATH` environment variable to be set. They are skipped if the variable is absent (so CI without GPU passes). The designated hardware gates run them explicitly.

**File:** `inference-engine/tests/correctness.rs` and `inference-api/tests/e2e.rs`

---

**Test 1: Factual knowledge**

```rust
#[tokio::test]
#[ignore = "requires GEMMA_WEIGHTS_PATH and GPU"]
async fn test_capital_of_china() {
    let output = run_inference("What is the capital of China?", default_params()).await;
    assert!(
        output.to_lowercase().contains("beijing"),
        "expected 'beijing' in output, got: {output:?}"
    );
}
```

---

**Test 2: Arithmetic**

```rust
#[tokio::test]
#[ignore = "requires GEMMA_WEIGHTS_PATH and GPU"]
async fn test_one_plus_one() {
    let params = SamplingParams {
        temperature: 0.0,  // greedy
        max_new_tokens: 5,
        ..Default::default()
    };
    let output = run_inference(
        "What is 1+1? Answer with only a number.",
        params,
    ).await;
    assert_eq!(
        output.trim(),
        "2",
        "expected '2', got: {output:?}"
    );
}
```

---

**Test 3: Concurrent requests**

```rust
#[tokio::test]
#[ignore = "requires GEMMA_WEIGHTS_PATH and GPU"]
async fn test_8_concurrent_requests() {
    let handles: Vec<_> = (0..8).map(|_| {
        tokio::spawn(run_inference(
            "What is the capital of China?",
            default_params(),
        ))
    }).collect();

    for handle in handles {
        let output = handle.await.unwrap();
        assert!(
            output.to_lowercase().contains("beijing"),
            "concurrent request failed: {output:?}"
        );
    }
}
```

---

**Test 4: Radix cache hit**

```rust
#[tokio::test]
#[ignore = "requires GEMMA_WEIGHTS_PATH and GPU"]
async fn test_radix_cache_hit() {
    let prompt = "What is the capital of China?";

    // First request: cold cache
    let output1 = run_inference(prompt, default_params()).await;

    // Second request: warm cache (same prompt)
    let (output2, stats) = run_inference_with_stats(prompt, default_params()).await;

    assert!(output1.to_lowercase().contains("beijing"));
    assert!(output2.to_lowercase().contains("beijing"));
    assert_eq!(output1, output2, "cache hit must produce identical output");
    assert!(
        stats.cache_hit_tokens > 0,
        "expected radix cache hit on second request"
    );
}
```

---

**Test 5: Chunked prefill**

```rust
#[tokio::test]
#[ignore = "requires GEMMA_WEIGHTS_PATH and GPU"]
async fn test_chunked_prefill_4096_tokens() {
    // Construct a 4096-token prompt by repeating a known phrase.
    let long_prompt = build_prompt_of_length(4096);
    let engine_config = EngineConfig {
        max_prefill_tokens_per_iter: 1024,
        ..default_engine_config()
    };

    let (output, metrics) = run_inference_with_metrics(
        &long_prompt,
        default_params(),
        engine_config,
    ).await;

    assert!(
        metrics.prefill_iterations >= 4,
        "expected >= 4 prefill iterations for 4096-token prompt with 1024 budget, got {}",
        metrics.prefill_iterations
    );
    // Output should be coherent (not empty, not garbage).
    assert!(!output.trim().is_empty(), "output must be non-empty");
}
```

---

**Test 6: Idle CPU usage**

```rust
#[tokio::test]
#[ignore = "requires GEMMA_WEIGHTS_PATH and GPU"]
async fn test_idle_cpu_usage() {
    let _engine = start_engine(default_engine_config()).await;

    // Let the engine sit idle for 10 seconds.
    tokio::time::sleep(Duration::from_secs(10)).await;

    // Measure CPU usage of the engine thread over the idle period.
    // Uses /proc/self/task/<tid>/stat on Linux or mach_thread_info on macOS.
    let cpu_percent = measure_engine_thread_cpu_percent(Duration::from_secs(10));

    assert!(
        cpu_percent < 1.0,
        "idle engine CPU usage must be < 1%, got {cpu_percent:.2}%"
    );
}
```

### 6.7 End-to-End API Tests

**File:** `inference-api/tests/e2e.rs` (uses embedded `StubBackend`; no GPU required)

| Test | Assertion |
|------|-----------|
| `test_non_streaming_response_schema` | POST with `stream:false`; response body matches OpenAI schema exactly. |
| `test_streaming_sse_format` | POST with `stream:true`; each SSE event parses as a valid delta chunk; final event contains `finish_reason`; `data: [DONE]` is last. |
| `test_health_endpoint` | GET `/health` → `{"status":"ok"}` with HTTP 200. |
| `test_models_endpoint` | GET `/v1/models` → array with one entry; `id` matches loaded model name. |
| `test_invalid_role_422` | POST with `"role": "unknown"` → HTTP 422. |
| `test_negative_max_tokens_422` | POST with `"max_tokens": -1` → HTTP 422. |
| `test_back_pressure_503` | Fill engine queue; next request → HTTP 503. |

---

## 7. Acceptance Criteria

Phase 0 is **done** when every item in this checklist is ticked.

### 7.1 Build and Test

- [ ] `cargo build --workspace` succeeds with zero errors and zero warnings (`RUSTFLAGS="-D warnings"`).
- [ ] `cargo test --workspace` passes 100% on both:
  - Ubuntu latest (Vulkan via `RUST_LOG=warn cargo test`).
  - macOS latest (Metal via `RUST_LOG=warn cargo test`).
- [ ] `cargo clippy --workspace -- -D warnings` passes.
- [ ] `cargo fmt --check` passes.

### 7.2 Correctness Tests (hardware gates)

All six tests from Section 6.6 must pass on both platforms:

| Test | Platform: Linux/Vulkan | Platform: macOS/Metal |
|------|----------------------|----------------------|
| Test 1: `"What is the capital of China?"` → contains `"beijing"` | PASS | PASS |
| Test 2: `"What is 1+1?"` → trim == `"2"` | PASS | PASS |
| Test 3: 8 concurrent requests, all correct | PASS | PASS |
| Test 4: Radix cache hit → identical output, `cache_hit_tokens > 0` | PASS | PASS |
| Test 5: 4096-token prompt, 1024 budget → `prefill_iterations >= 4` | PASS | PASS |
| Test 6: 10s idle → CPU < 1% | PASS | PASS |

### 7.3 Performance Targets

Measured on:
- **Linux:** Intel Meteor Lake / Arrow Lake laptop, Intel Arc iGPU, Vulkan, Gemma 3 4B BF16.
- **macOS:** Apple Silicon M3/M4 MacBook Air, Metal, Gemma 3 4B BF16.

| Metric | Target | Measurement method |
|--------|--------|-------------------|
| TTFT (512-token prompt, batch 1) | < 2 s | `first_token_time - arrival_time` |
| Decode throughput (batch 1) | > 15 tok/s | `output_ids.len() / (done_time - first_token_time)` |
| Radix cache hit rate (repeated prompt benchmark) | > 80% | `cache_hit_tokens / total_input_tokens` |
| Scheduler overhead per iteration (StubBackend, batch 32) | < 1 ms | `schedule_batch` wall time, 1000-iteration average |

### 7.4 Code Review Checklist

The following must be verified by code review before Phase 0 is declared complete:

**Idle loop correctness:**
- [ ] `tokio::task::yield_now()` does not appear anywhere in `inference-engine` or `inference-api` source code. (`grep -rn "yield_now" inference-engine/ inference-api/` returns no results.)
- [ ] The truly-idle branch of `run_overlapped_loop` uses `sched_rx.recv().await` (blocking receive, not `try_recv` in a loop).

**Memory correctness:**
- [ ] No `Arc<RwLock<RadixNode>>` or `Arc<Mutex<RadixNode>>` per-node exists in `radix_cache.rs`. Nodes are stored by value in `Slab<RadixNode>`.
- [ ] Every request that reaches `Done` or `Aborted` state has its KV pages freed and its locked radix nodes unlocked before being dropped. Verified by `test_kv_pages_freed_on_done`.

**Backend seam:**
- [ ] `inference-engine/Cargo.toml` does not have a direct dependency on `burn`, `burn-wgpu`, `burn-ggml`, or `inference-model-gemma`.
- [ ] `inference-api/Cargo.toml` does not have a direct dependency on `burn`, `burn-wgpu`, `burn-ggml`, or `inference-model-gemma`.
- [ ] The `BackendHandle` trait definition in `inference-backend/src/backend.rs` imports nothing from `inference-engine`.

**StubBackend isolation:**
- [ ] All tests in `inference-engine/` compile and run with `StubBackend` only — no GPU, no model weights required.
- [ ] `WgpuBackendHandle` and `Gemma3Model` are behind a `#[cfg(feature = "wgpu")]` feature flag. The default feature set for `inference-engine` does not include `wgpu`.

**Async correctness:**
- [ ] No `std::thread::sleep` in async contexts. Only `tokio::time::sleep`.
- [ ] No blocking I/O calls (file reads, network) inside async functions except where explicitly wrapped in `tokio::task::spawn_blocking`.

**Request lifecycle:**
- [ ] The `RequestState` machine transitions are exhaustive: every arm of every match on `RequestState` is handled.
- [ ] No request remains in `ctx.prefilling` or `ctx.decoding` after its `token_tx` channel is closed (client disconnect).

---

## 8. Non-Goals

The following items are explicitly deferred to Phase 1 or later and must not be started during Phase 0 implementation:

| Item | Deferred to |
|------|------------|
| KV cache SSD offloading (`KvOffloadManager`) | Phase 1 |
| Weight streaming from disk (`WeightCache`) | Phase 1 |
| `burn-ggml` backend (ggml + Metal) | Phase 1 |
| INT8 / INT4 quantization (GGUF Q4_K_M, Q8_0) | Phase 1 |
| Models larger than 4B parameters | Phase 1+ |
| Prefill-decode disaggregated workers | Phase 2 |
| Expert weight streaming (`ExpertWeightCache`, FlashMoE) | Phase 2 |
| MoE model support (Qwen2.5-MoE, etc.) | Phase 2 |
| ANE (Apple Neural Engine) integration | Phase 3+ |
| Intel NPU | Not planned |
| Custom paged-attention WGSL kernel | Phase 1 (replaces gather-then-attend) |
| Priority scheduling (High/Normal/Low) | Phase 1 |
| Request preemption with swap-to-SSD | Phase 2 |
| Prometheus / OpenTelemetry metrics | Phase 1 |
| Batch size auto-tuning | Phase 2 |
| Speculative decoding | Phase 3+ |
| Vision / multimodal inputs | Not planned |
| Multi-GPU / multi-node | Not planned |
| Kubernetes / cloud deployment | Not planned |

---

## 9. Dependencies

The following table lists every external crate dependency with exact version constraints and justification. All versions are pinned to minor versions (e.g., `"0.14"`) in `Cargo.toml` to prevent silent breaking changes.

### 9.1 `inference-engine`

| Crate | Version | Justification |
|-------|---------|--------------|
| `tokio` | `"1"` with features `["rt-multi-thread", "sync", "time", "macros"]` | Async runtime. `mpsc`, `oneshot`, `Mutex` for scheduler channels and idle-wait. |
| `parking_lot` | `"0.12"` | `Mutex<Request>` for request state. Lower overhead than `std::sync::Mutex` for short critical sections. |
| `uuid` | `"1"` with feature `["v4"]` | `RequestId` generation. |
| `slab` | `"0.4"` | Generational arena for `RadixCache` node storage. O(1) insert/remove/lookup by `NodeId`. |
| `rustc-hash` | `"2"` | `FxHashMap<u32, NodeId>` for radix node child lookup. ~2x faster than `std::HashMap` for small integer keys. |
| `thiserror` | `"2"` | `EngineError` derive. Zero-cost error type generation. |

### 9.2 `inference-api`

| Crate | Version | Justification |
|-------|---------|--------------|
| `axum` | `"0.8"` | HTTP server framework. SSE support via `axum::response::Sse`. |
| `tokio` | `"1"` | Same as above; shared workspace dependency. |
| `serde` | `"1"` with feature `["derive"]` | JSON serialization of request/response types. |
| `serde_json` | `"1"` | JSON parsing for request bodies. |
| `tokenizers` | `"0.21"` | HuggingFace tokenizers library. Fast BPE/SentencePiece encode/decode. Chat template application. |
| `tracing` | `"0.1"` | Structured logging. |
| `tracing-subscriber` | `"0.3"` | Log subscriber (stdout, JSON). |
| `uuid` | `"1"` | `ChatCompletionResponse` id generation. |

### 9.3 `inference-backend`

| Crate | Version | Justification |
|-------|---------|--------------|
| `burn` | `"0.16"` | Core tensor framework. |
| `burn-wgpu` | `"0.16"` | WebGPU backend. Vulkan (Linux), Metal (macOS). Only in `[features] wgpu`. |
| `tokio` | `"1"` | `oneshot::channel` for GPU thread result return. |
| `thiserror` | `"2"` | Error types. |

### 9.4 `inference-model-gemma`

| Crate | Version | Justification |
|-------|---------|--------------|
| `burn` | `"0.16"` | Module trait, tensor ops, parameter loading. |
| `safetensors` | `"0.4"` | Load Gemma 3 weights from HuggingFace safetensors format. |
| `gguf` | `"0.1"` or custom parser | Load GGUF files. If a maintained crate is not available at pinned quality, implement a minimal GGUF parser (~200 lines) for the header and BF16/F16 tensor types. |
| `half` | `"2"` | `bf16`, `f16` types for weight loading and KV cache buffers. |
| `memmap2` | `"0.9"` | Memory-map weight files. Avoids reading multi-GB files into heap. |

### 9.5 Workspace `Cargo.toml` settings

```toml
[workspace]
resolver = "2"
members = [
    "inference-engine",
    "inference-api",
    "inference-backend",
    "inference-model-gemma",
]

[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1

[profile.dev]
opt-level = 1   # faster compile than opt-level=0; debug asserts remain active
```

---

## 10. Open Questions

The following questions must be resolved before or during Phase 0 implementation. Each has an owner and a decision deadline.

### Q1: Paged attention implementation strategy

**Question:** `burn-wgpu` does not have a built-in paged attention operator. Phase 0 proposes a gather-then-attend approach (copy KV pages into a contiguous buffer, then run standard attention). Is this approach fast enough to meet the TTFT < 2s target on Gemma 3 4B?

**Options:**
1. Gather-then-attend: contiguous KV copy before each attention call. Simple, correct, possibly 20–40% slower than optimal.
2. Custom WGSL paged attention kernel: requires writing and debugging WGSL compute shaders. Estimated +2 weeks of work.

**Proposed resolution:** Start with gather-then-attend in Phase 0. Benchmark against target. If TTFT > 1.5s, escalate to custom kernel as a Phase 0.5 task before declaring Phase 0 complete.

**Decision needed by:** Before starting `inference-model-gemma/src/model.rs`.

---

### Q2: GGUF parser dependency

**Question:** Is there a maintained Rust GGUF parsing crate at sufficient quality for production use, or do we need to write a minimal custom parser?

**Candidates to evaluate:**
- `gguf-rs` (search crates.io; version and maintenance status unknown as of April 2026).
- `llama-cpp-sys` bindings (heavy, pulls in C build dependency).
- Custom minimal parser (~200 lines, supports only BF16/F16 tensors needed for Phase 0).

**Proposed resolution:** Use a custom minimal parser for Phase 0. The GGUF format is stable and well-documented; the subset needed (header, tensor metadata, BF16/F16 data) is simple. Evaluate crate alternatives for Phase 1.

**Decision needed by:** Before starting `inference-model-gemma/src/loader.rs`.

---

### Q3: Gemma 3 weight availability

**Question:** What is the canonical source for Gemma 3 1B and 4B weights in BF16/F16, and what format (safetensors vs. GGUF) should Phase 0 prioritize?

**Current status:** Gemma 3 weights are available on HuggingFace in safetensors format. GGUF conversions exist but may not cover all variants at BF16 precision.

**Proposed resolution:** Prioritize safetensors loading for Phase 0 (more directly maintained by Google). GGUF loading can be implemented in parallel but is not on the critical path if safetensors works.

**Decision needed by:** Start of implementation sprint.

---

### Q4: KV buffer handle abstraction in `WgpuBackendHandle`

**Question:** The paged KV buffer is a Burn tensor of shape `[num_layers, 2, num_pages, page_size, num_kv_heads, head_dim]`. Burn tensors are generic over backend. `WgpuBackendHandle` must expose KV pool metadata to the engine without exposing Burn types.

**Options:**
1. `WgpuBackendHandle` holds `KvCachePool` (CPU metadata only) and separately holds the Burn tensor. The engine only sees the CPU metadata via `dyn KvPool`.
2. Define a `KvBufferHandle` opaque type wrapping a pointer/handle to the GPU buffer; engine passes it to the backend as part of the `Batch`.

**Proposed resolution:** Option 1. The engine never needs to touch the GPU buffer directly. `KvCachePool` in `inference-engine` is CPU-only metadata. The physical GPU buffer is private to `WgpuBackendHandle`.

**Decision needed by:** Before starting `inference-backend/src/wgpu.rs`.

---

### Q5: Sampling implementation location

**Question:** Should sampling (temperature scaling, top-p/top-k filtering, argmax/multinomial draw) run on the GPU (inside `WgpuBackendHandle`) or on the CPU (inside `process_and_transition` in the engine)?

**Tradeoffs:**
- GPU sampling: avoids transferring `[batch_size, vocab_size]` float array to CPU. But requires a sampling kernel (custom WGSL or Burn op). vocab_size for Gemma 3 is 256K — transferring 256K × 4 bytes × batch_size per iteration could be 1 MB+ at batch 32.
- CPU sampling: simpler, no custom kernel. At batch 1 (Phase 0 target), 256K × 4 = 1 MB transfer per iteration. At 15 tok/s this is 15 MB/s — well within PCIe bandwidth but adds ~1 ms of host-device transfer per step on discrete GPUs. On unified memory (Apple Silicon, Intel iGPU), zero-copy is possible.

**Proposed resolution:** CPU sampling for Phase 0. The target hardware (Apple Silicon, Intel iGPU) has unified memory; the transfer cost is negligible. Revisit for Phase 2 if batch decode throughput targets require it.

**Decision needed by:** Before starting `inference-engine/src/scheduler.rs`.

---

### Q6: Radix cache max_pages sizing

**Question:** How should `RadixCache::max_pages` be set relative to `KvCachePool::num_pages`?

**Consideration:** The radix cache and the KV pool share the same physical GPU memory pages. A page is either in the radix cache (evictable, available for reuse) or in the KV pool as in-use by an active request. Setting `max_pages` too high starves the pool; too low degrades hit rate.

**Proposed resolution:** `max_pages = total_kv_pages`. The radix cache is just the pool's free list organized as a tree. When a page is freed by a request, it goes into the radix cache (not directly to the free list). When the pool needs pages, it calls `radix_cache.evict_pages(n)` to get them back. The pool's `free_pages` is always a subset of `radix_cache.total_cached_pages + pool.free_pages`.

This requires a minor architectural adjustment: `KvCachePool.free` routes through `RadixCache.promote_request` rather than directly returning to the free list. Clarify this in the KV pool / radix cache interaction before implementation.

**Decision needed by:** Before starting `inference-engine/src/kv_pool.rs` and `radix_cache.rs`.

---

*End of Phase 0 Deliverable Specification.*
