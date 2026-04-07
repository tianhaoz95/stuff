# Phase 2A Deliverable Specification: burn-ggml macOS Backend

**Status:** Draft v0.1  
**Date:** 2026-04-04  
**Scope:** Phase 2A only — `burn-ggml` backend on macOS Apple Silicon; Gemma 4 31B dense Q3_K_M; 16 GB unified memory

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

Phase 2A ports the already-validated engine and offloading stack from `burn-wgpu` (Linux) to
`burn-ggml` (macOS Apple Silicon). The scheduler, radix cache, HTTP API, tokenizer service,
`WeightCache`, `KvOffloadManager`, and `PrefetchOps` are all unchanged — they were proven
correct in Phase 0 and Phase 1. Phase 2A **only** changes what is behind the `BackendHandle`
trait.

### What Phase 2A proves

1. **The `BackendHandle` seam is genuinely portable.** Swapping `WgpuOffloadBackendHandle`
   for `GgmlBackendHandle` requires zero changes to `inference-engine` or `inference-api`.
   The seam that was designed in Phase 0 survives a complete backend replacement across OS
   and GPU API boundaries (Vulkan → Metal, WGPU → ggml).

2. **A 31B dense model runs on 16 GB unified memory on Apple Silicon.** Layer weight
   streaming with a 3–4 slot circular buffer (≈1.5 GB VRAM) plus SSD-backed global KV
   ping-pong fits Gemma 4 31B Q3_K_M on a MacBook Air M3 16 GB.

3. **ggml Metal kernels are competitive.** The `burn-ggml` backend achieves ≥ 85% of
   `llama.cpp` direct throughput on the same model and quantization, validating that the
   Burn integration overhead is acceptable.

4. **Hybrid attention (local + global) works correctly with Proportional RoPE.** The 256K
   context window of Gemma 4 31B requires alternating local sliding-window (1024-token) and
   global (full context) attention layers with different RoPE configurations; Phase 2A
   validates this at 8192-token prompt length.

5. **Q3_K_M quantized matmul is correct.** The ggml Metal kernels for Q3_K_M matmul produce
   output within 1% of F32 reference, confirming no numeric catastrophe from the quantization
   path.

### What is unchanged from Phase 0 and Phase 1

| Component | Status |
|-----------|--------|
| `inference-engine` crate | Unchanged — no modifications |
| `inference-api` crate | Unchanged — no modifications |
| `TokenizerService` | Unchanged |
| `RadixCache` | Unchanged |
| Scheduler (`schedule_batch`, `run_overlapped_loop`) | Unchanged |
| `BackendHandle` trait | Unchanged — `prefetch` default no-op from Phase 1 retained |
| `WeightCache<K>` | Unchanged |
| `KvOffloadManager` | Unchanged |
| `PrefetchOps` trait | Unchanged |
| `StubBackend` | Unchanged |
| All Phase 0 and Phase 1 cargo tests | Must continue to pass |

### Platform

- **OS:** macOS 14+ (Sonoma / Sequoia)
- **Hardware:** Apple Silicon M-series (M1 / M2 / M3 / M4)
- **RAM:** 16 GB unified memory
- **SSD:** NVMe (MacBook Pro/Air internal)
- **GPU API:** Apple Metal (via ggml Metal backend)

### Model

- **Gemma 4 31B dense** (NOT MoE — dense FFN, 46 transformer layers)
- **Quantization:** Q3_K_M (≈14.5 GB weights on SSD)
- **Context window:** 256K tokens
- **Attention:** hybrid local (sliding window 1024) + global (full context)
- **RoPE:** standard for local layers; Proportional RoPE (base=1M, scale=0.125) for global layers
- **Shared KV:** last N layers reuse K/V projections from earlier layers

### Backend

- **`burn-ggml`** — ggml C library + Apple Metal GPU via FFI
- **`ggml-sys`** — new crate providing Rust FFI bindings to ggml

### Timeline

- Weeks 15–18 (Phase 1 covered Weeks 7–14; Phase 0 covered Weeks 1–6)

---

## 2. Scope

### In scope

- New crate `ggml-sys`: FFI bindings to ggml/llama.cpp; `build.rs` cmake + bindgen; thin C
  wrapper `ggml_wrapper.c` for stable ABI
- New crate `inference-backend-ggml`: `GgmlBackendHandle`, `GgmlContext`, `GgmlTensor`,
  `GgmlBackend`, `GgmlDevice`, `GgmlQuantizedTensor`
- New crate `inference-model-gemma4-dense`: `Gemma4DenseModel`, `GemmaRunner`, `Gemma4DenseConfig`
- Hybrid attention graph construction: `build_local_attention`, `build_global_attention`
- Quantized tensor ops: `QTensorOps::quantize` (F32 → Q3_K_M), `QTensorOps::dequantize`
- Layer weight streaming prefetch schedule inside `GemmaRunner::decode_step`
- All 7 new correctness tests plus all Phase 0 and Phase 1 tests passing (Section 6)
- All performance targets met on MacBook Air M3 16 GB (Section 7)
- `cargo build` on macOS with no manual steps beyond `git submodule update --init`
- AddressSanitizer clean: no leaks in `GgmlContext` / `GgmlTensor`

### Explicitly out of scope

- Any change to `inference-engine`, `inference-api`, `WeightCache`, or `KvOffloadManager`
  (the `BackendHandle` seam must not move)
- ANE (Apple Neural Engine) acceleration — Phase 3+
- MoE expert routing on macOS — Gemma 4 31B is dense; MoE remains Linux-only Phase 1
- Multi-GPU or distributed inference — not planned
- iOS / tvOS / watchOS — not planned
- Windows or Linux support for the `burn-ggml` backend — macOS only in Phase 2A
- Gemma 4 31B IT (instruction-tuned) fine-tuning — inference only
- Models other than Gemma 4 31B dense — deferred to Phase 2B
- Quantization schemes other than Q3_K_M and Q4_K_M — deferred
- Prometheus metrics endpoint — stretch goal, not a hard gate
- Priority scheduling beyond FCFS — unchanged from Phase 0
- Continuous batching improvements — Phase 2B
- Speculative decoding — Phase 3+

---

## 3. New Components

This section specifies every new component introduced in Phase 2A. Existing components
(`inference-engine`, `inference-api`, `WeightCache`, `KvOffloadManager`) are not modified.

---

### 3.1 `ggml-sys` — FFI Bindings Crate

**Crate path:** `ggml-sys/`  
**Purpose:** Compile llama.cpp as a static library with Metal enabled, run bindgen over the
ggml public headers, and expose a thin Rust-safe C ABI wrapper.

#### 3.1.1 `build.rs`

```rust
// ggml-sys/build.rs
use std::{env, path::PathBuf};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let llama_src = PathBuf::from("vendor/llama.cpp");

    // 1. CMake configure
    cmake::Config::new(&llama_src)
        .define("LLAMA_METAL", "ON")
        .define("LLAMA_STATIC", "ON")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release")
        .build_target("ggml")
        .build();

    // 2. Link static library and Apple frameworks
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-search={}", out_dir.join("build/src").display());
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalKit");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Accelerate");

    // 3. Thin C wrapper (stable ABI surface)
    cc::Build::new()
        .file("src/ggml_wrapper.c")
        .include(&llama_src)
        .include(llama_src.join("ggml/include"))
        .compile("ggml_wrapper");
    println!("cargo:rustc-link-lib=static=ggml_wrapper");

    // 4. bindgen
    let bindings = bindgen::Builder::default()
        .header("src/wrapper.h")
        .clang_arg(format!("-I{}", llama_src.display()))
        .clang_arg(format!("-I{}", llama_src.join("ggml/include").display()))
        .allowlist_type("ggml_.*")
        .allowlist_function("ggml_.*")
        .allowlist_var("GGML_.*")
        .derive_debug(false)
        .layout_tests(false)
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings");
}
```

#### 3.1.2 `src/wrapper.h`

```c
// ggml-sys/src/wrapper.h
// Public headers exposed to bindgen.
#include "ggml/include/ggml.h"
#include "ggml/include/ggml-alloc.h"
#include "ggml/include/ggml-backend.h"
```

#### 3.1.3 `src/ggml_wrapper.c` — Thin C wrapper for stable ABI

```c
// ggml-sys/src/ggml_wrapper.c
// Exposes a stable extern "C" surface so that bindgen-generated symbols do not
// depend on inlined or static inline functions in ggml headers.
#include "ggml/include/ggml.h"
#include "ggml/include/ggml-alloc.h"
#include "ggml/include/ggml-backend.h"

// Backend initializers
struct ggml_backend * ggmlw_metal_init(void) {
    return ggml_backend_metal_init();
}

struct ggml_backend * ggmlw_cpu_init(void) {
    return ggml_backend_cpu_init();
}

// Scheduler
struct ggml_backend_sched * ggmlw_sched_new(
    struct ggml_backend ** backends,
    struct ggml_backend_buffer_type_t ** buf_types,
    int n_backends,
    size_t graph_size
) {
    return ggml_backend_sched_new(backends, buf_types, n_backends, graph_size, false);
}

// Quantize F32 -> Q3_K_M in-place
size_t ggmlw_quantize_q3_k(
    const float * src,
    void       * dst,
    int64_t      nrows,
    int64_t      n_per_row,
    const float * imatrix
) {
    return ggml_quantize_q3_K(src, dst, nrows, n_per_row, imatrix);
}
```

#### 3.1.4 Key bindgen-generated types

| Rust type | C type | Notes |
|-----------|--------|-------|
| `ggml_context` | `struct ggml_context` | Owns tensor allocations |
| `ggml_tensor` | `struct ggml_tensor` | 4-D; `.data` pointer into backend buffer |
| `ggml_cgraph` | `struct ggml_cgraph` | Forward computation graph |
| `ggml_backend_t` | `ggml_backend_t` (opaque ptr) | Metal or CPU |
| `ggml_backend_buffer_t` | `ggml_backend_buffer_t` | Device memory allocation |
| `ggml_backend_sched_t` | `ggml_backend_sched_t` | Multi-backend scheduler |

Key functions exposed:

```
ggml_backend_metal_init()
ggml_backend_cpu_init()
ggml_backend_free()
ggml_backend_sched_new()
ggml_backend_sched_graph_compute()
ggml_backend_sched_free()
ggml_new_context_default()
ggml_free()
ggml_new_tensor_4d()
ggml_mul_mat()
ggml_rms_norm()
ggml_rope_ext()
ggml_flash_attn_ext()
ggml_mul_mat_id()      // not used for 31B dense; included for completeness
ggml_graph_compute_with_ctx()
ggml_backend_alloc_ctx_tensors()
```

---

### 3.2 `GgmlBackendHandle` — implements `BackendHandle`

**Crate:** `inference-backend-ggml`  
**File:** `src/handle.rs`

```rust
use std::{path::PathBuf, sync::Arc};
use inference_backend::{BackendHandle, Batch, ForwardOutput, KvPool, ModelConfig};
use crate::{GgmlContext, model::Gemma4DenseModel, weight_cache::LayerWeightCache};
use inference_offload::{KvOffloadManager};

pub struct GgmlBackendHandle {
    ctx:         Arc<GgmlContext>,
    model:       Arc<Gemma4DenseModel>,
    layer_cache: Arc<LayerWeightCache>,   // Phase 1 WeightCache<LayerKey>
    kv_offload:  Arc<KvOffloadManager>,   // Phase 1 KvOffloadManager
}

impl GgmlBackendHandle {
    pub fn new(
        model_path:       PathBuf,
        kv_cache_dir:     PathBuf,
        max_layers_in_ram: usize,          // circular buffer slots (3-4)
        config:           Gemma4DenseConfig,
    ) -> anyhow::Result<Self> {
        let ctx         = Arc::new(GgmlContext::new_metal()?);
        let layer_cache = Arc::new(LayerWeightCache::new(max_layers_in_ram, model_path.clone())?);
        let kv_offload  = Arc::new(KvOffloadManager::new(kv_cache_dir, &config.kv_config())?);
        let model       = Arc::new(Gemma4DenseModel::new(
            ctx.clone(), layer_cache.clone(), kv_offload.clone(), config,
        )?);
        Ok(Self { ctx, model, layer_cache, kv_offload })
    }
}

impl BackendHandle for GgmlBackendHandle {
    fn forward(
        &self,
        batch: &Batch,
    ) -> impl std::future::Future<Output = ForwardOutput> + Send {
        let model = self.model.clone();
        let batch = batch.clone();
        async move { model.runner().decode_step(&batch).await }
    }

    fn kv_pool(&self) -> &dyn KvPool {
        self.kv_offload.kv_pool()
    }

    fn model_config(&self) -> &ModelConfig {
        self.model.config()
    }

    fn prefetch(&self, batch: &Batch) {
        // Fire layer weight prefetch and KV prefetch in parallel.
        // Both are async but we intentionally do not await — they enqueue
        // background I/O and return immediately.
        self.layer_cache.prefetch(batch.next_layer_hint());
        self.kv_offload.prefetch_next(batch.next_global_layer_hint());
    }
}
```

---

### 3.3 `GgmlContext` — owns ggml lifetime

**File:** `src/context.rs`

```rust
use std::ptr;
use crate::ggml_sys;

pub struct GgmlContext {
    pub(crate) ctx:         *mut ggml_sys::ggml_context,
    pub(crate) backend:     *mut ggml_sys::ggml_backend,      // Metal (primary)
    pub(crate) cpu_backend: *mut ggml_sys::ggml_backend,      // CPU (fallback)
    pub(crate) sched:       *mut ggml_sys::ggml_backend_sched, // multi-backend scheduler
    pub(crate) device:      GgmlDevice,
}

// Safety: all mutations to ggml state are serialized through a single
// ggml_backend_sched_graph_compute call per forward pass, which itself
// is &mut-guarded via the Tokio task structure.
unsafe impl Send for GgmlContext {}
unsafe impl Sync for GgmlContext {}

impl GgmlContext {
    /// Initialize Metal (primary) + CPU (fallback) backends and the scheduler.
    pub fn new_metal() -> anyhow::Result<Self> {
        unsafe {
            let backend = ggml_sys::ggmlw_metal_init();
            anyhow::ensure!(!backend.is_null(), "ggml Metal init failed");

            let cpu_backend = ggml_sys::ggmlw_cpu_init();
            anyhow::ensure!(!cpu_backend.is_null(), "ggml CPU init failed");

            let mut backends  = [backend, cpu_backend];
            let mut buf_types = [
                ggml_sys::ggml_backend_get_default_buffer_type(backend),
                ggml_sys::ggml_backend_get_default_buffer_type(cpu_backend),
            ];
            let sched = ggml_sys::ggmlw_sched_new(
                backends.as_mut_ptr(),
                buf_types.as_mut_ptr(),
                2,
                GGML_DEFAULT_GRAPH_SIZE,
            );
            anyhow::ensure!(!sched.is_null(), "ggml sched init failed");

            // ctx: 512 MB scratch for activations; no persistent tensors here
            let params = ggml_sys::ggml_init_params {
                mem_size:   512 * 1024 * 1024,
                mem_buffer: ptr::null_mut(),
                no_alloc:   false,
            };
            let ctx = ggml_sys::ggml_init(params);
            anyhow::ensure!(!ctx.is_null(), "ggml context init failed");

            Ok(Self { ctx, backend, cpu_backend, sched, device: GgmlDevice::Metal })
        }
    }
}

impl Drop for GgmlContext {
    fn drop(&mut self) {
        unsafe {
            if !self.sched.is_null()       { ggml_sys::ggml_backend_sched_free(self.sched); }
            if !self.ctx.is_null()         { ggml_sys::ggml_free(self.ctx); }
            if !self.backend.is_null()     { ggml_sys::ggml_backend_free(self.backend); }
            if !self.cpu_backend.is_null() { ggml_sys::ggml_backend_free(self.cpu_backend); }
        }
    }
}

const GGML_DEFAULT_GRAPH_SIZE: usize = 4096;
```

---

### 3.4 `GgmlTensor` — Burn tensor primitive backed by ggml

**File:** `src/tensor.rs`

```rust
use std::sync::Arc;
use crate::{GgmlContext, ggml_sys};

/// A Burn tensor primitive whose storage lives in a ggml_tensor.
/// This is the type that satisfies `Backend::FloatTensorPrimitive`.
pub struct GgmlTensor {
    pub(crate) ptr:   *mut ggml_sys::ggml_tensor,
    pub(crate) ctx:   Arc<GgmlContext>,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: ggml_sys::ggml_type,
}

// Safety: GgmlTensor is only mutated inside ggml graph compute, which is
// serialized by the Tokio runtime (one forward pass at a time).
unsafe impl Send for GgmlTensor {}
unsafe impl Sync for GgmlTensor {}

pub struct GgmlQuantizedTensor {
    pub(crate) ptr:   *mut ggml_sys::ggml_tensor,
    pub(crate) ctx:   Arc<GgmlContext>,
    pub(crate) shape: Vec<usize>,
    pub(crate) qtype: ggml_sys::ggml_type, // GGML_TYPE_Q3_K or GGML_TYPE_Q4_K
}

unsafe impl Send for GgmlQuantizedTensor {}
unsafe impl Sync for GgmlQuantizedTensor {}
```

---

### 3.5 `GgmlBackend` — implements Burn `Backend` trait

**File:** `src/backend.rs`

```rust
use std::path::PathBuf;
use burn::backend::Backend;

#[derive(Clone, Debug, Default)]
pub struct GgmlBackend;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GgmlDevice {
    Cpu,
    Metal,
    MetalWithOffload {
        kv_cache_dir:      PathBuf,
        max_layers_in_ram: usize,
    },
}

impl Backend for GgmlBackend {
    type Device                  = GgmlDevice;
    type FloatTensorPrimitive    = GgmlTensor;
    type FloatElem               = f32;
    type IntTensorPrimitive      = GgmlTensor;
    type IntElem                 = i32;
    type BoolTensorPrimitive     = GgmlTensor;
    type QuantizedTensorPrimitive = GgmlQuantizedTensor;
    type QuantizedEncoding       = burn::tensor::quantization::QuantizationScheme;
}
```

The `Backend` impl delegates all ops to ggml graph nodes via `GgmlContext`. Float ops
(`matmul`, `rms_norm`, `rope`) build ggml graph nodes; execution is deferred until
`ggml_backend_sched_graph_compute` is called at the end of each layer.

---

### 3.6 `Gemma4DenseModel` — 31B dense Burn module

**Crate:** `inference-model-gemma4-dense`  
**File:** `src/model.rs`

#### 3.6.1 Architecture constants

```rust
pub struct Gemma4DenseConfig {
    pub num_layers:          usize,   // 46
    pub hidden_size:         usize,   // 5120
    pub intermediate_size:   usize,   // 27648
    pub num_attention_heads: usize,   // 32
    pub num_kv_heads:        usize,   // 16  (GQA)
    pub head_dim:            usize,   // 256
    pub vocab_size:          usize,   // 262144
    pub max_position_embeddings: usize, // 131072 (256K tokens = 2 x this for sliding rope)
    pub sliding_window:      usize,   // 1024 (local attention)
    pub rope_local_base:     f32,     // 10000.0
    pub rope_global_base:    f32,     // 1_000_000.0
    pub rope_global_scale:   f32,     // 0.125 (Proportional RoPE)
    pub rms_norm_eps:        f32,     // 1e-6
    pub shared_kv_layers:    usize,   // layers that reuse K/V from an earlier layer
}
```

#### 3.6.2 Layer type classification

```rust
/// Each of the 46 layers is either Local (sliding window) or Global (full context).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType {
    Local,   // layers 0, 2, 4, ... (even indices, sliding window 1024)
    Global,  // layers 1, 3, 5, ... (odd indices, Proportional RoPE)
}

impl AttentionType {
    pub fn for_layer(layer_idx: usize) -> Self {
        if layer_idx % 2 == 0 { Self::Local } else { Self::Global }
    }
}
```

#### 3.6.3 `GemmaRunner::decode_step` — layer streaming prefetch schedule

```rust
impl GemmaRunner {
    /// Run a full forward pass with layer weight streaming.
    /// Prefetch schedule: while layer i is computing on Metal, load layer i+2 from SSD.
    pub async fn decode_step(&self, batch: &Batch) -> ForwardOutput {
        let n_layers = self.config.num_layers;

        // Initial prefetch: seed the circular buffer with layers 0 and 1
        self.layer_cache.load_layer(0).await;
        self.layer_cache.load_layer(1).await;

        let mut hidden = self.embed_tokens(batch);

        for i in 0..n_layers {
            // Enqueue prefetch for layer i+2 (returns immediately)
            if i + 2 < n_layers {
                self.layer_cache.prefetch(i + 2);
            }

            // For global layers, also enqueue KV prefetch for the *next* global layer
            if AttentionType::for_layer(i) == AttentionType::Global {
                let next_global = (i + 2..n_layers)
                    .find(|&j| AttentionType::for_layer(j) == AttentionType::Global);
                if let Some(ng) = next_global {
                    self.kv_offload.prefetch_next(ng);
                }
            }

            // Acquire layer weights (blocks only if prefetch not yet complete)
            let weights = self.layer_cache.acquire(i).await;

            // Build ggml graph for this layer and compute
            let layer_out = self.run_layer(i, &hidden, &weights, batch).await;

            hidden = layer_out;

            // Release layer i weights back to circular buffer
            self.layer_cache.release(i);
        }

        self.lm_head(&hidden)
    }
}
```

---

### 3.7 Quantized Matmul (Q3_K_M, Q4_K_M)

**File:** `src/ops/quantize.rs`

Phase 2A adds `QTensorOps` for the `GgmlBackend`:

```rust
impl QTensorOps<GgmlBackend> for GgmlBackend {
    fn quantize(
        tensor: &GgmlTensor,
        scheme: QuantizationScheme,
    ) -> GgmlQuantizedTensor {
        // scheme must be Q3_K_M or Q4_K_M
        let qtype = match scheme {
            QuantizationScheme::Q3_K_M => ggml_sys::GGML_TYPE_Q3_K,
            QuantizationScheme::Q4_K_M => ggml_sys::GGML_TYPE_Q4_K,
            other => panic!("unsupported quant scheme: {:?}", other),
        };
        let q_tensor = unsafe {
            // Allocate quantized buffer
            let n_elems = tensor.shape.iter().product::<usize>();
            let buf_size = ggml_sys::ggml_type_size(qtype)
                * n_elems / ggml_sys::ggml_blck_size(qtype) as usize;
            let q_ptr = libc::malloc(buf_size);
            ggml_sys::ggmlw_quantize_q3_k(
                tensor.ptr as *const f32,
                q_ptr,
                tensor.shape[0] as i64,          // nrows
                tensor.shape[1..].iter().product::<usize>() as i64, // n_per_row
                std::ptr::null(),                // no importance matrix
            );
            q_ptr as *mut ggml_sys::ggml_tensor
        };
        GgmlQuantizedTensor { ptr: q_tensor, ctx: tensor.ctx.clone(), shape: tensor.shape.clone(), qtype }
    }

    fn dequantize(tensor: &GgmlQuantizedTensor) -> GgmlTensor {
        // Build a ggml_op_get_rows or ggml_cast node — executed in next graph compute
        unsafe {
            let f32_tensor = ggml_sys::ggml_cast(
                tensor.ctx.ctx,
                tensor.ptr,
                ggml_sys::GGML_TYPE_F32,
            );
            GgmlTensor {
                ptr:   f32_tensor,
                ctx:   tensor.ctx.clone(),
                shape: tensor.shape.clone(),
                dtype: ggml_sys::GGML_TYPE_F32,
            }
        }
    }
}
```

`ggml_mul_mat` with a Q3_K_M weight tensor and F32 activation tensor is handled natively
by ggml Metal kernels — no explicit dequantization is required on the hot path.

---

### 3.8 Hybrid Attention ggml Graph Construction

**File:** `src/ops/attention.rs`

#### 3.8.1 Local attention (sliding window, standard RoPE)

```rust
/// Build the ggml compute graph for a local (sliding-window) attention layer.
/// window_size: number of tokens in the causal sliding window (1024 for Gemma 4 31B)
pub unsafe fn build_local_attention(
    ctx:         *mut ggml_sys::ggml_context,
    q:           *mut ggml_sys::ggml_tensor,  // [batch, n_heads, seq_len, head_dim]
    k:           *mut ggml_sys::ggml_tensor,  // [batch, n_kv_heads, kv_len, head_dim]
    v:           *mut ggml_sys::ggml_tensor,  // [batch, n_kv_heads, kv_len, head_dim]
    rope_base:   f32,                          // 10000.0
    window_size: usize,                        // 1024
    is_causal:   bool,                         // always true for decode
) -> *mut ggml_sys::ggml_tensor {
    // Apply standard RoPE to Q and K
    let q_rope = ggml_sys::ggml_rope_ext(
        ctx, q,
        /*positions*/ std::ptr::null_mut(),
        /*freq_factors*/ std::ptr::null_mut(),
        /*n_dims*/ 0,
        /*mode*/ ggml_sys::LLAMA_ROPE_TYPE_NEOX as i32,
        /*n_ctx_orig*/ 0,
        rope_base,
        /*freq_scale*/ 1.0f32,
        /*ext_factor*/ 0.0f32,
        /*attn_factor*/ 1.0f32,
        /*beta_fast*/ 32.0f32,
        /*beta_slow*/ 1.0f32,
    );
    let k_rope = ggml_sys::ggml_rope_ext(
        ctx, k, std::ptr::null_mut(), std::ptr::null_mut(),
        0, ggml_sys::LLAMA_ROPE_TYPE_NEOX as i32, 0,
        rope_base, 1.0, 0.0, 1.0, 32.0, 1.0,
    );

    // Sliding-window causal mask is applied inside flash_attn_ext via max_bias=0
    // and n_ctx window parameter
    ggml_sys::ggml_flash_attn_ext(
        ctx, q_rope, k_rope, v,
        /*mask*/ std::ptr::null_mut(),
        /*scale*/ (1.0f32 / (256.0f32.sqrt())),   // 1/sqrt(head_dim)
        /*max_bias*/ 0.0f32,
        /*logit_softcap*/ 50.0f32,  // Gemma 4 uses logit softcap
    )
}
```

#### 3.8.2 Global attention (full context, Proportional RoPE)

```rust
/// Build the ggml compute graph for a global (full-context) attention layer.
/// Uses Proportional RoPE: rope_base=1_000_000, rope_scale=0.125.
pub unsafe fn build_global_attention(
    ctx:        *mut ggml_sys::ggml_context,
    q:          *mut ggml_sys::ggml_tensor,
    k:          *mut ggml_sys::ggml_tensor,
    v:          *mut ggml_sys::ggml_tensor,
    rope_base:  f32,   // 1_000_000.0
    rope_scale: f32,   // 0.125
    is_causal:  bool,
) -> *mut ggml_sys::ggml_tensor {
    // Proportional RoPE: freq_scale = rope_scale (compresses position frequencies)
    let q_rope = ggml_sys::ggml_rope_ext(
        ctx, q, std::ptr::null_mut(), std::ptr::null_mut(),
        0, ggml_sys::LLAMA_ROPE_TYPE_NEOX as i32, 0,
        rope_base,
        rope_scale,   // freq_scale = 0.125 for Proportional RoPE
        0.0, 1.0, 32.0, 1.0,
    );
    let k_rope = ggml_sys::ggml_rope_ext(
        ctx, k, std::ptr::null_mut(), std::ptr::null_mut(),
        0, ggml_sys::LLAMA_ROPE_TYPE_NEOX as i32, 0,
        rope_base, rope_scale, 0.0, 1.0, 32.0, 1.0,
    );

    // Full-context causal attention via ggml_flash_attn_ext on Metal
    ggml_sys::ggml_flash_attn_ext(
        ctx, q_rope, k_rope, v,
        std::ptr::null_mut(),
        1.0f32 / (256.0f32.sqrt()),
        0.0f32,
        50.0f32,  // logit_softcap
    )
}
```

---

## 4. Memory Architecture

### 4.1 16 GB Unified Memory Budget

Apple Silicon uses unified memory — the Metal GPU and CPU share the same physical DRAM.
The following budget allocates that memory for Gemma 4 31B Q3_K_M inference.

```
Component                           Size        Notes
─────────────────────────────────────────────────────────────────────────────
OS + runtime                       ~1.5 GB     macOS kernel, dylibs, Tokio
Layer weight circular buffer       ~1.5 GB     3–4 layer slots × ~315 MB/layer (Q3_K_M)
  slot_0 (current layer)            315 MB
  slot_1 (current+1, prefetch)      315 MB
  slot_2 (current+2, prefetch)      315 MB
  slot_3 (spare / double-buffer)    315 MB
Local attention KV cache (RAM)     ~0.3 GB     38 local layers × ~8 MB
  = 38 × 2 × n_kv_heads × head_dim × sliding_window × dtype_size
  = 38 × 2 × 16 × 256 × 1024 × 2B (INT16) ≈ 0.3 GB
Global KV ping buffer (RAM)        ~1.1 GB     1 global layer, FP16 → INT8 at 8K ctx
Global KV pong buffer (RAM)        ~1.1 GB     prefetch slot for next global layer
Activation scratch / ggml ctx       ~0.5 GB    forward pass intermediates
Headroom                           ~1.0 GB     Metal driver, OS peaks
─────────────────────────────────────────────────────────────────────────────
Total RAM used                     ~7.0 GB     fits within 16 GB ✓

Remaining weights on SSD           ~13 GB      43–46 layer slots not in RAM
Global KV cache on SSD             ~8.5 GB     INT8 (full 256K context, all global layers)
                                   ~17 GB      FP16 (alternative, requires SSD headroom)
```

**Why 3–4 layer slots fit in 1.5 GB:**
Gemma 4 31B Q3_K_M: each layer has Q, K, V, O projections and the dense FFN
(gate, up, down). At 5120 hidden dim × 3 bit average ≈ 315 MB per layer.

### 4.2 Layer Weight Circular Buffer

The `LayerWeightCache` (Phase 1 `WeightCache<LayerKey>`) maintains a circular buffer
of N=4 slots. The invariant is:

```
 SSD layout:
 ┌────────┬────────┬────────┬ ··· ┬────────┐
 │layer_0 │layer_1 │layer_2 │     │layer_45│   (Q3_K_M GGUF on NVMe SSD)
 └────────┴────────┴────────┴ ··· ┴────────┘
       14.5 GB total, sequential layout for fast streaming

 RAM circular buffer (4 slots, ~1.5 GB):
 ┌──────────┬──────────┬──────────┬──────────┐
 │ slot_0   │ slot_1   │ slot_2   │ slot_3   │
 │ layer_i  │layer_i+1 │layer_i+2 │ (loading)│
 │ COMPUTE  │ READY    │ READY    │ PREFETCH │
 └──────────┴──────────┴──────────┴──────────┘
      ↑
  currently executing on Metal
                                       ↑
                               background SSD read
```

State transitions:
1. `PREFETCH`: background Tokio task reads layer weights from GGUF file on SSD
2. `READY`: DMA into unified memory complete, pinned for Metal access
3. `COMPUTE`: Metal GPU is executing this layer's ggml graph
4. After `release(i)`: slot is returned to pool, evicted by LRU

### 4.3 KV Cache Layout on SSD

Global KV layers produce large KV tensors that cannot fit in RAM for 256K context.
They are offloaded to SSD using the Phase 1 `KvOffloadManager`.

```
SSD directory: $KV_CACHE_DIR/
├── global_layer_01/        (layer index 1 — first global layer)
│   ├── k_cache.int8        (n_kv_heads × max_seq × head_dim, INT8 quantized)
│   └── v_cache.int8
├── global_layer_03/
│   ├── k_cache.int8
│   └── v_cache.int8
├── ...
└── global_layer_45/        (layer index 45 — last global layer)
    ├── k_cache.int8
    └── v_cache.int8

Total global layers: 23 (odd indices 1,3,...,45)
KV tensor per layer at 8192-token context (INT8):
  2 × 16 kv_heads × 8192 seq × 256 head_dim × 1B = 67 MB
  23 layers × 67 MB = 1.5 GB at 8K context
  23 layers × 67 MB × 32 (scale to 256K) = 48 GB at 256K context (SSD backed)
```

### 4.4 Ping-Pong Double Buffer for Global KV

While a global attention layer is executing with `ping` buffer, the `pong` buffer
loads the next global layer's KV data from SSD:

```
Time →

Metal:    [global_layer_1 compute (ping)]   [global_layer_3 compute (pong)]
SSD→RAM:       [load global_layer_3 (pong)]      [load global_layer_5 (ping)]

Overlap:  SSD read of layer N+2 overlaps Metal compute of layer N.
Stall:    Only if SSD read takes longer than Metal compute.
          At Q3_K_M decode rates (~10 tok/s), each layer takes ~20ms.
          67 MB at SSD sequential read ~3 GB/s = ~22ms — tight but feasible at INT8.
```

---

## 5. Data Flow

The following diagram traces a single decode step through the full stack.

```
HTTP Client
    │
    ▼
inference-api (axum)              [unchanged from Phase 0]
    │  POST /v1/completions
    │  SSE stream
    ▼
TokenizerService                  [unchanged]
    │  encode(prompt) → token_ids
    ▼
Scheduler (run_overlapped_loop)   [unchanged]
    │  schedule_batch() → Batch
    │  collect_decode_requests()
    ▼
BackendHandle::forward(batch)     ← GgmlBackendHandle (NEW in Phase 2A)
    │
    ├─► GgmlBackendHandle::prefetch(batch)
    │       ├─ LayerWeightCache::prefetch(i+2)  → SSD read (background, async)
    │       └─ KvOffloadManager::prefetch_next(next_global) → SSD read (background)
    │
    └─► GemmaRunner::decode_step(batch)
            │
            ├─ embed_tokens(batch) → hidden [Metal, ggml_mul_mat on embedding table]
            │
            ├─ for layer i in 0..46:
            │   │
            │   ├─ LayerWeightCache::acquire(i)  ← blocks if prefetch not done
            │   │       Weights: Q,K,V,O,gate,up,down (315 MB)
            │   │       Loaded from: SSD GGUF → unified RAM → Metal buffer
            │   │
            │   ├─ if AttentionType::Local(i):
            │   │   ├─ ggml_rms_norm(hidden)
            │   │   ├─ ggml_mul_mat(Q_weight, hidden)   → q  [Q3_K_M × F32, Metal]
            │   │   ├─ ggml_mul_mat(K_weight, hidden)   → k
            │   │   ├─ ggml_mul_mat(V_weight, hidden)   → v
            │   │   ├─ KvPool::append_local(i, k, v)    → local RAM KV cache
            │   │   ├─ build_local_attention(ctx,q,k,v,rope_base=10000,window=1024)
            │   │   │       ggml_rope_ext (standard RoPE)
            │   │   │       ggml_flash_attn_ext (Metal, sliding window causal mask)
            │   │   ├─ ggml_mul_mat(O_weight, attn_out) → attn_proj
            │   │   ├─ residual add
            │   │   ├─ ggml_rms_norm
            │   │   ├─ ggml_mul_mat(gate), silu, ggml_mul_mat(up), elementwise mul
            │   │   ├─ ggml_mul_mat(down)              → ffn_out
            │   │   └─ residual add → hidden
            │   │
            │   └─ if AttentionType::Global(i):
            │       ├─ ggml_rms_norm(hidden)
            │       ├─ ggml_mul_mat(Q,K,V weights)     → q, k, v
            │       ├─ KvOffloadManager::load_kv(i)    ← ping buffer (may block if pong not ready)
            │       ├─ build_global_attention(ctx,q,k,v,rope_base=1e6,rope_scale=0.125)
            │       │       ggml_rope_ext (Proportional RoPE)
            │       │       ggml_flash_attn_ext (Metal, full context)
            │       ├─ KvOffloadManager::store_kv(i)   → write updated KV to pong (async SSD)
            │       ├─ ggml_mul_mat(O_weight, attn_out)
            │       ├─ residual add, rms_norm, ffn
            │       └─ residual add → hidden
            │
            │       [ping/pong swap happens here for next global layer]
            │
            ├─ ggml_backend_sched_graph_compute(sched)  ← executes all queued ggml ops on Metal
            │       (called once per layer to keep activation memory bounded)
            │
            └─ lm_head(hidden) → logits → sample → token_id
                    │
                    ▼
            Scheduler: decode_token → ForwardOutput
                    │
                    ▼
            SSE stream → HTTP Client
```

### 5.1 Prefetch Timing

```
Decode step timeline (single layer, wall clock):

t=0ms   ┌─ LayerWeightCache::prefetch(i+2) issued (non-blocking)
        │  KvOffloadManager::prefetch_next(ng) issued (non-blocking)
        │
t=0ms   ├─ LayerWeightCache::acquire(i) → weights already resident ✓
        │
t=0ms   └─ Metal ggml graph compute begins for layer i
              ...
t=20ms  Metal compute done for layer i
        LayerWeightCache::release(i)

        Layer i+2 prefetch (315 MB @ ~3 GB/s SSD) ≈ 105ms — completes by layer i+1 end
        (two layers of compute = ~40ms covers the 105ms SSD read only if overlapped)
        → Real overlap: prefetch i+2 while computing i and i+1 (~40ms window) is tight;
          the implementation uses a 4-slot buffer so layer i+3 is also being loaded.
```

---

## 6. Test Specification

All Phase 0 and Phase 1 tests must continue to pass without modification. The following
7 new tests are required for Phase 2A acceptance.

---

### Test 6.1 — Semantic correctness: geography

**Location:** `inference-backend-ggml/tests/integration.rs`  
**Name:** `test_gemma4_31b_beijing`

```rust
#[tokio::test]
#[ignore = "requires model weights on SSD"]
async fn test_gemma4_31b_beijing() {
    let handle = make_ggml_handle_for_testing(); // loads Gemma 4 31B Q3_K_M
    let output = generate(&handle, "What is the capital of China?", 64).await;
    assert!(
        output.to_lowercase().contains("beijing"),
        "Expected 'beijing' in output, got: {output:?}"
    );
}
```

**Assertion:** `output.to_lowercase().contains("beijing")` must be true.  
**Platform:** macOS Apple Silicon (Metal).  
**Model:** Gemma 4 31B dense Q3_K_M.

---

### Test 6.2 — Semantic correctness: arithmetic

**Name:** `test_gemma4_31b_arithmetic`

```rust
#[tokio::test]
#[ignore = "requires model weights on SSD"]
async fn test_gemma4_31b_arithmetic() {
    let handle = make_ggml_handle_for_testing();
    let output = generate(
        &handle,
        "What is 1+1? Answer with only a number.",
        8,
    ).await;
    assert_eq!(
        output.trim(),
        "2",
        "Expected '2', got: {output:?}"
    );
}
```

**Assertion:** `output.trim() == "2"` (exact string match after trimming whitespace).

---

### Test 6.3 — Layer streaming correctness

**Name:** `test_layer_streaming_identical_output`

```rust
#[tokio::test]
#[ignore = "requires model weights on SSD"]
async fn test_layer_streaming_identical_output() {
    let prompt = "Describe the water cycle in two sentences.";

    // Reference: all 46 layers resident in RAM (N=46 slots)
    let handle_full = make_ggml_handle_with_slots(46);
    let output_full = generate_deterministic(&handle_full, prompt, 128).await;

    // Streaming: 4-slot circular buffer
    let handle_stream = make_ggml_handle_with_slots(4);
    let output_stream = generate_deterministic(&handle_stream, prompt, 128).await;

    assert_eq!(
        output_full, output_stream,
        "Layer streaming (N=4) must produce bit-identical output to all-resident (N=46)"
    );
}
```

**Assertion:** Token sequences must be bit-identical (same greedy decoding, same logits).  
**Note:** `generate_deterministic` uses temperature=0.0 (greedy), seed=42.

---

### Test 6.4 — KV ping-pong correctness (macOS)

**Name:** `test_kv_pingpong_identical_output`

```rust
#[tokio::test]
#[ignore = "requires model weights on SSD"]
async fn test_kv_pingpong_identical_output() {
    let prompt = "Explain quantum entanglement.";

    // Reference: global KV fully in RAM (no SSD offload)
    let handle_ram = make_ggml_handle_kv_in_ram();
    let output_ram = generate_deterministic(&handle_ram, prompt, 128).await;

    // SSD-backed global KV with ping-pong
    let handle_ssd = make_ggml_handle_kv_on_ssd();
    let output_ssd = generate_deterministic(&handle_ssd, prompt, 128).await;

    assert_eq!(
        output_ram, output_ssd,
        "SSD-backed KV ping-pong must produce identical output to RAM-resident KV"
    );
}
```

**Assertion:** Token sequences must be bit-identical.

---

### Test 6.5 — Quantized matmul correctness (Q3_K_M)

**Name:** `test_q3km_matmul_within_tolerance`

```rust
#[test]
fn test_q3km_matmul_within_tolerance() {
    // Generate a random F32 weight matrix and input vector
    let weight_f32 = random_f32_matrix(4096, 4096, /*seed=*/0);
    let input_f32  = random_f32_vector(4096, /*seed=*/1);

    // F32 reference matmul
    let output_f32 = matmul_f32(&weight_f32, &input_f32);

    // Q3_K_M quantize + dequantize weight, then matmul
    let weight_q3k = quantize_q3km(&weight_f32);
    let output_q3k = matmul_q3km_x_f32(&weight_q3k, &input_f32);

    // Check element-wise relative error < 1%
    let max_rel_err = output_f32.iter().zip(output_q3k.iter())
        .map(|(r, q)| (r - q).abs() / r.abs().max(1e-6))
        .fold(0.0f32, f32::max);

    assert!(
        max_rel_err < 0.01,
        "Q3_K_M matmul max relative error {max_rel_err:.4} exceeds 1% threshold"
    );
}
```

**Assertion:** Max element-wise relative error < 1% for a 4096×4096 random F32 matrix.

---

### Test 6.6 — Proportional RoPE correctness

**Name:** `test_proportional_rope_matches_reference`

```rust
#[test]
fn test_proportional_rope_matches_reference() {
    // HuggingFace reference values for Gemma 4 31B global layer RoPE at 8K positions
    // Pre-computed offline and checked into test fixtures.
    let reference: Vec<f32> = load_fixture("tests/fixtures/gemma4_31b_rope_8k_ref.bin");

    let ctx = GgmlContext::new_metal().unwrap();
    let q   = load_fixture_tensor(&ctx, "tests/fixtures/gemma4_31b_q_input.bin");

    let q_rope = unsafe {
        apply_proportional_rope(&ctx, q.ptr, /*base=*/1_000_000.0, /*scale=*/0.125)
    };

    let output = read_tensor_to_vec(&ctx, q_rope);
    let max_abs_err = reference.iter().zip(output.iter())
        .map(|(r, o)| (r - o).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_abs_err < 1e-3,
        "Proportional RoPE max absolute error {max_abs_err:.6} exceeds 1e-3"
    );
}
```

**Assertion:** Max absolute error vs. HuggingFace reference < 1e-3.  
**Fixture generation:** Python script `scripts/gen_rope_fixtures.py` using HuggingFace
`transformers` + Gemma4 model config, checked into the repository at a stable commit.

---

### Test 6.7 — Long context integrity (8192-token prompt)

**Name:** `test_long_context_8k_no_corruption`

```rust
#[tokio::test]
#[ignore = "requires model weights on SSD"]
async fn test_long_context_8k_no_corruption() {
    // Construct a 8192-token prompt that includes a unique marker near token 7000
    // to verify the KV chain was not truncated or wrapped.
    let marker = "UNIQUE_MARKER_7000";
    let prompt  = build_long_prompt_with_marker(8192, 7000, marker);

    let handle  = make_ggml_handle_for_testing();
    let output  = generate(&handle, &prompt, 64).await;

    // The model must reference the marker in its output (it was asked to repeat it)
    assert!(
        output.contains(marker),
        "8K context test: model output did not contain marker at position 7000.          This indicates KV truncation or sequence corruption. Output: {output:?}"
    );
}
```

**Assertion:** Output must contain the injected marker string.  
**Prompt construction:** The prompt is structured as:
`"Remember this code: UNIQUE_MARKER_7000. [6990 tokens of filler text]. What was the code I asked you to remember?"`

---

## 7. Acceptance Criteria

All items below are hard gates. Phase 2A is not complete until every item is checked.

### 7.1 Regression Gate — Phase 0 and Phase 1 Tests

- [ ] `cargo test -p inference-engine` — all tests pass, no modifications to the crate
- [ ] `cargo test -p inference-api` — all tests pass, no modifications to the crate
- [ ] `cargo test -p inference-backend-wgpu-offload` — all Phase 1 tests pass
- [ ] All Phase 0 correctness tests pass on macOS (scheduler, radix cache, stub backend)

### 7.2 Build Gate

- [ ] `cargo build --workspace` on macOS 14+ Apple Silicon completes with no errors
- [ ] Only prerequisite: `git submodule update --init --recursive` (pulls llama.cpp)
- [ ] `cmake`, `bindgen`, `cc` crates resolve without manual path configuration
- [ ] `cargo clippy -- -D warnings` passes on `ggml-sys` and `inference-backend-ggml`
- [ ] `cargo fmt --check` passes on all new crates

### 7.3 Correctness Gate — New Tests

- [ ] Test 6.1 passes: `output.to_lowercase().contains("beijing")` — Gemma 4 31B macOS
- [ ] Test 6.2 passes: `output.trim() == "2"` — Gemma 4 31B macOS
- [ ] Test 6.3 passes: layer streaming N=4 output bit-identical to N=46 all-resident
- [ ] Test 6.4 passes: SSD-backed KV ping-pong output identical to RAM-resident KV
- [ ] Test 6.5 passes: Q3_K_M matmul max relative error < 1% vs. F32 reference
- [ ] Test 6.6 passes: Proportional RoPE max absolute error < 1e-3 vs. HuggingFace ref
- [ ] Test 6.7 passes: 8192-token context preserves marker at position 7000

### 7.4 Performance Gate (MacBook Air M3 16 GB)

| Metric | Target | Measured | Status |
|--------|--------|----------|--------|
| TTFT (batch=1, cold, 512-token prompt) | < 3s | — | [ ] |
| TTFT (batch=1, warm radix cache hit) | < 0.3s | — | [ ] |
| Decode throughput (batch=1) | ≥ 10 tok/s | — | [ ] |
| Decode throughput (batch=4) | ≥ 30 tok/s aggregate | — | [ ] |
| Long-context TTFT (8192-token) | < 30s | — | [ ] |
| Radix cache hit rate (repeated system prompt) | > 85% | — | [ ] |
| Scheduler overhead (per batch) | < 0.5ms | — | [ ] |
| `burn-ggml` vs. `llama.cpp` direct | ≥ 85% throughput | — | [ ] |

### 7.5 Memory Safety Gate

- [ ] `cargo test` under AddressSanitizer: no leaks in `GgmlContext::drop`
- [ ] `cargo test` under AddressSanitizer: no leaks in `GgmlTensor` when graph fails
- [ ] `cargo test` under AddressSanitizer: no use-after-free in layer cache eviction

  Run with:
  ```bash
  RUSTFLAGS="-Z sanitizer=address" cargo +nightly test       -p inference-backend-ggml       -Z build-std       --target aarch64-apple-darwin
  ```

### 7.6 Seam Integrity Gate

- [ ] Zero lines changed in `inference-engine/` (verified by `git diff --stat`)
- [ ] Zero lines changed in `inference-api/` (verified by `git diff --stat`)
- [ ] Zero lines changed in `inference-offload/src/weight_cache.rs`
- [ ] Zero lines changed in `inference-offload/src/kv_offload.rs`
- [ ] `BackendHandle` trait has no new required methods (only default-impl additions
      from Phase 1 are retained; no breaking changes)

---

## 8. Non-Goals

The following are explicitly deferred to Phase 2B or later. Implementing any of these
in Phase 2A is a scope violation.

| Item | Reason deferred |
|------|----------------|
| ANE (Apple Neural Engine) | Requires Core ML integration; separate R&D track |
| Continuous batching improvements | Scheduler changes are Phase 2B scope |
| Speculative decoding | Requires draft model infrastructure |
| Multi-GPU / distributed inference | Not planned for this system |
| iOS / tvOS support | Different deployment target; separate phase |
| Gemma 4 31B MoE variant | This model is dense; MoE is Linux-only Phase 1 |
| Q8_0 or BF16 inference on Metal | Q3_K_M is the only validated quant in Phase 2A |
| Prefill-decode disaggregation | Phase 2B |
| KV cache quantization below INT8 | Correctness risk; Phase 2B |
| GGUF model conversion tooling | Assumed pre-converted; tooling is out of scope |
| Prometheus metrics endpoint | Stretch goal; not a hard gate |
| Priority scheduling | Unchanged from Phase 0 FCFS |
| Windows support for burn-ggml | macOS only in Phase 2A |
| llama.cpp-compatible server API | This is the burn-inference HTTP API |
| Flash attention for non-Metal compute | Only Metal path is supported |
| Dynamic quantization at runtime | Weights are pre-quantized Q3_K_M on disk |

---

## 9. Dependencies

### 9.1 New Rust crates (`Cargo.toml` additions)

```toml
# ggml-sys/Cargo.toml
[build-dependencies]
cmake    = "0.1"
bindgen  = "0.69"
cc       = "1.0"

[dependencies]
libc = "0.2"

# inference-backend-ggml/Cargo.toml
[dependencies]
ggml-sys           = { path = "../ggml-sys" }
inference-backend  = { path = "../inference-backend" }
inference-offload  = { path = "../inference-offload" }
tokio              = { version = "1", features = ["full"] }
anyhow             = "1"
tracing            = "0.1"
```

### 9.2 llama.cpp submodule

The `vendor/llama.cpp` directory is a git submodule pinned to a specific commit.

```toml
# .gitmodules
[submodule "vendor/llama.cpp"]
    path   = vendor/llama.cpp
    url    = https://github.com/ggerganov/llama.cpp.git
    branch = master
```

**Pinned commit:** The `Cargo.lock` equivalent for submodules is enforced via:

```bash
# In CI and developer onboarding:
git submodule update --init --recursive
# Then verify the pinned SHA:
git -C vendor/llama.cpp rev-parse HEAD
# Expected: <pinned-sha-here>  (updated when llama.cpp is intentionally bumped)
```

The pinned commit must be:
1. Tagged or at minimum named in `vendor/llama.cpp-version.txt`
2. Verified to include Metal Flash Attention support (`ggml_flash_attn_ext`)
3. Verified to include `ggml_rope_ext` with `freq_scale` parameter
4. Verified to build cleanly on macOS 14+ with `LLAMA_METAL=ON`

### 9.3 System requirements (macOS build host)

| Tool | Minimum version | Install |
|------|----------------|---------|
| macOS | 14.0 (Sonoma) | System |
| Xcode Command Line Tools | 15.0 | `xcode-select --install` |
| CMake | 3.20 | `brew install cmake` |
| Rust toolchain | 1.78 stable | `rustup update stable` |
| bindgen CLI (optional) | 0.69 | `cargo install bindgen-cli` |

No additional Metal SDK setup is required — Metal is included in Xcode CLT.

### 9.4 CI configuration

Phase 2A adds a macOS CI job to the existing GitHub Actions workflow:

```yaml
# .github/workflows/ci.yml (addition)
  macos-metal:
    runs-on: macos-14  # Apple Silicon runner (M1)
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
      - uses: dtolnay/rust-toolchain@stable
      - name: Build ggml-sys
        run: cargo build -p ggml-sys
      - name: Build inference-backend-ggml
        run: cargo build -p inference-backend-ggml
      - name: Run unit tests (no model weights)
        run: cargo test -p ggml-sys -p inference-backend-ggml
        # Integration tests requiring weights are #[ignore] and run only on
        # self-hosted runners with the model pre-loaded.
```

Integration tests (Tests 6.1–6.7) are marked `#[ignore]` and run only on a
dedicated self-hosted macOS runner that has the Gemma 4 31B Q3_K_M GGUF pre-loaded
at the path configured via `GEMMA4_31B_MODEL_PATH` environment variable.

### 9.5 GGUF model asset

**Model:** Gemma 4 31B dense Q3_K_M  
**File:** `gemma-4-31b-dense-q3_k_m.gguf`  
**Expected size:** ~14.5 GB  
**SHA-256:** (to be recorded after obtaining the final model file)  
**Source:** HuggingFace Hub, `google/gemma-4-31b-gguf` (or community conversion)  
**Pre-conversion script:** `scripts/convert_to_gguf_q3km.sh` (documented separately)

---

## 10. Open Questions

The following questions are unresolved at the time of this specification. Each must be
answered before or during Phase 2A implementation; answers should be recorded here as
the phase progresses.

**Q1. ggml Metal Flash Attention window parameter**  
Does `ggml_flash_attn_ext` in the pinned llama.cpp commit accept a `window_size`
parameter for sliding-window attention, or must the sliding-window causal mask be
constructed manually and passed as a `mask` tensor?  
*Impact:* If manual mask construction is required, `build_local_attention` needs a
mask-building helper, adding ~100 lines and a separate Metal buffer allocation per
local layer.  
*Resolution target:* Week 15, day 1 (first task after submodule pin).

**Q2. Shared KV layers in Gemma 4 31B**  
The architecture spec mentions "shared KV: last N layers reuse K/V projections from
earlier layers." The exact layer indices and sharing pattern must be confirmed from
the HuggingFace model config before implementing `Gemma4DenseModel`.  
*Impact:* Affects `LayerWeightCache` key design and the `KvPool` shard assignment.  
*Resolution target:* Week 15 (requires inspecting model config JSON).

**Q3. logit_softcap value for Gemma 4 31B**  
Gemma 2 uses `logit_softcap = 50.0`. Gemma 4 31B may use a different value or none.
The spec above assumes 50.0; confirm from model config.  
*Impact:* If softcap is 0.0, pass 0.0 to `ggml_flash_attn_ext`; no structural change.  
*Resolution target:* Week 15 (model config inspection).

**Q4. Proportional RoPE exact parameters**  
The spec uses `rope_base=1_000_000`, `rope_scale=0.125` for global layers. These must
be confirmed from the official Gemma 4 31B model card or config.json. Other
configurations (e.g., NTK-aware scaling with attn_factor) may also apply.  
*Impact:* Affects Test 6.6 fixture generation and `build_global_attention`.  
*Resolution target:* Week 15, confirmed against HuggingFace reference outputs.

**Q5. INT8 KV quantization scheme**  
The memory budget uses INT8 for global KV on SSD. The exact quantization scheme
(symmetric per-tensor, asymmetric per-channel, or block-float) is not yet specified.
Using the wrong scheme will cause Test 6.4 (KV correctness) to fail.  
*Impact:* Affects `KvOffloadManager` integration (Phase 1 component) and the
dequantize path on KV load.  
*Resolution target:* Week 15-16; may require extending `KvOffloadManager` with a
`QuantizationConfig` parameter (if so, this is the one allowed extension to a
Phase 1 component, and it must not break existing Phase 1 tests).

**Q6. SSD read throughput on MacBook Air M3**  
The memory budget assumes ~3 GB/s NVMe sequential read. MacBook Air M3 SSD benchmarks
show 3.5–4.0 GB/s. If actual throughput is lower (e.g., thermal throttling),
the KV ping-pong overlap window shrinks and Test 6.7 (8K long context TTFT < 30s)
may be at risk.  
*Impact:* May require reducing context window for the long-context test, or adding a
read-ahead buffer. Measure on target hardware in Week 15.

**Q7. ggml context per-layer vs. shared**  
The current design allocates a single `GgmlContext` (512 MB scratch) for the full
forward pass and resets it between layers by calling `ggml_free` + `ggml_init`.
An alternative is to use `ggml_cgraph` checkpointing.  
Which approach produces lower peak memory and avoids Metal command buffer overflow?  
*Impact:* Affects `GgmlContext` lifetime management and the reset protocol in
`GemmaRunner::decode_step`.  
*Resolution target:* Week 16 (profile both approaches with Instruments).

**Q8. llama.cpp submodule pinning policy**  
Should the llama.cpp submodule be pinned to a released tag (e.g., `b3400`) or a
specific commit SHA? Tags are more readable; SHAs are more reproducible.  
*Resolution target:* Team decision before Week 15 starts. Default: pin to the most
recent tag that passes the metal build test.
