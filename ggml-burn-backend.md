# `burn-ggml`: A GGML Backend for Burn
## Design Document

**Status:** Draft  
**Date:** 2026-04-05  
**Author:** (working document)  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Motivation & Use Case](#2-motivation--use-case)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Model Memory Analysis](#4-model-memory-analysis)
   - 4.1 [Gemma 4 31B Dense](#41-gemma-4-31b-dense)
   - 4.2 [Gemma 4 26B MoE](#42-gemma-4-26b-moe)
5. [Core Components](#5-core-components)
6. [Mixture-of-Experts Support](#6-mixture-of-experts-support)
7. [ggml Graph Construction for Attention](#7-ggml-graph-construction-for-attention)
8. [Burn Integration Points](#8-burn-integration-points)
   - 8.4 [PrefetchOps: Burn API Extension](#84-prefetchops-burn-api-extension)
9. [GGUF Model Loading](#9-gguf-model-loading)
10. [Implementation Plan](#10-implementation-plan)
11. [Key Risks & Mitigations](#11-key-risks--mitigations)
12. [Performance Targets](#12-performance-targets)
13. [Correctness Test Cases](#13-correctness-test-cases)

---

## 1. Executive Summary

`burn-ggml` is a Burn backend crate that delegates tensor operations to ggml,
the C tensor library at the heart of llama.cpp. The goal is to give Rust
applications written against the Burn ML framework access to ggml's battle-tested
quantization kernels, Apple Metal GPU acceleration, and its ecosystem of GGUF
model files — without leaving the Burn API surface.

Two target models drive the design:

- **Gemma 4 26B MoE** (128 experts, 8 active per token, 256K context) —
  the Phase 1 validation target, running on **Linux with an Intel iGPU via
  wgpu/Vulkan**.
- **Gemma 4 31B dense** (256K context) — the Phase 2 target, running on a
  **MacBook Air with 16 GB unified memory via ggml + Apple Metal**.

Both share the same hard constraint: model weights plus KV cache exceed available
RAM by a large margin at long context. The solution is a three-part memory
management strategy:

1. **Expert / layer weight streaming** — for MoE models, keep only the ~32
   hottest expert weight matrices in RAM (LRU cache); for dense models, keep
   only 3-4 transformer layers. Stream cold weights from NVMe SSD on demand.
2. **KV cache SSD offload with ping-pong buffering** — global attention layers
   store their KV cache on SSD; a double-buffer scheme hides I/O latency behind
   GPU compute.
3. **Compute-prefetch overlap** — async SSD reads are pipelined with GPU
   compute via a new `PrefetchOps` trait added to Burn's backend abstraction.
   Prefetch calls are issued explicitly from model-specific inference code
   (the schedule is model-architecture-dependent and cannot be derived
   generically). All existing Burn backends receive a no-op default; only
   `burn-ggml` and the wgpu offload variant trigger real async I/O.

### Two-Phase Delivery

| Phase | Backend | Platform | Benchmark model | Key new work |
|-------|---------|----------|-----------------|--------------|
| **Phase 1** | `burn-wgpu` + offload layer | Linux, Intel iGPU | Gemma 4 26B MoE Q4 | `PrefetchOps`, expert cache, `MUL_MAT_ID` WGSL kernel, KV offload |
| **Phase 2** | `burn-ggml` + Metal | macOS, Apple Silicon | Gemma 4 31B Q3_K_M | ggml FFI, Metal backend, quantized matmul, layer streaming |

Phase 1 validates the entire memory management architecture on accessible Linux
hardware before introducing ggml FFI complexity. The expert streaming cache built
in Phase 1 is a strict superset of the layer streaming cache needed in Phase 2 —
the same `WeightCache<T>` abstraction serves both.

---

## 2. Motivation & Use Case

### 2.1 The Gap in the Burn Ecosystem

Burn is a modern, ergonomic ML framework written in pure Rust. Its backend
abstraction allows the same model code to run on WGPU, LibTorch, CUDA (via
candle-style bindings), and a hand-written NdArray CPU backend. However:

- None of the existing backends expose **integer quantization** (INT4/INT3) at
  the kernel level. The `QTensorOps` trait exists but production-grade Q4/Q3
  dequant + GEMM is not available for large LLMs.
- The WGPU backend does not use Apple's **Metal Performance Shaders** (MPS),
  missing out on hardware-tuned matrix multiply kernels on Apple Silicon.
- There is no built-in mechanism for **weight streaming from disk** or **KV
  cache offloading** — both required for >16B models on consumer hardware.

### 2.2 What ggml Brings

ggml (from llama.cpp) provides:

- A rich library of **quantized matrix multiply kernels** (Q4_K, Q3_K, IQ3_S,
  IQ2_XXS, and many more) highly tuned for ARM NEON and x86 AVX2/AVX-512.
- A production-quality **Metal backend** that offloads matrix multiplications
  and attention to Apple Silicon's GPU via MSL shaders.
- The **GGUF file format**: a self-describing model file that carries tensor
  metadata, quantization types, and model hyperparameters in a single file.
- A **backend abstraction layer** (`ggml-backend.h`) that supports plugging in
  custom compute devices, making it straightforward to wrap in FFI.
- Years of community optimization for consumer hardware: CPU offload, split
  layers between CPU and GPU, KV cache quantization, etc.

### 2.3 Target Hardware & Scenario

| Parameter              | Value                                      |
|------------------------|--------------------------------------------|
| Hardware               | MacBook Air, Apple M-series (M1/M2/M3/M4)  |
| Unified memory         | 16 GB                                      |
| NVMe SSD               | ~3-7 GB/s sequential read bandwidth        |
| GPU                    | Integrated Apple Silicon GPU (no discrete) |
| Model                  | Gemma 4 31B                                |
| Quantization target    | Q3_K_M or IQ3_S                            |
| Context window         | 256K tokens                                |
| Use case               | Interactive chat, document Q&A             |

### 2.4 Why Not Use llama.cpp Directly?

llama.cpp is a complete inference stack. Using it directly means:

- Model code lives in C/C++, not Rust — no Burn ergonomics, no `Module` trait,
  no automatic differentiation (for future fine-tuning).
- Integrating custom preprocessing, post-processing, or retrieval pipelines
  written in Rust requires awkward FFI plumbing at every seam.
- The `llama.cpp` monolith couples model architecture with the backend; `burn-ggml`
  separates them: model architecture in idiomatic Rust (Burn), compute in ggml.

`burn-ggml` uses only the lower layers of llama.cpp — the ggml tensor library and
its Metal/CPU backends — while keeping model architecture, tokenization, and
sampling logic in Rust.

---

## 3. System Architecture Overview

### 3.1 Crate Stack

```
┌─────────────────────────────────────────────────────────┐
│           User Rust Application Code                    │
│  (Burn model definition, tokenizer, sampling loop)      │
└─────────────────────────────┬───────────────────────────┘
                              │  Burn public API
                              ▼
┌─────────────────────────────────────────────────────────┐
│                  burn-ggml                              │
│  (Backend trait impl, tensor ops, memory management)    │
└───────────────┬─────────────────────┬───────────────────┘
                │ Rust FFI             │ async I/O
                ▼                      ▼
┌───────────────────────┐   ┌──────────────────────────────┐
│       ggml-sys        │   │  Layer/KV cache offload I/O  │
│  (bindgen bindings)   │   │  (tokio + pread/mmap)        │
└───────────┬───────────┘   └──────────────────────────────┘
            │ C ABI
            ▼
┌─────────────────────────────────────────────────────────┐
│         ggml + Metal / CPU backend                      │
│         (from llama.cpp, compiled as static lib)        │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│       Apple Unified Memory / NVMe SSD                   │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Memory Hierarchy

```
┌────────────────────────────────────────────────────────────────┐
│  NVMe SSD (1-2 TB)                                             │
│                                                                │
│  ┌─────────────────────────────┐  ┌─────────────────────────┐ │
│  │  model.gguf                 │  │  kv_cache/              │ │
│  │  (all layer weights, ~14 GB)│  │  global_layer_N.bin     │ │
│  │                             │  │  (~1 GB each, cold)     │ │
│  └─────────────────────────────┘  └─────────────────────────┘ │
└─────────────────────────────┬──────────────────────────────────┘
                              │ async prefetch (3-7 GB/s)
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  CPU / Unified RAM (16 GB)                                     │
│                                                                │
│  ┌───────────────────┐  ┌───────────────────────────────────┐ │
│  │ Layer weight      │  │  KV ping-pong buffers             │ │
│  │ circular buffer   │  │  ping: [global KV active]         │ │
│  │ (3-4 layers,      │  │  pong: [global KV prefetching]    │ │
│  │  ~1-2 GB)         │  │  local KV: resident (small)       │ │
│  └─────────┬─────────┘  └─────────────────┬─────────────────┘ │
└────────────┼────────────────────────────────┼───────────────────┘
             │ Metal copy                     │ Metal copy
             ▼                                ▼
┌────────────────────────────────────────────────────────────────┐
│  Apple Silicon GPU (shared from unified memory)                │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  ggml Metal command buffers                             │  │
│  │  active_weights: layer K                                │  │
│  │  active_kv: attention for layer K                       │  │
│  └─────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### 3.3 Data Flow for a Single Forward Pass (Decode Step)

```
Token IDs
    │
    ▼  embed
Hidden state [1, d_model]
    │
    ├──── For each transformer layer i:
    │         1. Ensure layer i weights in RAM (LayerWeightCache)
    │         2. Copy weights to Metal buffer
    │         3. RMSNorm → QKV projection (Metal)
    │         4a. [Local layer]  Sliding-window attention (KV in RAM)
    │         4b. [Global layer] Load KV from SSD → ping buffer (async)
    │                            Attention over full context (Metal)
    │                            Write updated KV → pong → SSD (async)
    │         5. MLP (Metal)
    │         6. Trigger prefetch of layer i+1 weights + KV
    │
    ▼
Logits [1, vocab_size]
    │
    ▼  sample
Next token ID
```

---

## 4. Model Memory Analysis

## 4.1 Gemma 4 31B Dense

### 4.1.1 Model Architecture Facts

Gemma 4 is Google's fourth-generation Gemma model family. The 31B variant
relevant to this project has the following key characteristics:

| Property                   | Value                                        |
|----------------------------|----------------------------------------------|
| Total parameters           | ~31 billion                                  |
| Architecture               | Decoder-only transformer                     |
| Context window             | 256K tokens                                  |
| Attention type             | Hybrid: local sliding-window + global full   |
| Local window size          | 1024 tokens                                  |
| Attention interleaving     | ~5 local : 1 global (follows Gemma 3 design) |
| Number of layers           | 46                                           |
| KV heads                   | 8 (grouped-query attention, GQA)             |
| Head dimension             | 256                                          |
| Model dimension (d_model)  | ~5120                                        |
| Vocabulary size            | 256K tokens                                  |
| RoPE variant               | Dual: standard (local) + Proportional (global)|
| Proportional RoPE base     | 1M, scale factor 8                           |
| Shared KV                  | Yes — last N layers reuse earlier K/V projs  |

### 4.1.2 Weight Memory by Quantization

The following table shows approximate model weight size on disk/in-memory for
the most relevant GGUF quantization levels:

| Quantization | Bits/weight | Model size | Notes                              |
|--------------|-------------|------------|------------------------------------|
| FP16         | 16          | ~62 GB     | Full precision, impractical        |
| Q8_0         | 8.5         | ~33 GB     | Near-lossless, still too large     |
| Q4_K_M       | 4.5         | ~18 GB     | Good quality, slightly over 16 GB  |
| Q3_K_M       | 3.5         | ~14.5 GB   | **Primary target** — fits in 16 GB |
| IQ3_S        | ~3.4        | ~13 GB     | Importance-quantized, good quality |
| IQ2_XXS      | ~2.4        | ~8.7 GB    | Aggressive, more quality loss      |

**Conclusion:** Q3_K_M at ~14.5 GB is the primary target. It fits in 16 GB RAM
but leaves virtually no headroom for the KV cache, OS, and runtime overhead.
IQ3_S (~13 GB) provides additional headroom while being close in quality.

### 4.1.3 KV Cache Size Calculation

The KV cache stores key and value tensors for every layer, head, and past token.

**Formula:**
```
kv_cache_bytes =
    num_layers × 2 (K and V) × num_kv_heads × head_dim × context_len × bytes_per_element

For Gemma 4 31B FP16:
    46 × 2 × 8 × 256 × context_len × 2 bytes
    = 46 × 4096 × context_len × 2
    = 376,832 bytes per context token (all layers)
    ≈ 368 KB per token (all layers)
```

| Context length | KV cache size (FP16, all layers) | KV cache (INT8) |
|----------------|----------------------------------|-----------------|
| 4K tokens      | ~1.5 GB                          | ~0.7 GB         |
| 32K tokens     | ~11.8 GB                         | ~5.9 GB         |
| 128K tokens    | ~47 GB                           | ~23.5 GB        |
| 256K tokens    | **~94 GB**                       | ~47 GB          |

Note: When KV cache is quantized to INT8 (as ggml supports via `ggml_type = GGML_TYPE_Q8_0`),
the sizes halve, but 256K context still requires ~47 GB — far exceeding 16 GB RAM.

**Global layers only (1 in 6, ~8 layers out of 46):**
```
global_kv_cache_bytes =
    8 global_layers × 2 × 8 heads × 256 head_dim × 256K tokens × 2 bytes
    = 8 × 4096 × 262144 × 2
    ≈ 17 GB for global layers only (FP16)

Per global layer:
    2 × 8 × 256 × 262144 × 2 bytes = ~2.1 GB per global layer (FP16)
    With INT8: ~1.07 GB per global layer
```

**Local layers (sliding window, 1024 tokens max):**
```
local_kv_per_layer = 2 × 8 × 256 × 1024 × 2 bytes = 8 MB per local layer
Total local KV (38 layers × 8 MB) ≈ 304 MB — resident in RAM
```

### 4.1.4 Shared KV Cache Optimization

Gemma 4 implements a shared KV cache where certain later layers reuse K/V
projection outputs from designated earlier layers rather than computing their
own. This reduces the number of independent KV caches that must be maintained.
The exact sharing pattern is encoded in the GGUF metadata.

In practical terms, for the offload strategy:
- Shared KV layers do not need their own SSD storage slot — they reference
  the buffer of the layer they share with.
- This can reduce effective global KV storage by 20-30% depending on the
  specific sharing configuration in Gemma 4.

### 4.1.5 Memory Budget at Inference Time (Q3_K_M, 256K context)

```
Total RAM: 16 GB

  OS + runtime overhead:         ~1.5 GB
  Layer weight circular buffer:  ~1.5 GB  (3-4 layers in RAM at Q3_K_M)
  Local attention KV cache:      ~0.3 GB  (38 local layers × 8 MB)
  Ping buffer (global KV):       ~1.1 GB  (1 global layer at INT8)
  Pong buffer (global KV):       ~1.1 GB  (prefetch slot)
  Activation/scratch buffers:    ~0.5 GB
  Miscellaneous / headroom:      ~1.0 GB
                                --------
  Total used:                   ~7.0 GB  <-- fits comfortably

  Remaining model weights:       ~13 GB   (on SSD)
  Global KV cache (cold):        ~17 GB   (on SSD)
```

This confirms that Q3_K_M + layer streaming + KV offload fits within 16 GB.

---

## 4.2 Gemma 4 26B MoE

### 4.2.1 Model Architecture Facts

Gemma 4 26B is a sparse Mixture-of-Experts model. Only a fraction of expert
weights are active per token, making it memory-bandwidth-efficient at inference
despite its large total parameter count.

| Property                   | Value                                          |
|----------------------------|------------------------------------------------|
| Total parameters           | ~25.2B (~26B marketed)                         |
| Active parameters per token| ~3.8B (~4B marketed)                           |
| Architecture               | Decoder-only transformer, sparse MoE FFN       |
| Context window             | 256K tokens                                    |
| Number of layers           | 30                                             |
| Total experts              | **128** per MoE layer                          |
| Active experts per token   | **8** (top-8 routing) + **1 shared** (always)  |
| Expert sparsity            | 6.25% (8/128)                                  |
| Attention type             | Hybrid: local sliding-window + global full     |
| Local window size          | 1024 tokens                                    |
| Shared KV cache            | Yes (same as 31B dense)                        |
| Expert routing             | Softmax over 128 → top-8 + 1 shared            |
| Shared expert design       | DeepSeek V2 pattern: 1 always-active base expert|

### 4.2.2 Expert Weight Memory

At Q4_K_M, each expert FFN (gate + up + down projections) is approximately:

```
expert_size_Q4 ≈ 3 × (d_model × d_expert_ffn) × 4.5 bits/weight / 8
               ≈ 3 × 4096 × 2048 × 0.5625 bytes
               ≈ ~14 MB per expert

Total expert weights: 128 experts × 30 layers × ~14 MB = ~53 GB (Q4_K_M)
```

Note: non-FFN parameters (attention, embeddings, norms) add ~2–3 GB.

| Quantization | Total model size | Expert weights only |
|--------------|-----------------|---------------------|
| F16          | ~52 GB          | ~50 GB              |
| Q8_0         | ~27 GB          | ~25 GB              |
| Q4_K_M       | ~14 GB          | ~12.5 GB            |
| Q3_K_M       | ~10.5 GB        | ~9 GB               |

**At Q4_K_M (~14 GB total), the full model fits on a 16 GB device — but leaves
only ~2 GB for KV cache, activations, and OS overhead at short contexts.**
For 256K context the KV cache alone exceeds available RAM, requiring SSD offload
exactly as for the 31B dense model.

### 4.2.3 Expert Cache Design

Unlike layer streaming (sequential access, easy to prefetch), expert access is
**driven by the router output** — which experts are hot depends on the input
token. The `ExpertWeightCache` maintains an LRU cache of the N most recently
used expert weight matrices:

```
Total experts per layer:  128 × ~14 MB = ~1.8 GB/layer (Q4)
Hot expert cache (N=32):   32 × ~14 MB = ~448 MB
Cold experts on SSD:       96 × ~14 MB = ~1.3 GB/layer

Across all 30 layers (all cold):  128 × 30 × ~14 MB = ~53 GB on SSD
In-RAM hot cache (N=32 per layer, but shared across layers via global LRU):
  ~32 × ~14 MB = ~448 MB total hot cache
```

The key insight: with 128 fine-grained experts, cache hit rates are high for
typical text (vocabulary-driven token distributions cluster into recurring
expert subsets). A global LRU across all layers with N=32 slots is the starting
point; N is tunable.

### 4.2.4 Prefetch After Routing

The router selects 8 expert indices before the expert FFN executes. This creates
a natural prefetch window: immediately after routing, fire `tensor.prefetch()`
for the 8 selected experts. The SSD read (~14 MB × 8 = ~112 MB at ~5 GB/s =
~22 ms) overlaps with the attention computation that precedes the FFN.

```
Per-layer timeline (MoE layer):

  [Attention (local or global)]
       [Router: select top-8 experts]
       [prefetch(expert[0..7])]  ← fire immediately after routing
                    [Expert FFN: MUL_MAT_ID × 8]
                    ← by now, prefetch should be done (22ms vs ~30ms attn)
```

### 4.2.5 Memory Budget at Inference Time (Q4_K_M, 256K context, Linux 16 GB)

```
Total RAM: 16 GB

  OS + runtime overhead:          ~1.5 GB
  Non-expert model weights (RAM): ~1.5 GB  (attn, embed, norms)
  Expert hot cache (N=32):        ~0.45 GB
  Local attention KV cache:       ~0.25 GB (25 local layers × 8 MB)
  Ping buffer (global KV, INT8):  ~1.1 GB  (1 global layer)
  Pong buffer (global KV, INT8):  ~1.1 GB  (prefetch slot)
  Activation / scratch buffers:   ~0.5 GB
  Miscellaneous / headroom:       ~1.0 GB
                                  --------
  Total used:                     ~7.4 GB  <-- fits comfortably

  Cold expert weights on SSD:     ~53 GB   (all 128 × 30 layers)
  Global KV cache (cold):         ~14 GB   (INT8, 256K ctx)
```

**Conclusion:** Gemma 4 26B MoE at Q4_K_M fits the 16 GB budget with the
expert cache + KV offload strategy. The SSD holds ~67 GB of cold data —
a 256 GB or larger NVMe drive is required.

---

## 5. Core Components

### 5.1 `ggml-sys` -- FFI Bindings Crate

#### 5.1.1 Purpose

`ggml-sys` is a low-level `-sys` crate that:
- Compiles the ggml C/C++ library (from the llama.cpp submodule) as a static
  library via the `cmake` crate.
- Generates Rust FFI bindings from the ggml headers using `bindgen`.
- Re-exports all `ggml_*` symbols for use by `burn-ggml`.

No Rust logic lives here -- only the build machinery and generated bindings.

#### 5.1.2 `build.rs` Skeleton

```rust
// ggml-sys/build.rs
use std::path::PathBuf;

fn main() {
    let llama_dir = PathBuf::from("../llama.cpp");

    // 1. Build ggml + Metal backend via cmake
    let dst = cmake::Config::new(&llama_dir)
        .define("LLAMA_METAL", "ON")
        .define("LLAMA_STATIC", "ON")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .build();

    // 2. Tell cargo to link the static library
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=llama");

    // On macOS: link Metal framework
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // 3. Generate bindings with bindgen
    let bindings = bindgen::Builder::default()
        .header(llama_dir.join("ggml/include/ggml.h").to_str().unwrap())
        .header(llama_dir.join("ggml/include/ggml-backend.h").to_str().unwrap())
        .header(llama_dir.join("ggml/include/ggml-alloc.h").to_str().unwrap())
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .allowlist_var("GGML_.*")
        .allowlist_function("ggml_backend_metal_.*")
        .allowlist_function("ggml_backend_cpu_.*")
        .blocklist_function("ggml_internal_.*")
        .generate_comments(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Failed to generate ggml bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("ggml_bindings.rs"))
        .expect("Couldn't write bindings");

    println!("cargo:rerun-if-changed=../llama.cpp/ggml/include/ggml.h");
    println!("cargo:rerun-if-changed=../llama.cpp/ggml/include/ggml-backend.h");
}
```

#### 5.1.3 `lib.rs` for ggml-sys

```rust
// ggml-sys/src/lib.rs
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

include!(concat!(env!("OUT_DIR"), "/ggml_bindings.rs"));
```

#### 5.1.4 Static vs Dynamic Linking

Static linking is strongly preferred for `ggml-sys` because:
- ggml's Metal backend requires compiling `.metal` shader files into a default
  Metal library that is embedded in the binary. Dynamic linking complicates
  finding the shader library at runtime.
- Simpler deployment: a single Rust binary with no runtime ggml dependency.
- llama.cpp frequently changes its ABI; pinning to a git submodule SHA avoids
  version mismatches.

The tradeoff is longer incremental build times. Mitigated by caching the cmake
output (`OUT_DIR`) and using `cargo:rerun-if-changed` conservatively.

### 5.2 `burn-ggml` -- Backend Implementation Crate

#### 5.2.1 Crate Structure

```
burn-ggml/
├── Cargo.toml
├── build.rs
└── src/
    ├── lib.rs
    ├── backend.rs          # Backend trait impl + GgmlBackend struct
    ├── device.rs           # GgmlDevice enum + DeviceOps
    ├── tensor.rs           # GgmlTensor wrapper around *mut ggml_tensor
    ├── context.rs          # GgmlContext (owns ggml_context*)
    ├── graph.rs            # ggml_cgraph construction utilities
    ├── ops/
    │   ├── mod.rs
    │   ├── float_ops.rs    # FloatTensorOps impl
    │   ├── int_ops.rs      # IntTensorOps impl
    │   ├── bool_ops.rs     # BoolTensorOps impl
    │   ├── module_ops.rs   # ModuleOps: matmul, conv, norm
    │   ├── activation_ops.rs # ActivationOps: ReLU, GELU, SiLU
    │   └── quant_ops.rs    # QTensorOps: quantize/dequantize
    └── memory/
        ├── mod.rs
        ├── layer_cache.rs  # LayerWeightCache: streaming circular buffer
        ├── kv_offload.rs   # KvOffloadManager: SSD KV cache
        └── ping_pong.rs    # PingPongBuffer: double-buffer for KV I/O
```

#### 5.2.2 GgmlBackend Struct and Backend Trait Implementation

```rust
// burn-ggml/src/backend.rs
use burn::backend::Backend;
use burn::tensor::{Element, TensorMetadata};
use std::sync::Arc;
use crate::{GgmlDevice, GgmlTensor, GgmlContext};

/// The top-level backend type. Clone is cheap (Arc clone).
#[derive(Clone, Debug, Default)]
pub struct GgmlBackend;

impl Backend for GgmlBackend {
    type Device = GgmlDevice;

    type FloatTensorPrimitive = GgmlTensor;
    type FloatElem = f32;

    type IntTensorPrimitive = GgmlTensor;
    type IntElem = i32;

    type BoolTensorPrimitive = GgmlTensor;

    type QuantizedTensorPrimitive = GgmlQuantizedTensor;

    fn name(device: &Self::Device) -> String {
        match device {
            GgmlDevice::Cpu => "ggml-cpu".into(),
            GgmlDevice::Metal => "ggml-metal".into(),
            GgmlDevice::MetalWithOffload { .. } => "ggml-metal-offload".into(),
        }
    }

    fn seed(_device: &Self::Device, seed: u64) {
        GGML_SEED.with(|s| *s.borrow_mut() = seed);
    }

    fn sync(device: &Self::Device)
        -> Result<(), burn::tensor::backend::ExecutionError>
    {
        if matches!(device, GgmlDevice::Metal | GgmlDevice::MetalWithOffload { .. }) {
            unsafe {
                ggml_sys::ggml_backend_metal_synchronize(get_metal_backend());
            }
        }
        Ok(())
    }
}
```

#### 5.2.3 GgmlDevice Enum

```rust
// burn-ggml/src/device.rs
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GgmlDevice {
    /// Pure CPU backend using ggml CPU kernels (ARM NEON / AVX2).
    Cpu,

    /// Apple Metal GPU backend.
    Metal,

    /// Metal GPU with NVMe SSD offload for layer weights and KV cache.
    MetalWithOffload {
        kv_cache_dir: PathBuf,
        max_layers_in_ram: usize,
    },
}

impl Default for GgmlDevice {
    fn default() -> Self {
        #[cfg(target_os = "macos")]
        return GgmlDevice::Metal;
        #[cfg(not(target_os = "macos"))]
        return GgmlDevice::Cpu;
    }
}
```

#### 5.2.4 GgmlTensor Wrapper

```rust
// burn-ggml/src/tensor.rs
use std::sync::Arc;
use ggml_sys::{ggml_tensor, ggml_type};
use crate::context::GgmlContext;

/// A Burn tensor primitive backed by a ggml_tensor*.
///
/// Safety: the ggml_tensor is valid for as long as the Arc<GgmlContext>
/// is alive. GgmlTensor holds Arc<GgmlContext> as a lifetime anchor.
#[derive(Clone, Debug)]
pub struct GgmlTensor {
    /// Raw pointer -- NOT owned here. GgmlContext owns the memory.
    pub(crate) ptr: *mut ggml_tensor,
    pub(crate) ctx: Arc<GgmlContext>,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: ggml_type,
}

// Manually assert Send + Sync:
// 1. All mutations are serialised through ggml graph compute.
// 2. Reads are safe after graph completion.
unsafe impl Send for GgmlTensor {}
unsafe impl Sync for GgmlTensor {}

impl TensorMetadata for GgmlTensor {
    fn dtype(&self) -> burn::tensor::DType {
        match self.dtype {
            ggml_sys::GGML_TYPE_F32 => burn::tensor::DType::F32,
            ggml_sys::GGML_TYPE_F16 => burn::tensor::DType::F16,
            ggml_sys::GGML_TYPE_I32 => burn::tensor::DType::I32,
            _ => burn::tensor::DType::F32,
        }
    }

    fn shape(&self) -> burn::tensor::Shape {
        burn::tensor::Shape::from(self.shape.clone())
    }
}

impl GgmlTensor {
    /// Create from raw pointer. Caller guarantees ptr validity and that
    /// ctx is the owning context for this tensor.
    pub unsafe fn from_raw(ptr: *mut ggml_tensor, ctx: Arc<GgmlContext>) -> Self {
        let ne = (*ptr).ne;
        let n_dims = (*ptr).n_dims as usize;
        // ggml stores dims in ne[] with ne[0] as innermost (C-minor order)
        let shape: Vec<usize> = (0..n_dims).rev().map(|i| ne[i] as usize).collect();
        let dtype = (*ptr).type_;
        GgmlTensor { ptr, ctx, shape, dtype }
    }
}
```

### 5.3 Layer Weight Streaming Manager

The central challenge is that Gemma 4 31B at Q3_K_M weighs ~14.5 GB but only
~1.5 GB of RAM is available for weights (after OS, KV, and activation buffers).
The `LayerWeightCache` maintains a circular buffer of N=3-4 layer slots.

#### 5.3.1 Design

```
RAM: circular buffer of N=4 layer weight slots

  slot 0: [layer i-1 weights] -- may evict
  slot 1: [layer i   weights] -- in use
  slot 2: [layer i+1 weights] -- prefetched
  slot 3: [layer i+2 weights] -- prefetching...

SSD: model.gguf (all layers, mmap)

Timeline:
  T=0: GPU runs layer i,    async prefetch layer i+1 from SSD
  T=1: GPU runs layer i+1,  async prefetch layer i+2 from SSD,
                            GPU compute started immediately (i+1 already ready)
  T=2: GPU runs layer i+2,  ...
```

#### 5.3.2 LayerWeightCache Struct

```rust
// burn-ggml/src/memory/layer_cache.rs
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use tokio::sync::{Mutex, Notify};

struct LayerSlot {
    layer_idx: Option<usize>,
    ggml_buffer: *mut ggml_sys::ggml_backend_buffer,
    in_use: bool,
    prefetch_complete: Arc<Notify>,
}

pub struct LayerWeightCache {
    model_path: PathBuf,
    max_slots: usize,
    slots: Mutex<Vec<LayerSlot>>,
    layer_offsets: Vec<LayerOffsetInfo>,
    layer_sizes: Vec<usize>,
}

impl LayerWeightCache {
    pub async fn get_layer(&self, layer_idx: usize) -> Arc<LayerWeightGuard> {
        let mut slots = self.slots.lock().await;
        if let Some(slot) = slots.iter_mut().find(|s| s.layer_idx == Some(layer_idx)) {
            slot.in_use = true;
            let notify = slot.prefetch_complete.clone();
            drop(slots);
            notify.notified().await;
            return self.make_guard(layer_idx).await;
        }
        self.evict_lru(&mut slots);
        drop(slots);
        self.load_layer_sync(layer_idx).await;
        self.make_guard(layer_idx).await
    }

    pub fn prefetch(&self, layer_idx: usize) {
        let cache = self.clone_handle();
        tokio::spawn(async move {
            cache.load_layer_sync(layer_idx).await;
        });
    }

    async fn load_layer_sync(&self, layer_idx: usize) {
        let offset = self.layer_offsets[layer_idx];
        let size   = self.layer_sizes[layer_idx];
        let file = tokio::fs::File::open(&self.model_path).await.unwrap();
        let mut buf = allocate_ggml_buffer(size);
        file.seek_and_read(offset.data_offset, &mut buf).await.unwrap();
        let mut slots = self.slots.lock().await;
        let slot = self.find_free_slot(&mut slots);
        slot.layer_idx = Some(layer_idx);
        slot.ggml_buffer = buf.into_raw();
        slot.prefetch_complete.notify_waiters();
    }
}
```

#### 5.3.3 Eviction Policy

The cache uses LRU with forward-pass prefetch hints:

1. During the forward pass, layer access order is deterministic (0, 1, 2, ...
   N-1 for decode; same for prefill).
2. The prefetch window is always `[current_layer + 1, current_layer + N-1]`.
3. Eviction always targets the layer with the lowest index less than
   `current_layer - 1` (already passed) or the globally least-recently-used.
4. For speculative decoding or beam search, the eviction policy would need
   to account for branching access patterns.

#### 5.3.4 I/O Strategy on macOS

macOS does not expose `io_uring`. The recommended approach is:

- **mmap the GGUF file**: `mmap(MAP_SHARED)` the entire model file. The OS
  maps pages lazily from NVMe. Call `madvise(MADV_SEQUENTIAL)` on the next
  layer's region to trigger read-ahead in the page cache.
- **Tokio async I/O**: Use `tokio::fs::File` with `read_at` for explicit
  pread calls from an async task, overlapping with Metal compute.
- **F_RDADVISE fcntl**: macOS equivalent of `posix_fadvise(POSIX_FADV_WILLNEED)`
  for explicit read-ahead hints to the kernel.

Empirically, mmap + MADV_SEQUENTIAL achieves close to peak SSD read bandwidth
(3-7 GB/s) on Apple Silicon MacBooks due to the unified memory architecture.

### 5.4 KV Cache Ping-Pong Offload

#### 5.4.1 Gemma 4 Attention Layer Classification

Gemma 4 uses a hybrid attention scheme:

| Layer type  | Frequency | KV window    | KV per layer (FP16) | Strategy        |
|-------------|-----------|--------------|---------------------|-----------------|
| Local attn  | ~5 in 6   | 1024 tokens  | ~8 MB               | Resident in RAM |
| Global attn | ~1 in 6   | Full 256K    | ~2.1 GB             | SSD offload     |

For a 46-layer model with 1:5 global:local ratio:
- ~8 global attention layers
- ~38 local attention layers

Total resident local KV: 38 x 8 MB = 304 MB -- fits comfortably in RAM.
Total global KV on SSD: 8 x 2.1 GB = ~17 GB (FP16); ~8.5 GB at INT8.

KV quantization: ggml supports GGML_TYPE_Q8_0 for KV cache tensors, halving
memory. At INT8, each global layer's KV is ~1.07 GB.

#### 5.4.2 Ping-Pong Buffer Design

```
Global KV files on SSD (kv_cache/ directory):
  global_layer_07.kv  -- 1.07 GB (INT8)
  global_layer_13.kv  -- 1.07 GB (INT8)
  global_layer_19.kv  -- 1.07 GB (INT8)
  global_layer_25.kv  -- 1.07 GB (INT8)
  global_layer_31.kv  -- 1.07 GB (INT8)
  global_layer_37.kv  -- 1.07 GB (INT8)
  global_layer_43.kv  -- 1.07 GB (INT8)
  global_layer_45.kv  -- 1.07 GB (INT8)

RAM (ping-pong buffers):
  ping: [global_layer_07 KV data -- being used by Metal GPU]
  pong: [prefetching global_layer_13 KV from SSD          ]

                     time -->
  Metal GPU:  [--layer7 attn--][--layer13 attn--][--layer19 attn--]
  SSD read:       [--load 13--]     [--load 19--]     [--load 25--]
  SSD write:                   [-save 7-]        [-save 13-]

Double-buffer invariant:
  While GPU runs attention on ping, CPU reads next layer KV into pong.
  On attention completion: swap ping/pong.
  Write old ping back to SSD asynchronously.
```

#### 5.4.3 SSD I/O Strategy for KV Cache

Two distinct access patterns must be handled:

**Prefill (prompt processing):**
- All K/V positions are written sequentially from token 0 to seq_len.
- Use mmap(MAP_SHARED | MAP_POPULATE) or pwrite with large aligned writes.
- Batch all token writes for a layer into one sequential I/O operation.
- Pattern: sequential write -- achieves near-peak SSD write bandwidth.

**Decode (generation):**
- Each step appends one new token's K/V to the end of the cache.
- Requires a single ~(2 * 8 * 256 * 2) = 8 KB write per global layer per step.
- Batch multiple decode steps before writing to reduce write amplification.
- Pattern: append-only writes with small granularity; batch 16-32 tokens.

#### 5.4.4 KvOffloadManager Struct

```rust
// burn-ggml/src/memory/kv_offload.rs
use std::{path::PathBuf, sync::Arc};
use tokio::sync::Mutex;

struct KvLayerFile {
    layer_idx: usize,
    path: PathBuf,
    mmap: Option<memmap2::MmapMut>,
    stored_tokens: usize,
    max_tokens: usize,
}

pub struct KvOffloadManager {
    cache_dir: PathBuf,
    num_global_layers: usize,
    files: Mutex<Vec<KvLayerFile>>,
    ping: Arc<KvBuffer>,
    pong: Arc<KvBuffer>,
}

impl KvOffloadManager {
    pub async fn swap_and_get(&self, global_layer_idx: usize) -> Arc<KvBuffer> {
        self.swap_buffers().await;
        self.ping.clone()
    }

    pub fn prefetch_next(&self, next_global_layer_idx: usize) {
        let mgr = self.clone_handle();
        tokio::spawn(async move {
            mgr.load_into_pong(next_global_layer_idx).await;
        });
    }

    pub fn writeback_async(&self, global_layer_idx: usize, buf: Arc<KvBuffer>) {
        let mgr = self.clone_handle();
        tokio::spawn(async move {
            let mut files = mgr.files.lock().await;
            let kv_file = &mut files[global_layer_idx];
            kv_file.write_from_buffer(&buf).await;
        });
    }

    async fn load_into_pong(&self, global_layer_idx: usize) {
        let files = self.files.lock().await;
        let kv_file = &files[global_layer_idx];
        let mmap = kv_file.mmap.as_ref().expect("KV file must be initialized");
        self.pong.copy_from_slice(mmap.as_ref()).await;
    }
}
```

### 5.5 Compute-Prefetch Overlap: Latency Hiding

The key insight is that SSD reads and GPU compute can happen concurrently.
The timeline of a decode step looks like:

```
Layer K timeline (global attention layer):

     -------- wall clock ---------►

     [ Metal: layer K-1 MLP      ]
                [ SSD load: layer K weights (prefetch)  ]
                [ SSD load: layer K KV into pong        ]
                               [ Metal: layer K QKV proj ]
                               [ ping/pong swap          ]
                               [ Metal: layer K attention ]
                                         [ SSD writeback K-1 KV ]
                                         [ SSD load: K+1 KV     ]
                                                    [ Metal: layer K MLP ]

Overlap achieved:
  - Layer K weight prefetch overlaps with layer K-1 MLP
  - Layer K+1 KV prefetch overlaps with layer K attention
  - Layer K-1 KV writeback overlaps with layer K attention
```

#### 5.5.1 Throughput Analysis

```
SSD bandwidth: 3 GB/s (conservative; M-series MacBooks often achieve 5-7 GB/s)

Per-layer weight size at Q3_K_M (46 layers, ~14.5 GB total):
  ~14.5 GB / 46 layers = ~315 MB per layer

Time to prefetch one layer:
  315 MB / 3000 MB/s = ~105 ms

Metal compute time per layer at 2 tok/s with 46 layers:
  1000 ms / 2 tok/s / 46 layers = ~11 ms per layer

For global KV prefetch (INT8, 1.07 GB/layer):
  1070 MB / 3000 MB/s = ~357 ms
  Interval between global layers: ~6 layers x 11 ms = ~66 ms of compute

Implication: weight streaming at 3 GB/s is marginal at 2 tok/s without a
large prefetch window. With N=4 slots prefetched ahead:
  Available prefetch time = 4 x 11 ms = 44 ms  (at 5 GB/s: 63 ms load time)

IQ2_XXS (~8.7 GB, ~190 MB/layer) provides more comfortable streaming margin.
Empirical benchmarking on real hardware is essential to tune N.
```

#### 5.5.2 Prefetch Pipeline Task

```rust
/// Spawned once per forward pass. Manages the prefetch pipeline.
async fn prefetch_pipeline_task(
    layer_cache: Arc<LayerWeightCache>,
    kv_offload: Arc<KvOffloadManager>,
    mut layer_completion_rx: tokio::sync::mpsc::Receiver<usize>,
) {
    while let Some(completed_layer) = layer_completion_rx.recv().await {
        let next = completed_layer + 1;
        layer_cache.prefetch(next);
        if is_global_attention_layer(next) {
            kv_offload.prefetch_next(to_global_idx(next));
        }
    }
}
```

---

## 6. Mixture-of-Experts Support

This section covers the MoE-specific components needed for Gemma 4 26B. These
are introduced in **Phase 1** (wgpu/Linux) and reused without change in
**Phase 2** (ggml/Metal), since the expert cache and routing logic are
backend-agnostic Rust.

### 6.1 Expert Weight Cache (`ExpertWeightCache`)

The `ExpertWeightCache` is a generalization of `LayerWeightCache` (Section 5.3)
operating at expert granularity. The key difference: access order is
**router-driven** (unpredictable) rather than sequential.

```rust
// burn-ggml/src/memory/expert_cache.rs

pub struct ExpertKey {
    pub layer_idx:  usize,
    pub expert_idx: usize,
}

pub struct ExpertWeightCache {
    model_path:    PathBuf,
    max_slots:     usize,                        // e.g. 32
    slots:         Mutex<LruCache<ExpertKey, ExpertSlot>>,
    expert_index:  Vec<Vec<ExpertOffsetInfo>>,   // [layer][expert] → file offset
}

impl ExpertWeightCache {
    /// Ensure expert weights are resident, blocking if not yet loaded.
    pub async fn get(&self, key: ExpertKey) -> Arc<ExpertWeightGuard> { ... }

    /// Fire-and-forget: begin loading expert weights from SSD.
    /// Called immediately after the router produces its top-k indices.
    pub fn prefetch(&self, keys: &[ExpertKey]) {
        for key in keys {
            if !self.is_resident(key) {
                let cache = self.clone_handle();
                let key = *key;
                tokio::spawn(async move { cache.load(key).await; });
            }
        }
    }

    async fn load(&self, key: ExpertKey) {
        let offset = self.expert_index[key.layer_idx][key.expert_idx];
        // mmap + pread from GGUF, copy into backend buffer
        let buf = self.read_expert_bytes(offset).await;
        let mut slots = self.slots.lock().await;
        if slots.len() == self.max_slots {
            slots.pop_lru(); // evict least-recently-used expert
        }
        slots.put(key, ExpertSlot::ready(buf));
    }
}
```

### 6.2 MoE Routing Kernel

The router computes a softmax over 128 expert logits and selects the top-8
indices. This is a small operation (~128 floats per token) but must be fast
since it gates the prefetch.

#### 6.2.1 wgpu WGSL Implementation (Phase 1)

```wgsl
// shaders/moe_router.wgsl
//
// Dispatch: 1 workgroup per token, 128 threads per workgroup (1 per expert).
// Each thread owns one expert's logit score.

@group(0) @binding(0) var<storage, read>       gate_logits: array<f32>;  // [n_tokens × 128]
@group(0) @binding(1) var<storage, read_write> top_k_idx:   array<u32>;  // [n_tokens × 8]
@group(0) @binding(2) var<storage, read_write> top_k_wt:    array<f32>;  // [n_tokens × 8]

var<workgroup> scores:    array<f32, 128>;
var<workgroup> indices:   array<u32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let token = wg.x;
    let e     = lid.x;

    // Load logit for this expert
    scores[e]  = gate_logits[token * 128u + e];
    indices[e] = e;
    workgroupBarrier();

    // Softmax: find max for numerical stability
    var local_max = scores[e];
    // ... parallel reduction for max, then exp(score - max), then sum
    // (standard workgroup reduction in log2(128) = 7 steps)

    // Top-8 selection: partial insertion sort in shared memory
    // Each of 8 rounds: find global max, write to output, set to -inf
    for (var k = 0u; k < 8u; k++) {
        workgroupBarrier();
        // round k: find argmax via reduction, write index + weight
        // (128 threads cooperate; winner thread writes output)
    }
}
```

For the subgroup-capable path (Intel iGPU with `Features::SUBGROUP`), the
reduction steps use `subgroupMax` and `subgroupBallot` to eliminate most
`workgroupBarrier()` calls, reducing latency by ~4 barrier steps.

#### 6.2.2 ggml Implementation (Phase 2)

ggml already implements this via `ggml_argsort` + `ggml_top_k` in the CPU
path and `topk-moe.cu` on CUDA. For Metal, the ggml scheduler falls back to
the CPU path for the routing step (it is small enough that CPU is acceptable).
No new kernel needed in Phase 2.

### 6.3 Expert-Indexed Batched GEMM (`MUL_MAT_ID`)

After routing, the FFN executes 8 independent matrix multiplications — one per
selected expert. Naively this is 8 separate `float_matmul` calls; the efficient
path batches them into a single indexed GEMM dispatch.

#### 6.3.1 wgpu WGSL Implementation (Phase 1)

This is the most complex new kernel. The reference is ggml's Vulkan
`GGML_OP_MUL_MAT_ID` shader (`ggml-vulkan.cpp`).

```wgsl
// shaders/mul_mat_id.wgsl
//
// Executes: out[token] = weight[expert_id[token]] @ input[token]
// for each token, where expert_id is the routing assignment.
//
// Dispatch: one workgroup per (expert, output-tile).

@group(0) @binding(0) var<storage, read>       weights: array<f32>;  // [n_experts × K × N]
@group(0) @binding(1) var<storage, read>        inputs: array<f32>;  // [n_tokens × K]
@group(0) @binding(2) var<storage, read>    expert_ids: array<u32>;  // [n_tokens × top_k]
@group(0) @binding(3) var<storage, read_write> outputs: array<f32>;  // [n_tokens × N]

var<workgroup> tile_w: array<f32, 256>;  // weight tile
var<workgroup> tile_i: array<f32, 256>;  // input tile

struct PushConstants {
    M: u32, N: u32, K: u32,
    expert_stride: u32,  // K × N
    n_tokens: u32,
}
var<push_constant> pc: PushConstants;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let expert = wg.z;
    // Tiled GEMM: load tile_w from weights[expert * expert_stride ...],
    //             load tile_i from inputs for tokens assigned to this expert,
    //             accumulate dot products in registers,
    //             write to outputs.
    // Standard 16×16 tiled matmul with workgroupBarrier() between tiles.
}
```

#### 6.3.2 ggml Implementation (Phase 2)

`GGML_OP_MUL_MAT_ID` is a first-class ggml op with Metal and CPU
implementations. In Phase 2 this kernel is provided by ggml at no extra cost —
the `GemmaRunner` calls `ggml_mul_mat_id` directly in the graph construction.

### 6.4 Shared Expert

Gemma 4 26B includes one always-active shared expert per MoE layer (DeepSeek
V2 pattern). Its weights are always resident in RAM (never evicted from the
expert cache) and its output is added to the sum of the top-8 routed expert
outputs before the residual connection.

```rust
impl MoeLayer {
    pub fn forward(&self, x: &Tensor, cache: &ExpertWeightCache) -> Tensor {
        // 1. Compute router logits
        let logits = self.gate.forward(x);

        // 2. Select top-8 experts; fire prefetch immediately
        let (top_k_idx, top_k_weights) = router_top_k(&logits, 8);
        cache.prefetch(&top_k_idx.to_expert_keys(self.layer_idx));

        // 3. Shared expert (always resident, no cache lookup needed)
        let shared_out = self.shared_expert.forward(x);

        // 4. Routed experts via MUL_MAT_ID (waits on prefetch if needed)
        let expert_out = self.mul_mat_id(x, &top_k_idx, &top_k_weights, cache);

        // 5. Combine
        shared_out + expert_out
    }
}
```

### 6.5 `WeightCache<T>`: Unified Abstraction

`ExpertWeightCache` and `LayerWeightCache` share the same underlying pattern.
In the implementation they are unified as a single generic type:

```rust
pub struct WeightCache<K: CacheKey> {
    model_path:  PathBuf,
    max_slots:   usize,
    slots:       Mutex<LruCache<K, WeightSlot>>,
    index:       K::Index,   // byte-offset lookup table
}

impl<K: CacheKey> WeightCache<K> {
    pub async fn get(&self, key: K) -> Arc<WeightGuard> { ... }
    pub fn prefetch(&self, keys: &[K]) { ... }
}

pub type LayerWeightCache  = WeightCache<LayerKey>;
pub type ExpertWeightCache = WeightCache<ExpertKey>;
```

This means the prefetch pipeline, LRU eviction, async I/O, and
`PrefetchOps` integration are written once and shared between both use cases.

---

## 7. ggml Graph Construction for Attention


ggml uses a **deferred computation graph** model: you build an expression
graph of tensor operations, then call `ggml_graph_compute` once to execute
the entire graph. This maps well to how Metal command buffers work.

### 7.1 Key ggml Operations

| ggml function                | Description                              |
|------------------------------|------------------------------------------|
| `ggml_mul_mat(ctx, A, B)`    | Matrix multiply: C = A @ B               |
| `ggml_add(ctx, a, b)`        | Element-wise addition                    |
| `ggml_rms_norm(ctx, x, eps)` | RMS normalization                        |
| `ggml_rope(ctx, x, pos, ...)`| Rotary position embedding                |
| `ggml_soft_max(ctx, x)`      | Softmax                                  |
| `ggml_flash_attn_ext(ctx, Q, K, V, mask, scale, ...)` | Fused flash attention |
| `ggml_silu(ctx, x)`          | SiLU activation                          |
| `ggml_cont(ctx, x)`          | Make tensor contiguous                   |
| `ggml_view_2d(ctx, x, ...)`  | View (no copy) of a tensor region        |
| `ggml_cpy(ctx, src, dst)`    | Copy tensor data                         |

### 7.2 Local Sliding-Window Attention Graph

```c
// Simplified C pseudocode for local attention via ggml
// In practice called from Rust via ggml-sys FFI

struct ggml_tensor* build_local_attention(
    struct ggml_context* ctx,
    struct ggml_tensor*  hidden,       // [seq_len, d_model]
    struct ggml_tensor*  W_q, *W_k, *W_v, *W_o,  // weight matrices
    struct ggml_tensor*  kv_cache_k,   // [window, n_kv_heads, head_dim]
    struct ggml_tensor*  kv_cache_v,   // [window, n_kv_heads, head_dim]
    int                  window_size,
    int                  n_heads,
    int                  n_kv_heads,
    int                  head_dim,
    float                scale
) {
    // 1. QKV projections
    struct ggml_tensor* Q = ggml_mul_mat(ctx, W_q, hidden);  // [seq, n_heads*head_dim]
    struct ggml_tensor* K = ggml_mul_mat(ctx, W_k, hidden);  // [seq, n_kv_heads*head_dim]
    struct ggml_tensor* V = ggml_mul_mat(ctx, W_v, hidden);

    // 2. Reshape for multi-head layout
    Q = ggml_reshape_3d(ctx, Q, head_dim, n_heads, seq_len);
    K = ggml_reshape_3d(ctx, K, head_dim, n_kv_heads, seq_len);
    V = ggml_reshape_3d(ctx, V, head_dim, n_kv_heads, seq_len);

    // 3. Apply local RoPE (standard base freq)
    Q = ggml_rope(ctx, Q, positions, head_dim, LLAMA_ROPE_TYPE_NORM, n_ctx);
    K = ggml_rope(ctx, K, positions, head_dim, LLAMA_ROPE_TYPE_NORM, n_ctx);

    // 4. Update sliding-window KV cache
    K = ggml_cpy(ctx, K, ggml_view_kv_window(kv_cache_k, cur_pos, window_size));
    V = ggml_cpy(ctx, V, ggml_view_kv_window(kv_cache_v, cur_pos, window_size));

    // 5. Build local attention mask (causal within window)
    struct ggml_tensor* mask = build_sliding_window_mask(ctx, seq_len, window_size);

    // 6. Flash attention (fused QK^T softmax V)
    struct ggml_tensor* attn = ggml_flash_attn_ext(
        ctx, Q, K, V, mask, scale,
        0.0f,  // max_bias
        0.0f   // logit_softcap (Gemma uses 0)
    );

    // 7. Output projection
    attn = ggml_reshape_2d(ctx, attn, n_heads * head_dim, seq_len);
    return ggml_mul_mat(ctx, W_o, attn);
}
```

### 7.3 Global Full-Context Attention Graph

Global attention differs in three ways:
1. Uses Proportional RoPE (base 1M, scale 8) instead of standard RoPE.
2. KV cache spans the full context (up to 256K tokens), not a sliding window.
3. The KV cache is loaded from SSD via the ping-pong buffer rather than
   being resident in RAM.

```c
struct ggml_tensor* build_global_attention(
    struct ggml_context* ctx,
    struct ggml_tensor*  hidden,
    struct ggml_tensor*  W_q, *W_k, *W_v, *W_o,
    struct ggml_tensor*  kv_ping,   // ping buffer, loaded from SSD
    int                  n_ctx,      // full context length (up to 256K)
    ...)
{
    // 1-2: Same QKV projections and reshape as local attention

    // 3. Apply Proportional RoPE (global variant)
    //    rope_type = LLAMA_ROPE_TYPE_NEOX with:
    //    rope_freq_base = 1_000_000.0f
    //    rope_freq_scale = 1.0f / 8.0f
    Q = ggml_rope_ext(ctx, Q, positions, NULL, head_dim,
                      LLAMA_ROPE_TYPE_NEOX,
                      n_ctx,
                      1000000.0f,   // base freq (Proportional RoPE)
                      0.125f,       // scale = 1/8
                      0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(ctx, K, positions, NULL, head_dim,
                      LLAMA_ROPE_TYPE_NEOX,
                      n_ctx, 1000000.0f, 0.125f,
                      0.0f, 1.0f, 0.0f, 0.0f);

    // 4. Write new K/V to the ping buffer (at current token position)
    struct ggml_tensor* K_view = ggml_view_kv_at(kv_ping, cur_pos);
    struct ggml_tensor* V_view = ggml_view_kv_at(kv_ping, cur_pos);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, K, K_view));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, V, V_view));

    // 5. Full causal mask (no window restriction)
    struct ggml_tensor* mask = build_causal_mask(ctx, n_ctx);

    // 6. Flash attention over full context
    //    NOTE: KQ_scale must account for head_dim
    float kq_scale = 1.0f / sqrtf((float)head_dim);
    struct ggml_tensor* kv_K = ggml_view_k(kv_ping, n_ctx, n_kv_heads, head_dim);
    struct ggml_tensor* kv_V = ggml_view_v(kv_ping, n_ctx, n_kv_heads, head_dim);
    struct ggml_tensor* attn = ggml_flash_attn_ext(
        ctx, Q, kv_K, kv_V, mask, kq_scale, 0.0f, 0.0f);

    // 7. Output projection
    attn = ggml_reshape_2d(ctx, attn, n_heads * head_dim, 1);
    return ggml_mul_mat(ctx, W_o, attn);
}
```

### 7.4 Graph Compute Execution

```rust
// burn-ggml/src/graph.rs
use ggml_sys::*;

pub struct GgmlGraphExecutor {
    backend: *mut ggml_backend,
    /// Allocator that assigns tensors to backend buffers.
    allocr: *mut ggml_backend_sched,
}

impl GgmlGraphExecutor {
    pub unsafe fn compute(&self, gf: *mut ggml_cgraph) -> Result<(), GgmlError> {
        // 1. Allocate graph tensors to backend memory
        let ok = ggml_backend_sched_alloc_graph(self.allocr, gf);
        if !ok {
            return Err(GgmlError::AllocationFailed);
        }

        // 2. Execute the graph on the backend (Metal or CPU)
        let status = ggml_backend_sched_graph_compute(self.allocr, gf);
        if status != GGML_STATUS_SUCCESS {
            return Err(GgmlError::ComputeFailed(status));
        }

        Ok(())
    }
}
```

### 7.5 MLP Block

Gemma 4 uses a gated MLP (GeGLU variant with SiLU activation):

```c
struct ggml_tensor* build_mlp(
    struct ggml_context* ctx,
    struct ggml_tensor*  x,          // [seq, d_model]
    struct ggml_tensor*  W_gate,     // [d_model, d_ff]
    struct ggml_tensor*  W_up,       // [d_model, d_ff]
    struct ggml_tensor*  W_down      // [d_ff, d_model]
) {
    // Gated SiLU: output = W_down @ (silu(W_gate @ x) * (W_up @ x))
    struct ggml_tensor* gate = ggml_mul_mat(ctx, W_gate, x);
    struct ggml_tensor* up   = ggml_mul_mat(ctx, W_up,   x);
    gate = ggml_silu(ctx, gate);
    struct ggml_tensor* hidden = ggml_mul(ctx, gate, up);
    return ggml_mul_mat(ctx, W_down, hidden);
}
```

---

## 8. Burn Integration Points

### 8.1 Module-Level Operations (ModuleOps)

Burn's `ModuleOps` trait provides high-level operations used directly by
model modules. These map to ggml graph nodes as follows:

| Burn ModuleOps method        | ggml equivalent                          |
|------------------------------|------------------------------------------|
| `linear_forward`             | `ggml_mul_mat` + optional `ggml_add`     |
| `embedding_forward`          | `ggml_get_rows`                          |
| `layer_norm_forward`         | `ggml_norm` + `ggml_mul` + `ggml_add`   |
| `rms_norm_forward`           | `ggml_rms_norm` + `ggml_mul`            |
| `conv2d_forward`             | `ggml_conv_2d`                           |
| `adaptive_avg_pool2d`        | `ggml_pool_2d`                           |

Implementation skeleton for linear and RMS norm:

```rust
// burn-ggml/src/ops/module_ops.rs
use burn::ops::ModuleOps;
use ggml_sys::*;
use crate::{GgmlBackend, GgmlTensor};

impl ModuleOps<GgmlBackend> for GgmlBackend {
    fn linear_forward(
        x: GgmlTensor,
        weight: GgmlTensor,
        bias: Option<GgmlTensor>,
    ) -> GgmlTensor {
        let ctx = x.ctx.clone();
        unsafe {
            let gf = ggml_new_graph(ctx.ptr);
            // ggml_mul_mat: first arg is the weight (transposed convention)
            let mut out = ggml_mul_mat(ctx.ptr, weight.ptr, x.ptr);
            if let Some(b) = bias {
                out = ggml_add(ctx.ptr, out, b.ptr);
            }
            ggml_build_forward_expand(gf, out);
            ctx.executor.compute(gf).expect("linear_forward compute failed");
            GgmlTensor::from_raw(out, ctx)
        }
    }

    fn embedding_forward(
        weight: GgmlTensor,   // [vocab_size, d_model]
        indices: GgmlTensor,  // [seq_len] i32
    ) -> GgmlTensor {
        let ctx = weight.ctx.clone();
        unsafe {
            let gf = ggml_new_graph(ctx.ptr);
            // ggml_get_rows: rows of weight indexed by indices
            let out = ggml_get_rows(ctx.ptr, weight.ptr, indices.ptr);
            ggml_build_forward_expand(gf, out);
            ctx.executor.compute(gf).expect("embedding_forward compute failed");
            GgmlTensor::from_raw(out, ctx)
        }
    }

    fn rms_norm_forward(
        x: GgmlTensor,
        weight: GgmlTensor,  // elementwise scale (gamma)
        eps: f64,
    ) -> GgmlTensor {
        let ctx = x.ctx.clone();
        unsafe {
            let gf = ggml_new_graph(ctx.ptr);
            let normed = ggml_rms_norm(ctx.ptr, x.ptr, eps as f32);
            let out = ggml_mul(ctx.ptr, normed, weight.ptr);
            ggml_build_forward_expand(gf, out);
            ctx.executor.compute(gf).expect("rms_norm_forward compute failed");
            GgmlTensor::from_raw(out, ctx)
        }
    }
}
```

### 8.2 Quantized Tensor Support (QTensorOps)

Burn's `QTensorOps` trait handles quantization and dequantization at the
framework level. `burn-ggml` maps Burn's quantization scheme to ggml types:

| Burn QuantizationScheme | ggml type         | Notes                           |
|-------------------------|-------------------|---------------------------------|
| Symmetric INT8          | GGML_TYPE_Q8_0    | 8-bit per-block symmetric       |
| Affine INT8             | GGML_TYPE_Q8_1    | 8-bit per-block affine          |
| INT4 block-wise         | GGML_TYPE_Q4_K    | K-quant INT4 (best for weights) |
| INT3 block-wise         | GGML_TYPE_Q3_K    | K-quant INT3                    |
| IQ3 importance          | GGML_TYPE_IQ3_S   | Importance-quantized INT3       |
| IQ2 importance          | GGML_TYPE_IQ2_XXS | Aggressive 2-bit                |

#### 7.2.1 GgmlQuantizedTensor

```rust
// burn-ggml/src/tensor.rs (continued)

/// Quantized tensor backed by a ggml quantized tensor.
/// The underlying ggml_tensor has a quantized type (Q4_K, Q3_K, etc.).
#[derive(Clone, Debug)]
pub struct GgmlQuantizedTensor {
    pub inner: GgmlTensor,
    pub scheme: burn::tensor::quantization::QuantizationScheme,
}

impl burn::tensor::quantization::QTensorPrimitive for GgmlQuantizedTensor {
    fn scheme(&self) -> &burn::tensor::quantization::QuantizationScheme {
        &self.scheme
    }
}
```

#### 7.2.2 GGUF Weight Loading into Quantized Tensors

When a GGUF model is loaded, quantized weight tensors are read directly into
ggml-managed memory and wrapped as `GgmlQuantizedTensor`. No dequantization
happens at load time -- ggml's `ggml_mul_mat` kernel handles Q4_K/Q3_K weights
transparently, dequantizing blocks on-the-fly during matrix multiply.

This is zero-copy on the hot path: the GGUF tensor data bytes are the same
format as what ggml uses internally.

### 8.3 Device Management

Each `GgmlDevice` variant maps to a different ggml backend configuration:

```rust
// burn-ggml/src/context.rs

pub struct GgmlContext {
    pub(crate) ptr: *mut ggml_sys::ggml_context,
    /// The ggml backend (Metal or CPU)
    pub(crate) backend: *mut ggml_sys::ggml_backend,
    /// Scheduler for multi-backend execution (CPU fallback for unsupported ops)
    pub(crate) sched: *mut ggml_sys::ggml_backend_sched,
    pub(crate) executor: Arc<GgmlGraphExecutor>,
    pub(crate) device: GgmlDevice,
    /// Optional offload managers (present only for MetalWithOffload)
    pub(crate) layer_cache: Option<Arc<LayerWeightCache>>,
    pub(crate) kv_offload: Option<Arc<KvOffloadManager>>,
}

impl GgmlContext {
    pub fn new(device: GgmlDevice) -> Self {
        unsafe {
            let params = ggml_sys::ggml_init_params {
                mem_size: 256 * 1024 * 1024,  // 256MB scratch for graph nodes
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,  // tensors allocated via backend buffer, not here
            };
            let ctx = ggml_sys::ggml_init(params);

            let (backend, layer_cache, kv_offload) = match &device {
                GgmlDevice::Cpu => {
                    let b = ggml_sys::ggml_backend_cpu_init();
                    (b, None, None)
                }
                GgmlDevice::Metal => {
                    let b = ggml_sys::ggml_backend_metal_init();
                    (b, None, None)
                }
                GgmlDevice::MetalWithOffload { kv_cache_dir, max_layers_in_ram } => {
                    let b = ggml_sys::ggml_backend_metal_init();
                    let lc = Arc::new(LayerWeightCache::new(*max_layers_in_ram));
                    let kv = Arc::new(KvOffloadManager::new(kv_cache_dir.clone()));
                    (b, Some(lc), Some(kv))
                }
            };

            // Set up scheduler with Metal primary + CPU fallback
            let backends = [backend, ggml_sys::ggml_backend_cpu_init()];
            let sched = ggml_sys::ggml_backend_sched_new(
                backends.as_ptr(),
                std::ptr::null_mut(),  // no custom buffer types
                backends.len() as i32,
                GGML_DEFAULT_GRAPH_SIZE,
                false,
            );

            GgmlContext {
                ptr: ctx,
                backend,
                sched,
                executor: Arc::new(GgmlGraphExecutor { backend, allocr: sched }),
                device,
                layer_cache,
                kv_offload,
            }
        }
    }
}
```

### 8.4 `PrefetchOps`: Burn API Extension

#### 7.4.1 Motivation

Burn's `Backend` trait has no mechanism for a backend to begin loading tensor
data asynchronously before it is needed for computation. This is a fundamental
gap for the layer-streaming and KV-offload use case: by the time a tensor
reaches an operation like `float_matmul`, it must already be in device memory.
There is no hook to say "start fetching this tensor now, I'll need it in two
layers."

The solution is a small, non-breaking addition to the Burn backend trait
surface: a `PrefetchOps` trait with a single fire-and-forget method. All
existing backends get a no-op default. Only `burn-ggml` (and any future
disk-backed backend) overrides it.

Critically, **the call site lives in model-specific inference code**, not inside
the backend. The prefetch schedule is inherently model-architecture-dependent:
Gemma 4's 5:1 local/global interleaving pattern is completely different from a
pure MLA model or a mixture-of-experts architecture. No framework can derive
this schedule generically — it must be hand-authored in the `GemmaRunner`
forward loop.

#### 7.4.2 Trait Definition

The following additions are proposed to `burn-backend`:

```rust
// burn-backend/src/backend/ops/prefetch.rs  (new file)

use crate::Backend;
use burn_tensor::ops::{FloatTensor, IntTensor};

/// Tensors to be prefetched toward a device.
#[derive(Default)]
pub struct PrefetchPrimitive<B: Backend> {
    pub floats: Vec<FloatTensor<B>>,
    pub ints:   Vec<IntTensor<B>>,
}

/// Optional backend capability: begin moving tensors toward `device`
/// asynchronously, before they are needed for computation.
///
/// This is a fire-and-forget hint. The backend may ignore it entirely.
/// The contract is:
///   - Returns immediately (non-blocking).
///   - By the time the tensors are used in an operation, the backend
///     should have them resident in device memory.
///   - It is safe to call prefetch on a tensor that is already resident
///     (the backend should detect this and no-op).
pub trait PrefetchOps<B: Backend> {
    fn prefetch(primitive: PrefetchPrimitive<B>, device: &B::Device) {
        // Default: no-op. CPU/GPU backends where data is already resident
        // do nothing here.
        let _ = (primitive, device);
    }
}
```

Add `PrefetchOps<Self>` to the `Backend` supertrait in `base.rs`:

```rust
// burn-backend/src/backend/base.rs  (modified)
pub trait Backend:
    FloatTensorOps<Self>
    + BoolTensorOps<Self>
    + IntTensorOps<Self>
    + ModuleOps<Self>
    + ActivationOps<Self>
    + QTensorOps<Self>
    + TransactionOps<Self>
    + PrefetchOps<Self>          // <-- new
    + Clone + Default + Sized + Send + Sync + core::fmt::Debug + 'static
{ ... }
```

#### 7.4.3 Public Tensor API

Expose prefetch on `Tensor<B, D>` in `burn-tensor`:

```rust
// burn-tensor/src/tensor/api/base.rs  (addition)

impl<B: Backend, const D: usize> Tensor<B, D> {
    /// Hint to the backend to begin loading this tensor's data toward `device`.
    ///
    /// Non-blocking. Returns `self` unchanged so calls can be chained.
    /// Safe to call speculatively — backends that don't support prefetching
    /// ignore this call entirely (the default implementation is a no-op).
    pub fn prefetch(self, device: &B::Device) -> Self {
        use burn_backend::ops::prefetch::{PrefetchOps, PrefetchPrimitive};
        B::prefetch(
            PrefetchPrimitive { floats: vec![self.primitive.clone()], ints: vec![] },
            device,
        );
        self
    }
}
```

#### 7.4.4 `burn-ggml` Implementation

`burn-ggml` is the only backend that provides a real implementation:

```rust
// burn-ggml/src/ops/prefetch_ops.rs

use burn_backend::ops::prefetch::{PrefetchOps, PrefetchPrimitive};
use crate::{GgmlBackend, GgmlDevice, GgmlTensor};

impl PrefetchOps<GgmlBackend> for GgmlBackend {
    fn prefetch(primitive: PrefetchPrimitive<GgmlBackend>, device: &GgmlDevice) {
        let GgmlDevice::MetalWithOffload { .. } = device else {
            return; // Cpu and Metal variants: data already resident, no-op
        };
        let ctx = GgmlContext::get(device);
        let Some(layer_cache) = &ctx.layer_cache else { return };

        for tensor in primitive.floats {
            if let Some(layer_idx) = tensor.layer_hint {
                // Fire-and-forget: spawn async task to load layer weights from SSD
                layer_cache.prefetch(layer_idx);
            }
        }

        let Some(kv_offload) = &ctx.kv_offload else { return };
        // KV prefetch is handled separately via KvOffloadManager::prefetch_next,
        // called directly from GemmaRunner with the next global layer index.
    }
}
```

#### 7.4.5 Call Site: GemmaRunner Forward Loop

The prefetch schedule is hand-authored in the model's inference loop. This is
intentional — it encodes Gemma 4's specific 5:1 local/global interleaving:

```rust
// Application code: gemma_runner.rs

impl GemmaRunner {
    pub fn decode_step(&mut self, token_id: u32) -> u32 {
        let device = &self.device;
        let mut x = self.model.embed(token_id);

        for (i, layer) in self.model.layers.iter().enumerate() {
            // Prefetch layer i+2 weights while layer i is computing.
            // For all backends other than MetalWithOffload this is a no-op.
            if i + 2 < self.model.layers.len() {
                self.model.layers[i + 2].weights()
                    .iter()
                    .for_each(|t| { t.clone().prefetch(device); });
            }

            // Prefetch next global KV block from SSD.
            if let Some(next_global) = self.next_global_layer_after(i) {
                self.kv_offload.prefetch_next(next_global);
            }

            // Run the current layer.
            x = layer.forward(x, &mut self.kv_cache, i);
        }

        self.model.lm_head(x)
    }
}
```

#### 7.4.6 Backend Behaviour Summary

| Backend                    | `PrefetchOps::prefetch` behaviour                          |
|----------------------------|------------------------------------------------------------|
| `burn-ndarray`             | No-op (default impl). Data already in RAM.                 |
| `burn-wgpu` / `burn-cubecl`| No-op (default impl). Data already on GPU.                 |
| `burn-tch` / `burn-candle` | No-op (default impl). PyTorch/Candle manage residency.     |
| `burn-ggml` (Cpu/Metal)    | No-op (explicit early return). Data already resident.      |
| `burn-ggml` (MetalWithOffload) | **Real impl**: fires async pread + DMA into Metal buffer. |

Zero breaking changes to existing backends. The trait addition is purely
additive; the default no-op satisfies the bound for every existing backend
without any code changes.

#### 7.4.7 Upstream Contribution Path

This is a small, well-motivated addition suitable for an upstream Burn PR:

1. Add `burn-backend/src/backend/ops/prefetch.rs` (~40 lines)
2. Add `PrefetchOps<Self>` to `Backend` supertrait in `base.rs` (~1 line)
3. Add `Tensor::prefetch()` to `burn-tensor/src/tensor/api/base.rs` (~10 lines)
4. Add default no-op impl to all existing backends via the trait default (~0 lines)

The PR description would frame it as a general "async data movement hint" API,
not specific to ggml or SSD offloading — making it useful for any future backend
that has non-trivial data residency costs (e.g., a hypothetical NVMe-backed
training backend, or a distributed backend that fetches shards over the network).

---

## 9. GGUF Model Loading

GGUF (Generic GPU Unified Format) is the self-describing model file format
used by llama.cpp. It encodes tensor names, shapes, quantization types, and
all model hyperparameters in a single binary file.

### 9.1 GGUF File Structure

```
GGUF file layout:

  [magic: 4 bytes]           'GGUF'
  [version: 4 bytes]         3
  [n_tensors: 8 bytes]       number of tensors
  [n_kv_pairs: 8 bytes]      number of metadata key-value pairs

  [metadata section]
    kv_pair_0: key (string) + value (typed)
    kv_pair_1: ...  (model architecture, context length, etc.)
    ...

  [tensor info section]
    tensor_0: name, n_dims, shape[4], type, offset
    tensor_1: ...
    ...

  [padding to alignment]

  [tensor data section]
    raw quantized/float tensor data back-to-back
```

### 9.2 Loading Strategy

The loader uses **lazy, memory-mapped loading**:

1. Parse the GGUF header and tensor metadata into a `GgufIndex` struct.
   This is cheap (metadata is small, typically <1 MB).
2. `mmap` the entire GGUF file into virtual address space.
   No data is actually read from disk until accessed.
3. When a layer's weights are needed (during `LayerWeightCache::get_layer`),
   the relevant byte range is read from the mmap'd region.
4. The data is copied into a ggml-managed `ggml_backend_buffer` and
   wrapped as `GgmlTensor` / `GgmlQuantizedTensor`.

### 9.3 Gemma 4 Tensor Name Mapping

GGUF tensor names follow the llama.cpp naming convention. The Gemma 4
model maps to these names:

| Gemma 4 parameter              | GGUF tensor name                               |
|--------------------------------|------------------------------------------------|
| Token embedding                | `token_embd.weight`                            |
| Final RMSNorm                  | `output_norm.weight`                           |
| Output logit projection        | `output.weight`                                |
| Layer i attention norm         | `blk.{i}.attn_norm.weight`                     |
| Layer i Q projection           | `blk.{i}.attn_q.weight`                        |
| Layer i K projection           | `blk.{i}.attn_k.weight`                        |
| Layer i V projection           | `blk.{i}.attn_v.weight`                        |
| Layer i output projection      | `blk.{i}.attn_output.weight`                   |
| Layer i FFN norm               | `blk.{i}.ffn_norm.weight`                      |
| Layer i FFN gate               | `blk.{i}.ffn_gate.weight`                      |
| Layer i FFN up                 | `blk.{i}.ffn_up.weight`                        |
| Layer i FFN down               | `blk.{i}.ffn_down.weight`                      |

### 9.4 GgufLoader Implementation

```rust
// burn-ggml/src/gguf.rs
use std::{collections::HashMap, path::Path, sync::Arc};
use memmap2::Mmap;

/// Metadata about a single tensor in a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub ggml_type: u32,  // ggml_type enum value
    /// Byte offset of tensor data from the start of the tensor data section.
    pub data_offset: u64,
    pub data_size: usize,
}

/// Parsed GGUF index: metadata + tensor directory, no tensor data loaded yet.
pub struct GgufIndex {
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
    /// Byte offset in the file where tensor data begins.
    pub data_section_offset: u64,
    mmap: Mmap,
}

impl GgufIndex {
    pub fn open(path: &Path) -> Result<Self, GgufError> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut cursor = std::io::Cursor::new(mmap.as_ref());
        let magic = read_u32(&mut cursor)?;
        if magic != 0x46554747 {  // 'GGUF' in little-endian
            return Err(GgufError::InvalidMagic);
        }
        let version = read_u32(&mut cursor)?;
        let n_tensors = read_u64(&mut cursor)?;
        let n_kv = read_u64(&mut cursor)?;

        let metadata = parse_metadata(&mut cursor, n_kv)?;
        let tensors = parse_tensor_infos(&mut cursor, n_tensors)?;

        // Align to GGUF alignment (default 32 bytes)
        let alignment = metadata.get("general.alignment")
            .and_then(|v| v.as_u32()).unwrap_or(32) as u64;
        let pos = cursor.position();
        let data_section_offset = (pos + alignment - 1) / alignment * alignment;

        Ok(GgufIndex { metadata, tensors, data_section_offset, mmap })
    }

    /// Get a slice of raw tensor data from the mmap'd file.
    /// Zero-copy: returns a reference into the mmap'd region.
    pub fn tensor_data_bytes(&self, name: &str) -> Result<&[u8], GgufError> {
        let info = self.tensors.get(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;
        let start = (self.data_section_offset + info.data_offset) as usize;
        let end = start + info.data_size;
        Ok(&self.mmap[start..end])
    }

    /// Load a tensor into a ggml backend buffer as a GgmlTensor.
    pub unsafe fn load_tensor(
        &self,
        name: &str,
        ctx: &Arc<GgmlContext>,
    ) -> Result<GgmlTensor, GgufError> {
        let info = self.tensors.get(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;
        let data = self.tensor_data_bytes(name)?;

        // Allocate ggml tensor on the backend
        let ne: [i64; 4] = shape_to_ggml_ne(&info.shape);
        let t = ggml_sys::ggml_new_tensor_4d(
            ctx.ptr,
            info.ggml_type,
            ne[0], ne[1], ne[2], ne[3],
        );

        // Copy from mmap into ggml buffer (one memcpy, unavoidable)
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            ggml_sys::ggml_get_data(t) as *mut u8,
            data.len(),
        );

        Ok(GgmlTensor::from_raw(t, ctx.clone()))
    }
}
```

### 9.5 Lazy Layer Loading

For the streaming mode (`GgmlDevice::MetalWithOffload`), weights are not
loaded at model initialization. Instead:

1. `GgufIndex::open()` parses the header (fast, ~1ms).
2. The `LayerWeightCache` receives the `GgufIndex` and per-layer byte ranges.
3. On the first call to `layer_cache.get_layer(0)`, the embedding and layer 0
   weights are loaded synchronously.
4. Subsequent layers are prefetched asynchronously as described in Section 5.3.

This means **time-to-first-token** only waits for:
- Header parse + mmap setup (~1 ms)
- Embedding weights load (~few MB, ~1 ms)
- First N=4 layers preload (~4 x 315 MB = 1.26 GB at 5 GB/s = ~252 ms)

Rather than waiting for all 14.5 GB to load (~3 seconds at 5 GB/s).

---

## 10. Implementation Plan

The implementation is split into two phases. **Phase 1** builds the full memory
management stack on Linux using `burn-wgpu` (Vulkan/Intel iGPU) — no ggml FFI,
no macOS dependency. **Phase 2** ports to `burn-ggml` on Apple Metal, gaining
quantized kernels and the full 31B dense model target. Each phase ends with a
working, benchmarked system.

```
Phase 1 (Linux, wgpu/Vulkan)          Phase 2 (macOS, ggml/Metal)
─────────────────────────────          ──────────────────────────────
PrefetchOps trait (Burn)          →    reused unchanged
WeightCache<T> (expert + layer)   →    reused unchanged
KvOffloadManager (ping-pong)      →    reused unchanged
GgufIndex (model loader)          →    reused unchanged
MoE router WGSL kernel            →    replaced by ggml CPU/Metal path
MUL_MAT_ID WGSL kernel            →    replaced by ggml_mul_mat_id
Standard matmul (wgpu)            →    replaced by ggml quantized matmul
Test bench: Gemma 4 26B MoE Q4    →    Test bench: Gemma 4 31B Q3_K_M
```

---

### Phase 1: wgpu + Offload on Linux (Weeks 1-10)

**Goal:** Gemma 4 26B MoE at Q4_K_M running on Linux with Intel iGPU via
`burn-wgpu`, with expert streaming from SSD and KV offload, targeting
>=1.5 tok/s at 32K context.

#### P1-T1 — `PrefetchOps` trait in Burn

- [ ] Create `burn-backend/src/backend/ops/prefetch.rs`:
  `PrefetchPrimitive<B>` struct + `PrefetchOps<B>` trait with default no-op
- [ ] Add `+ PrefetchOps<Self>` to `Backend` supertrait in `base.rs`
- [ ] Add `Tensor::prefetch(self, device) -> Self` in `burn-tensor`
- [ ] Verify all existing backends (`ndarray`, `wgpu`, `tch`, `candle`) compile
  with zero changes — default no-op satisfies the bound
- [ ] (Optional) Open upstream PR to `tracel-ai/burn`

#### P1-T2 — `WeightCache<T>`: unified expert + layer cache

- [ ] Define `CacheKey` trait (`ExpertKey { layer, expert }`, `LayerKey { layer }`)
- [ ] Implement `WeightCache<K>` with `LruCache`, async `get()`, fire-and-forget
  `prefetch(keys: &[K])`
- [ ] Implement async SSD I/O: mmap GGUF + `tokio::fs` pread for cache misses
- [ ] Implement LRU eviction: pop least-recently-used slot on capacity overflow
- [ ] Type aliases: `ExpertWeightCache = WeightCache<ExpertKey>`,
  `LayerWeightCache = WeightCache<LayerKey>`
- [ ] Unit test: 128-expert cache with N=32 slots, random access pattern,
  verify LRU eviction and correctness
- [ ] Unit test: prefetch fires before `get()` blocks (timing assertion)

#### P1-T3 — KV cache SSD offload (ping-pong)

- [ ] Implement `KvBuffer` (wgpu buffer wrapper, ping/pong pair)
- [ ] Implement `KvOffloadManager`: per-global-layer files, `swap_and_get`,
  `prefetch_next`, `writeback_async`
- [ ] Implement async write-back batching (accumulate 16 decode steps before
  flushing to disk)
- [ ] Unit test: 4 sequential global attention layers, verify KV correctness
  across swap cycles
- [ ] Unit test: concurrent prefetch + writeback with tokio, no data races

#### P1-T4 — GGUF model loader

- [ ] Implement `GgufIndex::open`: parse magic, version, metadata KV pairs,
  tensor info table
- [ ] Implement `GgufIndex::tensor_bytes(name)`: return mmap slice for any
  named tensor
- [ ] Implement `GgufIndex::expert_offsets(layer)`: return per-expert byte
  ranges for `WeightCache` index
- [ ] Implement `GgufIndex::layer_offsets()`: per-layer byte ranges
- [ ] Test: open a Gemma 4 9B GGUF, verify all tensor names resolve correctly

#### P1-T5 — wgpu backend core ops

- [ ] Implement `WgpuOffloadDevice` variant in `GgmlDevice` (or a parallel
  `WgpuOffloadBackend` struct) with `ssd_path`, `max_expert_slots` config
- [ ] Implement `PrefetchOps` for the wgpu offload backend: on prefetch call,
  fire `WeightCache::prefetch` for listed tensors
- [ ] Implement standard matmul, RMSNorm, embedding, softmax via existing
  `burn-wgpu` ops (no new kernels needed for non-MoE ops)
- [ ] Implement hybrid attention dispatch: route layers to local (sliding
  window) or global (full context) attention based on GGUF metadata
- [ ] Implement `KvOffloadManager` integration: global layers use ping-pong
  SSD buffers; local layers use resident RAM buffers

#### P1-T6 — MoE routing kernel (WGSL)

- [ ] Write `shaders/moe_router.wgsl`: 128-thread workgroup, one thread per
  expert, shared-memory parallel reduction for softmax + top-8 selection
- [ ] Add subgroup-accelerated fast path (`Features::SUBGROUP`): use
  `subgroupMax` / `subgroupBallot` to eliminate 4 barrier steps
- [ ] Runtime fallback: if `Features::SUBGROUP` not advertised by adapter,
  use pure shared-memory path
- [ ] Expose as `MoeRouter` Burn module wrapping the wgpu compute dispatch
- [ ] Unit test: compare top-8 output against reference PyTorch `torch.topk`

#### P1-T7 — `MUL_MAT_ID` grouped GEMM kernel (WGSL)

- [ ] Write `shaders/mul_mat_id.wgsl`: tiled 16×16 GEMM with expert-index
  dispatch (port from ggml-vulkan reference in `ggml-vulkan.cpp`)
- [ ] Push constants: `M, N, K, expert_stride, n_tokens`
- [ ] Storage bindings: weight buffer (all experts contiguous), input buffer,
  expert-index buffer, output buffer
- [ ] Expose as `MulMatId` op in the backend's `ModuleOps`
- [ ] Unit test: 8-expert dispatch, compare output against 8 separate matmuls
- [ ] Benchmark: single `MUL_MAT_ID` vs 8 sequential matmuls on Intel iGPU

#### P1-T8 — Gemma 4 26B MoE model definition

- [ ] Define `Gemma4MoeConfig` (from GGUF metadata: n_experts, top_k,
  shared_expert, attention interleaving, context length)
- [ ] Define `Gemma4MoeLayer` Burn module: attention + `MoeLayer`
  (router + `ExpertWeightCache` + `MulMatId` + shared expert)
- [ ] Define `Gemma4MoeModel`: embedding + 30 layers + RMSNorm + lm_head
- [ ] Implement `GemmaRunner::decode_step` with explicit prefetch schedule:
  - After routing: `expert_cache.prefetch(top_k_indices)`
  - Before next global layer: `kv_offload.prefetch_next(next_global)`
- [ ] Test: instantiate from Gemma 4 9B MoE GGUF (smaller, faster iteration)
- [ ] Test: single forward pass output matches HuggingFace reference (greedy)

#### P1-T9 — End-to-end benchmark on Linux Intel iGPU

- [ ] Run Gemma 4 26B MoE Q4_K_M on Linux with 16 GB RAM
- [ ] Verify no OOM at 32K context (confirm memory budget from Section 4.2.5)
- [ ] Benchmark decode throughput at 4K, 16K, 32K context
- [ ] Profile: confirm SSD prefetch overlaps with GPU compute (tracing spans)
- [ ] Profile: measure expert cache hit rate (target >80% for typical text)
- [ ] Confirm `Features::SUBGROUP` is advertised on test Intel iGPU

**Milestone 1:** Gemma 4 26B MoE Q4_K_M running at >=1.5 tok/s at 32K
context on Linux 16 GB with Intel iGPU. Expert cache hit rate >80%.
All memory management components (`WeightCache`, `KvOffloadManager`,
`PrefetchOps`) validated and ready for Phase 2 reuse.

---

### Phase 2: ggml + Metal on macOS (Weeks 11-20)

**Goal:** Gemma 4 31B dense at Q3_K_M running on macOS Apple Silicon via
`burn-ggml`, with layer streaming and KV offload, targeting >=2 tok/s at
256K context.

All memory management components from Phase 1 (`WeightCache`, `KvOffloadManager`,
`GgufIndex`, `PrefetchOps`) are reused without modification.

#### P2-T1 — `ggml-sys` crate with bindgen

- [ ] Add llama.cpp as a git submodule (pin to specific SHA)
- [ ] Write `ggml-sys/build.rs`: cmake build (LLAMA_METAL=ON, static) +
  bindgen for `ggml.h`, `ggml-backend.h`, `ggml-alloc.h`
- [ ] Link Metal, MetalKit, Foundation, Accelerate frameworks
- [ ] Smoke test: call `ggml_init` / `ggml_free` / `ggml_backend_metal_init`
  from a Rust test
- [ ] Write thin C wrapper `ggml_wrapper.c` exposing a stable API surface
  to insulate `ggml-sys` from ggml header churn

#### P2-T2 — `burn-ggml` backend skeleton

- [ ] Implement `GgmlDevice` enum: `Cpu`, `Metal`,
  `MetalWithOffload { kv_cache_dir, max_layers_in_ram }`
- [ ] Implement `GgmlContext` with `ggml_init`, `ggml_backend_sched`
  (Metal primary + CPU fallback), optional `LayerWeightCache` + `KvOffloadManager`
- [ ] Implement `GgmlTensor` wrapper (`*mut ggml_tensor` + `Arc<GgmlContext>`)
- [ ] Implement `PrefetchOps` for `GgmlBackend`:
  - `Cpu` / `Metal`: early return (no-op)
  - `MetalWithOffload`: call `WeightCache::prefetch` → triggers pread + Metal DMA
- [ ] Implement core float ops: `float_matmul`, `float_add`, `float_softmax`,
  `embedding_forward`, `rms_norm_forward`, `linear_forward`
- [ ] Run Burn tensor test suite against `GgmlBackend::Cpu`; fix failures

#### P2-T3 — Quantized matmul (Q4_K_M, Q3_K_M)

- [ ] Implement `QTensorOps::quantize` (F32 → Q4_K_M, Q3_K_M via `ggml_quantize`)
- [ ] Implement `QTensorOps::dequantize`
- [ ] Implement `GgmlQuantizedTensor` wrapper
- [ ] Verify `ggml_mul_mat` with Q4_K weight + F32 input on Metal
- [ ] Benchmark: Q3_K_M matmul throughput on M-series vs F32 baseline

#### P2-T4 — Hybrid attention (local + global) on Metal

- [ ] Implement `build_local_attention` ggml graph (sliding window, 1024 tokens,
  standard RoPE)
- [ ] Implement `build_global_attention` ggml graph (full context, Proportional
  RoPE: base=1M, scale=0.125)
- [ ] Integrate `KvOffloadManager` for global layers: load from ping buffer,
  write back to pong → SSD
- [ ] Implement shared KV cache aliasing (last N layers reuse earlier K/V)
- [ ] Unit test: attention output vs PyTorch reference

#### P2-T5 — Gemma 4 31B model definition + layer streaming

- [ ] Define `Gemma4DenseConfig` + `Gemma4DenseModel` Burn modules
- [ ] Implement `GemmaRunner` with `LayerWeightCache` (N=4 slots):
  - `decode_step`: prefetch layer i+2 while layer i runs
  - KV prefetch: fire `kv_offload.prefetch_next` before each global layer
- [ ] Test: load Gemma 4 9B dense GGUF, run forward pass, verify output
- [ ] Test: layer streaming correctness (output identical with N=4 vs N=46)

#### P2-T6 — End-to-end benchmark on macOS Apple Silicon

- [ ] Run Gemma 4 31B Q3_K_M on MacBook Air M-series 16 GB
- [ ] Verify no OOM at 256K context
- [ ] Benchmark decode throughput at 4K, 32K, 128K, 256K context
- [ ] Benchmark prefill TTFT at 4K, 32K context
- [ ] Profile: verify compute/IO overlap via Metal GPU trace (Instruments)
- [ ] Compare against llama.cpp direct (ceiling benchmark)

**Milestone 2:** Gemma 4 31B Q3_K_M running at >=2 tok/s at 256K context
on MacBook Air M-series 16 GB. Performance within 85% of llama.cpp direct.
Full performance report published.

---

## 11. Key Risks & Mitigations

### Risk 1: ggml API Instability

**Risk:** llama.cpp evolves rapidly. Internal ggml APIs (`ggml_backend_i`,
`ggml_flash_attn_ext`, `ggml_backend_sched`) have changed multiple times
in 2023-2024. A major refactor could break `ggml-sys` bindings.

**Probability:** High (llama.cpp has 500+ commits/month).

**Mitigations:**
- Pin llama.cpp to a specific git SHA as a submodule. Only upgrade
  intentionally with a dedicated upgrade task.
- Use a thin C wrapper (`ggml_wrapper.c`) that exposes a stable C API to
  Rust, shielding `ggml-sys` from ggml header churn. The wrapper absorbs
  signature changes.
- Minimize the surface of ggml symbols exposed in `ggml-sys`. Only bindgen
  what is actively used.
- Write integration tests that run on every llama.cpp version upgrade and
  catch regressions early.

### Risk 2: FFI Safety and Memory Corruption

**Risk:** Raw pointer manipulation in `GgmlTensor` and `GgmlContext` is
inherently unsafe. Use-after-free, double-free, or data races are possible
if the `Arc<GgmlContext>` invariant is violated (e.g., a tensor outliving
its context due to clones in unexpected places).

**Probability:** Medium. Easy to get right for simple cases; tricky under
async/multi-threaded Burn usage patterns.

**Mitigations:**
- Encapsulate all unsafe in a small, well-reviewed module (`context.rs`,
  `tensor.rs`) with clear safety invariants documented via `# Safety` comments.
- Use `Arc<GgmlContext>` as the single lifetime anchor; never store raw
  `*mut ggml_context` outside this type.
- Run tests under `valgrind` (Linux) and `AddressSanitizer` / `ThreadSanitizer`
  during CI.
- Add `#[cfg(test)] mod safety_tests` with adversarial clone/drop patterns.
- Consider a `GgmlContextGuard` RAII type for graph execution to ensure
  buffers are not freed mid-computation.

### Risk 3: SSD Write Endurance

**Risk:** Each token generated at 256K context triggers a small (~8 KB per
global layer) write to the KV cache SSD file. At 2 tok/s over an 8-hour
session:
  2 tok/s * 28800 s * 8 global_layers * 8 KB = ~3.6 GB of writes/session

Apple Silicon SSDs are rated for ~1500 TB written (TBW) for a 1 TB drive.
At 3.6 GB/session, 100 sessions/day = 360 GB/day = ~4400 days to reach TBW.

**Probability:** Low (endurance is not a practical concern at this usage rate).

**Mitigations:**
- Batch KV writes: accumulate 16-32 decode tokens before flushing to disk.
  Reduces write frequency by 16-32x.
- Use `mmap` with `msync(MS_ASYNC)` to let the OS decide when to flush,
  coalescing multiple writes into fewer I/O operations.
- Document the endurance math in the README so users can make informed
  decisions.

### Risk 4: SSD Prefetch Latency Spikes

**Risk:** NVMe SSD latency is not uniform. GC pauses, thermal throttling,
or OS scheduler jitter can cause individual reads to take 10-100ms instead
of the expected 1-5ms, causing the GPU to stall waiting for data.

**Probability:** Medium (thermal throttling is observed on MacBook Air under
sustained load; the fanless design limits sustained SSD bandwidth).

**Mitigations:**
- Increase prefetch depth N from 4 to 8 when a latency spike is detected,
  giving more buffer against SSD jitter.
- Implement adaptive N: start with N=4, monitor whether `get_layer` ever
  blocks (prefetch not ready), increase N if blocking occurs.
- Add telemetry: record actual prefetch completion times, expose as metrics.
- Accept a modest decode speed reduction (1-2 tok/s) is more realistic than
  2 tok/s under sustained thermal load on fanless hardware.

### Risk 5: Gemma 4 Architecture Deviations

**Risk:** Gemma 4's exact architecture details (number of global layers,
shared KV cache sharing pattern, Proportional RoPE parameters) may differ
from Gemma 3 assumptions. If the GGUF metadata encoding of hybrid attention
differs from what is assumed here, the routing logic will be wrong.

**Probability:** Medium (Gemma 4 is relatively new; community GGUF files
may encode hybrid attention in a non-standard way initially).

**Mitigations:**
- Parse all relevant GGUF metadata keys for attention type per layer
  (`gemma4.attention.sliding_window`, `gemma4.attention.is_global`, etc.).
- Fall back to hard-coded interleaving pattern (1 global per 6 layers) if
  metadata key is absent.
- Add a `--debug-attention-types` flag that prints the detected attention
  type for each layer at model load time.

### Risk 6: Burn API Changes (v0.21 pre-release)

**Risk:** Burn is at v0.21.0-pre.2. The `Backend` trait, `ModuleOps`, and
`QTensorOps` interfaces may change before stable release.

**Probability:** Medium (pre-release software changes frequently).

**Mitigations:**
- Pin Burn to the exact commit SHA used during development.
- Track the Burn changelog / GitHub issues for breaking changes.
- Isolate Burn trait implementations behind a `burn_compat` module that
  can be updated without touching core tensor logic.

### Risk 7: Expert Cache Miss Rate on Diverse Inputs

**Risk:** The `ExpertWeightCache` LRU with N=32 slots assumes that token
distributions cluster into a recurring subset of hot experts. For highly
diverse inputs (code, multilingual, structured data), the effective working
set of experts may exceed 32, causing frequent cache misses and SSD reads
that can't be hidden by prefetch.

**Probability:** Medium (depends heavily on workload; typical chat is
relatively clustered; code/multilingual may be more diverse).

**Mitigations:**
- Make N tunable at runtime (`--expert-cache-slots N`); default 32, allow
  up to 64 (using ~900 MB RAM).
- Instrument cache hit rate per session; surface as a warning if hit rate
  drops below 60%.
- For cache miss scenarios, the fallback is synchronous SSD load (latency
  spike of ~22 ms per miss); this is acceptable for occasional misses but
  degrades throughput significantly if sustained.
- Consider a two-level cache: L1 = 8 slots pinned to the current layer's
  top-8 selection; L2 = 24 LRU slots shared across layers.

### Risk 8: `MUL_MAT_ID` WGSL Kernel Correctness on Intel iGPU

**Risk:** The tiled grouped GEMM kernel is the most complex WGSL shader in
the project. Correctness bugs (off-by-one in tile indexing, incorrect expert
stride calculation) produce silently wrong outputs that are hard to detect
without reference comparison.

**Probability:** Medium (complex kernel, no existing wgpu reference).

**Mitigations:**
- Develop against the ggml-vulkan SPIR-V reference (`ggml-vulkan.cpp`);
  port tile-by-tile with direct comparison at each step.
- Unit test: for each batch size (1, 8, 16, 32 tokens) and each expert
  count (1, 8, 128), compare `MUL_MAT_ID` output against 8 separate
  `float_matmul` calls with the same weights.
- Run under `vulkan validation layers` during development to catch
  out-of-bounds buffer accesses.
- Add a CPU reference path as a correctness oracle for CI.

---

## 12. Performance Targets

### 12.1 Target Hardware

Two hardware targets, one per phase:

**Phase 1:** Linux desktop/laptop with Intel iGPU (e.g. Intel Iris Xe),
16 GB RAM, 256 GB+ NVMe SSD, Mesa 22+ / Vulkan 1.3.

**Phase 2:** MacBook Air with Apple M3, 16 GB unified memory,
1 TB NVMe SSD, macOS Sequoia.

### 12.2 Phase 1 Targets — Gemma 4 26B MoE, Linux Intel iGPU

#### Decode throughput

| Quantization | Context | Target tok/s | Notes                              |
|--------------|---------|--------------|-------------------------------------|
| Q4_K_M       | 4K      | >= 2 tok/s   | Expert cache warm, short context    |
| Q4_K_M       | 16K     | >= 1.5 tok/s | Some KV offload for global layers   |
| Q4_K_M       | 32K     | >= 1 tok/s   | Full KV offload active              |

#### Expert cache

| Metric                        | Target  |
|-------------------------------|---------|
| Cache hit rate (typical chat) | >= 80%  |
| Cache miss penalty (cold SSD) | <= 25 ms per expert |
| Hot cache RAM footprint       | <= 500 MB (N=32 slots) |

#### Memory footprint

| State                          | RAM budget | Target  |
|--------------------------------|------------|---------|
| Model load (non-expert weights)| <= 2 GB    | ~1.5 GB |
| During decode (32K ctx)        | <= 14 GB   | ~7.4 GB |

### 12.3 Phase 2 Targets — Gemma 4 31B Dense, macOS Apple Silicon

#### Decode throughput

| Quantization | Context | Target tok/s | Notes                         |
|--------------|---------|--------------|-------------------------------|
| Q3_K_M       | 4K      | >= 3 tok/s   | Baseline, weights fit in RAM  |
| Q3_K_M       | 32K     | >= 2.5 tok/s | KV offload for global layers  |
| Q3_K_M       | 128K    | >= 2 tok/s   | KV + layer weight offload     |
| Q3_K_M       | 256K    | >= 1.5 tok/s | Full offload mode             |

#### Prefill (time-to-first-token)

| Quantization | Context | Target TTFT |
|--------------|---------|-------------|
| Q3_K_M       | 4K      | <= 5 s      |
| Q3_K_M       | 32K     | <= 30 s     |
| Q3_K_M       | 128K    | <= 120 s    |

Note: 256K context prefill takes several minutes (O(n²) global attention).
Accepted for initial implementation; chunked prefill is future work.

#### Memory footprint

| State                    | RAM budget | Target  |
|--------------------------|------------|---------|
| Model load (streaming)   | <= 2 GB    | ~1.5 GB |
| During decode (256K ctx) | <= 14 GB   | ~7.0 GB |
| Peak (prefill 256K)      | <= 15.5 GB | ~8 GB   |

### 12.4 SSD I/O Targets (both phases)

| Operation                        | Target latency | Target bandwidth |
|----------------------------------|----------------|------------------|
| Single expert prefetch (~14 MB)  | <= 25 ms       | >= 3 GB/s        |
| Single layer prefetch (~315 MB)  | <= 100 ms      | >= 3 GB/s        |
| Global KV prefetch (INT8, 1 GB)  | <= 400 ms      | >= 3 GB/s        |
| KV writeback per decode step     | < 5 ms         | (async, batched) |

### 12.5 Comparison Baseline

For Phase 2, llama.cpp direct is the ceiling benchmark:
- Graph construction overhead: <1 ms per layer
- FFI call overhead: negligible (batched, not per-element)
- Memory copy: ~0 for quantized weight path

Target: `burn-ggml` achieves **>=85% of llama.cpp throughput** for identical
model, quantization, and context.

For Phase 1, no direct comparison baseline exists (no wgpu MoE inference
stack to compare against). The target is absolute throughput only.

---

## 13. Correctness Test Cases

These are end-to-end black-box tests that must pass on every model variant
and every backend (wgpu Phase 1, ggml Phase 2) before a milestone is declared
complete. They validate that the full stack — weight loading, attention,
expert routing, KV cache, sampling — produces semantically correct output,
not just that individual ops are numerically correct.

All tests use **greedy decoding** (temperature=0, top-p=1, top-k=1) to make
outputs deterministic. Tests are run at short context (<=512 tokens) so they
complete quickly and do not exercise the offload path; separate offload
correctness tests are listed in Section 13.3.

### 13.1 Factual Recall

**Test ID:** `correctness::factual::capital_of_china`

```
Prompt:   "What is the capital of China?"
Max tokens: 64
Assertion: output.to_lowercase().contains("beijing")
```

Rationale: a single unambiguous factual question with a one-word answer. If
the model produces garbled output, repeats the prompt, or hallucinates a
different city, the assertion fails. This catches weight-loading bugs,
embedding errors, and broken attention that produce coherent-looking but
wrong text.

```rust
#[test]
fn test_capital_of_china() {
    let model = load_test_model();  // small variant, e.g. Gemma 4 E2B or 4B
    let output = model.generate("What is the capital of China?", GenerateConfig {
        max_new_tokens: 64,
        temperature: 0.0,
        ..Default::default()
    });
    assert!(
        output.to_lowercase().contains("beijing"),
        "Expected output to contain 'beijing', got: {:?}", output
    );
}
```

### 13.2 Instruction Following

**Test ID:** `correctness::instruction::arithmetic_strict`

```
Prompt:   "What is 1+1? Answer with only a number."
Max tokens: 8
Assertion: output.trim() == "2"
```

Rationale: tests that the model follows an explicit formatting constraint.
The `trim()` strips a single trailing newline or space; the assertion is
otherwise strict. This catches tokenizer bugs (wrong token IDs), broken
sampling (outputting multiple tokens when one suffices), and instruction
tuning regressions in the loaded weights.

The 8-token budget is intentional: if the model produces more than a few
tokens it is not following instructions, and the test should fail rather
than pass on a substring match.

```rust
#[test]
fn test_arithmetic_strict() {
    let model = load_test_model();
    let output = model.generate(
        "What is 1+1? Answer with only a number.",
        GenerateConfig {
            max_new_tokens: 8,
            temperature: 0.0,
            ..Default::default()
        },
    );
    assert_eq!(
        output.trim(), "2",
        "Expected exactly '2', got: {:?}", output
    );
}
```

### 13.3 Offload Correctness Tests

These tests verify that the offload path (expert cache, KV ping-pong, layer
streaming) produces **identical output** to a non-offload reference run on
the same model. They use the same two prompts above as inputs.

**Test ID:** `correctness::offload::expert_cache_matches_reference`

```
Setup:
  reference = run with ExpertWeightCache disabled (all experts resident)
  offload   = run with ExpertWeightCache enabled, N=4 slots (forces eviction)
Assertion: reference.output == offload.output  (token-for-token identical)
```

**Test ID:** `correctness::offload::kv_pingpong_matches_reference`

```
Setup:
  reference = run with KvOffloadManager disabled (all KV resident in RAM)
  offload   = run with KvOffloadManager enabled (global layers use SSD)
Assertion: reference.output == offload.output  (token-for-token identical)
```

**Test ID:** `correctness::offload::layer_streaming_matches_reference`

```
Setup:
  reference = run with LayerWeightCache disabled (all layers resident)
  streaming = run with LayerWeightCache enabled, N=2 slots (forces eviction)
Assertion: reference.output == streaming.output  (token-for-token identical)
```

These three tests are the most important correctness signal. If offload
introduces any numerical difference (wrong buffer swap, stale cache entry,
partial write to SSD), it will show up as a diverging token within a few
steps of generation.

### 13.4 Test Matrix

Each test must pass on all applicable configurations:

| Test ID                              | Phase 1 (wgpu) | Phase 2 (ggml) | Model           |
|--------------------------------------|----------------|----------------|-----------------|
| `factual::capital_of_china`          | ✓              | ✓              | 26B MoE / 31B   |
| `instruction::arithmetic_strict`     | ✓              | ✓              | 26B MoE / 31B   |
| `offload::expert_cache_matches`      | ✓              | —              | 26B MoE         |
| `offload::kv_pingpong_matches`       | ✓              | ✓              | 26B MoE / 31B   |
| `offload::layer_streaming_matches`   | —              | ✓              | 31B dense       |

---

## Appendix A: Cargo.toml Files

### A.1 Workspace `Cargo.toml`

```toml
[workspace]
members = [
    "ggml-sys",
    "burn-ggml",
]
resolver = "2"

[workspace.dependencies]
burn       = { path = "../burn", features = ["std"] }
ggml-sys   = { path = "ggml-sys" }
tokio      = { version = "1", features = ["full"] }
memmap2    = "0.9"
tracing    = "0.1"
```

### A.2 `ggml-sys/Cargo.toml`

```toml
[package]
name    = "ggml-sys"
version = "0.1.0"
edition = "2021"
links   = "ggml"

[build-dependencies]
cmake   = "0.1"
bindgen = { version = "0.69", features = ["runtime"] }
```

### A.3 `burn-ggml/Cargo.toml`

```toml
[package]
name    = "burn-ggml"
version = "0.1.0"
edition = "2021"

[dependencies]
burn     = { workspace = true }
ggml-sys = { workspace = true }
tokio    = { workspace = true }
memmap2  = { workspace = true }
tracing  = { workspace = true }

[dev-dependencies]
burn = { workspace = true, features = ["test"] }
```

## Appendix B: Key ggml C Types Reference

```c
// From ggml.h -- key types used in this design

enum ggml_type {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    // ...
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_IQ2_XXS = 18,
    GGML_TYPE_IQ3_S   = 24,
    // ...
};

struct ggml_tensor {
    enum ggml_type type;
    int64_t ne[GGML_MAX_DIMS];  // number of elements per dim (ne[0]=innermost)
    size_t  nb[GGML_MAX_DIMS];  // byte stride per dim
    int     n_dims;
    char    name[GGML_MAX_NAME];
    void  * data;               // pointer to tensor data (or NULL if deferred)
    // ... (src, op, grad fields omitted)
};

// From ggml-backend.h -- backend interface
struct ggml_backend_i {
    const char * (*get_name)(ggml_backend_t backend);
    void         (*free)(ggml_backend_t backend);
    // Compute a complete graph
    enum ggml_status (*graph_compute)(ggml_backend_t, struct ggml_cgraph *);
    // Optional: asynchronous compute, events, graph optimization
    enum ggml_status (*graph_compute_async)(ggml_backend_t, struct ggml_cgraph *);
    ggml_backend_event_t (*event_new)(ggml_backend_t);
    void (*event_synchronize)(ggml_backend_t, ggml_backend_event_t);
};
```

## Appendix C: Burn Backend Trait (v0.21 Reference)

```rust
// From burn/crates/burn-tensor/src/tensor/backend/base.rs (simplified)
pub trait Backend:
    FloatTensorOps<Self>
    + BoolTensorOps<Self>
    + IntTensorOps<Self>
    + ModuleOps<Self>
    + ActivationOps<Self>
    + QTensorOps<Self>
    + Clone + Default + Send + Sync + 'static
{
    type Device: DeviceOps;
    type FloatTensorPrimitive: TensorMetadata + 'static;
    type FloatElem: Element;
    type IntTensorPrimitive: TensorMetadata + 'static;
    type IntElem: Element;
    type BoolTensorPrimitive: TensorMetadata + 'static;
    type QuantizedTensorPrimitive: TensorMetadata + QTensorPrimitive + 'static;

    fn name(device: &Self::Device) -> String;
    fn seed(device: &Self::Device, seed: u64);
    fn sync(_device: &Self::Device) -> Result<(), ExecutionError> { Ok(()) }
    fn ad_enabled() -> bool { false }
}
```

---

*End of design document. Total estimated implementation: 20 weeks, 1 engineer.*
