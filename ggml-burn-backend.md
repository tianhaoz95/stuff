# `burn-ggml`: A GGML Backend for Burn
## Design Document

**Status:** Draft  
**Date:** 2026-04-04  
**Author:** (working document)  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Motivation & Use Case](#2-motivation--use-case)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Gemma 4 31B Memory Analysis](#4-gemma-4-31b-memory-analysis)
5. [Core Components](#5-core-components)
6. [ggml Graph Construction for Attention](#6-ggml-graph-construction-for-attention)
7. [Burn Integration Points](#7-burn-integration-points)
8. [GGUF Model Loading](#8-gguf-model-loading)
9. [Implementation Plan](#9-implementation-plan)
10. [Key Risks & Mitigations](#10-key-risks--mitigations)
11. [Performance Targets](#11-performance-targets)

---

## 1. Executive Summary

`burn-ggml` is a Burn backend crate that delegates tensor operations to ggml,
the C tensor library at the heart of llama.cpp. The goal is to give Rust
applications written against the Burn ML framework access to ggml's battle-tested
quantization kernels, Apple Metal GPU acceleration, and its ecosystem of GGUF
model files — without leaving the Burn API surface.

The primary motivating use case is running **Gemma 4 31B** at **256K context**
on a **MacBook Air with 16 GB of unified memory**. This is a hard problem: the
model weights alone occupy ~14.5 GB at Q3_K_M quantization, and the KV cache at
256K context exceeds 24 GB. Neither fits in 16 GB simultaneously. The solution is
a three-part memory management strategy built into this backend:

1. **Layer weight streaming** — keep only 3-4 transformer layers in RAM at any
   time; stream remaining weights from NVMe SSD on demand.
2. **KV cache SSD offload with ping-pong buffering** — global attention layers
   (those that attend over the full 256K context) store their KV cache on SSD;
   a double-buffer scheme hides I/O latency behind GPU compute.
3. **Compute-prefetch overlap** — async SSD reads for the next layer's weights
   and KV data are pipelined with the Metal GPU executing the current layer.

Together these techniques allow a 31B-parameter model with a 256K-token context
window to run on commodity Apple Silicon hardware, targeting >=2 tokens/second
generation throughput.

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

## 4. Gemma 4 31B Memory Analysis

### 4.1 Model Architecture Facts

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

### 4.2 Weight Memory by Quantization

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

### 4.3 KV Cache Size Calculation

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

### 4.4 Shared KV Cache Optimization

Gemma 4 implements a shared KV cache where certain later layers reuse K/V
projection outputs from designated earlier layers rather than computing their
own. This reduces the number of independent KV caches that must be maintained.
The exact sharing pattern is encoded in the GGUF metadata.

In practical terms, for the offload strategy:
- Shared KV layers do not need their own SSD storage slot — they reference
  the buffer of the layer they share with.
- This can reduce effective global KV storage by 20-30% depending on the
  specific sharing configuration in Gemma 4.

### 4.5 Memory Budget at Inference Time (Q3_K_M, 256K context)

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

## 6. ggml Graph Construction for Attention

ggml uses a **deferred computation graph** model: you build an expression
graph of tensor operations, then call `ggml_graph_compute` once to execute
the entire graph. This maps well to how Metal command buffers work.

### 6.1 Key ggml Operations

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

### 6.2 Local Sliding-Window Attention Graph

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

### 6.3 Global Full-Context Attention Graph

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

### 6.4 Graph Compute Execution

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

### 6.5 MLP Block

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

## 7. Burn Integration Points

### 7.1 Module-Level Operations (ModuleOps)

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

### 7.2 Quantized Tensor Support (QTensorOps)

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

### 7.3 Device Management

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

---

## 8. GGUF Model Loading

GGUF (Generic GPU Unified Format) is the self-describing model file format
used by llama.cpp. It encodes tensor names, shapes, quantization types, and
all model hyperparameters in a single binary file.

### 8.1 GGUF File Structure

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

### 8.2 Loading Strategy

The loader uses **lazy, memory-mapped loading**:

1. Parse the GGUF header and tensor metadata into a `GgufIndex` struct.
   This is cheap (metadata is small, typically <1 MB).
2. `mmap` the entire GGUF file into virtual address space.
   No data is actually read from disk until accessed.
3. When a layer's weights are needed (during `LayerWeightCache::get_layer`),
   the relevant byte range is read from the mmap'd region.
4. The data is copied into a ggml-managed `ggml_backend_buffer` and
   wrapped as `GgmlTensor` / `GgmlQuantizedTensor`.

### 8.3 Gemma 4 Tensor Name Mapping

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

### 8.4 GgufLoader Implementation

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

### 8.5 Lazy Layer Loading

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

## 9. Implementation Plan

The implementation is divided into four phases. Each phase produces working,
testable code before the next begins.

### Phase 1: Foundation (Weeks 1-4)

Goal: A working `burn-ggml` backend that passes the Burn CPU test suite
using ggml's CPU backend.

#### T1.1 — Set up `ggml-sys` crate with bindgen

- [ ] Add llama.cpp as a git submodule under `/idea-01/llama.cpp`
- [ ] Create `ggml-sys/Cargo.toml` with `cmake` and `bindgen` build deps
- [ ] Write `ggml-sys/build.rs`: cmake build + bindgen for `ggml.h`,
  `ggml-backend.h`, `ggml-alloc.h`
- [ ] Verify: `cargo build -p ggml-sys` compiles without errors
- [ ] Smoke test: call `ggml_init` / `ggml_free` from a Rust test

#### T1.2 — Basic `burn-ggml` backend struct + device

- [ ] Create `burn-ggml/Cargo.toml` with burn, ggml-sys dependencies
- [ ] Implement `GgmlDevice` enum (Cpu, Metal, MetalWithOffload)
- [ ] Implement `GgmlContext` struct with `ggml_init` / `ggml_free` lifecycle
- [ ] Implement `GgmlBackend` struct satisfying `Backend` trait skeleton
  (panicking impls for all ops initially)
- [ ] Implement `GgmlTensor` wrapper with `TensorMetadata`
- [ ] Verify: `cargo build -p burn-ggml` compiles

#### T1.3 — Core float ops via ggml CPU

Priority ops (most used in LLM inference):

- [ ] `float_add` / `float_mul` / `float_sub` via `ggml_add` / `ggml_mul` / `ggml_sub`
- [ ] `float_matmul` via `ggml_mul_mat`
- [ ] `float_reshape` / `float_transpose` via `ggml_reshape_*` / `ggml_transpose`
- [ ] `float_slice` via `ggml_view_*`
- [ ] `float_softmax` via `ggml_soft_max`
- [ ] `float_from_data` / `float_to_data` (tensor I/O)
- [ ] `embedding_forward` via `ggml_get_rows`
- [ ] `linear_forward` via `ggml_mul_mat` + `ggml_add`
- [ ] `rms_norm_forward` via `ggml_rms_norm` + `ggml_mul`

#### T1.4 — Run Burn backend test suite on CPU

- [ ] Run `burn/crates/burn-tensor/src/tests/` against GgmlBackend
- [ ] Fix failures from shape/stride mismatches (ggml uses column-major internally)
- [ ] Achieve >95% pass rate on float tensor tests
- [ ] Benchmark: compare ggml CPU matmul vs NdArray backend on 4096x4096 f32

**Milestone 1:** `GgmlBackend::Cpu` passes the Burn tensor test suite.

---

### Phase 2: Metal Acceleration (Weeks 5-8)

Goal: All core ops run on Apple Metal GPU, quantized weights work.

#### T2.1 — Metal backend integration

- [ ] Enable LLAMA_METAL cmake flag in ggml-sys build.rs
- [ ] Link Metal, MetalKit, Foundation, Accelerate frameworks
- [ ] Verify `ggml_backend_metal_init()` succeeds in a test binary
- [ ] Implement `GgmlDevice::Metal` path in `GgmlContext::new`
- [ ] Set up `ggml_backend_sched` with Metal primary + CPU fallback
- [ ] Verify matmul runs on GPU via Metal profiler (Instruments)

#### T2.2 — Attention op implementation (local + global)

- [ ] Implement `build_local_attention` ggml graph function (C-side wrapper)
- [ ] Implement `build_global_attention` ggml graph function
- [ ] Expose both through `module_ops.rs` as `attention_forward`
- [ ] Implement sliding-window causal mask construction
- [ ] Implement Proportional RoPE (`ggml_rope_ext` with base=1M, scale=0.125)
- [ ] Unit test: compare attention output against reference PyTorch impl

#### T2.3 — Quantized tensor ops (Q4, Q3 types)

- [ ] Implement `QTensorOps::quantize` (F32 -> Q4_K_M, Q3_K_M via ggml)
- [ ] Implement `QTensorOps::dequantize` (quantized -> F32)
- [ ] Implement `GgmlQuantizedTensor` wrapper
- [ ] Verify: `ggml_mul_mat` with Q4_K weight and F32 input works on Metal
- [ ] Benchmark: quantized matmul throughput vs F32 on M2

#### T2.4 — GGUF model loader

- [ ] Implement `GgufIndex::open` (header + metadata parsing)
- [ ] Implement `GgufIndex::tensor_data_bytes` (mmap slice)
- [ ] Implement `GgufIndex::load_tensor` (into ggml backend buffer)
- [ ] Test: load a small GGUF (e.g. Gemma 2B) and run a forward pass
- [ ] Verify tensor names match expected Gemma naming convention

**Milestone 2:** Load a Gemma model from GGUF and run Metal-accelerated
inference for short sequences (<=4K tokens).

---

### Phase 3: Memory Management (Weeks 9-14)

Goal: Enable inference of models that don't fit in RAM, with KV offload.

#### T3.1 — Layer weight streaming (LayerWeightCache)

- [ ] Implement `LayerWeightCache` with N-slot circular buffer
- [ ] Implement `get_layer` (blocking, with wait on prefetch completion)
- [ ] Implement `prefetch` (async tokio task + pread)
- [ ] Implement LRU eviction policy
- [ ] Implement `GgufIndex` integration (per-layer byte offsets)
- [ ] Test: load 8 layers cycling through a 4-slot cache, verify correctness
- [ ] Test: verify prefetch completes before `get_layer` returns (timing test)

#### T3.2 — KV cache SSD offload (KvOffloadManager)

- [ ] Implement `KvOffloadManager` with per-global-layer file handles
- [ ] Implement `swap_and_get` (ping/pong swap + return active buffer)
- [ ] Implement `prefetch_next` (async load into pong buffer)
- [ ] Implement `writeback_async` (async write of updated KV to SSD)
- [ ] Create per-layer KV files on first use, sized for max_context tokens
- [ ] Test: 4 sequential global attention layers, verify KV correctness

#### T3.3 — Ping-pong buffer implementation (PingPongBuffer)

- [ ] Implement `KvBuffer` struct (ggml_backend_buffer wrapper)
- [ ] Implement `swap_buffers` (atomic swap of ping/pong Arc pointers)
- [ ] Implement thread-safe buffer handoff between GPU and prefetch task
- [ ] Test: concurrent read/write with tokio, verify no data races

#### T3.4 — Async prefetch pipeline

- [ ] Implement `prefetch_pipeline_task` (receives layer completion events)
- [ ] Wire up Metal command buffer completion -> tokio channel notification
- [ ] Integrate with GgmlContext: spawn pipeline task at context creation
- [ ] Test: full forward pass with N=4 layer streaming, measure overlap
- [ ] Instrument with tracing spans to verify compute/IO overlap

**Milestone 3:** Run Gemma 9B (or similar) with more weight than fits in
RAM, using layer streaming. Verify output matches non-streaming mode.

---

### Phase 4: Gemma 4 31B Integration (Weeks 15-20)

Goal: End-to-end Gemma 4 31B inference at 256K context on 16GB MacBook Air.

#### T4.1 — Gemma 4 model definition in Burn

- [ ] Define `Gemma4Config` struct (from GGUF metadata)
- [ ] Define `Gemma4TransformerLayer` Burn module
- [ ] Define `Gemma4Model` Burn module with embedding + layers + norm + lm_head
- [ ] Implement weight loading from GGUF into Burn modules
- [ ] Test: instantiate Gemma4Model from a Gemma 4 9B GGUF

#### T4.2 — Hybrid attention (local + global) implementation

- [ ] Detect layer type (local vs global) from GGUF metadata or config
- [ ] Route each layer to the appropriate attention implementation
- [ ] Pass `is_global_attention` flag through the ggml graph builder
- [ ] Test: verify output of hybrid-attention forward pass matches reference

#### T4.3 — Shared KV cache support

- [ ] Parse shared KV layer mapping from GGUF metadata
- [ ] Implement KV cache aliasing: shared layers point to the same buffer
- [ ] Verify memory savings (number of distinct KV buffers reduced)

#### T4.4 — End-to-end inference at 256K context

- [ ] Configure `GgmlDevice::MetalWithOffload` for Gemma 4 31B at Q3_K_M
- [ ] Run prefill for 32K token prompt, verify no OOM
- [ ] Run decode for 100 tokens at 256K context, verify coherent output
- [ ] Tune max_layers_in_ram (N) for best throughput on M-series hardware
- [ ] Tune KV quantization (FP16 vs INT8 tradeoff)

#### T4.5 — Performance benchmarking

- [ ] Benchmark decode throughput (tok/s) at 4K, 32K, 128K, 256K context
- [ ] Benchmark prefill time for 32K and 128K token prompts
- [ ] Profile SSD utilization: confirm compute/IO overlap is happening
- [ ] Compare against llama.cpp direct (ceiling benchmark)
- [ ] Document: memory footprint breakdown at 256K context

**Milestone 4:** Gemma 4 31B Q3_K_M running at >=2 tok/s with 256K context
on a 16GB MacBook Air M-series. Performance report published.

---

## 10. Key Risks & Mitigations

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

---

## 11. Performance Targets

### 11.1 Target Hardware

All targets are measured on a **MacBook Air with Apple M3, 16 GB unified
memory, 1 TB NVMe SSD** running macOS Sequoia.

### 11.2 Decode Throughput Targets

| Model           | Quantization | Context  | Target tok/s | Notes                    |
|-----------------|--------------|----------|--------------|--------------------------|
| Gemma 4 31B     | Q3_K_M       | 4K       | >= 3 tok/s   | Baseline, all in RAM     |
| Gemma 4 31B     | Q3_K_M       | 32K      | >= 2.5 tok/s | Some KV offload          |
| Gemma 4 31B     | Q3_K_M       | 128K     | >= 2 tok/s   | KV + weight offload      |
| Gemma 4 31B     | Q3_K_M       | 256K     | >= 1.5 tok/s | Full offload mode        |
| Gemma 4 31B     | IQ2_XXS      | 256K     | >= 2.5 tok/s | Aggressive quant         |

### 11.3 Prefill (Time-to-First-Token) Targets

| Model       | Quantization | Context  | Target TTFT |
|-------------|--------------|----------|-------------|
| Gemma 4 31B | Q3_K_M       | 4K       | <= 5 s      |
| Gemma 4 31B | Q3_K_M       | 32K      | <= 30 s     |
| Gemma 4 31B | Q3_K_M       | 128K     | <= 120 s    |

Note: 256K context prefill is expected to take several minutes due to the
O(n^2) attention cost in global layers. This is accepted for the initial
implementation; optimizations (sparse attention, chunked prefill) are
future work.

### 11.4 Memory Footprint Targets

| Phase                     | RAM budget | Actual (target) |
|---------------------------|------------|-----------------|
| Model load (streaming)    | <= 2 GB    | ~1.5 GB         |
| During decode (256K ctx)  | <= 14 GB   | ~7 GB           |
| Peak (prefill 256K)       | <= 15.5 GB | ~8 GB           |

### 11.5 SSD I/O Targets

| Operation                     | Target latency | Target bandwidth  |
|-------------------------------|----------------|-------------------|
| Single layer weight prefetch  | <= 100 ms      | >= 3 GB/s         |
| Global KV prefetch (INT8)     | <= 400 ms      | >= 3 GB/s         |
| KV writeback per decode step  | < 5 ms         | (async, batched)  |

### 11.6 Comparison Baseline

llama.cpp (direct, not through Burn) is the ceiling benchmark. The expected
overhead of the Burn/Rust layer is:
- Graph construction overhead: <1 ms per layer (Rust ggml graph building)
- FFI call overhead: negligible (batch calls, not per-element)
- Memory copy overhead: ~0 for quantized weight path (ggml manages buffers)

Target: `burn-ggml` achieves **>=85% of llama.cpp throughput** for identical
model, quantization, and context configuration.

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
