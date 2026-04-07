# burn-inference: Rust Inference Engine Design

**Status:** Draft v0.1  
**Date:** 2026-04-06  
**Audience:** Core engine contributors

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Data Structures](#3-core-data-structures)
4. [Scheduler Design](#4-scheduler-design)
5. [Radix Cache](#5-radix-cache)
6. [KV Cache Management](#6-kv-cache-management)
7. [Prefill-Decode Disaggregation](#7-prefill-decode-disaggregation)
8. [FlashMoE Integration](#8-flashmoe-integration)
9. [Compute-Data Movement Overlap](#9-compute-data-movement-overlap)
10. [API Layer](#10-api-layer)
11. [Tokenizer Service](#11-tokenizer-service)
12. [Implementation Plan](#12-implementation-plan)
13. [Correctness Tests](#13-correctness-tests)
14. [Performance Targets](#14-performance-targets)
15. [Key Risks](#15-key-risks)

---

## 1. Executive Summary

`burn-inference` is a Rust-native LLM inference engine designed for edge and consumer hardware. It is the serving layer that sits on top of the `burn-ggml` (macOS) and `burn-wgpu` (Linux) compute backends described in the sibling design document `ggml-burn-backend.md`. Think of it as a Rust reimplementation of the core scheduling and memory management logic from SGLang or vLLM, but purpose-built for hardware that does not have 80 GB of HBM.

### What it is not

This is not a cloud inference stack. We do not target A100/H100 clusters, we do not target Kubernetes, and we do not assume NVLink or high-speed interconnects. Every design decision is calibrated for:

- **Platform A (macOS):** Apple Silicon MacBook Air M3/M4, 16–32 GB unified memory, Metal GPU, Neural Engine. Primary model: a 31B dense model (e.g., Qwen3-32B-Instruct in Q4 quantization, ~18 GB).
- **Platform B (Linux):** Intel Meteor Lake / Arrow Lake laptop, 16–32 GB DDR5, Intel Arc iGPU (8 GB shared VRAM), Vulkan via `burn-wgpu`. Primary model: a 26B MoE model (e.g., Qwen2.5-VL-7B-Instruct or a 26B sparse MoE where only ~6B parameters are active per token).

### Relationship to burn-ggml backend

`burn-inference` is the *engine* layer. It owns:
- Request lifecycle (admission, scheduling, completion)
- KV cache pool allocation and radix-tree-based prefix caching
- Continuous batching and chunked prefill
- Overlap scheduling (CPU scheduling overlaps GPU forward pass)
- HTTP API and tokenizer service

`burn-ggml` / `burn-wgpu` is the *compute* layer. It owns:
- Tensor operations (GEMM, attention, MoE routing)
- Weight loading and layer-wise streaming from disk (`WeightCache`)
- KV cache SSD offloading (`KvOffloadManager`)
- Expert weight streaming (`ExpertWeightCache` / FlashMoE)
- Compute-data overlap (`PrefetchOps`)

The boundary is the `ForwardPass` trait: the engine calls `backend.forward(batch)` and gets back logits; the backend handles all memory movement internally.

### Design lineage

The scheduling logic is modeled after mini-sglang (Python, ~5000 lines), which is itself a clean-room reimplementation of the core scheduling ideas from SGLang. The Rust async patterns are modeled after mistral.rs. The radix cache is a faithful translation of mini-sglang's `RadixCache` class.

---
## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client (HTTP)                            │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ POST /v1/chat/completions
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Layer (axum)                           │
│   - OpenAI-compatible REST                                      │
│   - Server-Sent Events streaming                                │
│   - Request validation & rate limiting                          │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ mpsc::Sender<ApiRequest>
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Tokenizer Service (async task)                │
│   - HuggingFace tokenizers crate                                │
│   - Detokenizes streamed tokens back to the HTTP handler        │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ mpsc::Sender<EngineRequest>
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Scheduler                                │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐ │
│  │    Prefill Manager   │  │       Decode Manager             │ │
│  │  - chunked prefill   │  │  - collect decodable reqs        │ │
│  │  - token budget      │  │  - pad to captured CUDA sizes    │ │
│  │  - PrefillAdder      │  │    (Linux only)                  │ │
│  └──────────┬───────────┘  └──────────────────┬───────────────┘ │
│             │                                  │                 │
│  ┌──────────▼──────────────────────────────────▼───────────────┐│
│  │                  Radix Cache                                 ││
│  │  - Prefix tree over token sequences                         ││
│  │  - Page-aligned KV cache storage                            ││
│  │  - LRU eviction with ref-count locking                      ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────┬───────────────────────────────┘
                                  │ Batch (forward request)
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Engine (forward pass)                        │
│  - Assembles input tensors (input_ids, position_ids, attn_mask) │
│  - Calls backend.forward(batch)                                 │
│  - Runs sampling (temperature, top-p, top-k, repetition pen.)  │
│  - Routes generated tokens back to waiting requests             │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │   Backend (trait object)    │
                    │                             │
                    │  ┌────────────────────────┐ │
                    │  │  burn-wgpu (Linux)     │ │
                    │  │  Vulkan / Intel iGPU   │ │
                    │  └────────────────────────┘ │
                    │  ┌────────────────────────┐ │
                    │  │  burn-ggml (macOS)     │ │
                    │  │  ggml + Metal          │ │
                    │  └────────────────────────┘ │
                    └─────────────┬──────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
┌─────────────────┐  ┌────────────────────────┐  ┌────────────────────────┐
│  WeightCache    │  │   KvOffloadManager     │  │  ExpertWeightCache     │
│  (layer-wise    │  │   (SSD KV offload for  │  │  (FlashMoE expert      │
│   weight        │  │    global attn layers) │  │   weight streaming)    │
│   streaming)    │  └────────────────────────┘  └────────────────────────┘
└─────────────────┘            │
                               ▼
                    ┌────────────────────────┐
                    │    PrefetchOps         │
                    │  (async I/O overlap)   │
                    └────────────────────────┘
```

### Component interaction summary

| Component | Owns | Calls |
|-----------|------|-------|
| API Layer | HTTP server, SSE streams | Tokenizer Service |
| Tokenizer Service | encode/decode | Scheduler |
| Scheduler | request queues, radix cache, page table | Engine |
| Engine | forward pass assembly, sampling | Backend trait |
| Backend | tensors, weights, KV | WeightCache, KvOffloadManager, ExpertWeightCache |
| PrefetchOps | async I/O scheduling | OS async I/O (io_uring / macOS kqueue) |

---
## 3. Core Data Structures

### 3.1 `SamplingParams`

```rust
/// All parameters that control token sampling.
/// Matches the OpenAI API surface.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Maximum tokens to generate (not counting the prompt).
    pub max_new_tokens: usize,
    /// Softmax temperature. 1.0 = no change, <1.0 = sharper, >1.0 = flatter.
    pub temperature: f32,
    /// Nucleus sampling: keep the smallest set of tokens whose cumulative
    /// probability exceeds `top_p`. 1.0 = disabled.
    pub top_p: f32,
    /// Keep only the `top_k` most probable tokens. 0 = disabled.
    pub top_k: usize,
    /// Penalise tokens that have already appeared. 1.0 = no penalty.
    pub repetition_penalty: f32,
    /// Stop generating when any of these strings are produced.
    pub stop_sequences: Vec<String>,
    /// If true, stream partial results back to the client.
    pub stream: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            stop_sequences: vec![],
            stream: false,
        }
    }
}
```

### 3.2 Request State Machine

A request transitions through the following states:

```
Waiting ──► Prefilling ──► Decoding ──► Done
                │                │
                └────────────────┴──► Aborted
```

- **Waiting**: queued, not yet assigned any KV cache pages.
- **Prefilling**: has KV cache pages, currently processing its prompt (may span multiple iterations if chunked prefill is in effect).
- **Decoding**: prompt fully processed; generates one token per iteration.
- **Done**: stop criterion met (EOS token, stop string, `max_new_tokens` reached).
- **Aborted**: client disconnected or error.

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestState {
    Waiting,
    Prefilling {
        /// Number of tokens processed so far (across all prefill chunks).
        processed_tokens: usize,
    },
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
```

### 3.3 `Request`

```rust
use std::sync::Arc;
use tokio::sync::mpsc;

/// A single inference request.
pub struct Request {
    /// Unique identifier (UUID v4).
    pub id: RequestId,

    /// The full token sequence: system prompt + user turns + assistant turns
    /// concatenated. This is the canonical source of truth for the token IDs.
    pub input_ids: Vec<u32>,

    /// Sampling configuration.
    pub sampling_params: SamplingParams,

    /// Current state in the lifecycle.
    pub state: RequestState,

    // ── KV cache accounting ──────────────────────────────────────────────────

    /// How many tokens are covered by the radix cache hit (prefix reuse).
    /// These tokens already have KV entries in the pool; we skip them
    /// during prefill.
    pub cached_len: usize,

    /// How many tokens currently have KV cache entries on the device
    /// (i.e., have been processed by the forward pass at least once).
    pub device_len: usize,

    /// How many new tokens to process in the *current* scheduler iteration.
    /// Set by PrefillAdder; zero for decode-only requests.
    pub extend_len: usize,

    /// Indices of the KV cache pages allocated to this request, in order.
    /// Page `kv_pages[i]` holds KV for tokens at positions
    /// `[i * PAGE_SIZE, (i+1) * PAGE_SIZE)`.
    pub kv_pages: Vec<PageIndex>,

    // ── Output ───────────────────────────────────────────────────────────────

    /// Tokens generated so far in the decode phase.
    pub output_ids: Vec<u32>,

    /// Channel to stream generated tokens back to the HTTP handler.
    /// None if streaming is disabled.
    pub token_tx: Option<mpsc::Sender<TokenEvent>>,

    // ── Timing ───────────────────────────────────────────────────────────────

    pub arrival_time: std::time::Instant,
    pub first_token_time: Option<std::time::Instant>,
}

pub type RequestId = uuid::Uuid;
pub type PageIndex = u32;

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

### 3.4 `Batch`

```rust
/// A batch handed from the scheduler to the engine for a single forward pass.
#[derive(Debug)]
pub struct Batch {
    /// Whether this batch contains prefill tokens, decode tokens, or both
    /// (continuous batching mixes them).
    pub phase: BatchPhase,

    /// Requests participating in this batch, in order.
    /// The engine uses this to interpret `input_ids` and `page_table_entries`.
    pub requests: Vec<Arc<parking_lot::Mutex<Request>>>,

    /// Flat list of token IDs to process.
    /// For prefill requests: their `extend_len` new tokens.
    /// For decode requests: their last generated token (single token each).
    pub input_ids: Vec<u32>,

    /// Position IDs matching `input_ids`. Needed for RoPE.
    pub position_ids: Vec<u32>,

    /// For each request, the KV page indices it owns, used by the attention
    /// kernel to locate KV entries in the pool.
    pub page_table: Vec<Vec<PageIndex>>,

    /// Total number of KV slots consumed by this batch.
    /// Used by the engine to validate the pool has enough space.
    pub num_kv_slots: usize,

    /// For MoE models: pre-computed expert routing decisions (optional).
    /// If Some, the engine uses these directly; if None, routing is computed
    /// during the forward pass.
    pub expert_routing: Option<ExpertRouting>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchPhase {
    /// All requests are in the prefill phase.
    PrefillOnly,
    /// All requests are in the decode phase.
    DecodeOnly,
    /// Mix of prefill and decode (continuous batching).
    Mixed,
}

/// Pre-computed expert assignments for a MoE forward pass.
/// Computed by the router during scheduling to allow expert weight prefetching.
#[derive(Debug)]
pub struct ExpertRouting {
    /// For each token, the selected expert indices (top-k per token).
    /// Shape: [num_tokens, top_k]
    pub token_expert_ids: Vec<Vec<u32>>,
    /// Which experts are needed across the whole batch.
    pub unique_expert_ids: Vec<u32>,
}
```

### 3.5 `EngineContext`

```rust
use std::collections::HashMap;

/// Shared mutable state owned by the engine task.
/// Protected by a `parking_lot::Mutex` at the outer engine level;
/// never cloned — all operations go through `&mut EngineContext`.
pub struct EngineContext {
    /// The compute backend (burn-wgpu or burn-ggml).
    pub backend: Box<dyn BackendHandle>,

    /// The radix cache.
    pub radix_cache: RadixCache,

    /// The KV cache pool.
    pub kv_pool: KvCachePool,

    /// The page table: maps (request_id, page_slot_index) to PageIndex.
    /// Entries are inserted when pages are allocated and removed on request
    /// completion.
    pub page_table: HashMap<(RequestId, usize), PageIndex>,

    /// Requests in the waiting queue (FCFS by default).
    pub waiting: std::collections::VecDeque<Arc<parking_lot::Mutex<Request>>>,

    /// Requests currently prefilling.
    pub prefilling: Vec<Arc<parking_lot::Mutex<Request>>>,

    /// Requests currently decoding.
    pub decoding: Vec<Arc<parking_lot::Mutex<Request>>>,

    /// Configuration.
    pub config: EngineConfig,

    /// Statistics (for Prometheus / tracing).
    pub stats: EngineStats,
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Maximum total tokens (prompt + generated) across all requests in flight.
    pub max_total_tokens: usize,
    /// Number of KV cache pages in the pool.
    pub num_kv_pages: usize,
    /// Tokens per KV cache page.
    pub page_size: usize,
    /// Maximum tokens to process in one prefill iteration (token budget).
    pub max_prefill_tokens_per_iter: usize,
    /// Maximum requests in a single decode batch.
    pub max_decode_batch_size: usize,
    /// Model architecture metadata.
    pub model_config: ModelConfig,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub is_moe: bool,
    pub num_experts: Option<usize>,
    pub top_k_experts: Option<usize>,
}

#[derive(Debug, Default)]
pub struct EngineStats {
    pub total_requests_served: u64,
    pub total_tokens_generated: u64,
    pub cache_hit_tokens: u64,
    pub cache_miss_tokens: u64,
    pub current_waiting: usize,
    pub current_prefilling: usize,
    pub current_decoding: usize,
}
```

### 3.6 Backend trait

```rust
use burn::tensor::{Tensor, backend::Backend};

/// The interface between the engine and the compute backend.
/// Implemented by both burn-wgpu and burn-ggml adapters.
#[async_trait::async_trait]
pub trait BackendHandle: Send + Sync {
    /// Run one forward pass.
    /// Returns logits of shape [num_tokens, vocab_size].
    async fn forward(&mut self, batch: &Batch, ctx: &EngineContext) -> Result<Logits, EngineError>;

    /// Signal the backend to begin prefetching weights / KV for the next batch.
    /// Called while the current batch is executing on the GPU.
    async fn prefetch_next(&mut self, next_batch: &Batch) -> Result<(), EngineError>;

    /// Return the number of free KV pages currently available.
    fn free_kv_pages(&self) -> usize;

    /// Return memory usage statistics.
    fn memory_stats(&self) -> BackendMemoryStats;
}

/// Raw logits returned by the backend.
/// We deliberately keep this as a plain Vec to avoid lifetime entanglement
/// between the backend's GPU memory and the scheduler.
pub struct Logits {
    /// Shape: [num_tokens_or_last_tokens, vocab_size]
    /// For prefill batches, only the logits for the last token of each
    /// request are included (shape [num_requests, vocab_size]).
    pub data: Vec<f32>,
    pub num_rows: usize,
    pub vocab_size: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("backend error: {0}")]
    Backend(String),
    #[error("out of KV pages")]
    OutOfKvPages,
    #[error("request aborted: {0:?}")]
    Aborted(AbortReason),
}
```

---
## 4. Scheduler Design

The scheduler is the heart of `burn-inference`. It runs in a tight loop, assembling batches from the waiting queue, managing the radix cache, and handing batches to the engine. It is modeled directly on mini-sglang's scheduler.

### 4.1 The Scheduling Loop

```rust
/// The main engine loop. Runs on a dedicated OS thread (via spawn_blocking)
/// so that the synchronous GPU forward pass does not block the Tokio runtime.
pub async fn engine_loop(
    mut ctx: EngineContext,
    mut request_rx: mpsc::Receiver<Arc<parking_lot::Mutex<Request>>>,
) {
    loop {
        // ── Step 1: Drain incoming requests ──────────────────────────────
        while let Ok(req) = request_rx.try_recv() {
            ctx.waiting.push_back(req);
        }

        // ── Step 2: Schedule the next batch ──────────────────────────────
        let maybe_batch = schedule_batch(&mut ctx);

        if let Some(batch) = maybe_batch {
            // ── Step 3: Kick off prefetch for *this* batch (async I/O) ────
            // This is a non-blocking call; the backend queues I/O.
            let _ = ctx.backend.prefetch_next(&batch).await;

            // ── Step 4: Execute the forward pass (blocks the async task
            //    via spawn_blocking; Tokio runtime stays live) ──────────────
            let logits = tokio::task::spawn_blocking({
                // SAFETY: BackendHandle is Send; we move the raw pointer
                // across the spawn_blocking boundary. The lock on ctx
                // is held for the duration by the outer loop.
                let batch_ref = &batch as *const Batch as usize;
                let ctx_ref = &ctx as *const EngineContext as usize;
                move || {
                    let batch = unsafe { &*(batch_ref as *const Batch) };
                    let ctx = unsafe { &*(ctx_ref as *const EngineContext) };
                    // In practice we use a channel to send the batch to a
                    // dedicated GPU thread and receive results; see §4.4.
                    todo!("forward pass via GPU thread channel")
                }
            })
            .await
            .expect("GPU thread panicked");

            // ── Step 5: Process logits, sample tokens, update state ───────
            process_logits(&mut ctx, &batch, logits).await;
        } else {
            // No runnable requests. Block until a new request arrives.
            // sched_rx.recv().await suspends the task completely — zero CPU.
            if let Some(req) = request_rx.recv().await {
                ctx.waiting.push_back(req);
            }
        }
    }
}
```

In practice, the forward pass runs on a dedicated thread that owns the GPU context (Metal/Vulkan command queues are not `Send` in general). The scheduler communicates with this thread via `std::sync::mpsc` channels.

### 4.2 `schedule_batch`

```rust
/// Produce one batch from the current state. Returns None if there is
/// nothing runnable.
fn schedule_batch(ctx: &mut EngineContext) -> Option<Batch> {
    // ── Phase A: collect decode requests (always highest priority) ────────
    let decode_reqs = collect_decode_requests(ctx);

    // ── Phase B: add prefill requests up to token budget ──────────────────
    let prefill_reqs = add_prefill_requests(ctx, &decode_reqs);

    if decode_reqs.is_empty() && prefill_reqs.is_empty() {
        return None;
    }

    // ── Phase C: build the Batch struct ───────────────────────────────────
    Some(assemble_batch(ctx, prefill_reqs, decode_reqs))
}
```

### 4.3 Decode Manager: `collect_decode_requests`

All requests in the `Decoding` state are eligible every iteration. This is the key property of continuous batching: decode never waits.

```rust
fn collect_decode_requests(
    ctx: &mut EngineContext,
) -> Vec<Arc<parking_lot::Mutex<Request>>> {
    let mut decode_reqs = Vec::new();

    ctx.decoding.retain(|req_arc| {
        let req = req_arc.lock();
        match req.state {
            RequestState::Decoding => {
                decode_reqs.push(req_arc.clone());
                true // keep in ctx.decoding
            }
            RequestState::Done | RequestState::Aborted { .. } => {
                // Release KV pages back to the pool.
                drop(req);
                free_kv_pages(ctx, req_arc);
                false // remove from ctx.decoding
            }
            _ => true,
        }
    });

    // Enforce max decode batch size (admission control for new prefill work).
    if decode_reqs.len() > ctx.config.max_decode_batch_size {
        decode_reqs.truncate(ctx.config.max_decode_batch_size);
    }

    decode_reqs
}
```

### 4.4 Prefill Manager: `add_prefill_requests`

The prefill manager uses a **token budget** (`max_prefill_tokens_per_iter`) to decide how many new tokens to process. Long prompts are chunked: if a request has 4096 tokens left and the budget is 1024, we process only 1024 this iteration and the remaining 3072 in future iterations.

```rust
fn add_prefill_requests(
    ctx: &mut EngineContext,
    decode_reqs: &[Arc<parking_lot::Mutex<Request>>],
) -> Vec<Arc<parking_lot::Mutex<Request>>> {
    // Budget: total token budget minus tokens already consumed by decode.
    let decode_tokens = decode_reqs.len(); // 1 token per decode request
    let remaining_budget = ctx
        .config
        .max_prefill_tokens_per_iter
        .saturating_sub(decode_tokens);

    PrefillAdder::new(ctx, remaining_budget).run()
}

/// Greedy algorithm that fills the token budget with prefill work.
struct PrefillAdder<'a> {
    ctx: &'a mut EngineContext,
    budget: usize,
    added: Vec<Arc<parking_lot::Mutex<Request>>>,
    tokens_used: usize,
}

impl<'a> PrefillAdder<'a> {
    fn new(ctx: &'a mut EngineContext, budget: usize) -> Self {
        Self { ctx, budget, added: Vec::new(), tokens_used: 0 }
    }

    fn run(mut self) -> Vec<Arc<parking_lot::Mutex<Request>>> {
        // First, continue any in-progress prefill requests (they already
        // have KV pages allocated).
        for req_arc in self.ctx.prefilling.iter() {
            if self.tokens_used >= self.budget {
                break;
            }
            self.schedule_prefill_chunk(req_arc.clone());
        }

        // Then, admit new requests from the waiting queue.
        while self.tokens_used < self.budget && !self.ctx.waiting.is_empty() {
            // Check if we have enough free KV pages for the next request.
            let req_arc = self.ctx.waiting.front().unwrap();
            let req = req_arc.lock();
            let pages_needed = pages_for_tokens(
                req.input_ids.len(),
                self.ctx.config.page_size,
            );
            drop(req);

            if self.ctx.kv_pool.free_pages() < pages_needed {
                // Try to evict from radix cache.
                let evicted = self.ctx.radix_cache.evict_pages(pages_needed);
                if evicted < pages_needed {
                    // Cannot admit; stop.
                    break;
                }
            }

            let req_arc = self.ctx.waiting.pop_front().unwrap();
            // Match prefix against radix cache.
            let cached_len = self.ctx.radix_cache.match_prefix(&req_arc.lock().input_ids);
            {
                let mut req = req_arc.lock();
                req.cached_len = cached_len;
                req.device_len = cached_len;
                req.state = RequestState::Prefilling { processed_tokens: 0 };
            }
            // Allocate KV pages.
            allocate_kv_pages(self.ctx, &req_arc);

            self.schedule_prefill_chunk(req_arc.clone());
            self.ctx.prefilling.push(req_arc);
        }

        self.added
    }

    fn schedule_prefill_chunk(&mut self, req_arc: Arc<parking_lot::Mutex<Request>>) {
        let remaining_tokens;
        {
            let req = req_arc.lock();
            remaining_tokens = req.input_ids.len() - req.device_len;
        }

        let chunk = remaining_tokens.min(self.budget - self.tokens_used);
        if chunk == 0 {
            return;
        }

        {
            let mut req = req_arc.lock();
            req.extend_len = chunk;
        }

        self.tokens_used += chunk;
        self.added.push(req_arc);
    }
}

fn pages_for_tokens(num_tokens: usize, page_size: usize) -> usize {
    num_tokens.div_ceil(page_size)
}
```

### 4.5 Batch Assembly: `assemble_batch`

```rust
fn assemble_batch(
    ctx: &EngineContext,
    prefill_reqs: Vec<Arc<parking_lot::Mutex<Request>>>,
    decode_reqs: Vec<Arc<parking_lot::Mutex<Request>>>,
) -> Batch {
    let mut input_ids = Vec::new();
    let mut position_ids = Vec::new();
    let mut page_table = Vec::new();
    let mut all_reqs = Vec::new();

    // Prefill requests first (packed into the batch).
    for req_arc in &prefill_reqs {
        let req = req_arc.lock();
        let start = req.device_len;
        let end = req.device_len + req.extend_len;
        input_ids.extend_from_slice(&req.input_ids[start..end]);
        position_ids.extend((start as u32)..(end as u32));
        page_table.push(req.kv_pages.clone());
        drop(req);
        all_reqs.push(req_arc.clone());
    }

    // Decode requests (one token each — the last generated token).
    for req_arc in &decode_reqs {
        let req = req_arc.lock();
        let last_token = *req.output_ids.last()
            .unwrap_or(req.input_ids.last().unwrap());
        input_ids.push(last_token);
        position_ids.push(req.device_len as u32);
        page_table.push(req.kv_pages.clone());
        drop(req);
        all_reqs.push(req_arc.clone());
    }

    let phase = match (prefill_reqs.is_empty(), decode_reqs.is_empty()) {
        (false, true) => BatchPhase::PrefillOnly,
        (true, false) => BatchPhase::DecodeOnly,
        _ => BatchPhase::Mixed,
    };

    Batch {
        phase,
        requests: all_reqs,
        input_ids,
        position_ids,
        page_table,
        num_kv_slots: 0, // filled in after assembly
        expert_routing: None, // filled in by MoE routing pass if applicable
    }
}
```

### 4.6 Overlap Scheduling

Overlap scheduling means: **while the GPU executes the current batch, the CPU schedules the next batch.** This hides scheduling latency behind GPU computation.

The architecture uses two Tokio tasks:

```
  Tokio Task A (scheduler)              GPU Thread
  ─────────────────────────             ─────────────────────────
  schedule_batch(ctx) ─────────────►   forward(current_batch)
         │                                     │
         │  (concurrent with GPU)              │
         ▼                                     │
  next_batch = schedule_batch(ctx)             │
  prefetch_next(next_batch)            ◄───────┘ logits ready
         │                             
         ▼
  process_logits(logits)
  (update request state)
         │
         ▼
  send next_batch to GPU ─────────────► forward(next_batch)
```

The loop has three distinct idle states that must be handled correctly:

| State | Condition | Correct behaviour |
|---|---|---|
| **Active** | batch scheduled, GPU running | `result_rx.await` suspends until GPU done |
| **GPU draining** | GPU running, no new batch schedulable | `result_rx.await` suspends until GPU done |
| **Truly idle** | no in-flight work, no waiting requests | `sched_rx.recv().await` suspends until new request |

The naive approach of using `tokio::task::yield_now().await` in the else branch
is **wrong**: `yield_now` re-queues the task immediately after one poll cycle, so
the loop spins at full CPU speed even when the server is completely idle. For a
10-minute idle period this burns one CPU core the entire time doing nothing.

`result_rx.await` and `sched_rx.recv().await` are true suspensions — they tell
the Tokio runtime "do not poll me again until this specific event fires." Zero
CPU usage until the event occurs.

```rust
pub async fn run_overlapped_loop(
    mut ctx:      EngineContext,
    mut sched_rx: mpsc::Receiver<Arc<Mutex<Request>>>,
    gpu_tx:       std::sync::mpsc::SyncSender<GpuWork>,
) {
    // Holds (batch sent to GPU, channel to receive logits back).
    let mut in_flight: Option<(Batch, oneshot::Receiver<Logits>)> = None;

    loop {
        // ── A. Drain any newly arrived requests ───────────────────────────
        // try_recv() is non-blocking: takes what is already in the channel
        // without suspending. We drain all of them before scheduling.
        while let Ok(req) = sched_rx.try_recv() {
            ctx.waiting.push_back(req);
        }

        // ── B. Collect results from the previous forward pass ─────────────
        if let Some((prev_batch, result_rx)) = in_flight.take() {
            let logits = result_rx.await.unwrap();
            process_and_transition(&mut ctx, &prev_batch, logits);
        }

        // ── C. Schedule the next batch ────────────────────────────────────
        let Some(batch) = schedule_batch(&mut ctx) else {
            // Nothing schedulable. Two sub-cases:
            match in_flight.take() {
                Some((prev_batch, result_rx)) => {
                    // ── State: GPU draining ──────────────────────────────
                    // GPU is still running a batch but we have nothing new
                    // to schedule alongside it. Suspend until GPU finishes.
                    // Tokio wakes this task only when the oneshot fires.
                    let logits = result_rx.await.unwrap();
                    process_and_transition(&mut ctx, &prev_batch, logits);
                    // Re-drain: new requests may have arrived while GPU ran.
                    while let Ok(req) = sched_rx.try_recv() {
                        ctx.waiting.push_back(req);
                    }
                }
                None => {
                    // ── State: truly idle ────────────────────────────────
                    // No in-flight GPU work, no waiting requests.
                    // Suspend until a new request arrives on the channel.
                    // Tokio parks this task completely — zero CPU usage
                    // for the entire idle period (seconds, minutes, hours).
                    let req = sched_rx.recv().await.unwrap();
                    ctx.waiting.push_back(req);
                    // Drain any others that arrived at the same time.
                    while let Ok(req) = sched_rx.try_recv() {
                        ctx.waiting.push_back(req);
                    }
                }
            }
            continue;
        };

        // ── D. Fire prefetch I/O for this batch (non-blocking) ───────────
        // For burn-ggml: triggers pread() for next layer weights / KV pages.
        // For burn-wgpu (Phase 0): no-op default impl.
        ctx.backend.prefetch(&batch);

        // ── E. Send batch to GPU thread ───────────────────────────────────
        // gpu_tx is a sync channel with capacity 1. This send is non-blocking
        // because the GPU thread always drains it before we can fill it again
        // (we only send when in_flight is None).
        let (result_tx, result_rx) = oneshot::channel();
        gpu_tx.send(GpuWork { batch: batch.clone(), result_tx }).unwrap();
        in_flight = Some((batch, result_rx));

        // Loop back to A immediately. While the GPU runs the batch we just
        // sent, we drain new requests and schedule the batch after next.
        // The next iteration hits step B and suspends on result_rx.await,
        // which is the correct overlap point.
    }
}
```

**Why `in_flight` is checked twice (steps B and C):**

Step B handles the normal active case — we always try to collect the previous
result before scheduling the next batch, so we can transition request states
(prefill→decode, decode→done) and make freed KV pages available to the scheduler.
Step C's `in_flight.take()` handles the edge case where `schedule_batch` returns
`None` *after* step B already consumed `in_flight` — meaning we just finished the
last batch and there is nothing new to run. In that case `in_flight` is `None` in
step C and we fall through to the truly idle branch.

### 4.7 Priority and Preemption Policy

For Phase 1, the scheduling policy is **FCFS with preemption by eviction**:

- Requests are admitted FCFS from the waiting queue.
- If the KV pool is exhausted and no eviction is possible, the *last-admitted* prefill-phase request is preempted: its KV pages are freed, it returns to Waiting state.
- Decode-phase requests are never preempted (doing so would require expensive KV recomputation; swap-to-SSD is in scope for Phase 2).
- Priority levels (`High`, `Normal`, `Low`) are planned but not implemented in Phase 1.

```rust
fn preempt_one_prefill_request(ctx: &mut EngineContext) -> bool {
    // Find the most recently admitted prefill request (last in the list).
    if let Some(req_arc) = ctx.prefilling.pop() {
        let mut req = req_arc.lock();
        // Free its KV pages.
        for &page in &req.kv_pages {
            ctx.kv_pool.free(page);
        }
        req.kv_pages.clear();
        req.device_len = 0;
        req.extend_len = 0;
        req.state = RequestState::Waiting;
        drop(req);
        // Return to front of waiting queue so it gets priority next iteration.
        ctx.waiting.push_front(req_arc);
        true
    } else {
        false
    }
}
```

---
## 5. Radix Cache

The radix cache is a prefix tree over token sequences. Each node stores the KV
cache pages for the token prefix that leads to it. When a new request arrives,
we walk the tree to find the longest matching prefix — the matching portion does
not need to be recomputed.

### 5.1 Why Not Use an Existing Crate

Before implementing from scratch, we evaluated every maintained Rust trie/radix
crate on crates.io:

| Crate | Downloads | `Vec<u32>` keys | Longest-prefix match | Custom node metadata | Verdict |
|---|---|---|---|---|---|
| `radix_trie` v0.3.0 | 57M | Yes (nibble-encoded) | `get_ancestor()` | ❌ private internals | Closest, but unusable |
| `trie-rs` v0.4.2 | 4.9M | u8/u16 only | ❌ no | ❌ | Immutable after build |
| `patricia_tree` v0.10.1 | 1.5M | Byte-oriented | ❌ no | ❌ | Wrong key type |
| `qp-trie` v0.8.2 | 266K | Bytes only | `longest_common_prefix` | ❌ | Stale (2023), byte-only |
| `sequence_trie` v0.3.6 | 2.85M | Yes (native u32) | Manual traversal | ❌ | Unmaintained since 2018 |
| `trie-hard` v0.2.0 | 672K | No | ❌ | ❌ | Bulk-load only (Cloudflare) |

**The fundamental blocker is the same across all crates: every crate hides its
internal node structure.** There is no way to attach `lock_ref: i32`,
`last_used: Instant`, and `page_indices: Vec<PageIndex>` to a tree node without
forking the crate. The LRU eviction logic (heap of evictable leaves, parent-chain
ref-count propagation) also has no hook points in any of them.

`radix_trie` is the closest — it has `get_ancestor()` for longest-prefix matching
and handles edge splitting automatically. But after matching you still need to
split the terminal edge on a partial match, and that API is not public. You would
be fighting the abstraction, and the crate's README notes it "hasn't been used in
production."

**mistral.rs** (the most complete Rust inference engine) avoids this entirely by
using a flat `IndexMap<Tokens, CacheElement>` with a linear scan +
`shared_prefix_len()`. This is O(N·L) and does not scale to large concurrent
caches.

**No existing Rust port of SGLang's radix cache exists** as of April 2026.

**Decision: implement from scratch, ~400–600 lines, using `slab` as the node
arena.** The SGLang Python source (`radix_cache.py`, ~300 lines) is the direct
algorithm reference.

### 5.2 Implementation Strategy: Slab-Based Arena

The key design insight is to store all nodes in a `Slab<RadixNode>` (generational
arena) and use `NodeId = usize` handles for parent/child relationships. This
sidesteps Rust's borrow checker issues with self-referential tree mutation
entirely — no `unsafe`, no `Rc<RefCell<>>`, no `Arc<RwLock<>>` per node.

**Building blocks:**

| Crate | Version | Downloads | Purpose |
|---|---|---|---|
| `slab` | 0.4.x | 563M | O(1) generational arena for node storage |
| `rustc-hash` | 2.x | — | `FxHashMap<u32, NodeId>` for child lookup by first token |
| `tokio::sync::Mutex` | — | — | Single async lock over the whole cache |
| `std::collections::BinaryHeap` | — | — | LRU eviction heap over leaf nodes |

The entire `RadixCache` is protected by a single `tokio::sync::Mutex`. This is
correct and efficient: prefix matching is a short critical section (O(tree depth),
typically O(1) for LLM prompt lengths), and the scheduler holds the lock only
during `match_prefix` and `insert_prefix` calls — not during the forward pass.

### 5.3 Node Structure

```rust
use slab::Slab;
use rustc_hash::FxHashMap;
use std::time::Instant;
use std::collections::BinaryHeap;
use std::cmp::Reverse;

pub type NodeId = usize;
pub type PageIndex = u32;

/// A single node in the radix tree.
/// Stored by value in a `Slab<RadixNode>` — no heap allocation per node.
pub struct RadixNode {
    /// Tokens on the edge from parent → this node.
    /// An empty vec means this is the root.
    pub edge_tokens: Vec<u32>,

    /// KV cache pages for `edge_tokens`, appended to the parent's page list.
    /// Only full pages (page_size tokens) are stored here.
    pub kv_pages: Vec<PageIndex>,

    /// Children keyed by the first token of their edge.
    /// FxHashMap is ~2x faster than std HashMap for small integer keys.
    pub children: FxHashMap<u32, NodeId>,

    /// Parent node id. None only for root.
    pub parent: Option<NodeId>,

    /// Ref count: number of in-flight requests using this node.
    /// A node with lock_ref > 0 cannot be evicted.
    pub lock_ref: i32,

    /// LRU timestamp: updated on every match and on child insertion.
    pub last_used: Instant,
}

impl RadixNode {
    fn new_root() -> Self {
        RadixNode {
            edge_tokens: vec![],
            kv_pages: vec![],
            children: FxHashMap::default(),
            parent: None,
            lock_ref: 0,
            last_used: Instant::now(),
        }
    }

    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    fn is_evictable(&self) -> bool {
        self.is_leaf() && self.lock_ref == 0 && !self.kv_pages.is_empty()
    }
}
```

### 5.4 `RadixCache`

```rust
pub struct RadixCache {
    /// Arena: O(1) insert/remove/lookup by NodeId.
    nodes: Slab<RadixNode>,
    root: NodeId,
    /// Heap of (last_used, node_id) for evictable leaves.
    /// Entries may be stale (node no longer evictable); checked lazily.
    evictable: BinaryHeap<(Reverse<Instant>, NodeId)>,
    /// Tokens per KV cache page.
    page_size: usize,
    /// Current total pages held across all nodes.
    total_pages: usize,
    /// Eviction is triggered when total_pages >= max_pages.
    max_pages: usize,
}

impl RadixCache {
    pub fn new(page_size: usize, max_pages: usize) -> Self {
        let mut nodes = Slab::new();
        let root = nodes.insert(RadixNode::new_root());
        RadixCache {
            nodes,
            root,
            evictable: BinaryHeap::new(),
            page_size,
            total_pages: 0,
            max_pages,
        }
    }
```

### 5.5 `match_prefix`

Walks the tree following `tokens`, returns `(matched_token_count, Vec<PageIndex>)`.
Increments `lock_ref` on every matched node to prevent eviction while the
request is live.

```rust
    /// Returns (matched_len, kv_pages) for the longest cached prefix of `tokens`.
    /// Locks matched nodes (lock_ref += 1) — caller must call unlock_nodes().
    pub fn match_prefix(&mut self, tokens: &[u32]) -> (usize, Vec<PageIndex>, Vec<NodeId>) {
        let mut matched_pages: Vec<PageIndex> = Vec::new();
        let mut matched_nodes: Vec<NodeId>   = Vec::new();
        let mut offset = 0usize;
        let mut cur    = self.root;

        loop {
            let first = match tokens.get(offset) {
                Some(&t) => t,
                None => break,
            };

            let child_id = match self.nodes[cur].children.get(&first).copied() {
                Some(id) => id,
                None => break,
            };

            let edge = &self.nodes[child_id].edge_tokens;
            let remaining = &tokens[offset..];
            let match_len = edge.iter().zip(remaining).take_while(|(a, b)| a == b).count();

            if match_len == 0 { break; }

            // Only count full pages.
            let full_pages = match_len / self.page_size;
            if full_pages == 0 { break; }

            matched_pages.extend_from_slice(&self.nodes[child_id].kv_pages[..full_pages]);
            matched_nodes.push(child_id);
            self.nodes[child_id].lock_ref += 1;
            self.nodes[child_id].last_used = Instant::now();
            offset += full_pages * self.page_size;

            if match_len < edge.len() { break; } // partial edge match: stop
            cur = child_id;
        }

        (offset, matched_pages, matched_nodes)
    }
```

### 5.6 `insert_prefix` and `split_edge`

Called after a request finishes prefill to cache newly computed KV pages.

```rust
    /// Insert `tokens[..N*page_size]` → `pages[..N]` into the tree.
    /// Handles edge splitting when a partial match exists.
    pub fn insert_prefix(&mut self, tokens: &[u32], pages: &[PageIndex]) {
        let full_pages  = pages.len();
        let full_tokens = full_pages * self.page_size;
        if full_pages == 0 { return; }
        let tokens = &tokens[..full_tokens];

        // Evict if needed before inserting.
        if self.total_pages + full_pages > self.max_pages {
            self.evict_pages(self.total_pages + full_pages - self.max_pages);
        }

        let mut offset  = 0usize;
        let mut cur     = self.root;

        loop {
            if offset >= tokens.len() { break; }
            let first = tokens[offset];

            match self.nodes[cur].children.get(&first).copied() {
                None => {
                    // No matching child: insert new leaf.
                    let new_id = self.nodes.insert(RadixNode {
                        edge_tokens: tokens[offset..].to_vec(),
                        kv_pages:    pages[offset / self.page_size..].to_vec(),
                        children:    FxHashMap::default(),
                        parent:      Some(cur),
                        lock_ref:    0,
                        last_used:   Instant::now(),
                    });
                    self.nodes[cur].children.insert(first, new_id);
                    let added = pages[offset / self.page_size..].len();
                    self.total_pages += added;
                    // New leaf is immediately evictable.
                    self.evictable.push((Reverse(Instant::now()), new_id));
                    break;
                }
                Some(child_id) => {
                    let edge_len = self.nodes[child_id].edge_tokens.len();
                    let remaining = &tokens[offset..];
                    let match_len = self.nodes[child_id]
                        .edge_tokens
                        .iter()
                        .zip(remaining)
                        .take_while(|(a, b)| a == b)
                        .count();

                    if match_len == edge_len {
                        // Full edge match: descend.
                        self.nodes[child_id].last_used = Instant::now();
                        offset += edge_len;
                        cur = child_id;
                    } else {
                        // Partial match: split the edge at match_len.
                        let mid_id = self.split_edge(cur, child_id, match_len);
                        offset += match_len;
                        cur = mid_id;
                    }
                }
            }
        }
    }

    /// Split the edge from `parent` → `child` at `split_at` tokens.
    /// Creates a new intermediate node; returns its NodeId.
    fn split_edge(&mut self, parent: NodeId, child: NodeId, split_at: usize) -> NodeId {
        let page_split = split_at / self.page_size;
        let prefix_tokens = self.nodes[child].edge_tokens[..split_at].to_vec();
        let suffix_tokens = self.nodes[child].edge_tokens[split_at..].to_vec();
        let prefix_pages  = self.nodes[child].kv_pages[..page_split].to_vec();
        let suffix_pages  = self.nodes[child].kv_pages[page_split..].to_vec();
        let first_suffix  = suffix_tokens[0];
        let first_prefix  = prefix_tokens[0];

        // Trim child to suffix.
        self.nodes[child].edge_tokens = suffix_tokens;
        self.nodes[child].kv_pages    = suffix_pages;
        self.nodes[child].parent      = None; // will be set below

        // Create intermediate node with prefix, pointing to child.
        let mid_id = self.nodes.insert(RadixNode {
            edge_tokens: prefix_tokens,
            kv_pages:    prefix_pages,
            children:    { let mut m = FxHashMap::default(); m.insert(first_suffix, child); m },
            parent:      Some(parent),
            lock_ref:    0,
            last_used:   Instant::now(),
        });
        self.nodes[child].parent = Some(mid_id);

        // Replace child pointer in parent.
        self.nodes[parent].children.insert(first_prefix, mid_id);
        mid_id
    }
```

### 5.7 Lock/Unlock Lifecycle

```rust
    /// Release locks on nodes acquired during match_prefix.
    /// Updates last_used so recently-released nodes are evicted last.
    pub fn unlock_nodes(&mut self, node_ids: &[NodeId]) {
        for &id in node_ids {
            if let Some(node) = self.nodes.get_mut(id) {
                node.lock_ref = (node.lock_ref - 1).max(0);
                node.last_used = Instant::now();
                // If now evictable, add to heap.
                if node.is_evictable() {
                    self.evictable.push((Reverse(node.last_used), id));
                }
            }
        }
    }
```

### 5.8 LRU Eviction

```rust
    /// Evict LRU leaf nodes until `target_pages` pages have been freed.
    /// Returns the freed page indices (caller returns them to the KV pool).
    pub fn evict_pages(&mut self, target_pages: usize) -> Vec<PageIndex> {
        let mut freed_pages: Vec<PageIndex> = Vec::new();

        while freed_pages.len() < target_pages {
            let candidate = loop {
                match self.evictable.pop() {
                    None => return freed_pages, // nothing left to evict
                    Some((_, id)) => {
                        // The heap may contain stale entries.
                        if self.nodes.contains(id) && self.nodes[id].is_evictable() {
                            break id;
                        }
                    }
                }
            };

            // Collect pages from this leaf.
            let pages = std::mem::take(&mut self.nodes[candidate].kv_pages);
            self.total_pages -= pages.len();
            freed_pages.extend_from_slice(&pages);

            // Remove from parent's child map.
            if let Some(parent_id) = self.nodes[candidate].parent {
                let first_token = self.nodes[candidate].edge_tokens.first().copied();
                if let Some(t) = first_token {
                    self.nodes[parent_id].children.remove(&t);
                    // If parent is now a childless leaf with lock_ref==0, it
                    // becomes evictable too.
                    if self.nodes[parent_id].is_evictable() {
                        let ts = self.nodes[parent_id].last_used;
                        self.evictable.push((Reverse(ts), parent_id));
                    }
                }
            }

            self.nodes.remove(candidate);
        }

        freed_pages
    }
}
```

### 5.9 Page-Aligned Storage

Only **complete pages** are inserted into the radix cache. A page holds exactly
`page_size` tokens. If a prompt length is not a multiple of `page_size`, the
partial last page stays in the request's private allocation and is promoted only
when it fills.

```rust
/// Promote a completed request's KV pages into the radix cache.
/// Called when a request transitions from Prefilling → Decoding.
pub fn promote_to_cache(cache: &mut RadixCache, req: &Request) {
    let full_pages  = req.device_len / cache.page_size;
    let cache_tokens = full_pages * cache.page_size;
    if full_pages == 0 { return; }
    cache.insert_prefix(&req.input_ids[..cache_tokens], &req.kv_pages[..full_pages]);
}
```

### 5.10 Interaction with KvOffloadManager (SSD offload)

When SSD offloading is enabled, evicted pages are written to SSD rather than freed.
The page state transitions from `Resident` → `Offloaded`. Subsequent cache hits
that land on `Offloaded` pages trigger a `backend.prefetch()` call before the
forward pass.

```rust
pub fn evict_to_ssd(
    cache:      &mut RadixCache,
    kv_offload: &KvOffloadManager,
    target:     usize,
) {
    let pages = cache.evict_pages(target);
    for page in pages {
        // Fire-and-forget async write; page state tracked in KvOffloadManager.
        kv_offload.offload_page_async(page);
    }
}
```

### 5.11 Thread Safety

The entire `RadixCache` is wrapped in a single `tokio::sync::Mutex<RadixCache>`.
This is the correct choice because:

- Prefix matching is a short critical section (O(tree depth) ≈ O(1) for typical
  LLM prompts where depth rarely exceeds 10–20 nodes).
- The scheduler holds the lock only during `match_prefix` and `insert_prefix`,
  never during the forward pass itself.
- Per-node locking (the `Arc<RwLock<RadixNode>>` approach used in the original
  draft) creates complex deadlock scenarios during edge splitting and parent-chain
  ref-count propagation.
- `tokio::sync::Mutex` is async-aware: `.lock().await` yields the Tokio task
  rather than blocking the thread, keeping the scheduler responsive.

```rust
// In EngineContext:
pub radix_cache: tokio::sync::Mutex<RadixCache>,

// Usage in scheduler:
async fn schedule_batch(ctx: &EngineContext, req: &mut Request) {
    let mut cache = ctx.radix_cache.lock().await;  // async, non-blocking
    let (matched_len, pages, locked_nodes) = cache.match_prefix(&req.input_ids);
    req.cached_len   = matched_len;
    req.locked_nodes = locked_nodes;
    drop(cache); // release before forward pass
    // ...
}
```
---
## 6. KV Cache Management

### 6.1 KV Cache Pool Layout

The KV cache pool is a single contiguous allocation that holds all KV tensors for all in-flight requests. The layout mirrors mini-sglang's:

```
kv_pool: [num_layers, 2, num_pages, page_size, num_kv_heads, head_dim]
          ─────────  ─  ─────────  ─────────  ────────────  ────────
          layer idx  K/V page idx  pos in pg  head index    dim
```

The "2" dimension is key (K) and value (V) stored contiguously. Page `p` contains KV for `page_size` sequential token positions, for all layers simultaneously.

```rust
pub struct KvCachePool {
    /// Total number of pages in the pool.
    pub num_pages: usize,
    /// Tokens per page.
    pub page_size: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of KV attention heads.
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Free list of page indices.
    pub free_pages: Vec<PageIndex>,
    /// Page state: Free, InUse(request_id), Offloaded(ssd_offset).
    pub page_states: Vec<PageState>,
    /// The actual tensor buffer, allocated on the GPU device.
    /// Accessed by the backend, not directly by the scheduler.
    pub buffer_handle: KvBufferHandle,
}

#[derive(Debug, Clone)]
pub enum PageState {
    Free,
    InUse(RequestId),
    /// Page has been evicted to SSD; logical content is preserved.
    Offloaded { ssd_offset: u64, size_bytes: usize },
    /// Page is being written to SSD asynchronously.
    Offloading,
}

impl KvCachePool {
    pub fn allocate(&mut self) -> Option<PageIndex> {
        self.free_pages.pop()
    }

    pub fn free(&mut self, page: PageIndex) {
        self.page_states[page as usize] = PageState::Free;
        self.free_pages.push(page);
    }

    pub fn free_pages(&self) -> usize {
        self.free_pages.len()
    }

    /// Compute the byte size of the full buffer.
    pub fn buffer_size_bytes(
        num_pages: usize,
        page_size: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype_bytes: usize,
    ) -> usize {
        // 2 = K + V
        2 * num_layers * num_pages * page_size * num_kv_heads * head_dim * dtype_bytes
    }
}
```

### 6.2 Sizing the Pool

On a 16 GB system, we need to budget carefully:

| Component | macOS (31B Q4) | Linux (26B MoE Q4) |
|-----------|----------------|--------------------|
| Model weights | ~18 GB | ~8 GB active experts |
| KV cache pool | 2–4 GB | 4–6 GB |
| Working memory | 0.5 GB | 0.5 GB |
| OS + other | 1 GB | 1 GB |

For the 31B dense model on macOS with 32 GB RAM:
- KV entry size per token: `2 * num_layers * num_kv_heads * head_dim * sizeof(fp16)` bytes
- For Qwen3-32B: 64 layers, 8 KV heads, 128 head_dim → `2 * 64 * 8 * 128 * 2 = 262,144 bytes` per token
- At page_size=16: `262,144 * 16 / 1024 / 1024 = 4 MB` per page
- 512 pages = 2 GB for KV cache pool

For the 26B MoE on Linux with 16 GB VRAM (shared + system):
- Qwen2.5-MoE variants have fewer KV heads; estimate ~1 GB for a 1024-page pool with page_size=16

### 6.3 Page Table

The page table maps a request's sequence positions to pool page indices:

```rust
/// Maps (request_id, page_slot) -> PageIndex.
/// page_slot 0 = tokens [0, page_size), slot 1 = [page_size, 2*page_size), etc.
pub struct PageTable {
    inner: HashMap<(RequestId, usize), PageIndex>,
}

impl PageTable {
    pub fn insert(&mut self, req_id: RequestId, slot: usize, page: PageIndex) {
        self.inner.insert((req_id, slot), page);
    }

    pub fn lookup(&self, req_id: RequestId, slot: usize) -> Option<PageIndex> {
        self.inner.get(&(req_id, slot)).copied()
    }

    pub fn remove_request(&mut self, req_id: RequestId) -> Vec<PageIndex> {
        let keys: Vec<_> = self.inner.keys()
            .filter(|(r, _)| *r == req_id)
            .cloned()
            .collect();
        keys.into_iter()
            .filter_map(|k| self.inner.remove(&k))
            .collect()
    }
}
```

For the attention kernel, we need a flat representation per request:

```rust
/// Convert a request's page table entries into a flat Vec for the attention kernel.
/// The kernel receives `kv_indptr` and `kv_indices` in the FlashInfer/PagedAttention style.
pub fn build_paged_attn_metadata(
    page_table: &PageTable,
    requests: &[RequestId],
    page_size: usize,
    max_seq_len: usize,
) -> PagedAttnMetadata {
    let mut kv_indptr = vec![0u32]; // CSR row pointers
    let mut kv_indices = vec![];    // page indices

    for &req_id in requests {
        let mut slot = 0;
        loop {
            match page_table.lookup(req_id, slot) {
                Some(page) => {
                    kv_indices.push(page);
                    slot += 1;
                }
                None => break,
            }
        }
        kv_indptr.push(kv_indices.len() as u32);
    }

    PagedAttnMetadata { kv_indptr, kv_indices, page_size }
}
```

### 6.4 Unified Page Allocator: Sliding-Window and Global Layers

For models with **sliding-window attention** (local attention layers) interleaved with **global attention layers** (full-context layers, as in Qwen2.5's hybrid attention), the KV cache has two regimes:

- **Local layers**: KV only needed for the last `window_size` tokens. These pages can be reused once they slide out of the window.
- **Global layers**: KV needed for the entire context. These may be offloaded to SSD via `KvOffloadManager` when memory pressure is high.

```rust
pub struct UnifiedPageAllocator {
    /// Pool for local attention layers (recycled as the window slides).
    pub local_pool: KvCachePool,
    /// Pool for global attention layers (may be offloaded to SSD).
    pub global_pool: KvCachePool,
    /// The KV offload manager for global layers.
    pub kv_offload: KvOffloadManager,
    /// Window size (tokens) for local attention layers.
    pub window_size: usize,
}

impl UnifiedPageAllocator {
    /// Allocate pages for a request that has just been admitted.
    /// Returns (local_pages, global_pages).
    pub fn allocate_for_request(
        &mut self,
        req: &Request,
        num_local_layers: usize,
        num_global_layers: usize,
    ) -> Result<(Vec<PageIndex>, Vec<PageIndex>), EngineError> {
        let seq_len = req.input_ids.len() - req.cached_len;
        let pages_per_seq = seq_len.div_ceil(self.local_pool.page_size);

        // Local layers: allocate min(pages_per_seq, window_pages) pages.
        let window_pages = self.window_size.div_ceil(self.local_pool.page_size);
        let local_needed = pages_per_seq.min(window_pages);

        // Global layers: allocate pages for the full sequence.
        let global_needed = pages_per_seq;

        let mut local_pages = Vec::with_capacity(local_needed);
        let mut global_pages = Vec::with_capacity(global_needed);

        for _ in 0..local_needed {
            match self.local_pool.allocate() {
                Some(p) => local_pages.push(p),
                None => return Err(EngineError::OutOfKvPages),
            }
        }

        for _ in 0..global_needed {
            match self.global_pool.allocate() {
                Some(p) => global_pages.push(p),
                None => {
                    // Try to offload the oldest global pages to SSD.
                    self.kv_offload.offload_lru(&mut self.global_pool);
                    match self.global_pool.allocate() {
                        Some(p) => global_pages.push(p),
                        None => return Err(EngineError::OutOfKvPages),
                    }
                }
            }
        }

        Ok((local_pages, global_pages))
    }
}
```

---
## 7. Prefill-Decode Disaggregation

### 7.1 Background

Splitwise, DistServe, and Mooncake have demonstrated that separating prefill (compute-bound) from decode (memory-bandwidth-bound) can meaningfully improve throughput and SLO attainment in datacenter settings. The key insight:

- **Prefill** is dominated by GEMMs with large matrices (high arithmetic intensity). It benefits from high FLOP/s.
- **Decode** is dominated by GEMVs — one row of a weight matrix per token, per layer — giving ~1 FLOP per byte of memory moved. It is limited by memory bandwidth.

On datacenter hardware, this means separate GPU types (A100 for prefill, T4 for decode). On edge hardware, the analogous distinction is:

| Phase | Best hardware (macOS) | Best hardware (Linux) |
|-------|----------------------|----------------------|
| Prefill | GPU (Metal) or ANE | GPU (Intel Arc) |
| Decode | GPU (Metal) | GPU (Intel Arc) |

For single-machine edge deployment in 2026, the primary benefit of disaggregation is **not** different hardware but **different scheduling policies**:
- Prefill workers can take long-running prefill requests without starving decode.
- Decode workers have predictable, bounded latency per iteration (no jitter from long prefills).

### 7.2 Single-Machine Architecture

We implement disaggregation as two **Tokio tasks** (or OS threads) sharing the same process and GPU context:

```
┌──────────────────────────────────────────────────────────┐
│                    burn-inference process                 │
│                                                           │
│  ┌───────────────────────┐  ┌────────────────────────┐   │
│  │   Prefill Worker Task │  │  Decode Worker Task    │   │
│  │                       │  │                        │   │
│  │  - Admits new reqs    │  │  - Runs decode loop    │   │
│  │  - Runs chunked       │  │  - Continuous batching │   │
│  │    prefill batches    │  │  - Low per-iter jitter │   │
│  │  - On completion,     │  │                        │   │
│  │    transfers KV via   │  │                        │   │
│  │    shared ring buffer │  │                        │   │
│  └──────────┬────────────┘  └──────────┬─────────────┘   │
│             │ KV transfer               │                  │
│             │ (shared memory)           │                  │
│  ┌──────────▼───────────────────────────▼──────────────┐  │
│  │         Shared KV Cache Ring Buffer                  │  │
│  │  (lock-free SPSC queue of completed-prefill KV slots)│  │
│  └──────────────────────────────────────────────────────┘  │
│                                                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         GPU (shared between workers)                 │  │
│  │  Workers take turns via a mutex/semaphore.           │  │
│  │  Prefill gets priority for large batches;            │  │
│  │  Decode gets priority for latency SLO.              │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### 7.3 KV Transfer Ring Buffer

When a request completes prefill, its KV pages must be visible to the decode worker. Since both workers share the same process and GPU memory, "transfer" is essentially a pointer handoff:

```rust
/// Message sent from prefill worker to decode worker after prefill completes.
pub struct PrefillComplete {
    pub request: Arc<parking_lot::Mutex<Request>>,
    /// The KV pages computed during prefill, already in the shared KV pool.
    pub kv_pages: Vec<PageIndex>,
    /// The first decode token (sampled from the prefill's last-position logits).
    pub first_decode_token: u32,
}

/// Lock-free SPSC queue for KV handoff.
/// Uses a fixed-size ring buffer to avoid allocation on the hot path.
pub struct KvHandoffQueue {
    inner: crossbeam_queue::ArrayQueue<PrefillComplete>,
}

impl KvHandoffQueue {
    pub fn new(capacity: usize) -> Self {
        Self { inner: crossbeam_queue::ArrayQueue::new(capacity) }
    }

    pub fn push(&self, item: PrefillComplete) -> Result<(), PrefillComplete> {
        self.inner.push(item).map_err(|e| e)
    }

    pub fn pop(&self) -> Option<PrefillComplete> {
        self.inner.pop()
    }
}
```

The decode worker polls this queue at the start of each scheduling iteration and admits newly-completed prefill requests.

### 7.4 GPU Time Sharing

Both workers use the same `BackendHandle`. We protect it with a `tokio::sync::Mutex`:

```rust
pub struct SharedGpuBackend {
    inner: tokio::sync::Mutex<Box<dyn BackendHandle>>,
}

impl SharedGpuBackend {
    /// Acquire the GPU for one forward pass.
    /// The prefill worker holds the lock for the duration of a prefill batch;
    /// the decode worker holds it for decode batches.
    pub async fn forward(&self, batch: &Batch, ctx: &EngineContext) -> Result<Logits, EngineError> {
        let mut backend = self.inner.lock().await;
        backend.forward(batch, ctx).await
    }
}
```

In Phase 1 this is a simple mutex. In Phase 2 we can implement priority: decode has a shorter lock-acquisition deadline; if the prefill worker holds the lock for too long, the decode worker pre-empts it by signaling the backend to yield the GPU command queue.

### 7.5 ANE Integration (Future: Phase 3+)

Apple Neural Engine specifics:
- M4 ANE delivers ~38 TOPS in FP16.
- Access from Rust: via CoreML FFI or `subprocess` invoking a Swift helper that loads a `.mlpackage`.
- **Fixed sequence lengths only**: CoreML models must be compiled with a specific input shape. We bucket prefill lengths to {128, 256, 512, 1024} and pad.
- Model restructuring required: `nn.Linear(in, out)` → `nn.Conv2d(in, out, kernel=1)` with channels-first layout.

The ANE path is architecturally isolated behind a `PrefillBackend` trait:

```rust
#[async_trait::async_trait]
pub trait PrefillBackend: Send + Sync {
    async fn prefill(
        &mut self,
        tokens: &[u32],   // padded to a bucket boundary
        mask: &[bool],    // true = real token, false = padding
    ) -> Result<PrefillOutput, EngineError>;
}

pub struct PrefillOutput {
    /// KV cache entries computed for each real token, all layers.
    /// Shape: [num_real_tokens, 2, num_layers, num_kv_heads, head_dim]
    pub kv_entries: Vec<f16>,
    /// Logits for the last real token.
    pub last_logits: Vec<f32>,
}
```

Practical verdict: not in Phase 1 or Phase 2. The CoreML integration complexity and the model restructuring effort are high. This is a Phase 3+ item. The architecture supports it via `PrefillBackend` trait dispatch.

### 7.6 Intel NPU (Linux) — Not Recommended

Intel NPU (Meteor Lake/Lunar Lake `intel_vpu.ko`, Level Zero API) as of Q1 2026:
- **INT4 only**: weights must be pre-quantized to INT4; mixed precision not supported.
- **Static shapes only**: no dynamic batching, no variable sequence lengths.
- **Limited model support**: SmolLM, Qwen2.5 only in the Intel OpenVINO acceleration library.
- **Archived**: the `intel-npu-acceleration-library` GitHub repo was archived in late 2025.
- **Linux driver stability**: requires Ubuntu 22.04 or 24.04; kernel 6.x; not stable on other distros.

**Recommendation**: Do not target Intel NPU in any planned phase. The Intel iGPU (Arc, Vulkan, `burn-wgpu`) is a much better target and is already integrated.

### 7.7 Practical Recommendation for Phase 2

Implement the disaggregated architecture (two workers, handoff queue) but use the **same backend** for both phases initially. This gives us:
1. Predictable decode latency (no long prefills blocking decode iterations).
2. Clean separation that makes future NPU/ANE integration a matter of implementing a new `PrefillBackend`.
3. Minimal added complexity in Phase 2.

The GPU-time-sharing mutex can be replaced by a smarter scheduler (e.g., credit-based) in Phase 3 if decode latency SLOs require it.

---
## 8. FlashMoE Integration

### 8.1 Background

Mixture-of-Experts (MoE) models activate a small subset of experts per token. For a 26B MoE with 64 experts and top-2 routing, only ~6B parameters are active per token. However, all 64 experts must be accessible, so the full ~26B parameter model must be stored somewhere. On a 16 GB device, most expert weights live on SSD and are streamed on demand.

The `ExpertWeightCache` in `burn-ggml`/`burn-wgpu` manages this streaming. `burn-inference` coordinates with it by:
1. Running the router **before** the FFN layers.
2. Prefetching the selected experts while the attention layers execute.
3. Grouping tokens by expert assignment to maximize MUL_MAT_ID throughput.

### 8.2 Routing Before Prefetch

In a standard transformer forward pass, the router runs inside the FFN block. For expert streaming, we need to know which experts are needed **before** we start the FFN. We restructure the forward pass as:

```
Layer i:
  1. Attention (uses KV from pool)
  2. Router (lightweight: linear + softmax + top-k)  ← moved forward
  3. FIRE prefetch for top-k experts needed in layer i FFN
  4. (other layers execute: attention for layers i+1, i+2, ...)
  5. Expert FFN for layer i (experts now in cache)
```

The engine communicates routing decisions to the backend via `ExpertRouting` in the `Batch` struct:

```rust
/// Called by the engine before the full forward pass.
/// Runs only the attention + router sublayers to get routing decisions.
/// These are then used to prefetch expert weights.
pub async fn compute_routing(
    backend: &mut dyn BackendHandle,
    batch: &mut Batch,
    ctx: &EngineContext,
) -> Result<(), EngineError> {
    let routing = backend.compute_router_only(batch, ctx).await?;
    batch.expert_routing = Some(routing);
    Ok(())
}
```

### 8.3 Expert Prefetch Schedule

```rust
/// Given routing decisions for a batch, issue prefetch requests to the
/// ExpertWeightCache for all experts needed in the batch.
/// Called while the attention layers are executing on the GPU.
pub async fn prefetch_experts(
    backend: &mut dyn BackendHandle,
    routing: &ExpertRouting,
    num_layers: usize,
) -> Result<(), EngineError> {
    // Each layer needs the same set of unique experts (routing is shared
    // across layers in most MoE designs).
    for layer_idx in 0..num_layers {
        backend.prefetch_expert_layer(layer_idx, &routing.unique_expert_ids).await?;
    }
    Ok(())
}
```

The backend's `prefetch_expert_layer` is non-blocking: it queues SSD reads for the expert weight tensors and returns immediately. The actual data movement happens concurrently with GPU computation.

### 8.4 MoE-Aware Batching

Grouping tokens by expert assignment before dispatch maximizes contiguous memory access in the `MUL_MAT_ID` kernel (the GGML operation that runs all selected experts in one fused call):

```rust
/// Reorder tokens in the batch so that tokens assigned to the same expert
/// are contiguous. This maximizes cache locality in the expert FFN kernels.
pub fn reorder_by_expert(
    batch: &mut Batch,
    routing: &ExpertRouting,
    top_k: usize,
) {
    // Build (expert_id, token_idx) pairs, sorted by expert_id.
    let mut assignments: Vec<(u32, usize)> = routing
        .token_expert_ids
        .iter()
        .enumerate()
        .flat_map(|(tok_idx, experts)| {
            experts.iter().map(move |&exp_id| (exp_id, tok_idx))
        })
        .collect();
    assignments.sort_unstable_by_key(|&(exp_id, _)| exp_id);

    // Reorder input_ids and position_ids accordingly.
    // The attention pass already completed, so this only affects the FFN pass.
    let orig_input = batch.input_ids.clone();
    let orig_pos = batch.position_ids.clone();
    let order: Vec<usize> = assignments.iter().map(|&(_, tok_idx)| tok_idx).collect();

    batch.input_ids = order.iter().map(|&i| orig_input[i]).collect();
    batch.position_ids = order.iter().map(|&i| orig_pos[i]).collect();
    // Store the permutation so we can un-permute outputs.
    // (stored in Batch metadata, not shown for brevity)
}
```

### 8.5 Expert Cache Warming

When a request is **admitted** (moved from Waiting to Prefilling), we can pre-warm the expert cache based on the first few tokens of the prompt. The idea: route the first 32 tokens through the router cheaply (CPU-side inference of the router MLP, which is small), identify the top-10 most-used experts, and prefetch them before the request's first forward pass.

```rust
pub async fn warm_expert_cache_on_admission(
    backend: &mut dyn BackendHandle,
    req: &Request,
    warm_tokens: usize,
) -> Result<(), EngineError> {
    let warm_ids = &req.input_ids[..warm_tokens.min(req.input_ids.len())];
    // Run router cheaply (router weights are small, always in cache).
    let approx_routing = backend.quick_route(warm_ids).await?;
    backend.prefetch_expert_layer(0, &approx_routing.unique_expert_ids).await?;
    Ok(())
}
```

### 8.6 Interaction with Radix Cache

For MoE models, the radix cache stores KV entries as usual, but expert activations are NOT cached (they are recomputed on each use). This is by design: expert weights are large, and caching the intermediate activations would cost more memory than it saves. Only KV cache (attention heads) benefits from the radix cache.

---
## 9. Compute-Data Movement Overlap

### 9.1 Overview

`PrefetchOps` (from the `burn-ggml`/`burn-wgpu` backend design) provides an async interface to queue disk reads and weight copies before they are needed. The engine exploits this by running a **double-buffered batch pipeline**:

```
Time:   ──────────────────────────────────────────────────────────────►

CPU:    [ schedule(B1) ]  [ schedule(B2) ]  [ schedule(B3) ]
         prefetch(B1)      prefetch(B2)      prefetch(B3)

GPU:                      [ forward(B1) ]  [ forward(B2) ]  [ forward(B3) ]

I/O:    [====prefetch B1====]  [=====prefetch B2=====]  [=====prefetch B3=====]
```

The key insight: I/O for batch N+1 is in flight while the GPU executes batch N. By the time the GPU finishes batch N, batch N+1's weights and KV pages are already in GPU memory.

### 9.2 `PrefetchOps` Interface

```rust
/// Issued by the engine to the backend to start prefetching for the next batch.
/// The call is non-blocking; data moves asynchronously.
#[async_trait::async_trait]
pub trait PrefetchOps: Send + Sync {
    /// Prefetch the weight tensors for the layers that will be used in `batch`.
    /// For MoE: also prefetch the selected expert weights.
    async fn prefetch_weights(&mut self, batch: &Batch) -> Result<(), EngineError>;

    /// Prefetch the KV cache pages needed by `batch` from SSD (if offloaded).
    async fn prefetch_kv_pages(&mut self, pages: &[PageIndex]) -> Result<(), EngineError>;

    /// Block until all previously issued prefetch operations are complete.
    /// Called just before `forward()`.
    async fn wait_prefetch(&mut self) -> Result<(), EngineError>;
}
```

### 9.3 Engine Integration

The engine calls `PrefetchOps` as part of the overlap scheduling loop described in §4.6:

```rust
pub async fn run_with_overlap(
    mut ctx: EngineContext,
    mut request_rx: mpsc::Receiver<Arc<parking_lot::Mutex<Request>>>,
) {
    let mut current_batch: Option<Batch> = None;

    loop {
        // Drain incoming requests into the waiting queue.
        while let Ok(req) = request_rx.try_recv() {
            ctx.waiting.push_back(req);
        }

        // ── Phase 1: Issue prefetch for the NEXT batch (non-blocking) ────
        // We do this BEFORE waiting for the current GPU result so that
        // I/O runs concurrently with the GPU.
        let next_batch = schedule_batch(&mut ctx);

        if let Some(ref nb) = next_batch {
            // Fire weight prefetches for next batch (non-blocking).
            let _ = ctx.backend.prefetch_weights(nb).await;

            // Fire KV page prefetches for any offloaded pages.
            let offloaded_pages = collect_offloaded_pages(nb, &ctx.kv_pool);
            if !offloaded_pages.is_empty() {
                let _ = ctx.backend.prefetch_kv_pages(&offloaded_pages).await;
            }

            // For MoE: fire expert weight prefetches.
            if let Some(ref routing) = nb.expert_routing {
                let _ = prefetch_experts(ctx.backend.as_mut(), routing, ctx.config.model_config.num_layers).await;
            }
        }

        // ── Phase 2: Collect results from the CURRENT GPU batch ──────────
        if let Some(batch) = current_batch.take() {
            // Wait for the GPU to finish. Because prefetches are already
            // in flight, this wait is (ideally) very short.
            let logits = ctx.backend.wait_current_forward().await
                .expect("GPU forward pass failed");
            process_logits(&mut ctx, &batch, logits).await;
            finalize_prefill_requests(&mut ctx, &batch);
        }

        // ── Phase 3: Wait for prefetches and launch the next batch ────────
        if let Some(nb) = next_batch {
            // Block until all I/O is ready.
            ctx.backend.wait_prefetch().await
                .expect("Prefetch wait failed");

            // Launch forward pass (non-blocking: submits GPU commands).
            ctx.backend.launch_forward(&nb, &ctx).await
                .expect("GPU launch failed");

            current_batch = Some(nb);
        } else {
            // Nothing to schedule. Block until a new request arrives.
            // Zero CPU usage while idle.
            if let Some(req) = sched_rx.recv().await {
                ctx.waiting.push_back(req);
            }
        }
    }
}

fn collect_offloaded_pages(batch: &Batch, pool: &KvCachePool) -> Vec<PageIndex> {
    batch
        .page_table
        .iter()
        .flatten()
        .filter(|&&page| matches!(pool.page_states[page as usize], PageState::Offloaded { .. }))
        .copied()
        .collect()
}
```

### 9.4 Double-Buffering at Batch Level

The loop above implements double-buffering:
- **Slot A**: the batch currently executing on the GPU.
- **Slot B**: the batch being scheduled and prefetched on the CPU.

When A finishes, B slides into A's slot and a new B is scheduled. This ensures the GPU is never idle waiting for I/O or CPU scheduling work to complete.

Timing analysis for a 26B MoE on Intel Arc (Linux):
- GPU forward pass (decode batch of 32): ~80ms
- CPU scheduling (schedule_batch): ~0.5ms
- I/O prefetch (4 experts, 512 MB total): ~40ms at 10 GB/s NVMe
- With overlap: effective GPU utilization ~99% (I/O hidden behind GPU compute)
- Without overlap: effective GPU utilization ~67% (GPU waits 40ms per 80ms GPU cycle)

### 9.5 Layer-Wise Weight Streaming

For the 31B dense model on macOS (18 GB weights, 32 GB unified memory), all weights fit in RAM. But for systems with 16 GB unified memory, weights must be streamed layer by layer from SSD:

```
Iteration timeline (layer-wise streaming, 64 layers):

Time: ──────────────────────────────────────────────────────►
GPU:  [attn_0][ffn_0][attn_1][ffn_1]...[attn_63][ffn_63]
I/O:  [load_1][load_2][load_3][load_4]...
         └────►loaded before GPU reaches layer 1
```

`WeightCache` in the backend handles this. The engine's role is to call `prefetch_weights` with enough lead time (at least 2 layers ahead) that the GPU never stalls waiting for weights.

---
## 10. API Layer

### 10.1 Overview

The API layer is an `axum` HTTP server implementing the OpenAI-compatible REST API. It handles:
- POST `/v1/chat/completions`
- POST `/v1/completions`
- GET `/v1/models`
- GET `/health`

Streaming responses use Server-Sent Events (SSE). All request/response serialization uses `serde_json`.

### 10.2 Dependencies

```toml
[dependencies]
axum = { version = "0.7", features = ["macros"] }
tokio = { version = "1", features = ["full"] }
tower = "0.4"
tower-http = { version = "0.5", features = ["cors", "trace"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
uuid = { version = "1", features = ["v4"] }
tokenizers = "0.19"
async-stream = "0.3"
futures = "0.3"
```

### 10.3 Request / Response Types

```rust
use serde::{Deserialize, Serialize};

// ── Chat Completions ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Vec<String>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
}

fn default_max_tokens() -> usize { 512 }
fn default_temperature() -> f32 { 1.0 }
fn default_top_p() -> f32 { 1.0 }

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,  // "system" | "user" | "assistant"
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: UsageStats,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Default)]
pub struct UsageStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

// ── Streaming (SSE) ──────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: DeltaContent,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DeltaContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}
```

### 10.4 HTTP Handlers

```rust
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use std::sync::Arc;
use tokio::sync::mpsc;

pub type AppState = Arc<InferenceServer>;

pub struct InferenceServer {
    pub engine_tx: mpsc::Sender<Arc<parking_lot::Mutex<Request>>>,
    pub tokenizer: Arc<tokenizers::Tokenizer>,
    pub model_name: String,
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/completions", post(completions_handler))
        .route("/v1/models", get(list_models_handler))
        .route("/health", get(health_handler))
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .with_state(state)
}

async fn health_handler() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

async fn list_models_handler(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "created": 0,
            "owned_by": "burn-inference"
        }]
    }))
}

async fn chat_completions_handler(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    // Apply chat template to convert messages to a single token sequence.
    let prompt = apply_chat_template(&req.messages);

    // Tokenize.
    let encoding = state.tokenizer.encode(&prompt[..], false)
        .expect("tokenizer failed");
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();

    let sampling_params = SamplingParams {
        max_new_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k.unwrap_or(0),
        repetition_penalty: req.repetition_penalty.unwrap_or(1.0),
        stop_sequences: req.stop,
        stream: req.stream,
    };

    if req.stream {
        // Streaming response: SSE.
        let (token_tx, token_rx) = mpsc::channel(256);
        let request = Arc::new(parking_lot::Mutex::new(Request {
            id: uuid::Uuid::new_v4(),
            input_ids,
            sampling_params,
            state: RequestState::Waiting,
            cached_len: 0,
            device_len: 0,
            extend_len: 0,
            kv_pages: vec![],
            output_ids: vec![],
            token_tx: Some(token_tx),
            arrival_time: std::time::Instant::now(),
            first_token_time: None,
        }));

        let _ = state.engine_tx.send(request).await;

        let tokenizer = state.tokenizer.clone();
        let model_name = state.model_name.clone();

        let stream = async_stream::stream! {
            let mut rx = token_rx;
            let req_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
            let created = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            // First chunk: role delta.
            let first_chunk = ChatCompletionChunk {
                id: req_id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: DeltaContent { role: Some("assistant".to_string()), content: None },
                    finish_reason: None,
                }],
            };
            yield Ok::<_, std::convert::Infallible>(
                axum::response::sse::Event::default()
                    .data(serde_json::to_string(&first_chunk).unwrap())
            );

            while let Some(event) = rx.recv().await {
                match event {
                    TokenEvent::Token(token_id) => {
                        let text = tokenizer
                            .decode(&[token_id], true)
                            .unwrap_or_default();
                        let chunk = ChatCompletionChunk {
                            id: req_id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model_name.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: DeltaContent { role: None, content: Some(text) },
                                finish_reason: None,
                            }],
                        };
                        yield Ok(
                            axum::response::sse::Event::default()
                                .data(serde_json::to_string(&chunk).unwrap())
                        );
                    }
                    TokenEvent::Done { finish_reason } => {
                        let finish_str = match finish_reason {
                            FinishReason::EosToken | FinishReason::StopString => "stop",
                            FinishReason::MaxTokens => "length",
                            FinishReason::Aborted => "stop",
                        };
                        let chunk = ChatCompletionChunk {
                            id: req_id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model_name.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: DeltaContent { role: None, content: None },
                                finish_reason: Some(finish_str.to_string()),
                            }],
                        };
                        yield Ok(
                            axum::response::sse::Event::default()
                                .data(serde_json::to_string(&chunk).unwrap())
                        );
                        // OpenAI protocol requires a final "[DONE]" SSE event.
                        yield Ok(
                            axum::response::sse::Event::default().data("[DONE]")
                        );
                        break;
                    }
                    TokenEvent::Error(e) => {
                        yield Ok(
                            axum::response::sse::Event::default()
                                .data(format!("{{\"error\": \"{}\"}}", e))
                        );
                        break;
                    }
                }
            }
        };

        Sse::new(stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response()
    } else {
        // Non-streaming: collect all tokens, return full response.
        let (token_tx, mut token_rx) = mpsc::channel(256);
        let request = Arc::new(parking_lot::Mutex::new(Request {
            id: uuid::Uuid::new_v4(),
            input_ids: input_ids.clone(),
            sampling_params,
            state: RequestState::Waiting,
            cached_len: 0,
            device_len: 0,
            extend_len: 0,
            kv_pages: vec![],
            output_ids: vec![],
            token_tx: Some(token_tx),
            arrival_time: std::time::Instant::now(),
            first_token_time: None,
        }));

        let _ = state.engine_tx.send(request).await;

        let mut output_tokens = Vec::new();
        let mut finish_reason = "stop".to_string();

        while let Some(event) = token_rx.recv().await {
            match event {
                TokenEvent::Token(id) => output_tokens.push(id),
                TokenEvent::Done { finish_reason: fr } => {
                    finish_reason = match fr {
                        FinishReason::MaxTokens => "length".to_string(),
                        _ => "stop".to_string(),
                    };
                    break;
                }
                TokenEvent::Error(e) => {
                    return (StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": e}))).into_response();
                }
            }
        }

        let content = state.tokenizer.decode(&output_tokens, true).unwrap_or_default();

        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion",
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model: state.model_name.clone(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage { role: "assistant".to_string(), content },
                finish_reason,
            }],
            usage: UsageStats {
                prompt_tokens: input_ids.len(),
                completion_tokens: output_tokens.len(),
                total_tokens: input_ids.len() + output_tokens.len(),
            },
        };

        Json(response).into_response()
    }
}

fn apply_chat_template(messages: &[ChatMessage]) -> String {
    // Minimal Qwen-style chat template.
    // In production, load the template from the tokenizer config.
    let mut result = String::new();
    result.push_str("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");
    for msg in messages {
        result.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role, msg.content
        ));
    }
    result.push_str("<|im_start|>assistant\n");
    result
}
```

### 10.5 Server Entry Point

```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let (engine_tx, engine_rx) = mpsc::channel(1024);

    // Load tokenizer.
    let tokenizer = Arc::new(
        tokenizers::Tokenizer::from_pretrained("Qwen/Qwen3-32B-Instruct", None)?
    );

    // Initialize engine context and start engine loop.
    let engine_ctx = EngineContext::new(EngineConfig::default())?;
    tokio::spawn(engine_loop(engine_ctx, engine_rx));

    let state = Arc::new(InferenceServer {
        engine_tx,
        tokenizer,
        model_name: "qwen3-32b-instruct".to_string(),
    });

    let app = build_router(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await?;
    tracing::info!("Listening on http://0.0.0.0:8000");
    axum::serve(listener, app).await?;

    Ok(())
}
```

---
## 11. Tokenizer Service

### 11.1 Design

The tokenizer service is a dedicated `async` Tokio task that handles all tokenization and detokenization. This mirrors mini-sglang's pattern of a separate tokenizer process (using ZMQ), but since we are in a single Rust process, we use `tokio::sync::mpsc` channels instead.

The separation is important because:
1. Tokenization can be slow for long prompts (Unicode normalization, BPE encoding).
2. Detokenization of streaming tokens requires state (some tokenizers produce multi-byte UTF-8 sequences that span token boundaries).
3. The tokenizer holds a large vocabulary table; keeping it in one place avoids duplication.

### 11.2 The Tokenizer Task

```rust
use tokenizers::{Tokenizer, EncodeInput};
use tokio::sync::{mpsc, oneshot};

/// A request to the tokenizer service.
pub enum TokenizerRequest {
    /// Encode a string into token IDs.
    Encode {
        text: String,
        response_tx: oneshot::Sender<Vec<u32>>,
    },
    /// Decode a sequence of token IDs into a string.
    Decode {
        token_ids: Vec<u32>,
        skip_special_tokens: bool,
        response_tx: oneshot::Sender<String>,
    },
    /// Decode a single token ID incrementally, maintaining per-request state
    /// to handle multi-byte sequences.
    DecodeSingle {
        request_id: RequestId,
        token_id: u32,
        response_tx: oneshot::Sender<Option<String>>,
    },
    /// Clean up incremental decode state for a completed request.
    CleanupRequest {
        request_id: RequestId,
    },
}

pub struct TokenizerService {
    tokenizer: Tokenizer,
    /// Per-request incremental decoder state.
    /// Tracks the carry buffer for incomplete UTF-8 sequences.
    incremental_decoders: std::collections::HashMap<RequestId, IncrementalDecoder>,
}

struct IncrementalDecoder {
    pending_ids: Vec<u32>,
    byte_buffer: Vec<u8>,
}

impl TokenizerService {
    pub fn new(tokenizer: Tokenizer) -> Self {
        TokenizerService {
            tokenizer,
            incremental_decoders: std::collections::HashMap::new(),
        }
    }

    pub async fn run(mut self, mut rx: mpsc::Receiver<TokenizerRequest>) {
        while let Some(req) = rx.recv().await {
            match req {
                TokenizerRequest::Encode { text, response_tx } => {
                    let ids = self.encode(&text);
                    let _ = response_tx.send(ids);
                }
                TokenizerRequest::Decode { token_ids, skip_special_tokens, response_tx } => {
                    let text = self.tokenizer
                        .decode(&token_ids, skip_special_tokens)
                        .unwrap_or_default();
                    let _ = response_tx.send(text);
                }
                TokenizerRequest::DecodeSingle { request_id, token_id, response_tx } => {
                    let text = self.decode_single(request_id, token_id);
                    let _ = response_tx.send(text);
                }
                TokenizerRequest::CleanupRequest { request_id } => {
                    self.incremental_decoders.remove(&request_id);
                }
            }
        }
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer
            .encode(text, false)
            .map(|enc| enc.get_ids().to_vec())
            .unwrap_or_default()
    }

    /// Incrementally decode a single token ID.
    /// Returns Some(text) when a complete UTF-8 sequence is ready,
    /// None if we are still accumulating a multi-byte sequence.
    fn decode_single(&mut self, req_id: RequestId, token_id: u32) -> Option<String> {
        let decoder = self.incremental_decoders
            .entry(req_id)
            .or_insert_with(|| IncrementalDecoder {
                pending_ids: Vec::new(),
                byte_buffer: Vec::new(),
            });

        decoder.pending_ids.push(token_id);

        // Attempt to decode the accumulated IDs.
        let decoded = self.tokenizer
            .decode(&decoder.pending_ids, true)
            .unwrap_or_default();

        if decoded.is_empty() {
            // Token might be a BPE fragment; wait for more tokens.
            return None;
        }

        // Validate UTF-8; if incomplete, keep accumulating.
        // (The tokenizers crate usually handles this, but we double-check.)
        if std::str::from_utf8(decoded.as_bytes()).is_err() {
            return None;
        }

        // Clear the pending buffer; output is ready.
        decoder.pending_ids.clear();
        Some(decoded)
    }
}
```

### 11.3 Client Handle

The HTTP handler and the engine use a `TokenizerHandle` to communicate with the tokenizer service:

```rust
#[derive(Clone)]
pub struct TokenizerHandle {
    tx: mpsc::Sender<TokenizerRequest>,
}

impl TokenizerHandle {
    pub fn new(tx: mpsc::Sender<TokenizerRequest>) -> Self {
        TokenizerHandle { tx }
    }

    pub async fn encode(&self, text: &str) -> Vec<u32> {
        let (resp_tx, resp_rx) = oneshot::channel();
        let _ = self.tx.send(TokenizerRequest::Encode {
            text: text.to_string(),
            response_tx: resp_tx,
        }).await;
        resp_rx.await.unwrap_or_default()
    }

    pub async fn decode(&self, ids: Vec<u32>, skip_special: bool) -> String {
        let (resp_tx, resp_rx) = oneshot::channel();
        let _ = self.tx.send(TokenizerRequest::Decode {
            token_ids: ids,
            skip_special_tokens: skip_special,
            response_tx: resp_tx,
        }).await;
        resp_rx.await.unwrap_or_default()
    }

    pub async fn decode_single(
        &self,
        req_id: RequestId,
        token_id: u32,
    ) -> Option<String> {
        let (resp_tx, resp_rx) = oneshot::channel();
        let _ = self.tx.send(TokenizerRequest::DecodeSingle {
            request_id: req_id,
            token_id,
            response_tx: resp_tx,
        }).await;
        resp_rx.await.ok().flatten()
    }
}
```

### 11.4 Startup

```rust
pub fn start_tokenizer_service(
    tokenizer_path: &str,
) -> anyhow::Result<TokenizerHandle> {
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    let (tx, rx) = mpsc::channel(1024);
    let service = TokenizerService::new(tokenizer);

    tokio::spawn(service.run(rx));

    Ok(TokenizerHandle::new(tx))
}
```

---
## 12. Implementation Plan

The implementation is structured in four phases. **Phase 0** validates the engine orchestration logic in isolation using the stock `burn-wgpu` backend with a small model — no custom backend, no offloading, no FFI. Only once the scheduler, radix cache, continuous batching, and API are proven correct does the project layer in offloading complexity (Phase 1) and then the ggml/Metal backend (Phase 2).

```
Phase 0 (burn-wgpu, small model, no offload)
  └─ Validate: scheduler, radix cache, continuous batching, API, correctness tests
       ↓ BackendHandle trait is the only seam that changes
Phase 1 (burn-wgpu + offload, 26B MoE, Linux)
  └─ Validate: WeightCache, ExpertCache, KvOffload, PrefetchOps, FlashMoE
       ↓ Same scheduler, same API, same radix cache — only backend changes
Phase 2A (burn-ggml + Metal, 31B dense, macOS)
  └─ Validate: ggml FFI, quantized matmul, layer streaming, Metal KV offload
Phase 2B (disaggregation, API polish, benchmarks)
```

The `BackendHandle` trait is the critical abstraction boundary. Everything above it (scheduler, radix cache, API) is written once in Phase 0 and never touched again when switching backends.

---

### Phase 0 — Weeks 1–6: Engine Validation on burn-wgpu (Small Model)

**Goal:** A fully working inference engine — continuous batching, radix cache, chunked prefill, overlap scheduling, HTTP API, streaming responses — running on `burn-wgpu` with a small model (Gemma 3 1B or 4B, or Qwen2.5-1.5B) that fits entirely in GPU/iGPU memory. No custom backend, no SSD offloading, no FFI.

**Why this first:** The scheduler, radix cache, and API layer are ~80% of the total code. Validating them against a simple backend eliminates the largest class of bugs before adding offload complexity. If the Beijing/1+1 correctness tests fail in Phase 0, the problem is definitively in the engine, not the backend.

**Model choice:** Gemma 3 1B–4B in BF16/F16. Fits in 4–8 GB VRAM. Fast enough to iterate in seconds. Close enough to Gemma 4's architecture that the model definition transfers with minimal changes.

#### P0-T1 — Crate structure and `BackendHandle` trait

- [ ] Create `burn-inference` workspace with sub-crates:
  - `inference-engine/` — scheduler, request lifecycle, KV pool, radix cache
  - `inference-api/` — HTTP server (axum), tokenizer service
  - `inference-backend/` — `BackendHandle` trait + `StubBackend` for unit tests
- [ ] Define `BackendHandle` trait (the only seam between engine and backend):
  ```rust
  pub trait BackendHandle: Send + Sync + 'static {
      fn forward(&self, batch: &Batch) -> impl Future<Output = ForwardOutput> + Send;
      fn kv_pool(&self) -> &dyn KvPool;
      fn model_config(&self) -> &ModelConfig;
  }
  ```
- [ ] Implement `StubBackend`: returns random logits, configurable latency — used for all unit tests
- [ ] Verify: `cargo build` and `cargo test` pass with stub backend

#### P0-T2 — Core data structures and scheduler skeleton

- [ ] Implement `SamplingParams`, `Request`, `RequestState` state machine
- [ ] Implement `Batch`, `BatchPhase` (Prefill / Decode)
- [ ] Implement `KvCachePool`: fixed-size page allocator (free list, `allocate`, `free`)
- [ ] Implement `PageTable`: `(RequestId, seq_pos) → PageIndex`
- [ ] Implement `EngineContext`: waiting / prefilling / decoding queues
- [ ] Implement `PrefillAdder` greedy algorithm with `max_prefill_tokens` budget
- [ ] Implement `collect_decode_requests`
- [ ] Implement `assemble_batch` and `process_logits` (greedy argmax only)
- [ ] Integration test: 10 concurrent requests, 128-token prompts, 64 max_new_tokens — all complete without deadlock

#### P0-T3 — Radix cache

- [ ] Implement `RadixNode` and `RadixCache`: `match_prefix`, `insert_prefix`, `split_edge`, `evict_pages`
- [ ] Implement `lock_nodes` / `unlock_nodes` ref-count lifecycle
- [ ] Implement `promote_to_cache`: called when prefill completes for a request
- [ ] Wire radix cache into `schedule_batch`: call `match_prefix` on admission, set `cached_len`
- [ ] Unit tests:
  - [ ] `test_radix_miss`: new request, no cache hit, `cached_len == 0`
  - [ ] `test_radix_hit`: two requests sharing 512-token system prompt; second gets `cached_len == 512`
  - [ ] `test_radix_eviction`: fill cache to capacity; verify LRU eviction, no use-after-free
  - [ ] `test_radix_partial_match`: requests sharing first 256 of 512 tokens; verify split edge

#### P0-T4 — Chunked prefill and overlap scheduling

- [ ] Implement chunked prefill: requests with `input_ids.len() > max_prefill_tokens_per_iter` span multiple iterations
- [ ] Update `PrefillAdder` to track cross-iteration prefill state (`ChunkedPrefillState`)
- [ ] Implement `run_overlapped_loop` (Tokio double-buffer pipeline):
  - Schedule next batch (CPU) while previous batch executes on GPU (async)
  - Use `tokio::select!` to multiplex incoming requests and forward-pass completion
- [ ] Implement `preempt_one_prefill_request` (evict lowest-priority in-flight prefill)
- [ ] Implement `temperature` + `top_p` + `top_k` sampling, stop sequences
- [ ] Unit tests:
  - [ ] `test_chunked_prefill`: 4096-token prompt, 1024-token budget → 4 iterations
  - [ ] `test_overlap_scheduling`: mock timing verifies CPU scheduling overlaps GPU execution

#### P0-T5 — burn-wgpu backend integration (small model)

- [ ] Implement `WgpuBackendHandle` (no offload variant): wraps `burn-wgpu` forward pass
- [ ] Implement model definition for Gemma 3 1B/4B (or Qwen2.5-1.5B) using Burn modules
- [ ] Implement GGUF/safetensors weight loader for the small model
- [ ] Wire paged attention metadata into the wgpu attention op
- [ ] No `WeightCache`, no `KvOffloadManager`, no `PrefetchOps` — all weights resident in GPU memory

#### P0-T6 — Tokenizer service and HTTP API

- [ ] Implement `TokenizerService` async Tokio task using `tokenizers` crate
- [ ] Implement axum HTTP server: `POST /v1/chat/completions` (streaming SSE + non-streaming)
- [ ] Implement per-request `mpsc::Sender<Token>` for streaming token delivery
- [ ] Implement graceful shutdown: drain in-flight requests before stopping

#### P0-T7 — Correctness validation

- [ ] Run on Linux with Intel iGPU (burn-wgpu/Vulkan):
  - [ ] `"What is the capital of China?"` → `output.to_lowercase().contains("beijing")`
  - [ ] `"What is 1+1? Answer with only a number."` → `output.trim() == "2"`
- [ ] Run same tests on macOS with burn-wgpu/Metal (same backend, different platform)
- [ ] Concurrent correctness: 8 simultaneous requests with the two prompts above; all pass
- [ ] Radix cache correctness: send the same prompt twice; second request gets cache hit, output is identical to first

**Milestone 0:** Both correctness tests pass on Linux (Vulkan) and macOS (Metal) with a small model. Radix cache hit rate >80% on repeated-prompt benchmark. Chunked prefill verified correct. All `cargo test` pass. The engine orchestration is proven correct before any offloading complexity is introduced.

---

### Phase 1 — Weeks 7–14: Offloading on burn-wgpu (26B MoE, Linux)

**Goal:** Add the full offloading stack — `ExpertWeightCache`, `KvOffloadManager`, `PrefetchOps` — to the already-validated engine. Switch to Gemma 4 26B MoE Q4 as the benchmark model. Zero changes to the scheduler, radix cache, or API layer.

**Key insight:** Because `BackendHandle` is the only seam, the scheduler does not know or care whether the backend uses SSD offloading. The only new code in the engine layer is wiring `PrefetchOps` calls into the overlap loop (§9).

#### P1-T1 — Implement `WgpuOffloadBackendHandle`

- [ ] Add `WgpuOffloadDevice { ssd_path, max_expert_slots, max_kv_pages_on_ssd }` variant
- [ ] Integrate `WeightCache<ExpertKey>` (`ExpertWeightCache`) from burn-ggml design
- [ ] Integrate `KvOffloadManager` (ping-pong SSD buffers for global attention layers)
- [ ] Implement `PrefetchOps` for wgpu offload backend: fire async pread on `prefetch()` call

#### P1-T2 — MoE routing kernel and MUL_MAT_ID (WGSL)

- [ ] Write `shaders/moe_router.wgsl` (128-thread workgroup, shared-memory top-k, subgroup fast path)
- [ ] Write `shaders/mul_mat_id.wgsl` (tiled 16×16 grouped GEMM, port from ggml-vulkan reference)
- [ ] Unit test: `MUL_MAT_ID` output matches 8 separate matmuls for same weights

#### P1-T3 — Wire PrefetchOps into engine overlap loop

- [ ] After `schedule_batch`, call `backend.prefetch(next_batch.weight_hints)`
- [ ] After routing (MoE layers), call `backend.prefetch(expert_indices)`
- [ ] Verify via tracing spans that SSD reads overlap with GPU compute

#### P1-T4 — Gemma 4 26B MoE model definition

- [ ] Define `Gemma4MoeModel` Burn modules (30 layers, 128 experts, 8 active + 1 shared)
- [ ] Implement `GemmaRunner::decode_step` with explicit prefetch schedule
- [ ] Load from GGUF via `GgufIndex`

#### P1-T5 — End-to-end benchmark on Linux

- [ ] Both correctness tests pass with 26B MoE
- [ ] Benchmark: decode throughput at batch 1, 4, 8 with 26B MoE Q4
- [ ] Confirm expert cache hit rate >75% on typical text
- [ ] Confirm SSD compute overlap via tracing

**Milestone 1:** Gemma 4 26B MoE Q4 running on Linux 16 GB with expert streaming and KV offload. Both correctness tests pass. Decode throughput ≥ 5 tok/s at batch 1.

---

### Phase 2A — Weeks 15–18: burn-ggml + Metal (macOS, 31B Dense)

**Goal:** Swap `WgpuOffloadBackendHandle` for `GgmlBackendHandle`. Zero scheduler changes.

- [ ] Implement `GgmlBackendHandle`: wraps `burn-ggml` forward pass as `BackendHandle`
- [ ] Integrate `WeightCache<LayerKey>` (layer-wise streaming for 31B dense)
- [ ] Integrate `KvOffloadManager` for global attention layers
- [ ] Wire `PrefetchOps` with macOS `madvise`/`pread` async I/O
- [ ] Correctness tests on macOS (same two prompts)
- [ ] Benchmark: decode throughput at batch 1, 4, 8 with 31B Q3_K_M

**Milestone 2A:** 31B Q3_K_M running on macOS 16 GB. Both correctness tests pass. Decode throughput ≥ 10 tok/s at batch 1.

---

### Phase 2B — Weeks 19–22: Disaggregation, API Polish, Benchmarks

**Goal:** Prefill-decode disaggregation; production-grade API; full benchmark suite.

- [ ] Implement `PrefillWorker` and `DecodeWorker` Tokio tasks (§7)
- [ ] Implement `KvHandoffQueue`: lock-free SPSC ring buffer for KV page transfer
- [ ] Implement `SharedGpuBackend`: mutex-based GPU time sharing between workers
- [ ] API polish: CORS, request timeouts, graceful shutdown, `/v1/completions`, Prometheus `/metrics`
- [ ] Full benchmark suite: TTFT (cold/warm), throughput at 1/4/8/16/32 concurrency, shared-prefix benchmark, long-context benchmark
- [ ] Document: performance report comparing Phase 0 (small model) → Phase 1 (26B MoE) → Phase 2A (31B dense)

**Milestone 2B:** All performance targets in §14 achieved. Benchmark results published.
---
## 13. Correctness Tests

### 13.1 End-to-End Semantic Tests

These are the canonical correctness tests, run against a real model on real hardware.

```rust
#[tokio::test]
async fn test_capital_of_china() {
    let server = start_test_server().await;

    let response = server
        .chat_completion(ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "What is the capital of China?".to_string(),
            }],
            max_tokens: 64,
            temperature: 0.0,  // greedy; deterministic
            ..Default::default()
        })
        .await
        .expect("request failed");

    let content = response.choices[0].message.content.to_lowercase();
    assert!(
        content.contains("beijing"),
        "Expected 'beijing' in response, got: {:?}",
        content
    );
}

#[tokio::test]
async fn test_arithmetic_one_plus_one() {
    let server = start_test_server().await;

    let response = server
        .chat_completion(ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "What is 1+1? Answer with only a number.".to_string(),
            }],
            max_tokens: 4,
            temperature: 0.0,
            stop: vec!["\n".to_string()],
            ..Default::default()
        })
        .await
        .expect("request failed");

    let content = response.choices[0].message.content.trim();
    assert_eq!(content, "2", "Expected '2', got: {:?}", content);
}
```

### 13.2 Scheduler-Specific Tests

#### Concurrent Requests

```rust
#[tokio::test]
async fn test_concurrent_requests_do_not_corrupt_each_other() {
    let server = start_test_server_with_stub_backend().await;

    // Send 16 concurrent requests with distinct prompts.
    let handles: Vec<_> = (0..16)
        .map(|i| {
            let server = server.clone();
            tokio::spawn(async move {
                server.chat_completion(ChatCompletionRequest {
                    messages: vec![ChatMessage {
                        role: "user".to_string(),
                        content: format!("Request number {}", i),
                    }],
                    max_tokens: 32,
                    temperature: 0.0,
                    ..Default::default()
                }).await
            })
        })
        .collect();

    let results: Vec<_> = futures::future::join_all(handles).await;
    for (i, result) in results.into_iter().enumerate() {
        let resp = result.unwrap().expect(&format!("request {} failed", i));
        // Each response must be non-empty and not contain another request's content.
        assert!(!resp.choices[0].message.content.is_empty(),
            "Request {} returned empty response", i);
    }
}
```

#### Radix Cache Hit

```rust
#[tokio::test]
async fn test_radix_cache_hit_reduces_ttft() {
    let server = start_test_server_with_stub_backend().await;

    // First request: cold cache. Measure TTFT.
    let t0 = std::time::Instant::now();
    let _ = server.chat_completion(make_request_with_long_system_prompt(
        "What day is it?",
        SYSTEM_PROMPT_1024_TOKENS,
    )).await.unwrap();
    let cold_ttft = t0.elapsed();

    // Second request: same system prompt, different user message.
    // Should hit radix cache for the system prompt portion.
    let t1 = std::time::Instant::now();
    let _ = server.chat_completion(make_request_with_long_system_prompt(
        "What is your name?",
        SYSTEM_PROMPT_1024_TOKENS,
    )).await.unwrap();
    let warm_ttft = t1.elapsed();

    // With a stub backend that has linear cost in tokens processed,
    // warm TTFT should be significantly less than cold TTFT.
    assert!(
        warm_ttft < cold_ttft / 2,
        "Expected warm TTFT ({:?}) to be <50% of cold TTFT ({:?})",
        warm_ttft, cold_ttft
    );
}
```

#### Radix Cache Miss (Different Prefix)

```rust
#[tokio::test]
async fn test_radix_cache_miss_on_different_prefix() {
    let mut cache = RadixCache::new(16, 1024);

    // Insert a prefix.
    let tokens: Vec<u32> = (0..128).collect();
    let pages: Vec<PageIndex> = (0..8).collect(); // 8 pages * 16 tokens = 128 tokens
    cache.insert_prefix(&tokens, &pages);

    // Match with the exact same prefix: should hit.
    let (matched, returned_pages) = cache.match_prefix(&tokens);
    assert_eq!(matched, 128);
    assert_eq!(returned_pages, pages);

    // Match with a completely different prefix: should miss.
    let different: Vec<u32> = (1000..1128).collect();
    let (matched, returned_pages) = cache.match_prefix(&different);
    assert_eq!(matched, 0);
    assert!(returned_pages.is_empty());

    // Match with a partial overlap (first 64 tokens match): should get 64 tokens.
    let mut partial = tokens[..64].to_vec();
    partial.extend((200..264u32).collect::<Vec<_>>());
    let (matched, returned_pages) = cache.match_prefix(&partial);
    assert_eq!(matched, 64);
    assert_eq!(returned_pages, pages[..4]); // 4 pages * 16 = 64 tokens
}
```

#### Chunked Prefill Correctness

```rust
#[tokio::test]
async fn test_chunked_prefill_produces_correct_kv() {
    // Simulate a 4096-token prompt with a 1024-token budget.
    // The scheduler should process it in 4 chunks of 1024.
    // After all 4 chunks, the KV state must be identical to a
    // single-chunk prefill (if one were possible).

    let stub_backend = StubBackend::new_recording(); // records all batch calls
    let mut ctx = EngineContext::new_with_backend(stub_backend, EngineConfig {
        max_prefill_tokens_per_iter: 1024,
        ..Default::default()
    });

    let (resp_tx, mut resp_rx) = mpsc::channel(1);
    let (tok_tx, tok_rx) = mpsc::channel(256);
    let req = Arc::new(parking_lot::Mutex::new(Request {
        id: uuid::Uuid::new_v4(),
        input_ids: (0u32..4096).collect(),
        sampling_params: SamplingParams { max_new_tokens: 1, ..Default::default() },
        state: RequestState::Waiting,
        cached_len: 0,
        device_len: 0,
        extend_len: 0,
        kv_pages: vec![],
        output_ids: vec![],
        token_tx: Some(tok_tx),
        arrival_time: std::time::Instant::now(),
        first_token_time: None,
    }));

    ctx.waiting.push_back(req);

    // Run 5 scheduler iterations: 4 prefill + 1 decode.
    for _ in 0..5 {
        let batch = schedule_batch(&mut ctx).unwrap();
        let logits = ctx.backend.forward_sync(&batch, &ctx).unwrap();
        process_logits_sync(&mut ctx, &batch, logits);
    }

    // Verify: 4 prefill batches of 1024 tokens each were issued.
    let recorded = ctx.backend.downcast::<StubBackend>().recorded_batches();
    assert_eq!(recorded.len(), 5); // 4 prefill + 1 decode
    for i in 0..4 {
        assert_eq!(recorded[i].phase, BatchPhase::PrefillOnly);
        assert_eq!(recorded[i].input_ids.len(), 1024);
    }
    assert_eq!(recorded[4].phase, BatchPhase::DecodeOnly);
    assert_eq!(recorded[4].input_ids.len(), 1);

    // Verify: the request produced exactly 1 output token.
    let event = tok_rx.try_recv().unwrap();
    assert!(matches!(event, TokenEvent::Token(_)));
}
```

#### Page Allocator Under Pressure

```rust
#[test]
fn test_page_allocator_eviction_under_pressure() {
    let mut pool = KvCachePool::new_test(num_pages: 64, page_size: 16);
    let mut cache = RadixCache::new(16, 64);

    // Fill the cache with 60 pages worth of prefixes.
    for i in 0..60u32 {
        let tokens: Vec<u32> = (i * 16..(i + 1) * 16).collect();
        let page = pool.allocate().unwrap();
        cache.insert_prefix(&tokens, &[page]);
    }

    assert_eq!(pool.free_pages(), 4);

    // Now try to admit a request that needs 10 pages.
    // The cache should evict 6 LRU pages to make room.
    let evicted = cache.evict_pages(6);
    assert_eq!(evicted, 6);
    assert_eq!(pool.free_pages(), 10); // 4 + 6 freed
}
```

### 13.3 Sampling Tests

```rust
#[test]
fn test_greedy_sampling() {
    let logits = vec![0.1f32, 0.2, 0.9, 0.3, 0.1];
    let params = SamplingParams { temperature: 0.0, ..Default::default() };
    let token = sample_token(&logits, &params, &[]);
    assert_eq!(token, 2); // argmax
}

#[test]
fn test_temperature_sampling_is_deterministic_with_seed() {
    let logits = vec![0.1f32, 0.5, 0.4];
    let params = SamplingParams { temperature: 0.8, top_p: 0.95, ..Default::default() };
    // With the same RNG seed, output must be deterministic.
    let t1 = sample_token_with_seed(&logits, &params, &[], 42);
    let t2 = sample_token_with_seed(&logits, &params, &[], 42);
    assert_eq!(t1, t2);
}

#[test]
fn test_repetition_penalty_suppresses_repeated_tokens() {
    let logits = vec![0.9f32, 0.1, 0.0]; // token 0 would be chosen greedily
    let params = SamplingParams {
        temperature: 0.0,
        repetition_penalty: 1.5,
        ..Default::default()
    };
    let generated_so_far = vec![0u32]; // token 0 already appeared
    let token = sample_token(&logits, &params, &generated_so_far);
    assert_eq!(token, 1); // token 0 is penalized; token 1 wins
}
```

---
## 14. Performance Targets

All targets measured on the respective primary hardware with the primary model, at a context length of 2048 tokens (prompt + generated) unless otherwise noted.

### 14.0 Phase 0 Targets: Small Model Validation (burn-wgpu, no offload)

**Hardware**: Any machine with a GPU or iGPU supported by `burn-wgpu` (Vulkan/Metal). Primary test platforms: Linux Intel iGPU and macOS Metal.

**Model**: Gemma 3 1B–4B (or Qwen2.5-1.5B) in BF16/F16. Fits entirely in GPU memory — no offloading.

**Purpose**: These targets validate the engine orchestration logic, not raw inference performance. The absolute numbers are intentionally modest; correctness and scheduler behavior matter here.

| Metric | Target | Notes |
|--------|--------|-------|
| Correctness test: capital of China | Pass | `output.to_lowercase().contains("beijing")` |
| Correctness test: 1+1 arithmetic | Pass | `output.trim() == "2"` |
| Concurrent correctness (8 requests) | Pass | All requests produce correct output |
| Radix cache hit rate (repeated prompt) | > 80% | Same 512-token prompt sent 10 times |
| Chunked prefill correctness | Pass | Output identical with and without chunking |
| TTFT (batch 1, 512-token prompt, 1B model) | < 2 s | Fast enough for interactive testing |
| Decode throughput (batch 1, 1B model) | > 15 tok/s | Sufficient for rapid iteration |
| Scheduler overhead per iteration | < 1 ms | Measured with stub backend |
| `cargo test` pass rate | 100% | All unit and integration tests |

### 14.1 Phase 1 Targets: Linux (26B MoE, Intel Arc iGPU, with offload)

**Hardware**: Intel Core Ultra 9 185H (Meteor Lake), Intel Arc iGPU (128 EU / ~16 TOPS FP16), 32 GB DDR5-5600, 2 TB NVMe SSD (PCIe 4.0, ~7 GB/s sequential read).

**Model**: Qwen2.5-MoE-7B-A2.7B or similar 26B sparse MoE in Q4 (6B active parameters per token).

| Metric | Target | Notes |
|--------|--------|-------|
| TTFT (batch 1, cold cache, 512-token prompt) | < 4 s | First token latency |
| TTFT (batch 1, warm cache, 512-token prompt) | < 0.5 s | Radix cache hit |
| Decode throughput (batch 1) | > 8 tok/s | Single request |
| Decode throughput (batch 4) | > 20 tok/s | Aggregate |
| Decode throughput (batch 8) | > 30 tok/s | Aggregate |
| Decode throughput (batch 16) | > 35 tok/s | Aggregate |
| Radix cache hit rate (repeated system prompt) | > 85% | 1024-token shared prefix |
| Scheduler overhead per iteration | < 1 ms | Time in `schedule_batch` |
| KV pool utilization at batch 16 | > 70% | Efficient page allocation |
| Expert cache hit rate (after warmup) | > 75% | Experts in GPU memory |

### 14.2 Phase 2A Targets: macOS (31B Dense, Apple Silicon, burn-ggml)

**Hardware**: MacBook Air M3, 16 GB unified memory (fallback to SSD streaming if needed), 512 GB NVMe SSD.

**Model**: Qwen3-32B-Instruct in Q4_K_M quantization (~18 GB, fits in 16 GB with aggressive KV management or in 24–32 GB comfortably).

| Metric | Target | Notes |
|--------|--------|-------|
| TTFT (batch 1, cold cache, 512-token prompt) | < 3 s | Metal compute |
| TTFT (batch 1, warm cache, 512-token prompt) | < 0.3 s | Radix cache hit |
| Decode throughput (batch 1) | > 12 tok/s | Single request |
| Decode throughput (batch 4) | > 35 tok/s | Aggregate |
| Decode throughput (batch 8) | > 55 tok/s | Aggregate |
| Long-context TTFT (8192-token prompt) | < 30 s | With KV SSD offload |
| Radix cache hit rate (repeated system prompt) | > 85% | 1024-token shared prefix |
| Scheduler overhead per iteration | < 0.5 ms | macOS is faster than Linux iGPU |

### 14.3 Phase 2B Targets: Disaggregated Architecture and API Polish

| Metric | Target | Notes |
|--------|--------|-------|
| P50 decode latency per token (batch 8) | < 40 ms | Predictable decode SLO |
| P99 decode latency per token (batch 8) | < 80 ms | No prefill jitter |
| Throughput improvement vs. non-disaggregated | > 20% | At 16+ concurrent requests |
| KV handoff latency (prefill → decode) | < 1 ms | Shared memory, no copy needed |

### 14.4 Radix Cache Effectiveness

The radix cache is only valuable when requests share long common prefixes. Typical deployment patterns:

- **System-prompt-heavy chatbots**: 512–2048 token system prompts, many different user messages. Expected cache hit rate: 80–95% of system prompt tokens.
- **RAG (retrieval-augmented generation)**: retrieved passages are often reused. Expected hit rate: 40–70%.
- **Single-use requests (no shared prefix)**: hit rate ~0%. Radix cache does not degrade performance in this case (the eviction path is fast).

### 14.5 Memory Budget (macOS, 16 GB)

| Component | Budget |
|-----------|--------|
| Model weights (31B Q4_K_M) | ~18 GB |
| KV cache pool | 2 GB |
| OS + framework | 1 GB |
| Activations / working memory | 0.5 GB |
| Total | ~21.5 GB |

This exceeds 16 GB. On a 16 GB MacBook Air, the 31B model requires either:
1. A more aggressive quantization (Q2 or Q3, ~10–12 GB) with accuracy tradeoff, or
2. Layer-wise weight streaming from SSD (WeightCache), accepting slower inference.
3. The 13B or 14B model class instead.

On a 24–32 GB MacBook Pro (M3 Max / M4 Pro), the 31B Q4 model fits comfortably with a 4–6 GB KV pool.

---
## 15. Key Risks

### 15.1 Backend Paged Attention Support

**Risk**: `burn-wgpu` and `burn-ggml` may not support paged (non-contiguous) attention natively. The KV cache pool relies on paged KV: each page can be anywhere in the pool buffer, and the attention kernel must scatter/gather KV from non-contiguous locations.

**Mitigation**:
- Phase 1 can use a contiguous KV layout (concatenate pages into a contiguous buffer before the forward pass). This copies KV data but avoids the dependency on a native paged attention kernel.
- Phase 2 can implement a custom paged attention kernel in WGSL (wgpu) or GGML (Metal). FlashInfer's paged attention GPU kernel is well-documented and the algorithm is straightforward to port.
- Worst case: use the copy-to-contiguous approach permanently; it adds ~5–10% overhead per iteration.

**Probability**: High that native paged attention is not available in burn backends v1. Fallback is well-defined.

### 15.2 Intel Arc iGPU Shared Memory Bandwidth

**Risk**: Intel Arc integrated GPUs share memory bandwidth with the CPU. In a 16 GB system, the GPU and CPU compete for ~90 GB/s DDR5 bandwidth. For a memory-bandwidth-bound workload (decode), this is the primary bottleneck.

**Mitigation**:
- Use INT4 quantization (reduces memory traffic by 4x vs FP32).
- Minimize unnecessary CPU-GPU data movement (do not copy logits back to CPU on every step; batch the sampling).
- Use `burn-wgpu`'s async compute path to pipeline GPU work.
- Accept lower throughput targets for Linux vs macOS (Apple's unified memory is faster and has no sharing overhead).

**Probability**: High that this is the binding constraint on Linux. Targets in §14.1 account for this.

### 15.3 Radix Cache Memory Overhead

**Risk**: The radix tree itself (without the KV pages) can become large if there are many unique prefixes. Each `RadixNode` holds a `Vec<u32>` (edge tokens), a `Vec<PageIndex>` (KV pages), and a `HashMap` (children). At millions of nodes, this could consume gigabytes of CPU RAM.

**Mitigation**:
- In practice, the tree is bounded by the KV pool size (eviction keeps `total_pages <= max_pages`).
- The tree structure overhead is small: a node with 16-token edges and 1 child is ~300 bytes. 10,000 nodes = 3 MB. Not a concern.
- Add a cap on the number of tree nodes as a secondary safeguard.

**Probability**: Low. The KV pool size naturally bounds tree growth.

### 15.4 MoE Expert Thrashing

**Risk**: If consecutive requests activate disjoint sets of experts, the `ExpertWeightCache` is effectively useless — every forward pass loads a completely different set of experts from SSD. This "expert thrashing" can reduce throughput to the SSD read speed (~5–10 tokens/s), far below the GPU compute limit.

**Mitigation**:
- Expert cache warming on admission (§8.5): inspect the first tokens of incoming requests and preferentially admit requests that activate cached experts.
- Expert-affinity batching: when multiple requests are in the waiting queue, prefer to batch those that activate the same experts.
- Increase expert cache size if SSD bandwidth is the bottleneck (trade model weight RAM for expert cache RAM).
- Accept that MoE throughput on SSD is fundamentally limited by SSD bandwidth if expert diversity is high.

**Probability**: Medium. Depends heavily on the workload. For typical chat workloads (similar tasks, similar expert activation patterns), thrashing should be rare.

### 15.5 Tokenizer Bottleneck

**Risk**: The HuggingFace `tokenizers` crate is fast but single-threaded per tokenizer instance. Under high concurrency (100+ requests/s), the tokenizer could become the bottleneck.

**Mitigation**:
- The tokenizer service (§11) handles requests serially; this is fine for < 50 req/s.
- For higher throughput: run multiple tokenizer tasks with a `round_robin` dispatcher.
- Tokenization is not on the GPU critical path; requests are tokenized before they enter the engine queue.

**Probability**: Low for edge hardware targets (< 10 req/s typical).

### 15.6 Rust async + GPU Thread Interaction

**Risk**: GPU command queues (Metal, Vulkan) are typically not `Send` and require all access from a single thread. Tokio's async executor runs tasks on a thread pool. Getting async scheduling and GPU execution to interact correctly without deadlocks requires careful design.

**Mitigation**:
- The GPU forward pass runs on a dedicated OS thread (not a Tokio worker thread). The scheduler communicates with it via `std::sync::mpsc` channels.
- `spawn_blocking` in Tokio pins a closure to a dedicated thread; we use this for the GPU→scheduler result path.
- All `BackendHandle` trait objects must be `Send + Sync`; the GPU thread owns the only mutable reference.
- Use `tokio::sync::Mutex` (not `std::sync::Mutex`) for any state shared between the scheduler task and the GPU thread to avoid blocking the Tokio runtime.

**Probability**: Medium. Requires careful implementation in Phase 0 but the pattern is well-established in projects like mistral.rs.

### 15.7 Chunked Prefill KV Consistency

**Risk**: Chunked prefill processes a long prompt in multiple iterations. Between iterations, the scheduler might preempt the request (§4.7) and free its KV pages. Resuming a preempted chunked-prefill request requires recomputing the KV for all previously processed chunks, which is expensive.

**Mitigation**:
- Phase 1: never preempt a chunked-prefill request that is more than 25% complete.
- Phase 2: swap the partial KV to SSD (KvOffloadManager) instead of freeing it; restore on resume.
- Accept occasional full recomputation as a rare event (preemption is rare when the pool is sized correctly).

**Probability**: Low-medium. Preemption is rare; the full recomputation fallback is acceptable.

### 15.8 Apple ANE Integration (Future)

**Risk**: CoreML-based ANE access requires model export, compilation, and runtime in a completely different framework (CoreML). Keeping the Burn model and the CoreML model synchronized as the model changes is a maintenance burden. Debugging ANE execution is nearly impossible (no kernel-level profiling, black-box execution).

**Mitigation**: Defer ANE integration to Phase 3+. The `PrefillBackend` trait provides a clean insertion point. Treat ANE as a pure optimization that does not affect correctness.

**Probability**: The risk exists but is not in scope for the current plan.

---

## Appendix A: Crate Structure

```
burn-inference/
├── Cargo.toml                  # workspace
├── crates/
│   ├── engine/                 # Scheduler, radix cache, KV pool, request lifecycle
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── scheduler.rs    # schedule_batch, PrefillAdder, collect_decode_requests
│   │   │   ├── radix_cache.rs  # RadixNode, RadixCache
│   │   │   ├── kv_pool.rs      # KvCachePool, PageTable, UnifiedPageAllocator
│   │   │   ├── request.rs      # Request, RequestState, Batch, SamplingParams
│   │   │   ├── sampling.rs     # sample_token, nucleus sampling, repetition penalty
│   │   │   ├── engine_loop.rs  # engine_loop, run_overlapped_loop
│   │   │   └── context.rs      # EngineContext, EngineConfig, EngineStats
│   │   └── Cargo.toml
│   │
│   ├── api/                    # HTTP server, tokenizer service
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── http.rs         # axum router, handlers
│   │   │   ├── types.rs        # ChatCompletionRequest/Response, etc.
│   │   │   ├── tokenizer.rs    # TokenizerService, TokenizerHandle
│   │   │   └── chat_template.rs # apply_chat_template
│   │   └── Cargo.toml
│   │
│   ├── backend-trait/          # BackendHandle trait definition (shared)
│   │   ├── src/lib.rs
│   │   └── Cargo.toml
│   │
│   ├── backend-wgpu/           # burn-wgpu adapter
│   │   ├── src/lib.rs
│   │   └── Cargo.toml
│   │
│   ├── backend-ggml/           # burn-ggml adapter
│   │   ├── src/lib.rs
│   │   └── Cargo.toml
│   │
│   └── server/                 # Binary: wires everything together
│       ├── src/main.rs
│       └── Cargo.toml
│
├── tests/
│   ├── correctness/
│   │   ├── capital_of_china.rs
│   │   └── arithmetic.rs
│   └── scheduler/
│       ├── radix_cache_tests.rs
│       ├── chunked_prefill_tests.rs
│       └── concurrent_requests_tests.rs
│
└── benches/
    ├── throughput.rs
    ├── ttft.rs
    └── radix_cache_bench.rs
```

## Appendix B: Key Crate Dependencies

```toml
[workspace.dependencies]
# Async runtime
tokio = { version = "1", features = ["full"] }
async-trait = "0.1"

# HTTP
axum = { version = "0.7", features = ["macros"] }
tower-http = { version = "0.5", features = ["cors", "trace"] }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Tokenization
tokenizers = "0.19"

# Concurrency
parking_lot = "0.12"
crossbeam-queue = "0.3"

# Radix cache node arena
slab = "0.4"            # O(1) generational arena; avoids Arc<RwLock<>> per node
rustc-hash = "2"        # FxHashMap — ~2x faster than std HashMap for u32 keys

# IDs
uuid = { version = "1", features = ["v4"] }

# Error handling
thiserror = "1"
anyhow = "1"

# Streaming
async-stream = "0.3"
futures = "0.3"

# Observability
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Burn
burn = { version = "0.14", features = [] }

# Half precision
half = { version = "2", features = ["bytemuck"] }
```

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **Continuous batching** | Scheduling policy where decode requests run every iteration; new prefill requests are added to the batch as capacity allows |
| **Radix cache** | Prefix tree that stores KV cache entries for common token prefixes, enabling reuse across requests |
| **Chunked prefill** | Splitting a long prompt into chunks across multiple iterations to stay within a token budget |
| **KV cache pool** | A single pre-allocated buffer holding KV tensors for all in-flight requests, addressed via page indices |
| **Page** | A fixed-size unit of KV cache storage holding `page_size` tokens for all layers |
| **Page table** | Mapping from (request_id, sequence_position) to page index in the pool |
| **Overlap scheduling** | Running CPU scheduling of the next batch concurrently with GPU execution of the current batch |
| **PrefetchOps** | Interface to queue asynchronous I/O (weight or KV page loads from SSD) in advance of when they are needed |
| **WeightCache** | Backend component managing layer-wise weight streaming from SSD when model does not fit in memory |
| **KvOffloadManager** | Backend component that evicts KV pages from GPU memory to SSD and restores them on demand |
| **ExpertWeightCache** | Backend component managing MoE expert weight streaming from SSD |
| **FlashMoE** | Technique to run MoE expert selection and dispatch via GGML's `MUL_MAT_ID` operation |
| **TTFT** | Time To First Token: latency from request arrival to first generated token |
| **Disaggregation** | Separating prefill and decode into distinct workers to isolate their different compute characteristics |
| **ANE** | Apple Neural Engine: fixed-function accelerator on Apple Silicon, FP16 only, CoreML API |

---

*End of burn-inference Design Document v0.1*
