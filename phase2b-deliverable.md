# Phase 2B Deliverable Specification: Prefill-Decode Disaggregation, API Polish, and Benchmark Suite

**Status:** Draft v0.1
**Date:** 2026-04-04
**Scope:** Phase 2B only — orchestration-level features; all compute backends are complete and unchanged

---

## Table of Contents

1. [Overview](#1-overview)
2. [Scope](#2-scope)
3. [New Components](#3-new-components)
4. [Disaggregation Architecture](#4-disaggregation-architecture)
5. [Data Flow](#5-data-flow)
6. [Test Specification](#6-test-specification)
7. [Acceptance Criteria](#7-acceptance-criteria)
8. [Non-Goals](#8-non-goals)
9. [Dependencies](#9-dependencies)
10. [Open Questions](#10-open-questions)

---

## 1. Overview

Phase 2B is the final phase of the burn-inference project. It adds prefill-decode disaggregation, production API polish, and a comprehensive benchmark suite on top of the already-validated compute backends from Phase 1 (burn-wgpu on Linux) and Phase 2A (burn-ggml on macOS). Phase 2B introduces no new compute kernels, no new GPU API surface, and no changes to the `BackendHandle` trait. Every addition in Phase 2B is at the orchestration layer.

### What Phase 2B adds

1. **Prefill-Decode Disaggregation.** A single-machine disaggregated scheduler that runs prefill and decode in separate Tokio tasks sharing one GPU via a mutex. The decode worker has GPU priority. KV pages are handed off from the prefill worker to the decode worker via a lock-free SPSC ring buffer.

2. **API Polish.** New HTTP endpoints (`POST /v1/completions`, `GET /v1/models`, `GET /health`, `GET /metrics`), CORS support, configurable request timeout with 408 response, graceful shutdown on SIGTERM, `X-Request-Id` header echo, and optional rate limiting.

3. **Prometheus Metrics.** A `/metrics` endpoint in Prometheus text format covering request counts, latency histograms (P50/P90/P99), token throughput, radix cache hit/miss, KV pool utilization, and expert cache hit/miss (Phase 1 only).

4. **Benchmark Suite.** A standalone binary `burn-inference-bench` covering TTFT (cold and warm), decode throughput, shared-prefix cache efficiency, long-context, concurrent ramp, and disaggregation comparison. Output is JSON plus a human-readable table.

5. **Configuration File.** A TOML configuration file `burn-inference.toml` that replaces CLI flags for all tunable parameters.

6. **Documentation.** `README.md` (quickstart, installation, model download, example curl commands) and `BENCHMARKS.md` (results tables for both platforms).

### What Phase 2B proves

1. **Disaggregation is correct.** Output with disaggregation enabled is identical to colocated output on both platforms.
2. **Disaggregation improves throughput.** At 16 concurrent requests, disaggregated mode achieves >= 15% higher decode throughput than colocated on Linux.
3. **The API is production-ready.** Health, metrics, CORS, timeout, graceful shutdown, and request ID all behave correctly.
4. **Performance targets are met on both platforms.** All final targets in Section 7 are met reproducibly (< 5% variance across 3 benchmark runs).

### What is unchanged from Phase 0, Phase 1, and Phase 2A

| Component | Status |
|-----------|--------|
| `inference-engine` crate | Unchanged |
| `inference-api` crate | Unchanged — HTTP routing extended, internals not modified |
| `BackendHandle` trait | Unchanged |
| `TokenizerService` | Unchanged |
| `RadixCache` | Unchanged |
| Scheduler (`schedule_batch`, `run_overlapped_loop`) | Unchanged |
| `WeightCache<K>` | Unchanged |
| `KvOffloadManager` | Unchanged |
| `PrefetchOps` trait | Unchanged |
| `GgmlBackendHandle` | Unchanged |
| `WgpuOffloadBackendHandle` | Unchanged |
| `StubBackend` | Unchanged |
| All Phase 0, Phase 1, Phase 2A cargo tests | Must continue to pass |

### Platforms

- **Linux:** Intel iGPU (Iris Xe / Arc), burn-wgpu backend, Gemma 4 26B MoE Q4_K_M, 16 GB RAM
- **macOS:** Apple Silicon M-series, burn-ggml backend, Gemma 4 31B dense Q3_K_M, 16 GB unified memory

### Timeline

- Weeks 19–22 (Phase 2A covered Weeks 15–18; Phase 1 covered Weeks 7–14; Phase 0 covered Weeks 1–6)

---

## 2. Scope

### In scope

- New crate `inference-disagg`: `PrefillWorker`, `DecodeWorker`, `KvHandoffQueue`, `DisaggEngine`
- New crate `inference-bench`: standalone `burn-inference-bench` binary with all 7 benchmark scenarios
- `inference-api` HTTP route additions: `POST /v1/completions`, `GET /v1/models`, `GET /health`, `GET /metrics`
- CORS middleware via `tower-http::cors`
- Request timeout middleware (configurable, default 120s, returns 408)
- Graceful shutdown handler (SIGTERM, drain up to 30s)
- `X-Request-Id` header echo middleware
- Optional rate limiting (max concurrent requests, configurable)
- Prometheus metrics registry and all metrics listed in Section 3.4
- `burn-inference.toml` configuration file with full schema (Section 3.5)
- All 8 new correctness tests (Section 6)
- All performance targets met on both platforms (Section 7)
- `README.md` and `BENCHMARKS.md`
- `cargo clippy -- -D warnings` clean
- `cargo fmt --check` clean

### Explicitly out of scope

- Any change to `inference-engine`, `BackendHandle`, `WeightCache`, `KvOffloadManager`, `RadixCache`, or `PrefetchOps`
- Multi-machine disaggregation (network KV transfer) — future work
- ANE (Apple Neural Engine) acceleration — Phase 3+
- Speculative decoding — Phase 3+
- Continuous batching improvements beyond what Phase 0 delivered — Phase 3+
- Multi-GPU inference — not planned
- Windows support — not planned
- Models other than Gemma 4 26B MoE (Linux) and 31B dense (macOS) — not tested in Phase 2B
- Fine-tuning or training — inference only
- OpenAI-compatible streaming (`text/event-stream`) for `/v1/completions` — already in `/v1/chat/completions` from Phase 0; `/v1/completions` non-streaming only in Phase 2B

---


## 3. New Components

This section specifies every new component introduced in Phase 2B. Existing components are not modified.

---

### 3.1 `inference-disagg` Crate

**Crate path:** `inference-disagg/`  
**Purpose:** Single-machine prefill-decode disaggregation. Two Tokio tasks share one GPU via an `Arc<Mutex<Box<dyn BackendHandle>>>`. KV pages produced by the prefill worker are delivered to the decode worker via a lock-free SPSC ring buffer. The decode worker has GPU priority.

#### 3.1.1 `PrefillWorker`

```rust
// inference-disagg/src/prefill_worker.rs
use std::{sync::Arc, time::Duration};
use tokio::sync::{Mutex, mpsc};
use inference_engine::{EngineContext, Request, RequestId};
use inference_backend::BackendHandle;
use crate::{KvHandoff, KvHandoffQueue};

pub struct PrefillWorker {
    /// Shared engine context (scheduler state, radix cache, KV pool).
    ctx:      Arc<Mutex<EngineContext>>,
    /// Shared GPU handle — same instance as DecodeWorker's gpu field.
    gpu:      Arc<Mutex<Box<dyn BackendHandle>>>,
    /// Receives new requests from the HTTP layer.
    sched_rx: mpsc::Receiver<Arc<Mutex<Request>>>,
    /// Sends completed KV pages to the decode worker.
    kv_tx:    KvHandoffQueue,
}

impl PrefillWorker {
    pub fn new(
        ctx:      Arc<Mutex<EngineContext>>,
        gpu:      Arc<Mutex<Box<dyn BackendHandle>>>,
        sched_rx: mpsc::Receiver<Arc<Mutex<Request>>>,
        kv_tx:    KvHandoffQueue,
    ) -> Self {
        Self { ctx, gpu, sched_rx, kv_tx }
    }

    /// Main loop. Runs in a dedicated Tokio task.
    pub async fn run(mut self) {
        loop {
            // Drain the scheduler channel, batch up to max_prefill_tokens.
            let batch = self.collect_batch().await;
            if batch.is_empty() {
                tokio::task::yield_now().await;
                continue;
            }

            // Try to acquire GPU with a timeout so decode worker is not starved.
            let gpu_result = tokio::time::timeout(
                Duration::from_millis(5),
                self.gpu.lock(),
            )
            .await;

            let gpu = match gpu_result {
                Ok(guard) => guard,
                Err(_) => {
                    // Decode worker needed GPU. Yield and retry next iteration.
                    tokio::task::yield_now().await;
                    continue;
                }
            };

            // Run prefill forward pass.
            let mut ctx = self.ctx.lock().await;
            let handoffs = ctx.run_prefill_batch(&*gpu, &batch)
                .expect("prefill forward pass failed");
            drop(ctx);
            drop(gpu);

            // Send KV handoffs to decode worker.
            for handoff in handoffs {
                // Non-blocking push; if ring buffer is full, spin once.
                while self.kv_tx.push(handoff.clone()).is_err() {
                    tokio::task::yield_now().await;
                }
            }
        }
    }

    async fn collect_batch(&mut self) -> Vec<Arc<Mutex<Request>>> {
        let mut batch = Vec::new();
        // Non-blocking drain up to max batch size.
        while let Ok(req) = self.sched_rx.try_recv() {
            batch.push(req);
            if batch.len() >= 32 {
                break;
            }
        }
        batch
    }
}
```

#### 3.1.2 `DecodeWorker`

```rust
// inference-disagg/src/decode_worker.rs
use std::sync::Arc;
use tokio::sync::Mutex;
use inference_engine::{EngineContext, RequestId};
use inference_backend::BackendHandle;
use crate::KvHandoffQueue;

pub struct DecodeWorker {
    /// Shared engine context — same Arc as PrefillWorker.
    ctx:   Arc<Mutex<EngineContext>>,
    /// Shared GPU — same Arc as PrefillWorker. DecodeWorker acquires without timeout.
    gpu:   Arc<Mutex<Box<dyn BackendHandle>>>,
    /// Receives completed KV pages from prefill worker.
    kv_rx: KvHandoffQueue,
}

impl DecodeWorker {
    pub fn new(
        ctx:   Arc<Mutex<EngineContext>>,
        gpu:   Arc<Mutex<Box<dyn BackendHandle>>>,
        kv_rx: KvHandoffQueue,
    ) -> Self {
        Self { ctx, gpu, kv_rx }
    }

    /// Main loop. Runs in a dedicated Tokio task. Acquires GPU without timeout.
    pub async fn run(self) {
        loop {
            // Drain all available KV handoffs and register them in the engine context.
            {
                let mut ctx = self.ctx.lock().await;
                while let Some(handoff) = self.kv_rx.pop() {
                    ctx.register_kv_handoff(handoff);
                }
            }

            // Acquire GPU (no timeout — decode has priority).
            let gpu = self.gpu.lock().await;
            let mut ctx = self.ctx.lock().await;

            let has_active = ctx.has_active_decode_requests();
            if !has_active {
                drop(ctx);
                drop(gpu);
                tokio::task::yield_now().await;
                continue;
            }

            // Run one decode step for all active sequences.
            ctx.run_decode_step(&*gpu)
                .expect("decode step failed");
        }
    }
}
```

#### 3.1.3 `KvHandoff` — KV Page Transfer Record

```rust
// inference-disagg/src/types.rs
use inference_engine::{RequestId, PageIndex};

/// Sent from PrefillWorker to DecodeWorker after prefill completes for one request.
#[derive(Clone, Debug)]
pub struct KvHandoff {
    /// The request whose prefill just completed.
    pub request_id: RequestId,
    /// Indices of KV cache pages that now hold the prefill KV state.
    /// The DecodeWorker reads these pages during its decode steps.
    pub kv_pages:   Vec<PageIndex>,
    /// Number of tokens whose KV state is now resident in kv_pages.
    pub cached_len: usize,
}
```

#### 3.1.4 `KvHandoffQueue` — Lock-Free SPSC Ring Buffer

```rust
// inference-disagg/src/queue.rs
//
// Wraps an `rtrb::RingBuffer<KvHandoff>` (or `crossbeam::queue::ArrayQueue`)
// in a pair of Arc-wrapped halves. One Arc goes to PrefillWorker (producer),
// the other to DecodeWorker (consumer).

use crate::KvHandoff;

#[cfg(feature = "rtrb")]
pub use rtrb_impl::{KvHandoffQueue, new_kv_handoff_queue};

#[cfg(feature = "rtrb")]
mod rtrb_impl {
    use super::KvHandoff;
    use rtrb::{Consumer, Producer, RingBuffer};
    use std::sync::Arc;
    use parking_lot::Mutex;

    /// Capacity of the SPSC ring buffer (number of KvHandoff slots).
    /// At most 32 concurrent requests, so 64 slots is double-buffered.
    pub const QUEUE_CAPACITY: usize = 64;

    /// Producer half — held by PrefillWorker.
    pub struct KvHandoffQueue {
        pub(crate) inner: Arc<Mutex<Producer<KvHandoff>>>,
    }

    /// Consumer half — held by DecodeWorker.
    pub struct KvHandoffConsumer {
        pub(crate) inner: Arc<Mutex<Consumer<KvHandoff>>>,
    }

    impl KvHandoffQueue {
        /// Attempt to push one handoff. Returns Err(handoff) if the buffer is full.
        pub fn push(&self, handoff: KvHandoff) -> Result<(), KvHandoff> {
            let mut prod = self.inner.lock();
            prod.push(handoff).map_err(|e| e.into_inner())
        }
    }

    impl KvHandoffConsumer {
        /// Pop one handoff, or None if the buffer is empty.
        pub fn pop(&self) -> Option<KvHandoff> {
            let mut cons = self.inner.lock();
            cons.pop().ok()
        }
    }

    /// Construct a matched producer/consumer pair.
    pub fn new_kv_handoff_queue() -> (KvHandoffQueue, KvHandoffConsumer) {
        let (prod, cons) = RingBuffer::<KvHandoff>::new(QUEUE_CAPACITY);
        (
            KvHandoffQueue  { inner: Arc::new(Mutex::new(prod)) },
            KvHandoffConsumer { inner: Arc::new(Mutex::new(cons)) },
        )
    }
}
```

**Note on crate choice:** `rtrb` is preferred because it is a purpose-built lock-free SPSC ring buffer with no std dependency and bounded latency. `crossbeam::ArrayQueue` is the fallback if `rtrb` is unavailable. The feature flag `rtrb` in `Cargo.toml` selects between them. Both implementations expose the same `push`/`pop` API.

#### 3.1.5 `DisaggEngine` — Top-Level Assembly

```rust
// inference-disagg/src/engine.rs
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use inference_engine::EngineContext;
use inference_backend::BackendHandle;
use crate::{PrefillWorker, DecodeWorker, new_kv_handoff_queue};

/// Assembles PrefillWorker and DecodeWorker, starts both as Tokio tasks.
pub struct DisaggEngine {
    /// Sender half for submitting new requests to the prefill worker.
    pub request_tx: mpsc::Sender<Arc<Mutex<inference_engine::Request>>>,
}

impl DisaggEngine {
    pub fn start(
        ctx: Arc<Mutex<EngineContext>>,
        gpu: Box<dyn BackendHandle>,
    ) -> Self {
        let gpu = Arc::new(Mutex::new(gpu));
        let (request_tx, request_rx) = mpsc::channel(256);
        let (kv_tx, kv_rx) = new_kv_handoff_queue();

        let prefill = PrefillWorker::new(
            ctx.clone(), gpu.clone(), request_rx, kv_tx,
        );
        let decode = DecodeWorker::new(
            ctx.clone(), gpu.clone(), kv_rx,
        );

        tokio::spawn(prefill.run());
        tokio::spawn(decode.run());

        Self { request_tx }
    }
}
```

**Colocated mode compatibility:** When `[disaggregation] enabled = false` (the default), `inference-api` uses the existing `run_overlapped_loop` path from Phase 0 unchanged. `DisaggEngine` is constructed only when disaggregation is enabled. The HTTP layer submits requests to `DisaggEngine::request_tx` or to the scheduler directly, depending on the config.

---

### 3.2 API Polish — New Endpoints and Middleware

All additions are in the `inference-api` crate. Existing routes (`POST /v1/chat/completions`, `POST /tokenize`) are not modified.

#### 3.2.1 `POST /v1/completions` — Legacy Non-Chat Completion

```rust
// inference-api/src/routes/completions.rs
use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use crate::AppState;

#[derive(Deserialize)]
pub struct CompletionRequest {
    pub model:       String,
    pub prompt:      String,   // plain string, not a messages array
    pub max_tokens:  Option<u32>,
    pub temperature: Option<f32>,
    pub top_p:       Option<f32>,
    pub stream:      Option<bool>,  // must be false or absent; streaming not supported
    pub stop:        Option<Vec<String>>,
}

#[derive(Serialize)]
pub struct CompletionResponse {
    pub id:      String,
    pub object:  String,       // "text_completion"
    pub created: u64,
    pub model:   String,
    pub choices: Vec<CompletionChoice>,
    pub usage:   UsageStats,
}

#[derive(Serialize)]
pub struct CompletionChoice {
    pub text:          String,
    pub index:         u32,
    pub finish_reason: String,  // "stop" | "length"
}

#[derive(Serialize)]
pub struct UsageStats {
    pub prompt_tokens:     u32,
    pub completion_tokens: u32,
    pub total_tokens:      u32,
}

pub async fn post_completions(
    State(state): State<AppState>,
    Json(req):    Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, ApiError> {
    if req.stream == Some(true) {
        return Err(ApiError::bad_request("streaming not supported on /v1/completions"));
    }
    let result = state.engine.complete(req.into()).await?;
    Ok(Json(result.into()))
}
```

#### 3.2.2 `GET /v1/models` — List Available Models

```rust
// inference-api/src/routes/models.rs
#[derive(Serialize)]
pub struct ModelList {
    pub object: String,      // "list"
    pub data:   Vec<ModelCard>,
}

#[derive(Serialize)]
pub struct ModelCard {
    pub id:       String,
    pub object:   String,    // "model"
    pub created:  u64,
    pub owned_by: String,
}

pub async fn get_models(
    State(state): State<AppState>,
) -> Json<ModelList> {
    Json(ModelList {
        object: "list".into(),
        data: vec![ModelCard {
            id:       state.config.model.path.file_stem()
                          .map(|s| s.to_string_lossy().into_owned())
                          .unwrap_or_else(|| "unknown".into()),
            object:   "model".into(),
            created:  0,
            owned_by: "burn-inference".into(),
        }],
    })
}
```

#### 3.2.3 `GET /health` — Health Check

```rust
// inference-api/src/routes/health.rs
use axum::http::StatusCode;
use serde_json::json;

pub async fn get_health() -> (StatusCode, axum::Json<serde_json::Value>) {
    (StatusCode::OK, axum::Json(json!({"status": "ok"})))
}
```

Behavior: always returns `HTTP 200 {"status": "ok"}` as long as the process is alive and the HTTP listener is accepting connections. No liveness checks on GPU or model are performed in Phase 2B.

#### 3.2.4 `GET /metrics` — Prometheus Metrics

```rust
// inference-api/src/routes/metrics.rs
use axum::{http::StatusCode, response::IntoResponse};
use prometheus::{Encoder, TextEncoder};
use crate::metrics::REGISTRY;

pub async fn get_metrics() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer)
        .expect("metric encoding failed");
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        buffer,
    )
}
```

#### 3.2.5 CORS Middleware

```rust
// inference-api/src/middleware/cors.rs
use tower_http::cors::{CorsLayer, Any};
use http::Method;

pub fn cors_layer() -> CorsLayer {
    CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any)
        .allow_origin(Any)
}
```

CORS is applied globally to all routes. `OPTIONS` preflight requests are handled automatically by `tower-http`. The allowed origin is `Any` in Phase 2B; production deployments should restrict this via config.

#### 3.2.6 Request Timeout Middleware

```rust
// inference-api/src/middleware/timeout.rs
use std::time::Duration;
use tower::timeout::TimeoutLayer;
use axum::http::StatusCode;

/// Returns a TimeoutLayer configured from the server config.
/// On timeout, returns HTTP 408 Request Timeout.
pub fn timeout_layer(secs: u64) -> TimeoutLayer {
    TimeoutLayer::new(Duration::from_secs(secs))
}

/// Map TimeoutError to 408 in the error handler.
pub fn handle_timeout_error(
    err: axum::BoxError,
) -> (StatusCode, String) {
    if err.is::<tower::timeout::error::Elapsed>() {
        (StatusCode::REQUEST_TIMEOUT, "Request timed out".into())
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Unhandled error: {err}"))
    }
}
```

#### 3.2.7 Graceful Shutdown

```rust
// inference-api/src/shutdown.rs
use tokio::signal;
use std::time::Duration;

/// Returns a future that completes when SIGTERM (or Ctrl-C) is received.
pub async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c().await.expect("failed to install Ctrl-C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c    => {},
        _ = terminate => {},
    }
}

/// Passed to axum::serve::with_graceful_shutdown.
/// Stops accepting new connections immediately; waits up to drain_secs for
/// in-flight requests to complete before forcing exit.
pub async fn graceful_shutdown_with_drain(drain_secs: u64) {
    shutdown_signal().await;
    // axum handles draining; this task exits after drain_secs.
    tokio::time::sleep(Duration::from_secs(drain_secs)).await;
}
```

Server startup wires this in:

```rust
axum::serve(listener, app)
    .with_graceful_shutdown(graceful_shutdown_with_drain(config.server.graceful_shutdown_secs))
    .await
    .unwrap();
```

#### 3.2.8 `X-Request-Id` Header Echo

```rust
// inference-api/src/middleware/request_id.rs
use axum::{
    middleware::{self, Next},
    extract::Request,
    response::Response,
    http::HeaderValue,
};
use uuid::Uuid;

pub async fn request_id_middleware(
    mut req: Request,
    next: Next,
) -> Response {
    let id = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_owned())
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    let id_header = HeaderValue::from_str(&id).unwrap_or_else(|_| HeaderValue::from_static("invalid"));
    let mut response = next.run(req).await;
    response.headers_mut().insert("x-request-id", id_header);
    response
}
```

#### 3.2.9 Rate Limiting (Optional)

```rust
// inference-api/src/middleware/rate_limit.rs
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use axum::{
    middleware::Next,
    extract::Request,
    response::Response,
    http::StatusCode,
};

#[derive(Clone)]
pub struct ConcurrentRequestLimiter {
    current: Arc<AtomicUsize>,
    max:     usize,
}

impl ConcurrentRequestLimiter {
    pub fn new(max: usize) -> Self {
        Self { current: Arc::new(AtomicUsize::new(0)), max }
    }

    pub async fn middleware(
        self,
        req: Request,
        next: Next,
    ) -> Result<Response, StatusCode> {
        let prev = self.current.fetch_add(1, Ordering::SeqCst);
        if prev >= self.max {
            self.current.fetch_sub(1, Ordering::SeqCst);
            return Err(StatusCode::TOO_MANY_REQUESTS);
        }
        let response = next.run(req).await;
        self.current.fetch_sub(1, Ordering::SeqCst);
        Ok(response)
    }
}
```

Rate limiting is enabled only when `[server] max_concurrent_requests` is set in `burn-inference.toml` and is less than `usize::MAX`. The middleware layer is conditionally added at startup.

---

### 3.3 Prometheus Metrics

**File:** `inference-api/src/metrics.rs`

A global `prometheus::Registry` is initialized at startup. All counters, histograms, and gauges are registered once and updated inline in the hot path.

```rust
// inference-api/src/metrics.rs
use once_cell::sync::Lazy;
use prometheus::{
    CounterVec, Histogram, HistogramOpts, HistogramVec, IntCounter, IntGauge,
    IntGaugeVec, Opts, Registry,
};

pub static REGISTRY: Lazy<Registry> = Lazy::new(Registry::new);

// --- Request counters ---
pub static REQUESTS_TOTAL: Lazy<CounterVec> = Lazy::new(|| {
    let opts = Opts::new(
        "burn_inference_requests_total",
        "Total inference requests by status",
    );
    let c = CounterVec::new(opts, &["status"]).unwrap();
    REGISTRY.register(Box::new(c.clone())).unwrap();
    c
});

// --- Request duration (P50/P90/P99 via histogram) ---
pub static REQUEST_DURATION_SECONDS: Lazy<Histogram> = Lazy::new(|| {
    let opts = HistogramOpts::new(
        "burn_inference_request_duration_seconds",
        "End-to-end request latency in seconds",
    )
    .buckets(vec![0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]);
    let h = Histogram::with_opts(opts).unwrap();
    REGISTRY.register(Box::new(h.clone())).unwrap();
    h
});

// --- Token throughput ---
pub static TOKENS_GENERATED_TOTAL: Lazy<IntCounter> = Lazy::new(|| {
    let c = IntCounter::new(
        "burn_inference_tokens_generated_total",
        "Total decode tokens generated",
    ).unwrap();
    REGISTRY.register(Box::new(c.clone())).unwrap();
    c
});

pub static TOKENS_PER_SECOND: Lazy<prometheus::Gauge> = Lazy::new(|| {
    let g = prometheus::Gauge::new(
        "burn_inference_tokens_per_second",
        "Rolling average decode tokens per second (last 10s window)",
    ).unwrap();
    REGISTRY.register(Box::new(g.clone())).unwrap();
    g
});

// --- Radix cache ---
pub static RADIX_CACHE_HITS: Lazy<IntCounter> = Lazy::new(|| {
    let c = IntCounter::new("burn_inference_radix_cache_hits_total", "Radix cache hits").unwrap();
    REGISTRY.register(Box::new(c.clone())).unwrap();
    c
});

pub static RADIX_CACHE_MISSES: Lazy<IntCounter> = Lazy::new(|| {
    let c = IntCounter::new("burn_inference_radix_cache_misses_total", "Radix cache misses").unwrap();
    REGISTRY.register(Box::new(c.clone())).unwrap();
    c
});

// --- KV pool utilization ---
pub static KV_POOL_PAGES_USED: Lazy<IntGauge> = Lazy::new(|| {
    let g = IntGauge::new("burn_inference_kv_pool_pages_used", "KV pool pages in use").unwrap();
    REGISTRY.register(Box::new(g.clone())).unwrap();
    g
});

pub static KV_POOL_PAGES_TOTAL: Lazy<IntGauge> = Lazy::new(|| {
    let g = IntGauge::new("burn_inference_kv_pool_pages_total", "KV pool total pages").unwrap();
    REGISTRY.register(Box::new(g.clone())).unwrap();
    g
});

// --- Expert cache (Phase 1 Linux only; no-op on Phase 2A macOS) ---
pub static EXPERT_CACHE_HITS: Lazy<IntCounter> = Lazy::new(|| {
    let c = IntCounter::new(
        "burn_inference_expert_cache_hits_total",
        "Expert weight cache hits (Linux MoE only)",
    ).unwrap();
    REGISTRY.register(Box::new(c.clone())).unwrap();
    c
});

pub static EXPERT_CACHE_MISSES: Lazy<IntCounter> = Lazy::new(|| {
    let c = IntCounter::new(
        "burn_inference_expert_cache_misses_total",
        "Expert weight cache misses (Linux MoE only)",
    ).unwrap();
    REGISTRY.register(Box::new(c.clone())).unwrap();
    c
});
```

**Update sites:**
- `REQUESTS_TOTAL` is incremented in the HTTP layer after each request completes, with label `"success"`, `"error"`, or `"timeout"`.
- `REQUEST_DURATION_SECONDS` is observed in the request ID middleware, measuring wall time from request receipt to response sent.
- `TOKENS_GENERATED_TOTAL` is incremented by the decode step, once per token per sequence.
- `TOKENS_PER_SECOND` is recomputed every 1 second by a background Tokio task that divides the token delta by elapsed time.
- `RADIX_CACHE_HITS` / `RADIX_CACHE_MISSES` are incremented in `RadixCache::lookup`.
- `KV_POOL_PAGES_USED` / `KV_POOL_PAGES_TOTAL` are set by a background task that polls `KvPool::stats()` every 500ms.
- `EXPERT_CACHE_HITS` / `EXPERT_CACHE_MISSES` are incremented in `WeightCache::get` (Phase 1 only).

**Prometheus text format example** (what `GET /metrics` returns):

```
# HELP burn_inference_requests_total Total inference requests by status
# TYPE burn_inference_requests_total counter
burn_inference_requests_total{status="success"} 1024
burn_inference_requests_total{status="error"} 3
burn_inference_requests_total{status="timeout"} 1
# HELP burn_inference_request_duration_seconds End-to-end request latency in seconds
# TYPE burn_inference_request_duration_seconds histogram
burn_inference_request_duration_seconds_bucket{le="0.05"} 12
burn_inference_request_duration_seconds_bucket{le="0.5"} 400
burn_inference_request_duration_seconds_bucket{le="1.0"} 750
burn_inference_request_duration_seconds_bucket{le="+Inf"} 1024
burn_inference_request_duration_seconds_sum 312.7
burn_inference_request_duration_seconds_count 1024
# HELP burn_inference_tokens_generated_total Total decode tokens generated
# TYPE burn_inference_tokens_generated_total counter
burn_inference_tokens_generated_total 204800
# HELP burn_inference_tokens_per_second Rolling average decode tokens per second
# TYPE burn_inference_tokens_per_second gauge
burn_inference_tokens_per_second 42.3
```

---

### 3.4 Benchmark Suite — `burn-inference-bench`

**Crate path:** `inference-bench/`  
**Binary:** `burn-inference-bench`

The benchmark binary connects to a running `burn-inference` server over HTTP and drives it with synthetic requests. It does not link against `inference-engine` or any GPU code.

#### 3.4.1 Benchmark Scenarios

```rust
// inference-bench/src/main.rs
use clap::{Parser, Subcommand};

#[derive(Parser)]
pub struct Cli {
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    pub server: String,

    #[arg(long, default_value = "results.json")]
    pub output: String,

    #[command(subcommand)]
    pub bench: BenchCommand,
}

#[derive(Subcommand)]
pub enum BenchCommand {
    /// TTFT, cold radix cache, batch 1
    TtftCold,
    /// TTFT, warm radix cache (512-token shared prefix), batch 1
    TtftWarm,
    /// Decode throughput at batch 1, 4, 8, 16, 32
    Throughput,
    /// 100 requests with 1024-token shared system prompt; report cache hit rate
    SharedPrefix,
    /// Single 8192-token prompt; report TTFT and decode tok/s
    LongContext,
    /// Ramp 1 to 32 concurrent requests; report P50/P99 latency
    Concurrent,
    /// Compare colocated vs disaggregated at 16 concurrent requests
    Disagg,
}
```

#### 3.4.2 Scenario Specifications

| Scenario | Prompt | Max tokens | Concurrency | Repetitions | Metric |
|----------|--------|------------|-------------|-------------|--------|
| `ttft-cold` | 128-token synthetic (no shared prefix) | 1 | 1 | 10 | TTFT (ms) |
| `ttft-warm` | 512-token shared prefix + 16-token tail | 1 | 1 | 10 | TTFT (ms) |
| `throughput` | 128-token prompt | 128 | 1, 4, 8, 16, 32 | 3 per batch | tok/s |
| `shared-prefix` | 1024-token system prompt + 32-token tail | 64 | 1 | 100 | cache hit rate (%), tok/s |
| `long-context` | 8192-token prompt | 64 | 1 | 3 | TTFT (ms), tok/s |
| `concurrent` | 128-token prompt | 64 | 1..32 (step 4) | 3 per step | P50/P99 latency (ms) |
| `disagg` | 128-token prompt | 128 | 16 | 5 | tok/s colocated vs disagg |

**Warm cache protocol for `ttft-warm`:** Send the shared prefix as a dummy request with `max_tokens=1` before starting the timed measurement. Verify that the radix cache hit count increased (via `GET /metrics`) before running the timed request.

**Disagg comparison protocol for `disagg`:** Run 5 repetitions with `[disaggregation] enabled = false`, record throughput. Restart server with `[disaggregation] enabled = true`, run 5 more repetitions. Report both numbers and the percentage gain.

#### 3.4.3 Output Format

```json
{
  "timestamp": "2026-04-04T10:00:00Z",
  "server_version": "0.1.0",
  "platform": "linux",
  "model": "gemma4-26b-moe-q4_k_m",
  "results": [
    {
      "scenario": "ttft-cold",
      "unit": "ms",
      "values": [3120, 3085, 3201, 3099, 3150, 3080, 3210, 3090, 3140, 3100],
      "p50": 3115,
      "p90": 3195,
      "p99": 3210
    },
    {
      "scenario": "throughput",
      "concurrency": 8,
      "unit": "tok/s",
      "values": [25.1, 25.4, 24.9],
      "mean": 25.1,
      "stddev": 0.25
    }
  ]
}
```

Human-readable table printed to stdout:

```
Benchmark Results — Linux (Gemma 4 26B MoE Q4_K_M)
=====================================================
Scenario           Metric        Value     Target    Pass?
-----------        ----------    --------  --------  -----
ttft-cold          TTFT P50 ms   3,115     < 4,000   YES
ttft-warm          TTFT P50 ms   280       < 500     YES
throughput (b=1)   tok/s         5.8       >= 5      YES
throughput (b=8)   tok/s         25.1      >= 25     YES
throughput (b=8)   P99 lat ms    88        < 100     YES
disagg vs coloc    gain %        18.3%     >= 15%    YES
```

Variance check: if `stddev / mean > 0.05` for any metric across repetitions, the run is flagged with `"high_variance": true` and the benchmark reports a warning. The acceptance gate requires no `high_variance` flags.

---

### 3.5 Configuration File — `burn-inference.toml`

```toml
# burn-inference.toml
# All fields have defaults shown below. Fields marked REQUIRED must be set.

[server]
host                    = "127.0.0.1"
port                    = 8080
max_concurrent_requests = 32          # 0 = unlimited
request_timeout_secs    = 120
graceful_shutdown_secs  = 30

[model]
path    = "/path/to/model.gguf"       # REQUIRED
backend = "ggml-metal"                # "ggml-metal" | "wgpu-vulkan" | "wgpu-metal"

[scheduler]
max_prefill_tokens_per_iter = 4096
page_size                   = 16      # tokens per KV cache page
max_pages                   = 4096

[radix_cache]
enabled   = true
max_pages = 2048

[offload]                             # optional — omit section to disable
enabled          = false
kv_cache_dir     = "/tmp/burn-inference-kv"
max_layers_in_ram = 4
max_expert_slots  = 32               # Linux MoE only; ignored on macOS

[disaggregation]                      # optional — omit section to disable
enabled               = false
prefill_gpu_timeout_ms = 5
```

**Rust config struct:**

```rust
// inference-api/src/config.rs
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    pub server:          ServerConfig,
    pub model:           ModelConfig,
    pub scheduler:       SchedulerConfig,
    pub radix_cache:     RadixCacheConfig,
    pub offload:         Option<OffloadConfig>,
    pub disaggregation:  Option<DisaggConfig>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_requests: usize,
    #[serde(default = "default_request_timeout")]
    pub request_timeout_secs: u64,
    #[serde(default = "default_graceful_shutdown")]
    pub graceful_shutdown_secs: u64,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ModelConfig {
    pub path:    PathBuf,
    pub backend: BackendKind,
}

#[derive(Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum BackendKind {
    GgmlMetal,
    WgpuVulkan,
    WgpuMetal,
}

#[derive(Deserialize, Debug, Clone)]
pub struct SchedulerConfig {
    #[serde(default = "default_max_prefill")]
    pub max_prefill_tokens_per_iter: usize,
    #[serde(default = "default_page_size")]
    pub page_size: usize,
    #[serde(default = "default_max_pages")]
    pub max_pages: usize,
}

#[derive(Deserialize, Debug, Clone)]
pub struct RadixCacheConfig {
    #[serde(default = "bool_true")]
    pub enabled:   bool,
    #[serde(default = "default_radix_max_pages")]
    pub max_pages: usize,
}

#[derive(Deserialize, Debug, Clone)]
pub struct OffloadConfig {
    pub enabled:           bool,
    pub kv_cache_dir:      PathBuf,
    pub max_layers_in_ram: usize,
    #[serde(default = "default_expert_slots")]
    pub max_expert_slots:  usize,
}

#[derive(Deserialize, Debug, Clone)]
pub struct DisaggConfig {
    pub enabled:                bool,
    #[serde(default = "default_prefill_timeout")]
    pub prefill_gpu_timeout_ms: u64,
}

// Default functions
fn default_host()            -> String { "127.0.0.1".into() }
fn default_port()            -> u16    { 8080 }
fn default_max_concurrent()  -> usize  { 32 }
fn default_request_timeout() -> u64    { 120 }
fn default_graceful_shutdown()-> u64   { 30 }
fn default_max_prefill()     -> usize  { 4096 }
fn default_page_size()       -> usize  { 16 }
fn default_max_pages()       -> usize  { 4096 }
fn default_radix_max_pages() -> usize  { 2048 }
fn default_expert_slots()    -> usize  { 32 }
fn default_prefill_timeout() -> u64    { 5 }
fn bool_true()               -> bool   { true }
```

Config is loaded at startup with:

```rust
let config: Config = toml::from_str(
    &std::fs::read_to_string("burn-inference.toml")
        .expect("could not read burn-inference.toml")
).expect("invalid burn-inference.toml");
```

A `--config` CLI flag overrides the default path. Config is immutable after startup; no hot reload in Phase 2B.

---

## 4. Disaggregation Architecture

### 4.1 Component Diagram

```
                         HTTP Layer
                         (axum router)
                              |
                    POST /v1/chat/completions
                    POST /v1/completions
                              |
                 +------------v-------------+
                 |     AppState             |
                 |  (config, metrics, ...)  |
                 |                          |
                 |  if disagg disabled:     |
                 |    run_overlapped_loop() |  <-- Phase 0 path, unchanged
                 |                          |
                 |  if disagg enabled:      |
                 |    DisaggEngine          |
                 |    .request_tx.send(req) |
                 +------------+-------------+
                              |
             +----------------+------------------+
             |                                   |
   +---------v----------+           +------------v---------+
   |   PrefillWorker    |           |    DecodeWorker       |
   |                    |           |                       |
   |  sched_rx.recv()   |           |  kv_rx.pop()          |
   |  collect_batch()   |           |  register_kv_handoff()|
   |                    |           |                       |
   |  gpu.lock()        |           |  gpu.lock()           |
   |  (5ms timeout)     |           |  (no timeout)         |
   |                    |           |                       |
   |  ctx.run_prefill   |           |  ctx.run_decode_step  |
   |  _batch()          |           |                       |
   |                    |  KvHandoff|                       |
   |  kv_tx.push(h)  ---+---------->+  kv_rx.pop()         |
   +--------------------+           +----------------------+
             |                                   |
             +------------ Arc<Mutex<GPU>> ------+
                         (single shared GPU)
```

### 4.2 GPU Time-Sharing Policy

The GPU is represented as `Arc<Mutex<Box<dyn BackendHandle>>>`. Both workers share the same `Arc`. The decode worker acquires the mutex unconditionally (with `gpu.lock().await`). The prefill worker acquires with a 5ms timeout:

```rust
// PrefillWorker GPU acquisition
let gpu_result = tokio::time::timeout(
    Duration::from_millis(config.prefill_gpu_timeout_ms),
    self.gpu.lock(),
).await;

match gpu_result {
    Ok(guard)  => { /* proceed with prefill */ },
    Err(_)     => {
        // Decode worker needed the GPU. Yield and re-enqueue the batch.
        tokio::task::yield_now().await;
        continue;
    }
}
```

**Priority guarantee:** The decode worker never waits for the prefill worker. The prefill worker yields within 5ms if the decode worker is waiting for the GPU. This prevents TTFT from penalizing time-to-next-token for in-flight decode sequences.

**Starvation prevention:** The prefill worker is not permanently blocked. After yielding, it retries the next iteration of its loop. If the decode worker finishes its step before the prefill worker retries, the prefill worker acquires the GPU on the next attempt.

### 4.3 KV Handoff Flow

```
Prefill worker completes prefill for request R:
  ctx.run_prefill_batch(gpu, [R])
    -> allocates KV pages [P1, P2, P3, ..., Pk] in shared KvPool
    -> writes KV tensors into those pages via BackendHandle
    -> returns Vec<KvHandoff> {
         request_id: R,
         kv_pages:   [P1, P2, P3, ..., Pk],
         cached_len: len(R.prompt_tokens)
       }

PrefillWorker pushes handoff to SPSC ring buffer:
  kv_tx.push(handoff)   -- O(1), no allocation, no lock (rtrb)

DecodeWorker pops at start of each decode iteration:
  while let Some(h) = kv_rx.pop():
    ctx.register_kv_handoff(h)
      -> marks request h.request_id as ready for decode
      -> records h.kv_pages as the KV state for that request
      -> sets ctx.cached_len[h.request_id] = h.cached_len

DecodeWorker runs decode step:
  ctx.run_decode_step(gpu)
    -> for each request with status == ReadyForDecode:
         fetches KV from h.kv_pages
         generates next token
         appends new KV to the page list
```

**Page ownership:** KV pages are allocated in the shared `KvPool` by the prefill worker and owned by the request. The decode worker reads and appends to the same pages. No copy of KV data occurs — the handoff passes page indices only. This is why correctness test T-02 (Section 6) verifies page contents rather than just token outputs.

### 4.4 Colocated vs Disaggregated: Decision at Startup

```rust
// inference-api/src/app.rs
let engine_handle: EngineHandle = match &config.disaggregation {
    Some(d) if d.enabled => {
        let disagg = DisaggEngine::start(ctx.clone(), backend);
        EngineHandle::Disagg(disagg)
    }
    _ => {
        // Phase 0 colocated path
        EngineHandle::Colocated(ColocatedEngine::start(ctx.clone(), backend))
    }
};
```

`EngineHandle` is an enum used by all HTTP route handlers. Both variants implement the same `submit_request` / `await_completion` interface, so route handlers are unaware of which mode is active.

---

## 5. Data Flow

### 5.1 Colocated Path (disaggregation disabled)

This is the Phase 0 path, unchanged in Phase 2B:

```
Client
  |-- POST /v1/chat/completions -->
  |                                HTTP handler
  |                                  |
  |                                  v
  |                              Tokenizer (TokenizerService)
  |                                  |
  |                                  v
  |                              RadixCache.lookup(tokens)
  |                              -- hit: reuse KV pages, skip prefill tokens
  |                              -- miss: full prefill
  |                                  |
  |                                  v
  |                              Scheduler.schedule_batch()
  |                                  |
  |                                  v
  |                              run_overlapped_loop():
  |                                loop:
  |                                  BackendHandle.forward_prefill()
  |                                  BackendHandle.forward_decode()
  |                                  yield tokens via channel
  |                                  |
  |<-- streaming SSE tokens --------+
  |<-- final JSON response ----------+
```

### 5.2 Disaggregated Path (disaggregation enabled)

```
Client
  |-- POST /v1/chat/completions -->
  |                                HTTP handler
  |                                  |
  |                            [request_id middleware: assign X-Request-Id]
  |                            [timeout middleware: start 120s timer]
  |                            [rate limit middleware: check concurrent cap]
  |                                  |
  |                                  v
  |                              Tokenizer (TokenizerService)
  |                                  |
  |                                  v
  |                              RadixCache.lookup(tokens)
  |                                  |
  |                                  v
  |                              Request assembled with:
  |                                prompt_tokens, cached_pages, status=PendingPrefill
  |                                  |
  |                                  v
  |                          DisaggEngine.request_tx.send(request)
  |                                  |
  |                    +-------------+
  |                    |
  |           PrefillWorker
  |             |
  |             |  collect_batch(): drain up to 32 requests from sched_rx
  |             |
  |             |  gpu = tokio::time::timeout(5ms, gpu.lock()).await
  |             |  -- if timeout: yield, retry next loop iteration
  |             |
  |             |  ctx.run_prefill_batch(gpu, batch):
  |             |    for each req:
  |             |      BackendHandle.forward_prefill(req.prompt_tokens)
  |             |      -> writes KV to KV pool pages
  |             |      -> returns KvHandoff {request_id, kv_pages, cached_len}
  |             |
  |             |  kv_tx.push(handoff)  -- SPSC ring buffer, O(1)
  |             |
  |         DecodeWorker
  |             |
  |             |  kv_rx.pop() for all pending handoffs
  |             |  ctx.register_kv_handoff(h) for each
  |             |    -> request status: PendingPrefill -> ReadyForDecode
  |             |
  |             |  gpu = gpu.lock().await  -- no timeout, decode has priority
  |             |
  |             |  ctx.run_decode_step(gpu):
  |             |    for each ReadyForDecode request:
  |             |      BackendHandle.forward_decode(kv_pages, last_token)
  |             |      -> appends one new KV page entry
  |             |      -> produces one output token
  |             |      -> if stop condition: request status -> Finished
  |             |
  |             |  finished tokens sent via per-request response channel
  |             |
  |<-- token response channel --------+
  |                                  |
  |                          HTTP handler completes:
  |                            waits on response channel
  |                            assembles JSON / SSE
  |<-- HTTP response ----------------+
```

### 5.3 Request Status State Machine

```
New
 |
 v
PendingPrefill   <-- request submitted to DisaggEngine
 |
 v (after PrefillWorker runs forward_prefill)
HandoffSent      <-- KvHandoff pushed to ring buffer
 |
 v (after DecodeWorker pops and registers handoff)
ReadyForDecode
 |
 v (after each decode step)
Decoding         <-- cycling per token
 |
 v (EOS token or max_tokens reached)
Finished
```

In colocated mode, the states `PendingPrefill`, `HandoffSent`, and `ReadyForDecode` are collapsed — the scheduler manages prefill and decode in the same overlapped loop without explicit state transitions.

---

## 6. Test Specification

All Phase 0, Phase 1, and Phase 2A tests must continue to pass without modification. This section specifies only the new tests added in Phase 2B.

### T-01: Disaggregation Output Correctness

**File:** `inference-disagg/tests/disagg_correctness.rs`

**Purpose:** Verify that running two prompts through the disaggregated engine produces output identical to running the same prompts through the colocated engine (same model, same seed, same sampling parameters).

**Setup:**
- Use `StubBackend` (from Phase 0) to make the test hermetic and GPU-free.
- Seed the RNG with a fixed value.
- Prompts: `["The capital of France is", "1 + 1 ="]`.
- `max_tokens = 16`, `temperature = 0.0` (greedy).

```rust
#[tokio::test]
async fn test_disagg_output_equals_colocated() {
    let prompts = vec![
        "The capital of France is",
        "1 + 1 =",
    ];
    let params = SamplingParams {
        max_tokens:  16,
        temperature: 0.0,
        top_p:       1.0,
    };

    // Run colocated (Phase 0 path).
    let colocated_outputs: Vec<String> = run_colocated(&prompts, &params).await;

    // Run disaggregated.
    let disagg_outputs: Vec<String> = run_disaggregated(&prompts, &params).await;

    assert_eq!(colocated_outputs.len(), disagg_outputs.len());
    for (i, (colocated, disagg)) in colocated_outputs.iter().zip(&disagg_outputs).enumerate() {
        assert_eq!(
            colocated, disagg,
            "prompt {i}: disagg output differs from colocated.
  colocated: {colocated:?}
  disagg:    {disagg:?}"
        );
    }
}
```

**Assertion:** `disagg_outputs[i] == colocated_outputs[i]` for both prompts. Token-for-token identical, not just semantically similar.

---

### T-02: KV Handoff Page Integrity

**File:** `inference-disagg/tests/kv_handoff.rs`

**Purpose:** Verify that the KV pages transferred from the prefill worker to the decode worker are not corrupted (no byte-level errors, correct page indices, correct `cached_len`).

```rust
#[tokio::test]
async fn test_kv_handoff_page_integrity() {
    let (ctx, gpu, kv_pool) = setup_stub_engine();
    let (kv_tx, kv_rx) = new_kv_handoff_queue();

    let prompt_tokens = tokenize("The quick brown fox");
    let request_id = RequestId::new();

    // Simulate prefill: allocate pages and fill them with a known pattern.
    let handoff = {
        let gpu = gpu.lock().await;
        let mut ctx = ctx.lock().await;
        ctx.run_prefill_single(&*gpu, request_id, &prompt_tokens)
    };

    // Push handoff to queue.
    kv_tx.push(handoff.clone()).unwrap();

    // Pop from queue and verify.
    let received = kv_rx.pop().unwrap();
    assert_eq!(received.request_id, handoff.request_id);
    assert_eq!(received.kv_pages, handoff.kv_pages);
    assert_eq!(received.cached_len, prompt_tokens.len(),
        "cached_len must equal prompt token count");

    // Verify page contents match what was written by prefill.
    for &page_idx in &received.kv_pages {
        let page_data = kv_pool.read_page(page_idx);
        assert!(
            page_data.iter().any(|&b| b != 0),
            "page {page_idx} is all zeros — prefill did not write KV data"
        );
    }
}
```

**Assertions:**
- `received.request_id == handoff.request_id`
- `received.kv_pages == handoff.kv_pages`
- `received.cached_len == prompt_tokens.len()`
- Every transferred page contains non-zero data (confirms prefill wrote KV tensors into the pages).

---

### T-03: Graceful Shutdown Drains In-Flight Requests

**File:** `inference-api/tests/graceful_shutdown.rs`

**Purpose:** Verify that in-flight requests complete before the server exits when SIGTERM is received.

```rust
#[tokio::test]
async fn test_graceful_shutdown_drains_requests() {
    let server = spawn_test_server().await;

    // Submit a request that takes ~500ms to complete.
    let response_future = tokio::spawn({
        let addr = server.addr();
        async move {
            reqwest::Client::new()
                .post(format!("http://{addr}/v1/chat/completions"))
                .json(&slow_request())
                .send()
                .await
                .unwrap()
        }
    });

    // Wait 100ms, then send SIGTERM.
    tokio::time::sleep(Duration::from_millis(100)).await;
    server.send_sigterm();

    // The in-flight request must complete with HTTP 200, not be dropped.
    let response = tokio::time::timeout(
        Duration::from_secs(5),
        response_future,
    )
    .await
    .expect("response future timed out")
    .unwrap();

    assert_eq!(response.status(), 200,
        "in-flight request should complete with 200, not be dropped on shutdown");

    // Server should exit cleanly within drain timeout.
    let exited = tokio::time::timeout(
        Duration::from_secs(35),
        server.wait_for_exit(),
    )
    .await;
    assert!(exited.is_ok(), "server did not exit within graceful shutdown timeout");
}
```

**Assertions:**
- In-flight request receives HTTP 200 (not a connection reset or 503).
- Server process exits within `graceful_shutdown_secs + 5s` after SIGTERM.

---

### T-04: Health Endpoint

**File:** `inference-api/tests/health.rs`

```rust
#[tokio::test]
async fn test_health_endpoint_returns_ok() {
    let server = spawn_test_server().await;
    let resp = reqwest::get(format!("http://{}/health", server.addr()))
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok",
        "health endpoint must return {{"status": "ok"}}");
}
```

**Assertions:**
- HTTP status 200.
- Response body is exactly `{"status": "ok"}`.

---

### T-05: Metrics Endpoint Returns Valid Prometheus Text Format

**File:** `inference-api/tests/metrics.rs`

```rust
#[tokio::test]
async fn test_metrics_endpoint_valid_prometheus() {
    let server = spawn_test_server().await;

    // Fire one successful request to generate some metrics.
    let _ = reqwest::Client::new()
        .post(format!("http://{}/v1/chat/completions", server.addr()))
        .json(&minimal_chat_request())
        .send()
        .await
        .unwrap();

    let resp = reqwest::get(format!("http://{}/metrics", server.addr()))
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);

    let content_type = resp.headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        content_type.starts_with("text/plain"),
        "content-type must be text/plain, got {content_type}"
    );

    let body = resp.text().await.unwrap();

    // Required metric names must be present.
    let required_metrics = [
        "burn_inference_requests_total",
        "burn_inference_request_duration_seconds",
        "burn_inference_tokens_generated_total",
        "burn_inference_radix_cache_hits_total",
        "burn_inference_radix_cache_misses_total",
        "burn_inference_kv_pool_pages_used",
        "burn_inference_kv_pool_pages_total",
    ];
    for metric in required_metrics {
        assert!(
            body.contains(metric),
            "metrics output missing required metric: {metric}"
        );
    }

    // After one successful request, requests_total{status="success"} must be >= 1.
    assert!(
        body.contains(r#"burn_inference_requests_total{status="success"}"#),
        "success counter missing after one successful request"
    );
}
```

**Assertions:**
- HTTP status 200 with `Content-Type: text/plain`.
- All required metric names present in response body.
- `burn_inference_requests_total{status="success"}` present and >= 1 after one request.

---

### T-06: CORS Preflight

**File:** `inference-api/tests/cors.rs`

```rust
#[tokio::test]
async fn test_cors_preflight_returns_correct_headers() {
    let server = spawn_test_server().await;

    let resp = reqwest::Client::new()
        .request(
            reqwest::Method::OPTIONS,
            format!("http://{}/v1/chat/completions", server.addr()),
        )
        .header("Origin", "https://example.com")
        .header("Access-Control-Request-Method", "POST")
        .header("Access-Control-Request-Headers", "content-type")
        .send()
        .await
        .unwrap();

    // CORS preflight must return 200 (some implementations return 204).
    assert!(
        resp.status() == 200 || resp.status() == 204,
        "CORS preflight must return 200 or 204, got {}",
        resp.status()
    );

    let headers = resp.headers();
    assert!(
        headers.contains_key("access-control-allow-origin"),
        "CORS response missing Access-Control-Allow-Origin header"
    );
    assert!(
        headers.contains_key("access-control-allow-methods"),
        "CORS response missing Access-Control-Allow-Methods header"
    );
}
```

**Assertions:**
- HTTP status 200 or 204.
- `Access-Control-Allow-Origin` header present.
- `Access-Control-Allow-Methods` header present.

---

### T-07: Request Timeout Returns 408

**File:** `inference-api/tests/timeout.rs`

```rust
#[tokio::test]
async fn test_request_timeout_returns_408() {
    // Start server with 1-second timeout for this test.
    let server = spawn_test_server_with_timeout(1).await;

    // Submit a request to the stub backend that takes 3 seconds.
    let resp = reqwest::Client::new()
        .post(format!("http://{}/v1/chat/completions", server.addr()))
        .json(&request_with_delay_secs(3))
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(), 408,
        "request that exceeds timeout must return 408, got {}",
        resp.status()
    );
}
```

**Assertions:**
- HTTP status 408 (Request Timeout) when the request duration exceeds `request_timeout_secs`.

---

### T-08: `/v1/completions` Semantic Correctness

**File:** `inference-api/tests/completions_endpoint.rs`

```rust
#[tokio::test]
async fn test_v1_completions_beijing() {
    let server = spawn_test_server().await;
    let resp = reqwest::Client::new()
        .post(format!("http://{}/v1/completions", server.addr()))
        .json(&serde_json::json!({
            "model": "test-model",
            "prompt": "The capital of China is",
            "max_tokens": 8,
            "temperature": 0.0
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();

    assert_eq!(body["object"], "text_completion");
    assert!(body["choices"].is_array());
    assert_eq!(body["choices"][0]["index"], 0);

    let text = body["choices"][0]["text"].as_str().unwrap();
    assert!(
        text.to_lowercase().contains("beijing"),
        "completion for 'The capital of China is' must contain 'beijing', got: {text:?}"
    );
}

#[tokio::test]
async fn test_v1_completions_arithmetic() {
    let server = spawn_test_server().await;
    let resp = reqwest::Client::new()
        .post(format!("http://{}/v1/completions", server.addr()))
        .json(&serde_json::json!({
            "model": "test-model",
            "prompt": "1 + 1 =",
            "max_tokens": 4,
            "temperature": 0.0
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    let text = body["choices"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("2"),
        "completion for '1 + 1 =' must contain '2', got: {text:?}"
    );
}

#[tokio::test]
async fn test_v1_completions_usage_fields_present() {
    let server = spawn_test_server().await;
    let resp = reqwest::Client::new()
        .post(format!("http://{}/v1/completions", server.addr()))
        .json(&serde_json::json!({
            "model": "test-model",
            "prompt": "Hello",
            "max_tokens": 4
        }))
        .send()
        .await
        .unwrap();

    let body: serde_json::Value = resp.json().await.unwrap();
    let usage = &body["usage"];
    assert!(usage["prompt_tokens"].as_u64().unwrap_or(0) > 0);
    assert!(usage["completion_tokens"].as_u64().unwrap_or(0) > 0);
    assert_eq!(
        usage["total_tokens"],
        usage["prompt_tokens"].as_u64().unwrap() + usage["completion_tokens"].as_u64().unwrap()
    );
}
```

**Assertions for `test_v1_completions_beijing`:**
- HTTP status 200.
- `body.object == "text_completion"`.
- `body.choices[0].text` contains the string `"beijing"` (case-insensitive).

**Assertions for `test_v1_completions_arithmetic`:**
- HTTP status 200.
- `body.choices[0].text` contains the string `"2"`.

**Assertions for `test_v1_completions_usage_fields_present`:**
- `prompt_tokens > 0`.
- `completion_tokens > 0`.
- `total_tokens == prompt_tokens + completion_tokens`.

---

## 7. Acceptance Criteria

All criteria below are hard gates. Phase 2B is not complete unless every item passes.

### 7.1 Regression: All Prior Tests Pass

- [ ] All Phase 0 cargo tests pass with no modifications (`cargo test -p inference-engine -p inference-api -p inference-backend-stub`)
- [ ] All Phase 1 cargo tests pass (`cargo test -p inference-backend-wgpu-offload -p inference-model-gemma4-moe`)
- [ ] All Phase 2A cargo tests pass (`cargo test -p ggml-sys -p inference-backend-ggml -p inference-model-gemma4-dense`)

### 7.2 New Tests Pass

- [ ] T-01: `test_disagg_output_equals_colocated` passes on both platforms
- [ ] T-02: `test_kv_handoff_page_integrity` passes
- [ ] T-03: `test_graceful_shutdown_drains_requests` passes
- [ ] T-04: `test_health_endpoint_returns_ok` passes
- [ ] T-05: `test_metrics_endpoint_valid_prometheus` passes
- [ ] T-06: `test_cors_preflight_returns_correct_headers` passes
- [ ] T-07: `test_request_timeout_returns_408` passes
- [ ] T-08a: `test_v1_completions_beijing` passes on both platforms
- [ ] T-08b: `test_v1_completions_arithmetic` passes on both platforms
- [ ] T-08c: `test_v1_completions_usage_fields_present` passes

### 7.3 Performance Targets — Linux (26B MoE, Intel iGPU, burn-wgpu)

| Metric | Target | Gate |
|--------|--------|------|
| TTFT cold (batch 1) | < 4,000 ms | Hard |
| TTFT warm (batch 1) | < 500 ms | Hard |
| Decode throughput (batch 1) | >= 5 tok/s | Hard |
| Decode throughput (batch 8) | >= 25 tok/s | Hard |
| P99 decode latency (batch 8) | < 100 ms/token | Hard |
| Disaggregation throughput gain (16 concurrent) | >= 15% vs colocated | Hard |

### 7.4 Performance Targets — macOS (31B dense, Apple Silicon, burn-ggml)

| Metric | Target | Gate |
|--------|--------|------|
| TTFT cold (batch 1) | < 3,000 ms | Hard |
| TTFT warm (batch 1) | < 300 ms | Hard |
| Decode throughput (batch 1) | >= 10 tok/s | Hard |
| Decode throughput (batch 8) | >= 45 tok/s | Hard |
| P99 decode latency (batch 8) | < 60 ms/token | Hard |
| Long-context TTFT (8192-token prompt) | < 30,000 ms | Hard |

### 7.5 Benchmark Reproducibility

- [ ] All `burn-inference-bench` scenarios produce variance < 5% (stddev/mean < 0.05) across 3 runs on each platform
- [ ] No scenario outputs `"high_variance": true` in the JSON report
- [ ] JSON output is valid according to the schema in Section 3.4.3

### 7.6 Code Quality

- [ ] `cargo clippy -- -D warnings` passes on all crates (both Linux and macOS feature flags)
- [ ] `cargo fmt --check` passes on all crates
- [ ] `cargo build --release` succeeds on Linux (wgpu backend) with no manual steps beyond `cargo build`
- [ ] `cargo build --release` succeeds on macOS (ggml backend) with no manual steps beyond `git submodule update --init && cargo build`

### 7.7 Documentation

- [ ] `README.md` present at repo root with: quickstart section, installation instructions, model download instructions for both platforms, at least 3 example `curl` commands (`/v1/chat/completions`, `/health`, `/metrics`)
- [ ] `BENCHMARKS.md` present with results tables for both Linux (Intel iGPU) and macOS (Apple Silicon M3), using actual measured values from the final benchmark run

---

## 8. Non-Goals

### 8.1 Explicitly Out of Scope Forever

These items will not be addressed in any future phase of this project:

- **Multi-GPU inference.** The design assumes a single GPU per machine. Tensor parallelism across multiple GPUs is not planned.
- **Windows support.** The project targets Linux (Vulkan/wgpu) and macOS (Metal/ggml) only. Win32 support is not planned.
- **iOS / tvOS / watchOS.** Mobile platforms are out of scope.
- **Training or fine-tuning.** This is an inference-only engine.
- **Models with more than 256K context window.** Longer contexts require fundamental changes to the KV paging design.

### 8.2 Deferred to Future Work (Phase 3+)

These items are explicitly deferred and not implied by Phase 2B:

- **Multi-machine disaggregation.** The current design is single-machine only (two Tokio tasks, one GPU, one process). Network-based KV transfer (e.g., RDMA or TCP) is Phase 3+ work.
- **Speculative decoding.** Requires a separate draft model; deferred.
- **Continuous batching improvements.** The Phase 0 `run_overlapped_loop` scheduler is unchanged. Chunked prefill, priority-aware scheduling, and preemption are deferred.
- **ANE (Apple Neural Engine) acceleration.** Metal-only in Phase 2A; ANE integration requires CoreML and is deferred.
- **Hot config reload.** Config is immutable after startup. SIGHUP-triggered reload is deferred.
- **OpenAI-compatible streaming on `/v1/completions`.** SSE streaming is already supported on `/v1/chat/completions`. Adding it to the legacy endpoint is deferred.
- **Authentication and TLS.** The server listens on plain HTTP. Reverse proxy (nginx, Caddy) handles TLS termination.
- **Prometheus push gateway / remote write.** Only the pull-based `/metrics` scrape endpoint is implemented. Push-based metrics export is deferred.
- **Quantization schemes beyond Q3_K_M and Q4_K_M.** Q2_K, Q8_0, F16, and other formats are deferred.
- **Models beyond Gemma 4 26B MoE and 31B dense.** Other model architectures are deferred.
- **Disaggregation over multiple GPU machines.** Current architecture is single-machine only.
- **Expert parallelism for MoE.** All experts reside on one GPU (with SSD offload); distributing experts across GPUs is Phase 3+.

---

## 9. Dependencies

### 9.1 New Crates Added in Phase 2B

| Crate | Version | Use | Feature flags |
|-------|---------|-----|---------------|
| `rtrb` | 0.3+ | Lock-free SPSC ring buffer for `KvHandoffQueue` | default |
| `prometheus` | 0.13+ | Prometheus metrics registry, counters, histograms, gauges | default |
| `tower-http` | 0.5+ | CORS middleware (`CorsLayer`), request timeout | `cors`, `timeout` |
| `tower` | 0.4+ | `TimeoutLayer` | `timeout` |
| `toml` | 0.8+ | Config file deserialization | default |
| `uuid` | 1.x | Request ID generation | `v4` |
| `once_cell` | 1.x | Lazy global metric registry initialization | default |
| `parking_lot` | 0.12+ | Mutex wrapping `rtrb` producer/consumer halves | default |
| `reqwest` | 0.12+ | Benchmark binary HTTP client | `json`, `blocking` feature not used |

### 9.2 Existing Crates — Phase 2B Additions

| Crate | Prior use | Phase 2B addition |
|-------|-----------|-------------------|
| `tokio` | Async runtime, mpsc | Add `time::timeout`, signal handling (`SIGTERM`) |
| `axum` | HTTP routing | Add new route handlers; `with_graceful_shutdown` |
| `serde` / `serde_json` | Serialization | No change |
| `clap` | CLI flags | Add `--config` flag; add `BenchCommand` subcommands |
| `anyhow` | Error handling | No change |
| `tracing` | Structured logging | Add tracing spans in PrefillWorker and DecodeWorker |

### 9.3 Crate Feature Flag Decisions

**`rtrb` vs `crossbeam::ArrayQueue`:**
`rtrb` is preferred because it is purpose-built for SPSC use (single producer, single consumer), has bounded worst-case latency, and produces no allocations on push/pop. `crossbeam::ArrayQueue` is MPMC and uses CAS loops that are suboptimal for SPSC. The `rtrb` feature flag allows swapping to `crossbeam` if `rtrb` cannot be compiled on a target.

**`prometheus` crate:**
The `prometheus` crate (from the official Prometheus Rust client) is chosen over `metrics` + `metrics-exporter-prometheus` because it provides direct control over the text format encoding and allows a custom `Registry` (needed to avoid polluting the global registry in tests).

**`tower-http` CORS:**
`tower-http::cors::CorsLayer` is the canonical solution for CORS in the axum ecosystem. It handles `OPTIONS` preflight automatically with no custom route handler required.

### 9.4 No New Native Dependencies

Phase 2B adds no new C/C++ libraries, no new `build.rs` steps, and no new `git submodule` entries. All new dependencies are pure Rust crates available on crates.io.

### 9.5 Cargo.toml Changes

```toml
# inference-disagg/Cargo.toml  (new crate)
[dependencies]
tokio          = { version = "1", features = ["full"] }
inference-engine = { path = "../inference-engine" }
inference-backend = { path = "../inference-backend" }
parking_lot    = "0.12"
tracing        = "0.1"
anyhow         = "1"

[dependencies.rtrb]
version  = "0.3"
optional = true

[features]
default = ["rtrb"]
```

```toml
# inference-api/Cargo.toml  (additions only)
[dependencies]
prometheus   = "0.13"
tower-http   = { version = "0.5", features = ["cors", "timeout"] }
tower        = { version = "0.4", features = ["timeout"] }
toml         = "0.8"
uuid         = { version = "1", features = ["v4"] }
once_cell    = "1"
parking_lot  = "0.12"
inference-disagg = { path = "../inference-disagg", optional = true }

[features]
disagg = ["dep:inference-disagg"]
```

```toml
# inference-bench/Cargo.toml  (new crate)
[dependencies]
tokio    = { version = "1", features = ["full"] }
reqwest  = { version = "0.12", features = ["json"] }
clap     = { version = "4", features = ["derive"] }
serde    = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow   = "1"

[[bin]]
name = "burn-inference-bench"
path = "src/main.rs"
```

---

## 10. Open Questions

### OQ-01: SPSC Queue Capacity

The `KvHandoffQueue` ring buffer is sized at 64 slots (`QUEUE_CAPACITY = 64`). At a maximum of 32 concurrent requests, this is 2x the worst-case outstanding handoffs. The question is whether the prefill worker can outpace the decode worker enough to fill the buffer.

**Scenario:** If prefill takes 3s per request (cold, Linux) and decode takes 200ms per token, with 32 concurrent requests all completing prefill simultaneously, the ring buffer needs to hold 32 entries. 64 slots provides a 2x safety margin.

**Resolution needed:** Profile on real hardware to confirm that `push` never returns `Err` in practice under the target workload. If it does, increase `QUEUE_CAPACITY` to 128 or add a retry-with-backoff in the prefill worker. The current implementation already retries via `yield_now`, but this burns a Tokio scheduling slot.

### OQ-02: Disaggregation Benefit on macOS

The 15% throughput gain target for disaggregation is specified for Linux only. On macOS with the ggml Metal backend, the decode latency is already very low (< 60ms P99 per token at batch 8), and the prefill time is also shorter. It is unclear whether disaggregation provides any benefit on macOS, or whether the overhead of the two-task scheduling model erases any gain.

**Resolution needed:** Run the `bench disagg` scenario on macOS M3 to measure actual effect. If disaggregation shows no benefit or negative benefit on macOS, the performance target for disaggregation gain should be Linux-only in the acceptance criteria. The correctness test T-01 still applies to both platforms regardless of throughput.

### OQ-03: GPU Mutex Contention Under High Concurrency

At 32 concurrent requests, the decode worker runs decode steps back-to-back with minimal gaps. The prefill worker has a 5ms timeout window to acquire the GPU. Under high load, the prefill worker may be starved for extended periods if each decode step takes < 5ms (which is possible at low batch sizes on macOS).

**Resolution needed:** Measure actual decode step duration at batch 32 on both platforms. If decode steps are consistently < 5ms, increase `prefill_gpu_timeout_ms` to 10ms or 20ms, or implement a token-budget-based yielding policy where the decode worker voluntarily releases the GPU every N tokens to allow prefill to proceed.

### OQ-04: `GET /metrics` Scrape Latency Under Load

The `prometheus::TextEncoder` gathers and serializes all metric families on each scrape. Under high request load with many counters, this serialization could take > 1ms and block the HTTP thread.

**Resolution needed:** Benchmark `GET /metrics` scrape duration under the 32-concurrent-request load scenario. If scrape takes > 5ms, move metric serialization to a background task that pre-builds the response every 5 seconds and serves a cached copy.

### OQ-05: Graceful Shutdown Interaction with Disaggregated Engine

In colocated mode, graceful shutdown is handled entirely by axum's `with_graceful_shutdown`. In disaggregated mode, the PrefillWorker and DecodeWorker Tokio tasks must also be cancelled gracefully. If the shutdown signal is received while a prefill step is holding the GPU mutex, the decode worker must be able to complete its current step before both tasks exit.

**Resolution needed:** Implement a cancellation token (`tokio_util::CancellationToken`) shared between both workers and the HTTP server. On SIGTERM: (1) stop accepting new requests (axum), (2) signal CancellationToken, (3) both workers check token between iterations and exit cleanly. Confirm that the cancellation does not leave KV pages in a leaked state in the KvPool.

### OQ-06: `/v1/completions` Streaming

The Phase 2B spec explicitly defers SSE streaming on `/v1/completions`. However, some callers (OpenAI-compatible clients) may set `"stream": true` in their requests. The current implementation returns `400 Bad Request` in that case.

**Resolution needed:** Decide whether to return 400 (current plan), silently ignore `stream: true` and return non-streaming, or implement streaming before Phase 2B ships. The semantic tests T-08 do not cover streaming, but real-world usage may require it sooner than Phase 3.

### OQ-07: Prometheus Histogram Bucket Selection

The default latency histogram buckets are `[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]` seconds. These are appropriate for end-to-end request latency on slow hardware (Linux Intel iGPU with 3–4s cold TTFT). For macOS with < 1s typical latency, the resolution at the sub-second range may be insufficient.

**Resolution needed:** Consider adding finer-grained buckets at the low end for macOS: `[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]`. Alternatively, use per-platform bucket configuration in `burn-inference.toml`.

---

*End of Phase 2B Deliverable Specification*
