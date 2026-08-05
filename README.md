# llamad

In-process GGUF model inference over channels — no HTTP, no sidecar.

```rust
use llamad::client::Client;

let client = Client::new("/path/to/model.gguf")?;
let text = client.complete_text("tell me a joke")?;
```

## Architecture

A `Client` owns an `Engine`: one loaded model plus two threads. Inference runs
on a dedicated OS thread with **slotted continuous batching** — up to
`LLAMAD_N_SLOTS` concurrent sequences share a single `LlamaContext`. The
inference loop reuses one `LlamaBatch` (cleared per step) with all active
slots, decodes once, then samples per-slot.

**Concurrency is opt-in.** `LLAMAD_N_SLOTS` defaults to **1**, so a default
engine gives every request the whole `LLAMAD_N_CTX` (2048 tokens). Slots
partition that context *statically* — `N_CTX / N_SLOTS`, and an idle slot's
share is not lendable to a busy one. Raising `LLAMAD_N_SLOTS` to 4 therefore
does not add capacity; it cuts each request's prompt-plus-generation budget to
512 tokens. Raise `LLAMAD_N_CTX` alongside it, and only when you actually
issue concurrent requests.

```
┌──────────────┐   mpsc::channel    ┌────────────────────┐   mpsc::channel    ┌─────────────────────────────┐
│  your code   │ ─────────────────> │  preprocess thread │ ─────────────────> │  inference thread           │
│  (sync/async)│                    │  chat template +   │  (pre-tokenized)   │  owns model + ctx + slots   │
│              │ <oneshot::channel> │  tokenize + budget │                    │  continuous batching loop   │
└──────────────┘                    └────────────────────┘                    └─────────────────────────────┘
```

Engines are independent: any number can be alive at once, each with its own
model, context and slots. An engine joins both of its threads when dropped, so
`drop(client)` cancels in-flight requests and returns only once the model is
fully released. Teardown is bounded by one decode step even when every slot is
busy — a shutdown flag is checked per loop iteration, not just when the command
channel is drained.

**Slot lifecycle**: `empty → prefill (prompt phase) → generate → finish/cancel → empty`. A slot finishes on end-of-generation, the token cap, the per-slot budget, or a stop sequence.
- Each slot has a per-slot token budget (`N_CTX / N_SLOTS`, default 2048 — one slot holding all of `N_CTX`), configurable via `LLAMAD_N_CTX` / `LLAMAD_N_SLOTS`. A prompt that exceeds its slot's budget is rejected at preprocess time rather than truncated.
- When all `LLAMAD_N_SLOTS` slots are full, new requests queue in the `mpsc` channel and are dequeued as slots free up.
- **Slot routing is prefix-aware**: an incoming request goes to the free slot whose retained KV prefix shares the most tokens with it, not simply to the lowest free index. Without this, a request arriving while slot 0 is busy would full-prefill on slot 1 even when an idle slot held an exact prefix of it — the common shape for a repeated system prompt. Ties resolve to the lowest free index.
- KV cache is cleared per-sequence (`ctx.clear_kv_cache_seq`) on finish, cancel, and streaming client disconnect. When reuse is on (`LLAMAD_KV_CACHE=on` and the model supports partial KV rewind), a completed sequence's KV prefix is retained and reused on the next request into the same slot — but only while its length stayed within the per-slot budget; a sequence that hit the generation budget is fully cleared instead.

**Preprocessing**: chat templating, tokenization, and the per-slot budget
check run on a dedicated `llamad-preprocess` thread that shares the model via
`Arc<LlamaModel>` (immutable after load; `apply_chat_template` and
`str_to_token` are pure `&self` calls). A long prompt never stalls in-flight
generations on the inference thread.

### Structure

```
src/
├── lib.rs         Library root — crate docs, public modules
├── client.rs      High-level Client API, TokenStream (sync + async)
├── config.rs      InferenceConfig, LLAMAD_* env knobs, sane() clamping
├── protocol.rs    Request/Response types, LlamaError, InferCmd
├── preprocess.rs  Chat template + tokenization offload, sampling resolution, stop normalization
├── inference.rs   Engine, slotted inference loop, Slot lifecycle, inference_loop()
├── server.rs      Unix socket server (JSON/NDJSON over UDS)
└── main.rs        Daemon entry point (requires the `bin` feature, on by default)
```

### Design tradeoffs

These are deliberate choices, not oversights. Read them before embedding the
library — each one is the cost of something else it buys you.

- **No fault isolation: llama.cpp shares your process.** The library calls
  ggml through FFI on a thread inside the host process — that is what "no
  HTTP, no sidecar" means, and it is why there is no IPC hop or serialization
  on the request path. The consequence is that a fault inside the C++ backend
  (an out-of-bounds tensor access, a driver-level allocation failure, an
  uncaught abort) terminates the **whole host process**. It bypasses Rust
  destructors, so `Engine::drop` does not run and nothing is unwound.
  Rust-level errors are still returned normally as `LlamaError`; this applies
  only to faults below the FFI boundary. If your host process is doing
  something it cannot afford to lose, run the `llamad` daemon binary and talk
  to it over the Unix socket — that puts a process boundary back in, at the
  cost of the socket round-trip. The in-process path is the right default for
  CLI tools and local utilities, where the inference *is* the program.

- **Static context partitioning, not paged KV.** `N_CTX` is divided evenly
  across slots up front. There is no unified KV pool and no PagedAttention-style
  block allocator, so an idle slot's tokens cannot be lent to a busy one: with
  4 slots, a 1000-token prompt is rejected even when three slots are empty.
  Defaulting to a single slot keeps this from biting the common case; a
  multi-tenant backend with heterogeneous prompt sizes is not what this
  library is good at.

- **Prefill is not chunked.** `n_batch` is set to `n_ctx`, so a prompt is
  prefilled in one `decode` call. Under multi-slot load a long prompt entering
  one slot stalls token generation on the others for that step (ggml still
  splits the work into `n_ubatch` chunks internally, which bounds memory but
  not the stall). Prefill dominates TTFT for long prompts — see the batch
  table. Chunked prefill would fix the cross-slot jitter and is not
  implemented.

- **KV reuse is per-slot and non-persistent.** Prefix reuse compares against
  the cache retained by the *selected slot* (routing is prefix-aware, see
  above), not a global content-addressed prefix tree. Nothing survives engine
  shutdown.

## Usage

```rust
use llamad::client::Client;
use llamad::protocol::Request;

let client = Client::new("model.gguf")?;

// Simple text completion
let text = client.complete_text("tell me a joke")?;

// Full control with builder
let result = client.complete(
    Request::new("what is rust?")
        .with_system("answer concisely")
        .with_temperature(0.3)
        .with_max_tokens(200)
        .with_seed(42)              // omit for a fresh random seed per request
        .push_stop("\n\n")          // ends generation, excluded from the result
        .push_history("user", "previous question"),
)?;

// Streaming
let mut stream = client.complete_stream(Request::new("count to 5"))?;
while let Some(token) = stream.next_token() {
    print!("{token}");
}
let result = stream.into_result()?;   // ← returns InferResult with stats
```

`TokenStream` supports peek-as-you-go (`text()` for accumulated text so far, `finish()` to consume and return it) and `into_result()`, which blocks until the inference thread finishes and returns `InferResult { text, prompt_tokens, generated_tokens }`. Drop the `TokenStream` early to cancel streaming.

### Blocking vs. async

`complete`, `complete_text`, `TokenStream::next_token` and
`TokenStream::into_result` block the calling thread, and **panic if called
from inside a tokio runtime** — that is tokio's own guard against stalling a
reactor thread. Async callers use the `_async` variants:

```rust
let result = client.complete_async(Request::new("hi")).await?;

let mut stream = client.complete_stream(Request::new("count to 5"))?;
while let Some(token) = stream.next_token_async().await {
    print!("{token}");
}
let result = stream.into_result_async().await?;
```

`complete_stream` itself never blocks and is safe to call in either context.
`tokio::task::spawn_blocking` also works if you would rather keep the sync API.

### Binary / sidecar

```sh
cargo run --release -- /path/to/model.gguf [/tmp/llamad.sock]
```

Listens on a Unix socket (default `/tmp/llamad.sock`, mode `0600`) and accepts
JSON requests. Intended for out-of-process use from non-Rust consumers:

```sh
echo '{"prompt":"hi","system":"you are helpful","max_tokens":20}' | nc -N -U /tmp/llamad.sock
```

The daemon binary is behind the `bin` feature, enabled by default. Library
consumers should depend on the crate with `default-features = false` to drop
its argument-handling and log-subscriber dependencies.

## Protocol

One request per connection: write a JSON object, half-close the write side,
read the reply.

Fields: `prompt` (required), `system`, `max_tokens`, `temperature`, `top_k`, `top_p`, `repeat_penalty`, `repeat_last_n`, `seed`, `stop` (`[string]`), `stream` (bool), `history` (`[{role, content}]`). See [Generation parameters](#generation-parameters) for defaults and ranges. Unknown fields rejected. Request bodies are capped at 1 MiB (`server::MAX_REQUEST_BYTES`).

`Request` and `Response` both implement `Serialize` and `Deserialize`, so a
Rust consumer of the socket protocol can use the crate's own types rather than
hand-rolling the JSON. `None` fields are skipped on serialization, so a
round-trip reproduces the original request.

Non-streaming response:
```json
{"text":"2 + 2 equals 4.","prompt_tokens":24,"generated_tokens":8}
```

Streaming response (NDJSON):
```json
{"token":"2"}
{"token":" +"}
{"token":" 2"}
{"token":" equals"}
{"token":" 4"}
{"done":true,"text":"2 + 2 equals 4.","prompt_tokens":24,"generated_tokens":8}
```

Errors (both modes):
```json
{"error":"inference: ..."}
```

Other error forms: `invalid request` (parse failure or unknown field), `empty request`, `request too large`, `read timeout` (30 s to send a request), `stream timeout` (60 s without tokens), `inference thread unavailable`.

## Error handling

`LlamaError` (typed via `thiserror`) implements `Clone` for use across channel boundaries:

| Variant | Meaning |
|---|---|
| `ModelLoad(String)` | Model failed to load, or the llama.cpp backend failed to initialize |
| `InferenceCrashed` | Inference thread died, or the engine was shut down mid-request |
| `Inference(String)` | Runtime failure (batch overflow, chat template application, prompt-over-budget rejection, streaming disconnect, detokenization) |
| `Protocol(String)` | Message construction: null-byte / invalid role-or-content rejection (`NewLlamaChatMessageError`) |
| `Io(std::io::Error)` | Thread spawn or I/O failure |

`LlamaError` is `#[non_exhaustive]`; match with a `_` arm.

Streaming errors propagate through `done_tx` (`oneshot::Sender<Result<InferResult, LlamaError>>`) — the client calls `TokenStream::into_result()` which drains remaining tokens then checks the error. A dropped `done_tx` reports `InferenceCrashed`.

## Dependencies

- `llama-cpp-4` v0.4 with `default-features = false`; acceleration backends are opt-in cargo features (see below)
- `llama-cpp-sys-4` v0.4, only for the `ggml_log_level` type in the log callback (the `llama_log_set` call goes through `llama_cpp_4::log_set`)
- `tokio` for async UDS and channels (`net`, `sync`, `time`, `io-util`; the `bin` feature adds the multi-threaded runtime and signal handling)
- `serde` / `serde_json` for protocol
- `thiserror` for typed errors
- `num_cpus` for portable physical core detection
- `tracing` for instrumentation
- `anyhow` and `tracing-subscriber`, binary-only (behind the `bin` feature)
- `serial_test` (dev) for serializing model-loading tests

### Acceleration features

No backend is enabled by default, so a stock `cargo build` produces a binary
that runs on any machine with the same target triple. Opt in per deployment:

| Feature | Effect |
|---|---|
| `native` | Builds llama.cpp with `-march=native` — enables AVX-512/AVX2/FMA for the LFM2.5 GQA attention kernels. Fastest, but the binary may `SIGILL` on any other CPU model. |
| `openmp` | OpenMP threading inside ggml. |
| `prebuilt` | Downloads a prebuilt generic x86-64 ggml instead of compiling llama.cpp. Much faster to build; no arch-specific SIMD. |
| `cuda`, `metal`, `vulkan`, `hip`, `blas` | Forwarded to the corresponding `llama-cpp-4` feature. |

The benchmark numbers below were measured with `--features native`.

## Supported models

Tested with the **Liquid AI LFM2.5 family** (all available as GGUF on HuggingFace at `LiquidAI/<model-name>-GGUF`). The real-model test suite needs the GGUFs in `models/` (gitignored) — fetch them with `models/download.sh` (also grabs SmolLM2-135M and Qwen2.5-0.5B, the attention-only models used by the KV-reuse tests):

| Model | Params | Q4 size | Use case | Speed (i5-1135G7, 4 threads) |
|---|---|---|---|---|
| [LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M-GGUF) | 230M | 146 MB | Mechanical tasks: classification, routing, extraction, formatting | 37–70 tok/s |
| [LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF) | 1.2B | 731 MB | Reasoning: math, logic, multi-step problems | 15–16 tok/s |

KV prefix reuse is probe-gated per model: it engages on pure-attention GGUFs (e.g. SmolLM2, Qwen2.5 — any llama.cpp-supported attention arch) and degrades to full-prefill-per-request on hybrid/SSM/recurrent models like the LFM2.5 family (see "KV-reuse path" under Tests).

Single-request throughput measured 2026-07-31 on release builds (`cargo build --release --features native`, 4 physical threads, 256-token generations). Spread on the 230M is thermal/load variance between runs (avg 63 tok/s in a cool run, 39 tok/s in a warm one) — not code variance. Reproduce with the bundled benchmark harness: `cargo run --release --features native --example bench -- <model.gguf> <n_repeat> [system_prompt]` (see `examples/bench.rs`).

### Batch throughput and latency (i5-1135G7, 4 threads, `LLAMAD_N_SLOTS=4`)

| Model | Single req | 4 concurrent (aggregate) | TTFT short prompt | TTFT ~480-tok prompt |
|---|---|---|---|---|
| LFM2.5-230M | 37–70 tok/s | ~103 tok/s (4 × 25.7) | 0.14 s | 0.96 s |
| LFM2.5-1.2B-Thinking | 15–16 tok/s | ~30 tok/s (4 × 7.6) | 0.63 s | 5.3 s |

Continuous batching scales aggregate throughput ~1.6–1.9x over single-request decode under 4-slot load. TTFT for long prompts is prefill-dominated; tokenization no longer adds to it on the inference thread (see Performance notes).

Both models share the same architecture (LFM2.5 dense hybrid), chat template (ChatML), vocabulary (65K), and tool-use format (`<|tool_call_start|>` / `<|tool_call_end|>`).

## Generation parameters

Every sampling parameter is per-request and overridable. The **defaults** match
the Liquid AI recommended settings from the LFM2.5 model card — they are a
starting point for the bundled models, not a policy imposed on every GGUF:

| Field | Default | Range |
|---|---|---|
| `temperature` | 0.1 | clamped `[0, 2]`; `<= 0` selects greedy decoding |
| `top_k` | 50 | clamped `>= 0`; `0` disables |
| `top_p` | 0.95 | clamped `[0, 1]` |
| `repeat_penalty` | 1.05 | clamped `[0, 2]`; `1.0` disables |
| `repeat_last_n` | -1 (whole context) | clamped `>= -1`; `0` disables the lookback |
| `seed` | a fresh random seed per request | any `u32` |
| `max_tokens` | 256 | further capped by the per-slot budget |

Sampler chain: penalties → top-k → top-p → temperature → distribution. When
`temperature <= 0` a greedy sampler replaces the chain entirely, and the other
sampling fields do not apply.

Unparsable or out-of-range values are clamped rather than rejected; `NaN`
floats fall back to the default.

### Seeding

`seed` defaults to a **fresh random seed per request**, so two identical
requests at a non-zero temperature produce different text. Pin it for
reproducible output:

```rust
let req = Request::new("invent a sentence")
    .with_temperature(1.0)
    .with_seed(42);            // same text every run
```

Greedy decoding (`temperature <= 0`) is deterministic regardless and ignores
the seed.

### Stop sequences

Generation ends as soon as any stop sequence appears in the output. The matched
text is excluded from the result and is never streamed:

```rust
let req = Request::new("emit a tool call")
    .push_stop("<|tool_call_end|>")
    .push_stop("\n\n");
```

A stop sequence may span token boundaries — output that could still grow into
one is withheld from the stream until the match either completes or is ruled
out, so a partial match never leaks to the client. Text that looks nothing like
a stop sequence streams with no added latency. Empty stop strings are ignored.

## Performance notes

- **Preprocess offload**: chat templating and tokenization run on a dedicated
  thread (`llamad-preprocess`); a long prompt no longer stalls in-flight
  generations on the inference thread. TTFT for long prompts is prefill-
  dominated, not tokenization-dominated (see batch table above).
- **Slotted batching**: multiple requests decode simultaneously — measured
  ~1.6–1.9x aggregate throughput scaling over single-request decode under
  4-slot load (see batch table above). This is opt-in: the default is one slot
  holding all 2048 tokens of context, and the batch figures were taken with
  `LLAMAD_N_SLOTS=4` (4 slots × 512 tokens). Aggregate throughput scales
  because batched slots amortize one pass of weight reads across sequences;
  *per-sequence* speed still drops, so single-request latency is best at the
  default single slot.
- **Thread counts**: `n_threads` (decode) and `n_threads_batch` (prefill)
  default to `num_cpus::get_physical()` (not `thread::available_parallelism()`)
  to avoid hyperthread collapse. On the test hardware (i5-1135G7, 4 physical /
  8 logical cores), both default to 4. Prefill is GEMM-heavy and wants all
  cores; decode is GEMV-bound and may prefer fewer on small models — tune with
  `LLAMAD_N_THREADS` / `LLAMAD_N_THREADS_BATCH`.
- **mlock**: Explicitly disabled (`with_use_mlock(false)`). The default `true` fails on most Linux systems without root or `ulimit -l` elevation.
- **Logging**: llama.cpp C-level logs are routed through a custom `tracing` callback: errors → `tracing::error!`, warnings → `tracing::warn!`, info/debug → `tracing::debug!`.
- **Backend singleton**: `LlamaBackend::init()` runs exactly once per process, *inside* a `OnceLock` initializer so simultaneous engine startups cannot race it. Failure returns `LlamaError::ModelLoad` rather than panicking, and is cached — a backend that cannot initialize is an environmental fact, not a transient error worth retrying per engine.

## Security notes

- The Unix socket is chmod'ed to `0600` after bind: the default path lives in
  world-writable `/tmp`, and a permissive mode would let any local user submit
  inference requests.
- `bind_socket` reclaims a socket left behind by a crash, but refuses to
  unlink a path that is not a socket, and refuses to rebind a path a live
  daemon is still listening on.
- Request bodies are capped at 1 MiB so a local client cannot drive the daemon
  out of memory by writing without end.

## Two-model strategy

For agentic workloads, pair the models by task complexity:

1. **LFM2.5-230M** — fast mechanical sub-agent tasks (classify, route, extract, format). ~37–70 tok/s, 146 MB.
2. **LFM2.5-1.2B-Thinking** — reasoning that needs step-by-step thought (math, logic, planning). Outputs `<think>` chains internally, then answers. ~15–16 tok/s, 731 MB.

## Configuration

Environment variables (read once per engine at startup; unparsable values fall back to defaults, zero or absurd values are clamped):

| Variable | Default | Meaning |
|---|---|---|
| `LLAMAD_N_SLOTS` | 1 | Concurrent sequences (clamped 1–512); KV context is split *statically and evenly* across slots, so raising this divides each request's budget rather than adding capacity — raise `LLAMAD_N_CTX` with it |
| `LLAMAD_N_CTX` | 2048 | Total KV-context capacity in tokens, all slots (clamped to `n_slots`…1,048,576) |
| `LLAMAD_N_THREADS` | physical cores | Threads for single-token decode (clamped 1–256) |
| `LLAMAD_N_THREADS_BATCH` | physical cores | Threads for batched prompt processing, prefill (clamped 1–256) |
| `LLAMAD_KV_CACHE` | on | Reuse the KV prefix of completed sequences on the next request into the same slot (runtime-probed; disabled on models that cannot partially rewind KV; retention capped at the per-slot budget — sequences at/over the budget are fully cleared) |

## Tests

First fetch the model fixtures (re-runnable; skips already-downloaded files):

```sh
./models/download.sh                  # all four GGUFs (~1.4 GB)
./models/download.sh SmolLM2          # just the KV-reuse test model
```

Then:

```sh
cargo test --features native          # all tests
cargo test --lib                      # unit tests only (no model needed)
cargo test --test integration         # real-model integration tests (LFM2.5)
cargo test --test lifecycle           # engine/client lifecycle contracts
cargo test --test reuse               # KV-prefix-reuse path tests (attention model)
cargo test --test cancellation        # slot-recycling cancellation test
```

141 unit tests + 24 real-model tests (15 in `tests/integration.rs`, 5 in `tests/lifecycle.rs`, 3 in `tests/reuse.rs`, 1 in `tests/cancellation.rs`) + 4 doc tests. Real model tests use the GGUFs in `models/` (paths default via `CARGO_MANIFEST_DIR`; override per-model with `LLAMAD_TEST_MODEL_230M`, `LLAMAD_TEST_MODEL_1_2B`, or `LLAMAD_TEST_MODEL`). Every model-loading test is `#[serial]`-enforced (`serial_test`) within its binary, and cargo runs test binaries sequentially, so the heavy llama.cpp loads never contend — no `--test-threads` flags needed. Log-contract assertions (the "degrade loudly" warning, the reuse debug line) use a minimal capture `Layer` in `tests/common/mod.rs` (tracing-test was evaluated and rejected: its per-test subscriber filters to the test crate's targets, dropping library events).

### KV-reuse path (`LLAMAD_TEST_MODEL` / bundled SmolLM2)

The KV-prefix-reuse machinery (anchor, retention, fill reconciliation) engages only when the startup probe finds a model that supports partial KV rewind — which the repo's LFM2.5 hybrids do not (they run the degraded full-prefill path). The reuse-path tests in `tests/reuse.rs` run against `LLAMAD_TEST_MODEL` (an attention-only GGUF, e.g. Qwen2.5-0.5B) when set, else the bundled SmolLM2-135M in `models/` when present. With a rewind-capable model the identical-resend and partial-prefix tests exercise the real reuse path, and a dedicated test asserts the probe verdict is `true` on it (no degradation warning; reuse debug lines present). With neither model available, the reuse binary skips with a message — the degraded-path contract stays covered by `probe_verdict_degrades_loudly_on_lfm2` in `tests/integration.rs`.

### Test coverage

| Area | Tests | Description |
|---|---|---|
| Protocol | 17 | Deserialization, serialization round-trip, null-byte rejection, history |
| Preprocess | 29 | Chat templating (`build_messages`), sampling resolution and clamping, stop-sequence normalization, budget checks, defaults |
| Slot lifecycle + KV reuse | 52 | finish, cancel, streaming disconnect, UTF-8 assembly, stop-sequence matching (split fragments, withheld partial matches, multi-byte boundaries), sampler chain and seeding, LCP, prefix-aware slot routing, `begin_request` mirror, `finish_prefill` invariant, probe verdict gating, fallback |
| Client API | 15 | complete/complete_stream, async variants, error propagation, done-signal ordering, `Send + Sync` guard |
| Server | 15 | Socket binding and permissions, stale/live-socket handling, JSON round-trips, size cap, error responses |
| Config | 13 | Env parsing, defaults, `sane()` clamping, true-spelling flip-from-disabled |
| Integration | 15 | Real model (LFM2.5): simple completion, streaming, multi-turn, batching, 5-request overflow, probe verdict (degrade loudly), stop-sequence truncation (buffered + streamed), unseeded variation and seeded reproducibility |
| Lifecycle | 5 | Real model: two concurrent clients, simultaneous engine startup (backend-init race), prompt shutdown under load, explicit shutdown, async API inside a tokio runtime |
| KV reuse e2e | 3 | Real model (attention, `tests/reuse.rs`): identical resend, partial-prefix triple, reuse-engages probe verdict |
| Cancellation | 1 | Single-slot server: slot recycled after stream drop (bounded first-token wait) |

### Test architecture

- **Mock client**: A background thread processes `InferCmd`s and returns canned responses — tests run without a real model, via a test-only `Engine::from_sender`.
- **`oneshot_bridge`**: Converts `tokio::sync::oneshot` to `std::sync::mpsc` for use in sync test contexts.
- **Real-model tests**: Load actual GGUF files (from `models/`, fetched by `models/download.sh`) and verify token counts, temperature effects, system prompts, and multi-request batching. Serialized with `serial_test`; log-contract assertions via the capture layer in `tests/common/mod.rs`.

## Contributing

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(inference): reuse KV prefix across requests
fix(server): drain oversized bodies before replying
docs: document the async API
perf: avoid cloning the prompt into a slot
```

The changelog and GitHub release notes are generated from these, so the
subject line is what users read. `chore:`, `ci:` and `style:` are excluded
from the changelog; anything with a `!` suffix or a `BREAKING CHANGE:` footer
is flagged as breaking. Non-conventional commits are dropped entirely, so a
stray message never reaches a published changelog.

Before pushing:

```sh
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test --features native
```

## Releasing

Changelog and release notes come from [git-cliff](https://git-cliff.org)
(config in `cliff.toml`).

```sh
cargo install git-cliff          # once
```

1. Fold the new commits into the changelog and review the result:

   ```sh
   git cliff --unreleased --prepend CHANGELOG.md
   ```

   Run this once per release — `--prepend` adds a section each time it runs.
   The prose under `## [Unreleased]` is hand-written for the first release;
   rename that heading to the version instead of prepending for `0.1.0`.

2. Bump `version` in `Cargo.toml`, then `cargo update -p llamad --offline` so
   `Cargo.lock` follows.

3. Commit, tag, push:

   ```sh
   git commit -am "chore(release): v0.1.0"
   git tag -a v0.1.0 -m "v0.1.0"
   git push && git push --tags
   ```

   The tag triggers `.github/workflows/release.yml`, which verifies the tag
   matches `Cargo.toml`, runs the tests, and publishes a GitHub Release with
   generated notes.

4. Publish manually:

   ```sh
   cargo publish
   ```

   This step is deliberately not automated — a crates.io version can be
   yanked but never replaced.

Preview the notes for an unreleased set of commits at any time:

```sh
git cliff --unreleased --strip header
```

## License

MIT — see [LICENSE](LICENSE).
