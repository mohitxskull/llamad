# llamad

In-process GGUF model inference over channels — no HTTP, no sidecar.

```rust
use llamad::client::Client;

let client = Client::new("/path/to/model.gguf")?;
let text = client.complete_text("tell me a joke")?;
```

**llamad is a CPU-first inference library for embedding a local GGUF model
directly in a Rust program.** It gives you a finished decode loop — slotted
batching, streaming, drop-to-cancel, KV-prefix reuse, grammar-constrained
output, stop sequences, UTF-8 assembly across token boundaries — so you do not
write one against raw llama.cpp bindings.

## Where this fits

The Rust ecosystem makes you pick one of two bad options for local inference.
Bindings like [`llama-cpp-2`](https://crates.io/crates/llama-cpp-2) hand you
primitives and leave the decode loop to you — batch management, slot
scheduling, KV bookkeeping, cancellation, partial-UTF-8 reassembly. That loop
is small to sketch and unpleasant to get right. The alternative is a
server — [Ollama](https://github.com/ollama/ollama),
[llama.cpp's `llama-server`](https://github.com/ggml-org/llama.cpp) — which
means a second process to install, supervise and version, plus an HTTP hop for
inference that was always going to run on the same machine.

llamad is the middle: the scheduler and the loop, as a library call, in your
process.

| | llamad | `llama-cpp-2` / `-4` | Ollama / `llama-server` | [`mistral.rs`](https://github.com/EricLBuehler/mistral.rs) |
|---|---|---|---|---|
| Runs in your process | yes | yes | no — separate daemon | yes (heavier) |
| Decode loop written for you | yes | **no** — you write it | yes | yes |
| Continuous batching | yes | no | yes | yes |
| Transport | in-process channels; optional local IPC (Unix socket / named pipe) | none | HTTP | in-process / HTTP |
| Primary target | **CPU** | either | either, GPU-leaning | GPU-leaning |
| Platforms | Linux, macOS, Windows (all CI-tested) | all | all | all |
| Grammar-constrained output | yes | via raw sampler | yes | yes |
| GPU | untested pass-through | yes | yes | yes |
| Multi-model | one `Client` per model, sized independently | manual | registry, pull, hot-swap | multi-pipeline |
| Model source | local paths | local paths | registry + pull | HF Hub |
| Dependency weight | llama.cpp + tokio | llama.cpp | external binary | large |

**Use llamad when** the inference belongs inside your program: CLI tools,
desktop apps, editor and shell integrations, batch and offline pipelines,
privacy-sensitive local processing, and test suites that want a real model
without a GPU or a network.

**Use something else when** you want GPU offload, a model registry with
hot-swap, an OpenAI-compatible HTTP API, or multi-tenant serving. Ollama and
`llama-server` are better at all of those, and this is not trying to compete
with them.

**When you do need a process boundary, it costs nothing measurable.** The
daemon speaks JSON over a Unix socket rather than HTTP, and the round trip
disappears into the noise: across three runs of 60 single-token requests
against the 230M model, the median difference between the socket path and the
in-process path was 13 ms, 1.7 ms, and — on the third run — negative, with the
socket marginally ahead. That spread is measurement noise on a ~270 ms request,
not a transport cost. Against a real multi-second generation it is not worth
thinking about.

The reason to prefer local IPC here is not throughput, it is that there is no
network surface to get wrong: no port to allocate, no firewall rule, no
accidental bind to `0.0.0.0`. On Unix, access is filesystem permissions — the
socket is `0600` — instead of a network ACL. On Windows the daemon uses a named
pipe with `reject_remote_clients`, so it is unreachable over SMB; local access
falls back to the default pipe DACL (the creating user, `SYSTEM` and
`Administrators`), which is comparable to `0600` but not identical — an
administrator on the machine can open it.

**Not included, deliberately:** HTTP, an OpenAI-compatible endpoint, a model
registry with lazy loading or eviction (several models at once is just several
clients — see [Running more than one model](#running-more-than-one-model)), LoRA, speculative decoding, embeddings/rerank, vision, and paged KV.
See [Design tradeoffs](#design-tradeoffs) for the ones that are load-bearing
rather than merely absent.

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

### Running more than one model

A `Client` owns one model. To run several — the common shape is a small fast
model that routes or classifies, plus a larger one that does the actual
reasoning — construct one client per model and keep both alive:

```rust
use llamad::{client::Client, config::InferenceConfig};

// Small router: short context, one thread.
let router = Client::with_config("LFM2.5-230M-Q4_K_M.gguf", InferenceConfig {
    n_ctx: 1024,
    n_threads: 1,
    n_threads_batch: 1,
    ..Default::default()
})?;

// Large thinker: long context, the rest of the cores.
let thinker = Client::with_config("LFM2.5-1.2B-Thinking-Q4_K_M.gguf", InferenceConfig {
    n_ctx: 8192,
    n_threads: 3,
    n_threads_batch: 3,
    ..Default::default()
})?;

let route = router.complete_text("Classify: math, prose, or code? 2+2")?;
let answer = if route.contains("math") {
    thinker.complete_text("What is 2+2? Think step by step.")?
} else {
    router.complete_text("Reply briefly.")?
};
```

Each client is an independent engine: its own model, context, slots and pair
of threads. They do not share a KV cache and neither can disturb the other's
slots. Both are `Send + Sync`, so `Arc<Client>` works for concurrent callers.

Two things to get right:

- **Use `with_config`, not `new`.** The `LLAMAD_*` variables are process-global,
  so `Client::new` gives every model the same context and thread budget. A
  230M router does not need the 1.2B model's context, and paying for it wastes
  memory on every slot.

- **Decide thread counts from how often the models run at the same time.**
  Each engine defaults to `num_cpus::get_physical()` threads, so several
  engines each claim every core. That costs concurrency *scaling* but is not a
  collapse — see the measurements below. It is a decision made once per engine
  at construction, not per request.

There is no hard limit on how many models you load; three or four behave like
two. What actually bounds it is RAM and cores.

**Cost of keeping models resident.** Three models (135M + 230M + 1.2B, all
Q4_K_M) measured on the 4-core i5-1135G7:

| | measured |
|---|---|
| Resident memory | 1066 MB — roughly the sum of the GGUF files plus KV cache |
| OS threads | 2 per engine (preprocess + inference), 7 total including main |
| CPU while idle | **0%** over 5 s with all three loaded |

Idle engines are free: the inference thread blocks on its command channel when
no slot is active, so an unused model costs memory and nothing else. Keeping a
rarely-used model loaded is cheap.

**Cost of running two at once.** Two models generating 60 tokens each, three
runs, same box:

| Threads per engine | Each alone (sum) | Both concurrently | Speedup |
|---|---|---|---|
| 2 + 2 (cores split) | ~6.7 s | ~5.2 s | 1.27–1.28× |
| 4 + 4 (both claim all cores) | ~4.0 s | ~3.6–4.0 s | 1.00–1.12× |

Oversubscription is visible — it erodes the concurrency gain toward 1.0×, i.e.
the two models end up serializing on the cores — but it did **not** make things
slower than running them sequentially, and the higher per-model thread count
made every individual request faster. Splitting cores gives better parallel
scaling; not splitting gives better latency whenever only one model is active.

So: if your models mostly take turns (a router deciding, then a thinker
answering), give each all cores. If they genuinely generate simultaneously and
sustained, split the cores. Measure on your own hardware before assuming —
these numbers are one CPU, two small models, and do not extrapolate to a
16-core box.

A runnable version of all this is in
[`examples/multi_model.rs`](examples/multi_model.rs):

```
cargo run --release --example multi_model -- router.gguf thinker.gguf "your prompt"
```

#### Lazy loading, and why there is no registry

Models stay resident for as long as their client is alive. There is deliberately
no registry, no lazy loading and no eviction in the crate: dropping a client
unloads its model and joins its threads, and that is the whole lifecycle.

Eviction is the reason. Any registry worth the name eventually unloads
something, which means some later request stalls for however long it takes to
read a GGUF back off disk — at a moment neither you nor your user chose. For a
CLI tool or an editor integration that is a worse failure than running out of
memory, because it is silent and intermittent. That policy belongs to the
application, which knows which model must never be evicted.

Lazy loading without the eviction policy is about fifteen lines, and you keep
control of when the model goes away:

```rust
use std::sync::Mutex;

struct Lazy { path: String, config: InferenceConfig, slot: Mutex<Option<Client>> }

impl Lazy {
    fn with<R>(&self, f: impl FnOnce(&Client) -> R) -> Result<R, LlamaError> {
        let mut guard = self.slot.lock().unwrap();
        if guard.is_none() {
            *guard = Some(Client::with_config(&self.path, self.config.clone())?);
        }
        Ok(f(guard.as_ref().expect("just loaded")))
    }

    /// Release the memory. The next `with` reloads.
    fn unload(&self) { drop(self.slot.lock().unwrap().take()); }
}
```

Loading a small quantized GGUF takes on the order of a second, so reconstructing
on demand is viable when you genuinely cannot hold everything at once.

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

One request per connection: write a JSON object, end it with **either a
newline or a half-close**, then read the reply.

The transport is a Unix domain socket on Linux and macOS
(`/tmp/llamad.sock`) and a named pipe on Windows (`\\.\pipe\llamad`). The wire
format is identical on both.

The newline terminator exists because Windows named pipes have no half-close —
closing the handle closes both directions, so a request that ended only at EOF
would deadlock: the daemon would wait forever for bytes it already had.
`serde_json` escapes newlines inside strings and never emits a raw one, so a
newline is an unambiguous frame boundary. Clients that half-close still work
unchanged.

Replies are framed the same way: every line the daemon writes — the
non-streaming response, each streamed token, and every error — ends with a
newline. A client can therefore stop at the terminator instead of reading to
EOF, which on a named pipe means waiting for the daemon to drop the
connection. Trailing whitespace is insignificant to JSON, so a client that
does read to EOF is unaffected.

Fields: `prompt` (required), `system`, `max_tokens`, `temperature`, `top_k`, `top_p`, `repeat_penalty`, `repeat_last_n`, `seed`, `stop` (`[string]`), `stream` (bool), `history` (`[{role, content}]`). See [Generation parameters](#generation-parameters) for defaults and ranges. Unknown fields rejected. Request bodies are capped at 1 MiB (`server::MAX_REQUEST_BYTES`).

Every wire shape has a type in `protocol` — `Request`, `Response`,
`TokenChunk` (one streamed token), `StreamDone` (the streaming terminator) and
`ErrorResponse` — and all implement `Serialize` and `Deserialize`, so a Rust
consumer of the socket protocol can use the crate's own types rather than
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

| Variant | Meaning | Caller's move |
|---|---|---|
| `ModelLoad(String)` | Model failed to load, or the llama.cpp backend failed to initialize | Fix the path or the environment |
| `InferenceCrashed` | Inference thread died, or the engine was shut down mid-request | Rebuild the engine |
| `Busy` | Every slot was occupied when the request arrived | **Retry** — transient |
| `PromptTooLong { tokens, budget }` | The templated prompt alone exhausts the per-slot budget | Shorten the prompt, or raise `LLAMAD_N_CTX` / lower `LLAMAD_N_SLOTS`. Retrying is pointless |
| `Disconnected` | The streaming client hung up, cancelling generation | Expected when a `TokenStream` is dropped early — that *is* how cancellation is requested |
| `Inference(String)` | Runtime failure with no specific caller response: batch overflow, chat template application, detokenization | Treat as a bug or a resource limit |
| `Protocol(String)` | Message construction: null-byte / invalid role-or-content rejection (`NewLlamaChatMessageError`), or a grammar that does not compile | Fix the request |
| `Io(std::io::Error)` | Thread spawn or I/O failure | Depends on the `ErrorKind` |

`Busy`, `PromptTooLong` and `Disconnected` are split out of the `Inference`
catch-all precisely because they imply different responses — the first is worth
retrying, the second never is, and the third is usually self-inflicted and
benign. Matching on the message string is never necessary.

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

| Feature | Effect | Status |
|---|---|---|
| `native` | Builds llama.cpp with `-march=native` — enables AVX-512/AVX2/FMA. Fastest, but the binary may `SIGILL` on any other CPU model. | Tested; all benchmarks below use it |
| `openmp` | OpenMP threading inside ggml. | Builds; not benchmarked |
| `prebuilt` | Downloads a prebuilt generic x86-64 ggml instead of compiling llama.cpp. Much faster to build; no arch-specific SIMD. | Builds; not benchmarked |
| `cuda`, `metal`, `vulkan`, `hip`, `blas` | Forwarded verbatim to the corresponding `llama-cpp-4` feature. | **Untested — see below** |

**The GPU and BLAS features are unverified.** They are pass-through flags to
`llama-cpp-4`; nothing in this repo builds, runs, or benchmarks them, and CI
does not cover them. They are exposed because forwarding a feature costs
nothing and blocking it would help no one — not because llamad is known to work
on a GPU. If you enable one and it works, that is llama.cpp working; if it
breaks, the bug is as likely to be here as upstream. Reports welcome.

llamad is developed and tested as a **CPU inference library**. Everything below
— the throughput figures, the thread defaults, the slot budgets — was measured
and tuned on CPU.

## Supported models

Any GGUF that llama.cpp can load and that carries a chat template. The loader,
sampler and decode loop are model-agnostic; nothing in the crate is specific to
one family. Sampling **defaults** happen to be the Liquid AI LFM2.5 model-card
values, and every one of them is overridable per request (see
[Generation parameters](#generation-parameters)).

What is *tested* is narrower: the LFM2.5 family (hybrid/SSM) plus
SmolLM2-135M (pure attention, which is what makes the KV-reuse path reachable).
KV-prefix reuse is probe-gated per model and degrades to full prefill on
architectures that cannot partially rewind their KV cache — hybrid/SSM models
like LFM2.5 take that path.
The real-model test suite needs the GGUFs in `models/` (gitignored) — fetch
them with `models/download.sh`:

| Model | Params | Q4 size | Use case | Speed (i5-1135G7, 4 threads) |
|---|---|---|---|---|
| [LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M-GGUF) | 230M | 146 MB | Mechanical tasks: classification, routing, extraction, formatting | 37–70 tok/s |
| [LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF) | 1.2B | 731 MB | Reasoning: math, logic, multi-step problems | 15–16 tok/s |

KV prefix reuse is probe-gated per model: it engages on pure-attention GGUFs (SmolLM2 is the bundled one; any llama.cpp-supported attention arch works) and degrades to full-prefill-per-request on hybrid/SSM/recurrent models like the LFM2.5 family (see "KV-reuse path" under Tests).

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
| `grammar` | none | GBNF text; see [Grammar-constrained output](#grammar-constrained-output) |
| `grammar_root` | `"root"` | grammar start rule; ignored without `grammar` |
| `max_tokens` | 256 | further capped by the per-slot budget |

Sampler chain: grammar → penalties → top-k → top-p → temperature →
distribution. Grammar is first so every later stage picks only from the allowed
set. When
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

### Grammar-constrained output

A GBNF grammar masks disallowed tokens before every other sampling stage, so
output is guaranteed to parse. This is the reliable way to get structured
output from a small model, which will otherwise drift out of format:

```rust
let req = Request::new("Reply with a JSON object having key ok.")
    .with_grammar(r#"
root ::= "{" ws "\"ok\"" ws ":" ws bool ws "}"
bool ::= "true" | "false"
ws   ::= " "?
"#);
```

`grammar_root` selects the start rule and defaults to `"root"`. A grammar that
does not compile is rejected with `LlamaError::Protocol` — never silently
ignored, since unconstrained output from a request that asked to be constrained
is worse than an error. Grammars are validated on the preprocess thread, so a
malformed one fails that single request and leaves the engine serving.

Grammars are GBNF only; there is no JSON-Schema-to-GBNF converter in this crate
(`llama-cpp-4` does not expose one). Generate the GBNF with an external tool if
you are starting from a schema.

**Write grammars permissively at the start.** If a grammar cannot match
anything the model wants to emit — a common cause is forbidding the leading
space or newline most chat models produce — llama.cpp can be left with no legal
token. Allowing optional leading whitespace (`root ::= [ \n]* ...`) avoids the
usual case.

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

## Examples

| Example | What it shows |
|---|---|
| [`bench.rs`](examples/bench.rs) | TTFT and decode throughput for one model. `cargo run --release --example bench -- <model.gguf> <n_repeat> [system_prompt]` |
| [`multi_model.rs`](examples/multi_model.rs) | A small router plus a large thinker in one process: per-model config, grammar-constrained routing, and lazy loading. `cargo run --release --example multi_model -- <router.gguf> <thinker.gguf> [prompt]` |

## Tests

First fetch the model fixtures (re-runnable; skips already-downloaded files):

```sh
./models/download.sh                  # all three GGUFs (~950 MB)
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
cargo test --test daemon              # the llamad binary, end to end
cargo test --test fixtures            # fixture/download.sh drift (no model needed)
```

156 unit tests + 5 fixture-drift tests + 36 real-model tests (21 in `tests/integration.rs`, 7 in `tests/lifecycle.rs`, 4 in `tests/daemon.rs`, 3 in `tests/reuse.rs`, 1 in `tests/cancellation.rs`) + 6 doc tests.

Fixtures are named for the **capability** a test needs, not for a model's size: `HYBRID` (KV rewind unsupported — the degraded path), `HYBRID_LARGE`, and `ATTENTION` (rewind supported — reuse engages). Paths default under `CARGO_MANIFEST_DIR`; override with `LLAMAD_TEST_MODEL_HYBRID`, `LLAMAD_TEST_MODEL_HYBRID_LARGE`, or `LLAMAD_TEST_MODEL_ATTENTION`. Set `LLAMAD_REQUIRE_MODELS=1` — as CI does — to turn a missing fixture from a skip into a failure; without it a suite that tested nothing still reports as passing. `tests/fixtures.rs` checks the fixture list against `models/download.sh` in both directions, so a renamed quant and a download nothing uses are both caught. Every model-loading test is `#[serial]`-enforced (`serial_test`) within its binary, and cargo runs test binaries sequentially, so the heavy llama.cpp loads never contend — no `--test-threads` flags needed. Log-contract assertions (the "degrade loudly" warning, the reuse debug line) use a minimal capture `Layer` in `tests/common/mod.rs` (tracing-test was evaluated and rejected: its per-test subscriber filters to the test crate's targets, dropping library events).

### KV-reuse path (`LLAMAD_TEST_MODEL_ATTENTION` / bundled SmolLM2)

The KV-prefix-reuse machinery (anchor, retention, fill reconciliation) engages only when the startup probe finds a model that supports partial KV rewind — which the repo's LFM2.5 hybrids do not (they run the degraded full-prefill path). The reuse-path tests in `tests/reuse.rs` run against `LLAMAD_TEST_MODEL_ATTENTION` (any attention-only GGUF) when set, else the bundled SmolLM2-135M in `models/`. With a rewind-capable model the identical-resend and partial-prefix tests exercise the real reuse path, and a dedicated test asserts the probe verdict is `true` on it (no degradation warning; reuse debug lines present). Without one the binary skips — and skips report as passes, so CI sets `LLAMAD_REQUIRE_MODELS=1` to make that a failure instead. The degraded-path contract stays covered by `probe_verdict_degrades_loudly_on_lfm2` in `tests/integration.rs`.

### Test coverage

| Area | Tests | Description |
|---|---|---|
| Protocol | 20 | Deserialization, serialization round-trip, null-byte rejection, history, and the wire shape of `StreamDone` (flat, not nested) and `ErrorResponse` |
| Preprocess | 29 | Chat templating (`build_messages`), sampling resolution and clamping, grammar validation, stop-sequence normalization, budget checks, defaults |
| Slot lifecycle + KV reuse | 56 | finish, cancel, streaming disconnect, UTF-8 assembly, stop-sequence matching (split fragments, withheld partial matches, multi-byte boundaries), sampler chain and seeding, LCP, prefix-aware slot routing, `begin_request` mirror, `finish_prefill` invariant, probe verdict gating, fallback. Includes 5 **property** tests (proptest) over UTF-8 reassembly, stop-sequence hold-back and prefix matching — the areas where hand-picked examples only cover the cases someone thought of |
| Client API | 15 | complete/complete_stream, async variants, error propagation, done-signal ordering, `Send + Sync` guard |
| Server | 18 | Socket binding and permissions, stale/live-socket handling, JSON round-trips, newline-framed request with no half-close (the mechanism Windows needs), size cap, error responses. A `cfg(windows)` module mirrors the protocol assertions over a named pipe; those run on the `windows-latest` CI job, not locally. |
| Config | 18 | Env parsing, defaults, `sane()` clamping, true-spelling parsing (via `read_env_bool` directly — through `from_env` the property is untestable, since its default is already `true`), and the physical-core thread default |
| Integration | 21 | Real model (LFM2.5): simple completion, streaming, multi-turn, batching, 5-request overflow, probe verdict (degrade loudly), stop-sequence truncation (buffered + streamed), unseeded variation and seeded reproducibility, grammar constraint (sampled + greedy + JSON) and rejection of malformed/null-byte/unknown-root grammars without killing the engine |
| Lifecycle | 7 | Real model: two models side by side with independent configs, degenerate-config clamping, two concurrent clients, simultaneous engine startup (backend-init race), prompt shutdown under load, explicit shutdown, async API inside a tokio runtime |
| KV reuse e2e | 3 | Real model (attention, `tests/reuse.rs`): identical resend, partial-prefix triple, reuse-engages probe verdict |
| Cancellation | 1 | Single-slot server: slot recycled after stream drop (bounded first-token wait) |
| Daemon binary | 4 | The real `llamad` binary via `CARGO_BIN_EXE_llamad`: socket mode `0600`, newline-framed request round-trip, NDJSON stream terminated by a done line, SIGINT exits zero and unlinks the socket, usage/`--help` exit codes |
| Fixture drift | 5 | No model needed: the fixture list and `models/download.sh` must name the same files in *both* directions — a renamed quant and a download nothing uses are both failures |

### Test architecture

- **Mock client**: A background thread processes `InferCmd`s and returns canned responses — tests run without a real model, via a test-only `Engine::from_sender`.
- **`oneshot_bridge`**: Converts `tokio::sync::oneshot` to `std::sync::mpsc` for use in sync test contexts.
- **Real-model tests**: Load actual GGUF files (from `models/`, fetched by `models/download.sh`) and verify token counts, temperature effects, system prompts, and multi-request batching. Serialized with `serial_test`; log-contract assertions via the capture layer in `tests/common/mod.rs`.
- **Property tests**: `proptest` covers the two places where the failure mode is a panic on a character boundary or a silently dropped fragment. Failing seeds are committed under `tests/proptest-regressions/`, so a case found once re-runs for everyone.
- **Mutation testing** (evaluated 2026-08-06, not adopted): `cargo-mutants` answers what coverage cannot — whether a test would *fail* if the code were wrong. It earned its keep in one pass, finding four gaps that are now fixed: `complete_text_async` had no coverage at all; four `Request` builders were unasserted at unit level; a config test documented itself as catching a defect it structurally could not catch; and the thread-count default was untestable on any machine with four physical cores.

  It is not wired in, because the economics do not work for this crate. Mutation testing rebuilds and relinks once per mutant. That is sub-second for pure Rust, but every scenario here links llama.cpp's static library — 3-4s at best on a warm `target/`, ~570s on a cold one, times 273 mutants. Worse, one mutant (`TokenStream::next_token` returning `Some` unconditionally) turns every `while let Some(t) = next_token()` loop in the suite into an infinite allocator that exhausts a 16 GB machine in about twenty seconds, well inside the default timeout. Making that survivable needs a cgroup cap or a hand-tuned test timeout, and the whole apparatus then buys progressively less as coverage stays near 94%.

  Worth re-running by hand if a future change makes the test suite's *assertions* suspect. Not worth a scheduled job.
- **Coverage**: `cargo llvm-cov` over the whole suite, real-model tests included (without them the decode loop reads as uncovered). Reported, not gated: a percentage threshold mostly rewards tests that execute lines without asserting on them.

### Enforced by lints, not review

The library denies `unwrap_used`, `expect_used`, `panic` and
`undocumented_unsafe_blocks`, plus `unsafe_op_in_unsafe_fn` and
`unused_unsafe` at the manifest. Scoped to the library crate — the surface the
daemon exposes to socket input — so tests and the binary, where a panic is an
assertion or a startup abort, are unaffected. What lints cannot catch is a
panic inside a dependency: the real hazard is `LlamaSampler::grammar`, which
panics on malformed GBNF from the socket, handled explicitly with
`catch_unwind` in `build_grammar_sampler`.

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

## Development

Enable the pre-commit hook once per clone — `core.hooksPath` is local config,
so it does not come with the repository:

```sh
git config core.hooksPath .githooks
```

`.githooks/pre-commit` runs the same checks as CI's `check` job — rustfmt,
clippy at `-D warnings`, rustdoc at `-D warnings`, unit tests, doc tests, and a
compile of the real-model test binaries. About 12 seconds on a warm `target/`,
and no model files needed.

`cargo doc` is in there for a specific reason: an intra-doc link to a
`cfg`-gated item resolves fine on the platform that defines it and fails
everywhere else, and fmt, clippy and the tests all pass straight through that.
It is the one check that has caught a problem nothing else did.

What the hook deliberately does *not* run: the real-model suite (needs ~950 MB
of GGUFs) and the macOS/Windows jobs. Those belong in CI. A hook slow enough to
be irritating is a hook people bypass.

Bypass with `git commit --no-verify` for work-in-progress commits on a branch —
not for anything about to reach `main`.

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

3. Run the **full** suite locally, including the real-model tests. CI cannot:
   the GGUFs are ~950 MB, so `.github/workflows/ci.yml` compiles those test
   binaries but does not run them.

   ```sh
   cargo test
   ```

4. Commit, tag, push:

   ```sh
   git commit -am "chore(release): v0.1.0"
   git tag -a v0.1.0 -m "v0.1.0"
   git push && git push --tags
   ```

The tag is the only trigger. Pushing it runs
`.github/workflows/release.yml`, which:

1. **verifies** — tag matches `Cargo.toml`, `CHANGELOG.md` has a section for
   the version, unit and doc tests pass, clippy is clean, and the packaged
   tarball builds;
2. **publishes to crates.io** — using Trusted Publishing, so no registry token
   is stored in this repository;
3. **creates the GitHub Release** with notes generated by git-cliff.

The order matters: publishing is irreversible, so it runs after every cheap
check and before the release announcement. A failed publish leaves no GitHub
Release claiming a version that does not exist.

Authentication is [Trusted
Publishing](https://crates.io/docs/trusted-publishing): the publish job mints a
short-lived token over OIDC, so there is no registry credential stored in this
repository. Nothing to rotate, and nothing to leak.

The workflow also accepts a `CARGO_REGISTRY_TOKEN` secret and prefers it when
present. That path exists because a trusted publisher is configured per crate
at `https://crates.io/crates/<name>/settings`, a page that does not exist until
the crate has been published once — so a brand-new crate cannot use OIDC for
its first release. It is unused here and should stay that way.

If authentication fails the crate is simply not published; delete the tag, fix
the setup, and re-tag.

A version can be yanked but never replaced or reused, which is why the tag —
not a branch push or a commit subject — is what triggers this.

Preview the notes for an unreleased set of commits at any time:

```sh
git cliff --unreleased --strip header
```

## License

MIT — see [LICENSE](LICENSE).
