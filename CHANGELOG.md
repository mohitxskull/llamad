# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries below the first release heading are generated from
[Conventional Commits](https://www.conventionalcommits.org/) by
[git-cliff](https://git-cliff.org) — see "Releasing" in the README.

## [0.1.0] - 2026-08-05

First public release. Everything below is new.

### Added

**Library API**

- `Client` — load a GGUF model and run completions, streaming or not.
  `complete`, `complete_text`, `complete_stream`, and a `TokenStream` that
  yields tokens as they are generated.
- Async variants for every blocking call: `complete_async`,
  `complete_text_async`, `TokenStream::next_token_async` and
  `into_result_async`. The blocking methods panic inside a tokio runtime by
  tokio's design, so async callers have a first-class path.
- `Engine` — the loaded model and its two threads. Owns them and joins on
  drop, so releasing an engine cancels in-flight requests and returns only
  once the model is actually freed. Engines are independent: several can be
  alive at once, each with its own model.
- `Request` builder with system prompt, history, token cap, stop sequences and
  the full sampling set (temperature, top-k, top-p, repetition penalty and its
  lookback, seed).
  `Request`, `Response` and `TokenChunk` all round-trip through serde, so a
  Rust consumer of the socket protocol uses these types rather than
  hand-rolled JSON.
- `Client` is `Send + Sync`, so `Arc<Client>` serves concurrent requests from
  one model.
- `Client::with_config` / `Engine::start_with_config` take an
  `InferenceConfig` directly, bypassing the environment. Needed by any process
  running more than one model: `LLAMAD_*` is process-global, so the env-based
  constructors give a small routing model and a large reasoning model the same
  context and thread budget. Hand-built configs are clamped, not trusted — the
  struct's fields are public and `n_slots: 0` would otherwise divide by zero.

**Inference**

- Slotted continuous batching: up to `LLAMAD_N_SLOTS` concurrent sequences
  share a single `LlamaContext`, decoded in one batch per step and sampled
  per slot. Measured ~1.6–1.9x aggregate throughput over single-request
  decode with `LLAMAD_N_SLOTS=4`.
- Concurrency is opt-in: `LLAMAD_N_SLOTS` defaults to **1**, so a default
  engine gives each request the whole `LLAMAD_N_CTX`. Slots partition context
  statically (`N_CTX / N_SLOTS`) and an idle slot's share is not lendable, so
  a multi-slot default would silently cut every single-request caller's prompt
  budget to a fraction of the context for concurrency it never uses.
- KV-prefix reuse: a completed sequence's KV prefix is retained and reused on
  the next request into the same slot. Gated on a startup probe for partial
  KV rewind support, degrading loudly to full-prefill-per-request on
  hybrid/SSM models that cannot rewind.
- Prefix-aware slot routing: a request is placed in the free slot whose
  retained KV prefix shares the most tokens with it, rather than the lowest
  free index. First-free placement would strand the reuse cache whenever a
  lower-numbered slot was busy — the common case for a repeated system prompt
  under concurrency. Ties resolve to the lowest free index.
- Preprocess offload: chat templating and tokenization run on a dedicated
  thread, so a long prompt never stalls in-flight generations.
- Per-request sampling: `temperature`, `top_k`, `top_p`, `repeat_penalty`,
  `repeat_last_n` and `seed` are all set per request. The defaults are the
  Liquid AI model-card values, but they are defaults only — a crate that loads
  arbitrary GGUFs must not impose one vendor's recommendation. Values are
  clamped into range rather than rejected, and `NaN` falls back to the default.
- `seed` defaults to a fresh random seed per request, so two identical requests
  at a non-zero temperature return different text. Pin it for reproducibility.
- Grammar-constrained generation: `grammar` (GBNF) masks disallowed tokens
  ahead of every other sampling stage, so output is guaranteed to parse —
  the reliable way to get structured output from a small model.
  `grammar_root` selects the start rule, defaulting to `"root"`. Grammars are
  compiled and validated on the preprocess thread, so a malformed one fails
  that single request with `LlamaError::Protocol` instead of reaching the
  inference thread: `LlamaSampler::grammar` panics on unparseable GBNF and on
  null bytes, and grammars arrive over the socket, so an unguarded panic would
  have been a remote kill of the whole engine. GBNF only — no JSON-Schema
  converter, which `llama-cpp-4` does not provide.
- Stop sequences: `stop` ends generation as soon as any entry appears, and the
  matched text is excluded from the result. A stop sequence may span token
  boundaries — output that could still grow into one is withheld from the
  stream until the match completes or is ruled out, so a partial match never
  leaks to a streaming client. Text unlike any stop sequence streams with no
  added latency.
- Per-slot token budgets, incremental UTF-8 assembly across token boundaries,
  and cancellation that frees a slot as soon as a streaming client goes away.
- Sampler chain order: grammar → penalties → top-k → top-p → temperature →
  distribution. Greedy decoding (temperature ≤ 0) replaces the sampling tail
  but still runs behind the grammar mask.
- Tokens are accepted into the sampler exactly once. `llama_sampler_sample`
  already accepts internally, so the additional explicit `accept` advanced
  every stateful sampler twice per token — double-counting each token in the
  repetition penalty's history, and (once grammars existed) stepping the
  grammar twice per token until its stack emptied and llama.cpp threw a C++
  exception that aborted the process from across the FFI boundary.

**Daemon**

- Unix-socket server speaking JSON, and NDJSON when streaming.
- Socket is created with mode `0600`; the default path is in world-writable
  `/tmp`, where a permissive mode would let any local user drive the model.
- Reclaims a socket left by a crashed daemon, but refuses to unlink a path
  that is not a socket and refuses to rebind a path a live daemon is using.
- Request bodies capped at 1 MiB so a local client cannot exhaust memory.
- Graceful shutdown on SIGINT that joins the inference threads.

**Configuration**

- `LLAMAD_N_SLOTS`, `LLAMAD_N_CTX`, `LLAMAD_N_THREADS`,
  `LLAMAD_N_THREADS_BATCH`, `LLAMAD_KV_CACHE`, all range-clamped so a bad
  value degrades instead of panicking.
- Thread counts default to physical cores rather than logical, avoiding
  hyperthread collapse. Several engines in one process each claim every core
  by default; measurements for splitting versus not are in the README, along
  with the memory and idle-CPU cost of keeping several models resident.
- `n_ctx` is floored at 2 as well as at `n_slots`. `n_batch` is set to `n_ctx`
  and llama.cpp hard-asserts `n_tokens_all <= n_batch`, so a smaller context
  aborted the process on the startup KV-rewind probe — reachable from
  `LLAMAD_N_CTX=1` — instead of failing a request.

**Build**

- Acceleration backends are opt-in cargo features — `native`, `openmp`,
  `prebuilt`, `cuda`, `metal`, `vulkan`, `hip`, `blas`. None are on by
  default, so a stock build is portable to any machine with the same target
  triple.
- The daemon binary and its argument/logging dependencies sit behind the
  `bin` feature (on by default); library consumers can drop them with
  `default-features = false`.
- MSRV 1.88, enforced in CI.

### Notes

- Tested against the Liquid AI LFM2.5 family; KV-prefix reuse additionally
  exercised against pure-attention models (SmolLM2, Qwen2.5).
- 142 unit tests, 32 real-model tests, 6 doc tests. CI runs clippy and
  rustdoc at `-D warnings`.
