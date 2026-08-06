//! In-process GGUF model inference over channels — no HTTP, no sidecar.
//!
//! ```rust,no_run
//! use llamad::client::Client;
//!
//! # fn main() -> Result<(), llamad::protocol::LlamaError> {
//! let client = Client::new("model.gguf")?;
//! let text = client.complete_text("tell me a joke")?;
//! # Ok(())
//! # }
//! ```
//!
//! # Architecture
//!
//! A [`Client`](client::Client) owns an [`Engine`](inference::Engine): a
//! loaded model plus two threads. Requests flow
//! `caller → llamad-preprocess → llamad-inference`, where the preprocess
//! thread does chat templating and tokenization so a long prompt never stalls
//! in-flight generations, and the inference thread runs a slotted continuous
//! batching loop over a single `LlamaContext`.
//!
//! Engines are independent — several clients with different models can be
//! alive at once — and each joins its threads on drop.
//!
//! # Running several models
//!
//! One [`Client`](client::Client) owns one model. For a small routing model
//! alongside a large reasoning one, construct a client per model with
//! [`Client::with_config`](client::Client::with_config) and give each its own
//! context and thread budget — the `LLAMAD_*` variables are process-global, so
//! [`Client::new`](client::Client::new) would give both the same settings.
//! There is no limit on how many engines are alive; RAM and cores bound it.
//! An idle engine costs memory but no CPU — its inference thread blocks on the
//! command channel whenever no slot is active. Thread counts are worth
//! choosing deliberately: every engine otherwise claims all physical cores,
//! which is fine when the models take turns and costs parallel scaling when
//! they generate simultaneously. See the README for measurements.
//!
//! # Blocking and async
//!
//! [`Client`](client::Client)'s default methods block the calling thread and
//! **panic inside a tokio runtime**. Async callers should use the `_async`
//! variants, e.g. [`Client::complete_async`](client::Client::complete_async).
//!
//! # Acceleration
//!
//! No backend is enabled by default, so a stock build is portable. Enable one
//! with a cargo feature: `native` (host-specific SIMD, fastest but pins the
//! binary to the build machine), `openmp`, `prebuilt`, `cuda`, `metal`,
//! `vulkan`, `hip`, or `blas`.
//!
//! # Configuration
//!
//! Runtime knobs are read from `LLAMAD_*` environment variables — see
//! [`InferenceConfig`](config::InferenceConfig). Pass a config explicitly with
//! [`Client::with_config`](client::Client::with_config) to bypass the
//! environment, which is what a process running more than one model wants.

#![warn(missing_docs)]
#![warn(clippy::doc_markdown)]
// ── Panic discipline ─────────────────────────────────────────────────────────
//
// This library is reachable from untrusted input: the daemon hands socket
// bytes to the same code paths a caller reaches in process. A panic on the
// inference thread takes the engine down for *every* client, so "does not
// panic" is a property worth having the compiler enforce rather than review
// for.
//
// The crate already satisfies this — the only two `expect`s outside test code
// are infallible and carry an `#[allow]` with the reason at the call site.
// These lints exist to keep it that way, not to schedule work.
//
// Scoped to the library crate rather than `[lints.clippy]` in Cargo.toml so
// they do not apply to `tests/`, `examples/` or the daemon binary, where a
// panic is an assertion or a startup abort rather than a fault. The in-crate
// `#[cfg(test)] mod tests` blocks *are* part of this crate, so each carries
// its own `allow`.
//
// Note what this cannot catch: a panic raised inside a dependency. The real
// untrusted-input hazard here is `LlamaSampler::grammar`, which panics on a
// malformed GBNF arriving over the socket — handled explicitly with
// `catch_unwind` in `inference::build_grammar_sampler`, not by these lints.
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
// Every `unsafe` block states why it is sound. The crate's only production
// unsafe is the llama.cpp log callback; the rest is `env::set_var` in tests,
// which edition 2024 makes unsafe.
#![deny(clippy::undocumented_unsafe_blocks)]

pub mod client;
pub mod config;
pub mod inference;
mod preprocess;
pub mod protocol;
pub mod server;
