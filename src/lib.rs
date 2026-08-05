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

pub mod client;
pub mod config;
pub mod inference;
mod preprocess;
pub mod protocol;
/// The Unix-socket daemon protocol.
///
/// **Unix only.** This module is compiled on Linux and macOS and absent on
/// Windows, which has no `tokio::net::UnixListener`. The rest of the crate —
/// [`Client`](client::Client), [`Engine`](inference::Engine) and everything
/// they need — is portable; only the socket server and the `llamad` binary
/// depend on the platform.
#[cfg(unix)]
pub mod server;
