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
//! [`InferenceConfig`](config::InferenceConfig).

#![warn(missing_docs)]
#![warn(clippy::doc_markdown)]

pub mod client;
pub mod config;
pub mod inference;
mod preprocess;
pub mod protocol;
pub mod server;
