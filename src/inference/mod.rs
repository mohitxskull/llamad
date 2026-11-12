//! The inference thread: model and context ownership, the slotted continuous
//! batching loop, and KV-prefix reuse.
//!
//! [`Engine`] is the entry point — it owns the `llamad-inference` and
//! `llamad-preprocess` threads for one loaded model and joins them on drop.

mod cache;
mod engine;
pub mod metrics;
mod sampler;
mod scheduler;
mod slot;

#[cfg(test)]
mod tests;

pub use engine::{Engine, EngineStatus, start_inference};
pub use metrics::EngineMetrics;
pub(crate) use metrics::MetricsTracker;
pub(crate) use sampler::build_grammar_sampler;
