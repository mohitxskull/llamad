//! Shared helpers for the real-model integration test binaries
//! (`integration`, `reuse`, `cancellation`) and the global tracing capture
//! layer used to assert the "degrade loudly" / reuse-engages contracts.
//!
//! Each `tests/*.rs` file compiles as its own binary (cargo runs them
//! sequentially), isolating env vars and log capture across binaries. Within a
//! binary, every model-loading test is `#[serial]`-enforced (see
//! `serial_test`), so heavy llama.cpp contexts never contend for CPU in
//! parallel.
//!
//! Why a custom capture layer instead of `tracing-test` (evaluated and
//! rejected 2026-08-05): `#[traced_test]` installs a subscriber filtered to
//! the *test crate's* targets (`"<test-crate>=trace"`), so events emitted by
//! the library under test (`llamad::inference`) are dropped before capture;
//! and its `logs_contain` scope-matching only matches lines whose formatted
//! target is under the test module. The assertions below need exactly the
//! opposite — library-target events at DEBUG+ with raw message matching — so
//! a minimal `Layer` recording the `message` field into a per-binary sink is
//! the fit. (tracing-subscriber's own tests use the same pattern.)
#![allow(dead_code)] // shared across binaries; each binary uses a subset

use std::sync::Mutex;
use std::sync::OnceLock;

/// Model paths default to the crate's `models/` dir; every constant can be
/// overridden via env (`LLAMAD_TEST_MODEL_230M`, `LLAMAD_TEST_MODEL_1_2B`,
/// `LLAMAD_TEST_MODEL`) so the suite runs on any machine without editing code.
pub fn env_override(key: &str, default: &'static str) -> String {
    std::env::var(key)
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

pub const MODEL_230M: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/LFM2.5-230M-Q4_K_M.gguf"
);
pub const MODEL_1_2B: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf"
);

/// LFM2.5-230M path (hybrid arch — KV reuse degrades by probe verdict),
/// overridable via `LLAMAD_TEST_MODEL_230M`.
pub fn model_230m() -> String {
    env_override("LLAMAD_TEST_MODEL_230M", MODEL_230M)
}

/// LFM2.5-1.2B-Thinking path, overridable via `LLAMAD_TEST_MODEL_1_2B`.
pub fn model_1_2b() -> String {
    env_override("LLAMAD_TEST_MODEL_1_2B", MODEL_1_2B)
}

/// Optional rewind-capable (pure-attention) GGUF: `LLAMAD_TEST_MODEL` when
/// set, else the bundled SmolLM2-135M in `models/` when present. With a
/// rewind-capable model the probe returns true and the KV-reuse path actually
/// engages; without one, the reuse tests in `tests/reuse.rs` skip with a
/// message. All four models are fetched by `models/download.sh`.
pub fn reuse_model_path() -> Option<String> {
    if let Some(p) = std::env::var("LLAMAD_TEST_MODEL")
        .ok()
        .filter(|s| !s.trim().is_empty())
    {
        return Some(p);
    }
    let bundled = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/models/SmolLM2-135M-Instruct-Q4_K_M.gguf"
    );
    std::fs::metadata(bundled).ok().map(|_| bundled.to_string())
}

// ── Global tracing capture layer ────────────────────────────────────────────

pub use tracing_subscriber::layer::SubscriberExt;
pub use tracing_subscriber::util::SubscriberInitExt;

/// Shared capture sink for the global tracing layer installed by
/// [`install_capture_layer`]. The library never installs a subscriber (only
/// `src/main.rs` does), so the test binary owns the global default.
pub static LOG_SINK: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
static SUBSCRIBER_INSTALLED: OnceLock<()> = OnceLock::new();

/// Minimal `Layer` that records the `message` field of every event at DEBUG
/// and above into [`LOG_SINK`]. No formatting, no timestamps — the captured
/// lines are the raw message strings. DEBUG is the floor so the reuse-path
/// debug line ("slot N reused M cached tokens") is captured too; WARN
/// assertions (the "degrade loudly" contract) still work since WARN ≥ DEBUG.
struct CaptureLayer;

impl<S> tracing_subscriber::Layer<S> for CaptureLayer
where
    S: tracing::Subscriber,
{
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        // CAUTION: tracing-core's `Level` ordering is inverted from severity —
        // `PartialOrd` is defined as `other >= self` (metadata.rs: ge/le), so
        // "DEBUG and above" is `level() <= &Level::DEBUG`, NOT `>=` (which
        // would admit only DEBUG/TRACE and drop INFO/WARN/ERROR). Verified
        // against vendored tracing-core 0.1.36.
        if event.metadata().level() <= &tracing::Level::DEBUG {
            let mut recorder = SinkRecorder::default();
            event.record(&mut recorder);
            if let Some(msg) = recorder.message {
                LOG_SINK
                    .get()
                    .expect("sink initialized before subscriber install")
                    .lock()
                    .expect("sink lock")
                    .push(msg);
            }
        }
    }
}

/// Extracts the `message` field of an event. All macro format-string messages
/// (`event!` wraps them in `fmt::Arguments`) and `field::display`/`field::debug`
/// wrappers arrive via `record_debug`, which is what the assertion code reads.
/// Plain `&str` fields would take `record_str` instead — implemented below as
/// belt-and-braces so an explicitly string-typed `message` field is never
/// silently missed.
#[derive(Default)]
struct SinkRecorder {
    message: Option<String>,
}

impl tracing::field::Visit for SinkRecorder {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = Some(format!("{value:?}"));
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.message = Some(value.to_string());
        }
    }

    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        if field.name() == "message" {
            self.message = Some(value.to_string());
        }
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        if field.name() == "message" {
            self.message = Some(value.to_string());
        }
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        if field.name() == "message" {
            self.message = Some(value.to_string());
        }
    }
}

/// Drop all captured lines so each test's poll asserts only its own traffic
/// (llama.cpp log_callback WARN lines accumulate across tests otherwise).
pub fn clear_sink() {
    LOG_SINK
        .get()
        .expect("sink initialized before subscriber install")
        .lock()
        .expect("sink lock")
        .clear();
}

/// Install the capture layer as the process-global subscriber exactly once
/// (idempotent; safe under `--test-threads=1` and parallel runs alike).
pub fn install_capture_layer() {
    LOG_SINK.get_or_init(|| Mutex::new(Vec::new()));
    SUBSCRIBER_INSTALLED.get_or_init(|| {
        // `Err` means a subscriber already exists — nothing else in this
        // process installs one, but ignore it regardless.
        let _ = tracing_subscriber::registry().with(CaptureLayer).try_init();
    });
}

/// Snapshot of the captured lines for a poll loop.
pub fn captured_lines() -> Vec<String> {
    LOG_SINK
        .get()
        .expect("sink initialized")
        .lock()
        .expect("sink lock")
        .clone()
}
