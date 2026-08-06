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

// ── Model fixtures ──────────────────────────────────────────────────────────

/// One GGUF fixture, named for the **capability** a test needs from it rather
/// than for its size or vendor.
///
/// The distinction is load-bearing, not cosmetic. `probe_kv_rewind` decides at
/// startup whether a model supports partial KV rewind, and that verdict is the
/// thing several tests assert on: the reuse tests need a model where the probe
/// says *yes*, the degrade tests need one where it says *no*. Naming these
/// `model_230m` / `model_1_2b` invited pointing an override at a model with
/// the opposite property, which made the test fail for a reason its name did
/// not explain.
pub struct Fixture {
    /// Filename under `models/`. Must match a key in `models/download.sh` —
    /// `tests/fixtures.rs` fails if these two drift apart.
    pub file: &'static str,
    /// Environment variable that overrides the path to this fixture.
    pub env: &'static str,
    /// What a test gets from this model, for the message printed when it is
    /// missing.
    pub capability: &'static str,
}

/// Hybrid/SSM architecture: the startup probe refuses partial KV rewind, so
/// prefix reuse degrades to a full prefill. The default workhorse — small and
/// fast, and the model the degrade-path contracts are written against.
pub const HYBRID: Fixture = Fixture {
    file: "LFM2.5-230M-Q4_K_M.gguf",
    env: "LLAMAD_TEST_MODEL_HYBRID",
    capability: "hybrid/SSM arch (KV rewind unsupported — degraded path)",
};

/// A second, larger hybrid model. Only used to prove a bigger model loads and
/// generates; nothing depends on its size beyond that.
pub const HYBRID_LARGE: Fixture = Fixture {
    file: "LFM2.5-1.2B-Thinking-Q4_K_M.gguf",
    env: "LLAMAD_TEST_MODEL_HYBRID_LARGE",
    capability: "a larger hybrid model",
};

/// Pure-attention architecture: the probe accepts partial KV rewind, so the
/// prefix-reuse path actually engages. Without this the reuse assertions in
/// `tests/reuse.rs` are vacuous.
pub const ATTENTION: Fixture = Fixture {
    file: "SmolLM2-135M-Instruct-Q4_K_M.gguf",
    env: "LLAMAD_TEST_MODEL_ATTENTION",
    capability: "pure-attention arch (KV rewind supported — reuse engages)",
};

/// Every fixture the suite knows about. `tests/fixtures.rs` checks this list
/// against `models/download.sh` in both directions, so a renamed quant and an
/// orphaned download are both caught.
pub const ALL: &[&Fixture] = &[&HYBRID, &HYBRID_LARGE, &ATTENTION];

/// Set by CI. When present, a missing fixture is a test failure rather than a
/// skip — otherwise a suite that silently tested nothing reports as green.
pub const REQUIRE_ENV: &str = "LLAMAD_REQUIRE_MODELS";

fn require_models() -> bool {
    std::env::var(REQUIRE_ENV)
        .ok()
        .is_some_and(|v| !v.trim().is_empty() && v.trim() != "0")
}

impl Fixture {
    /// Path to this fixture: the override if set, else `models/<file>`.
    /// Does not check that anything exists there.
    pub fn path(&self) -> String {
        std::env::var(self.env)
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| format!("{}/models/{}", env!("CARGO_MANIFEST_DIR"), self.file))
    }

    /// Whether the fixture is actually on disk.
    pub fn available(&self) -> bool {
        std::fs::metadata(self.path()).is_ok()
    }

    /// Path to a fixture the test cannot run without.
    ///
    /// Panics with an actionable message rather than letting the model load
    /// fail later as a generic `InferenceCrashed`, which says nothing about
    /// what is missing or how to get it.
    pub fn require(&self) -> String {
        let path = self.path();
        assert!(
            self.available(),
            "missing fixture {} ({})\n  expected at: {path}\n  fix: run ./models/download.sh, or set {}=/path/to/model.gguf",
            self.file,
            self.capability,
            self.env
        );
        path
    }

    /// Path to a fixture the test can be skipped without — unless
    /// [`REQUIRE_ENV`] is set, where a missing fixture fails instead.
    ///
    /// Returns `None` to mean "skip". The caller must still return early; a
    /// skipped test reports as a pass, which is exactly why CI sets
    /// [`REQUIRE_ENV`].
    pub fn optional(&self) -> Option<String> {
        if self.available() {
            return Some(self.path());
        }
        assert!(
            !require_models(),
            "{REQUIRE_ENV} is set, so fixture {} ({}) must be present.\n  expected at: {}\n  fix: run ./models/download.sh",
            self.file,
            self.capability,
            self.path()
        );
        eprintln!(
            "SKIPPED: no {} available ({}). Run ./models/download.sh or set {}.",
            self.file, self.capability, self.env
        );
        None
    }
}

/// Hybrid model, required.
pub fn model_hybrid() -> String {
    HYBRID.require()
}

/// Larger hybrid model, required.
pub fn model_hybrid_large() -> String {
    HYBRID_LARGE.require()
}

/// Pure-attention model, or `None` to skip.
pub fn model_attention() -> Option<String> {
    ATTENTION.optional()
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
