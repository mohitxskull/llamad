//! Runtime configuration for the inference engine.
//!
//! All knobs are read from environment variables with sane defaults, so the
//! engine can be tuned per deployment without recompiling.

use std::env;

/// Smallest context the engine can start with.
///
/// The startup KV-rewind probe decodes two tokens, and `n_batch` is set to
/// `n_ctx`, so anything below this trips llama.cpp's
/// `GGML_ASSERT(n_tokens_all <= cparams.n_batch)` and aborts the process
/// rather than failing a request. Nothing useful fits in two tokens either,
/// but this is a crash floor, not a usefulness floor — a context too small for
/// a prompt still rejects that prompt cleanly at the budget check.
const MIN_N_CTX: u32 = 2;

/// Slots, context and thread configuration for a single inference loop.
///
/// The default is **one slot holding the whole context**. Concurrency is
/// opt-in because slots partition `n_ctx` statically: raising `n_slots` to 4
/// does not add capacity, it cuts every request's token budget to a quarter.
/// A caller that never issues concurrent requests would pay that cut for
/// nothing, so the cost is charged only to callers who ask for concurrency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceConfig {
    /// Number of concurrent sequences (slots).
    pub n_slots: usize,
    /// Total KV-context capacity in tokens, shared across all slots.
    pub n_ctx: u32,
    /// Threads used for single-token decode (generation).
    pub n_threads: usize,
    /// Threads used for batched/prompt processing (prefill).
    pub n_threads_batch: usize,
    /// Keep the KV cache of completed sequences and reuse the shared
    /// token prefix on the next request into the same slot.
    pub kv_cache: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            n_slots: 1,
            n_ctx: 2048,
            n_threads: 0,
            n_threads_batch: 0,
            kv_cache: true,
        }
        .with_default_threads()
    }
}

impl InferenceConfig {
    fn with_default_threads(mut self) -> Self {
        let physical = num_cpus::get_physical();
        let default = if physical > 0 { physical } else { 4 };
        if self.n_threads == 0 {
            self.n_threads = default;
        }
        if self.n_threads_batch == 0 {
            self.n_threads_batch = default;
        }
        self
    }

    /// Read configuration from environment variables, falling back to
    /// defaults. Recognized variables:
    ///
    /// - `LLAMAD_N_SLOTS` — concurrent sequences (default 1)
    /// - `LLAMAD_N_CTX` — total KV context in tokens (default 2048)
    /// - `LLAMAD_N_THREADS` — decode threads (default: physical cores)
    /// - `LLAMAD_N_THREADS_BATCH` — prefill threads (default: physical cores)
    ///
    /// Invalid values are ignored (a warning is logged) and the default kept.
    pub fn from_env() -> Self {
        let base = Self::default();
        Self {
            n_slots: read_env("LLAMAD_N_SLOTS", base.n_slots),
            n_ctx: read_env("LLAMAD_N_CTX", base.n_ctx),
            n_threads: read_env("LLAMAD_N_THREADS", base.n_threads),
            n_threads_batch: read_env("LLAMAD_N_THREADS_BATCH", base.n_threads_batch),
            kv_cache: read_env_bool("LLAMAD_KV_CACHE", base.kv_cache),
        }
        .sane()
    }

    /// Clamp values into usable ranges (never zero/overflowing; caps prevent
    /// truncation-induced div-by-zero and absurd allocations downstream).
    ///
    /// Applied to every config an engine starts from, including one a caller
    /// hand-built and passed to
    /// [`Engine::start_with_config`](crate::inference::Engine::start_with_config)
    /// — the struct's fields are public, so the values cannot be assumed sane.
    pub(crate) fn sane(self) -> Self {
        let n_slots = self.n_slots.clamp(1, 512);
        Self {
            n_slots,
            // Floored at `MIN_N_CTX` as well as at `n_slots`: `n_batch` is set
            // to `n_ctx`, and llama.cpp hard-asserts `n_tokens_all <= n_batch`
            // inside `llama_context::decode`. A smaller context would abort the
            // process on the startup KV-rewind probe, before any request ran.
            n_ctx: self.n_ctx.clamp((n_slots as u32).max(MIN_N_CTX), 1_048_576),
            n_threads: self.n_threads.clamp(1, 256),
            n_threads_batch: self.n_threads_batch.clamp(1, 256),
            kv_cache: self.kv_cache,
        }
    }

    /// Per-slot token budget: total context divided evenly across slots.
    ///
    /// This is a *static* partition — an idle slot's share is not lendable to
    /// a busy one. With the default single slot the budget is all of `n_ctx`;
    /// every additional slot divides it.
    pub fn per_slot_budget(&self) -> u32 {
        self.n_ctx / self.n_slots as u32
    }
}

fn read_env<T>(key: &str, default: T) -> T
where
    T: std::str::FromStr,
{
    match env::var(key) {
        Ok(raw) if !raw.trim().is_empty() => match raw.trim().parse::<T>() {
            Ok(v) => v,
            Err(_) => {
                tracing::warn!("invalid {key}={raw:?}, using default");
                default
            }
        },
        _ => default,
    }
}

fn read_env_bool(key: &str, default: bool) -> bool {
    match env::var(key) {
        Ok(raw) if raw.trim().is_empty() => default,
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            other => {
                tracing::warn!("invalid {key}={other:?}, using default");
                default
            }
        },
        _ => default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes tests that read/write `LLAMAD_*` env vars. The test harness
    /// runs tests in parallel threads by default (only `--test-threads=1`
    /// would serialize them), so without this lock two env tests can
    /// interleave and tear down each other's variables mid-assertion.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_defaults() {
        let cfg = InferenceConfig::default();
        // One slot by default: a single request gets the *whole* context, not
        // a fraction of it. A regression back to a multi-slot default would
        // silently cut every default caller's prompt budget.
        assert_eq!(cfg.n_slots, 1);
        assert_eq!(cfg.n_ctx, 2048);
        assert_eq!(cfg.n_threads, cfg.n_threads_batch);
        assert!(cfg.n_threads >= 1);
        assert_eq!(cfg.per_slot_budget(), 2048);
    }

    #[test]
    fn test_per_slot_budget_uneven_division_floors() {
        // 1000 / 3 must floor to 333 — a ceil regression would return 334
        // and let one slot exceed the shared context.
        let cfg = InferenceConfig {
            n_slots: 3,
            n_ctx: 1000,
            n_threads: 2,
            n_threads_batch: 2,
            kv_cache: true,
        };
        assert_eq!(cfg.per_slot_budget(), 333);
    }

    #[test]
    fn test_from_env_uses_defaults_when_unset() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // SAFETY: serialized by ENV_LOCK; no other test reads these vars.
        unsafe {
            std::env::remove_var("LLAMAD_N_SLOTS");
            std::env::remove_var("LLAMAD_N_CTX");
            std::env::remove_var("LLAMAD_N_THREADS");
            std::env::remove_var("LLAMAD_N_THREADS_BATCH");
        }
        let cfg = InferenceConfig::from_env();
        assert_eq!(cfg.n_slots, 1);
        assert_eq!(cfg.n_ctx, 2048);
        assert_eq!(cfg.per_slot_budget(), 2048);
    }

    #[test]
    fn test_from_env_reads_valid_values() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // SAFETY: serialized by ENV_LOCK; no other test reads these vars.
        unsafe {
            std::env::set_var("LLAMAD_N_SLOTS", "8");
            std::env::set_var("LLAMAD_N_CTX", "4096");
            std::env::set_var("LLAMAD_N_THREADS", "2");
            std::env::set_var("LLAMAD_N_THREADS_BATCH", "6");
        }
        let cfg = InferenceConfig::from_env();
        unsafe {
            std::env::remove_var("LLAMAD_N_SLOTS");
            std::env::remove_var("LLAMAD_N_CTX");
            std::env::remove_var("LLAMAD_N_THREADS");
            std::env::remove_var("LLAMAD_N_THREADS_BATCH");
        }
        assert_eq!(cfg.n_slots, 8);
        assert_eq!(cfg.n_ctx, 4096);
        assert_eq!(cfg.n_threads, 2);
        assert_eq!(cfg.n_threads_batch, 6);
        assert_eq!(cfg.per_slot_budget(), 512);
    }

    #[test]
    fn test_from_env_ignores_invalid_values_and_clamps() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        unsafe {
            std::env::set_var("LLAMAD_N_SLOTS", ""); // empty → default 1
            std::env::set_var("LLAMAD_N_CTX", "0"); // parses → sane() clamps up to n_slots
            std::env::set_var("LLAMAD_N_THREADS", "0"); // parses → sane() clamps to ≥ 1
            std::env::set_var("LLAMAD_N_THREADS_BATCH", "banana"); // unparsable → default
        }
        let cfg = InferenceConfig::from_env();
        unsafe {
            std::env::remove_var("LLAMAD_N_SLOTS");
            std::env::remove_var("LLAMAD_N_CTX");
            std::env::remove_var("LLAMAD_N_THREADS");
            std::env::remove_var("LLAMAD_N_THREADS_BATCH");
        }
        assert_eq!(cfg.n_slots, 1); // "" empty → default
        assert_eq!(cfg.n_ctx, MIN_N_CTX); // 0 clamps up to the crash floor
        assert_eq!(cfg.n_threads, 1); // 0 clamps to ≥ 1
        assert_eq!(
            cfg.n_threads_batch,
            InferenceConfig::default().n_threads_batch
        ); // "banana" unparsable → default
        assert_eq!(cfg.per_slot_budget(), 2); // 2 ctx / 1 slot
    }

    #[test]
    fn test_n_ctx_never_falls_below_the_batch_assert_floor() {
        // `n_batch` is set to `n_ctx`, and llama.cpp aborts the process if a
        // decode exceeds `n_batch`. The startup KV-rewind probe decodes two
        // tokens, so a context below `MIN_N_CTX` would kill the engine before
        // it served anything — a crash, not a rejected request.
        for n_ctx in [0, 1, 2] {
            let cfg = InferenceConfig {
                n_slots: 1,
                n_ctx,
                n_threads: 1,
                n_threads_batch: 1,
                kv_cache: true,
            }
            .sane();
            assert!(
                cfg.n_ctx >= MIN_N_CTX,
                "n_ctx {n_ctx} must clamp to at least {MIN_N_CTX}, got {}",
                cfg.n_ctx
            );
        }
    }

    #[test]
    fn test_many_slots_still_raise_n_ctx_above_the_floor() {
        // The floor is a maximum of the two constraints, not a replacement:
        // with more slots than MIN_N_CTX, n_slots is what binds.
        let cfg = InferenceConfig {
            n_slots: 64,
            n_ctx: 0,
            n_threads: 1,
            n_threads_batch: 1,
            kv_cache: true,
        }
        .sane();
        assert_eq!(cfg.n_ctx, 64);
        assert_eq!(cfg.per_slot_budget(), 1);
    }

    #[test]
    fn test_from_env_clamps_upper_bounds() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        unsafe {
            std::env::set_var("LLAMAD_N_SLOTS", "4294967296"); // 2^32 → truncates as u32 → clamps to 512
            std::env::set_var("LLAMAD_N_CTX", "4294967295"); // u32::MAX → clamps to 1_048_576
            std::env::set_var("LLAMAD_N_THREADS", "3000000000"); // would overflow i32 → clamps to 256
            std::env::set_var("LLAMAD_N_THREADS_BATCH", "3000000000");
        }
        let cfg = InferenceConfig::from_env();
        unsafe {
            std::env::remove_var("LLAMAD_N_SLOTS");
            std::env::remove_var("LLAMAD_N_CTX");
            std::env::remove_var("LLAMAD_N_THREADS");
            std::env::remove_var("LLAMAD_N_THREADS_BATCH");
        }
        assert_eq!(cfg.n_slots, 512);
        assert_eq!(cfg.n_ctx, 1_048_576); // still ≥ n_slots
        assert_eq!(cfg.n_threads, 256);
        assert_eq!(cfg.n_threads_batch, 256);
        assert_eq!(cfg.per_slot_budget(), 2048); // no div-by-zero
    }

    #[test]
    fn test_kv_cache_default_on() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        unsafe {
            std::env::remove_var("LLAMAD_KV_CACHE");
        }
        let cfg = InferenceConfig::from_env();
        unsafe {
            std::env::remove_var("LLAMAD_KV_CACHE");
        }
        assert!(cfg.kv_cache);
    }

    #[test]
    fn test_kv_cache_env_disables() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        unsafe {
            std::env::set_var("LLAMAD_KV_CACHE", "0");
        }
        let cfg = InferenceConfig::from_env();
        unsafe {
            std::env::remove_var("LLAMAD_KV_CACHE");
        }
        assert!(!cfg.kv_cache);
    }

    #[test]
    fn test_kv_cache_env_accepts_false_spellings() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        for v in ["false", "no", "off"] {
            unsafe {
                std::env::set_var("LLAMAD_KV_CACHE", v);
            }
            let cfg = InferenceConfig::from_env();
            unsafe {
                std::env::remove_var("LLAMAD_KV_CACHE");
            }
            assert!(!cfg.kv_cache, "{v} should disable");
        }
    }

    #[test]
    fn test_kv_cache_env_accepts_true_spellings() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        for v in ["1", "true", "yes", "on"] {
            unsafe {
                std::env::set_var("LLAMAD_KV_CACHE", v);
            }
            let cfg = InferenceConfig::from_env();
            unsafe {
                std::env::remove_var("LLAMAD_KV_CACHE");
            }
            assert!(cfg.kv_cache, "{v} should enable");
        }
    }

    #[test]
    fn test_kv_cache_env_true_spellings_flip_from_disabled() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // The default is also true, so the spellings loop alone cannot detect
        // a dropped true-arm (every value would fall through to the default
        // and still pass). Baseline on "0" first: each spelling must flip the
        // bool back to true, so a missing true-arm now fails the test.
        unsafe {
            std::env::set_var("LLAMAD_KV_CACHE", "0");
        }
        let cfg = InferenceConfig::from_env();
        unsafe {
            std::env::remove_var("LLAMAD_KV_CACHE");
        }
        assert!(!cfg.kv_cache, "baseline: 0 should disable");
        for v in ["1", "true", "yes", "on"] {
            unsafe {
                std::env::set_var("LLAMAD_KV_CACHE", v);
            }
            let cfg = InferenceConfig::from_env();
            unsafe {
                std::env::remove_var("LLAMAD_KV_CACHE");
            }
            assert!(
                cfg.kv_cache,
                "{v} should flip the disabled baseline to true"
            );
        }
    }

    #[test]
    fn test_kv_cache_env_empty_falls_back_to_default() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        unsafe {
            std::env::set_var("LLAMAD_KV_CACHE", "");
        }
        let cfg = InferenceConfig::from_env();
        unsafe {
            std::env::remove_var("LLAMAD_KV_CACHE");
        }
        assert!(cfg.kv_cache);
    }

    #[test]
    fn test_kv_cache_env_invalid_keeps_default() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        unsafe {
            std::env::set_var("LLAMAD_KV_CACHE", "banana");
        }
        let cfg = InferenceConfig::from_env();
        unsafe {
            std::env::remove_var("LLAMAD_KV_CACHE");
        }
        assert!(cfg.kv_cache);
    }
}
