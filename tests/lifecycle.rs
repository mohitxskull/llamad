//! Engine/Client lifecycle contracts, against a real model.
//!
//! Its own test binary so the multi-engine tests here cannot be perturbed by
//! the `#[serial]` single-engine assumptions of `integration.rs`.
//!
//! These cover the defects found in the pre-release review:
//!
//! 1. `Client::new` used to join a process-global "previous" thread pair, so
//!    constructing a second client while the first was alive blocked forever.
//! 2. The blocking API panics inside a tokio runtime; the `_async` variants
//!    are the supported path there and must actually work.
//! 3. Backend initialization raced when two engines started simultaneously.

use std::sync::mpsc;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use llamad::client::Client;
use llamad::config::InferenceConfig;
use llamad::inference::Engine;
use llamad::protocol::{InferCmd, LlamaError, Request};
use serial_test::serial;

mod common;
use common::{model_1_2b, model_230m};

fn short(text: &str) -> Request {
    Request::new(text).with_max_tokens(8)
}

/// Run `f` on a worker thread and fail if it has not finished within `budget`.
///
/// A hang here is the failure mode under test, so the assertion cannot be
/// `join()` — that would hang the suite instead of reporting.
fn within<F: FnOnce() + Send + 'static>(budget: Duration, what: &str, f: F) {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        f();
        let _ = tx.send(());
    });
    assert!(
        rx.recv_timeout(budget).is_ok(),
        "{what} did not finish within {budget:?}"
    );
}

#[serial]
#[test]
fn two_models_run_side_by_side_with_independent_configs() {
    // The small-router + large-thinker pattern. Environment config is
    // process-global, so `Client::new` would hand both models the same context
    // and thread budget; `with_config` is what makes them independently
    // sizable. Thread counts are deliberately under physical cores here —
    // two engines each defaulting to all cores oversubscribe ggml's workers.
    let router = Client::with_config(
        model_230m(),
        InferenceConfig {
            n_ctx: 512,
            n_threads: 1,
            n_threads_batch: 1,
            ..Default::default()
        },
    )
    .expect("router client");
    let thinker = Client::with_config(
        model_1_2b(),
        InferenceConfig {
            n_ctx: 1024,
            n_threads: 2,
            n_threads_batch: 2,
            ..Default::default()
        },
    )
    .expect("thinker client");

    // Both are live at the same time and each serves its own model.
    let routed = router
        .complete(short("Say hi."))
        .expect("router completion");
    let thought = thinker
        .complete(short("Say hi."))
        .expect("thinker completion");
    assert!(!routed.text.is_empty(), "router generated nothing");
    assert!(!thought.text.is_empty(), "thinker generated nothing");

    // The router still works after the thinker has been used: the two engines
    // hold separate contexts and neither disturbs the other's slots.
    let again = router.complete(short("Say hi.")).expect("router again");
    assert!(!again.text.is_empty());
}

#[serial]
#[test]
fn per_engine_config_is_clamped_not_trusted() {
    // `InferenceConfig`'s fields are public, so a caller can hand over
    // `n_slots: 0`, which would divide by zero when the per-slot budget is
    // computed. The engine must clamp rather than panic.
    let client = Client::with_config(
        model_230m(),
        InferenceConfig {
            n_slots: 0,
            n_ctx: 0,
            n_threads: 0,
            n_threads_batch: 0,
            kv_cache: true,
        },
    )
    .expect("client with a degenerate config");
    // n_ctx clamps up to n_slots (1), leaving a 1-token budget — too small for
    // any prompt, so this must be a clean error rather than a crash.
    let err = client.complete(short("Hi")).expect_err("budget is 1 token");
    assert!(matches!(err, LlamaError::Inference(_)), "got {err:?}");
}

#[serial]
#[test]
fn two_clients_can_be_alive_at_once() {
    // Regression: a global join made this deadlock. The second constructor
    // never returned while the first client was alive.
    let first = Client::new(model_230m()).expect("first client");
    within(Duration::from_secs(60), "second Client::new", || {
        let second = Client::new(model_230m()).expect("second client");
        let result = second.complete(short("Hi")).expect("second completion");
        assert!(!result.text.is_empty());
    });
    // The first client is still usable after the second has come and gone.
    let result = first.complete(short("Hi")).expect("first completion");
    assert!(!result.text.is_empty());
}

#[serial]
#[test]
fn engines_starting_simultaneously_all_survive_backend_init() {
    // Regression: `LlamaBackend::init()` ran outside the `OnceLock`
    // initializer, so simultaneous engine startups raced. The loser got
    // `BackendAlreadyInitialized` before the winner had stored its value, saw
    // an empty cell, and failed its entire engine with `InferenceCrashed`.
    //
    // `two_clients_can_be_alive_at_once` caught this only ~1 run in 5, and
    // only when it happened to run first in the binary (before another test
    // had already initialized the backend). Releasing the clients from a
    // barrier makes the contention deterministic instead of incidental.
    const N: usize = 4;
    let barrier = Arc::new(Barrier::new(N));
    let handles: Vec<_> = (0..N)
        .map(|i| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                let client = Client::new(model_230m()).expect("client");
                client
                    .complete(short("Hi"))
                    .unwrap_or_else(|e| panic!("client {i} failed: {e}"))
            })
        })
        .collect();
    for (i, h) in handles.into_iter().enumerate() {
        let result = h.join().unwrap_or_else(|_| panic!("client {i} panicked"));
        assert!(!result.text.is_empty(), "client {i} generated nothing");
    }
}

#[serial]
#[test]
fn dropping_an_engine_joins_promptly_while_a_slot_is_busy() {
    // The shutdown flag exists for exactly this: with every slot busy, the
    // command channel is not drained, so a Shutdown *message* alone would not
    // be seen until the in-flight request exhausted its generation budget.
    // SAFETY: this test is `#[serial]` and no other thread reads these vars.
    unsafe { std::env::set_var("LLAMAD_N_SLOTS", "1") };
    let engine = Engine::start(model_230m()).expect("start engine");

    let (token_tx, mut token_rx) = tokio::sync::mpsc::unbounded_channel();
    engine
        .send(InferCmd::RunStream {
            // Far more tokens than the drop below should ever wait for.
            request: Request::new("write a long story about a dragon").with_max_tokens(4000),
            token_tx,
            done_tx: None,
        })
        .expect("send stream");
    assert!(
        token_rx.blocking_recv().is_some(),
        "generation should be under way before the drop"
    );
    // Hold the receiver open so the slot is not cancelled by a disconnect —
    // the shutdown path is what must free it.
    let started = Instant::now();
    drop(engine);
    let elapsed = started.elapsed();
    drop(token_rx);
    assert!(
        elapsed < Duration::from_secs(10),
        "dropping a busy engine took {elapsed:?}; it should abort within a decode step"
    );

    unsafe { std::env::remove_var("LLAMAD_N_SLOTS") };
}

#[serial]
#[test]
fn engine_shutdown_is_explicit_and_idempotent_with_drop() {
    let engine = Engine::start(model_230m()).expect("start engine");
    within(Duration::from_secs(60), "Engine::shutdown", move || {
        engine.shutdown();
    });
}

#[serial]
#[tokio::test]
async fn async_api_works_inside_a_tokio_runtime() {
    // The blocking API panics here by tokio's design; these are the methods
    // async callers are told to use, so they must be exercised in a runtime.
    let client = Client::new(model_230m()).expect("load model");

    let result = client
        .complete_async(short("Hi"))
        .await
        .expect("async completion");
    assert!(!result.text.is_empty());
    assert!(result.generated_tokens > 0);

    let mut stream = client.complete_stream(short("Hi")).expect("start stream");
    let mut count = 0;
    while let Some(token) = stream.next_token_async().await {
        assert!(!token.is_empty());
        count += 1;
    }
    assert!(count > 0, "streaming should yield at least one token");

    let result = client
        .complete_stream(short("Hi"))
        .expect("start stream")
        .into_result_async()
        .await
        .expect("async stream result");
    assert!(!result.text.is_empty());
    assert!(result.prompt_tokens > 0);

    // Dropping a Client joins its threads; doing that on a runtime thread
    // must not deadlock against the runtime.
    drop(client);
}
