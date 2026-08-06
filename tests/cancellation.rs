//! Cancellation end-to-end. Runs in its own test binary (each `tests/*.rs`
//! file is a separate process), so the `LLAMAD_N_SLOTS=1` env var set here
//! affects only this test's server — it cannot leak into the parallel
//! processes of other integration test files.

use std::time::Duration;

use llamad::inference::start_inference;
use llamad::protocol::{InferCmd, Request};
use serial_test::serial;

mod common;
use common::model_hybrid;

#[serial]
#[test]
fn drop_token_stream_cancels_inference() {
    // Single-slot server: a cancel-does-nothing regression forces the next
    // request to queue behind the first's full generation budget instead of
    // occupying a spare slot.
    // SAFETY: this is the only test in this binary; no other thread reads
    // or writes these env vars.
    unsafe { std::env::set_var("LLAMAD_N_SLOTS", "1") };
    let engine = start_inference(model_hybrid()).expect("start inference thread");

    // First request: a long generation. Receive one token to confirm it is
    // mid-flight, then drop the receiver — the inference thread's next emit
    // fails and the slot is cancelled.
    let (token_tx1, mut token_rx1) = tokio::sync::mpsc::unbounded_channel();
    engine
        .send(InferCmd::RunStream {
            request: Request::new("write a long story about a dragon").with_max_tokens(5000),
            token_tx: token_tx1,
            done_tx: None,
        })
        .expect("send first stream");
    assert!(
        token_rx1.blocking_recv().is_some(),
        "first stream should start generating before cancel"
    );
    drop(token_rx1);

    // Second request, same single slot. With cancellation working the slot
    // frees immediately and the first token arrives in well under the
    // first request's remaining generation budget (~2048+ tokens). With a
    // cancel-does-nothing regression the second request queues behind that
    // remaining budget and produces no token within the bound.
    let (token_tx2, mut token_rx2) = tokio::sync::mpsc::unbounded_channel();
    let (done_tx2, done_rx2) = tokio::sync::oneshot::channel();
    engine
        .send(InferCmd::RunStream {
            request: Request::new("second request").with_max_tokens(20),
            token_tx: token_tx2,
            done_tx: Some(done_tx2),
        })
        .expect("send second stream");

    let first_token = poll_first_token(&mut token_rx2, Duration::from_secs(5));
    assert!(first_token.is_some(), "second stream ended without a token");

    // The slot was genuinely recycled: the second request completes with a
    // result, not just a first token.
    while token_rx2.blocking_recv().is_some() {}
    let result = done_rx2
        .blocking_recv()
        .expect("second stream done signal")
        .expect("second stream completion");
    assert!(!result.text.is_empty());
    assert!(result.generated_tokens <= 20);

    engine.shutdown();
}

/// First token of a stream, or `None` if it ends; panics on timeout.
/// (tokio's unbounded receiver has no `blocking_recv_timeout`, so poll with
/// a hard deadline.)
fn poll_first_token(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<String>,
    timeout: Duration,
) -> Option<String> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match rx.try_recv() {
            Ok(token) => return Some(token),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                if std::time::Instant::now() >= deadline {
                    panic!(
                        "second request produced no token within {timeout:?}: \
                         the first slot was not freed by cancellation"
                    );
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => return None,
        }
    }
}
