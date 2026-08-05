use std::sync::mpsc;
use std::time::Duration;

use llamad::client::Client;
use llamad::inference::start_inference;
use llamad::protocol::{InferCmd, InferResult, LlamaError, Request};
use serial_test::serial;

mod common;
use common::{captured_lines, install_capture_layer, model_1_2b, model_230m, reuse_model_path};

fn short_req(text: &str) -> Request {
    Request::new(text).with_max_tokens(10)
}

fn oneshot_bridge<T: Send + 'static>(rx: tokio::sync::oneshot::Receiver<T>) -> mpsc::Receiver<T> {
    let (tx, out_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let val = rx.blocking_recv().ok();
        if let Some(v) = val {
            let _ = tx.send(v);
        }
    });
    out_rx
}

// ── Client integration tests (230M model) ──────────────────────────────────

#[serial]
#[test]
fn client_completes_non_streaming() {
    let client = Client::new(model_230m()).expect("load 230M model");
    let result = client.complete(short_req("Hi")).expect("completion");
    assert!(!result.text.is_empty(), "should generate text");
    assert!(result.prompt_tokens > 0, "should count prompt tokens");
    assert!(result.generated_tokens > 0, "should count generated tokens");
    // EOG tokens are not counted and the budget caps generation: the count
    // must never exceed the requested max_tokens (10 via short_req).
    assert!(
        result.generated_tokens <= 10,
        "generated {} tokens, max_tokens=10",
        result.generated_tokens
    );
    eprintln!(
        "  prompt={} gen={} text={:?}",
        result.prompt_tokens,
        result.generated_tokens,
        &result.text[..result.text.len().min(80)]
    );
}

#[serial]
#[test]
fn client_completes_streaming() {
    let client = Client::new(model_230m()).expect("load 230M model");
    let mut stream = client
        .complete_stream(short_req("Hi"))
        .expect("start stream");
    let mut count = 0;
    while let Some(token) = stream.next_token() {
        assert!(!token.is_empty(), "token should not be empty");
        count += 1;
    }
    assert!(count > 0, "should receive at least one token");
    assert!(!stream.text().is_empty(), "accumulated text should exist");
    eprintln!(
        "  streamed {count} tokens, text={:?}",
        &stream.text()[..stream.text().len().min(80)]
    );
}

#[serial]
#[test]
fn client_stream_into_result_returns_stats() {
    let client = Client::new(model_230m()).expect("load 230M model");
    let stream = client
        .complete_stream(short_req("Hi"))
        .expect("start stream");
    let result = stream.into_result().expect("final result");
    assert!(!result.text.is_empty());
    assert!(result.prompt_tokens > 0);
    assert!(result.generated_tokens > 0);
    // EOG tokens are not counted and the budget caps generation (max 10).
    assert!(
        result.generated_tokens <= 10,
        "generated {} tokens, max_tokens=10",
        result.generated_tokens
    );
}

#[serial]
#[test]
fn client_completes_with_system_prompt() {
    let client = Client::new(model_230m()).expect("load 230M model");
    let req = Request::new("what is 1+1?")
        .with_system("answer in one word")
        .with_max_tokens(20);
    // complete_text takes &str, so use complete
    let result = client.complete(req).expect("completion");
    assert!(!result.text.is_empty());
    eprintln!(
        "  system-prompt result: {:?}",
        &result.text[..result.text.len().min(80)]
    );
}

#[serial]
#[test]
fn client_completes_with_temperature() {
    let client = Client::new(model_230m()).expect("load 230M model");
    let req = Request::new("hello")
        .with_temperature(0.8)
        .with_max_tokens(20);
    let result = client.complete(req).expect("completion");
    assert!(!result.text.is_empty());
}

#[serial]
#[test]
fn client_model_load_failure_surfaces_as_crash() {
    // End-to-end: the inference thread fails to load the model, the
    // preprocess thread exits, and the normal Client path reports the crash
    // to the caller (not just a channel-level probe).
    let client = Client::new("/nonexistent-model.gguf").expect("spawn should succeed");
    let err = client.complete(Request::new("x")).unwrap_err();
    assert!(matches!(err, LlamaError::InferenceCrashed));
}

// ── Slot queuing (N_SLOTS=4, 5 requests — 5th queued until a slot frees) ──

#[serial]
#[test]
fn batch_overflow_queues_fifth() {
    let engine = start_inference(model_230m()).expect("start inference thread");

    let mut bridges: Vec<mpsc::Receiver<Result<InferResult, LlamaError>>> = Vec::new();
    for i in 0..5 {
        let (oneshot_tx, oneshot_rx) = tokio::sync::oneshot::channel();
        let bridge_rx = oneshot_bridge(oneshot_rx);
        engine
            .send(InferCmd::Run {
                request: short_req(&format!("number {i}")),
                resp: oneshot_tx,
            })
            .expect("send cmd");
        bridges.push(bridge_rx);
    }

    let mut results: Vec<Result<InferResult, LlamaError>> = Vec::new();
    for rx in bridges {
        match rx.recv_timeout(Duration::from_secs(60)) {
            Ok(r) => results.push(r),
            Err(_) => panic!("timed out waiting for result"),
        }
    }

    let rejected: Vec<_> = results.iter().filter(|r| r.is_err()).collect();
    assert_eq!(
        rejected.len(),
        0,
        "all 5 requests should succeed (5th queued)"
    );

    engine.shutdown();
}

#[serial]
#[test]
fn batch_overflow_streaming_queues_fifth() {
    let engine = start_inference(model_230m()).expect("start inference thread");

    let mut bridges: Vec<mpsc::Receiver<Result<InferResult, LlamaError>>> = Vec::new();
    let mut keepalive: Vec<tokio::sync::mpsc::UnboundedReceiver<String>> = Vec::new();
    for _ in 0..5 {
        let (token_tx, token_rx) = tokio::sync::mpsc::unbounded_channel();
        keepalive.push(token_rx);
        let (done_tx, done_rx) = tokio::sync::oneshot::channel();
        let bridge_rx = oneshot_bridge(done_rx);
        engine
            .send(InferCmd::RunStream {
                request: short_req("hi"),
                token_tx,
                done_tx: Some(done_tx),
            })
            .expect("send cmd");
        bridges.push(bridge_rx);
    }
    // keepalive dropped at end of test scope

    let mut results: Vec<Result<InferResult, LlamaError>> = Vec::new();
    for rx in bridges {
        match rx.recv_timeout(Duration::from_secs(60)) {
            Ok(r) => results.push(r),
            Err(_) => panic!("timed out waiting for streaming result"),
        }
    }

    let rejected: Vec<_> = results.iter().filter(|r| r.is_err()).collect();
    assert_eq!(
        rejected.len(),
        0,
        "all 5 streaming requests should succeed (5th queued)"
    );

    engine.shutdown();
}

// ─── Multi-turn slot recycling ─────────────────────────────────────────────

#[serial]
#[test]
fn four_consecutive_completions_recycle_slots() {
    let client = Client::new(model_230m()).expect("load 230M model");
    for i in 0..4 {
        let result = client
            .complete(short_req(&format!("request {i}")))
            .expect("completion");
        assert!(!result.text.is_empty(), "request {i} should produce text");
    }
}

// ─── 1.2B model smoke test ─────────────────────────────────────────────────

#[serial]
#[test]
fn model_1_2b_completes() {
    let client = Client::new(model_1_2b()).expect("load 1.2B model");
    let result = client
        .complete(Request::new("Hello").with_max_tokens(20))
        .expect("completion");
    assert!(!result.text.is_empty());
    eprintln!(
        "  1.2B: prompt={} gen={} text={:?}",
        result.prompt_tokens,
        result.generated_tokens,
        &result.text[..result.text.len().min(80)]
    );
}

// ── KV-prefix-reuse e2e moved to tests/reuse.rs ─────────────────────────────
//
// The reuse-path real-model tests (identical resend, partial-prefix triple,
// probe-verdict-inverted) live in `tests/reuse.rs`, their own test binary: the
// warning-absence assertion there needs a sink that no LFM2 client from this
// binary can pollute, and binaries run sequentially so concurrent model loads
// stay bounded. This binary keeps the LFM2.5 degraded-path contracts, incl.
// `probe_verdict_degrades_loudly_on_lfm2` below.

// ── Probe verdict capture ("degrade loudly" contract) ──────────────────────
//
// Capture layer, sink, and model-path helpers live in `tests/common/mod.rs`;
// `install_capture_layer`/`captured_lines` are re-exported there. The warning
// is emitted by the inference thread strictly before the main loop serves any
// request, and `complete` returns only after the request is served — so a
// single assert after `complete` reads settled state (happens-before via the
// response channel; no polling needed).

#[serial]
#[test]
fn probe_verdict_degrades_loudly_on_lfm2() {
    // The "degrade loudly" contract (review gaps G1/G2): on LFM2.5 the
    // startup probe must return false and the inference thread must emit the
    // degradation warning. The warning is emitted strictly before the main
    // loop serves any request, and `complete` returns only after the request
    // is served — so by the time we assert, the warning is guaranteed
    // recorded (happens-before via the response channel; no polling needed).
    //
    // NOTE: this test's name pins model-specific behavior (03-M-4) — "probe
    // must be false" is a property of the hybrid-arch GGUF at MODEL_230M. If
    // that file is ever swapped for a rewind-capable model, the name, the
    // skip below, and the asserted expectation must move together.
    //
    // NOTE (03-M-3): the exact substring "KV prefix reuse disabled" is an
    // intentional degrade-contract assertion — rephrasing inference.rs's
    // warning breaks this test; keep the substring and the warning in sync
    // deliberately.
    //
    // Skip guards: this test pins the *default-on* degraded contract. An
    // ambient LLAMAD_KV_CACHE=false spelling ("0","false","no","off" per
    // config.rs read_env_bool) legitimately disables the machinery, so the
    // warning would never fire — skip instead of failing (03-I-2). And when
    // LLAMAD_TEST_MODEL is set, the caller is exercising the rewind-capable
    // inverted contract, which is `reuse_engages_on_rewind_capable_model`'s
    // job — skip here with a clear message.
    if let Ok(raw) = std::env::var("LLAMAD_KV_CACHE") {
        let trimmed = raw.trim().to_ascii_lowercase();
        if matches!(trimmed.as_str(), "0" | "false" | "no" | "off") {
            eprintln!("skipped: LLAMAD_KV_CACHE explicitly disabled");
            return;
        }
    }
    if reuse_model_path().is_some() {
        eprintln!(
            "skipped: LLAMAD_TEST_MODEL (or bundled attention model) present — the \
             inverted (reuse-engages) probe contract is covered by \
             reuse_engages_on_rewind_capable_model"
        );
        return;
    }
    install_capture_layer();
    let client = Client::new(model_230m()).expect("load 230M model");
    let result = client.complete(short_req("Hi")).expect("completion");
    assert!(!result.text.is_empty(), "completion should generate text");

    assert!(
        captured_lines()
            .iter()
            .any(|l| l.contains("KV prefix reuse disabled")),
        "inference thread never warned about KV prefix reuse being disabled \
         (the 'degrade loudly' contract is broken, or the probe returned \
         true on LFM2). Captured log lines: {:?}",
        captured_lines()
    );
}
