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

/// Run `f` with `LLAMAD_N_SLOTS` set, then restore the environment.
///
/// `Engine::start` reads its config synchronously before spawning any thread,
/// so the variable only needs to be live across the `start_inference` call.
/// Safe because every caller is `#[serial]`.
fn with_n_slots<T>(n: &str, f: impl FnOnce() -> T) -> T {
    // SAFETY: serialized by `#[serial]`; no other test in this binary reads
    // this variable concurrently.
    unsafe { std::env::set_var("LLAMAD_N_SLOTS", n) };
    let out = f();
    unsafe { std::env::remove_var("LLAMAD_N_SLOTS") };
    out
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

// ── Slot queuing (5 requests — 5th queued until a slot frees) ──
//
// These set `LLAMAD_N_SLOTS=4` explicitly: the default is 1 slot, under which
// all five requests would queue trivially and the test would no longer
// exercise the "more requests than slots" path it exists for.

#[serial]
#[test]
fn batch_overflow_queues_fifth() {
    let engine = with_n_slots("4", || {
        start_inference(model_230m()).expect("start inference thread")
    });

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
    let engine = with_n_slots("4", || {
        start_inference(model_230m()).expect("start inference thread")
    });

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

// ─── Stop sequences ────────────────────────────────────────────────────────

#[serial]
#[test]
fn stop_sequence_truncates_the_completion() {
    // Greedy decoding (temperature 0.0) makes the model's output stable, so a
    // marker lifted out of an unconstrained run is guaranteed to appear in a
    // second run of the same prompt. That sidesteps guessing what a 230M model
    // will say while still exercising the real generation loop.
    let client = Client::new(model_230m()).expect("load 230M model");
    let base = Request::new("Count from one to ten.")
        .with_temperature(0.0)
        .with_max_tokens(40);

    let full = client.complete(base.clone()).expect("unconstrained run");
    assert!(
        full.text.chars().count() > 6,
        "need a few characters to cut at, got {:?}",
        full.text
    );

    // A two-character marker starting at the third character, sliced on char
    // boundaries so multi-byte output cannot panic.
    let start = full.text.char_indices().nth(2).unwrap().0;
    let end = full.text[start..]
        .char_indices()
        .nth(2)
        .map(|(i, _)| start + i)
        .unwrap_or(full.text.len());
    let marker = full.text[start..end].to_owned();

    let stopped = client
        .complete(base.push_stop(marker.clone()))
        .expect("stopped run");

    assert!(
        !stopped.text.contains(&marker),
        "stop text {marker:?} must not appear in {:?}",
        stopped.text
    );
    assert!(
        full.text.starts_with(&stopped.text),
        "stopped output {:?} must be a prefix of {:?}",
        stopped.text,
        full.text
    );
    assert!(
        stopped.text.len() < full.text.len(),
        "stopping must shorten the output"
    );
}

#[serial]
#[test]
fn stop_sequence_truncates_a_streamed_completion() {
    // The streamed bytes must agree with the final result: no fragment of the
    // stop sequence may leak to the client before the match resolves.
    let client = Client::new(model_230m()).expect("load 230M model");
    let base = Request::new("Count from one to ten.")
        .with_temperature(0.0)
        .with_max_tokens(40);

    let full = client.complete(base.clone()).expect("unconstrained run");
    let start = full.text.char_indices().nth(2).unwrap().0;
    let end = full.text[start..]
        .char_indices()
        .nth(2)
        .map(|(i, _)| start + i)
        .unwrap_or(full.text.len());
    let marker = full.text[start..end].to_owned();

    let mut stream = client
        .complete_stream(base.push_stop(marker.clone()))
        .expect("start stream");
    let mut streamed = String::new();
    while let Some(tok) = stream.next_token() {
        streamed.push_str(&tok);
    }
    let result = stream.into_result().expect("stream result");

    assert_eq!(
        streamed, result.text,
        "streamed text must match the final result exactly"
    );
    assert!(
        !streamed.contains(&marker),
        "stop text {marker:?} leaked into the stream"
    );
}

// ─── Grammar-constrained generation ────────────────────────────────────────

/// A GBNF grammar admitting exactly one of three words.
const CHOICE_GRAMMAR: &str = r#"root ::= [ \n]* ("yes" | "no" | "maybe")"#;

#[serial]
#[test]
fn grammar_constrains_output_to_the_grammar() {
    // The point of grammar support: a 230M model asked an open question will
    // ramble, but under a constraint the output *cannot* be anything else.
    let client = Client::new(model_230m()).expect("load 230M model");
    let result = client
        .complete(
            Request::new("Is the sky blue? Answer in one word.")
                .with_grammar(CHOICE_GRAMMAR)
                .with_max_tokens(20),
        )
        .expect("grammar-constrained completion");

    assert!(
        ["yes", "no", "maybe"].contains(&result.text.trim()),
        "output {:?} escaped the grammar",
        result.text
    );
}

#[serial]
#[test]
fn grammar_applies_under_greedy_decoding() {
    // Greedy decoding replaces the whole sampler chain, so the grammar has to
    // be spliced in ahead of the greedy tail rather than dropped. A regression
    // here would silently produce unconstrained output at temperature 0.
    let client = Client::new(model_230m()).expect("load 230M model");
    let result = client
        .complete(
            Request::new("Is the sky blue? Answer in one word.")
                .with_grammar(CHOICE_GRAMMAR)
                .with_temperature(0.0)
                .with_max_tokens(20),
        )
        .expect("greedy grammar completion");

    assert!(
        ["yes", "no", "maybe"].contains(&result.text.trim()),
        "greedy output {:?} escaped the grammar",
        result.text
    );
}

#[serial]
#[test]
fn grammar_produces_parseable_json() {
    // The use case people actually want: structured extraction from a small
    // model. Unconstrained, a 230M model emits malformed JSON constantly.
    let grammar = r#"
root   ::= "{" ws "\"ok\"" ws ":" ws bool ws "}"
bool   ::= "true" | "false"
ws     ::= " "?
"#;
    let client = Client::new(model_230m()).expect("load 230M model");
    let result = client
        .complete(
            Request::new("Reply with a JSON object having key ok.")
                .with_grammar(grammar)
                .with_max_tokens(40),
        )
        .expect("json grammar completion");

    let parsed: serde_json::Value =
        serde_json::from_str(result.text.trim()).unwrap_or_else(|e| {
            panic!("grammar output {:?} did not parse as JSON: {e}", result.text)
        });
    assert!(parsed.get("ok").is_some_and(serde_json::Value::is_boolean));
}

#[serial]
#[test]
fn invalid_grammar_is_rejected_without_killing_the_engine() {
    // `LlamaSampler::grammar` panics on a grammar llama.cpp cannot parse
    // (llama.cpp returns null, the binding unwraps it). Grammars arrive over
    // the socket, so an unguarded panic would be a remote kill of the whole
    // engine. The bad request must fail alone and the engine keep serving.
    let client = Client::new(model_230m()).expect("load 230M model");

    let err = client
        .complete(Request::new("hi").with_grammar("root ::= ((( unterminated"))
        .expect_err("a malformed grammar must be rejected");
    assert!(
        matches!(err, LlamaError::Protocol(_)),
        "expected a protocol error, got {err:?}"
    );

    // The engine survived: an ordinary request still works.
    let ok = client
        .complete(Request::new("Say hi.").with_max_tokens(10))
        .expect("engine must still serve after a rejected grammar");
    assert!(!ok.text.is_empty());
}

#[serial]
#[test]
fn grammar_with_a_null_byte_is_rejected() {
    // The other panic path in the binding: `CString::new` on a string with an
    // interior null byte.
    let client = Client::new(model_230m()).expect("load 230M model");
    let err = client
        .complete(Request::new("hi").with_grammar("root ::= \"a\"\0"))
        .expect_err("a null byte must be rejected");
    assert!(matches!(err, LlamaError::Protocol(_)), "got {err:?}");
}

#[serial]
#[test]
fn unknown_grammar_root_is_rejected() {
    let client = Client::new(model_230m()).expect("load 230M model");
    let err = client
        .complete(
            Request::new("hi")
                .with_grammar(CHOICE_GRAMMAR)
                .with_grammar_root("nonexistent"),
        )
        .expect_err("an unknown start rule must be rejected");
    assert!(matches!(err, LlamaError::Protocol(_)), "got {err:?}");
}

// ─── Sampling seed ─────────────────────────────────────────────────────────

#[serial]
#[test]
fn identical_requests_vary_without_an_explicit_seed() {
    // The defect this guards: a hardcoded sampler seed made every request with
    // the same prompt return byte-identical text, so `temperature` had no
    // observable effect across requests. Three samples at temperature 1.0 over
    // 40 tokens of a 65K vocabulary — all three matching would mean the seed
    // is pinned, not that sampling got lucky.
    let client = Client::new(model_230m()).expect("load 230M model");
    let req = Request::new("Invent a short unusual sentence.")
        .with_temperature(1.0)
        .with_max_tokens(40);

    let a = client.complete(req.clone()).expect("first").text;
    let b = client.complete(req.clone()).expect("second").text;
    let c = client.complete(req).expect("third").text;

    assert!(
        !(a == b && b == c),
        "three unseeded samples were identical, seed appears pinned: {a:?}"
    );
}

#[serial]
#[test]
fn an_explicit_seed_reproduces_the_same_text() {
    // The other half of the contract: opting into a seed must give back
    // reproducibility, which is what makes the random default safe to ship.
    let client = Client::new(model_230m()).expect("load 230M model");
    let req = Request::new("Invent a short unusual sentence.")
        .with_temperature(1.0)
        .with_max_tokens(40)
        .with_seed(20260805);

    let first = client.complete(req.clone()).expect("first").text;
    let second = client.complete(req).expect("second").text;
    assert_eq!(first, second, "a pinned seed must be reproducible");
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
