//! KV-prefix-reuse real-model e2e (review round 2, 03-tests-quality I-1, rec 1).
//!
//! Own test binary (each `tests/*.rs` is its own process) so the
//! warning-absence assertion in `reuse_engages_on_rewind_capable_model` cannot
//! be polluted by LFM2 clients from the `integration` binary (cargo runs test
//! binaries sequentially, and within this binary every test uses the same
//! rewind-capable model — no degrade warning is ever legitimately emitted).
//!
//! These close the review's G1/I2 gaps: the anchor (zero-token prefill rewind),
//! the finish-site retention tail-clear, and the fill-time partial clear are
//! executed against real llama.cpp state. The determinism tests use greedy
//! decoding (temperature 0.0 → no sampler chain → `LlamaSampler::greedy`),
//! which is deterministic, so identical prompts must produce identical text.
//! The exact-equality contract is valid only because this build is CPU-only
//! (no BLAS/GPU), uses static ggml row partitioning, and compares within one
//! process; it would flake if BLAS/GPU offload or dynamic thread counts are
//! ever introduced.
//!
//! Model selection: `LLAMAD_TEST_MODEL` when set, else the bundled
//! SmolLM2-135M in `models/` when present. Without either, the
//! LFM2.5 hybrid falls back to the degraded path and the reuse assertions
//! would be vacuous — the tests skip with a clear message instead (the
//! degraded-path contract is covered by `probe_verdict_degrades_loudly_on_lfm2`
//! in the `integration` binary).

use llamad::client::Client;
use llamad::protocol::Request;
use serial_test::serial;

mod common;
use common::{captured_lines, install_capture_layer, reuse_model_path};

fn req(text: &str) -> Request {
    Request::new(text).with_temperature(0.0).with_max_tokens(20)
}

#[serial]
#[test]
fn identical_prompt_twice_succeeds() {
    // Greedy-determinism regression guard for the KV-reuse machinery: on a
    // rewind-capable model the second request's tokens exactly match the
    // retained cache (lcp == n_prompt), so the zero-token prefill anchor and
    // the C1 rewind ([n_prompt-1, ∞) clear) run. A mirror-corruption or
    // anchor-regression bug would fail the equality assertion loudly.
    let Some(model) = reuse_model_path() else {
        skip_without_reuse_model();
        return;
    };
    let client = Client::new(model).expect("load model");
    let first = client
        .complete(req("Count from one to five."))
        .expect("first completion");
    let second = client
        .complete(req("Count from one to five."))
        .expect("second completion");
    assert!(
        !first.text.is_empty(),
        "first completion should generate text"
    );
    assert!(
        !second.text.is_empty(),
        "second completion should generate text"
    );
    assert!(
        first.prompt_tokens > 0,
        "first completion should count prompt tokens"
    );
    assert!(
        second.prompt_tokens > 0,
        "second completion should count prompt tokens"
    );
    assert_eq!(
        first.text, second.text,
        "greedy decoding of the same prompt must be deterministic: the second \
         request shares the full cached prefix and must produce identical text"
    );
    eprintln!(
        "  identical resend: prompt={} gen1={} gen2={} text={:?}",
        first.prompt_tokens,
        first.generated_tokens,
        second.generated_tokens,
        &first.text[..first.text.len().min(80)]
    );
}

#[serial]
#[test]
fn partial_prefix_triple_keeps_outputs_consistent() {
    // Three greedy requests sharing the chat-template prefix and diverging
    // only in the tail, with the first and third identical. On a rewind-
    // capable model the first two populate the cache and the third shares a
    // real prefix, so the mirror + fill-time KV reconciliation (partial clear
    // [lcp, ∞) + fallback) ARE exercised — a mirror-corruption or
    // under-clear bug would silently diverge, caught here.
    let Some(model) = reuse_model_path() else {
        skip_without_reuse_model();
        return;
    };
    let client = Client::new(model).expect("load model");
    let prompts = [
        "Say hello in French.",
        "Say hello in German.",
        "Say hello in French.",
    ];
    let results: Vec<_> = prompts
        .iter()
        .map(|&p| client.complete(req(p)).expect("completion"))
        .collect();
    for (i, r) in results.iter().enumerate() {
        assert!(
            !r.text.is_empty(),
            "request {i} ({:?}) should generate text",
            prompts[i]
        );
        assert!(
            r.prompt_tokens > 0,
            "request {i} ({:?}) should count prompt tokens",
            prompts[i]
        );
    }
    assert_eq!(
        results[0].text, results[2].text,
        "request 3 re-sends request 1's prompt; greedy decoding must reproduce \
         its text (a mirror-corruption bug would silently diverge here)"
    );
    eprintln!(
        "  partial-prefix triple: req1={:?} req2={:?} req3={:?}",
        &results[0].text[..results[0].text.len().min(40)],
        &results[1].text[..results[1].text.len().min(40)],
        &results[2].text[..results[2].text.len().min(40)],
    );
}

#[serial]
#[serial]
#[test]
fn reuse_engages_on_rewind_capable_model() {
    // The inverted probe contract (03-tests-quality I-1, rec. 1; 02-rec-2):
    // on a rewind-capable (pure-attention) model the probe must return true,
    // so NO "KV prefix reuse disabled" warning fires AND the "slot X reused N
    // cached tokens" debug line (inference.rs) is emitted on an identical
    // resend. Both halves are asserted so a warning regression and a
    // reuse-silence regression each fail with a distinct message. The sink is
    // process-local (this binary loads only the rewind-capable model), so the
    // warning-absence half is unambiguous. The "reused" debug line is emitted
    // during the second fill and `complete` returns only after the request is
    // served — a single assert reads settled state (happens-before via the
    // response channel; no polling needed).
    let Some(model) = reuse_model_path() else {
        skip_without_reuse_model();
        return;
    };
    install_capture_layer();
    let client = Client::new(model).expect("load model");
    client
        .complete(req("Count from one to five."))
        .expect("first completion");
    client
        .complete(req("Count from one to five."))
        .expect("second completion");

    let lines = captured_lines();
    assert!(
        !lines.iter().any(|l| l.contains("KV prefix reuse disabled")),
        "the probe must return true for attention-only models — a 'KV prefix \
         reuse disabled' warning means the reuse path did not engage. \
         Captured log lines: {lines:?}"
    );
    assert!(
        lines.iter().any(|l| l.contains("reused")),
        "KV reuse did not engage: the identical resend must emit the 'slot N \
         reused M cached tokens' debug line (mirror/fill-time reconciliation \
         did not run). Captured log lines: {lines:?}"
    );
}

/// Shared skip path: the reuse-path assertions are vacuous on a hybrid model
/// (probe false → degraded full-prefill), so refuse to run rather than pass
/// silently (the "empty provider completions must fail loudly" preference,
/// applied to test coverage).
fn skip_without_reuse_model() {
    eprintln!(
        "skipped: no rewind-capable model available — set LLAMAD_TEST_MODEL to \
         an attention-only GGUF, or put SmolLM2-135M-Instruct-Q4_K_M.gguf in \
         models/"
    );
}
