// Panics are the assertion mechanism in tests: a failed `unwrap` here is a
// reported failure, not a fault reachable from untrusted input. The
// crate-level denies in `lib.rs` cover the library's production paths,
// which is the surface that matters.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::cache::*;
use super::sampler::*;
use super::slot::*;
use crate::protocol::*;
use llama_cpp_4::prelude::*;

fn base_slot() -> Slot {
    Slot {
        request_id: None,
        prompt_tokens: vec![],
        n_prompt: 0,
        gen_count: 0,
        max_tokens: 10,
        kv_pos: 0,
        sampler: LlamaSampler::greedy(),
        utf8_buf: vec![],
        text: String::new(),
        pending_token: None,
        prompt_phase: false,
        resp: None,
        token_tx: None,
        done_tx: None,
        cache_tokens: vec![],
        stop: vec![],
        stop_max_len: 0,
        emitted: 0,
    }
}

/// A slot that streams to `tx` and stops on `stop`.
fn streaming_slot(tx: tokio::sync::mpsc::Sender<String>, stop: &[&str]) -> Slot {
    let stop: Vec<String> = stop.iter().map(|s| (*s).to_owned()).collect();
    Slot {
        token_tx: Some(tx),
        stop_max_len: stop.iter().map(String::len).max().unwrap_or(0),
        stop,
        ..base_slot()
    }
}

/// Drain everything currently queued on a token receiver.
fn drain(rx: &mut tokio::sync::mpsc::Receiver<String>) -> String {
    let mut out = String::new();
    while let Ok(chunk) = rx.try_recv() {
        out.push_str(&chunk);
    }
    out
}

#[test]
fn test_slot_push_text_streams_and_accumulates() {
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let mut slot = streaming_slot(tx, &[]);
    assert_eq!(slot.push_text("hello"), Emit::Continue);
    assert_eq!(slot.text, "hello");
    // With no stop sequences there is nothing to withhold: the fragment
    // reaches the client in the same step it was generated.
    assert_eq!(drain(&mut rx), "hello");
}

#[test]
fn test_slot_push_text_no_token_tx() {
    let mut slot = base_slot();
    assert_eq!(slot.push_text("world"), Emit::Continue);
    assert_eq!(slot.text, "world");
}

#[test]
fn test_slot_finish_sends_via_resp() {
    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    let mut slot = Slot {
        n_prompt: 3,
        gen_count: 5,
        kv_pos: 5,
        text: "hello world".into(),
        resp: Some(resp_tx),
        ..base_slot()
    };
    assert!(slot.finish().is_ok());
    let result = resp_rx.blocking_recv().unwrap().unwrap();
    assert_eq!(result.text, "hello world");
    assert_eq!(result.prompt_tokens, 3);
    assert_eq!(result.generated_tokens, 5);
}

#[test]
fn test_slot_finish_sends_via_done_tx() {
    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    let mut slot = Slot {
        n_prompt: 1,
        gen_count: 2,
        kv_pos: 3,
        text: "hi".into(),
        done_tx: Some(done_tx),
        ..base_slot()
    };
    assert!(slot.finish().is_ok());
    let result = done_rx.blocking_recv().unwrap().unwrap();
    assert_eq!(result.text, "hi");
    assert_eq!(result.prompt_tokens, 1);
    assert_eq!(result.generated_tokens, 2);
}

// ── push_raw_bytes (UTF-8 decode buffering) ──────────────────────────

fn slot_with_utf8_buf(utf8_buf: Vec<u8>) -> Slot {
    Slot {
        utf8_buf,
        ..base_slot()
    }
}

#[test]
fn test_push_raw_bytes_ascii_in_single_push() {
    let mut slot = slot_with_utf8_buf(Vec::new());
    assert_eq!(
        slot.push_raw_bytes(b"hello").unwrap().as_deref(),
        Some("hello")
    );
    assert!(slot.utf8_buf.is_empty());
}

#[test]
fn test_push_raw_bytes_partial_multibyte_split_across_pushes() {
    let mut slot = slot_with_utf8_buf(Vec::new());
    // "€" = E2 82 AC: the first push ends mid-sequence...
    assert_eq!(slot.push_raw_bytes(&[0xE2, 0x82]).unwrap(), None);
    assert_eq!(slot.utf8_buf, vec![0xE2, 0x82]);
    // ...the second push completes it.
    assert_eq!(slot.push_raw_bytes(&[0xAC]).unwrap().as_deref(), Some("€"));
    assert!(slot.utf8_buf.is_empty());
}

#[test]
fn test_push_raw_bytes_multibyte_char_across_chunk_boundary() {
    let mut slot = slot_with_utf8_buf(Vec::new());
    assert_eq!(slot.push_raw_bytes(&[0xE2]).unwrap(), None);
    assert_eq!(
        slot.push_raw_bytes(&[0x82, 0xAC]).unwrap().as_deref(),
        Some("€")
    );
}

#[test]
fn test_push_raw_bytes_multiple_chars_in_one_push() {
    let mut slot = slot_with_utf8_buf(Vec::new());
    assert_eq!(
        slot.push_raw_bytes(b"a\xE2\x82\xACb").unwrap().as_deref(),
        Some("a€b")
    );
    assert!(slot.utf8_buf.is_empty());
}

#[test]
fn test_push_raw_bytes_empty_input_returns_none() {
    let mut slot = slot_with_utf8_buf(Vec::new());
    assert_eq!(slot.push_raw_bytes(&[]).unwrap(), None);
    assert!(slot.utf8_buf.is_empty());
}

#[test]
fn test_push_raw_bytes_invalid_continuation_byte_errors() {
    let mut slot = slot_with_utf8_buf(Vec::new());
    let err = slot.push_raw_bytes(&[0xE2, 0x82, 0x41]).unwrap_err();
    assert!(matches!(err, LlamaError::Inference(_)));
}

#[test]
fn test_push_raw_bytes_invalid_sequence_after_valid_prefix_errors() {
    let mut slot = slot_with_utf8_buf(Vec::new());
    // "ab" flushes; the dangling lead byte 0xC3 stays buffered.
    assert_eq!(
        slot.push_raw_bytes(b"ab\xC3").unwrap().as_deref(),
        Some("ab")
    );
    assert_eq!(slot.utf8_buf, vec![0xC3]);
    // A non-continuation byte after the lead byte is permanently invalid.
    let err = slot.push_raw_bytes(b"(").unwrap_err();
    assert!(matches!(err, LlamaError::Inference(_)));
}

#[test]
fn test_slot_cancel_sends_error_through_resp() {
    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    let mut slot = Slot {
        resp: Some(resp_tx),
        ..base_slot()
    };
    slot.cancel(LlamaError::Inference("test error".into()));
    let err = resp_rx.blocking_recv().unwrap().unwrap_err();
    assert!(matches!(err, LlamaError::Inference(_)));
}

#[test]
fn test_slot_push_text_reports_disconnect() {
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    drop(rx);
    let mut slot = streaming_slot(tx, &[]);
    assert_eq!(slot.push_text("lost"), Emit::Disconnected);
}

#[test]
fn test_cancel_all_clears_all_slots_and_sends_error() {
    let (resp_tx1, resp_rx1) = tokio::sync::oneshot::channel();
    let (resp_tx2, resp_rx2) = tokio::sync::oneshot::channel();
    let mut slots: Vec<Option<Slot>> = vec![
        Some(Slot {
            resp: Some(resp_tx1),
            ..base_slot()
        }),
        None,
        Some(Slot {
            resp: Some(resp_tx2),
            ..base_slot()
        }),
    ];
    let mut slot_cache: Vec<Vec<LlamaToken>> = vec![
        vec![LlamaToken::new(1), LlamaToken::new(2)],
        Vec::new(),
        Vec::new(),
    ];
    cancel_all(&mut slots, &mut slot_cache, LlamaError::InferenceCrashed);
    assert!(slots.iter().all(|s| s.is_none()));
    assert!(slot_cache.iter().all(|c| c.is_empty()));
    // Receivers kept alive: every active slot's client must observe the
    // crash, not just the slots being cleared.
    let err1 = resp_rx1.blocking_recv().unwrap().unwrap_err();
    let err2 = resp_rx2.blocking_recv().unwrap().unwrap_err();
    assert!(matches!(err1, LlamaError::InferenceCrashed));
    assert!(matches!(err2, LlamaError::InferenceCrashed));
}

#[test]
fn test_slot_finish_sends_via_resp_and_done_tx() {
    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    let mut slot = Slot {
        n_prompt: 3,
        gen_count: 5,
        kv_pos: 5,
        text: "hello world".into(),
        resp: Some(resp_tx),
        done_tx: Some(done_tx),
        ..base_slot()
    };
    assert!(slot.finish().is_ok());
    let via_resp = resp_rx.blocking_recv().unwrap().unwrap();
    let via_done = done_rx.blocking_recv().unwrap().unwrap();
    assert_eq!(via_resp.text, "hello world");
    assert_eq!(via_done.text, "hello world");
    assert_eq!(via_resp.prompt_tokens, 3);
    assert_eq!(via_done.generated_tokens, 5);
}

#[test]
fn test_slot_finish_fails_on_incomplete_utf8_tail() {
    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    let mut slot = Slot {
        n_prompt: 3,
        gen_count: 5,
        kv_pos: 5,
        text: "hello ".into(),
        utf8_buf: vec![0xF0, 0x9F], // incomplete 4-byte emoji
        resp: Some(resp_tx),
        done_tx: Some(done_tx),
        ..base_slot()
    };
    let res = slot.finish();
    assert!(res.is_err());
    let err = res.unwrap_err();
    assert!(matches!(err, LlamaError::Inference(_)));
    assert!(
        err.to_string()
            .contains("incomplete UTF-8 sequence at end of generation")
    );

    let r1_err = resp_rx.blocking_recv().unwrap().unwrap_err();
    let r2_err = done_rx.blocking_recv().unwrap().unwrap_err();
    assert!(matches!(r1_err, LlamaError::Inference(_)));
    assert!(matches!(r2_err, LlamaError::Inference(_)));
}

#[test]
fn test_slot_cancel_sends_error_through_done_tx() {
    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    let mut slot = Slot {
        done_tx: Some(done_tx),
        ..base_slot()
    };
    slot.cancel(LlamaError::Inference("test error".into()));
    let err = done_rx.blocking_recv().unwrap().unwrap_err();
    assert!(matches!(err, LlamaError::Inference(_)));
}

// ── stop sequences ────────────────────────────────────────────────────

#[test]
fn test_stop_sequence_within_a_single_fragment() {
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let mut slot = streaming_slot(tx, &["STOP"]);
    assert_eq!(slot.push_text("abcSTOPdef"), Emit::Stopped);
    // The stop text and everything after it is dropped from the result.
    assert_eq!(slot.text, "abc");
    assert_eq!(drain(&mut rx), "abc");
}

#[test]
fn test_stop_sequence_split_across_fragments() {
    // The case a per-token check cannot catch: no single fragment
    // contains the stop sequence, only their concatenation does.
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let mut slot = streaming_slot(tx, &["<|end|>"]);
    assert_eq!(slot.push_text("hello <|"), Emit::Continue);
    assert_eq!(slot.push_text("end"), Emit::Continue);
    assert_eq!(slot.push_text("|>tail"), Emit::Stopped);
    assert_eq!(slot.text, "hello ");
    assert_eq!(drain(&mut rx), "hello ");
}

#[test]
fn test_partial_stop_match_is_withheld_then_released() {
    // "<|" looks like the start of the stop sequence, so it must not be
    // streamed. When the next fragment rules the match out, the withheld
    // bytes are released — losing them would silently truncate output.
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let mut slot = streaming_slot(tx, &["<|end|>"]);
    assert_eq!(slot.push_text("hi <|"), Emit::Continue);
    assert_eq!(drain(&mut rx), "hi ", "the ambiguous tail must be held");
    assert_eq!(slot.push_text("not"), Emit::Continue);
    assert_eq!(drain(&mut rx), "<|not", "ruled out, so released");
    assert_eq!(slot.text, "hi <|not");
}

#[test]
fn test_no_stop_sequences_withholds_nothing() {
    // The default path must not pay any streaming latency.
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let mut slot = streaming_slot(tx, &[]);
    slot.push_text("anything at all");
    assert_eq!(drain(&mut rx), "anything at all");
    assert_eq!(slot.holdback(), 0);
}

#[test]
fn test_unrelated_text_is_not_withheld_despite_stop_sequences() {
    // Hold-back is computed from the actual tail, not from stop_max_len,
    // so output that looks nothing like a stop streams immediately.
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let mut slot = streaming_slot(tx, &["<|end|>"]);
    slot.push_text("plain prose");
    assert_eq!(drain(&mut rx), "plain prose");
}

#[test]
fn test_earliest_of_several_stop_sequences_wins() {
    let (tx, _rx) = tokio::sync::mpsc::channel(64);
    let mut slot = streaming_slot(tx, &["END", "XY"]);
    assert_eq!(slot.push_text("aXYbENDc"), Emit::Stopped);
    // "XY" occurs before "END": the cut is the earliest match, not the
    // first stop sequence in list order.
    assert_eq!(slot.text, "a");
}

#[test]
fn test_stop_sequence_at_the_very_start_yields_empty_text() {
    let (tx, _rx) = tokio::sync::mpsc::channel(64);
    let mut slot = streaming_slot(tx, &["STOP"]);
    assert_eq!(slot.push_text("STOP"), Emit::Stopped);
    assert_eq!(slot.text, "");
}

#[test]
fn test_stop_matching_across_multibyte_characters() {
    // Hold-back offsets are byte counts; landing mid-character would panic
    // on the `str` slice. "é" is two bytes, so the naive offset is unsafe.
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let mut slot = streaming_slot(tx, &["éé"]);
    assert_eq!(slot.push_text("café"), Emit::Continue);
    assert_eq!(slot.push_text("é!"), Emit::Stopped);
    assert_eq!(slot.text, "caf");
    assert_eq!(drain(&mut rx), "caf");
}

#[test]
fn test_multibyte_tail_is_not_split_when_withheld() {
    // A withheld tail must stay on a char boundary even when the stop
    // sequence is multi-byte and only partially matched.
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let mut slot = streaming_slot(tx, &["→→"]);
    assert_eq!(slot.push_text("go →"), Emit::Continue);
    assert_eq!(drain(&mut rx), "go ");
    assert_eq!(slot.push_text(" on"), Emit::Continue);
    assert_eq!(drain(&mut rx), "→ on");
}

#[test]
fn test_finish_releases_withheld_tail() {
    // Generation ending on EOG or the token cap while a partial stop match
    // is outstanding must still deliver those bytes.
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    let mut slot = Slot {
        done_tx: Some(done_tx),
        ..streaming_slot(tx, &["<|end|>"])
    };
    assert_eq!(slot.push_text("done <|"), Emit::Continue);
    assert_eq!(drain(&mut rx), "done ");
    assert!(slot.finish().is_ok());
    assert_eq!(drain(&mut rx), "<|", "withheld tail must be flushed");
    assert_eq!(done_rx.blocking_recv().unwrap().unwrap().text, "done <|");
}

#[test]
fn test_stop_sequence_without_streaming_client_still_truncates() {
    // Non-streaming requests have no token_tx; the stop logic must work
    // off `text` alone.
    let mut slot = Slot {
        stop: vec!["STOP".to_owned()],
        stop_max_len: 4,
        ..base_slot()
    };
    assert_eq!(slot.push_text("keepSTOPdrop"), Emit::Stopped);
    assert_eq!(slot.text, "keep");
}

// ── sampler chain ─────────────────────────────────────────────────────

/// The model-card defaults, as `resolve_sampling` produces them for a
/// request that overrides nothing.
fn default_params() -> SamplingParams {
    SamplingParams {
        temperature: 0.1,
        top_k: 50,
        top_p: 0.95,
        repeat_penalty: 1.05,
        repeat_last_n: -1,
        seed: crate::protocol::RANDOM_SEED,
    }
}

#[test]
fn test_sampler_chain_content_and_order_for_temperature() {
    assert_eq!(
        sampler_chain(default_params()),
        vec![
            SamplerKind::Penalties {
                penalty_last_n: -1,
                penalty_repeat: 1.05,
            },
            SamplerKind::TopK { k: 50 },
            SamplerKind::TopP {
                p: 0.95,
                min_keep: 1
            },
            SamplerKind::Temp { t: 0.1 },
            SamplerKind::Dist {
                seed: crate::protocol::RANDOM_SEED
            },
        ]
    );
}

#[test]
fn test_sampler_chain_threads_every_parameter_through() {
    // Regression guard for hardcoding: each field must reach its sampler.
    // These values are deliberately unlike the defaults, so a chain that
    // ignored the argument would fail on every element.
    let params = SamplingParams {
        temperature: 1.5,
        top_k: 7,
        top_p: 0.3,
        repeat_penalty: 1.9,
        repeat_last_n: 64,
        seed: 12345,
    };
    assert_eq!(
        sampler_chain(params),
        vec![
            SamplerKind::Penalties {
                penalty_last_n: 64,
                penalty_repeat: 1.9,
            },
            SamplerKind::TopK { k: 7 },
            SamplerKind::TopP {
                p: 0.3,
                min_keep: 1
            },
            SamplerKind::Temp { t: 1.5 },
            SamplerKind::Dist { seed: 12345 },
        ]
    );
}

#[test]
fn test_sampler_chain_default_seed_is_the_random_sentinel() {
    // The defect this guards: a fixed seed makes every request with the
    // same prompt return identical text, so `temperature` has no
    // observable effect across requests. Each slot builds its own sampler,
    // so the sentinel is the only thing that varies them.
    let chain = sampler_chain(default_params());
    assert!(matches!(
        chain.last(),
        Some(SamplerKind::Dist { seed }) if *seed == crate::protocol::RANDOM_SEED
    ));
}

#[test]
fn test_sampler_chain_zero_or_negative_temperature_means_greedy() {
    let zero = SamplingParams {
        temperature: 0.0,
        ..default_params()
    };
    let negative = SamplingParams {
        temperature: -0.5,
        ..default_params()
    };
    assert_eq!(sampler_chain(zero), vec![]);
    assert_eq!(sampler_chain(negative), vec![]);
}

#[test]
fn test_sampler_kinds_construct_over_ffi() {
    // Smoke: every `SamplerKind` builds a real llama.cpp sampler. This is
    // the part of `build_sampler` that needs no model; the grammar branch
    // and the full assembly are covered by the real-model tests in
    // `tests/integration.rs`, since `LlamaSampler::grammar` needs a
    // vocabulary.
    for kind in sampler_chain(default_params()) {
        drop(kind.into_sampler(32000));
    }
    drop(LlamaSampler::greedy());
}

// ── longest_common_prefix ────────────────────────────────────────────────

#[test]
fn test_lcp_full_match() {
    let a = vec![LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(3)];
    let b = a.clone();
    assert_eq!(longest_common_prefix(&a, &b), 3);
}

#[test]
fn test_lcp_partial_match() {
    let a = vec![LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(3)];
    let b = vec![LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(9)];
    assert_eq!(longest_common_prefix(&a, &b), 2);
}

#[test]
fn test_lcp_no_match() {
    let a = vec![LlamaToken::new(1), LlamaToken::new(2)];
    let b = vec![LlamaToken::new(9), LlamaToken::new(8)];
    assert_eq!(longest_common_prefix(&a, &b), 0);
}

#[test]
fn test_lcp_new_prompt_is_prefix_of_cache() {
    let cache = vec![LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(3)];
    let prompt = vec![LlamaToken::new(1), LlamaToken::new(2)];
    assert_eq!(longest_common_prefix(&cache, &prompt), 2);
}

#[test]
fn test_lcp_empty_sides() {
    assert_eq!(longest_common_prefix(&[], &[LlamaToken::new(1)]), 0);
    assert_eq!(longest_common_prefix(&[LlamaToken::new(1)], &[]), 0);
    assert_eq!(longest_common_prefix(&[], &[]), 0);
}

// ── select_slot (prefix-aware routing) ───────────────────────────────────

fn toks(ids: &[i32]) -> Vec<LlamaToken> {
    ids.iter().copied().map(LlamaToken::new).collect()
}

/// `slots` from a spec of which indices are occupied.
fn slots_with_busy(n: usize, busy: &[usize]) -> Vec<Option<Slot>> {
    (0..n)
        .map(|i| {
            if busy.contains(&i) {
                Some(slot_for_request())
            } else {
                None
            }
        })
        .collect()
}

#[test]
fn test_select_slot_cold_start_picks_lowest_index() {
    // No slot holds a cache: every candidate scores 0, so the tie-break
    // must reproduce first-free selection exactly.
    let slots = slots_with_busy(4, &[]);
    let cache = vec![Vec::new(); 4];
    assert_eq!(select_slot(&slots, &cache, &toks(&[1, 2, 3])), Some(0));
}

#[test]
fn test_select_slot_prefers_free_slot_holding_the_prefix() {
    // The regression this whole function exists for: slot 0 is busy, so
    // first-free would hand the request to slot 1 and full-prefill —
    // while slot 3 sits idle holding an exact prefix of the prompt.
    let slots = slots_with_busy(4, &[0]);
    let mut cache = vec![Vec::new(); 4];
    cache[3] = toks(&[1, 2, 3]);
    assert_eq!(select_slot(&slots, &cache, &toks(&[1, 2, 3, 4])), Some(3));
}

#[test]
fn test_select_slot_picks_longest_prefix_not_merely_any_match() {
    let slots = slots_with_busy(4, &[]);
    let mut cache = vec![Vec::new(); 4];
    cache[1] = toks(&[1, 2]); // lcp 2
    cache[2] = toks(&[1, 2, 3, 4]); // lcp 4 — the best
    cache[3] = toks(&[1]); // lcp 1
    assert_eq!(select_slot(&slots, &cache, &toks(&[1, 2, 3, 4])), Some(2));
}

#[test]
fn test_select_slot_ignores_cache_of_busy_slots() {
    // Slot 2 holds the perfect prefix but is mid-generation; its KV is not
    // available for reuse, so routing must fall to a free slot.
    let slots = slots_with_busy(4, &[2]);
    let mut cache = vec![Vec::new(); 4];
    cache[2] = toks(&[1, 2, 3, 4]);
    assert_eq!(select_slot(&slots, &cache, &toks(&[1, 2, 3, 4])), Some(0));
}

#[test]
fn test_select_slot_equal_prefix_lengths_resolve_to_lowest_index() {
    // `max_by_key` keeps the last maximum; without the `Reverse(i)` term
    // this would return 3 and churn caches for no benefit.
    let slots = slots_with_busy(4, &[]);
    let mut cache = vec![Vec::new(); 4];
    cache[1] = toks(&[1, 2]);
    cache[3] = toks(&[1, 2]);
    assert_eq!(select_slot(&slots, &cache, &toks(&[1, 2, 9])), Some(1));
}

#[test]
fn test_select_slot_all_busy_returns_none() {
    let slots = slots_with_busy(2, &[0, 1]);
    let cache = vec![Vec::new(); 2];
    assert_eq!(select_slot(&slots, &cache, &toks(&[1])), None);
}

#[test]
fn test_select_slot_single_slot_always_routes_to_zero() {
    // The default configuration: one slot, and a non-matching cache must
    // not make selection fail — it routes to 0 and full-prefills.
    let slots = slots_with_busy(1, &[]);
    let cache = vec![toks(&[7, 8, 9])];
    assert_eq!(select_slot(&slots, &cache, &toks(&[1, 2])), Some(0));
}

// ── begin_request ────────────────────────────────────────────────────────

fn slot_for_request() -> Slot {
    Slot {
        prompt_phase: true,
        ..base_slot()
    }
}

#[test]
fn test_begin_request_reuses_shared_prefix() {
    let mut slot = slot_for_request();
    let cached = vec![LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(3)];
    let tokens = vec![LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(9)];
    let lcp = slot.begin_request(tokens.clone(), cached);
    assert_eq!(lcp, 2);
    assert_eq!(slot.kv_pos, 2);
    assert_eq!(slot.prompt_tokens, tokens);
    assert_eq!(
        slot.cache_tokens,
        vec![LlamaToken::new(1), LlamaToken::new(2)]
    );
}

#[test]
fn test_begin_request_no_match_clears_cache() {
    let mut slot = slot_for_request();
    let cached = vec![LlamaToken::new(1), LlamaToken::new(2)];
    let tokens = vec![LlamaToken::new(9)];
    let lcp = slot.begin_request(tokens.clone(), cached);
    assert_eq!(lcp, 0);
    assert_eq!(slot.kv_pos, 0);
    assert!(slot.cache_tokens.is_empty());
}

#[test]
fn test_begin_request_empty_cache() {
    let mut slot = slot_for_request();
    let tokens = vec![LlamaToken::new(7)];
    let lcp = slot.begin_request(tokens.clone(), vec![]);
    assert_eq!(lcp, 0);
    assert_eq!(slot.kv_pos, 0);
    assert!(slot.cache_tokens.is_empty());
    assert_eq!(slot.prompt_tokens, tokens);
}

#[test]
fn test_finish_prefill_keeps_mirror_in_sync() {
    // Mirror-corruption regression guard: after a partial-prefix reuse
    // (lcp = 2 of 3), the mirror must equal the full new prompt and
    // kv_pos == cache_tokens.len(). A `drain(..)` instead of
    // `drain(reused..)` would grow the mirror to lcp + n_prompt and break
    // the invariant — invisible to identical-prompt e2e, fatal for any
    // agentic loop that re-sends a growing prompt with a shared prefix.
    let mut slot = slot_for_request();
    let cached = vec![LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(3)];
    let tokens = vec![LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(9)];
    let lcp = slot.begin_request(tokens.clone(), cached);
    assert_eq!(lcp, 2);
    assert_eq!(
        slot.cache_tokens,
        vec![LlamaToken::new(1), LlamaToken::new(2)]
    );
    slot.finish_prefill();
    assert_eq!(slot.kv_pos as usize, slot.cache_tokens.len());
    assert_eq!(slot.cache_tokens, tokens);
    assert!(!slot.prompt_phase);
    assert!(slot.prompt_tokens.is_empty());
}

// ── KV-rewind probe degradation (KV-rewind amendment) ─────────────────────

#[test]
fn test_reuse_gated_on_probe_verdict() {
    // The finish sites retain a completed sequence only when the env knob
    // is on AND the startup probe found partial rewind support. With
    // kv_rewind == false (LFM2) nothing is ever retained.
    assert!(reuse_allowed(true, true));
    assert!(!reuse_allowed(true, false));
    assert!(!reuse_allowed(false, true));
    assert!(!reuse_allowed(false, false));
}

#[test]
fn test_probe_refused_means_next_request_full_prefills() {
    // With kv_rewind == false the finish sites never retain, so the slot
    // cache entry the next fill takes is empty: begin_request falls back
    // to a full prefill from position 0, matching the pre-plan behavior.
    let mut slot = slot_for_request();
    let cached = Vec::new(); // what fill_empty_slot mem::takes after a no-retain finish
    let tokens = vec![LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(3)];
    let lcp = slot.begin_request(tokens.clone(), cached);
    assert_eq!(lcp, 0);
    assert_eq!(slot.kv_pos, 0);
    assert!(slot.cache_tokens.is_empty());
    slot.finish_prefill();
    assert_eq!(slot.kv_pos as usize, slot.cache_tokens.len());
    assert_eq!(slot.cache_tokens, tokens);
}

#[test]
fn test_partial_rewind_refused_falls_back_to_full_prefill() {
    // Defense in depth (review 04-H1 spirit): if the checked partial
    // removal is refused at fill time, the slot must not trust the cache
    // mirror — warn, clear it, and rewind to a full prefill.
    let mut slot = slot_for_request();
    let cached = vec![LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(3)];
    let tokens = vec![LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(9)];
    let lcp = slot.begin_request(tokens.clone(), cached);
    assert_eq!(lcp, 2);
    assert_eq!(slot.kv_pos, 2);
    fallback_to_full_prefill(&mut slot, lcp, 0);
    assert!(slot.cache_tokens.is_empty());
    assert_eq!(slot.kv_pos, 0);
    // The prefill loop reads `reused = cache_tokens.len()`: 0 means the
    // whole prompt is re-added at [0, n_prompt) — full prefill converges.
    let reused = slot.cache_tokens.len() as i32;
    assert_eq!(reused, 0);
    slot.finish_prefill();
    assert_eq!(slot.kv_pos as usize, slot.cache_tokens.len());
    assert_eq!(slot.cache_tokens, tokens);
}
// ── Properties ───────────────────────────────────────────────────────────
//
// The example tests above cover the cases someone thought of. These cover
// the shape of the input space: arbitrary Unicode split at arbitrary byte
// offsets, which is exactly what a tokenizer hands `push_raw_bytes`, and
// what the stop-sequence hold-back has to slice without landing inside a
// character.

use proptest::prelude::*;
use proptest::test_runner::{FailurePersistence, FileFailurePersistence};

/// Where a failing case's seed is written.
///
/// Proptest's default is `SourceParallel("proptest-regressions")`, which
/// resolves to a sibling of `src/` — a directory in the repository root.
/// Pointed under `tests/` instead so the regression corpus sits with the
/// rest of the test material rather than at the top level.
///
/// `SourceParallel`, not `Direct`: the file is still named after the source
/// it came from, so property tests added to another module get their own
/// file instead of appending to this one.
///
/// The argument is resolved against the *sibling* of the directory holding
/// `lib.rs` — the crate root — so it is written without a leading `../`.
/// A `../` here silently escapes the repository: proptest reports no error,
/// the tests still pass, and the seeds land in the parent directory.
///
/// These seeds are committed on purpose. They are the inputs that killed
/// mutants of `floor_char_boundary` and the UTF-8 buffer — a two-byte "é"
/// sliced at offset 1, a four-byte 𐀀, and a string mixing NUL, vertical
/// tab and U+0080 — so keeping them turns a lucky random draw into a
/// deterministic case that runs before any new ones.
fn regression_file() -> Option<Box<dyn FailurePersistence>> {
    Some(Box::new(FileFailurePersistence::SourceParallel(
        "tests/proptest-regressions",
    )))
}

/// Split `bytes` at the given offsets, each taken modulo the remaining
/// length so any generated vector produces a valid chunking.
fn chunk(bytes: &[u8], cuts: &[usize]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut rest = bytes;
    for &c in cuts {
        if rest.is_empty() {
            break;
        }
        let at = c % (rest.len() + 1);
        let (head, tail) = rest.split_at(at);
        out.push(head.to_vec());
        rest = tail;
    }
    out.push(rest.to_vec());
    out
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: regression_file(),
        ..ProptestConfig::default()
    })]
    /// Reassembly is lossless however the tokenizer splits the bytes.
    ///
    /// This is the contract `push_raw_bytes` exists for: a multi-byte
    /// character may arrive across two detokenized pieces, and the decoder
    /// must buffer rather than emit a replacement character or error.
    #[test]
    fn prop_utf8_reassembly_is_lossless(
        text in ".{0,64}",
        cuts in prop::collection::vec(any::<usize>(), 0..8),
    ) {
        let mut slot = base_slot();
        let mut out = String::new();
        for piece in chunk(text.as_bytes(), &cuts) {
            // Valid UTF-8 split anywhere can never be *permanently*
            // invalid, only incomplete — so this must never error.
            let got = slot.push_raw_bytes(&piece)
                .expect("a split of valid utf-8 is never permanently invalid");
            if let Some(s) = got {
                out.push_str(&s);
            }
        }
        prop_assert_eq!(&out, &text, "reassembled text differs from input");
        prop_assert!(
            slot.utf8_buf.is_empty(),
            "buffer left holding {:?} after complete input",
            slot.utf8_buf
        );
    }

    /// Every byte in `text` is eventually streamed, and nothing else is.
    ///
    /// Hold-back delays bytes that might still grow into a stop sequence;
    /// the risk is that a delayed byte is never released, silently
    /// truncating output. Generation ending without a match must flush.
    #[test]
    fn prop_no_stop_match_streams_everything(
        pieces in prop::collection::vec(".{0,16}", 0..6),
        stop in "[a-zA-Z]{1,4}",
    ) {
        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        let mut slot = streaming_slot(tx, &[stop.as_str()]);

        let mut stopped = false;
        for piece in &pieces {
            if slot.push_text(piece) == Emit::Stopped {
                stopped = true;
                break;
            }
        }
        let expected = slot.text.clone();
        if !stopped {
            assert!(slot.finish().is_ok());
        }
        // `finish` takes `text`, so compare against the snapshot above.
        let streamed = drain(&mut rx);
        prop_assert_eq!(
            streamed,
            expected,
            "streamed bytes must equal the accumulated text exactly"
        );
    }

    /// Hold-back never splits a character.
    ///
    /// The offsets come from byte lengths, so a naive implementation
    /// slices mid-character and panics. Multi-byte stop sequences and
    /// multi-byte output are generated deliberately here.
    #[test]
    fn prop_holdback_lands_on_a_char_boundary(
        text in "[aé→𝄞]{0,24}",
        stop in "[aé→𝄞]{1,3}",
    ) {
        let (tx, _rx) = tokio::sync::mpsc::channel(64);
        let mut slot = streaming_slot(tx, &[stop.as_str()]);
        slot.text = text;
        let held = slot.holdback();
        prop_assert!(held <= slot.text.len());
        let cut = slot.text.len() - held;
        prop_assert!(
            slot.text.is_char_boundary(cut),
            "holdback {held} puts the cut at {cut}, inside a character of {:?}",
            slot.text
        );
    }

    /// `floor_char_boundary` returns a boundary at or below its argument.
    #[test]
    fn prop_floor_char_boundary_is_a_boundary(
        text in "[aé→𝄞]{0,24}",
        i in 0usize..64,
    ) {
        let got = floor_char_boundary(&text, i);
        prop_assert!(got <= i.min(text.len()));
        prop_assert!(text.is_char_boundary(got));
    }

    /// The prefix length is a real common prefix, and maximal.
    #[test]
    fn prop_lcp_is_a_maximal_common_prefix(
        a in prop::collection::vec(0i32..6, 0..12),
        b in prop::collection::vec(0i32..6, 0..12),
    ) {
        let ta: Vec<LlamaToken> = a.iter().copied().map(LlamaToken::new).collect();
        let tb: Vec<LlamaToken> = b.iter().copied().map(LlamaToken::new).collect();
        let n = longest_common_prefix(&ta, &tb);
        prop_assert!(n <= ta.len().min(tb.len()));
        prop_assert_eq!(&ta[..n], &tb[..n], "reported prefix is not shared");
        // Maximal: the next pair, if any, must differ.
        if n < ta.len().min(tb.len()) {
            prop_assert_ne!(ta[n], tb[n], "prefix could have been longer");
        }
    }
}
