//! Prompt preprocessing: chat templating + tokenization, off the inference
//! thread so long prompts never stall active decodes.
//!
//! The preprocess thread shares the loaded [`LlamaModel`] with the inference
//! thread via `Arc`. Both operations it performs (`apply_chat_template`,
//! `str_to_token`) are pure `&self` calls on the immutable model, which is
//! declared `Send + Sync` with no interior mutability.

use std::sync::Arc;
use std::sync::mpsc;

use llama_cpp_4::prelude::*;

use crate::config::InferenceConfig;
use crate::inference::build_grammar_sampler;
use crate::protocol::{
    Grammar, InferCmd, LlamaError, PreparedCmd, PreparedRequest, RANDOM_SEED, Request,
    SamplingParams,
};

/// Default sampling parameters: the Liquid AI LFM2.5 model-card values.
///
/// These are *defaults*, not policy — every one is overridable per request.
/// They are one vendor's recommendation for one model family, and a crate
/// that loads arbitrary GGUFs must not silently impose them.
const DEFAULT_TEMPERATURE: f32 = 0.1;
const DEFAULT_TOP_K: i32 = 50;
const DEFAULT_TOP_P: f32 = 0.95;
const DEFAULT_REPEAT_PENALTY: f32 = 1.05;
const DEFAULT_REPEAT_LAST_N: i32 = -1;

pub(crate) fn clamp_temperature(raw: f32) -> f32 {
    if raw.is_nan() || raw.is_sign_negative() {
        0.0
    } else if raw > 2.0 {
        2.0
    } else {
        raw
    }
}

/// Resolve one optional float: `None` or NaN falls back to `default`,
/// anything else is clamped into `[lo, hi]`.
///
/// NaN resolves to the default rather than to a bound: a NaN top-p is a
/// caller mistake, and silently turning it into 0.0 would collapse the
/// candidate set to a single token with no diagnostic.
fn resolve_f32(raw: Option<f32>, default: f32, lo: f32, hi: f32) -> f32 {
    match raw {
        Some(v) if !v.is_nan() => v.clamp(lo, hi),
        _ => default,
    }
}

/// Sampling parameters for a request, with defaults applied and every value
/// clamped into a range llama.cpp accepts.
pub(crate) fn resolve_sampling(request: &Request) -> SamplingParams {
    SamplingParams {
        temperature: clamp_temperature(request.temperature.unwrap_or(DEFAULT_TEMPERATURE)),
        // 0 disables top-k in llama.cpp; negatives are meaningless.
        top_k: request.top_k.unwrap_or(DEFAULT_TOP_K).max(0),
        top_p: resolve_f32(request.top_p, DEFAULT_TOP_P, 0.0, 1.0),
        repeat_penalty: resolve_f32(request.repeat_penalty, DEFAULT_REPEAT_PENALTY, 0.0, 2.0),
        // -1 means "the whole context"; anything below that is meaningless.
        repeat_last_n: request.repeat_last_n.unwrap_or(DEFAULT_REPEAT_LAST_N).max(-1),
        // No seed means a fresh random one per request, so two identical
        // requests at a non-zero temperature do not return identical text.
        seed: request.seed.unwrap_or(RANDOM_SEED),
    }
}

/// Drop stop sequences that cannot match usefully.
///
/// An empty string is a prefix of every position, so keeping one would end
/// generation before the first token and hand back an empty completion. It is
/// dropped rather than rejected: an empty entry is almost always an artifact
/// of building the list programmatically, not a deliberate request.
pub(crate) fn normalize_stop(stop: &[String]) -> Vec<String> {
    stop.iter().filter(|s| !s.is_empty()).cloned().collect()
}

/// Resolve and validate a request's grammar against the model's vocabulary.
///
/// Validation happens here, on the preprocess thread, because this is the
/// trust boundary: a grammar arriving over the socket is untrusted input, and
/// a bad one must come back to that one caller as an error rather than reach
/// the inference thread, where llama.cpp's null-pointer-on-parse-failure would
/// otherwise take down the engine for every client.
///
/// An unset grammar yields `None`. `grammar_root` defaults to `"root"`, the
/// GBNF convention, and is ignored when no grammar is given.
pub(crate) fn resolve_grammar(
    model: &LlamaModel,
    request: &Request,
) -> std::result::Result<Option<Grammar>, LlamaError> {
    let Some(text) = request.grammar.as_ref() else {
        return Ok(None);
    };
    let root = request.grammar_root.as_deref().unwrap_or("root");

    // Checked explicitly for a clear message; `build_grammar_sampler` would
    // otherwise catch the resulting `CString::new` panic and report it as a
    // syntax error, which would send the caller looking in the wrong place.
    if text.contains('\0') || root.contains('\0') {
        return Err(LlamaError::Protocol(
            "grammar and grammar_root must not contain null bytes".into(),
        ));
    }

    // Compile once to prove it parses, then discard: the inference thread
    // builds the sampler it actually uses, from this same validated text.
    build_grammar_sampler(model, text, root)?;
    Ok(Some(Grammar {
        text: text.clone(),
        root: root.to_owned(),
    }))
}

pub(crate) fn build_messages(
    req: &Request,
) -> std::result::Result<Vec<LlamaChatMessage>, LlamaError> {
    let mut msgs = Vec::new();
    if let Some(sys) = &req.system {
        msgs.push(LlamaChatMessage::new("system".into(), sys.clone())?);
    }
    for h in &req.history {
        msgs.push(LlamaChatMessage::new(h.role.clone(), h.content.clone())?);
    }
    msgs.push(LlamaChatMessage::new("user".into(), req.prompt.clone())?);
    Ok(msgs)
}

/// Compute the generation budget for a prompt, enforcing the per-slot cap.
///
/// Returns an error when the prompt alone exhausts the slot budget.
pub(crate) fn compute_max_gen(
    n_prompt: usize,
    max_tokens: Option<u32>,
    per_slot_budget: u32,
) -> std::result::Result<u32, LlamaError> {
    let max_tokens = max_tokens.unwrap_or(256);
    let max_gen = max_tokens.min(per_slot_budget.saturating_sub(n_prompt as u32));
    if max_gen == 0 {
        return Err(LlamaError::Inference(format!(
            "prompt too long ({} tokens, max {per_slot_budget} per slot)",
            n_prompt
        )));
    }
    Ok(max_gen)
}

/// Resolve the generation parameters for a tokenized prompt, applying the
/// request defaults: `max_tokens` defaults to 256, and every sampling field
/// to its model-card value (then clamped). Extracted from [`prepare_request`]
/// so the defaults contract is unit-testable without a loaded model.
pub(crate) fn resolve_defaults(
    n_prompt: usize,
    request: &Request,
    per_slot_budget: u32,
) -> std::result::Result<(u32, SamplingParams), LlamaError> {
    let max_gen = compute_max_gen(n_prompt, request.max_tokens, per_slot_budget)?;
    Ok((max_gen, resolve_sampling(request)))
}

/// Template, tokenize, and budget-check a request. Pure model reads only.
pub(crate) fn prepare_request(
    model: &LlamaModel,
    request: &Request,
    per_slot_budget: u32,
) -> std::result::Result<PreparedRequest, LlamaError> {
    let messages = build_messages(request)?;
    let prompt = model
        .apply_chat_template(None, &messages, true)
        .map_err(|e| LlamaError::Inference(format!("chat template: {e}")))?;
    let tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|e| LlamaError::Inference(e.to_string()))?;
    let (max_gen, sampling) = resolve_defaults(tokens.len(), request, per_slot_budget)?;
    Ok(PreparedRequest {
        tokens,
        max_gen,
        sampling,
        grammar: resolve_grammar(model, request)?,
        stop: normalize_stop(&request.stop),
    })
}

/// Preprocess thread: receive raw client commands, prepare them, forward
/// pre-tokenized work to the inference thread.
pub(super) fn preprocess_loop(
    cmd_rx: mpsc::Receiver<InferCmd>,
    model_rx: mpsc::Receiver<Arc<LlamaModel>>,
    prepared_tx: mpsc::Sender<PreparedCmd>,
    config: InferenceConfig,
) {
    let model = match model_rx.recv() {
        Ok(model) => model,
        Err(_) => {
            // Inference thread died during load; its channel closed, so any
            // pending client sends will fail with InferenceCrashed.
            tracing::warn!("inference thread exited before providing the model");
            return;
        }
    };

    let budget = config.per_slot_budget();
    for cmd in cmd_rx {
        match cmd {
            InferCmd::Run { request, resp } => match prepare_request(&model, &request, budget) {
                Ok(req) => {
                    match prepared_tx.send(PreparedCmd::Run { req, resp }) {
                        Ok(()) => {}
                        Err(mpsc::SendError(PreparedCmd::Run { resp, .. })) => {
                            let _ = resp.send(Err(LlamaError::InferenceCrashed));
                        }
                        Err(_) => unreachable!("Run arm can only fail with Run"),
                    }
                }
                Err(e) => {
                    let _ = resp.send(Err(e));
                }
            },
            InferCmd::RunStream {
                request,
                token_tx,
                done_tx,
            } => match prepare_request(&model, &request, budget) {
                Ok(req) => {
                    match prepared_tx.send(PreparedCmd::RunStream {
                        req,
                        token_tx,
                        done_tx,
                    }) {
                        Ok(()) => {}
                        Err(mpsc::SendError(PreparedCmd::RunStream {
                            token_tx, done_tx, ..
                        })) => {
                            if let Some(done) = done_tx {
                                let _ = done.send(Err(LlamaError::InferenceCrashed));
                            }
                            drop(token_tx);
                        }
                        Err(_) => unreachable!("RunStream arm can only fail with RunStream"),
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to prepare streaming request: {e}");
                    if let Some(done) = done_tx {
                        let _ = done.send(Err(e));
                    }
                    drop(token_tx);
                }
            },
            InferCmd::Shutdown => {
                let _ = prepared_tx.send(PreparedCmd::Shutdown);
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::HistoryMessage;

    #[test]
    fn test_compute_max_gen_default() {
        assert_eq!(compute_max_gen(10, None, 512).unwrap(), 256);
    }

    #[test]
    fn test_compute_max_gen_capped_by_request() {
        assert_eq!(compute_max_gen(10, Some(100), 512).unwrap(), 100);
    }

    #[test]
    fn test_compute_max_gen_capped_by_budget() {
        assert_eq!(compute_max_gen(400, Some(500), 512).unwrap(), 112);
    }

    #[test]
    fn test_compute_max_gen_prompt_too_long() {
        let err = compute_max_gen(512, Some(10), 512).unwrap_err();
        assert!(err.to_string().contains("prompt too long"));
        assert!(err.to_string().contains("512"));
    }

    #[test]
    fn test_compute_max_gen_prompt_exceeds_budget() {
        // n_prompt > budget: saturating_sub must clamp the difference to 0
        // (a plain `-` would underflow), surfacing the specific error with
        // both token counts.
        let err = compute_max_gen(600, Some(10), 512).unwrap_err();
        assert!(err.to_string().contains("prompt too long"));
        assert!(err.to_string().contains("600"));
        assert!(err.to_string().contains("512"));
    }

    #[test]
    fn test_compute_max_gen_zero_budget() {
        assert!(compute_max_gen(100, None, 0).is_err());
    }

    #[test]
    fn test_preprocess_exits_when_model_channel_closed() {
        // Inference thread died during model load → preprocess exits → client
        // sends fail → Client maps send failure to InferenceCrashed.
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (model_tx, model_rx) = mpsc::channel();
        drop(model_tx); // dropped immediately, as when the inference thread dies
        let (prepared_tx, _prepared_rx) = mpsc::channel();
        let cfg = InferenceConfig::default();
        let handle = std::thread::spawn(move || {
            preprocess_loop(cmd_rx, model_rx, prepared_tx, cfg);
        });
        handle.join().unwrap();
        let err = cmd_tx.send(InferCmd::Shutdown).unwrap_err();
        assert!(matches!(err, mpsc::SendError(InferCmd::Shutdown)));
    }

    // ── request defaults ─────────────────────────────────────────────────

    #[test]
    fn test_minimal_request_gets_default_temperature_and_max_tokens() {
        // Contract: a request without max_tokens/temperature is prepared
        // with temperature 0.1 and max_tokens 256. prepare_request applies
        // these via resolve_defaults after tokenizing; the defaults live
        // here, not in Request (the protocol.rs mirror test was removed).
        let req = Request::new("hello");
        let (max_gen, sampling) = resolve_defaults(10, &req, 512).unwrap();
        assert_eq!(sampling.temperature, 0.1);
        assert_eq!(max_gen, 256);
    }

    // ── resolve_sampling ─────────────────────────────────────────────────

    #[test]
    fn test_resolve_sampling_defaults_match_the_model_card() {
        let p = resolve_sampling(&Request::new("hi"));
        assert_eq!(p.temperature, DEFAULT_TEMPERATURE);
        assert_eq!(p.top_k, DEFAULT_TOP_K);
        assert_eq!(p.top_p, DEFAULT_TOP_P);
        assert_eq!(p.repeat_penalty, DEFAULT_REPEAT_PENALTY);
        assert_eq!(p.repeat_last_n, DEFAULT_REPEAT_LAST_N);
    }

    #[test]
    fn test_resolve_sampling_seed_defaults_to_random() {
        // Without this, every request shares one fixed seed and identical
        // prompts return identical text no matter the temperature.
        assert_eq!(resolve_sampling(&Request::new("hi")).seed, RANDOM_SEED);
    }

    #[test]
    fn test_resolve_sampling_honours_an_explicit_seed() {
        let req = Request::new("hi").with_seed(42);
        assert_eq!(resolve_sampling(&req).seed, 42);
    }

    #[test]
    fn test_resolve_sampling_passes_every_override_through() {
        let req = Request::new("hi")
            .with_temperature(0.8)
            .with_top_k(7)
            .with_top_p(0.3)
            .with_repeat_penalty(1.5)
            .with_repeat_last_n(64)
            .with_seed(9);
        let p = resolve_sampling(&req);
        assert_eq!(p.temperature, 0.8);
        assert_eq!(p.top_k, 7);
        assert_eq!(p.top_p, 0.3);
        assert_eq!(p.repeat_penalty, 1.5);
        assert_eq!(p.repeat_last_n, 64);
        assert_eq!(p.seed, 9);
    }

    #[test]
    fn test_resolve_sampling_clamps_out_of_range_values() {
        let req = Request::new("hi")
            .with_top_k(-5) // negative top-k is meaningless → 0 (disabled)
            .with_top_p(3.0) // above 1 → clamped to 1
            .with_repeat_penalty(99.0) // → clamped to 2
            .with_repeat_last_n(-9); // below -1 → -1 (whole context)
        let p = resolve_sampling(&req);
        assert_eq!(p.top_k, 0);
        assert_eq!(p.top_p, 1.0);
        assert_eq!(p.repeat_penalty, 2.0);
        assert_eq!(p.repeat_last_n, -1);
    }

    #[test]
    fn test_resolve_sampling_nan_falls_back_to_defaults() {
        let req = Request::new("hi")
            .with_top_p(f32::NAN)
            .with_repeat_penalty(f32::NAN);
        let p = resolve_sampling(&req);
        assert_eq!(p.top_p, DEFAULT_TOP_P);
        assert_eq!(p.repeat_penalty, DEFAULT_REPEAT_PENALTY);
    }

    #[test]
    fn test_resolve_sampling_zero_top_k_stays_zero() {
        // 0 is llama.cpp's "top-k disabled", not a value to clamp away.
        let req = Request::new("hi").with_top_k(0);
        assert_eq!(resolve_sampling(&req).top_k, 0);
    }

    // ── normalize_stop ───────────────────────────────────────────────────

    #[test]
    fn test_normalize_stop_keeps_real_sequences_in_order() {
        let stop = vec!["<|end|>".to_owned(), "STOP".to_owned()];
        assert_eq!(normalize_stop(&stop), stop);
    }

    #[test]
    fn test_normalize_stop_drops_empty_strings() {
        // An empty stop sequence matches at offset 0, which would end every
        // generation before its first token and return empty text.
        let stop = vec![String::new(), "END".to_owned(), String::new()];
        assert_eq!(normalize_stop(&stop), vec!["END".to_owned()]);
    }

    #[test]
    fn test_normalize_stop_empty_input_is_empty() {
        assert!(normalize_stop(&[]).is_empty());
    }

    // ── build_messages ───────────────────────────────────────────────────

    #[test]
    fn test_build_messages_system_and_history() {
        let req = Request::new("hello")
            .with_system("you are helpful")
            .push_history("assistant", "previous response");
        let msgs = build_messages(&req).unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(
            msgs[0],
            LlamaChatMessage::new("system".into(), "you are helpful".into()).unwrap()
        );
        assert_eq!(
            msgs[1],
            LlamaChatMessage::new("assistant".into(), "previous response".into()).unwrap()
        );
        assert_eq!(
            msgs[2],
            LlamaChatMessage::new("user".into(), "hello".into()).unwrap()
        );
    }

    #[test]
    fn test_build_messages_no_system() {
        let req = Request::new("hi");
        let msgs = build_messages(&req).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(
            msgs[0],
            LlamaChatMessage::new("user".into(), "hi".into()).unwrap()
        );
    }

    #[test]
    fn test_build_messages_empty_prompt() {
        let req = Request::new("");
        let msgs = build_messages(&req).unwrap();
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn test_build_messages_rejects_null_byte_in_role() {
        let req = Request::new("hi").push_history("user\x00admin", "hello");
        assert!(build_messages(&req).is_err());
    }

    #[test]
    fn test_build_messages_rejects_null_byte_in_content() {
        let req = Request::new("hi").with_system("sys\x00tem");
        assert!(build_messages(&req).is_err());
    }

    #[test]
    fn test_build_messages_many_history_entries() {
        let history: Vec<HistoryMessage> = (0..100)
            .map(|i| HistoryMessage {
                role: "user".into(),
                content: format!("message {i}"),
            })
            .collect();
        let req = Request::new("final").with_history(history);
        let msgs = build_messages(&req).unwrap();
        assert_eq!(msgs.len(), 101);
        // History lands in order, then the user prompt closes the list.
        for (i, m) in msgs[..100].iter().enumerate() {
            assert_eq!(
                *m,
                LlamaChatMessage::new("user".into(), format!("message {i}")).unwrap()
            );
        }
        assert_eq!(
            msgs[100],
            LlamaChatMessage::new("user".into(), "final".into()).unwrap()
        );
    }

    // ── temperature clamping ──────────────────────────────────────────────

    #[test]
    fn test_clamp_temperature_normal() {
        assert_eq!(clamp_temperature(0.7), 0.7);
        assert_eq!(clamp_temperature(0.0), 0.0);
        assert_eq!(clamp_temperature(2.0), 2.0);
    }

    #[test]
    fn test_clamp_temperature_negative_is_zero() {
        assert_eq!(clamp_temperature(-1.0), 0.0);
        assert_eq!(clamp_temperature(-0.5), 0.0);
    }

    #[test]
    fn test_clamp_temperature_nan_is_zero() {
        assert_eq!(clamp_temperature(f32::NAN), 0.0);
    }

    #[test]
    fn test_clamp_temperature_infinity_is_capped() {
        assert_eq!(clamp_temperature(f32::INFINITY), 2.0);
        assert_eq!(clamp_temperature(f32::NEG_INFINITY), 0.0);
    }

    #[test]
    fn test_clamp_temperature_above_two_is_capped() {
        assert_eq!(clamp_temperature(2.5), 2.0);
        assert_eq!(clamp_temperature(100.0), 2.0);
    }
}
