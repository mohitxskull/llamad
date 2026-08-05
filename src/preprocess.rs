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
use crate::protocol::{InferCmd, LlamaError, PreparedCmd, Request};

pub(crate) fn clamp_temperature(raw: f32) -> f32 {
    if raw.is_nan() || raw.is_sign_negative() {
        0.0
    } else if raw > 2.0 {
        2.0
    } else {
        raw
    }
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
/// request defaults: `max_tokens` defaults to 256, `temperature` to 0.1
/// (then clamped). Extracted from [`prepare_request`] so the defaults
/// contract is unit-testable without a loaded model.
pub(crate) fn resolve_defaults(
    n_prompt: usize,
    request: &Request,
    per_slot_budget: u32,
) -> std::result::Result<(u32, f32), LlamaError> {
    let max_gen = compute_max_gen(n_prompt, request.max_tokens, per_slot_budget)?;
    let temperature = clamp_temperature(request.temperature.unwrap_or(0.1));
    Ok((max_gen, temperature))
}

/// Template, tokenize, and budget-check a request. Pure model reads only.
pub(crate) fn prepare_request(
    model: &LlamaModel,
    request: &Request,
    per_slot_budget: u32,
) -> std::result::Result<(Vec<LlamaToken>, u32, f32), LlamaError> {
    let messages = build_messages(request)?;
    let prompt = model
        .apply_chat_template(None, &messages, true)
        .map_err(|e| LlamaError::Inference(format!("chat template: {e}")))?;
    let tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|e| LlamaError::Inference(e.to_string()))?;
    let (max_gen, temperature) = resolve_defaults(tokens.len(), request, per_slot_budget)?;
    Ok((tokens, max_gen, temperature))
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
                Ok((tokens, max_gen, temperature)) => {
                    match prepared_tx.send(PreparedCmd::Run {
                        tokens,
                        max_gen,
                        temperature,
                        resp,
                    }) {
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
                Ok((tokens, max_gen, temperature)) => {
                    match prepared_tx.send(PreparedCmd::RunStream {
                        tokens,
                        max_gen,
                        temperature,
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
        let (max_gen, temperature) = resolve_defaults(10, &req, 512).unwrap();
        assert_eq!(temperature, 0.1);
        assert_eq!(max_gen, 256);
    }

    // ── build_messages ───────────────────────────────────────────────────

    #[test]
    fn test_build_messages_system_and_history() {
        let req = Request {
            prompt: "hello".into(),
            system: Some("you are helpful".into()),
            max_tokens: None,
            temperature: None,
            stream: None,
            history: vec![HistoryMessage {
                role: "assistant".into(),
                content: "previous response".into(),
            }],
        };
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
        let req = Request {
            prompt: "hi".into(),
            system: None,
            max_tokens: None,
            temperature: None,
            stream: None,
            history: vec![],
        };
        let msgs = build_messages(&req).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(
            msgs[0],
            LlamaChatMessage::new("user".into(), "hi".into()).unwrap()
        );
    }

    #[test]
    fn test_build_messages_empty_prompt() {
        let req = Request {
            prompt: String::new(),
            system: None,
            max_tokens: None,
            temperature: None,
            stream: None,
            history: vec![],
        };
        let msgs = build_messages(&req).unwrap();
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn test_build_messages_rejects_null_byte_in_role() {
        let req = Request {
            prompt: "hi".into(),
            system: None,
            max_tokens: None,
            temperature: None,
            stream: None,
            history: vec![HistoryMessage {
                role: "user\x00admin".into(),
                content: "hello".into(),
            }],
        };
        assert!(build_messages(&req).is_err());
    }

    #[test]
    fn test_build_messages_rejects_null_byte_in_content() {
        let req = Request {
            prompt: "hi".into(),
            system: Some("sys\x00tem".into()),
            max_tokens: None,
            temperature: None,
            stream: None,
            history: vec![],
        };
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
        let req = Request {
            prompt: "final".into(),
            system: None,
            max_tokens: None,
            temperature: None,
            stream: None,
            history,
        };
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
