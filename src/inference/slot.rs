use llama_cpp_4::prelude::*;

use super::cache::{floor_char_boundary, longest_common_prefix};
use crate::protocol::{InferResult, LlamaError};

type Result<T> = std::result::Result<T, LlamaError>;

/// What a newly decoded fragment implies for the slot's lifetime.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Emit {
    /// Nothing special — keep generating.
    Continue,
    /// A stop sequence completed. `text` is already truncated at its start.
    Stopped,
    /// The streaming client hung up.
    Disconnected,
    /// The streaming buffer is full and the slow consumer was dropped.
    SlowConsumer,
}

/// Where a request's outcome is sent: `resp` for a non-streaming caller,
/// `token_tx`/`done_tx` for a streaming one.
#[derive(Default)]
pub(crate) struct Channels {
    pub(crate) resp: Option<tokio::sync::oneshot::Sender<Result<InferResult>>>,
    pub(crate) token_tx: Option<tokio::sync::mpsc::Sender<String>>,
    pub(crate) done_tx:
        Option<tokio::sync::oneshot::Sender<std::result::Result<InferResult, LlamaError>>>,
}

pub(crate) struct Slot {
    pub(crate) request_id: Option<String>,
    pub(crate) prompt_tokens: Vec<LlamaToken>,
    pub(crate) n_prompt: usize,
    pub(crate) gen_count: usize,
    pub(crate) max_tokens: u32,
    pub(crate) kv_pos: i32,
    pub(crate) sampler: LlamaSampler,
    pub(crate) utf8_buf: Vec<u8>,
    pub(crate) text: String,
    pub(crate) pending_token: Option<LlamaToken>,
    pub(crate) prompt_phase: bool,
    pub(crate) resp: Option<tokio::sync::oneshot::Sender<Result<InferResult>>>,
    pub(crate) token_tx: Option<tokio::sync::mpsc::Sender<String>>,
    pub(crate) done_tx:
        Option<tokio::sync::oneshot::Sender<std::result::Result<InferResult, LlamaError>>>,
    pub(crate) cache_tokens: Vec<LlamaToken>,
    pub(crate) stop: Vec<String>,
    pub(crate) stop_max_len: usize,
    pub(crate) emitted: usize,
}

impl Slot {
    pub(crate) fn push_token(
        &mut self,
        model: &LlamaModel,
        token: LlamaToken,
    ) -> Result<Option<String>> {
        let raw = model
            .token_to_raw_bytes(token, Special::Plaintext)
            .map_err(|e| LlamaError::Inference(e.to_string()))?;
        self.push_raw_bytes(&raw)
    }

    /// Buffer raw detokenized bytes and return any complete UTF-8 text prefix.
    pub(crate) fn push_raw_bytes(&mut self, raw: &[u8]) -> Result<Option<String>> {
        self.utf8_buf.extend_from_slice(raw);

        let valid = match std::str::from_utf8(&self.utf8_buf) {
            Ok(_) => self.utf8_buf.len(),
            Err(e) if e.error_len().is_some() => {
                return Err(LlamaError::Inference(format!(
                    "invalid utf-8 sequence: {e}"
                )));
            }
            Err(e) => e.valid_up_to(),
        };

        if valid == 0 {
            return Ok(None);
        }
        let complete: Vec<u8> = self.utf8_buf.drain(..valid).collect();
        String::from_utf8(complete)
            .map(Some)
            .map_err(|e| LlamaError::Inference(format!("internal: utf-8 prefix invalid: {e}")))
    }

    /// Byte offset of the earliest completed stop sequence, or `None`.
    pub(crate) fn find_stop(&self, piece_len: usize) -> Option<usize> {
        if self.stop.is_empty() {
            return None;
        }
        let window = piece_len + self.stop_max_len.saturating_sub(1);
        let from = floor_char_boundary(&self.text, self.text.len().saturating_sub(window));
        self.stop
            .iter()
            .filter_map(|s| self.text[from..].find(s.as_str()).map(|i| from + i))
            .min()
    }

    /// How many trailing bytes must be withheld from the stream because they
    /// could still grow into a stop sequence.
    pub(crate) fn holdback(&self) -> usize {
        if self.stop.is_empty() {
            return 0;
        }
        let max = self.stop_max_len.saturating_sub(1);
        let start = floor_char_boundary(&self.text, self.text.len().saturating_sub(max));
        for cut in start..self.text.len() {
            if !self.text.is_char_boundary(cut) {
                continue;
            }
            let tail = &self.text[cut..];
            if self.stop.iter().any(|s| s.starts_with(tail)) {
                return self.text.len() - cut;
            }
        }
        0
    }

    /// Stream `text[emitted..end]` to a streaming client, advancing `emitted`.
    pub(crate) fn flush_to(&mut self, end: usize) -> Emit {
        if end <= self.emitted {
            return Emit::Continue;
        }
        let chunk = self.text[self.emitted..end].to_owned();
        self.emitted = end;
        if let Some(ref tx) = self.token_tx {
            match tx.try_send(chunk) {
                Ok(()) => Emit::Continue,
                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => Emit::Disconnected,
                Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                    tracing::warn!("streaming token receiver full: slow consumer cancelled");
                    Emit::SlowConsumer
                }
            }
        } else {
            Emit::Continue
        }
    }

    /// Accumulate a generated fragment and release whatever is now safe to stream.
    pub(crate) fn push_text(&mut self, piece: &str) -> Emit {
        self.text.push_str(piece);

        if let Some(cut) = self.find_stop(piece.len()) {
            self.text.truncate(cut);
            return match self.flush_to(cut) {
                Emit::Continue => Emit::Stopped,
                other => other,
            };
        }

        let safe = self.text.len() - self.holdback();
        self.flush_to(safe)
    }

    pub(crate) fn finish(&mut self) -> Result<()> {
        if !self.utf8_buf.is_empty() {
            let err = LlamaError::Inference(format!(
                "incomplete UTF-8 sequence at end of generation ({} trailing bytes)",
                self.utf8_buf.len()
            ));
            self.cancel(err.clone());
            return Err(err);
        }
        let end = self.text.len();
        match self.flush_to(end) {
            Emit::Disconnected => return Err(LlamaError::Disconnected),
            Emit::SlowConsumer => {
                return Err(LlamaError::Inference(
                    "slow consumer: token channel buffer full".into(),
                ));
            }
            _ => {}
        }
        let result = InferResult {
            request_id: self.request_id.clone(),
            text: std::mem::take(&mut self.text),
            prompt_tokens: self.n_prompt,
            generated_tokens: self.gen_count,
        };
        if let Some(resp) = self.resp.take() {
            let _ = resp.send(Ok(result.clone()));
        }
        if let Some(done) = self.done_tx.take() {
            let _ = done.send(Ok(result));
        }
        drop(self.token_tx.take());
        Ok(())
    }

    pub(crate) fn cancel(&mut self, err: LlamaError) {
        if let Some(resp) = self.resp.take() {
            let _ = resp.send(Err(err.clone()));
        }
        if let Some(done) = self.done_tx.take() {
            let _ = done.send(Err(err));
        }
        drop(self.token_tx.take());
    }

    /// Whether the client has disconnected or dropped its response receiver.
    pub(crate) fn is_cancelled(&self) -> bool {
        if let Some(ref resp) = self.resp
            && resp.is_closed()
        {
            return true;
        }
        if let Some(ref token_tx) = self.token_tx
            && token_tx.is_closed()
        {
            return true;
        }
        false
    }

    /// Prepare the slot for a new prompt, reusing the cached KV prefix.
    pub(crate) fn begin_request(
        &mut self,
        tokens: Vec<LlamaToken>,
        cached: Vec<LlamaToken>,
    ) -> usize {
        self.prompt_tokens = tokens;
        self.n_prompt = self.prompt_tokens.len();
        self.kv_pos = 0;
        self.gen_count = 0;
        self.prompt_phase = true;
        self.pending_token = None;
        self.utf8_buf.clear();
        self.text.clear();

        let lcp = longest_common_prefix(&cached, &self.prompt_tokens);
        self.cache_tokens = if lcp > 0 {
            self.kv_pos = lcp as i32;
            let mut kept: Vec<LlamaToken> = cached;
            kept.truncate(lcp);
            kept
        } else {
            Vec::new()
        };
        lcp
    }

    /// Close the prefill phase: advance `kv_pos` past the prompt.
    pub(crate) fn finish_prefill(&mut self) {
        let reused = self.cache_tokens.len() as i32;
        debug_assert_eq!(self.kv_pos as usize, reused as usize);
        self.kv_pos += self.n_prompt as i32 - reused;
        self.prompt_phase = false;
        self.cache_tokens
            .extend(self.prompt_tokens.drain(reused as usize..));
        self.prompt_tokens.clear();
    }
}

pub(crate) fn create_slot(
    request_id: Option<String>,
    max_gen: u32,
    sampler: LlamaSampler,
    stop: Vec<String>,
    channels: Channels,
) -> Slot {
    let stop_max_len = stop.iter().map(String::len).max().unwrap_or(0);

    Slot {
        request_id,
        prompt_tokens: Vec::new(),
        n_prompt: 0,
        gen_count: 0,
        max_tokens: max_gen,
        kv_pos: 0,
        sampler,
        utf8_buf: Vec::new(),
        text: String::new(),
        pending_token: None,
        prompt_phase: false,
        resp: channels.resp,
        token_tx: channels.token_tx,
        done_tx: channels.done_tx,
        cache_tokens: Vec::new(),
        stop,
        stop_max_len,
        emitted: 0,
    }
}

pub(crate) fn reject(err: LlamaError, channels: Channels) {
    if let Some(resp) = channels.resp {
        let _ = resp.send(Err(err.clone()));
    }
    if let Some(done) = channels.done_tx {
        let _ = done.send(Err(err));
    }
    drop(channels.token_tx);
}
