//! Wire and channel types: the [`Request`]/[`Response`] protocol spoken over
//! the Unix socket, the [`LlamaError`] taxonomy, and the [`InferCmd`] messages
//! the [`Client`](crate::client::Client) sends to the inference threads.

use llama_cpp_4::token::LlamaToken;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::oneshot;

/// Everything that can go wrong between submitting a request and receiving a
/// completion.
///
/// Implements [`Clone`] so a single failure can be reported to both the
/// response channel and the streaming done-channel of the same slot.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LlamaError {
    /// The GGUF file could not be opened or parsed.
    #[error("model load failed: {0}")]
    ModelLoad(String),
    /// The inference or preprocess thread exited before answering. Also
    /// returned when a model failed to load, since the thread exits in that
    /// case, and when an engine is used after shutdown.
    #[error("inference thread crashed")]
    InferenceCrashed,
    /// A runtime failure during generation: batch overflow, chat-template
    /// application, a prompt over the per-slot budget, a disconnected
    /// streaming client, or detokenization.
    #[error("inference failed: {0}")]
    Inference(String),
    /// Message construction rejected the input — a null byte or an otherwise
    /// invalid role/content pair.
    #[error("protocol error: {0}")]
    Protocol(String),
    /// Thread spawn or socket I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl Clone for LlamaError {
    fn clone(&self) -> Self {
        match self {
            LlamaError::ModelLoad(s) => LlamaError::ModelLoad(s.clone()),
            LlamaError::InferenceCrashed => LlamaError::InferenceCrashed,
            LlamaError::Inference(s) => LlamaError::Inference(s.clone()),
            LlamaError::Protocol(s) => LlamaError::Protocol(s.clone()),
            LlamaError::Io(e) => LlamaError::Io(std::io::Error::new(e.kind(), e.to_string())),
        }
    }
}

impl From<llama_cpp_4::LLamaCppError> for LlamaError {
    fn from(e: llama_cpp_4::LLamaCppError) -> Self {
        LlamaError::Inference(e.to_string())
    }
}

impl From<llama_cpp_4::NewLlamaChatMessageError> for LlamaError {
    fn from(e: llama_cpp_4::NewLlamaChatMessageError) -> Self {
        LlamaError::Protocol(e.to_string())
    }
}

/// One completion request. Unknown JSON fields are rejected.
///
/// `None` fields are omitted when serializing, so a round-trip through the
/// socket protocol reproduces the same request.
#[derive(Serialize, Deserialize, Default, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Request {
    /// The user turn. Required.
    pub prompt: String,
    /// Optional system prompt, prepended before `history`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Generation cap. Defaults to 256, further capped by the per-slot budget.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Sampling temperature, clamped to `[0, 2]`. Defaults to 0.1. Zero or
    /// below selects greedy decoding, which ignores every other sampling
    /// field below except `seed` (which greedy decoding does not consult).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Top-k cutoff. Clamped to `>= 0`; `0` disables it. Defaults to 50.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    /// Nucleus-sampling threshold, clamped to `[0, 1]`. Defaults to 0.95.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Repetition penalty, clamped to `[0, 2]`. `1.0` disables it. Defaults
    /// to 1.05.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    /// How many recent tokens the repetition penalty looks back over. `-1`
    /// (the default) means the whole context; `0` disables the lookback.
    /// Clamped to `>= -1`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_last_n: Option<i32>,
    /// RNG seed for sampling.
    ///
    /// Defaults to a **fresh random seed per request**, so two identical
    /// requests at a non-zero temperature produce different text. Set it
    /// explicitly for reproducible output. Ignored by greedy decoding
    /// (`temperature <= 0`), which is deterministic regardless.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    /// Strings that end generation as soon as one appears in the output.
    ///
    /// The matched stop text is **not** included in the returned completion,
    /// and is never streamed: output that could still grow into a stop
    /// sequence is held back until the match resolves. Empty strings are
    /// ignored. A stop sequence may span token boundaries.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    /// Stream tokens as NDJSON instead of returning one JSON object. Only
    /// meaningful over the socket protocol; the in-process client selects
    /// streaming by calling [`Client::complete_stream`](crate::client::Client::complete_stream).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Prior turns, applied between the system prompt and `prompt`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history: Vec<HistoryMessage>,
}

/// One prior conversation turn.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HistoryMessage {
    /// Chat role, e.g. `"user"` or `"assistant"`. Must not contain null bytes.
    pub role: String,
    /// Turn content. Must not contain null bytes.
    pub content: String,
}

/// The non-streaming socket response body.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Response {
    /// The generated text.
    pub text: String,
    /// Tokens in the templated prompt.
    pub prompt_tokens: usize,
    /// Tokens generated, excluding the end-of-generation token.
    pub generated_tokens: usize,
}

/// One NDJSON line of a streaming socket response.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct TokenChunk<'a> {
    /// The token text. Always a complete UTF-8 fragment.
    #[serde(borrow)]
    pub token: &'a str,
}

/// A finished completion with its token accounting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferResult {
    /// The generated text.
    pub text: String,
    /// Tokens in the templated prompt.
    pub prompt_tokens: usize,
    /// Tokens generated, excluding the end-of-generation token.
    pub generated_tokens: usize,
}

impl Request {
    /// A request with only a prompt; every other field takes its default.
    #[must_use]
    pub fn new(prompt: impl Into<String>) -> Self {
        Request {
            prompt: prompt.into(),
            ..Default::default()
        }
    }

    /// Set the system prompt.
    #[must_use]
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Set the sampling temperature. Clamped to `[0, 2]` at preparation time;
    /// zero or below selects greedy decoding.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the generation cap. Further capped by the per-slot token budget.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set the top-k cutoff. `0` disables it.
    #[must_use]
    pub fn with_top_k(mut self, top_k: i32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set the nucleus-sampling threshold. Clamped to `[0, 1]`.
    #[must_use]
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set the repetition penalty. `1.0` disables it.
    #[must_use]
    pub fn with_repeat_penalty(mut self, penalty: f32) -> Self {
        self.repeat_penalty = Some(penalty);
        self
    }

    /// Set how many recent tokens the repetition penalty considers. `-1` is
    /// the whole context, `0` disables the lookback.
    #[must_use]
    pub fn with_repeat_last_n(mut self, last_n: i32) -> Self {
        self.repeat_last_n = Some(last_n);
        self
    }

    /// Pin the sampling seed for reproducible output. Without this each
    /// request draws a fresh random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Replace the stop sequences. Generation ends as soon as one appears,
    /// and the matched text is excluded from the completion.
    #[must_use]
    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.stop = stop;
        self
    }

    /// Append one stop sequence.
    #[must_use]
    pub fn push_stop(mut self, stop: impl Into<String>) -> Self {
        self.stop.push(stop.into());
        self
    }

    /// Set the `stream` flag of the socket protocol.
    #[must_use]
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Replace the conversation history.
    #[must_use]
    pub fn with_history(mut self, history: Vec<HistoryMessage>) -> Self {
        self.history = history;
        self
    }

    /// Append one turn to the conversation history.
    #[must_use]
    pub fn push_history(mut self, role: impl Into<String>, content: impl Into<String>) -> Self {
        self.history.push(HistoryMessage {
            role: role.into(),
            content: content.into(),
        });
        self
    }
}

/// A command for the inference threads.
///
/// Most users should go through [`Client`](crate::client::Client) instead;
/// this is the lower-level interface the socket server uses so it can await
/// results inside an async runtime.
#[non_exhaustive]
pub enum InferCmd {
    /// Generate a full completion and send it on `resp`.
    Run {
        /// The request to prepare and run.
        request: Request,
        /// Receives the completion, or the error that ended it.
        resp: oneshot::Sender<Result<InferResult, LlamaError>>,
    },
    /// Generate a completion, streaming each token on `token_tx`.
    ///
    /// Dropping the receiving end of `token_tx` cancels generation and frees
    /// the slot at the next emitted token.
    RunStream {
        /// The request to prepare and run.
        request: Request,
        /// Receives each complete UTF-8 fragment as it is generated.
        token_tx: tokio::sync::mpsc::UnboundedSender<String>,
        /// Receives the final result once generation ends. Sent before
        /// `token_tx` is dropped, so a closed token channel implies the done
        /// signal is already in flight.
        done_tx: Option<tokio::sync::oneshot::Sender<Result<InferResult, LlamaError>>>,
    },
    /// Cancel every active slot and stop both threads.
    Shutdown,
}

/// llama.cpp's "draw a fresh random seed" sentinel — `LLAMA_DEFAULT_SEED` in
/// `llama.h`. `llama_sampler_init_dist` maps it through `get_rng_seed`, which
/// pulls from `std::random_device` (or the system clock when that is not a
/// true RNG). Defined here rather than taken from the bindgen output so the
/// value does not depend on how the sys crate happens to name its constants.
pub(crate) const RANDOM_SEED: u32 = 0xFFFF_FFFF;

/// Sampling parameters resolved from a [`Request`]: every `None` replaced by
/// its default and every value clamped into a usable range.
///
/// Separate from `Request` so the inference thread receives values it can
/// hand to llama.cpp unchecked — validation happens once, on the preprocess
/// thread.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SamplingParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,
    pub seed: u32,
}

/// A fully-prepared inference command: the prompt is already templated,
/// tokenized, and budget-checked, and the sampling parameters are resolved
/// and clamped. Sent from the preprocess thread to the inference thread.
pub(crate) enum PreparedCmd {
    Run {
        tokens: Vec<LlamaToken>,
        max_gen: u32,
        sampling: SamplingParams,
        stop: Vec<String>,
        resp: oneshot::Sender<Result<InferResult, LlamaError>>,
    },
    RunStream {
        tokens: Vec<LlamaToken>,
        max_gen: u32,
        sampling: SamplingParams,
        stop: Vec<String>,
        token_tx: tokio::sync::mpsc::UnboundedSender<String>,
        done_tx: Option<tokio::sync::oneshot::Sender<Result<InferResult, LlamaError>>>,
    },
    Shutdown,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── deserialization ──────────────────────────────────────────────────

    #[test]
    fn test_deserialize_request_full() {
        let json = r#"{"prompt":"hello","system":"be nice","max_tokens":100,"temperature":0.5}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "hello");
        assert_eq!(req.system.as_deref(), Some("be nice"));
        assert_eq!(req.max_tokens, Some(100));
        assert_eq!(req.temperature, Some(0.5));
        assert!(!req.stream.unwrap_or(false));
    }

    #[test]
    fn test_deserialize_request_minimal() {
        let json = r#"{"prompt":"hi"}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "hi");
        assert!(req.system.is_none());
        assert!(req.max_tokens.is_none());
    }

    #[test]
    fn test_deserialize_request_with_history() {
        let json =
            r#"{"prompt":"what was my question?","history":[{"role":"user","content":"hello"}]}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        assert_eq!(req.history.len(), 1);
        assert_eq!(req.history[0].role, "user");
        assert_eq!(req.history[0].content, "hello");
    }

    #[test]
    fn test_deserialize_request_empty_history() {
        let json = r#"{"prompt":"hi","history":[]}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        assert!(req.history.is_empty());
    }

    #[test]
    fn test_deserialize_request_unknown_field_rejected() {
        let json = r#"{"prompt":"hi","unknown_field":"value"}"#;
        let result = serde_json::from_str::<Request>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_request_missing_prompt_rejected() {
        let json = r#"{"temperature":0.5}"#;
        let result = serde_json::from_str::<Request>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_request_bad_json_rejected() {
        let json = r#"{"prompt":"hi"trailing"#;
        let result = serde_json::from_str::<Request>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_request_wrong_type_for_prompt() {
        let json = r#"{"prompt":42}"#;
        let result = serde_json::from_str::<Request>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_request_very_long_strings() {
        let long_role = "a".repeat(65536);
        let long_content = "b".repeat(65536);
        let json = format!(
            r#"{{"prompt":"test","history":[{{"role":"{long_role}","content":"{long_content}"}}]}}"#
        );
        let req: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(req.history[0].role.len(), 65536);
        assert_eq!(req.history[0].content.len(), 65536);
    }

    // ── serialization ────────────────────────────────────────────────────

    #[test]
    fn test_serialize_response() {
        let resp = Response {
            text: "2 + 2 = 4".into(),
            prompt_tokens: 10,
            generated_tokens: 5,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"text\":\"2 + 2 = 4\""));
        assert!(json.contains("\"prompt_tokens\":10"));
        assert!(json.contains("\"generated_tokens\":5"));
    }

    #[test]
    fn test_request_round_trips_through_json() {
        // A Rust consumer of the socket protocol builds a Request, serializes
        // it, and the daemon deserializes it back. `deny_unknown_fields` makes
        // this fail loudly if a `None` field ever serializes as an explicit
        // null instead of being skipped.
        let req = Request::new("hi")
            .with_system("be nice")
            .with_max_tokens(20)
            .push_history("user", "earlier");
        let json = serde_json::to_string(&req).unwrap();
        assert!(
            !json.contains("null"),
            "None fields must be skipped: {json}"
        );
        assert!(!json.contains("temperature"));
        assert_eq!(serde_json::from_str::<Request>(&json).unwrap(), req);
    }

    #[test]
    fn test_minimal_request_serializes_to_prompt_only() {
        let json = serde_json::to_string(&Request::new("hi")).unwrap();
        assert_eq!(json, r#"{"prompt":"hi"}"#);
    }

    #[test]
    fn test_response_deserializes() {
        // Consumers of the socket protocol parse the daemon's reply with the
        // crate's own type rather than hand-rolling a struct.
        let resp: Response =
            serde_json::from_str(r#"{"text":"hi","prompt_tokens":3,"generated_tokens":1}"#)
                .unwrap();
        assert_eq!(resp.text, "hi");
        assert_eq!(resp.prompt_tokens, 3);
    }

    #[test]
    fn test_token_chunk_deserializes() {
        let chunk: TokenChunk = serde_json::from_str(r#"{"token":"ab"}"#).unwrap();
        assert_eq!(chunk.token, "ab");
    }

    #[test]
    fn test_deserialize_history_message_unknown_field_rejected() {
        let json = r#"{"role":"user","content":"hi","extra":"bad"}"#;
        let result = serde_json::from_str::<HistoryMessage>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_stream_mode_defaults_to_false() {
        let json = r#"{"prompt":"test"}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        assert!(!req.stream.unwrap_or(false));
    }

    #[test]
    fn test_request_stream_true() {
        let json = r#"{"prompt":"test","stream":true}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        assert!(req.stream.unwrap_or(false));
    }
}
