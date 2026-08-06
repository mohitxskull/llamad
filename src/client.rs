//! The high-level API: [`Client`] owns a model and answers requests.

use std::path::Path;

use tokio::sync::{mpsc, oneshot};

use crate::config::InferenceConfig;
use crate::inference::Engine;
use crate::protocol::{InferCmd, InferResult, LlamaError, Request};

/// High-level client for a local GGUF model.
///
/// Owns an [`Engine`] — a loaded model plus its inference and preprocess
/// threads — and shuts it down on drop. Clients are independent: several can
/// be alive at once, each with its own model.
///
/// # Blocking and async
///
/// [`complete`](Self::complete), [`complete_text`](Self::complete_text) and
/// [`TokenStream::next_token`] block the calling thread. **They panic if
/// called from inside a tokio runtime** — that is tokio's own guard against
/// stalling a reactor thread. Inside async code use the `_async` variants
/// ([`complete_async`](Self::complete_async),
/// [`TokenStream::next_token_async`]), or move the blocking call to
/// `tokio::task::spawn_blocking`.
///
/// # Example
///
/// ```rust,no_run
/// # use llamad::client::Client;
/// # use llamad::protocol::Request;
/// # fn example() -> Result<(), llamad::protocol::LlamaError> {
/// let client = Client::new("/path/to/model.gguf")?;
///
/// let result = client.complete(Request::new("tell me a joke"))?;
/// println!("{}", result.text);
///
/// let result = client.complete(
///     Request::new("what is rust?")
///         .with_system("answer concisely")
///         .with_temperature(0.3),
/// )?;
///
/// let mut stream = client.complete_stream(Request::new("count to 3"))?;
/// while let Some(token) = stream.next_token() {
///     print!("{token}");
/// }
/// # Ok(())
/// # }
/// ```
///
/// The same thing from async code:
///
/// ```rust,no_run
/// # use llamad::client::Client;
/// # use llamad::protocol::Request;
/// # async fn example() -> Result<(), llamad::protocol::LlamaError> {
/// let client = Client::new("/path/to/model.gguf")?;
/// let result = client.complete_async(Request::new("tell me a joke")).await?;
///
/// let mut stream = client.complete_stream(Request::new("count to 3"))?;
/// while let Some(token) = stream.next_token_async().await {
///     print!("{token}");
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Client {
    engine: Engine,
}

impl Client {
    /// Load a model and return a client connected to it.
    ///
    /// Returns as soon as the threads are spawned; the model loads in the
    /// background. A bad path or unreadable GGUF surfaces as
    /// [`LlamaError::InferenceCrashed`] on the first request, not here.
    ///
    /// # Errors
    ///
    /// [`LlamaError::Io`] if the inference threads cannot be spawned.
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self, LlamaError> {
        Ok(Client {
            engine: Engine::start(model_path)?,
        })
    }

    /// Load a model with an explicit configuration, ignoring the `LLAMAD_*`
    /// environment variables.
    ///
    /// Use this when one process runs **more than one model**. Environment
    /// configuration is process-global, so [`new`](Self::new) gives every
    /// client the same context size and thread budget — rarely what you want
    /// when a small routing model and a large reasoning model share a machine.
    ///
    /// Values are clamped into usable ranges, so a hand-built config with a
    /// zero or absurd field degrades rather than panicking.
    ///
    /// # Errors
    ///
    /// [`LlamaError::Io`] if the inference threads cannot be spawned.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use llamad::{client::Client, config::InferenceConfig};
    /// # fn example() -> Result<(), llamad::protocol::LlamaError> {
    /// let router = Client::with_config("small.gguf", InferenceConfig {
    ///     n_ctx: 1024,
    ///     n_threads: 2,
    ///     ..Default::default()
    /// })?;
    /// let thinker = Client::with_config("big.gguf", InferenceConfig {
    ///     n_ctx: 8192,
    ///     n_threads: 4,
    ///     ..Default::default()
    /// })?;
    /// # let _ = (router, thinker);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_config(
        model_path: impl AsRef<Path>,
        config: InferenceConfig,
    ) -> Result<Self, LlamaError> {
        Ok(Client {
            engine: Engine::start_with_config(model_path, config)?,
        })
    }

    /// Run non-streaming inference, blocking until the response is complete.
    ///
    /// # Panics
    ///
    /// If called from inside a tokio runtime. Use
    /// [`complete_async`](Self::complete_async) there instead.
    ///
    /// # Errors
    ///
    /// Whatever ended the request — see [`LlamaError`].
    pub fn complete(&self, request: Request) -> Result<InferResult, LlamaError> {
        let resp_rx = self.submit(request)?;
        resp_rx
            .blocking_recv()
            .map_err(|_| LlamaError::InferenceCrashed)?
    }

    /// Run non-streaming inference without blocking the calling thread.
    ///
    /// # Errors
    ///
    /// Whatever ended the request — see [`LlamaError`].
    pub async fn complete_async(&self, request: Request) -> Result<InferResult, LlamaError> {
        let resp_rx = self.submit(request)?;
        resp_rx.await.map_err(|_| LlamaError::InferenceCrashed)?
    }

    /// Non-streaming completion from a prompt string.
    ///
    /// # Panics
    ///
    /// If called from inside a tokio runtime. Use
    /// [`complete_text_async`](Self::complete_text_async) there instead.
    ///
    /// # Errors
    ///
    /// Whatever ended the request — see [`LlamaError`].
    pub fn complete_text(&self, prompt: &str) -> Result<String, LlamaError> {
        self.complete(Request::new(prompt)).map(|r| r.text)
    }

    /// Non-streaming completion from a prompt string, without blocking.
    ///
    /// # Errors
    ///
    /// Whatever ended the request — see [`LlamaError`].
    pub async fn complete_text_async(&self, prompt: &str) -> Result<String, LlamaError> {
        self.complete_async(Request::new(prompt))
            .await
            .map(|r| r.text)
    }

    /// Start streaming inference.
    ///
    /// This call does not block in either context — only consuming the
    /// returned [`TokenStream`] does. Drop the stream to cancel generation.
    ///
    /// # Errors
    ///
    /// [`LlamaError::InferenceCrashed`] if the engine's threads have exited.
    pub fn complete_stream(&self, request: Request) -> Result<TokenStream, LlamaError> {
        let (token_tx, token_rx) = mpsc::unbounded_channel();
        let (done_tx, done_rx) = oneshot::channel();
        self.engine.send(InferCmd::RunStream {
            request,
            token_tx,
            done_tx: Some(done_tx),
        })?;
        Ok(TokenStream {
            rx: token_rx,
            done_rx,
            text: String::new(),
        })
    }

    /// The underlying engine, for direct [`InferCmd`] submission.
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Queue a non-streaming request and hand back its response channel.
    fn submit(
        &self,
        request: Request,
    ) -> Result<oneshot::Receiver<Result<InferResult, LlamaError>>, LlamaError> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.engine.send(InferCmd::Run {
            request,
            resp: resp_tx,
        })?;
        Ok(resp_rx)
    }
}

/// A stream that yields tokens as the model generates them.
///
/// Drop this early to cancel generation: the inference thread's next emit
/// fails, and it cancels the slot and moves on.
///
/// [`next_token`](Self::next_token) and [`into_result`](Self::into_result)
/// block and panic inside a tokio runtime; use
/// [`next_token_async`](Self::next_token_async) and
/// [`into_result_async`](Self::into_result_async) there.
#[derive(Debug)]
pub struct TokenStream {
    rx: mpsc::UnboundedReceiver<String>,
    /// Not optional: `complete_stream` is the only constructor and always
    /// requests a done-channel. `InferCmd::RunStream` allows `done_tx: None`
    /// for a caller driving the engine directly, but a `TokenStream` never
    /// takes that shape — so there is no "finished without stats" case to
    /// represent, and `into_result` needs no fallback for one.
    done_rx: oneshot::Receiver<Result<InferResult, LlamaError>>,
    text: String,
}

impl TokenStream {
    /// Accumulated text so far.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Consume the stream and return the text accumulated so far.
    #[must_use]
    pub fn finish(self) -> String {
        self.text
    }

    /// Block until the next token arrives, or return `None` once generation
    /// has ended.
    ///
    /// # Panics
    ///
    /// If called from inside a tokio runtime. Use
    /// [`next_token_async`](Self::next_token_async) there instead.
    pub fn next_token(&mut self) -> Option<String> {
        let token = self.rx.blocking_recv()?;
        self.text.push_str(&token);
        Some(token)
    }

    /// Await the next token, or return `None` once generation has ended.
    pub async fn next_token_async(&mut self) -> Option<String> {
        let token = self.rx.recv().await?;
        self.text.push_str(&token);
        Some(token)
    }

    /// Drain the stream and return the final [`InferResult`] with its stats.
    ///
    /// # Panics
    ///
    /// If called from inside a tokio runtime. Use
    /// [`into_result_async`](Self::into_result_async) there instead.
    ///
    /// # Errors
    ///
    /// Whatever ended the request — see [`LlamaError`].
    pub fn into_result(mut self) -> Result<InferResult, LlamaError> {
        while let Some(token) = self.rx.blocking_recv() {
            self.text.push_str(&token);
        }
        // `blocking_recv`, not `try_recv`: the done signal is sent before the
        // token channel closes today, but a `try_recv` would silently report
        // "not received" the moment that ordering changed.
        self.done_rx
            .blocking_recv()
            .map_err(|_| LlamaError::InferenceCrashed)?
    }

    /// Drain the stream and return the final [`InferResult`], without blocking
    /// the calling thread.
    ///
    /// # Errors
    ///
    /// Whatever ended the request — see [`LlamaError`].
    pub async fn into_result_async(mut self) -> Result<InferResult, LlamaError> {
        while let Some(token) = self.rx.recv().await {
            self.text.push_str(&token);
        }
        self.done_rx
            .await
            .map_err(|_| LlamaError::InferenceCrashed)?
    }
}

#[cfg(test)]
mod tests {
    // Panics are the assertion mechanism in tests: a failed `unwrap` here is a
    // reported failure, not a fault reachable from untrusted input. The
    // crate-level denies in `lib.rs` cover the library's production paths,
    // which is the surface that matters.
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::protocol::InferResult;
    use std::sync::mpsc;

    /// The done signal must survive the token channel closing first — the
    /// ordering `into_result` depends on. A `try_recv` here would race.
    #[test]
    fn test_into_result_gets_stats_when_tokens_close_first() {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        std::thread::spawn(move || {
            for cmd in cmd_rx {
                if let InferCmd::RunStream {
                    token_tx, done_tx, ..
                } = cmd
                {
                    let _ = token_tx.send("hi".to_string());
                    if let Some(done) = done_tx {
                        let _ = done.send(Ok(InferResult {
                            text: "hi".into(),
                            prompt_tokens: 7,
                            generated_tokens: 1,
                        }));
                    }
                    drop(token_tx);
                }
            }
        });
        let client = client_from_sender(cmd_tx);
        let result = client
            .complete_stream(Request::new("x"))
            .unwrap()
            .into_result()
            .unwrap();
        assert_eq!(result.prompt_tokens, 7);
        assert_eq!(result.generated_tokens, 1);
    }

    /// A dropped done-channel means the engine died mid-request, and must be
    /// reported as such rather than as a generic "no done signal".
    #[test]
    fn test_into_result_reports_crash_when_done_dropped() {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        std::thread::spawn(move || {
            for cmd in cmd_rx {
                if let InferCmd::RunStream {
                    token_tx, done_tx, ..
                } = cmd
                {
                    drop(done_tx);
                    drop(token_tx);
                }
            }
        });
        let client = client_from_sender(cmd_tx);
        let err = client
            .complete_stream(Request::new("x"))
            .unwrap()
            .into_result()
            .unwrap_err();
        assert!(matches!(err, LlamaError::InferenceCrashed));
    }

    #[tokio::test]
    async fn test_complete_async_works_inside_a_runtime() {
        // The sync `complete` panics here (tokio forbids blocking a reactor
        // thread); the async variant is the supported path.
        let client = mock_client();
        let result = client.complete_async(Request::new("hi")).await.unwrap();
        assert_eq!(result.text, "echo: hi");
    }

    #[tokio::test]
    async fn test_stream_async_collects_tokens_inside_a_runtime() {
        let client = mock_client();
        let mut stream = client.complete_stream(Request::new("abc")).unwrap();
        let mut tokens = Vec::new();
        while let Some(t) = stream.next_token_async().await {
            tokens.push(t);
        }
        assert_eq!(tokens, vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn test_complete_text_async_returns_the_generated_text() {
        // Found by mutation testing: replacing this method's body with
        // `Ok(String::new())` survived the whole suite. `complete_text` and
        // `complete_async` were both covered, so the gap was invisible by
        // inspection — it is exactly the combination neither test reached.
        let client = mock_client();
        let text = client.complete_text_async("hello").await.unwrap();
        assert_eq!(text, "echo: hello");
    }

    #[tokio::test]
    async fn test_into_result_async_returns_stats() {
        let client = mock_client();
        let result = client
            .complete_stream(Request::new("hi"))
            .unwrap()
            .into_result_async()
            .await
            .unwrap();
        assert_eq!(result.text, "echo: hi");
        assert_eq!(result.prompt_tokens, 3);
    }

    /// `Arc<Client>` shared across tasks is the natural way to serve
    /// concurrent requests from one model, so the auto traits that permit it
    /// are part of the public contract. `std::sync::mpsc::Sender` only became
    /// `Sync` in Rust 1.72 — a channel swap could silently take this away.
    #[test]
    fn test_client_is_send_and_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<Client>();
        assert_sync::<Client>();
        assert_send::<TokenStream>();
        assert_send::<crate::inference::Engine>();
        assert_sync::<crate::inference::Engine>();
    }

    /// A client whose "inference thread" echoes the prompt back, so the
    /// client API can be exercised without loading a model.
    fn mock_client() -> Client {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        std::thread::spawn(move || {
            for cmd in cmd_rx {
                match cmd {
                    InferCmd::Run { request, resp } => {
                        let result = InferResult {
                            text: format!("echo: {}", request.prompt),
                            prompt_tokens: 3,
                            generated_tokens: 5,
                        };
                        let _ = resp.send(Ok(result));
                    }
                    InferCmd::RunStream {
                        request,
                        token_tx,
                        done_tx,
                    } => {
                        for ch in request.prompt.chars() {
                            let _ = token_tx.send(ch.to_string());
                        }
                        if let Some(done) = done_tx {
                            let _ = done.send(Ok(InferResult {
                                text: format!("echo: {}", request.prompt),
                                prompt_tokens: 3,
                                generated_tokens: request.prompt.len(),
                            }));
                        }
                        drop(token_tx);
                    }
                    InferCmd::Shutdown => break,
                }
            }
        });
        client_from_sender(cmd_tx)
    }

    fn client_from_sender(cmd_tx: mpsc::Sender<InferCmd>) -> Client {
        Client {
            engine: Engine::from_sender(cmd_tx),
        }
    }

    #[test]
    fn test_complete_text() {
        let client = mock_client();
        let text = client.complete_text("hello").unwrap();
        assert_eq!(text, "echo: hello");
    }

    #[test]
    fn test_complete_with_options() {
        let client = mock_client();
        let req = Request::new("hi")
            .with_system("be nice")
            .with_temperature(0.5)
            .with_max_tokens(100);
        let result = client.complete(req).unwrap();
        assert_eq!(result.text, "echo: hi");
        assert_eq!(result.prompt_tokens, 3);
        assert_eq!(result.generated_tokens, 5);
    }

    #[test]
    fn test_complete_stream_collects_tokens() {
        let client = mock_client();
        let mut stream = client.complete_stream(Request::new("abc")).unwrap();
        let mut tokens = Vec::new();
        while let Some(t) = stream.next_token() {
            tokens.push(t);
        }
        assert_eq!(tokens, vec!["a", "b", "c"]);
        assert_eq!(stream.text(), "abc");
        assert_eq!(stream.finish(), "abc");
    }

    #[test]
    fn test_complete_stream_into_result_returns_stats() {
        let client = mock_client();
        let stream = client.complete_stream(Request::new("hi")).unwrap();
        let result = stream.into_result().unwrap();
        assert_eq!(result.text, "echo: hi");
        assert_eq!(result.prompt_tokens, 3);
        assert!(result.generated_tokens > 0);
    }

    #[test]
    fn test_complete_empty_prompt() {
        let client = mock_client();
        let text = client.complete_text("").unwrap();
        assert_eq!(text, "echo: ");
    }

    #[test]
    fn test_complete_inference_thread_crashed() {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        drop(cmd_rx);
        let client = client_from_sender(cmd_tx);
        let err = client.complete(Request::new("x")).unwrap_err();
        assert!(matches!(err, LlamaError::InferenceCrashed));
    }

    #[test]
    fn test_complete_stream_inference_thread_crashed() {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        drop(cmd_rx);
        let client = client_from_sender(cmd_tx);
        let err = client.complete_stream(Request::new("x")).unwrap_err();
        assert!(matches!(err, LlamaError::InferenceCrashed));
    }

    #[test]
    fn test_complete_stream_error_through_done_tx() {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        std::thread::spawn(move || {
            for cmd in cmd_rx {
                if let InferCmd::RunStream { done_tx, .. } = cmd {
                    if let Some(done) = done_tx {
                        let _ = done.send(Err(LlamaError::Inference("model error".into())));
                    }
                    break;
                }
            }
        });
        let client = client_from_sender(cmd_tx);
        let result = client
            .complete_stream(Request::new("x"))
            .unwrap()
            .into_result();
        let err = result.unwrap_err();
        assert!(matches!(err, LlamaError::Inference(_)));
    }

    #[test]
    fn test_request_builder_defaults() {
        let req = Request::new("hello")
            .with_system("helpful")
            .with_temperature(0.7)
            .with_max_tokens(50)
            .with_stream(true)
            .push_history("user", "previous question");
        assert_eq!(req.prompt, "hello");
        assert_eq!(req.system.as_deref(), Some("helpful"));
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.max_tokens, Some(50));
        assert_eq!(req.stream, Some(true));
        assert_eq!(req.history.len(), 1);
    }
}
