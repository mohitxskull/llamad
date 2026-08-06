//! Local IPC server speaking the JSON (and NDJSON, when streaming) protocol.
//!
//! One request per connection. The client writes a JSON object and ends the
//! request with **either a newline or a half-close**, then reads the reply.
//!
//! The transport is a Unix domain socket on Unix and a named pipe on Windows;
//! the protocol above is identical on both. The handlers are generic over the
//! stream, so only the listener differs by platform: `bind_socket` on Unix,
//! `bind_pipe` on Windows.
//!
//! Those two are deliberately *not* intra-doc links — each exists on only one
//! platform, so linking either one breaks `cargo doc` on the other.
//!
//! # Why a newline terminates a request
//!
//! The original protocol ended a request at EOF, which requires the client to
//! half-close its write side while keeping the read side open. Windows named
//! pipes have no half-close — closing the handle closes both directions — so
//! an EOF-only frame would deadlock there: the server would wait forever for a
//! request it had already received in full.
//!
//! A newline terminator fixes that without breaking anything, because
//! `serde_json` escapes newlines inside strings and never emits a raw one. A
//! client that half-closes still works: EOF ends the frame too.
//!
//! Replies are framed the same way, for the same reason: every line the daemon
//! writes — the non-streaming response, each streamed token, and every error —
//! ends with a newline. A client can therefore parse a reply as soon as it
//! arrives instead of reading to EOF, which on a named pipe means waiting for
//! the server to drop the connection. Trailing whitespace is insignificant to
//! JSON, so a client that does read to EOF is unaffected.

use std::sync::mpsc;

use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::oneshot;

use crate::protocol::{
    ErrorResponse, InferCmd, LlamaError, Request, Response, StreamDone, TokenChunk,
};

#[cfg(unix)]
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
#[cfg(unix)]
use std::path::Path;
#[cfg(unix)]
use tokio::net::UnixListener;

/// Socket path used when none is given.
#[cfg(unix)]
pub const DEFAULT_SOCKET_PATH: &str = "/tmp/llamad.sock";
/// Pipe name used when none is given.
///
/// Windows named pipes live in a flat kernel namespace under `\\.\pipe\`, not
/// on the filesystem, so there is no directory to place this in and nothing to
/// clean up after a crash — the name is released when the last handle closes.
#[cfg(windows)]
pub const DEFAULT_SOCKET_PATH: &str = r"\\.\pipe\llamad";
/// How long a client may take to send its whole request.
pub const READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
/// How long a streaming client may wait between tokens before giving up.
pub const STREAM_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
/// Largest request body accepted, in bytes.
///
/// Without a cap, one local client can drive the daemon out of memory by
/// writing forever — the read below only ends at EOF. A megabyte is far above
/// any prompt that fits the per-slot token budget.
pub const MAX_REQUEST_BYTES: u64 = 1024 * 1024;

/// Bind the listener at `path`, replacing a stale socket left by a crash.
///
/// The socket is chmod'ed to `0600` so only its owner can submit inference
/// requests. The default path lives in a world-writable directory, so a
/// permissive mode would let any local user drive the model.
///
/// # Errors
///
/// [`LlamaError::Io`] if the path is taken by a live daemon, is occupied by
/// something other than a socket, or cannot be bound.
#[cfg(unix)]
pub fn bind_socket(path: impl AsRef<Path>) -> Result<UnixListener, LlamaError> {
    let path = path.as_ref();
    let listener = match UnixListener::bind(path) {
        Ok(listener) => listener,
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            // Only reclaim a socket nobody is listening on. A successful
            // connect means a live daemon owns this path, and unlinking it
            // would silently steal its clients.
            if std::os::unix::net::UnixStream::connect(path).is_ok() {
                return Err(LlamaError::Io(std::io::Error::new(
                    std::io::ErrorKind::AddrInUse,
                    format!("{} is in use by a running daemon", path.display()),
                )));
            }
            // Refuse to unlink anything that is not a socket — the default
            // path is in /tmp, where a regular file or symlink at this name
            // is more likely an attack than a leftover.
            let meta = std::fs::symlink_metadata(path)?;
            if !meta.file_type().is_socket() {
                return Err(LlamaError::Io(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    format!("{} exists and is not a socket", path.display()),
                )));
            }
            std::fs::remove_file(path)?;
            UnixListener::bind(path)?
        }
        Err(e) => return Err(e.into()),
    };
    // Narrow the mode as soon as the socket exists. The window between bind
    // and chmod is unavoidable without a per-process umask change, which is
    // not thread-safe; put the socket in a 0700 directory if that matters.
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    Ok(listener)
}

/// Create a named-pipe server instance at `name` (e.g. `\\.\pipe\llamad`).
///
/// Call once for the first instance with `first` set, then again after each
/// accepted connection to have an instance waiting for the next client —
/// unlike a Unix listener, a named-pipe server handle *is* the connection once
/// a client attaches, so the accept loop must create its successor.
///
/// `first_pipe_instance` on the first call makes creation fail if the name is
/// already taken, which is the closest equivalent to refusing to steal a live
/// daemon's socket. There is no stale pipe to reclaim: the name disappears
/// when the last handle closes, including when the process is killed.
///
/// # Access control
///
/// `reject_remote_clients` is set, so the pipe cannot be reached over SMB from
/// another machine — without it, a named pipe is remotely accessible by
/// default, which a Unix socket never is.
///
/// Local access falls back to the default DACL for the creating process,
/// which grants the creating user, `SYSTEM` and `Administrators`. That is
/// broadly comparable to the `0600` the Unix path sets, but it is *not* the
/// same guarantee — an administrator on the machine can open the pipe. If you
/// need stricter local control, run the daemon as a dedicated user.
///
/// # Errors
///
/// [`LlamaError::Io`] if the name is taken by a running daemon, or the pipe
/// cannot be created.
#[cfg(windows)]
pub fn bind_pipe(
    name: &str,
    first: bool,
) -> Result<tokio::net::windows::named_pipe::NamedPipeServer, LlamaError> {
    tokio::net::windows::named_pipe::ServerOptions::new()
        .first_pipe_instance(first)
        .reject_remote_clients(true)
        .create(name)
        .map_err(LlamaError::Io)
}

/// Read one request frame: everything up to the first newline, or to EOF.
///
/// Returns the body, or `None` if the client exceeded [`MAX_REQUEST_BYTES`].
/// The cap is read one byte past the limit so an over-long body is detected
/// rather than silently truncated into a confusing parse error.
async fn read_frame<S: AsyncRead + Unpin>(stream: &mut S) -> std::io::Result<Option<Vec<u8>>> {
    let mut buf = Vec::new();
    let mut reader = BufReader::new(stream.take(MAX_REQUEST_BYTES + 1));
    reader.read_until(b'\n', &mut buf).await?;
    if buf.len() as u64 > MAX_REQUEST_BYTES {
        return Ok(None);
    }
    // A trailing newline is a frame terminator, not part of the JSON.
    if buf.last() == Some(&b'\n') {
        buf.pop();
    }
    Ok(Some(buf))
}

/// Write one newline-terminated reply line.
///
/// Every reply the daemon writes — success, error, and each streamed token —
/// ends with a newline, mirroring the request framing. The terminator is what
/// lets a client parse a reply without waiting for the connection to close,
/// which matters most on Windows named pipes: they have no half-close, so
/// "read until EOF" is the one framing rule that transport cannot express
/// cheaply. Returns `false` once the client has hung up.
async fn write_line<S: AsyncWrite + Unpin>(stream: &mut S, line: &str) -> bool {
    stream.write_all(line.as_bytes()).await.is_ok() && stream.write_all(b"\n").await.is_ok()
}

/// Write an `{"error": ...}` reply. Failures are ignored: the connection is
/// being abandoned either way, and there is nobody left to report to.
async fn reply_error<S: AsyncWrite + Unpin>(stream: &mut S, message: &str) {
    let err = ErrorResponse {
        error: message.to_owned(),
    };
    if let Ok(json) = serde_json::to_string(&err) {
        let _ = write_line(stream, &json).await;
    }
}

/// Read one request from `stream`, run it, and write the reply.
///
/// Generic over the stream so the same protocol serves a Unix socket and a
/// Windows named pipe; only the listener differs by platform.
pub async fn handle_connection<S: AsyncRead + AsyncWrite + Unpin>(
    stream: &mut S,
    cmd_tx: &mpsc::Sender<InferCmd>,
) {
    let buf = match tokio::time::timeout(READ_TIMEOUT, read_frame(stream)).await {
        Ok(Ok(Some(buf))) => buf,
        Ok(Ok(None)) => {
            // Discard whatever the client is still writing before replying.
            // Closing a socket that has unread data queued makes the kernel
            // send RST, which throws away our error message along with it —
            // the client would see a bare connection reset instead of the
            // reason. Reading into a fixed scratch buffer keeps memory
            // bounded, and READ_TIMEOUT bounds how long a client can hold the
            // connection open.
            let mut sink = [0u8; 8192];
            let _ = tokio::time::timeout(READ_TIMEOUT, async {
                while matches!(stream.read(&mut sink).await, Ok(n) if n > 0) {}
            })
            .await;

            reply_error(
                stream,
                &format!("request too large (max {MAX_REQUEST_BYTES} bytes)"),
            )
            .await;
            return;
        }
        Ok(Err(_)) => return, // connection died mid-request; nobody to tell
        Err(_) => {
            reply_error(stream, "read timeout").await;
            return;
        }
    };

    if buf.is_empty() {
        reply_error(stream, "empty request").await;
        return;
    }

    let req: Request = match serde_json::from_slice(&buf) {
        Ok(r) => r,
        Err(e) => {
            reply_error(stream, &format!("invalid request: {e}")).await;
            return;
        }
    };

    let stream_mode = req.stream.unwrap_or(false);

    if stream_mode {
        handle_streaming(stream, cmd_tx, req).await;
    } else {
        handle_non_streaming(stream, cmd_tx, req).await;
    }
}

async fn handle_streaming<S: AsyncRead + AsyncWrite + Unpin>(
    stream: &mut S,
    cmd_tx: &mpsc::Sender<InferCmd>,
    req: Request,
) {
    let (token_tx, mut token_rx) = tokio::sync::mpsc::unbounded_channel();
    let (done_tx, done_rx) = oneshot::channel();

    if cmd_tx
        .send(InferCmd::RunStream {
            request: req,
            token_tx,
            done_tx: Some(done_tx),
        })
        .is_err()
    {
        reply_error(stream, "inference thread unavailable").await;
        return;
    }

    loop {
        match tokio::time::timeout(STREAM_TIMEOUT, token_rx.recv()).await {
            Ok(Some(token)) => {
                if let Ok(json) = serde_json::to_string(&TokenChunk { token: &token })
                    && !write_line(stream, &json).await
                {
                    return; // client hung up mid-stream
                }
            }
            Ok(None) => break,
            Err(_) => {
                reply_error(stream, "stream timeout").await;
                return;
            }
        }
    }

    match done_rx.await {
        Ok(Ok(result)) => {
            let done = StreamDone::from(Response::from(result));
            if let Ok(json) = serde_json::to_string(&done) {
                let _ = write_line(stream, &json).await;
            }
        }
        Ok(Err(e)) => {
            reply_error(stream, &format!("inference: {e}")).await;
        }
        Err(_) => {
            reply_error(stream, "inference stream ended unexpectedly").await;
        }
    }
}

async fn handle_non_streaming<S: AsyncRead + AsyncWrite + Unpin>(
    stream: &mut S,
    cmd_tx: &mpsc::Sender<InferCmd>,
    req: Request,
) {
    let (resp_tx, resp_rx) = oneshot::channel();

    if cmd_tx
        .send(InferCmd::Run {
            request: req,
            resp: resp_tx,
        })
        .is_err()
    {
        reply_error(stream, "inference thread unavailable").await;
        return;
    }

    match resp_rx.await {
        Ok(Ok(result)) => {
            if let Ok(json) = serde_json::to_string(&Response::from(result)) {
                let _ = write_line(stream, &json).await;
            }
        }
        Ok(Err(e)) => {
            reply_error(stream, &format!("inference: {e}")).await;
        }
        Err(_) => {
            reply_error(stream, "inference thread crashed").await;
        }
    }
}

// Test helpers with no transport in them, shared by the Unix and Windows
// suites below. Hoisted rather than duplicated per platform: the mock engine's
// echo behaviour is what both suites assert against, so a change to one that
// missed the other would silently weaken the transport it was not updated for.
// This module compiles on every platform, so Linux CI type-checks the copy the
// Windows suite uses.
#[cfg(test)]
mod test_support {
    use super::*;
    use std::thread;

    use crate::protocol::InferResult;

    /// An "inference thread" that echoes the prompt back, so the protocol can
    /// be exercised without loading a model.
    pub(super) fn spawn_mock_inference() -> mpsc::Sender<InferCmd> {
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            for cmd in rx {
                match cmd {
                    InferCmd::Run { request, resp } => {
                        let _ = resp.send(Ok(InferResult {
                            text: format!("echo: {}", request.prompt),
                            prompt_tokens: 5,
                            generated_tokens: 2,
                        }));
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
                                prompt_tokens: 5,
                                generated_tokens: request.prompt.len(),
                            }));
                        }
                        drop(token_tx);
                    }
                    InferCmd::Shutdown => break,
                }
            }
        });
        tx
    }

    /// A sender whose receiver is already dropped, so every send fails — the
    /// "inference thread is gone" condition.
    pub(super) fn dead_sender() -> mpsc::Sender<InferCmd> {
        let (tx, rx) = mpsc::channel();
        drop(rx);
        tx
    }
}

// The existing suite drives a Unix domain socket directly. The Windows
// transport has its own module below; the protocol assertions are duplicated
// rather than shared because the listener types have no common trait to
// abstract over — a named-pipe server handle becomes the connection, a Unix
// listener yields one.
#[cfg(all(test, unix))]
mod tests {
    use super::test_support::{dead_sender, spawn_mock_inference};
    use super::*;
    use std::path::PathBuf;
    use std::thread;

    /// A socket path unique to this process and test, in a temp directory that
    /// already exists. The `tag` keeps parallel tests in one binary from
    /// colliding on a name.
    fn socket_path(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("llamad-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join(format!("{tag}.sock"));
        let _ = std::fs::remove_file(&path);
        path
    }

    /// Serve exactly one connection at a fresh socket path, run `client_body`
    /// against it, and clean up.
    ///
    /// No sleep before connecting: `bind_socket` returns a *listening* socket,
    /// so the kernel queues a connect that arrives before the accept task is
    /// scheduled. The 100 ms sleeps this replaces were not synchronizing
    /// anything — they only made the suite a second slower.
    async fn with_server<F, Fut>(tag: &str, cmd_tx: mpsc::Sender<InferCmd>, client_body: F)
    where
        F: FnOnce(tokio::net::UnixStream) -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        let path = socket_path(tag);
        let listener = bind_socket(&path).expect("bind test socket");
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept");
            handle_connection(&mut stream, &cmd_tx).await;
        });

        let client = tokio::net::UnixStream::connect(&path)
            .await
            .expect("connect to test socket");
        client_body(client).await;

        server.await.expect("server task");
        let _ = std::fs::remove_file(&path);
    }

    /// Send `request` (newline-framed), read the whole reply, and return it.
    async fn round_trip(mut client: tokio::net::UnixStream, request: &str) -> String {
        client
            .write_all(format!("{request}\n").as_bytes())
            .await
            .expect("write request");
        let mut buf = Vec::new();
        client.read_to_end(&mut buf).await.expect("read reply");
        String::from_utf8(buf).expect("reply is utf-8")
    }

    /// The single JSON object a non-streaming reply consists of.
    async fn round_trip_json(client: tokio::net::UnixStream, request: &str) -> serde_json::Value {
        let raw = round_trip(client, request).await;
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("reply {raw:?} is not JSON: {e}"))
    }

    // ── bind_socket ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_bind_socket_creates_listener() {
        let path = socket_path("creates");
        let listener = bind_socket(&path).unwrap();
        assert!(path.exists());
        drop(listener);
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn test_bind_socket_reclaims_stale_socket() {
        // A crashed daemon leaves a socket file with nobody listening. That
        // one is safe to unlink and rebind.
        let path = socket_path("stale");
        let dead = UnixListener::bind(&path).unwrap();
        drop(dead); // tokio does not unlink on drop — the file survives
        assert!(path.exists(), "stale socket file should remain");

        let listener = bind_socket(&path).unwrap();
        assert!(path.exists());
        drop(listener);
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn test_bind_socket_refuses_to_unlink_a_regular_file() {
        // The default socket path is in /tmp. Blindly unlinking whatever sits
        // at that name lets any local user get the daemon to delete a file.
        let path = socket_path("not-a-socket");
        std::fs::write(&path, b"important").unwrap();

        let err = bind_socket(&path).unwrap_err();
        assert!(
            err.to_string().contains("not a socket"),
            "expected a not-a-socket refusal, got: {err}"
        );
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"important",
            "the file must not have been touched"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn test_bind_socket_refuses_a_live_daemons_path() {
        // Rebinding over a socket someone is still listening on would
        // silently hijack that daemon's incoming connections.
        let path = socket_path("live");
        let live = bind_socket(&path).unwrap();
        let err = bind_socket(&path).unwrap_err();
        assert!(
            err.to_string().contains("running daemon"),
            "expected an in-use refusal, got: {err}"
        );
        drop(live);
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn test_bind_socket_is_owner_only() {
        // The socket accepts inference requests; in a world-writable
        // directory a permissive mode hands the model to any local user.
        let path = socket_path("perms");
        let listener = bind_socket(&path).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600, "socket mode was {:o}", mode & 0o777);
        drop(listener);
        let _ = std::fs::remove_file(&path);
    }

    // ── request framing ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_request_over_size_cap_is_rejected() {
        with_server("too-big", dead_sender(), |mut client| async move {
            let oversized = format!(
                r#"{{"prompt":"{}"}}"#,
                "a".repeat(MAX_REQUEST_BYTES as usize + 16)
            );
            // The server stops reading at the cap, so the client's write may
            // fail once the receive buffer fills — that is the intended
            // backpressure, and the reply still arrives.
            let _ = client.write_all(oversized.as_bytes()).await;
            let _ = client.shutdown().await;

            let mut buf = Vec::new();
            client.read_to_end(&mut buf).await.unwrap();
            let resp: serde_json::Value = serde_json::from_slice(&buf).unwrap();
            assert!(
                resp["error"].as_str().unwrap().contains("too large"),
                "expected a size-cap rejection, got: {resp}"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn test_half_closed_request_is_framed_by_eof() {
        // The original framing, still supported: a client that half-closes its
        // write side ends the request at EOF, with no trailing newline.
        with_server(
            "half-close",
            spawn_mock_inference(),
            |mut client| async move {
                let req = serde_json::json!({"prompt": "hello world", "max_tokens": 10});
                client.write_all(req.to_string().as_bytes()).await.unwrap();
                client.shutdown().await.unwrap();

                let mut buf = Vec::new();
                client.read_to_end(&mut buf).await.unwrap();
                let resp: serde_json::Value = serde_json::from_slice(&buf).unwrap();
                assert_eq!(resp["text"], "echo: hello world");
            },
        )
        .await;
    }

    #[tokio::test]
    async fn test_newline_terminated_request_without_half_close() {
        // The mechanism Windows depends on. A named-pipe client cannot
        // half-close, so the request has to be able to end at a newline while
        // the connection stays open in both directions. Exercised here on a
        // Unix socket because the framing is transport-independent — if this
        // breaks, the Windows daemon deadlocks.
        with_server(
            "newline-frame",
            spawn_mock_inference(),
            |client| async move {
                let req = serde_json::json!({"prompt": "hello world", "max_tokens": 10});
                let resp = round_trip_json(client, &req.to_string()).await;
                assert_eq!(resp["text"], "echo: hello world");
            },
        )
        .await;
    }

    #[tokio::test]
    async fn test_every_reply_is_newline_terminated() {
        // Reply framing mirrors request framing: a client must be able to stop
        // at the newline instead of reading to EOF, which a named pipe cannot
        // signal without dropping the connection.
        with_server("reply-frame", spawn_mock_inference(), |client| async move {
            let req = serde_json::json!({"prompt": "hi"});
            let raw = round_trip(client, &req.to_string()).await;
            assert!(raw.ends_with('\n'), "reply was not terminated: {raw:?}");
        })
        .await;
    }

    // ── protocol round-trip with mock inference ──────────────────────────

    #[tokio::test]
    async fn test_protocol_non_streaming_roundtrip() {
        with_server("non-stream", spawn_mock_inference(), |client| async move {
            let req = serde_json::json!({"prompt": "hello world", "max_tokens": 10});
            let resp = round_trip_json(client, &req.to_string()).await;
            assert_eq!(resp["text"], "echo: hello world");
            assert_eq!(resp["prompt_tokens"], 5);
            assert_eq!(resp["generated_tokens"], 2);
            assert!(resp.get("done").is_none());
        })
        .await;
    }

    #[tokio::test]
    async fn test_protocol_streaming_roundtrip() {
        with_server("stream", spawn_mock_inference(), |client| async move {
            let req = serde_json::json!({"prompt": "hi", "stream": true});
            let output = round_trip(client, &req.to_string()).await;

            let lines: Vec<&str> = output.lines().collect();
            assert!(lines.len() >= 2, "expected at least 2 lines, got {lines:?}");
            assert_eq!(lines[0], r#"{"token":"h"}"#);
            assert_eq!(lines[1], r#"{"token":"i"}"#);

            let last: serde_json::Value = serde_json::from_str(lines.last().unwrap()).unwrap();
            assert_eq!(last["done"], true);
            assert_eq!(last["text"], "echo: hi");
        })
        .await;
    }

    #[tokio::test]
    async fn test_protocol_invalid_json_returns_error() {
        with_server("bad-json", dead_sender(), |client| async move {
            let resp = round_trip_json(client, "not valid json at all").await;
            assert!(resp["error"].as_str().unwrap().contains("invalid request"));
        })
        .await;
    }

    #[tokio::test]
    async fn test_protocol_empty_request_returns_error() {
        with_server("empty", dead_sender(), |mut client| async move {
            // Nothing written at all: the frame ends immediately at EOF.
            client.shutdown().await.unwrap();
            let mut buf = Vec::new();
            client.read_to_end(&mut buf).await.unwrap();
            let resp: serde_json::Value = serde_json::from_slice(&buf).unwrap();
            assert_eq!(resp["error"], "empty request");
        })
        .await;
    }

    #[tokio::test]
    async fn test_protocol_inference_thread_unavailable_non_streaming() {
        with_server("no-infer", dead_sender(), |client| async move {
            let req = serde_json::json!({"prompt": "hi"});
            let resp = round_trip_json(client, &req.to_string()).await;
            assert!(resp["error"].as_str().unwrap().contains("unavailable"));
        })
        .await;
    }

    #[tokio::test]
    async fn test_protocol_inference_thread_unavailable_streaming() {
        with_server("no-stream", dead_sender(), |client| async move {
            let req = serde_json::json!({"prompt": "hi", "stream": true});
            let resp = round_trip_json(client, &req.to_string()).await;
            assert!(resp["error"].as_str().unwrap().contains("unavailable"));
        })
        .await;
    }

    #[tokio::test]
    async fn test_protocol_inference_error_propagates() {
        let (cmd_tx, cmd_rx) = mpsc::channel::<InferCmd>();
        thread::spawn(move || {
            for cmd in cmd_rx {
                if let InferCmd::Run { resp, .. } = cmd {
                    let _ = resp.send(Err(LlamaError::Inference("model exploded".into())));
                }
            }
        });

        with_server("infer-err", cmd_tx, |client| async move {
            let req = serde_json::json!({"prompt": "crash me"});
            let resp = round_trip_json(client, &req.to_string()).await;
            assert!(resp["error"].as_str().unwrap().contains("model exploded"));
        })
        .await;
    }

    #[tokio::test]
    async fn test_protocol_streaming_inference_error_propagates() {
        let (cmd_tx, cmd_rx) = mpsc::channel::<InferCmd>();
        thread::spawn(move || {
            for cmd in cmd_rx {
                if let InferCmd::RunStream {
                    token_tx, done_tx, ..
                } = cmd
                {
                    let _ = token_tx.send("partial ".to_string());
                    if let Some(done) = done_tx {
                        let _ = done.send(Err(LlamaError::Inference(
                            "model exploded mid-stream".into(),
                        )));
                    }
                    drop(token_tx);
                }
            }
        });

        with_server("stream-infer-err", cmd_tx, |client| async move {
            let req = serde_json::json!({"prompt": "crash", "stream": true});
            let output = round_trip(client, &req.to_string()).await;

            let lines: Vec<&str> = output.lines().collect();
            assert!(!lines.is_empty(), "expected at least 1 line, got {lines:?}");
            assert_eq!(lines[0], r#"{"token":"partial "}"#);
            let err_line: serde_json::Value = serde_json::from_str(lines.last().unwrap()).unwrap();
            assert!(
                err_line["error"]
                    .as_str()
                    .unwrap()
                    .contains("model exploded mid-stream")
            );
        })
        .await;
    }

    #[tokio::test]
    async fn test_protocol_streaming_done_tx_dropped() {
        let (cmd_tx, cmd_rx) = mpsc::channel::<InferCmd>();
        thread::spawn(move || {
            for cmd in cmd_rx {
                if let InferCmd::RunStream { token_tx, .. } = cmd {
                    drop(token_tx);
                }
            }
        });

        with_server("stream-crash", cmd_tx, |client| async move {
            let req = serde_json::json!({"prompt": "hi", "stream": true});
            let resp = round_trip_json(client, &req.to_string()).await;
            assert!(
                resp["error"]
                    .as_str()
                    .unwrap()
                    .contains("ended unexpectedly")
            );
        })
        .await;
    }
}

// The Windows named-pipe transport. Mirrors the protocol assertions of the
// Unix suite above against the other listener type.
//
// These do not run on the maintainer's machine — CI's `windows-latest` runner
// is what verifies them, which is the whole reason the cross-platform job
// exists. Treat a failure here as a real bug, not runner flakiness.
#[cfg(all(test, windows))]
mod windows_tests {
    use super::test_support::spawn_mock_inference;
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::windows::named_pipe::ClientOptions;

    use crate::protocol::InferCmd;

    /// A pipe name unique to this process and test, so a parallel test run
    /// cannot collide in the flat `\\.\pipe\` namespace.
    fn pipe_name(tag: &str) -> String {
        format!(r"\\.\pipe\llamad-test-{}-{}", std::process::id(), tag)
    }

    /// Serve exactly one connection on a fresh pipe and return the whole reply
    /// to `request`.
    ///
    /// The request is newline-framed and the client never closes its write
    /// side: a named pipe has no half-close, so this is the only framing the
    /// transport can express — the property the Unix suite's
    /// `test_newline_terminated_request_without_half_close` guards from the
    /// other side.
    async fn round_trip(tag: &str, cmd_tx: mpsc::Sender<InferCmd>, request: &str) -> String {
        let name = pipe_name(tag);
        let mut server_pipe = bind_pipe(&name, true).expect("bind pipe");
        let server = tokio::spawn(async move {
            server_pipe.connect().await.expect("client attach");
            handle_connection(&mut server_pipe, &cmd_tx).await;
        });

        let mut client = ClientOptions::new().open(&name).expect("open pipe");
        client
            .write_all(format!("{request}\n").as_bytes())
            .await
            .expect("write request");

        let mut buf = Vec::new();
        client.read_to_end(&mut buf).await.expect("read reply");
        server.await.expect("server task");
        String::from_utf8(buf).expect("reply is utf-8")
    }

    #[tokio::test]
    async fn test_bind_pipe_refuses_a_duplicate_first_instance() {
        // The named-pipe equivalent of refusing to steal a live daemon's
        // socket: `first_pipe_instance` fails if the name is already owned.
        let name = pipe_name("dup");
        let _first = bind_pipe(&name, true).expect("first instance");
        assert!(
            bind_pipe(&name, true).is_err(),
            "a second first-instance claim on the same name must fail"
        );
    }

    #[tokio::test]
    async fn test_protocol_non_streaming_roundtrip_over_pipe() {
        let req = serde_json::json!({"prompt": "hello world", "max_tokens": 10});
        let raw = round_trip("nonstream", spawn_mock_inference(), &req.to_string()).await;
        let resp: serde_json::Value = serde_json::from_str(raw.trim_end()).unwrap();
        assert_eq!(resp["text"], "echo: hello world");
        assert_eq!(resp["prompt_tokens"], 5);
        assert_eq!(resp["generated_tokens"], 2);
    }

    #[tokio::test]
    async fn test_every_reply_is_newline_terminated_over_pipe() {
        // Reply framing matters more here than on Unix: without a terminator a
        // pipe client cannot tell "reply complete" from "more coming" short of
        // waiting for the server to drop the connection.
        let req = serde_json::json!({"prompt": "hi"});
        let raw = round_trip("replyframe", spawn_mock_inference(), &req.to_string()).await;
        assert!(raw.ends_with('\n'), "reply was not terminated: {raw:?}");
    }

    #[tokio::test]
    async fn test_protocol_streaming_roundtrip_over_pipe() {
        let req = serde_json::json!({"prompt": "hi", "stream": true});
        let output = round_trip("stream", spawn_mock_inference(), &req.to_string()).await;

        let lines: Vec<&str> = output.lines().collect();
        assert!(lines.len() >= 2, "expected at least 2 lines, got {lines:?}");
        assert_eq!(lines[0], r#"{"token":"h"}"#);
        assert_eq!(lines[1], r#"{"token":"i"}"#);

        let last: serde_json::Value = serde_json::from_str(lines.last().unwrap()).unwrap();
        assert_eq!(last["done"], true);
        assert_eq!(last["text"], "echo: hi");
    }

    #[tokio::test]
    async fn test_invalid_json_returns_error_over_pipe() {
        let raw = round_trip("badjson", spawn_mock_inference(), "{not json}").await;
        let resp: serde_json::Value = serde_json::from_str(raw.trim_end()).unwrap();
        assert!(
            resp["error"].as_str().unwrap().contains("invalid request"),
            "got {resp}"
        );
    }
}
