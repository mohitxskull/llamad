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

// The existing suite drives a Unix domain socket directly. The Windows
// transport has its own module below; the protocol assertions are duplicated
// rather than shared because the listener types have no common trait to
// abstract over — a named-pipe server handle becomes the connection, a Unix
// listener yields one.
#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    use crate::protocol::InferResult;

    fn spawn_mock_inference() -> mpsc::Sender<InferCmd> {
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            for cmd in rx {
                match cmd {
                    InferCmd::Run { request, resp } => {
                        let result = InferResult {
                            text: format!("echo: {}", request.prompt),
                            prompt_tokens: 5,
                            generated_tokens: 2,
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

    // ── bind_socket ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_bind_socket_creates_listener() {
        let dir = std::env::temp_dir().join(format!("llamad-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.sock");
        let _ = std::fs::remove_file(&path);

        let listener = bind_socket(path.to_str().unwrap()).unwrap();
        assert!(path.exists());
        drop(listener);
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn test_bind_socket_reclaims_stale_socket() {
        // A crashed daemon leaves a socket file with nobody listening. That
        // one is safe to unlink and rebind.
        let dir = std::env::temp_dir().join(format!("llamad-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("stale.sock");
        let _ = std::fs::remove_file(&path);

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
        let dir = std::env::temp_dir().join(format!("llamad-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("not-a-socket");
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
        let dir = std::env::temp_dir().join(format!("llamad-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("live.sock");
        let _ = std::fs::remove_file(&path);

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
        let dir = std::env::temp_dir().join(format!("llamad-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("perms.sock");
        let _ = std::fs::remove_file(&path);

        let listener = bind_socket(&path).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600, "socket mode was {:o}", mode & 0o777);
        drop(listener);
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn test_request_over_size_cap_is_rejected() {
        let dir = std::env::temp_dir().join(format!("llamad-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let socket_path = dir.join("too-big.sock");
        let _ = std::fs::remove_file(&socket_path);

        let listener = UnixListener::bind(&socket_path).unwrap();
        let cmd_tx = mpsc::channel::<InferCmd>().0;

        let path = socket_path.clone();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            handle_connection(&mut stream, &cmd_tx).await;
        });

        tokio::time::sleep(Duration::from_millis(100)).await;

        let mut client = tokio::net::UnixStream::connect(&path).await.unwrap();
        // A valid-looking request whose body exceeds the cap.
        let oversized = format!(
            r#"{{"prompt":"{}"}}"#,
            "a".repeat(MAX_REQUEST_BYTES as usize + 16)
        );
        // The server stops reading at the cap, so the client's write may fail
        // once the receive buffer fills — that is the intended backpressure.
        let _ = client.write_all(oversized.as_bytes()).await;
        let _ = client.shutdown().await;

        let mut buf = Vec::new();
        client.read_to_end(&mut buf).await.unwrap();
        let resp: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert!(
            resp["error"].as_str().unwrap().contains("too large"),
            "expected a size-cap rejection, got: {resp}"
        );

        server.await.unwrap();
        let _ = std::fs::remove_file(&socket_path);
    }

    // ── protocol round-trip with mock inference ──────────────────────────

    #[tokio::test]
    async fn test_protocol_non_streaming_roundtrip() {
        let dir = std::env::temp_dir().join(format!("llamad-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let socket_path = dir.join("non-stream.sock");
        let _ = std::fs::remove_file(&socket_path);

        let listener = UnixListener::bind(&socket_path).unwrap();
        let cmd_tx = spawn_mock_inference();

        let cmdtx = cmd_tx.clone();
        let path = socket_path.clone();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            handle_connection(&mut stream, &cmdtx).await;
        });

        tokio::time::sleep(Duration::from_millis(100)).await;

        let mut client = tokio::net::UnixStream::connect(&path).await.unwrap();
        let req = serde_json::json!({"prompt": "hello world", "max_tokens": 10});
        client.write_all(req.to_string().as_bytes()).await.unwrap();
        client.shutdown().await.unwrap();

        let mut buf = Vec::new();
        client.read_to_end(&mut buf).await.unwrap();

        let resp: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(resp["text"], "echo: hello world");
        assert_eq!(resp["prompt_tokens"], 5);
        assert_eq!(resp["generated_tokens"], 2);
        assert!(resp.get("done").is_none());

        server.await.unwrap();
        cmd_tx.send(InferCmd::Shutdown).ok();
        let _ = std::fs::remove_file(&socket_path);
    }

    #[tokio::test]
    async fn test_newline_terminated_request_without_half_close() {
        // The mechanism Windows depends on. A named-pipe client cannot
        // half-close, so the request has to be able to end at a newline while
        // the connection stays open in both directions. Exercised here on a
        // Unix socket because the framing is transport-independent — if this
        // breaks, the Windows daemon deadlocks.
        let dir = std::env::temp_dir().join(format!("llamad-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let socket_path = dir.join("newline-frame.sock");
        let _ = std::fs::remove_file(&socket_path);

        let listener = UnixListener::bind(&socket_path).unwrap();
        let cmd_tx = spawn_mock_inference();

        let cmdtx = cmd_tx.clone();
        let path = socket_path.clone();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            handle_connection(&mut stream, &cmdtx).await;
        });

        tokio::time::sleep(Duration::from_millis(100)).await;

        let mut client = tokio::net::UnixStream::connect(&path).await.unwrap();
        let req = serde_json::json!({"prompt": "hello world", "max_tokens": 10});
        // Note: no `shutdown()`. The newline is the only end-of-request signal.
        client
            .write_all(format!("{req}\n").as_bytes())
            .await
            .unwrap();

        let mut buf = Vec::new();
        client.read_to_end(&mut buf).await.unwrap();
        let resp: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(resp["text"], "echo: hello world");

        server.await.unwrap();
        cmd_tx.send(InferCmd::Shutdown).ok();
        let _ = std::fs::remove_file(&socket_path);
    }

    #[tokio::test]
    async fn test_protocol_streaming_roundtrip() {
        let dir = std::env::temp_dir().join(format!("llamad-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let socket_path = dir.join("stream.sock");
        let _ = std::fs::remove_file(&socket_path);

        let listener = UnixListener::bind(&socket_path).unwrap();
        let cmd_tx = spawn_mock_inference();

        let cmdtx = cmd_tx.clone();
        let path = socket_path.clone();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            handle_connection(&mut stream, &cmdtx).await;
        });

        tokio::time::sleep(Duration::from_millis(100)).await;

        let mut client = tokio::net::UnixStream::connect(&path).await.unwrap();
        let req = serde_json::json!({"prompt": "hi", "stream": true});
        client.write_all(req.to_string().as_bytes()).await.unwrap();
        client.shutdown().await.unwrap();

        let mut buf = Vec::new();
        client.read_to_end(&mut buf).await.unwrap();
        let output = String::from_utf8_lossy(&buf);

        let lines: Vec<&str> = output.lines().collect();
        assert!(lines.len() >= 2, "expected at least 2 lines, got {lines:?}");
        assert_eq!(lines[0], r#"{"token":"h"}"#);
        assert_eq!(lines[1], r#"{"token":"i"}"#);

        let last: serde_json::Value = serde_json::from_str(lines.last().unwrap()).unwrap();
        assert_eq!(last["done"], true);
        assert_eq!(last["text"], "echo: hi");

        server.await.unwrap();
        cmd_tx.send(InferCmd::Shutdown).ok();
        let _ = std::fs::remove_file(&socket_path);
    }

    #[tokio::test]
    async fn test_protocol_invalid_json_returns_error() {
        let dir = std::env::temp_dir().join(format!("llamad-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let socket_path = dir.join("bad-json.sock");
        let _ = std::fs::remove_file(&socket_path);

        let listener = UnixListener::bind(&socket_path).unwrap();
        let cmd_tx = mpsc::channel::<InferCmd>().0;

        let path = socket_path.clone();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            handle_connection(&mut stream, &cmd_tx).await;
        });

        tokio::time::sleep(Duration::from_millis(100)).await;

        let mut client = tokio::net::UnixStream::connect(&path).await.unwrap();
        client.write_all(b"not valid json at all").await.unwrap();
        client.shutdown().await.unwrap();

        let mut buf = Vec::new();
        client.read_to_end(&mut buf).await.unwrap();
        let resp: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert!(resp["error"].as_str().unwrap().contains("invalid request"));

        server.await.unwrap();
        let _ = std::fs::remove_file(&socket_path);
    }

    #[tokio::test]
    async fn test_protocol_empty_request_returns_error() {
        let dir = std::env::temp_dir().join(format!("llamad-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let socket_path = dir.join("empty.sock");
        let _ = std::fs::remove_file(&socket_path);

        let listener = UnixListener::bind(&socket_path).unwrap();
        let cmd_tx = mpsc::channel::<InferCmd>().0;

        let path = socket_path.clone();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            handle_connection(&mut stream, &cmd_tx).await;
        });

        tokio::time::sleep(Duration::from_millis(100)).await;

        let mut client = tokio::net::UnixStream::connect(&path).await.unwrap();
        client.shutdown().await.unwrap();

        let mut buf = Vec::new();
        client.read_to_end(&mut buf).await.unwrap();
        let resp: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(resp["error"], "empty request");

        server.await.unwrap();
        let _ = std::fs::remove_file(&socket_path);
    }

    #[tokio::test]
    async fn test_protocol_inference_thread_unavailable_non_streaming() {
        let dir = std::env::temp_dir().join(format!("llamad-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let socket_path = dir.join("no-infer.sock");
        let _ = std::fs::remove_file(&socket_path);

        let listener = UnixListener::bind(&socket_path).unwrap();
        let (cmd_tx, cmd_rx) = mpsc::channel::<InferCmd>();
        drop(cmd_rx);

        let cmdtx = cmd_tx.clone();
        let path = socket_path.clone();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            handle_connection(&mut stream, &cmdtx).await;
        });

        tokio::time::sleep(Duration::from_millis(100)).await;

        let mut client = tokio::net::UnixStream::connect(&path).await.unwrap();
        let req = serde_json::json!({"prompt": "hi"});
        client.write_all(req.to_string().as_bytes()).await.unwrap();
        client.shutdown().await.unwrap();

        let mut buf = Vec::new();
        client.read_to_end(&mut buf).await.unwrap();
        let resp: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert!(resp["error"].as_str().unwrap().contains("unavailable"));

        server.await.unwrap();
        let _ = std::fs::remove_file(&socket_path);
    }

    #[tokio::test]
    async fn test_protocol_inference_thread_unavailable_streaming() {
        let dir = std::env::temp_dir().join(format!("llamad-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let socket_path = dir.join("no-stream.sock");
        let _ = std::fs::remove_file(&socket_path);

        let listener = UnixListener::bind(&socket_path).unwrap();
        let (cmd_tx, cmd_rx) = mpsc::channel::<InferCmd>();
        drop(cmd_rx);

        let cmdtx = cmd_tx.clone();
        let path = socket_path.clone();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            handle_connection(&mut stream, &cmdtx).await;
        });

        tokio::time::sleep(Duration::from_millis(100)).await;

        let mut client = tokio::net::UnixStream::connect(&path).await.unwrap();
        let req = serde_json::json!({"prompt": "hi", "stream": true});
        client.write_all(req.to_string().as_bytes()).await.unwrap();
        client.shutdown().await.unwrap();

        let mut buf = Vec::new();
        client.read_to_end(&mut buf).await.unwrap();
        let resp: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert!(resp["error"].as_str().unwrap().contains("unavailable"));

        server.await.unwrap();
        let _ = std::fs::remove_file(&socket_path);
    }

    #[tokio::test]
    async fn test_protocol_inference_error_propagates() {
        let dir = std::env::temp_dir().join(format!("llamad-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let socket_path = dir.join("infer-err.sock");
        let _ = std::fs::remove_file(&socket_path);

        let listener = UnixListener::bind(&socket_path).unwrap();
        let (cmd_tx, cmd_rx) = mpsc::channel::<InferCmd>();

        thread::spawn(move || {
            for cmd in cmd_rx {
                if let InferCmd::Run { resp, .. } = cmd {
                    let _ = resp.send(Err(LlamaError::Inference("model exploded".into())));
                }
            }
        });

        let cmdtx = cmd_tx.clone();
        let path = socket_path.clone();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            handle_connection(&mut stream, &cmdtx).await;
        });

        tokio::time::sleep(Duration::from_millis(100)).await;

        let mut client = tokio::net::UnixStream::connect(&path).await.unwrap();
        let req = serde_json::json!({"prompt": "crash me"});
        client.write_all(req.to_string().as_bytes()).await.unwrap();
        client.shutdown().await.unwrap();

        let mut buf = Vec::new();
        client.read_to_end(&mut buf).await.unwrap();
        let resp: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert!(resp["error"].as_str().unwrap().contains("model exploded"));

        server.await.unwrap();
        cmd_tx.send(InferCmd::Shutdown).ok();
        let _ = std::fs::remove_file(&socket_path);
    }

    #[tokio::test]
    async fn test_protocol_streaming_inference_error_propagates() {
        let dir = std::env::temp_dir().join(format!("llamad-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let socket_path = dir.join("stream-infer-err.sock");
        let _ = std::fs::remove_file(&socket_path);

        let listener = UnixListener::bind(&socket_path).unwrap();
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

        let cmdtx = cmd_tx.clone();
        let path = socket_path.clone();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            handle_connection(&mut stream, &cmdtx).await;
        });

        tokio::time::sleep(Duration::from_millis(100)).await;

        let mut client = tokio::net::UnixStream::connect(&path).await.unwrap();
        let req = serde_json::json!({"prompt": "crash", "stream": true});
        client.write_all(req.to_string().as_bytes()).await.unwrap();
        client.shutdown().await.unwrap();

        let mut buf = Vec::new();
        client.read_to_end(&mut buf).await.unwrap();
        let output = String::from_utf8_lossy(&buf);

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

        server.await.unwrap();
        cmd_tx.send(InferCmd::Shutdown).ok();
        let _ = std::fs::remove_file(&socket_path);
    }

    #[tokio::test]
    async fn test_protocol_streaming_done_tx_dropped() {
        let dir = std::env::temp_dir().join(format!("llamad-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let socket_path = dir.join("stream-crash.sock");
        let _ = std::fs::remove_file(&socket_path);

        let listener = UnixListener::bind(&socket_path).unwrap();
        let (cmd_tx, cmd_rx) = mpsc::channel::<InferCmd>();

        thread::spawn(move || {
            for cmd in cmd_rx {
                if let InferCmd::RunStream { token_tx, .. } = cmd {
                    drop(token_tx);
                }
            }
        });

        let cmdtx = cmd_tx.clone();
        let path = socket_path.clone();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            handle_connection(&mut stream, &cmdtx).await;
        });

        tokio::time::sleep(Duration::from_millis(100)).await;

        let mut client = tokio::net::UnixStream::connect(&path).await.unwrap();
        let req = serde_json::json!({"prompt": "hi", "stream": true});
        client.write_all(req.to_string().as_bytes()).await.unwrap();
        client.shutdown().await.unwrap();

        let mut buf = Vec::new();
        client.read_to_end(&mut buf).await.unwrap();
        let resp: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert!(
            resp["error"]
                .as_str()
                .unwrap()
                .contains("ended unexpectedly")
        );

        server.await.unwrap();
        cmd_tx.send(InferCmd::Shutdown).ok();
        let _ = std::fs::remove_file(&socket_path);
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
    use super::*;
    use std::thread;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::windows::named_pipe::ClientOptions;

    use crate::protocol::{InferCmd, InferResult};

    /// A pipe name unique to this process and test, so a parallel test run
    /// cannot collide in the flat `\\.\pipe\` namespace.
    fn pipe_name(tag: &str) -> String {
        format!(r"\\.\pipe\llamad-test-{}-{}", std::process::id(), tag)
    }

    /// Same echo behaviour as the Unix suite's mock; duplicated because the
    /// two test modules are compiled on different platforms.
    fn spawn_mock_inference() -> mpsc::Sender<InferCmd> {
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
        let name = pipe_name("nonstream");
        let mut server_pipe = bind_pipe(&name, true).expect("bind pipe");
        let cmd_tx = spawn_mock_inference();
        let cmdtx = cmd_tx.clone();

        let server = tokio::spawn(async move {
            server_pipe.connect().await.expect("client attach");
            handle_connection(&mut server_pipe, &cmdtx).await;
        });

        let mut client = ClientOptions::new().open(&name).expect("open pipe");
        let req = serde_json::json!({"prompt": "hello world", "max_tokens": 10});
        // No `shutdown()` — a named pipe has no half-close, so the trailing
        // newline is the only thing that ends the request.
        client
            .write_all(format!("{req}\n").as_bytes())
            .await
            .unwrap();

        let mut buf = Vec::new();
        client.read_to_end(&mut buf).await.unwrap();
        let resp: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(resp["text"], "echo: hello world");
        assert_eq!(resp["prompt_tokens"], 5);
        assert_eq!(resp["generated_tokens"], 2);

        server.await.unwrap();
        cmd_tx.send(InferCmd::Shutdown).ok();
    }

    #[tokio::test]
    async fn test_protocol_streaming_roundtrip_over_pipe() {
        let name = pipe_name("stream");
        let mut server_pipe = bind_pipe(&name, true).expect("bind pipe");
        let cmd_tx = spawn_mock_inference();
        let cmdtx = cmd_tx.clone();

        let server = tokio::spawn(async move {
            server_pipe.connect().await.expect("client attach");
            handle_connection(&mut server_pipe, &cmdtx).await;
        });

        let mut client = ClientOptions::new().open(&name).expect("open pipe");
        let req = serde_json::json!({"prompt": "hi", "stream": true});
        client
            .write_all(format!("{req}\n").as_bytes())
            .await
            .unwrap();

        let mut buf = Vec::new();
        client.read_to_end(&mut buf).await.unwrap();
        let output = String::from_utf8_lossy(&buf);

        let lines: Vec<&str> = output.lines().collect();
        assert!(lines.len() >= 2, "expected at least 2 lines, got {lines:?}");
        assert_eq!(lines[0], r#"{"token":"h"}"#);
        assert_eq!(lines[1], r#"{"token":"i"}"#);

        let last: serde_json::Value = serde_json::from_str(lines.last().unwrap()).unwrap();
        assert_eq!(last["done"], true);
        assert_eq!(last["text"], "echo: hi");

        server.await.unwrap();
        cmd_tx.send(InferCmd::Shutdown).ok();
    }

    #[tokio::test]
    async fn test_invalid_json_returns_error_over_pipe() {
        let name = pipe_name("badjson");
        let mut server_pipe = bind_pipe(&name, true).expect("bind pipe");
        let cmd_tx = spawn_mock_inference();
        let cmdtx = cmd_tx.clone();

        let server = tokio::spawn(async move {
            server_pipe.connect().await.expect("client attach");
            handle_connection(&mut server_pipe, &cmdtx).await;
        });

        let mut client = ClientOptions::new().open(&name).expect("open pipe");
        client.write_all(b"{not json}\n").await.unwrap();

        let mut buf = Vec::new();
        client.read_to_end(&mut buf).await.unwrap();
        let resp: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert!(
            resp["error"].as_str().unwrap().contains("invalid request"),
            "got {resp}"
        );

        server.await.unwrap();
        cmd_tx.send(InferCmd::Shutdown).ok();
    }
}
