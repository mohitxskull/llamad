//! Unix-socket server speaking the JSON (and NDJSON, when streaming) protocol.
//!
//! One request per connection: the client writes a JSON object, half-closes
//! the write side, and reads the reply.

use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::path::Path;
use std::sync::mpsc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixListener;
use tokio::sync::oneshot;

use crate::protocol::{InferCmd, LlamaError, Request, Response, TokenChunk};

/// Socket path used when none is given.
pub const DEFAULT_SOCKET_PATH: &str = "/tmp/llamad.sock";
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

/// Read one request from `stream`, run it, and write the reply.
pub async fn handle_connection(
    stream: &mut tokio::net::UnixStream,
    cmd_tx: &mpsc::Sender<InferCmd>,
) {
    let mut buf = Vec::new();
    // Read one byte past the cap so an over-long body is detectable rather
    // than silently truncated into a parse error.
    let capped = (&mut *stream).take(MAX_REQUEST_BYTES + 1);
    tokio::pin!(capped);
    if tokio::time::timeout(READ_TIMEOUT, capped.read_to_end(&mut buf))
        .await
        .is_err()
    {
        let err = serde_json::json!({"error": "read timeout"});
        let _ = stream.write_all(err.to_string().as_bytes()).await;
        return;
    }

    if buf.len() as u64 > MAX_REQUEST_BYTES {
        // Discard whatever the client is still writing before replying.
        // Closing a socket that has unread data queued makes the kernel send
        // RST, which throws away our error message along with it — the client
        // would see a bare connection reset instead of the reason. Reading
        // into a fixed scratch buffer keeps memory bounded, and READ_TIMEOUT
        // bounds how long a client can hold the connection open.
        let mut sink = [0u8; 8192];
        let _ = tokio::time::timeout(READ_TIMEOUT, async {
            while matches!(stream.read(&mut sink).await, Ok(n) if n > 0) {}
        })
        .await;

        let err = serde_json::json!({
            "error": format!("request too large (max {MAX_REQUEST_BYTES} bytes)")
        });
        let _ = stream.write_all(err.to_string().as_bytes()).await;
        return;
    }

    if buf.is_empty() {
        let err = serde_json::json!({"error": "empty request"});
        let _ = stream.write_all(err.to_string().as_bytes()).await;
        return;
    }

    let req: Request = match serde_json::from_slice(&buf) {
        Ok(r) => r,
        Err(e) => {
            let err = serde_json::json!({"error": format!("invalid request: {e}")});
            let _ = stream.write_all(err.to_string().as_bytes()).await;
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

async fn handle_streaming(
    stream: &mut tokio::net::UnixStream,
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
        let err = serde_json::json!({"error": "inference thread unavailable"});
        let _ = stream.write_all(err.to_string().as_bytes()).await;
        return;
    }

    loop {
        match tokio::time::timeout(STREAM_TIMEOUT, token_rx.recv()).await {
            Ok(Some(token)) => {
                if let Ok(json) = serde_json::to_string(&TokenChunk { token: &token }) {
                    if stream.write_all(json.as_bytes()).await.is_err() {
                        return;
                    }
                    if stream.write_all(b"\n").await.is_err() {
                        return;
                    }
                }
            }
            Ok(None) => break,
            Err(_) => {
                let err = serde_json::json!({"error": "stream timeout"});
                let _ = stream.write_all(err.to_string().as_bytes()).await;
                return;
            }
        }
    }

    match done_rx.await {
        Ok(Ok(result)) => {
            let done = serde_json::json!({
                "text": result.text,
                "prompt_tokens": result.prompt_tokens,
                "generated_tokens": result.generated_tokens,
                "done": true,
            });
            let _ = stream.write_all(done.to_string().as_bytes()).await;
            let _ = stream.write_all(b"\n").await;
        }
        Ok(Err(e)) => {
            let err = serde_json::json!({"error": format!("inference: {e}")});
            let _ = stream.write_all(err.to_string().as_bytes()).await;
        }
        Err(_) => {
            let err = serde_json::json!({"error": "inference stream ended unexpectedly"});
            let _ = stream.write_all(err.to_string().as_bytes()).await;
        }
    }
}

async fn handle_non_streaming(
    stream: &mut tokio::net::UnixStream,
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
        let err = serde_json::json!({"error": "inference thread unavailable"});
        let _ = stream.write_all(err.to_string().as_bytes()).await;
        return;
    }

    match resp_rx.await {
        Ok(Ok(result)) => {
            let resp = Response {
                text: result.text,
                prompt_tokens: result.prompt_tokens,
                generated_tokens: result.generated_tokens,
            };
            if let Ok(json) = serde_json::to_string(&resp) {
                let _ = stream.write_all(json.as_bytes()).await;
            }
        }
        Ok(Err(e)) => {
            let err = serde_json::json!({"error": format!("inference: {e}")});
            let _ = stream.write_all(err.to_string().as_bytes()).await;
        }
        Err(_) => {
            let err = serde_json::json!({"error": "inference thread crashed"});
            let _ = stream.write_all(err.to_string().as_bytes()).await;
        }
    }
}

#[cfg(test)]
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
