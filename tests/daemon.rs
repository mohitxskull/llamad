//! End-to-end test of the `llamad` binary itself.
//!
//! Everything else tests the library: `src/server.rs`'s unit tests drive
//! `handle_connection` against a mock engine, and the other integration
//! binaries drive a real engine in process. Neither runs `src/main.rs`, so its
//! argument handling, accept loop, signal handling and socket cleanup had no
//! coverage at all — the daemon could fail to bind, leak its socket, or hang on
//! shutdown and the whole suite would stay green.
//!
//! Unix only. The binary is built for Windows too, but the shutdown path there
//! is a named pipe with no filesystem entry to clean up and no SIGINT to send;
//! the pipe protocol itself is covered by `server::windows_tests`.
#![cfg(unix)]

use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use serial_test::serial;

mod common;
use common::model_hybrid;

/// Loading a GGUF and starting the accept loop is not instant, and the machine
/// may be busy running other tests. Generous on purpose: this bound exists to
/// turn a hang into a readable failure, not to assert a startup time.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(120);
/// Shutdown joins both engine threads, which can take one decode step.
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(60);

/// A daemon child that is killed on drop, so a failed assertion cannot leave a
/// process holding a model in memory (~150 MB) for the rest of the run.
struct Daemon {
    child: Child,
    socket: PathBuf,
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.socket);
    }
}

impl Daemon {
    /// Start the daemon and wait until its socket is accepting connections.
    ///
    /// Waits for a successful *connect*, not merely for the file to exist:
    /// `bind_socket` creates the entry before `main` reaches the accept loop,
    /// so an existence check races the daemon and yields flaky connection
    /// refusals.
    fn start(tag: &str) -> Daemon {
        let socket = std::env::temp_dir().join(format!(
            "llamad-daemon-test-{}-{tag}.sock",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&socket);

        // `CARGO_BIN_EXE_<name>` is set by cargo for integration tests and
        // points at the binary built from this same source tree, so the test
        // cannot accidentally exercise an installed copy.
        let child = Command::new(env!("CARGO_BIN_EXE_llamad"))
            .arg(model_hybrid())
            .arg(&socket)
            .env("LLAMAD_N_CTX", "512")
            .env("LLAMAD_N_SLOTS", "1")
            .spawn()
            .expect("spawn the llamad binary");

        let mut daemon = Daemon { child, socket };
        let deadline = Instant::now() + STARTUP_TIMEOUT;
        while Instant::now() < deadline {
            if let Some(status) = daemon
                .child
                .try_wait()
                .expect("poll the daemon's exit status")
            {
                panic!("daemon exited during startup with {status}");
            }
            if UnixStream::connect(&daemon.socket).is_ok() {
                return daemon;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        panic!(
            "daemon did not accept a connection within {STARTUP_TIMEOUT:?} (socket {})",
            daemon.socket.display()
        );
    }

    /// Send one newline-framed request and read the whole reply.
    fn request(&self, body: &str) -> String {
        let mut stream = UnixStream::connect(&self.socket).expect("connect to the daemon");
        stream
            .set_read_timeout(Some(SHUTDOWN_TIMEOUT))
            .expect("set a read timeout");
        stream
            .write_all(format!("{body}\n").as_bytes())
            .expect("write the request");
        let mut reply = String::new();
        stream
            .read_to_string(&mut reply)
            .expect("read the daemon's reply");
        reply
    }

    /// Send SIGINT and wait for the process to exit on its own.
    ///
    /// `kill(1)` rather than a `libc` dev-dependency: `Child::kill` sends
    /// SIGKILL, which would bypass the graceful path this test exists to
    /// cover.
    fn interrupt_and_wait(&mut self) -> std::process::ExitStatus {
        let pid = self.child.id().to_string();
        let killed = Command::new("kill")
            .args(["-INT", &pid])
            .status()
            .expect("run kill(1)");
        assert!(killed.success(), "kill -INT {pid} failed");

        let deadline = Instant::now() + SHUTDOWN_TIMEOUT;
        while Instant::now() < deadline {
            if let Some(status) = self.child.try_wait().expect("poll the daemon") {
                return status;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        panic!("daemon did not exit within {SHUTDOWN_TIMEOUT:?} of SIGINT");
    }
}

#[serial]
#[test]
fn daemon_serves_a_request_and_shuts_down_cleanly() {
    let mut daemon = Daemon::start("roundtrip");

    // The socket accepts inference requests, so its mode is a security
    // property, not a detail — the default path lives in a world-writable
    // directory. Asserted against the real binary because `bind_socket`'s own
    // test only proves the library function does it.
    let mode = std::fs::metadata(&daemon.socket)
        .expect("stat the socket")
        .permissions()
        .mode();
    assert_eq!(mode & 0o777, 0o600, "socket mode was {:o}", mode & 0o777);

    let reply = daemon.request(r#"{"prompt":"Say hi.","max_tokens":8}"#);
    let json: serde_json::Value =
        serde_json::from_str(reply.trim_end()).unwrap_or_else(|e| panic!("reply {reply:?}: {e}"));
    assert!(
        json["text"].as_str().is_some_and(|t| !t.is_empty()),
        "daemon returned no text: {json}"
    );
    assert!(
        json["prompt_tokens"].as_u64().is_some_and(|n| n > 0),
        "daemon returned no prompt token count: {json}"
    );
    assert!(
        reply.ends_with('\n'),
        "reply was not newline terminated: {reply:?}"
    );

    let status = daemon.interrupt_and_wait();
    assert!(status.success(), "daemon exited with {status}");

    // The socket is unlinked on the way out. A daemon that leaves one behind
    // makes its own next start take the stale-socket reclaim path, which is
    // only safe because nothing is listening — a race worth not relying on.
    assert!(
        !Path::new(&daemon.socket).exists(),
        "daemon left its socket behind at {}",
        daemon.socket.display()
    );
}

#[serial]
#[test]
fn daemon_streams_ndjson_when_asked() {
    let mut daemon = Daemon::start("stream");
    let reply = daemon.request(r#"{"prompt":"Count: 1 2 3","max_tokens":8,"stream":true}"#);

    let lines: Vec<&str> = reply.lines().filter(|l| !l.trim().is_empty()).collect();
    assert!(
        lines.len() >= 2,
        "expected token lines plus a terminator, got {lines:?}"
    );
    // Every line is a complete JSON object — the framing contract a streaming
    // client depends on.
    for line in &lines {
        serde_json::from_str::<serde_json::Value>(line)
            .unwrap_or_else(|e| panic!("streamed line {line:?} is not JSON: {e}"));
    }
    let last: serde_json::Value = serde_json::from_str(lines.last().unwrap()).unwrap();
    assert_eq!(last["done"], true, "stream did not end with a done line");

    let status = daemon.interrupt_and_wait();
    assert!(status.success(), "daemon exited with {status}");
}

#[test]
fn daemon_rejects_a_missing_model_argument() {
    // Exits 2 rather than 0: a missing argument is a usage error, and a
    // supervisor that restarts on non-zero needs to see one.
    let out = Command::new(env!("CARGO_BIN_EXE_llamad"))
        .output()
        .expect("run the daemon with no arguments");
    assert_eq!(out.status.code(), Some(2), "expected exit code 2");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("Usage:"), "no usage text: {stderr}");
}

#[test]
fn daemon_help_exits_zero() {
    let out = Command::new(env!("CARGO_BIN_EXE_llamad"))
        .arg("--help")
        .output()
        .expect("run the daemon with --help");
    assert!(out.status.success(), "--help exited with {}", out.status);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("LLAMAD_N_CTX"), "no env docs: {stderr}");
}
