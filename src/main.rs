//! The `llamad` daemon: serves the JSON/NDJSON protocol over local IPC.
//!
//! Unix domain socket on Unix, named pipe on Windows. The wire protocol is
//! identical; only the listener and the accept loop differ.

use llamad::inference::Engine;
use llamad::server::{DEFAULT_SOCKET_PATH, handle_connection};

#[cfg(windows)]
use llamad::server::bind_pipe;
#[cfg(unix)]
use llamad::server::bind_socket;

/// Wait for the next client and return its stream.
///
/// A Unix listener yields a fresh stream per connection and keeps listening.
/// A named-pipe server handle *becomes* the connection once a client attaches,
/// so the Windows arm creates the next instance and swaps it in — otherwise
/// nothing would be listening between connections, and a client arriving in
/// that window would get "file not found" rather than waiting.
#[cfg(unix)]
async fn accept(
    listener: &mut tokio::net::UnixListener,
    _name: &str,
) -> anyhow::Result<tokio::net::UnixStream> {
    Ok(listener.accept().await?.0)
}

#[cfg(windows)]
async fn accept(
    pipe: &mut tokio::net::windows::named_pipe::NamedPipeServer,
    name: &str,
) -> anyhow::Result<tokio::net::windows::named_pipe::NamedPipeServer> {
    pipe.connect().await?;
    let next = bind_pipe(name, false)?;
    Ok(std::mem::replace(pipe, next))
}

fn usage(program: &str) {
    eprintln!("Usage: {program} <model.gguf> [socket-path]");
    eprintln!();
    eprintln!("Arguments:");
    eprintln!("  <model.gguf>              Path to GGUF model file (required)");
    eprintln!(
        "  [socket-path]             Socket path or pipe name (default: {DEFAULT_SOCKET_PATH})"
    );
    eprintln!();
    eprintln!("Environment:");
    eprintln!("  LLAMAD_N_SLOTS            Concurrent sequences (default 1; splits N_CTX)");
    eprintln!("  LLAMAD_N_CTX              Total KV context in tokens (default 2048)");
    eprintln!("  LLAMAD_N_THREADS          Decode threads (default: physical cores)");
    eprintln!("  LLAMAD_N_THREADS_BATCH    Prefill threads (default: physical cores)");
    eprintln!("  LLAMAD_KV_CACHE           Reuse KV prefixes across requests (default on)");
    eprintln!("  RUST_LOG                  Log filter (default: warn)");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(tracing_subscriber::filter::LevelFilter::WARN.into())
                .from_env_lossy(),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let program = args.first().map_or("llamad", String::as_str);

    // A missing model path is a usage error; an explicit --help is not.
    if args.len() < 2 {
        usage(program);
        std::process::exit(2);
    }
    if args[1] == "--help" || args[1] == "-h" {
        usage(program);
        return Ok(());
    }

    let model_path = &args[1];
    let socket_path = args
        .get(2)
        .map_or(DEFAULT_SOCKET_PATH, |s| s.as_str())
        .to_owned();

    #[cfg(unix)]
    let mut acceptor = bind_socket(&socket_path)?;
    #[cfg(windows)]
    let mut acceptor = bind_pipe(&socket_path, true)?;

    let engine = Engine::start(model_path)?;

    tracing::info!("llamad listening on {socket_path}");

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("Shutdown signal received");
                break;
            }
            result = accept(&mut acceptor, &socket_path) => {
                let mut stream = result?;
                let cmd_tx = engine.sender().clone();
                tokio::spawn(async move {
                    handle_connection(&mut stream, &cmd_tx).await;
                });
            }
        }
    }

    tracing::info!("Shutting down inference threads...");
    // Joins both threads rather than sleeping and hoping. The join runs on a
    // blocking-safe thread because nothing else is scheduled on this runtime
    // once the accept loop has exited.
    tokio::task::spawn_blocking(move || engine.shutdown()).await?;
    // Unix sockets leave a filesystem entry behind; named pipes do not exist
    // once the last handle closes, so there is nothing to unlink on Windows.
    #[cfg(unix)]
    let _ = std::fs::remove_file(&socket_path);
    tracing::info!("Goodbye");

    Ok(())
}
