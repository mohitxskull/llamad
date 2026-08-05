//! The `llamad` daemon: serves the JSON/NDJSON protocol over a Unix socket.

use llamad::inference::Engine;
use llamad::server::{DEFAULT_SOCKET_PATH, bind_socket, handle_connection};

fn usage(program: &str) {
    eprintln!("Usage: {program} <model.gguf> [socket-path]");
    eprintln!();
    eprintln!("Arguments:");
    eprintln!("  <model.gguf>              Path to GGUF model file (required)");
    eprintln!("  [socket-path]             Unix socket path (default: {DEFAULT_SOCKET_PATH})");
    eprintln!();
    eprintln!("Environment:");
    eprintln!("  LLAMAD_N_SLOTS            Concurrent sequences (default 4)");
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

    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        usage(program);
        // A missing model path is a usage error, not a successful run.
        return if args.len() < 2 {
            std::process::exit(2);
        } else {
            Ok(())
        };
    }

    let model_path = &args[1];
    let socket_path = args
        .get(2)
        .map_or(DEFAULT_SOCKET_PATH, |s| s.as_str())
        .to_owned();

    let listener = bind_socket(&socket_path)?;
    let engine = Engine::start(model_path)?;

    tracing::info!("llamad listening on {socket_path}");

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("Shutdown signal received");
                break;
            }
            result = listener.accept() => {
                let (mut stream, _) = result?;
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
    let _ = std::fs::remove_file(&socket_path);
    tracing::info!("Goodbye");

    Ok(())
}
