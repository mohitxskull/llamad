//! Benchmark harness: measures TTFT and decode throughput for one model.
//!
//! Usage:
//!   cargo run --release -p llamad --example bench -- <model.gguf> <n_repeat> [system_prompt]
//!
//! Prints one line per run: run=N prompt_tokens=P generated_tokens=G
//! ttft_ms=T decode_ms=D decode_tok_per_s=S
//!
//! `decode_tok_per_s` is steady-state throughput: it excludes the first token,
//! which is produced during the TTFT window rather than the decode window.
//! Then prints "SUMMARY <name>=<value>" lines for the report.

use std::time::Instant;

use llamad::client::Client;
use llamad::protocol::Request;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args[1] == "--help" || args[1] == "-h" {
        eprintln!(
            "Usage: {} <model.gguf> <n_repeat> [system_prompt]\n  n_repeat: how many identical requests to run",
            args[0]
        );
        std::process::exit(2);
    }
    let model = &args[1];
    let n_repeat: usize = args[2].parse()?;
    if n_repeat == 0 {
        eprintln!("n_repeat must be >= 1");
        std::process::exit(2);
    }
    let system = args.get(3).cloned();

    let client = Client::new(model)?;
    let prompt = "Explain, in one short paragraph, what the Fermi paradox is.";

    let mut runs = Vec::new();
    for run in 0..n_repeat {
        let mut req = Request::new(prompt).with_max_tokens(200).with_stream(true);
        if let Some(sys) = &system {
            req = req.with_system(sys.clone());
        }
        let started = Instant::now();
        let mut stream = client.complete_stream(req)?;
        let _ = stream.next_token(); // blocks until the first token
        let ttft_ms = started.elapsed().as_millis();
        let decode_started = Instant::now();
        while let Some(_t) = stream.next_token() {}
        let decode_ms = decode_started.elapsed().as_millis();
        let result = stream.into_result()?;
        // The first token was produced during the TTFT window, not the decode
        // window, so it must not be counted against `decode_ms`. Including it
        // overstates throughput by roughly 1/N — small at N=200, but these
        // numbers are quoted in the README.
        let decoded_tokens = result.generated_tokens.saturating_sub(1);
        println!(
            "RUN run={} prompt_tokens={} generated_tokens={} ttft_ms={} decode_ms={} decode_tok_per_s={}",
            run,
            result.prompt_tokens,
            result.generated_tokens,
            ttft_ms,
            decode_ms,
            decode_tok_per_s(decode_ms, decoded_tokens)
        );
        runs.push((ttft_ms, result.prompt_tokens, decoded_tokens, decode_ms));
    }

    let n = runs.len() as u64;
    let ttft_avg = runs.iter().map(|r| r.0 as u64).sum::<u64>() / n;
    let tps_avg = runs.iter().map(|r| decode_tok_per_s(r.3, r.2)).sum::<f32>() / n as f32;
    let tps_first = decode_tok_per_s(runs[0].3, runs[0].2);
    let tps_last = decode_tok_per_s(runs.last().unwrap().3, runs.last().unwrap().2);
    println!("SUMMARY model={model} n_repeat={n_repeat}");
    println!("SUMMARY ttft_ms_avg={ttft_avg}");
    println!("SUMMARY decode_tok_per_s_avg={tps_avg}");
    println!("SUMMARY decode_tok_per_s_first={tps_first}");
    println!("SUMMARY decode_tok_per_s_last={tps_last}");
    Ok(())
}

/// Steady-state decode rate: tokens produced *within* the decode window over
/// that window. The first token belongs to TTFT, so callers pass
/// `generated_tokens - 1`.
fn decode_tok_per_s(decode_ms: u128, decoded_tokens: usize) -> f32 {
    if decode_ms == 0 {
        return 0.0;
    }
    decoded_tokens as f32 / (decode_ms as f32 / 1000.0)
}
