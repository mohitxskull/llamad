//! Concurrent multi-slot load and stress benchmark.
//!
//! Spawns multiple concurrent tasks submitting requests across multiple slots,
//! validating continuous batching, bounded queue backpressure, and runtime metrics.
//!
//! Usage:
//!   cargo run --release -p llamad --example concurrent_load -- <model.gguf> [concurrent_clients] [requests_per_client]

use std::sync::Arc;
use std::time::Instant;

use llamad::client::Client;
use llamad::config::InferenceConfig;
use llamad::protocol::Request;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        eprintln!(
            "Usage: {} <model.gguf> [concurrent_clients] [requests_per_client]",
            args[0]
        );
        std::process::exit(2);
    }

    let model_path = &args[1];
    let concurrency: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let requests_per_client: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);

    println!(
        "Starting concurrent load test: {} clients x {} requests (total {}) against {}",
        concurrency,
        requests_per_client,
        concurrency * requests_per_client,
        model_path
    );

    let config = InferenceConfig {
        n_slots: concurrency,
        max_queue: concurrency * 4,
        ..InferenceConfig::from_env()
    };

    let client = Arc::new(Client::with_config(model_path, config)?);
    println!(
        "Model loaded successfully. Initial metrics: {:?}",
        client.metrics()
    );

    let started = Instant::now();
    let mut tasks = Vec::new();

    for client_id in 0..concurrency {
        let client = Arc::clone(&client);
        let task = tokio::spawn(async move {
            let mut latencies = Vec::new();
            for req_id in 0..requests_per_client {
                let req_start = Instant::now();
                let prompt = format!("Count from {} to {}: ", req_id, req_id + 5);
                let request = Request::new(&prompt)
                    .with_max_tokens(16)
                    .with_temperature(0.0)
                    .with_request_id(format!("client-{client_id}-req-{req_id}"));

                match client.complete_async(request).await {
                    Ok(res) => {
                        latencies.push(req_start.elapsed());
                        println!(
                            "[client {}] req {} done: generated {} tokens in {:?}",
                            client_id,
                            req_id,
                            res.generated_tokens,
                            req_start.elapsed()
                        );
                    }
                    Err(e) => {
                        eprintln!("[client {}] req {} failed: {:?}", client_id, req_id, e);
                    }
                }
            }
            latencies
        });
        tasks.push(task);
    }

    let mut all_latencies = Vec::new();
    for t in tasks {
        if let Ok(lats) = t.await {
            all_latencies.extend(lats);
        }
    }

    let elapsed = started.elapsed();
    let metrics = client.metrics();

    println!("\n=== Load Test Summary ===");
    println!("Elapsed time:       {:?}", elapsed);
    println!("Total completed:    {}", metrics.completed_requests);
    println!("Total failed:       {}", metrics.failed_requests);
    println!("Total cancelled:    {}", metrics.cancelled_requests);
    println!("Slow drops:         {}", metrics.slow_consumer_drops);
    println!("Rejected busy:      {}", metrics.rejected_busy);
    println!("Queue depth:        {}", metrics.queue_depth);
    println!("Active slots:       {}", metrics.active_slots);

    if !all_latencies.is_empty() {
        let avg_lat =
            all_latencies.iter().sum::<std::time::Duration>() / all_latencies.len() as u32;
        println!("Average latency:    {:?}", avg_lat);
    }

    Ok(())
}
