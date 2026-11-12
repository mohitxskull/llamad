use std::path::Path;

use llamad::client::Client;
use llamad::config::InferenceConfig;
use llamad::protocol::Request;

const MODEL: &str = "models/LFM3-1.7B-Q4_K_M.gguf";

fn model_present() -> bool {
    Path::new(MODEL).exists()
}

#[tokio::test]
async fn test_engine_metrics_lifecycle() {
    if !model_present() {
        eprintln!("skipping: {} not found", MODEL);
        return;
    }

    let config = InferenceConfig {
        n_slots: 2,
        max_queue: 16,
        ..InferenceConfig::from_env()
    };

    let client = Client::with_config(MODEL, config).expect("model load failed");
    let initial = client.metrics();

    assert_eq!(initial.total_requests, 0);
    assert_eq!(initial.completed_requests, 0);
    assert_eq!(initial.failed_requests, 0);
    assert_eq!(initial.cancelled_requests, 0);
    assert_eq!(initial.slow_consumer_drops, 0);
    assert_eq!(initial.rejected_busy, 0);
    assert_eq!(initial.queue_depth, 0);
    assert_eq!(initial.active_slots, 0);
    assert_eq!(initial.total_slots, 2);
    assert_eq!(initial.max_queue, 16);

    // Run one non-streaming completion
    let req = Request::new("say hi")
        .with_max_tokens(4)
        .with_temperature(0.0)
        .with_request_id("metrics-test-1");

    let res = client.complete_async(req).await;
    assert!(res.is_ok());

    let m1 = client.metrics();
    assert_eq!(m1.total_requests, 1);
    assert_eq!(m1.completed_requests, 1);
    assert_eq!(m1.failed_requests, 0);
    assert_eq!(m1.cancelled_requests, 0);
    assert_eq!(m1.queue_depth, 0);
    assert_eq!(m1.active_slots, 0);

    // Test cancellation metric increment
    let stream_req = Request::new("tell a long story")
        .with_max_tokens(100)
        .with_stream(true)
        .with_request_id("metrics-test-cancel");

    let stream = client.complete_stream(stream_req);
    assert!(stream.is_ok());
    // Drop the stream immediately before or after first token
    drop(stream);

    // Wait briefly for scheduler to retire/cancel
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let m2 = client.metrics();
    assert_eq!(m2.total_requests, 2);
    assert!(m2.cancelled_requests >= 1 || m2.completed_requests >= 2);
    assert_eq!(m2.queue_depth, 0);
    assert_eq!(m2.active_slots, 0);
}
