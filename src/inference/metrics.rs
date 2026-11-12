//! Operational and execution metrics for monitoring the engine.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Observable runtime metrics for an [`Engine`](super::Engine).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EngineMetrics {
    /// Total number of requests received by the engine.
    pub total_requests: u64,
    /// Requests successfully completed.
    pub completed_requests: u64,
    /// Requests failed due to errors.
    pub failed_requests: u64,
    /// Requests cancelled by client disconnection or dropped receivers.
    pub cancelled_requests: u64,
    /// Requests cancelled due to slow streaming consumer policy.
    pub slow_consumer_drops: u64,
    /// Requests rejected immediately due to saturated queues or slots (Busy).
    pub rejected_busy: u64,
    /// Currently queued requests waiting for preprocessing or slot allocation.
    pub queue_depth: usize,
    /// Maximum capacity of the command queue before backpressure rejection.
    pub max_queue: usize,
    /// Currently occupied inference slots.
    pub active_slots: usize,
    /// Maximum concurrent inference slots configured.
    pub total_slots: usize,
}

/// Internal atomic counters tracking engine execution metrics.
#[derive(Debug, Default)]
pub(crate) struct MetricsTracker {
    pub(crate) total_requests: AtomicU64,
    pub(crate) completed_requests: AtomicU64,
    pub(crate) failed_requests: AtomicU64,
    pub(crate) cancelled_requests: AtomicU64,
    pub(crate) slow_consumer_drops: AtomicU64,
    pub(crate) rejected_busy: AtomicU64,
    pub(crate) queue_depth: AtomicUsize,
    pub(crate) active_slots: AtomicUsize,
}

impl MetricsTracker {
    /// Take a point-in-time snapshot of the metrics.
    #[must_use]
    pub(crate) fn snapshot(&self, max_queue: usize, total_slots: usize) -> EngineMetrics {
        EngineMetrics {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            completed_requests: self.completed_requests.load(Ordering::Relaxed),
            failed_requests: self.failed_requests.load(Ordering::Relaxed),
            cancelled_requests: self.cancelled_requests.load(Ordering::Relaxed),
            slow_consumer_drops: self.slow_consumer_drops.load(Ordering::Relaxed),
            rejected_busy: self.rejected_busy.load(Ordering::Relaxed),
            queue_depth: self.queue_depth.load(Ordering::Relaxed),
            max_queue,
            active_slots: self.active_slots.load(Ordering::Relaxed),
            total_slots,
        }
    }
}
