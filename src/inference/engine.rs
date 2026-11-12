use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use crate::config::InferenceConfig;
use crate::preprocess::preprocess_loop;
use crate::protocol::{InferCmd, LlamaError};

use super::metrics::{EngineMetrics, MetricsTracker};
use super::scheduler::inference_loop;

type Result<T> = std::result::Result<T, LlamaError>;

/// The lifecycle status of an inference engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineStatus {
    /// The model is loading and initializing in the background.
    Loading,
    /// The model and inference context are initialized and ready to serve requests.
    Ready,
    /// The model failed to load or initialize.
    Failed(String),
}

#[derive(Clone)]
pub(crate) struct EngineReadiness {
    state: Arc<(Mutex<EngineStatus>, Condvar)>,
    notify: Arc<tokio::sync::Notify>,
}

impl std::fmt::Debug for EngineReadiness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineReadiness")
            .field("status", &self.status())
            .finish()
    }
}

impl EngineReadiness {
    pub(crate) fn new() -> Self {
        Self {
            state: Arc::new((Mutex::new(EngineStatus::Loading), Condvar::new())),
            notify: Arc::new(tokio::sync::Notify::new()),
        }
    }

    pub(crate) fn status(&self) -> EngineStatus {
        let (lock, _) = &*self.state;
        let guard = match lock.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.clone()
    }

    pub(crate) fn set_ready(&self) {
        let (lock, cvar) = &*self.state;
        let mut guard = match lock.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        *guard = EngineStatus::Ready;
        cvar.notify_all();
        self.notify.notify_waiters();
    }

    pub(crate) fn set_failed(&self, reason: String) {
        let (lock, cvar) = &*self.state;
        let mut guard = match lock.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        *guard = EngineStatus::Failed(reason);
        cvar.notify_all();
        self.notify.notify_waiters();
    }

    pub(crate) fn wait_ready(&self) -> Result<()> {
        let (lock, cvar) = &*self.state;
        let mut guard = match lock.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        loop {
            match &*guard {
                EngineStatus::Ready => return Ok(()),
                EngineStatus::Failed(reason) => return Err(LlamaError::ModelLoad(reason.clone())),
                EngineStatus::Loading => {
                    guard = match cvar.wait(guard) {
                        Ok(g) => g,
                        Err(p) => p.into_inner(),
                    };
                }
            }
        }
    }

    pub(crate) async fn wait_ready_async(&self) -> Result<()> {
        loop {
            match self.status() {
                EngineStatus::Ready => return Ok(()),
                EngineStatus::Failed(reason) => return Err(LlamaError::ModelLoad(reason)),
                EngineStatus::Loading => {
                    self.notify.notified().await;
                }
            }
        }
    }
}

/// A loaded model and the two threads that serve it.
///
/// The engine owns its `llamad-inference` and `llamad-preprocess` threads and
/// joins them on drop, so an engine's resources are fully released before the
/// value goes away. Engines are independent: any number can exist at once,
/// each with its own model, context and slots.
///
/// Most users want [`Client`](crate::client::Client), which wraps an engine in
/// a request/response API. Use `Engine` directly only to hold [`InferCmd`]
/// senders yourself — the socket server does this so it can `await` results.
///
/// # Example
///
/// ```rust,no_run
/// # use llamad::inference::Engine;
/// # fn example() -> Result<(), llamad::protocol::LlamaError> {
/// let engine = Engine::start("model.gguf")?;
/// let tx = engine.sender().clone();
/// // ... hand `tx` to request handlers ...
/// engine.shutdown(); // or just drop it
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Engine {
    cmd_tx: Option<mpsc::SyncSender<InferCmd>>,
    shutdown: Arc<AtomicBool>,
    inference: Option<thread::JoinHandle<()>>,
    preprocess: Option<thread::JoinHandle<()>>,
    readiness: EngineReadiness,
    metrics: Arc<MetricsTracker>,
    config: InferenceConfig,
}

impl Engine {
    /// Load `model_path` and start the inference and preprocess threads.
    ///
    /// Returns as soon as the threads are spawned — the model loads in the
    /// background. A load failure surfaces via [`wait_ready`](Self::wait_ready),
    /// [`wait_ready_async`](Self::wait_ready_async), or on the first request as
    /// [`LlamaError::ModelLoad`].
    ///
    /// # Errors
    ///
    /// [`LlamaError::Io`] if either thread cannot be spawned.
    pub fn start(model_path: impl AsRef<Path>) -> Result<Self> {
        Self::start_with_config(model_path, InferenceConfig::from_env())
    }

    /// Load `model_path` with an explicit configuration, ignoring the
    /// `LLAMAD_*` environment variables.
    ///
    /// Use this when one process runs **more than one model** and they need
    /// different budgets — a small router model and a large reasoning model
    /// want different context sizes and thread counts, and the environment is
    /// process-global, so [`start`](Self::start) would give both the same
    /// settings.
    ///
    /// Values are clamped into usable ranges, so a hand-built config with a
    /// zero or absurd field degrades rather than panicking.
    ///
    /// # Errors
    ///
    /// [`LlamaError::Io`] if either thread cannot be spawned.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use llamad::{config::InferenceConfig, inference::Engine};
    /// # fn example() -> Result<(), llamad::protocol::LlamaError> {
    /// // Two engines, sized independently, sharing one machine's cores.
    /// let router = Engine::start_with_config("small.gguf", InferenceConfig {
    ///     n_ctx: 1024,
    ///     n_threads: 2,
    ///     ..Default::default()
    /// })?;
    /// let thinker = Engine::start_with_config("big.gguf", InferenceConfig {
    ///     n_ctx: 8192,
    ///     n_threads: 4,
    ///     ..Default::default()
    /// })?;
    /// # let _ = (router, thinker);
    /// # Ok(())
    /// # }
    /// ```
    pub fn start_with_config(
        model_path: impl AsRef<Path>,
        config: InferenceConfig,
    ) -> Result<Self> {
        // Clamped here rather than trusting the caller: the fields are public,
        // so a hand-built config can carry `n_slots: 0`, which would divide by
        // zero in `per_slot_budget`.
        let config = config.sane();

        // client → preprocess (raw requests), preprocess → inference (prepared),
        // inference → preprocess (Arc<LlamaModel> once loaded).
        let (cmd_tx, cmd_rx) = mpsc::sync_channel(config.max_queue);
        let (prepared_tx, prepared_rx) = mpsc::sync_channel(config.max_queue);
        let (model_tx, model_rx) = mpsc::channel();

        let readiness = EngineReadiness::new();
        let infer_readiness = readiness.clone();
        let metrics = Arc::new(MetricsTracker::default());
        let infer_metrics = Arc::clone(&metrics);
        let prep_metrics = Arc::clone(&metrics);

        let path = model_path.as_ref().to_owned();
        let infer_config = config.clone();
        let prep_config = config.clone();
        let shutdown = Arc::new(AtomicBool::new(false));
        let infer_shutdown = Arc::clone(&shutdown);
        let inference = thread::Builder::new()
            .name("llamad-inference".into())
            .spawn(move || {
                inference_loop(
                    &path,
                    prepared_rx,
                    model_tx,
                    infer_config,
                    infer_shutdown,
                    infer_readiness,
                    infer_metrics,
                )
            })
            .map_err(LlamaError::Io)?;

        let preprocess = thread::Builder::new()
            .name("llamad-preprocess".into())
            .spawn(move || {
                preprocess_loop(cmd_rx, model_rx, prepared_tx, prep_config, prep_metrics)
            })
            .map_err(LlamaError::Io)?;

        Ok(Engine {
            cmd_tx: Some(cmd_tx),
            shutdown,
            inference: Some(inference),
            preprocess: Some(preprocess),
            readiness,
            metrics,
            config,
        })
    }

    /// The command channel. Clone it to hand senders to request handlers.
    ///
    /// # Panics
    ///
    /// If called after [`Engine::shutdown`], which consumes the engine — so
    /// this cannot happen through the public API.
    // Unlike the utf-8 case above, this one cannot degrade to an error without
    // changing the return type, and `Option<&Sender>` would push an
    // always-`Some` unwrap onto every caller instead of removing it. `shutdown`
    // takes `self` by value, so no `&self` can outlive the `None` state.
    #[allow(
        clippy::expect_used,
        reason = "unreachable: `cmd_tx` is only taken by `stop`, which runs behind `shutdown(self)` or `drop`"
    )]
    pub fn sender(&self) -> &mpsc::SyncSender<InferCmd> {
        self.cmd_tx.as_ref().expect("sender taken only on shutdown")
    }

    /// Submit a command to the bounded preprocessing queue.
    ///
    /// # Errors
    ///
    /// - [`LlamaError::Busy`] if the pending command queue is full.
    /// - [`LlamaError::ModelLoad`] if model loading failed during startup.
    /// - [`LlamaError::InferenceCrashed`] if the engine threads have exited.
    pub fn send(&self, cmd: InferCmd) -> Result<()> {
        self.metrics.total_requests.fetch_add(1, Ordering::Relaxed);
        self.metrics.queue_depth.fetch_add(1, Ordering::Relaxed);
        match self.sender().try_send(cmd) {
            Ok(()) => Ok(()),
            Err(mpsc::TrySendError::Full(_)) => {
                self.metrics.queue_depth.fetch_sub(1, Ordering::Relaxed);
                self.metrics.rejected_busy.fetch_add(1, Ordering::Relaxed);
                Err(LlamaError::Busy)
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                self.metrics.queue_depth.fetch_sub(1, Ordering::Relaxed);
                self.metrics.failed_requests.fetch_add(1, Ordering::Relaxed);
                if let EngineStatus::Failed(reason) = self.readiness.status() {
                    Err(LlamaError::ModelLoad(reason))
                } else {
                    Err(LlamaError::InferenceCrashed)
                }
            }
        }
    }

    /// The current lifecycle status of the engine.
    #[must_use]
    pub fn status(&self) -> EngineStatus {
        self.readiness.status()
    }

    /// Block until the model is loaded and ready, or return an error if startup failed.
    pub fn wait_ready(&self) -> Result<()> {
        self.readiness.wait_ready()
    }

    /// Await until the model is loaded and ready, or return an error if startup failed.
    pub async fn wait_ready_async(&self) -> Result<()> {
        self.readiness.wait_ready_async().await
    }

    /// Return a point-in-time snapshot of the engine's operational metrics.
    #[must_use]
    pub fn metrics(&self) -> EngineMetrics {
        self.metrics
            .snapshot(self.config.max_queue, self.config.n_slots)
    }

    /// Cancel every in-flight request and join both threads.
    ///
    /// Equivalent to dropping the engine, but explicit and able to block
    /// visibly at a chosen point. Active slots' clients receive
    /// [`LlamaError::InferenceCrashed`].
    pub fn shutdown(mut self) {
        self.stop();
    }

    /// An engine backed by a caller-driven channel instead of real threads,
    /// so the client API can be tested without loading a model.
    #[cfg(test)]
    pub(crate) fn from_sender(cmd_tx: mpsc::SyncSender<InferCmd>) -> Self {
        let readiness = EngineReadiness::new();
        readiness.set_ready();
        let config = InferenceConfig::from_env().sane();
        Engine {
            cmd_tx: Some(cmd_tx),
            shutdown: Arc::new(AtomicBool::new(false)),
            inference: None,
            preprocess: None,
            readiness,
            metrics: Arc::new(MetricsTracker::default()),
            config,
        }
    }

    /// Signal both threads and join them. Idempotent.
    fn stop(&mut self) {
        // Order matters. The flag is what a busy inference thread notices
        // between decode steps; the message is what unblocks a preprocess
        // thread parked on an empty channel. Neither alone covers both.
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(tx) = self.cmd_tx.take() {
            let _ = tx.send(InferCmd::Shutdown);
            // Drop our sender so the preprocess loop's `for cmd in cmd_rx`
            // terminates even if outstanding clones are also dropped.
            drop(tx);
        }
        // Preprocess first: it forwards the shutdown downstream, after which
        // the inference thread sees either the message or the flag.
        if let Some(h) = self.preprocess.take()
            && h.join().is_err()
        {
            tracing::error!("preprocess thread panicked");
        }
        if let Some(h) = self.inference.take()
            && h.join().is_err()
        {
            tracing::error!("inference thread panicked");
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Load a model and start its inference threads.
///
/// Thin wrapper over [`Engine::start`]. The returned engine must be kept alive
/// for as long as requests are in flight — dropping it cancels them and joins
/// the threads.
pub fn start_inference(model_path: impl AsRef<Path>) -> Result<Engine> {
    Engine::start(model_path)
}
