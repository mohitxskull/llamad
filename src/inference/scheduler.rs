use std::ffi::CStr;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use llama_cpp_4::prelude::*;

use crate::config::InferenceConfig;
use crate::protocol::{LlamaError, PreparedCmd};

use super::cache::{abandon_slot, fallback_to_full_prefill, retire_slot, select_slot};
use super::engine::EngineReadiness;
use super::metrics::MetricsTracker;
use super::sampler::build_sampler;
use super::slot::{Channels, Emit, Slot, create_slot, reject};

type Result<T> = std::result::Result<T, LlamaError>;

/// The process-wide llama.cpp backend, initialized at most once.
fn get_backend() -> Result<&'static LlamaBackend> {
    static BACKEND: std::sync::OnceLock<std::result::Result<LlamaBackend, String>> =
        std::sync::OnceLock::new();
    BACKEND
        .get_or_init(|| LlamaBackend::init().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| LlamaError::ModelLoad(format!("llama backend init failed: {e}")))
}

pub(crate) fn inference_loop(
    model_path: &Path,
    rx: mpsc::Receiver<PreparedCmd>,
    model_tx: mpsc::Sender<Arc<LlamaModel>>,
    config: InferenceConfig,
    shutdown: Arc<AtomicBool>,
    readiness: EngineReadiness,
    metrics: Arc<MetricsTracker>,
) {
    let backend = match get_backend() {
        Ok(b) => b,
        Err(e) => {
            let msg = format!("{e}");
            tracing::error!("{msg}");
            readiness.set_failed(msg);
            return;
        }
    };

    unsafe extern "C" fn log_callback(
        level: llama_cpp_sys_4::ggml_log_level,
        text: *const std::os::raw::c_char,
        _user_data: *mut std::os::raw::c_void,
    ) {
        let msg = if !text.is_null() {
            // SAFETY: llama.cpp guarantees `text` points to a valid, null-terminated
            // C string for the duration of the log callback when `!text.is_null()`.
            unsafe { CStr::from_ptr(text) }
                .to_string_lossy()
                .into_owned()
        } else {
            String::new()
        };
        if msg.trim().is_empty() {
            return;
        }
        match level {
            llama_cpp_sys_4::GGML_LOG_LEVEL_ERROR => tracing::error!("{msg}"),
            llama_cpp_sys_4::GGML_LOG_LEVEL_WARN => tracing::warn!("{msg}"),
            _ => tracing::debug!("{msg}"),
        }
    }
    // SAFETY: `log_callback` is a `'static` fn item and the user data pointer
    // is null, so both remain valid for the life of the process.
    unsafe {
        llama_cpp_4::log_set(Some(log_callback), std::ptr::null_mut());
    }

    let mut model_params = LlamaModelParams::default();
    if !config.kv_cache {
        model_params = model_params.with_n_gpu_layers(0);
    }

    if !model_path.exists() {
        let msg = format!("model file does not exist: {}", model_path.display());
        tracing::error!("{msg}");
        readiness.set_failed(msg);
        return;
    }

    let load_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        LlamaModel::load_from_file(backend, model_path, &model_params)
    }));
    let model = match load_res {
        Ok(Ok(m)) => Arc::new(m),
        Ok(Err(e)) => {
            let msg = format!("failed to load model {}: {e}", model_path.display());
            tracing::error!("{msg}");
            readiness.set_failed(msg);
            return;
        }
        Err(panic_err) => {
            let panic_msg = if let Some(s) = panic_err.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = panic_err.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            let msg = format!(
                "model loader panicked for {}: {panic_msg}",
                model_path.display()
            );
            tracing::error!("{msg}");
            readiness.set_failed(msg);
            return;
        }
    };

    if model_tx.send(Arc::clone(&model)).is_err() {
        let msg = "preprocess thread disconnected before receiving model".to_string();
        tracing::error!("{msg}");
        readiness.set_failed(msg);
        return;
    }

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(config.n_ctx))
        .with_n_seq_max(config.n_slots as u32)
        .with_n_threads(config.n_threads as i32)
        .with_n_threads_batch(config.n_threads_batch as i32)
        .with_n_batch(config.n_ctx);

    let mut ctx = match model.new_context(backend, ctx_params) {
        Ok(c) => c,
        Err(e) => {
            let msg = format!("failed to create llama context: {e}");
            tracing::error!("{msg}");
            readiness.set_failed(msg);
            return;
        }
    };

    let kv_rewind = super::cache::probe_kv_rewind(&model, &mut ctx);
    let reuse = super::cache::reuse_allowed(config.kv_cache, kv_rewind);
    if config.kv_cache && !kv_rewind {
        tracing::warn!(
            "KV prefix reuse disabled: partial KV rewind unsupported by this model; full prefill per request"
        );
    }

    let batch = LlamaBatch::new(config.n_ctx as usize, 1);
    let n_slots = config.n_slots;
    let per_slot_budget = config.per_slot_budget();

    let mut sched = Scheduler {
        model: &model,
        ctx,
        slots: (0..n_slots).map(|_| None).collect(),
        slot_cache: (0..n_slots).map(|_| Vec::new()).collect(),
        batch,
        slot_batch_idx: vec![None; n_slots],
        taken: Vec::with_capacity(n_slots),
        n_slots,
        per_slot_budget,
        reuse,
        metrics,
    };

    readiness.set_ready();
    tracing::info!("inference engine initialized and ready");

    loop {
        if shutdown.load(Ordering::Relaxed) {
            tracing::info!(
                "Shutdown flag set, draining {} active slots",
                sched.active_count()
            );
            sched.cancel_all(LlamaError::InferenceCrashed);
            break;
        }

        match sched.admit_pending(&rx) {
            Admit::Continue => {}
            Admit::Restart => continue,
            Admit::Stop => break,
        }

        sched.step();
    }
}

/// What the admission phase decided the main loop should do next.
enum Admit {
    Continue,
    Restart,
    Stop,
}

pub(crate) struct Scheduler<'a> {
    pub(crate) model: &'a LlamaModel,
    pub(crate) ctx: LlamaContext<'a>,
    pub(crate) slots: Vec<Option<Slot>>,
    pub(crate) slot_cache: Vec<Vec<LlamaToken>>,
    pub(crate) batch: LlamaBatch,
    pub(crate) slot_batch_idx: Vec<Option<usize>>,
    pub(crate) taken: Vec<(usize, Slot)>,
    pub(crate) n_slots: usize,
    pub(crate) per_slot_budget: u32,
    pub(crate) reuse: bool,
    pub(crate) metrics: Arc<MetricsTracker>,
}

impl Scheduler<'_> {
    pub(crate) fn active_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    pub(crate) fn abandon(&mut self, slot: &mut Slot, seq_id: usize, err: LlamaError) {
        self.metrics.active_slots.fetch_sub(1, Ordering::Relaxed);
        match &err {
            LlamaError::Disconnected => {
                self.metrics
                    .cancelled_requests
                    .fetch_add(1, Ordering::Relaxed);
            }
            _ => {
                self.metrics.failed_requests.fetch_add(1, Ordering::Relaxed);
            }
        }
        abandon_slot(&mut self.ctx, slot, &mut self.slot_cache, seq_id, err);
    }

    pub(crate) fn retire(&mut self, slot: &mut Slot, seq_id: usize) {
        self.metrics.active_slots.fetch_sub(1, Ordering::Relaxed);
        self.metrics
            .completed_requests
            .fetch_add(1, Ordering::Relaxed);
        retire_slot(
            &mut self.ctx,
            slot,
            &mut self.slot_cache,
            seq_id,
            self.reuse,
            self.per_slot_budget,
        );
    }

    pub(crate) fn cancel_all(&mut self, err: LlamaError) {
        let active = self.active_count();
        self.metrics
            .active_slots
            .fetch_sub(active, Ordering::Relaxed);
        self.metrics
            .cancelled_requests
            .fetch_add(active as u64, Ordering::Relaxed);
        super::cache::cancel_all(&mut self.slots, &mut self.slot_cache, err);
    }

    fn admit_pending(&mut self, rx: &mpsc::Receiver<PreparedCmd>) -> Admit {
        while self.slots.iter().any(|s| s.is_none()) {
            match rx.try_recv() {
                Ok(PreparedCmd::Shutdown) => {
                    tracing::info!(
                        "Shutdown received, draining {} active slots",
                        self.active_count()
                    );
                    self.cancel_all(LlamaError::InferenceCrashed);
                    return Admit::Stop;
                }
                Ok(cmd) => self.admit_one(cmd),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return Admit::Stop,
            }
        }

        if self.slots.iter().all(|s| s.is_none()) {
            match rx.recv() {
                Ok(PreparedCmd::Shutdown) => {
                    tracing::info!("Shutdown received");
                    return Admit::Stop;
                }
                Ok(cmd) => self.admit_one(cmd),
                Err(_) => return Admit::Stop,
            }
            return Admit::Restart;
        }

        Admit::Continue
    }

    fn admit_one(&mut self, cmd: PreparedCmd) {
        let req_id_str = match &cmd {
            PreparedCmd::Run { req, .. } | PreparedCmd::RunStream { req, .. } => {
                req.request_id.as_deref().unwrap_or("-")
            }
            PreparedCmd::Shutdown => "-",
        };
        let _span = tracing::info_span!("admit", request_id = %req_id_str).entered();

        let (req, channels) = match cmd {
            PreparedCmd::Run { req, resp } => (
                req,
                Channels {
                    resp: Some(resp),
                    ..Default::default()
                },
            ),
            PreparedCmd::RunStream {
                req,
                token_tx,
                done_tx,
            } => (
                req,
                Channels {
                    resp: None,
                    token_tx: Some(token_tx),
                    done_tx,
                },
            ),
            PreparedCmd::Shutdown => {
                tracing::warn!("Shutdown received outside of main loop");
                return;
            }
        };

        let sampler = match build_sampler(self.model, req.sampling, req.grammar.as_ref()) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("sampler construction failed: {e}");
                self.metrics.queue_depth.fetch_sub(1, Ordering::Relaxed);
                self.metrics.failed_requests.fetch_add(1, Ordering::Relaxed);
                reject(e, channels);
                return;
            }
        };

        let idx = match select_slot(&self.slots, &self.slot_cache, &req.tokens) {
            Some(i) => i,
            None => {
                tracing::warn!("All slots full, dropping request");
                self.metrics.queue_depth.fetch_sub(1, Ordering::Relaxed);
                self.metrics.rejected_busy.fetch_add(1, Ordering::Relaxed);
                reject(LlamaError::Busy, channels);
                return;
            }
        };
        let mut slot = create_slot(req.request_id, req.max_gen, sampler, req.stop, channels);
        let cached = std::mem::take(&mut self.slot_cache[idx]);
        let lcp = slot.begin_request(req.tokens, cached);
        if lcp > 0 {
            let rewound = self
                .ctx
                .clear_kv_cache_seq(Some(idx as u32), Some(lcp as u32), None)
                .unwrap_or(false);
            if rewound {
                tracing::debug!("slot {idx} reused {lcp} cached tokens");
            } else {
                fallback_to_full_prefill(&mut slot, lcp, idx);
                let _ = self.ctx.clear_kv_cache_seq(Some(idx as u32), None, None);
            }
        } else {
            let _ = self.ctx.clear_kv_cache_seq(Some(idx as u32), None, None);
        }
        self.metrics.queue_depth.fetch_sub(1, Ordering::Relaxed);
        self.metrics.active_slots.fetch_add(1, Ordering::Relaxed);
        self.slots[idx] = Some(slot);
    }

    fn step(&mut self) {
        self.build_batch();

        if self.batch.n_tokens() == 0 {
            let mut taken = std::mem::take(&mut self.taken);
            for (seq_id, mut slot) in taken.drain(..) {
                self.abandon(
                    &mut slot,
                    seq_id,
                    LlamaError::Inference(
                        "internal: empty batch after building from active slots".into(),
                    ),
                );
            }
            self.taken = taken;
            return;
        }

        if let Err(e) = self.ctx.decode(&mut self.batch) {
            let mut taken = std::mem::take(&mut self.taken);
            for (seq_id, mut slot) in taken.drain(..) {
                self.abandon(&mut slot, seq_id, LlamaError::Inference(e.to_string()));
            }
            self.taken = taken;
            return;
        }

        self.harvest();
    }

    fn build_batch(&mut self) {
        self.batch.clear();
        self.slot_batch_idx.fill(None);
        self.taken.clear();

        for seq_id in 0..self.n_slots {
            let mut slot = match self.slots[seq_id].take() {
                Some(s) => s,
                None => continue,
            };
            if slot.is_cancelled() {
                self.abandon(&mut slot, seq_id, LlamaError::Disconnected);
                continue;
            }
            let sid = seq_id as i32;

            if slot.prompt_phase {
                let mut ok = true;
                let mut reused = slot.cache_tokens.len() as i32;
                if reused > 0 && reused as usize == slot.n_prompt {
                    let rewound = self
                        .ctx
                        .clear_kv_cache_seq(
                            Some(seq_id as u32),
                            Some((slot.n_prompt - 1) as u32),
                            None,
                        )
                        .unwrap_or(false);
                    if rewound {
                        tracing::debug!(
                            "slot {seq_id}: zero-token prefill, anchor at {}",
                            slot.n_prompt - 1
                        );
                    } else {
                        fallback_to_full_prefill(&mut slot, reused as usize, seq_id);
                        let _ = self.ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
                        reused = 0;
                    }
                }
                for (i, &tok) in slot.prompt_tokens.iter().enumerate().skip(reused as usize) {
                    let idx = self.batch.n_tokens() as usize;
                    let is_last = i == slot.n_prompt - 1;
                    if self
                        .batch
                        .add(tok, slot.kv_pos + i as i32 - reused, &[sid], is_last)
                        .is_err()
                    {
                        tracing::error!("Batch overflow (prompt seq {seq_id})");
                        self.abandon(
                            &mut slot,
                            seq_id,
                            LlamaError::Inference("batch overflow during prompt processing".into()),
                        );
                        ok = false;
                        break;
                    }
                    if is_last {
                        self.slot_batch_idx[seq_id] = Some(idx);
                    }
                }
                if ok && self.slot_batch_idx[seq_id].is_none() && reused > 0 {
                    let idx = self.batch.n_tokens() as usize;
                    let last = (slot.n_prompt - 1) as i32;
                    if self
                        .batch
                        .add(slot.prompt_tokens[last as usize], last, &[sid], true)
                        .is_err()
                    {
                        tracing::error!("Batch overflow (prompt seq {seq_id})");
                        self.abandon(
                            &mut slot,
                            seq_id,
                            LlamaError::Inference("batch overflow during prompt processing".into()),
                        );
                        ok = false;
                    } else {
                        self.slot_batch_idx[seq_id] = Some(idx);
                    }
                }
                if ok {
                    slot.finish_prefill();
                    self.taken.push((seq_id, slot));
                }
            } else if let Some(token) = slot.pending_token {
                let idx = self.batch.n_tokens() as usize;
                if self.batch.add(token, slot.kv_pos, &[sid], true).is_ok() {
                    self.slot_batch_idx[seq_id] = Some(idx);
                    slot.kv_pos += 1;
                    self.taken.push((seq_id, slot));
                } else {
                    tracing::error!("Batch overflow (gen seq {seq_id})");
                    self.slot_cache[seq_id] = std::mem::take(&mut slot.cache_tokens);
                    self.metrics.active_slots.fetch_sub(1, Ordering::Relaxed);
                    self.metrics.failed_requests.fetch_add(1, Ordering::Relaxed);
                    slot.cancel(LlamaError::Inference(
                        "batch overflow during generation".into(),
                    ));
                }
            } else {
                self.slots[seq_id] = Some(slot);
            }
        }
    }

    fn harvest(&mut self) {
        let mut taken = std::mem::take(&mut self.taken);
        for (seq_id, mut slot) in taken.drain(..) {
            if slot.is_cancelled() {
                self.abandon(&mut slot, seq_id, LlamaError::Disconnected);
                continue;
            }
            let batch_idx = match self.slot_batch_idx[seq_id] {
                Some(i) => i as i32,
                None => {
                    tracing::error!("slot {seq_id}: no batch index after prefill");
                    self.abandon(
                        &mut slot,
                        seq_id,
                        LlamaError::Inference("internal: no batch index after prefill".into()),
                    );
                    continue;
                }
            };

            let token = slot.sampler.sample(&self.ctx, batch_idx);

            if self.model.is_eog_token(token) {
                if let Err(e) = slot.finish() {
                    self.abandon(&mut slot, seq_id, e);
                } else {
                    self.retire(&mut slot, seq_id);
                }
                continue;
            }

            slot.gen_count += 1;

            match slot.push_token(self.model, token) {
                Ok(Some(piece)) => match slot.push_text(&piece) {
                    Emit::Continue => {}
                    Emit::Stopped => {
                        if let Err(e) = slot.finish() {
                            self.abandon(&mut slot, seq_id, e);
                        } else {
                            self.retire(&mut slot, seq_id);
                        }
                        continue;
                    }
                    Emit::Disconnected => {
                        self.abandon(&mut slot, seq_id, LlamaError::Disconnected);
                        continue;
                    }
                    Emit::SlowConsumer => {
                        self.metrics
                            .slow_consumer_drops
                            .fetch_add(1, Ordering::Relaxed);
                        self.abandon(
                            &mut slot,
                            seq_id,
                            LlamaError::Inference(
                                "streaming token consumer too slow (buffer full)".into(),
                            ),
                        );
                        continue;
                    }
                },
                Ok(None) => {}
                Err(e) => {
                    tracing::error!("Detokenization error: {e}");
                    self.abandon(&mut slot, seq_id, e);
                    continue;
                }
            }

            if slot.gen_count as u32 >= slot.max_tokens
                || slot.kv_pos as u32 >= self.per_slot_budget
            {
                if let Err(e) = slot.finish() {
                    self.abandon(&mut slot, seq_id, e);
                } else {
                    self.retire(&mut slot, seq_id);
                }
                continue;
            }

            slot.pending_token = Some(token);
            self.slots[seq_id] = Some(slot);
        }
        self.taken = taken;
    }
}
