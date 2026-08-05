//! The inference thread: model and context ownership, the slotted continuous
//! batching loop, and KV-prefix reuse.
//!
//! [`Engine`] is the entry point — it owns the `llamad-inference` and
//! `llamad-preprocess` threads for one loaded model and joins them on drop.

use std::ffi::CStr;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;

use llama_cpp_4::prelude::*;

use crate::preprocess::preprocess_loop;
use crate::protocol::{Grammar, InferCmd, InferResult, LlamaError, PreparedCmd, SamplingParams};

type Result<T> = std::result::Result<T, LlamaError>;

use crate::config::InferenceConfig;

// ── Slot ─────────────────────────────────────────────────────────────────────

/// Length of the longest common prefix of two token slices.
fn longest_common_prefix(a: &[LlamaToken], b: &[LlamaToken]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Largest index `<= i` that lies on a `char` boundary of `s`.
///
/// Stop-sequence bookkeeping slices `text` at offsets derived from byte
/// lengths, which can land inside a multi-byte character; `str` indexing
/// panics there. (`str::floor_char_boundary` is still unstable.)
fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// What a newly decoded fragment implies for the slot's lifetime.
#[derive(Debug, PartialEq, Eq)]
enum Emit {
    /// Nothing special — keep generating.
    Continue,
    /// A stop sequence completed. `text` is already truncated at its start.
    Stopped,
    /// The streaming client hung up.
    Disconnected,
}

struct Slot {
    prompt_tokens: Vec<LlamaToken>,
    n_prompt: usize,
    gen_count: usize,
    max_tokens: u32,
    kv_pos: i32,
    sampler: LlamaSampler,
    utf8_buf: Vec<u8>,
    text: String,
    pending_token: Option<LlamaToken>,
    prompt_phase: bool,
    resp: Option<tokio::sync::oneshot::Sender<Result<InferResult>>>,
    token_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
    done_tx: Option<tokio::sync::oneshot::Sender<std::result::Result<InferResult, LlamaError>>>,
    cache_tokens: Vec<LlamaToken>,
    /// Stop sequences, already stripped of empty entries by preprocessing.
    stop: Vec<String>,
    /// Longest stop sequence in bytes; 0 when there are none. Bounds both the
    /// search window and the streaming hold-back.
    stop_max_len: usize,
    /// Byte offset in `text` already handed to `token_tx`. Trails `text.len()`
    /// whenever the tail could still grow into a stop sequence.
    emitted: usize,
}

impl Slot {
    fn push_token(&mut self, model: &LlamaModel, token: LlamaToken) -> Result<Option<String>> {
        let raw = model
            .token_to_raw_bytes(token, Special::Plaintext)
            .map_err(|e| LlamaError::Inference(e.to_string()))?;
        self.push_raw_bytes(&raw)
    }

    /// Buffer raw detokenized bytes and return any complete UTF-8 text
    /// prefix. Partial multi-byte sequences stay buffered until the next
    /// push; permanently invalid sequences return an error.
    fn push_raw_bytes(&mut self, raw: &[u8]) -> Result<Option<String>> {
        self.utf8_buf.extend_from_slice(raw);

        let valid = match std::str::from_utf8(&self.utf8_buf) {
            Ok(_) => self.utf8_buf.len(),
            Err(e) if e.error_len().is_some() => {
                return Err(LlamaError::Inference(format!(
                    "invalid utf-8 sequence: {e}"
                )));
            }
            Err(e) => e.valid_up_to(),
        };

        if valid == 0 {
            return Ok(None);
        }
        let complete: Vec<u8> = self.utf8_buf.drain(..valid).collect();
        Ok(Some(
            String::from_utf8(complete).expect("prefix up to valid_up_to is valid utf-8"),
        ))
    }

    /// Byte offset of the earliest completed stop sequence, or `None`.
    ///
    /// Only the tail is searched: a match that completes with the `piece_len`
    /// bytes just appended cannot start earlier than
    /// `piece_len + stop_max_len - 1` back from the end, and everything before
    /// that was already scanned on a previous token.
    fn find_stop(&self, piece_len: usize) -> Option<usize> {
        if self.stop.is_empty() {
            return None;
        }
        let window = piece_len + self.stop_max_len.saturating_sub(1);
        let from = floor_char_boundary(&self.text, self.text.len().saturating_sub(window));
        self.stop
            .iter()
            .filter_map(|s| self.text[from..].find(s.as_str()).map(|i| from + i))
            .min()
    }

    /// How many trailing bytes must be withheld from the stream because they
    /// could still grow into a stop sequence.
    ///
    /// This is the length of the longest suffix of `text` that is a proper
    /// prefix of some stop sequence — zero when the tail cannot lead anywhere.
    /// Computing it per token rather than always withholding `stop_max_len - 1`
    /// bytes keeps streaming latency at zero for the overwhelmingly common
    /// case where the output looks nothing like a stop sequence.
    fn holdback(&self) -> usize {
        if self.stop.is_empty() {
            return 0;
        }
        let max = self.stop_max_len.saturating_sub(1);
        let start = floor_char_boundary(&self.text, self.text.len().saturating_sub(max));
        // Ascending cut means descending tail length, so the first hit is the
        // longest qualifying suffix.
        for cut in start..self.text.len() {
            if !self.text.is_char_boundary(cut) {
                continue;
            }
            let tail = &self.text[cut..];
            if self.stop.iter().any(|s| s.starts_with(tail)) {
                return self.text.len() - cut;
            }
        }
        0
    }

    /// Stream `text[emitted..end]` to a streaming client, advancing `emitted`.
    /// Returns `false` when that client has disconnected.
    fn flush_to(&mut self, end: usize) -> bool {
        if end <= self.emitted {
            return true;
        }
        let chunk = self.text[self.emitted..end].to_owned();
        self.emitted = end;
        if let Some(ref tx) = self.token_tx
            && tx.send(chunk).is_err()
        {
            return false;
        }
        true
    }

    /// Accumulate a generated fragment and release whatever is now safe to
    /// stream, honouring stop sequences.
    ///
    /// A stop sequence may span token boundaries, so the decision cannot be
    /// made per token: bytes that could still become one are withheld until
    /// the match either completes (generation ends, the stop text is dropped)
    /// or is ruled out (the bytes are released).
    fn push_text(&mut self, piece: &str) -> Emit {
        self.text.push_str(piece);

        if let Some(cut) = self.find_stop(piece.len()) {
            // Drop the stop sequence and anything after it, then release what
            // preceded it — that text is real output the client must still
            // see. `emitted <= cut` always holds here: every byte from `cut`
            // on was withheld by `holdback` as a growing prefix of this very
            // match, and `flush_to` no-ops if that were ever violated.
            self.text.truncate(cut);
            if !self.flush_to(cut) {
                return Emit::Disconnected;
            }
            return Emit::Stopped;
        }

        let safe = self.text.len() - self.holdback();
        if !self.flush_to(safe) {
            return Emit::Disconnected;
        }
        Emit::Continue
    }

    fn finish(&mut self) {
        // Release anything still withheld for a possible stop match:
        // generation ended without one, so the tail is ordinary output.
        let end = self.text.len();
        let _ = self.flush_to(end);
        let result = InferResult {
            text: std::mem::take(&mut self.text),
            prompt_tokens: self.n_prompt,
            generated_tokens: self.gen_count,
        };
        if let Some(resp) = self.resp.take() {
            let _ = resp.send(Ok(result.clone()));
        }
        if let Some(done) = self.done_tx.take() {
            let _ = done.send(Ok(result));
        }
        drop(self.token_tx.take());
    }

    fn cancel(&mut self, err: LlamaError) {
        if let Some(resp) = self.resp.take() {
            let _ = resp.send(Err(err.clone()));
        }
        if let Some(done) = self.done_tx.take() {
            let _ = done.send(Err(err));
        }
        drop(self.token_tx.take());
    }

    /// Prepare the slot for a new prompt, reusing the cached KV prefix when the
    /// new prompt shares tokens with the cached sequence.
    ///
    /// `cached` is the token history of the previous completion (taken from the
    /// slot cache array, see `inference_loop`); it is empty when the slot is
    /// fresh. On return the slot is ready for prefill starting at `kv_pos`
    /// (0 when nothing was reused). The caller must reconcile the context's KV
    /// afterwards: clear the tail from `kv_pos` when `lcp > 0`, or clear the
    /// whole sequence when the cache was non-empty but unused.
    fn begin_request(&mut self, tokens: Vec<LlamaToken>, cached: Vec<LlamaToken>) -> usize {
        self.prompt_tokens = tokens;
        self.n_prompt = self.prompt_tokens.len();
        self.kv_pos = 0;
        self.gen_count = 0;
        self.prompt_phase = true;
        self.pending_token = None;
        self.utf8_buf.clear();
        self.text.clear();

        let lcp = longest_common_prefix(&cached, &self.prompt_tokens);
        self.cache_tokens = if lcp > 0 {
            self.kv_pos = lcp as i32;
            let mut kept: Vec<LlamaToken> = cached;
            kept.truncate(lcp);
            kept
        } else {
            Vec::new()
        };
        lcp
    }

    /// Close the prefill phase: advance `kv_pos` past the prompt and fold the
    /// newly-prefilled tail into the cache mirror.
    ///
    /// Only the tail `prompt_tokens[reused..]` is drained — draining the whole
    /// prompt on top of the kept prefix would grow the mirror to
    /// `lcp + n_prompt` while `kv_pos == n_prompt`, breaking the
    /// `kv_pos == cache_tokens.len()` invariant (the corruption surfaces only
    /// on a *partial*-prefix prompt, the third request of a triple).
    fn finish_prefill(&mut self) {
        let reused = self.cache_tokens.len() as i32;
        // Invariant: kv_pos == cache_tokens.len() throughout prefill (kept by
        // begin_request / fallback_to_full_prefill).
        debug_assert_eq!(self.kv_pos as usize, reused as usize);
        self.kv_pos += self.n_prompt as i32 - reused;
        self.prompt_phase = false;
        self.cache_tokens
            .extend(self.prompt_tokens.drain(reused as usize..));
        self.prompt_tokens.clear();
    }
}

// ── Sampler chain ────────────────────────────────────────────────────────────

/// Parameterized description of one sampler in the decoding chain.
///
/// Data rather than opaque [`LlamaSampler`] handles so the chain's content
/// and order are unit-testable (the FFI handle cannot be introspected).
#[derive(Debug, Clone, Copy, PartialEq)]
enum SamplerKind {
    Penalties {
        penalty_last_n: i32,
        penalty_repeat: f32,
    },
    TopK {
        k: i32,
    },
    TopP {
        p: f32,
        min_keep: usize,
    },
    Temp {
        t: f32,
    },
    Dist {
        seed: u32,
    },
}

impl SamplerKind {
    fn into_sampler(self) -> LlamaSampler {
        match self {
            SamplerKind::Penalties {
                penalty_last_n,
                penalty_repeat,
            } => LlamaSampler::penalties_simple(penalty_last_n, penalty_repeat),
            SamplerKind::TopK { k } => LlamaSampler::top_k(k),
            SamplerKind::TopP { p, min_keep } => LlamaSampler::top_p(p, min_keep),
            SamplerKind::Temp { t } => LlamaSampler::temp(t),
            SamplerKind::Dist { seed } => LlamaSampler::dist(seed),
        }
    }
}

/// The decoding chain for a set of resolved parameters. Empty means greedy
/// decoding (no sampler chain), which ignores every parameter but temperature.
fn sampler_chain(params: SamplingParams) -> Vec<SamplerKind> {
    if params.temperature <= 0.0 {
        Vec::new()
    } else {
        vec![
            SamplerKind::Penalties {
                penalty_last_n: params.repeat_last_n,
                penalty_repeat: params.repeat_penalty,
            },
            SamplerKind::TopK { k: params.top_k },
            SamplerKind::TopP {
                p: params.top_p,
                min_keep: 1,
            },
            SamplerKind::Temp {
                t: params.temperature,
            },
            // Seeded per request. `RANDOM_SEED` (llama.cpp's
            // `LLAMA_DEFAULT_SEED`) makes llama.cpp draw a fresh seed, so two
            // identical requests do not return identical text — a fixed seed
            // here would make `temperature` unobservable across requests,
            // since each slot builds its own sampler.
            SamplerKind::Dist { seed: params.seed },
        ]
    }
}

/// Build a grammar sampler, turning llama.cpp's failure modes into an error.
///
/// [`LlamaSampler::grammar`] **panics** twice over: `CString::new` panics on a
/// null byte in either string, and llama.cpp returns a null pointer for a
/// grammar that does not parse, which the binding unwraps. Both are reachable
/// from untrusted socket input, and a panic on the inference thread would take
/// the engine down for every client — a malformed grammar from one caller must
/// not be a denial of service for the rest.
///
/// Catching the unwind is sound here: this is a constructor, so a panic means
/// llama.cpp allocated nothing and no state was mutated. The default panic
/// hook still prints to stderr before unwinding, so a rejected grammar is
/// noisy in the log even though it is handled. Requires `panic = "unwind"`
/// (the default); under `panic = "abort"` a bad grammar aborts the process.
pub(crate) fn build_grammar_sampler(
    model: &LlamaModel,
    grammar: &str,
    root: &str,
) -> Result<LlamaSampler> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        LlamaSampler::grammar(model, grammar, root)
    }))
    .map_err(|_| {
        LlamaError::Protocol(format!(
            "grammar failed to compile (start rule {root:?}); check the GBNF syntax"
        ))
    })
}

/// Assemble the sampler for one request.
///
/// # Errors
///
/// [`LlamaError::Protocol`] if a supplied grammar does not compile.
fn build_sampler(
    model: &LlamaModel,
    params: SamplingParams,
    grammar: Option<&Grammar>,
) -> Result<LlamaSampler> {
    let mut samplers: Vec<LlamaSampler> = Vec::new();

    // Grammar goes first, and the order is load-bearing. It masks disallowed
    // tokens to -inf so every later stage picks only from the allowed set.
    // Placed after top-k/top-p it could mask away the entire truncated
    // candidate set; placed after `Dist` it would do nothing at all, since the
    // token is already chosen.
    if let Some(g) = grammar {
        samplers.push(build_grammar_sampler(model, &g.text, &g.root)?);
    }

    let kinds = sampler_chain(params);
    // An empty chain means temperature <= 0, i.e. greedy decoding.
    let greedy = kinds.is_empty();
    samplers.extend(kinds.into_iter().map(SamplerKind::into_sampler));
    if greedy {
        // Greedy is the chain's *tail*, not the whole sampler: with a grammar
        // present the mask has to run first. A lone grammar sampler would
        // never select a token.
        samplers.push(LlamaSampler::greedy());
    }

    Ok(LlamaSampler::chain_simple(samplers))
}

// ── Slot creation ─────────────────────────────────────────────────────────────

/// Assemble a slot from a prepared request and an already-built sampler.
///
/// The sampler is built by the caller rather than here so that a grammar
/// failure can be reported to the client *before* the response channels are
/// moved into the slot.
fn create_slot(
    tokens: Vec<LlamaToken>,
    max_gen: u32,
    sampler: LlamaSampler,
    stop: Vec<String>,
    resp: Option<tokio::sync::oneshot::Sender<Result<InferResult>>>,
    token_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
    done_tx: Option<tokio::sync::oneshot::Sender<std::result::Result<InferResult, LlamaError>>>,
) -> Slot {
    let n_prompt = tokens.len();
    let stop_max_len = stop.iter().map(String::len).max().unwrap_or(0);

    Slot {
        prompt_tokens: tokens,
        n_prompt,
        gen_count: 0,
        max_tokens: max_gen,
        kv_pos: 0,
        sampler,
        utf8_buf: Vec::new(),
        // ~4 bytes per token is a typical average for these vocabularies; the
        // cap keeps a large max_tokens from over-reserving up front.
        text: String::with_capacity((max_gen as usize).min(512) * 4),
        pending_token: None,
        prompt_phase: true,
        resp,
        token_tx,
        done_tx,
        cache_tokens: Vec::new(),
        stop,
        stop_max_len,
        emitted: 0,
    }
}

// ── KV rewind probe ───────────────────────────────────────────────────────────

/// Empirical probe: can this context do PARTIAL KV rewind? Partial suffix
/// removal (`seq_rm` with p0 > 0, p1 = `INT_MAX`) is refused by
/// `llama_memory_recurrent` when the rollback bound fails (`1 <= rollback
/// <= n_rs_seq` must hold; llamad never sets `n_rs_seq`, which defaults to
/// 0 (llama-context.cpp)) and by DSV4 compressed caches, which do not
/// support partial removals at all. Hybrid/SSM models (LFM2, LFM2MOE,
/// QWEN35, QWEN35MOE) and DSV4 models therefore refuse it. MSA-strict
/// (`msa_strict_slots`) refuses only non-suffix removals; pure-attention
/// `llama_kv_cache` never refuses suffix trims. Note: `llm_arch_supports_rs_rollback`
/// does NOT gate `seq_rm` — it only clamps `n_rs_seq` at context creation.
/// Probe once at startup into scratch seq id 0 before any slot exists;
/// every path below full-clears seq 0 afterwards.
fn probe_kv_rewind(model: &LlamaModel, ctx: &mut LlamaContext) -> bool {
    let mut batch = LlamaBatch::new(8, 2);
    let probe_seq: i32 = 0; // scratch seq id; full-cleared after the probe
    // BOS-less vocabularies report LLAMA_TOKEN_NULL (-1), which the batch
    // validator rejects; fall back to token 0.
    let bos = model.token_bos();
    let tok = if bos.0 == -1 { LlamaToken::new(0) } else { bos };
    let ok_add = batch.add(tok, 0, &[probe_seq], false).is_ok()
        && batch.add(tok, 1, &[probe_seq], true).is_ok();
    if !ok_add || ctx.decode(&mut batch).is_err() {
        let _ = ctx.clear_kv_cache_seq(Some(probe_seq as u32), None, None);
        return false; // decode itself failed — degrade to be safe
    }
    // Attempt to remove [1, ...): works only if partial rewind is supported.
    let rewinds = ctx
        .clear_kv_cache_seq(Some(probe_seq as u32), Some(1), None)
        .unwrap_or(false);
    // Always leave the scratch seq clean.
    let _ = ctx.clear_kv_cache_seq(Some(probe_seq as u32), None, None);
    rewinds
}

/// Whether a completed sequence should be retained for prefix reuse on the
/// next request into its slot. Reuse engages only when the env knob is on
/// (`LLAMAD_KV_CACHE`) AND the startup probe found the model supports
/// partial KV rewind.
fn reuse_allowed(kv_cache: bool, kv_rewind: bool) -> bool {
    kv_cache && kv_rewind
}

/// Handle a refused partial KV rewind: warn loudly and reset the slot to a
/// full prefill (the caller must also full-clear the sequence in the
/// context — not reachable from the unit tests, which exercise only the
/// slot-side reset).
fn fallback_to_full_prefill(slot: &mut Slot, lcp: usize, idx: usize) {
    tracing::warn!("slot {idx}: partial KV rewind refused ({lcp}), falling back to full prefill");
    slot.cache_tokens.clear();
    slot.kv_pos = 0;
}

// ── Command dispatch ──────────────────────────────────────────────────────────

/// Choose which free slot should serve `tokens`.
///
/// Prefers the free slot whose retained KV prefix shares the most tokens with
/// the incoming prompt. Selection must be prefix-aware because the reuse
/// machinery below only ever compares the prompt against `slot_cache[idx]`:
/// with plain first-free selection, a request that arrives while slot 0 is
/// busy lands on slot 1 and full-prefills, even when slot 2 sat idle holding
/// an exact prefix of it. That is the common shape for a repeated system
/// prompt under any concurrency at all.
///
/// Ties — including the all-zero case where no slot matches — resolve to the
/// lowest free index, so cold-start and single-slot behaviour is identical to
/// first-free selection.
fn select_slot(
    slots: &[Option<Slot>],
    slot_cache: &[Vec<LlamaToken>],
    tokens: &[LlamaToken],
) -> Option<usize> {
    slots
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_none())
        .map(|(i, _)| (i, longest_common_prefix(&slot_cache[i], tokens)))
        // `max_by_key` keeps the *last* maximum, so `Reverse(i)` is what makes
        // the lowest index win a tie instead of the highest.
        .max_by_key(|&(i, lcp)| (lcp, std::cmp::Reverse(i)))
        .map(|(i, _)| i)
}

/// Report `err` to whichever response channels a request carried.
fn reject(
    err: LlamaError,
    resp: Option<tokio::sync::oneshot::Sender<Result<InferResult>>>,
    token_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
    done_tx: Option<tokio::sync::oneshot::Sender<std::result::Result<InferResult, LlamaError>>>,
) {
    if let Some(resp) = resp {
        let _ = resp.send(Err(err.clone()));
    }
    if let Some(done) = done_tx {
        let _ = done.send(Err(err));
    }
    drop(token_tx);
}

fn fill_empty_slot(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    slots: &mut [Option<Slot>],
    slot_cache: &mut [Vec<LlamaToken>],
    cmd: PreparedCmd,
) {
    // Destructured by value, and before slot selection: cloning `req.tokens`
    // would copy the whole tokenized prompt on every request, and
    // `select_slot` needs them to score each free slot's cached prefix.
    let (req, resp, token_tx, done_tx) = match cmd {
        PreparedCmd::Run { req, resp } => (req, Some(resp), None, None),
        PreparedCmd::RunStream {
            req,
            token_tx,
            done_tx,
        } => (req, None, Some(token_tx), done_tx),
        PreparedCmd::Shutdown => {
            tracing::warn!("Shutdown received outside of main loop");
            return;
        }
    };

    // Built before the slot is claimed so a failure costs nothing. The
    // preprocess thread already validated any grammar against this same
    // vocabulary, so this error path is defence in depth rather than the
    // primary check — but it must not panic the inference thread either way.
    let sampler = match build_sampler(model, req.sampling, req.grammar.as_ref()) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("sampler construction failed: {e}");
            reject(e, resp, token_tx, done_tx);
            return;
        }
    };

    let idx = match select_slot(slots, slot_cache, &req.tokens) {
        Some(i) => i,
        None => {
            tracing::warn!("All slots full, dropping request");
            reject(
                LlamaError::Inference("all slots busy, try again later".into()),
                resp,
                token_tx,
                done_tx,
            );
            return;
        }
    };
    let mut slot = create_slot(
        Vec::new(),
        req.max_gen,
        sampler,
        req.stop,
        resp,
        token_tx,
        done_tx,
    );
    let cached = std::mem::take(&mut slot_cache[idx]);
    let lcp = slot.begin_request(req.tokens, cached);
    if lcp > 0 {
        // Drop the old tail beyond the shared prefix; the prefill below
        // rewrites [kv_pos, n_prompt) anyway, so clearing from lcp covers
        // both divergent and generated tails. Checked, not `let _ =`:
        // a refused partial removal means the cache cannot be trusted —
        // fall back to a full clear and rewind the slot to a full prefill.
        let rewound = ctx
            .clear_kv_cache_seq(Some(idx as u32), Some(lcp as u32), None)
            .unwrap_or(false);
        if rewound {
            tracing::debug!("slot {idx} reused {lcp} cached tokens");
        } else {
            fallback_to_full_prefill(&mut slot, lcp, idx);
            let _ = ctx.clear_kv_cache_seq(Some(idx as u32), None, None);
        }
    } else {
        // The clear must be unconditional, not gated on the mirror having
        // held tokens: an empty mirror cannot guarantee the seq's KV is
        // empty — prompt-overflow-canceled slots have their partially-added
        // tokens decoded into the KV *after* the cancel-site clear (the
        // cancel drops the slot from the batch, but the shared batch still
        // decodes its already-added tokens). Every empty-cache fill must
        // therefore reconcile with a full clear; on a fresh or already-clean
        // seq this is a cheap no-op scan.
        let _ = ctx.clear_kv_cache_seq(Some(idx as u32), None, None);
    }
    slots[idx] = Some(slot);
}

// ── Cancel all active slots ───────────────────────────────────────────────────

/// Reconcile a just-finished slot's KV cache with its mirror.
///
/// Retains the sequence for prefix reuse when reuse is enabled and the
/// sequence stayed under the per-slot budget; otherwise clears both the KV and
/// the mirror. Shared by every normal finish site (EOG, generation cap, stop
/// sequence) so the retention rules cannot drift apart between them.
fn retire_slot(
    ctx: &mut LlamaContext,
    slot: &mut Slot,
    slot_cache: &mut [Vec<LlamaToken>],
    seq_id: usize,
    reuse: bool,
    per_slot_budget: u32,
) {
    if reuse && (slot.kv_pos as u32) < per_slot_budget {
        slot_cache[seq_id] = std::mem::take(&mut slot.cache_tokens);
        // The generated tail [n_prompt, kv_pos) is unreachable by any future
        // lcp (lcp <= n_prompt always); suffix removal from n_prompt can never
        // be refused on attention models (the only models where reuse
        // engages), so `let _` is safe.
        let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), Some(slot.n_prompt as u32), None);
    } else {
        // Either reuse is off, or the retention cap fired — the sequence hit
        // the per-slot budget and is not worth keeping. Clear the seq's KV
        // *and* the mirror: with the cap, the mirror can still hold a
        // previously retained sequence, and a stale mirror would make the next
        // fill compute a bogus lcp and under-clear (silent corruption). When
        // reuse is off the mirror is always empty already, so clearing it is a
        // harmless no-op rather than a behaviour change.
        slot_cache[seq_id].clear();
        let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
    }
}

fn cancel_all(slots: &mut [Option<Slot>], slot_cache: &mut [Vec<LlamaToken>], err: LlamaError) {
    for (i, slot) in slots.iter_mut().enumerate() {
        if let Some(mut s) = slot.take() {
            s.cancel(err.clone());
            slot_cache[i].clear();
        }
    }
}

// ── Inference loop ────────────────────────────────────────────────────────────

/// The process-wide llama.cpp backend, initialized at most once.
///
/// Returns an error rather than panicking: backend init can fail for
/// environmental reasons (no usable compute device), and a library must not
/// abort its host process over that. The `OnceLock` stores only a successful
/// init, so a failed attempt can be retried by a later caller.
fn get_backend() -> Result<&'static LlamaBackend> {
    static BACKEND: std::sync::OnceLock<std::result::Result<LlamaBackend, String>> =
        std::sync::OnceLock::new();
    // `LlamaBackend::init()` MUST run inside the initializer. `OnceLock`
    // guarantees the closure runs exactly once, so concurrent engines never
    // race: without this, two engines starting together both call `init()`,
    // the loser gets `BackendAlreadyInitialized` before the winner has stored
    // its value, observes an empty cell, and wrongly fails its whole engine.
    // The outcome is cached either way — a backend that cannot initialize is
    // an environmental fact, not a transient error worth retrying per engine.
    BACKEND
        .get_or_init(|| LlamaBackend::init().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| LlamaError::ModelLoad(format!("llama backend init failed: {e}")))
}

fn inference_loop(
    model_path: &Path,
    rx: mpsc::Receiver<PreparedCmd>,
    model_tx: mpsc::Sender<Arc<LlamaModel>>,
    config: InferenceConfig,
    shutdown: Arc<AtomicBool>,
) {
    let backend = match get_backend() {
        Ok(b) => b,
        Err(e) => {
            tracing::error!("{e}");
            return;
        }
    };

    unsafe extern "C" fn log_callback(
        level: llama_cpp_sys_4::ggml_log_level,
        text: *const std::os::raw::c_char,
        _user_data: *mut std::os::raw::c_void,
    ) {
        let msg = if !text.is_null() {
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
    // Routed through llama-cpp-4's wrapper rather than calling into the sys
    // crate directly, so the callback is installed on the same llama.cpp
    // instance the rest of this module talks to.
    //
    // SAFETY: `log_callback` is a `'static` fn item and the user data pointer
    // is null, so both remain valid for the life of the process.
    unsafe {
        llama_cpp_4::log_set(Some(log_callback), std::ptr::null_mut());
    }

    let model_params = std::pin::pin!(LlamaModelParams::default().with_use_mlock(false));
    let model = match LlamaModel::load_from_file(backend, model_path, &model_params) {
        Ok(m) => Arc::new(m),
        Err(e) => {
            tracing::error!("Failed to load model: {e}");
            return;
        }
    };
    // Hand the model to the preprocess thread; it only calls pure &self
    // methods (chat template, tokenize), which are safe concurrently.
    if model_tx.send(Arc::clone(&model)).is_err() {
        tracing::error!("Preprocess thread exited before model was ready");
        return;
    }

    tracing::info!(
        "Model loaded: {} vocab, {} context",
        model.n_vocab(),
        model.n_ctx_train()
    );

    let per_slot_budget = config.per_slot_budget();
    tracing::debug!(
        "Using {} decode / {} prefill threads, {} slots, {} ctx",
        config.n_threads,
        config.n_threads_batch,
        config.n_slots,
        config.n_ctx
    );

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(config.n_ctx))
        .with_n_seq_max(config.n_slots as u32)
        .with_n_threads(config.n_threads as i32)
        .with_n_threads_batch(config.n_threads_batch as i32)
        // `llama_context::decode` hard-asserts n_tokens_all <= n_batch
        // (llama-context.cpp); with n_ctx > 2048 and concurrent prefill
        // slots a larger batch would abort the daemon, so align n_batch
        // with n_ctx. n_ubatch keeps its default (it sizes the compute
        // buffer; raising it to n_ctx could balloon memory).
        .with_n_batch(config.n_ctx);
    let mut ctx = match model.new_context(backend, ctx_params) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("Failed to create context: {e}");
            return;
        }
    };
    let kv_rewind = config.kv_cache && probe_kv_rewind(&model, &mut ctx);
    if config.kv_cache && !kv_rewind {
        tracing::warn!(
            "KV prefix reuse disabled: partial KV rewind unsupported by this model; full prefill per request"
        );
    }
    let mut slots: Vec<Option<Slot>> = (0..config.n_slots).map(|_| None).collect();

    // Token history of each slot's last completed sequence, kept so the KV
    // prefix can be reused on the next request into that slot.
    let mut slot_cache: Vec<Vec<LlamaToken>> = vec![Vec::new(); config.n_slots];

    // Reusable per-step scratch: allocated once, reset every iteration.
    let mut batch = LlamaBatch::new(config.n_ctx as usize, config.n_slots as i32);
    let mut slot_batch_idx: Vec<Option<usize>> = vec![None; config.n_slots];
    let mut taken: Vec<(usize, Slot)> = Vec::with_capacity(config.n_slots);

    // ── Main loop ────────────────────────────────────────────────────────
    loop {
        // 0. Cooperative shutdown. The `PreparedCmd::Shutdown` message alone
        //    is not enough: the drain below only reads the channel while a
        //    slot is free, so with every slot busy a shutdown would not be
        //    seen until some request finished its whole generation budget —
        //    making `Engine::drop` block for many seconds. The flag is checked
        //    once per decode step instead, bounding teardown to one step.
        if shutdown.load(Ordering::Relaxed) {
            tracing::info!(
                "Shutdown requested, cancelling {} active slots",
                slots.iter().filter(|s| s.is_some()).count()
            );
            cancel_all(&mut slots, &mut slot_cache, LlamaError::InferenceCrashed);
            return;
        }

        // 1. Non-blocking drain of new requests into empty slots
        loop {
            let has_empty = slots.iter().any(|s| s.is_none());
            if !has_empty {
                break;
            }
            match rx.try_recv() {
                Ok(PreparedCmd::Shutdown) => {
                    tracing::info!(
                        "Shutdown received, draining {} active slots",
                        slots.iter().filter(|s| s.is_some()).count()
                    );
                    cancel_all(&mut slots, &mut slot_cache, LlamaError::InferenceCrashed);
                    return;
                }
                Ok(cmd) => fill_empty_slot(&model, &mut ctx, &mut slots, &mut slot_cache, cmd),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return,
            }
        }

        // 2. If no slots active, block until the next request
        let has_active = slots.iter().any(|s| s.is_some());
        if !has_active {
            match rx.recv() {
                Ok(PreparedCmd::Shutdown) => {
                    tracing::info!("Shutdown received");
                    return;
                }
                Ok(cmd) => fill_empty_slot(&model, &mut ctx, &mut slots, &mut slot_cache, cmd),
                Err(_) => return,
            }
            continue;
        }

        // 3. Build batch from all active slots.
        batch.clear();
        slot_batch_idx.fill(None);
        taken.clear();

        for seq_id in 0..config.n_slots {
            let mut slot = match slots[seq_id].take() {
                Some(s) => s,
                None => continue,
            };
            let sid = seq_id as i32;

            if slot.prompt_phase {
                let mut ok = true;
                let mut reused = slot.cache_tokens.len() as i32;
                if reused > 0 && reused as usize == slot.n_prompt {
                    // Zero-token prefill (lcp == n_prompt — identical re-send, or
                    // a prompt that is a prefix of the cached one): the whole
                    // prompt is already in KV, so nothing can be added below. The
                    // anchor must re-add the last prompt token, but the batch
                    // validator requires the batch to start at seq_pos_max + 1
                    // (llama-batch.cpp: Y = X + 1) — re-adding an occupied
                    // position would fail decode. Free the last cell by rewinding
                    // from n_prompt - 1; n_prompt == 1 clears from 0 = full clear,
                    // which is also correct.
                    let rewound = ctx
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
                        // Defense in depth: partial rewind refused — cannot happen
                        // on attention models where reuse engages, but mirror the
                        // fill-time fallback and re-add the whole prompt from 0.
                        fallback_to_full_prefill(&mut slot, reused as usize, seq_id);
                        let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
                        reused = 0;
                    }
                }
                for (i, &tok) in slot.prompt_tokens.iter().enumerate().skip(reused as usize) {
                    let idx = batch.n_tokens() as usize;
                    let is_last = i == slot.n_prompt - 1;
                    // Positions are kv_pos + i - reused; this equals the absolute
                    // index i only because kv_pos == reused == lcp after
                    // begin_request. The batch must start exactly at
                    // seq_pos_max + 1 (llama-batch.cpp).
                    if batch
                        .add(tok, slot.kv_pos + i as i32 - reused, &[sid], is_last)
                        .is_err()
                    {
                        tracing::error!("Batch overflow (prompt seq {seq_id})");
                        slot.cancel(LlamaError::Inference(
                            "batch overflow during prompt processing".into(),
                        ));
                        slot_cache[seq_id].clear();
                        let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
                        ok = false;
                        break;
                    }
                    if is_last {
                        slot_batch_idx[seq_id] = Some(idx);
                    }
                }
                if ok && slot_batch_idx[seq_id].is_none() && reused > 0 {
                    // Zero-token prefill (lcp == n_prompt — identical re-send, or a
                    // prompt that is a prefix of the cached one): the whole prompt is
                    // already in KV, so the prefill loop added nothing. The cell at
                    // n_prompt - 1 was freed by the rewind clear above, so this
                    // re-add lands at seq_pos_max + 1 — a legal batch — and gives
                    // the logits anchor the generation phase samples from. Without it
                    // the slot is canceled with a bogus "internal: empty batch" error
                    // (solo slot) or silently dropped (other slots active) — the
                    // plan's own identical-request-twice e2e hits exactly this.
                    let idx = batch.n_tokens() as usize;
                    let last = (slot.n_prompt - 1) as i32;
                    if batch
                        .add(slot.prompt_tokens[last as usize], last, &[sid], true)
                        .is_err()
                    {
                        tracing::error!("Batch overflow (prompt seq {seq_id})");
                        slot.cancel(LlamaError::Inference(
                            "batch overflow during prompt processing".into(),
                        ));
                        slot_cache[seq_id].clear();
                        let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
                        ok = false;
                    } else {
                        slot_batch_idx[seq_id] = Some(idx);
                    }
                }
                if ok {
                    slot.finish_prefill();
                    taken.push((seq_id, slot));
                }
            } else if let Some(token) = slot.pending_token {
                let idx = batch.n_tokens() as usize;
                if batch.add(token, slot.kv_pos, &[sid], true).is_ok() {
                    slot_batch_idx[seq_id] = Some(idx);
                    slot.kv_pos += 1;
                    taken.push((seq_id, slot));
                } else {
                    tracing::error!("Batch overflow (gen seq {seq_id})");
                    // Refresh the mirror to the sequence actually in KV
                    // before dropping the slot: `cache_tokens` holds the full
                    // current prompt after `finish_prefill`, so the next
                    // fill's lcp is computed against the current sequence,
                    // and the fill-time clear [lcp, ∞) always covers its
                    // entire range. (Do NOT clear the mirror — that would
                    // recreate the prompt-overflow failure mode.)
                    slot_cache[seq_id] = std::mem::take(&mut slot.cache_tokens);
                    slot.cancel(LlamaError::Inference(
                        "batch overflow during generation".into(),
                    ));
                }
            } else {
                slots[seq_id] = Some(slot);
            }
        }

        // 4. If the batch is empty, skip decode.
        if batch.n_tokens() == 0 {
            if !taken.is_empty() {
                for (seq_id, mut slot) in taken.drain(..) {
                    slot.cancel(LlamaError::Inference(
                        "internal: empty batch after building from active slots".into(),
                    ));
                    slot_cache[seq_id].clear();
                    let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
                }
            }
            continue;
        }

        // 5. Decode — one call for every active sequence.
        if let Err(e) = ctx.decode(&mut batch) {
            for (seq_id, mut slot) in taken.drain(..) {
                slot.cancel(LlamaError::Inference(e.to_string()));
                slot_cache[seq_id].clear();
                let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
            }
            continue;
        }

        // 6. Sample and process each taken slot.
        for (seq_id, mut slot) in taken.drain(..) {
            let batch_idx = match slot_batch_idx[seq_id] {
                Some(i) => i as i32,
                None => {
                    tracing::error!("slot {seq_id}: no batch index after prefill");
                    slot.cancel(LlamaError::Inference(
                        "internal: no batch index after prefill".into(),
                    ));
                    slot_cache[seq_id].clear();
                    let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
                    continue;
                }
            };

            // No explicit `accept` here: `llama_sampler_sample` already calls
            // `llama_sampler_accept` on the chain before returning (see
            // llama-sampler.cpp, and the binding's own "Sample and accept a
            // token" doc). Calling it again advanced every stateful sampler
            // twice per token — double-counting each token in the repetition
            // penalty's history, and, once a grammar is in the chain, driving
            // the grammar two steps per token until its stack empties, at
            // which point llama.cpp throws a C++ exception that unwinds
            // through the thread boundary and aborts the process.
            let token = slot.sampler.sample(&ctx, batch_idx);

            if model.is_eog_token(token) {
                slot.finish();
                retire_slot(
                    &mut ctx,
                    &mut slot,
                    &mut slot_cache,
                    seq_id,
                    reuse_allowed(config.kv_cache, kv_rewind),
                    per_slot_budget,
                );
                continue;
            }

            slot.gen_count += 1;

            match slot.push_token(&model, token) {
                Ok(Some(piece)) => match slot.push_text(&piece) {
                    Emit::Continue => {}
                    Emit::Stopped => {
                        // A stop sequence completed. `text` is already
                        // truncated at the match, so this is an ordinary
                        // successful finish — the caller gets the output that
                        // preceded the stop, without the stop text itself.
                        slot.finish();
                        retire_slot(
                            &mut ctx,
                            &mut slot,
                            &mut slot_cache,
                            seq_id,
                            reuse_allowed(config.kv_cache, kv_rewind),
                            per_slot_budget,
                        );
                        continue;
                    }
                    Emit::Disconnected => {
                        slot.cancel(LlamaError::Inference(
                            "streaming client disconnected".into(),
                        ));
                        slot_cache[seq_id].clear();
                        let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
                        continue;
                    }
                },
                Ok(None) => {}
                Err(e) => {
                    tracing::error!("Detokenization error: {e}");
                    slot.utf8_buf.clear();
                }
            }

            if slot.gen_count as u32 >= slot.max_tokens || slot.kv_pos as u32 >= per_slot_budget {
                slot.finish();
                retire_slot(
                    &mut ctx,
                    &mut slot,
                    &mut slot_cache,
                    seq_id,
                    reuse_allowed(config.kv_cache, kv_rewind),
                    per_slot_budget,
                );
                continue;
            }

            slot.pending_token = Some(token);
            slots[seq_id] = Some(slot);
        }
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

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
    cmd_tx: Option<mpsc::Sender<InferCmd>>,
    shutdown: Arc<AtomicBool>,
    inference: Option<thread::JoinHandle<()>>,
    preprocess: Option<thread::JoinHandle<()>>,
}

impl Engine {
    /// Load `model_path` and start the inference and preprocess threads.
    ///
    /// Returns as soon as the threads are spawned — the model loads in the
    /// background. A load failure is not reported here; it surfaces as
    /// [`LlamaError::InferenceCrashed`] on the first request.
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
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (prepared_tx, prepared_rx) = mpsc::channel();
        let (model_tx, model_rx) = mpsc::channel();

        let path = model_path.as_ref().to_owned();
        let infer_config = config.clone();
        let shutdown = Arc::new(AtomicBool::new(false));
        let infer_shutdown = Arc::clone(&shutdown);
        let inference = thread::Builder::new()
            .name("llamad-inference".into())
            .spawn(move || {
                inference_loop(&path, prepared_rx, model_tx, infer_config, infer_shutdown)
            })
            .map_err(LlamaError::Io)?;

        let preprocess = thread::Builder::new()
            .name("llamad-preprocess".into())
            .spawn(move || preprocess_loop(cmd_rx, model_rx, prepared_tx, config))
            .map_err(LlamaError::Io)?;

        Ok(Engine {
            cmd_tx: Some(cmd_tx),
            shutdown,
            inference: Some(inference),
            preprocess: Some(preprocess),
        })
    }

    /// The command channel. Clone it to hand senders to request handlers.
    ///
    /// # Panics
    ///
    /// If called after [`Engine::shutdown`], which consumes the engine — so
    /// this cannot happen through the public API.
    pub fn sender(&self) -> &mpsc::Sender<InferCmd> {
        self.cmd_tx.as_ref().expect("sender taken only on shutdown")
    }

    /// Submit a command.
    ///
    /// # Errors
    ///
    /// [`LlamaError::InferenceCrashed`] if the preprocess thread has exited.
    pub fn send(&self, cmd: InferCmd) -> Result<()> {
        self.sender()
            .send(cmd)
            .map_err(|_| LlamaError::InferenceCrashed)
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
    pub(crate) fn from_sender(cmd_tx: mpsc::Sender<InferCmd>) -> Self {
        Engine {
            cmd_tx: Some(cmd_tx),
            shutdown: Arc::new(AtomicBool::new(false)),
            inference: None,
            preprocess: None,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A blank slot: no channels, no stop sequences, greedy sampler.
    ///
    /// Tests override only the fields they exercise via struct-update syntax
    /// (`Slot { text: ..., ..base_slot() }`), so adding a field to `Slot` does
    /// not require touching every test.
    fn base_slot() -> Slot {
        Slot {
            prompt_tokens: vec![],
            n_prompt: 0,
            gen_count: 0,
            max_tokens: 10,
            kv_pos: 0,
            sampler: LlamaSampler::greedy(),
            utf8_buf: vec![],
            text: String::new(),
            pending_token: None,
            prompt_phase: false,
            resp: None,
            token_tx: None,
            done_tx: None,
            cache_tokens: vec![],
            stop: vec![],
            stop_max_len: 0,
            emitted: 0,
        }
    }

    /// A slot that streams to `tx` and stops on `stop`.
    fn streaming_slot(
        tx: tokio::sync::mpsc::UnboundedSender<String>,
        stop: &[&str],
    ) -> Slot {
        let stop: Vec<String> = stop.iter().map(|s| (*s).to_owned()).collect();
        Slot {
            token_tx: Some(tx),
            stop_max_len: stop.iter().map(String::len).max().unwrap_or(0),
            stop,
            ..base_slot()
        }
    }

    /// Drain everything currently queued on a token receiver.
    fn drain(rx: &mut tokio::sync::mpsc::UnboundedReceiver<String>) -> String {
        let mut out = String::new();
        while let Ok(chunk) = rx.try_recv() {
            out.push_str(&chunk);
        }
        out
    }

    #[test]
    fn test_slot_push_text_streams_and_accumulates() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut slot = streaming_slot(tx, &[]);
        assert_eq!(slot.push_text("hello"), Emit::Continue);
        assert_eq!(slot.text, "hello");
        // With no stop sequences there is nothing to withhold: the fragment
        // reaches the client in the same step it was generated.
        assert_eq!(drain(&mut rx), "hello");
    }

    #[test]
    fn test_slot_push_text_no_token_tx() {
        let mut slot = base_slot();
        assert_eq!(slot.push_text("world"), Emit::Continue);
        assert_eq!(slot.text, "world");
    }

    #[test]
    fn test_slot_finish_sends_via_resp() {
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let mut slot = Slot {
            n_prompt: 3,
            gen_count: 5,
            kv_pos: 5,
            text: "hello world".into(),
            resp: Some(resp_tx),
            ..base_slot()
        };
        slot.finish();
        let result = resp_rx.blocking_recv().unwrap().unwrap();
        assert_eq!(result.text, "hello world");
        assert_eq!(result.prompt_tokens, 3);
        assert_eq!(result.generated_tokens, 5);
    }

    #[test]
    fn test_slot_finish_sends_via_done_tx() {
        let (done_tx, done_rx) = tokio::sync::oneshot::channel();
        let mut slot = Slot {
            n_prompt: 1,
            gen_count: 2,
            kv_pos: 3,
            text: "hi".into(),
            done_tx: Some(done_tx),
            ..base_slot()
        };
        slot.finish();
        let result = done_rx.blocking_recv().unwrap().unwrap();
        assert_eq!(result.text, "hi");
        assert_eq!(result.prompt_tokens, 1);
        assert_eq!(result.generated_tokens, 2);
    }

    // ── push_raw_bytes (UTF-8 decode buffering) ──────────────────────────

    fn slot_with_utf8_buf(utf8_buf: Vec<u8>) -> Slot {
        Slot {
            utf8_buf,
            ..base_slot()
        }
    }

    #[test]
    fn test_push_raw_bytes_ascii_in_single_push() {
        let mut slot = slot_with_utf8_buf(Vec::new());
        assert_eq!(
            slot.push_raw_bytes(b"hello").unwrap().as_deref(),
            Some("hello")
        );
        assert!(slot.utf8_buf.is_empty());
    }

    #[test]
    fn test_push_raw_bytes_partial_multibyte_split_across_pushes() {
        let mut slot = slot_with_utf8_buf(Vec::new());
        // "€" = E2 82 AC: the first push ends mid-sequence...
        assert_eq!(slot.push_raw_bytes(&[0xE2, 0x82]).unwrap(), None);
        assert_eq!(slot.utf8_buf, vec![0xE2, 0x82]);
        // ...the second push completes it.
        assert_eq!(slot.push_raw_bytes(&[0xAC]).unwrap().as_deref(), Some("€"));
        assert!(slot.utf8_buf.is_empty());
    }

    #[test]
    fn test_push_raw_bytes_multibyte_char_across_chunk_boundary() {
        let mut slot = slot_with_utf8_buf(Vec::new());
        assert_eq!(slot.push_raw_bytes(&[0xE2]).unwrap(), None);
        assert_eq!(
            slot.push_raw_bytes(&[0x82, 0xAC]).unwrap().as_deref(),
            Some("€")
        );
    }

    #[test]
    fn test_push_raw_bytes_multiple_chars_in_one_push() {
        let mut slot = slot_with_utf8_buf(Vec::new());
        assert_eq!(
            slot.push_raw_bytes(b"a\xE2\x82\xACb").unwrap().as_deref(),
            Some("a€b")
        );
        assert!(slot.utf8_buf.is_empty());
    }

    #[test]
    fn test_push_raw_bytes_empty_input_returns_none() {
        let mut slot = slot_with_utf8_buf(Vec::new());
        assert_eq!(slot.push_raw_bytes(&[]).unwrap(), None);
        assert!(slot.utf8_buf.is_empty());
    }

    #[test]
    fn test_push_raw_bytes_invalid_continuation_byte_errors() {
        let mut slot = slot_with_utf8_buf(Vec::new());
        let err = slot.push_raw_bytes(&[0xE2, 0x82, 0x41]).unwrap_err();
        assert!(matches!(err, LlamaError::Inference(_)));
    }

    #[test]
    fn test_push_raw_bytes_invalid_sequence_after_valid_prefix_errors() {
        let mut slot = slot_with_utf8_buf(Vec::new());
        // "ab" flushes; the dangling lead byte 0xC3 stays buffered.
        assert_eq!(
            slot.push_raw_bytes(b"ab\xC3").unwrap().as_deref(),
            Some("ab")
        );
        assert_eq!(slot.utf8_buf, vec![0xC3]);
        // A non-continuation byte after the lead byte is permanently invalid.
        let err = slot.push_raw_bytes(b"(").unwrap_err();
        assert!(matches!(err, LlamaError::Inference(_)));
    }

    #[test]
    fn test_slot_cancel_sends_error_through_resp() {
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let mut slot = Slot {
            resp: Some(resp_tx),
            ..base_slot()
        };
        slot.cancel(LlamaError::Inference("test error".into()));
        let err = resp_rx.blocking_recv().unwrap().unwrap_err();
        assert!(matches!(err, LlamaError::Inference(_)));
    }

    #[test]
    fn test_slot_push_text_reports_disconnect() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        drop(rx);
        let mut slot = streaming_slot(tx, &[]);
        assert_eq!(slot.push_text("lost"), Emit::Disconnected);
    }

    #[test]
    fn test_cancel_all_clears_all_slots_and_sends_error() {
        let (resp_tx1, resp_rx1) = tokio::sync::oneshot::channel();
        let (resp_tx2, resp_rx2) = tokio::sync::oneshot::channel();
        let mut slots: Vec<Option<Slot>> = vec![
            Some(Slot {
                resp: Some(resp_tx1),
                ..base_slot()
            }),
            None,
            Some(Slot {
                resp: Some(resp_tx2),
                ..base_slot()
            }),
        ];
        let mut slot_cache: Vec<Vec<LlamaToken>> = vec![
            vec![LlamaToken::new(1), LlamaToken::new(2)],
            Vec::new(),
            Vec::new(),
        ];
        cancel_all(&mut slots, &mut slot_cache, LlamaError::InferenceCrashed);
        assert!(slots.iter().all(|s| s.is_none()));
        assert!(slot_cache.iter().all(|c| c.is_empty()));
        // Receivers kept alive: every active slot's client must observe the
        // crash, not just the slots being cleared.
        let err1 = resp_rx1.blocking_recv().unwrap().unwrap_err();
        let err2 = resp_rx2.blocking_recv().unwrap().unwrap_err();
        assert!(matches!(err1, LlamaError::InferenceCrashed));
        assert!(matches!(err2, LlamaError::InferenceCrashed));
    }

    #[test]
    fn test_slot_finish_sends_via_resp_and_done_tx() {
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let (done_tx, done_rx) = tokio::sync::oneshot::channel();
        let mut slot = Slot {
            n_prompt: 3,
            gen_count: 5,
            kv_pos: 5,
            text: "hello world".into(),
            resp: Some(resp_tx),
            done_tx: Some(done_tx),
            ..base_slot()
        };
        slot.finish();
        let via_resp = resp_rx.blocking_recv().unwrap().unwrap();
        let via_done = done_rx.blocking_recv().unwrap().unwrap();
        assert_eq!(via_resp.text, "hello world");
        assert_eq!(via_done.text, "hello world");
        assert_eq!(via_resp.prompt_tokens, 3);
        assert_eq!(via_done.generated_tokens, 5);
    }

    #[test]
    fn test_slot_cancel_sends_error_through_done_tx() {
        let (done_tx, done_rx) = tokio::sync::oneshot::channel();
        let mut slot = Slot {
            done_tx: Some(done_tx),
            ..base_slot()
        };
        slot.cancel(LlamaError::Inference("test error".into()));
        let err = done_rx.blocking_recv().unwrap().unwrap_err();
        assert!(matches!(err, LlamaError::Inference(_)));
    }

    // ── stop sequences ────────────────────────────────────────────────────

    #[test]
    fn test_stop_sequence_within_a_single_fragment() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut slot = streaming_slot(tx, &["STOP"]);
        assert_eq!(slot.push_text("abcSTOPdef"), Emit::Stopped);
        // The stop text and everything after it is dropped from the result.
        assert_eq!(slot.text, "abc");
        assert_eq!(drain(&mut rx), "abc");
    }

    #[test]
    fn test_stop_sequence_split_across_fragments() {
        // The case a per-token check cannot catch: no single fragment
        // contains the stop sequence, only their concatenation does.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut slot = streaming_slot(tx, &["<|end|>"]);
        assert_eq!(slot.push_text("hello <|"), Emit::Continue);
        assert_eq!(slot.push_text("end"), Emit::Continue);
        assert_eq!(slot.push_text("|>tail"), Emit::Stopped);
        assert_eq!(slot.text, "hello ");
        assert_eq!(drain(&mut rx), "hello ");
    }

    #[test]
    fn test_partial_stop_match_is_withheld_then_released() {
        // "<|" looks like the start of the stop sequence, so it must not be
        // streamed. When the next fragment rules the match out, the withheld
        // bytes are released — losing them would silently truncate output.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut slot = streaming_slot(tx, &["<|end|>"]);
        assert_eq!(slot.push_text("hi <|"), Emit::Continue);
        assert_eq!(drain(&mut rx), "hi ", "the ambiguous tail must be held");
        assert_eq!(slot.push_text("not"), Emit::Continue);
        assert_eq!(drain(&mut rx), "<|not", "ruled out, so released");
        assert_eq!(slot.text, "hi <|not");
    }

    #[test]
    fn test_no_stop_sequences_withholds_nothing() {
        // The default path must not pay any streaming latency.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut slot = streaming_slot(tx, &[]);
        slot.push_text("anything at all");
        assert_eq!(drain(&mut rx), "anything at all");
        assert_eq!(slot.holdback(), 0);
    }

    #[test]
    fn test_unrelated_text_is_not_withheld_despite_stop_sequences() {
        // Hold-back is computed from the actual tail, not from stop_max_len,
        // so output that looks nothing like a stop streams immediately.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut slot = streaming_slot(tx, &["<|end|>"]);
        slot.push_text("plain prose");
        assert_eq!(drain(&mut rx), "plain prose");
    }

    #[test]
    fn test_earliest_of_several_stop_sequences_wins() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut slot = streaming_slot(tx, &["END", "XY"]);
        assert_eq!(slot.push_text("aXYbENDc"), Emit::Stopped);
        // "XY" occurs before "END": the cut is the earliest match, not the
        // first stop sequence in list order.
        assert_eq!(slot.text, "a");
    }

    #[test]
    fn test_stop_sequence_at_the_very_start_yields_empty_text() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut slot = streaming_slot(tx, &["STOP"]);
        assert_eq!(slot.push_text("STOP"), Emit::Stopped);
        assert_eq!(slot.text, "");
    }

    #[test]
    fn test_stop_matching_across_multibyte_characters() {
        // Hold-back offsets are byte counts; landing mid-character would panic
        // on the `str` slice. "é" is two bytes, so the naive offset is unsafe.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut slot = streaming_slot(tx, &["éé"]);
        assert_eq!(slot.push_text("café"), Emit::Continue);
        assert_eq!(slot.push_text("é!"), Emit::Stopped);
        assert_eq!(slot.text, "caf");
        assert_eq!(drain(&mut rx), "caf");
    }

    #[test]
    fn test_multibyte_tail_is_not_split_when_withheld() {
        // A withheld tail must stay on a char boundary even when the stop
        // sequence is multi-byte and only partially matched.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut slot = streaming_slot(tx, &["→→"]);
        assert_eq!(slot.push_text("go →"), Emit::Continue);
        assert_eq!(drain(&mut rx), "go ");
        assert_eq!(slot.push_text(" on"), Emit::Continue);
        assert_eq!(drain(&mut rx), "→ on");
    }

    #[test]
    fn test_finish_releases_withheld_tail() {
        // Generation ending on EOG or the token cap while a partial stop match
        // is outstanding must still deliver those bytes.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let (done_tx, done_rx) = tokio::sync::oneshot::channel();
        let mut slot = Slot {
            done_tx: Some(done_tx),
            ..streaming_slot(tx, &["<|end|>"])
        };
        assert_eq!(slot.push_text("done <|"), Emit::Continue);
        assert_eq!(drain(&mut rx), "done ");
        slot.finish();
        assert_eq!(drain(&mut rx), "<|", "withheld tail must be flushed");
        assert_eq!(done_rx.blocking_recv().unwrap().unwrap().text, "done <|");
    }

    #[test]
    fn test_stop_sequence_without_streaming_client_still_truncates() {
        // Non-streaming requests have no token_tx; the stop logic must work
        // off `text` alone.
        let mut slot = Slot {
            stop: vec!["STOP".to_owned()],
            stop_max_len: 4,
            ..base_slot()
        };
        assert_eq!(slot.push_text("keepSTOPdrop"), Emit::Stopped);
        assert_eq!(slot.text, "keep");
    }

    // ── sampler chain ─────────────────────────────────────────────────────

    /// The model-card defaults, as `resolve_sampling` produces them for a
    /// request that overrides nothing.
    fn default_params() -> SamplingParams {
        SamplingParams {
            temperature: 0.1,
            top_k: 50,
            top_p: 0.95,
            repeat_penalty: 1.05,
            repeat_last_n: -1,
            seed: crate::protocol::RANDOM_SEED,
        }
    }

    #[test]
    fn test_sampler_chain_content_and_order_for_temperature() {
        assert_eq!(
            sampler_chain(default_params()),
            vec![
                SamplerKind::Penalties {
                    penalty_last_n: -1,
                    penalty_repeat: 1.05,
                },
                SamplerKind::TopK { k: 50 },
                SamplerKind::TopP {
                    p: 0.95,
                    min_keep: 1
                },
                SamplerKind::Temp { t: 0.1 },
                SamplerKind::Dist {
                    seed: crate::protocol::RANDOM_SEED
                },
            ]
        );
    }

    #[test]
    fn test_sampler_chain_threads_every_parameter_through() {
        // Regression guard for hardcoding: each field must reach its sampler.
        // These values are deliberately unlike the defaults, so a chain that
        // ignored the argument would fail on every element.
        let params = SamplingParams {
            temperature: 1.5,
            top_k: 7,
            top_p: 0.3,
            repeat_penalty: 1.9,
            repeat_last_n: 64,
            seed: 12345,
        };
        assert_eq!(
            sampler_chain(params),
            vec![
                SamplerKind::Penalties {
                    penalty_last_n: 64,
                    penalty_repeat: 1.9,
                },
                SamplerKind::TopK { k: 7 },
                SamplerKind::TopP {
                    p: 0.3,
                    min_keep: 1
                },
                SamplerKind::Temp { t: 1.5 },
                SamplerKind::Dist { seed: 12345 },
            ]
        );
    }

    #[test]
    fn test_sampler_chain_default_seed_is_the_random_sentinel() {
        // The defect this guards: a fixed seed makes every request with the
        // same prompt return identical text, so `temperature` has no
        // observable effect across requests. Each slot builds its own sampler,
        // so the sentinel is the only thing that varies them.
        let chain = sampler_chain(default_params());
        assert!(matches!(
            chain.last(),
            Some(SamplerKind::Dist { seed }) if *seed == crate::protocol::RANDOM_SEED
        ));
    }

    #[test]
    fn test_sampler_chain_zero_or_negative_temperature_means_greedy() {
        let zero = SamplingParams {
            temperature: 0.0,
            ..default_params()
        };
        let negative = SamplingParams {
            temperature: -0.5,
            ..default_params()
        };
        assert_eq!(sampler_chain(zero), vec![]);
        assert_eq!(sampler_chain(negative), vec![]);
    }

    #[test]
    fn test_sampler_kinds_construct_over_ffi() {
        // Smoke: every `SamplerKind` builds a real llama.cpp sampler. This is
        // the part of `build_sampler` that needs no model; the grammar branch
        // and the full assembly are covered by the real-model tests in
        // `tests/integration.rs`, since `LlamaSampler::grammar` needs a
        // vocabulary.
        for kind in sampler_chain(default_params()) {
            drop(kind.into_sampler());
        }
        drop(LlamaSampler::greedy());
    }

    // ── longest_common_prefix ────────────────────────────────────────────────

    #[test]
    fn test_lcp_full_match() {
        let a = vec![LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(3)];
        let b = a.clone();
        assert_eq!(longest_common_prefix(&a, &b), 3);
    }

    #[test]
    fn test_lcp_partial_match() {
        let a = vec![LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(3)];
        let b = vec![LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(9)];
        assert_eq!(longest_common_prefix(&a, &b), 2);
    }

    #[test]
    fn test_lcp_no_match() {
        let a = vec![LlamaToken::new(1), LlamaToken::new(2)];
        let b = vec![LlamaToken::new(9), LlamaToken::new(8)];
        assert_eq!(longest_common_prefix(&a, &b), 0);
    }

    #[test]
    fn test_lcp_new_prompt_is_prefix_of_cache() {
        let cache = vec![LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(3)];
        let prompt = vec![LlamaToken::new(1), LlamaToken::new(2)];
        assert_eq!(longest_common_prefix(&cache, &prompt), 2);
    }

    #[test]
    fn test_lcp_empty_sides() {
        assert_eq!(longest_common_prefix(&[], &[LlamaToken::new(1)]), 0);
        assert_eq!(longest_common_prefix(&[LlamaToken::new(1)], &[]), 0);
        assert_eq!(longest_common_prefix(&[], &[]), 0);
    }

    // ── select_slot (prefix-aware routing) ───────────────────────────────────

    fn toks(ids: &[i32]) -> Vec<LlamaToken> {
        ids.iter().copied().map(LlamaToken::new).collect()
    }

    /// `slots` from a spec of which indices are occupied.
    fn slots_with_busy(n: usize, busy: &[usize]) -> Vec<Option<Slot>> {
        (0..n)
            .map(|i| {
                if busy.contains(&i) {
                    Some(slot_for_request())
                } else {
                    None
                }
            })
            .collect()
    }

    #[test]
    fn test_select_slot_cold_start_picks_lowest_index() {
        // No slot holds a cache: every candidate scores 0, so the tie-break
        // must reproduce first-free selection exactly.
        let slots = slots_with_busy(4, &[]);
        let cache = vec![Vec::new(); 4];
        assert_eq!(select_slot(&slots, &cache, &toks(&[1, 2, 3])), Some(0));
    }

    #[test]
    fn test_select_slot_prefers_free_slot_holding_the_prefix() {
        // The regression this whole function exists for: slot 0 is busy, so
        // first-free would hand the request to slot 1 and full-prefill —
        // while slot 3 sits idle holding an exact prefix of the prompt.
        let slots = slots_with_busy(4, &[0]);
        let mut cache = vec![Vec::new(); 4];
        cache[3] = toks(&[1, 2, 3]);
        assert_eq!(select_slot(&slots, &cache, &toks(&[1, 2, 3, 4])), Some(3));
    }

    #[test]
    fn test_select_slot_picks_longest_prefix_not_merely_any_match() {
        let slots = slots_with_busy(4, &[]);
        let mut cache = vec![Vec::new(); 4];
        cache[1] = toks(&[1, 2]); // lcp 2
        cache[2] = toks(&[1, 2, 3, 4]); // lcp 4 — the best
        cache[3] = toks(&[1]); // lcp 1
        assert_eq!(select_slot(&slots, &cache, &toks(&[1, 2, 3, 4])), Some(2));
    }

    #[test]
    fn test_select_slot_ignores_cache_of_busy_slots() {
        // Slot 2 holds the perfect prefix but is mid-generation; its KV is not
        // available for reuse, so routing must fall to a free slot.
        let slots = slots_with_busy(4, &[2]);
        let mut cache = vec![Vec::new(); 4];
        cache[2] = toks(&[1, 2, 3, 4]);
        assert_eq!(select_slot(&slots, &cache, &toks(&[1, 2, 3, 4])), Some(0));
    }

    #[test]
    fn test_select_slot_equal_prefix_lengths_resolve_to_lowest_index() {
        // `max_by_key` keeps the last maximum; without the `Reverse(i)` term
        // this would return 3 and churn caches for no benefit.
        let slots = slots_with_busy(4, &[]);
        let mut cache = vec![Vec::new(); 4];
        cache[1] = toks(&[1, 2]);
        cache[3] = toks(&[1, 2]);
        assert_eq!(select_slot(&slots, &cache, &toks(&[1, 2, 9])), Some(1));
    }

    #[test]
    fn test_select_slot_all_busy_returns_none() {
        let slots = slots_with_busy(2, &[0, 1]);
        let cache = vec![Vec::new(); 2];
        assert_eq!(select_slot(&slots, &cache, &toks(&[1])), None);
    }

    #[test]
    fn test_select_slot_single_slot_always_routes_to_zero() {
        // The default configuration: one slot, and a non-matching cache must
        // not make selection fail — it routes to 0 and full-prefills.
        let slots = slots_with_busy(1, &[]);
        let cache = vec![toks(&[7, 8, 9])];
        assert_eq!(select_slot(&slots, &cache, &toks(&[1, 2])), Some(0));
    }

    // ── begin_request ────────────────────────────────────────────────────────

    fn slot_for_request() -> Slot {
        Slot {
            prompt_phase: true,
            ..base_slot()
        }
    }

    #[test]
    fn test_begin_request_reuses_shared_prefix() {
        let mut slot = slot_for_request();
        let cached = vec![LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(3)];
        let tokens = vec![LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(9)];
        let lcp = slot.begin_request(tokens.clone(), cached);
        assert_eq!(lcp, 2);
        assert_eq!(slot.kv_pos, 2);
        assert_eq!(slot.prompt_tokens, tokens);
        assert_eq!(
            slot.cache_tokens,
            vec![LlamaToken::new(1), LlamaToken::new(2)]
        );
    }

    #[test]
    fn test_begin_request_no_match_clears_cache() {
        let mut slot = slot_for_request();
        let cached = vec![LlamaToken::new(1), LlamaToken::new(2)];
        let tokens = vec![LlamaToken::new(9)];
        let lcp = slot.begin_request(tokens.clone(), cached);
        assert_eq!(lcp, 0);
        assert_eq!(slot.kv_pos, 0);
        assert!(slot.cache_tokens.is_empty());
    }

    #[test]
    fn test_begin_request_empty_cache() {
        let mut slot = slot_for_request();
        let tokens = vec![LlamaToken::new(7)];
        let lcp = slot.begin_request(tokens.clone(), vec![]);
        assert_eq!(lcp, 0);
        assert_eq!(slot.kv_pos, 0);
        assert!(slot.cache_tokens.is_empty());
        assert_eq!(slot.prompt_tokens, tokens);
    }

    #[test]
    fn test_finish_prefill_keeps_mirror_in_sync() {
        // Mirror-corruption regression guard: after a partial-prefix reuse
        // (lcp = 2 of 3), the mirror must equal the full new prompt and
        // kv_pos == cache_tokens.len(). A `drain(..)` instead of
        // `drain(reused..)` would grow the mirror to lcp + n_prompt and break
        // the invariant — invisible to identical-prompt e2e, fatal for any
        // agentic loop that re-sends a growing prompt with a shared prefix.
        let mut slot = slot_for_request();
        let cached = vec![LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(3)];
        let tokens = vec![LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(9)];
        let lcp = slot.begin_request(tokens.clone(), cached);
        assert_eq!(lcp, 2);
        assert_eq!(
            slot.cache_tokens,
            vec![LlamaToken::new(1), LlamaToken::new(2)]
        );
        slot.finish_prefill();
        assert_eq!(slot.kv_pos as usize, slot.cache_tokens.len());
        assert_eq!(slot.cache_tokens, tokens);
        assert!(!slot.prompt_phase);
        assert!(slot.prompt_tokens.is_empty());
    }

    // ── KV-rewind probe degradation (KV-rewind amendment) ─────────────────────

    #[test]
    fn test_reuse_gated_on_probe_verdict() {
        // The finish sites retain a completed sequence only when the env knob
        // is on AND the startup probe found partial rewind support. With
        // kv_rewind == false (LFM2) nothing is ever retained.
        assert!(reuse_allowed(true, true));
        assert!(!reuse_allowed(true, false));
        assert!(!reuse_allowed(false, true));
        assert!(!reuse_allowed(false, false));
    }

    #[test]
    fn test_probe_refused_means_next_request_full_prefills() {
        // With kv_rewind == false the finish sites never retain, so the slot
        // cache entry the next fill takes is empty: begin_request falls back
        // to a full prefill from position 0, matching the pre-plan behavior.
        let mut slot = slot_for_request();
        let cached = Vec::new(); // what fill_empty_slot mem::takes after a no-retain finish
        let tokens = vec![LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(3)];
        let lcp = slot.begin_request(tokens.clone(), cached);
        assert_eq!(lcp, 0);
        assert_eq!(slot.kv_pos, 0);
        assert!(slot.cache_tokens.is_empty());
        slot.finish_prefill();
        assert_eq!(slot.kv_pos as usize, slot.cache_tokens.len());
        assert_eq!(slot.cache_tokens, tokens);
    }

    #[test]
    fn test_partial_rewind_refused_falls_back_to_full_prefill() {
        // Defense in depth (review 04-H1 spirit): if the checked partial
        // removal is refused at fill time, the slot must not trust the cache
        // mirror — warn, clear it, and rewind to a full prefill.
        let mut slot = slot_for_request();
        let cached = vec![LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(3)];
        let tokens = vec![LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(9)];
        let lcp = slot.begin_request(tokens.clone(), cached);
        assert_eq!(lcp, 2);
        assert_eq!(slot.kv_pos, 2);
        fallback_to_full_prefill(&mut slot, lcp, 0);
        assert!(slot.cache_tokens.is_empty());
        assert_eq!(slot.kv_pos, 0);
        // The prefill loop reads `reused = cache_tokens.len()`: 0 means the
        // whole prompt is re-added at [0, n_prompt) — full prefill converges.
        let reused = slot.cache_tokens.len() as i32;
        assert_eq!(reused, 0);
        slot.finish_prefill();
        assert_eq!(slot.kv_pos as usize, slot.cache_tokens.len());
        assert_eq!(slot.cache_tokens, tokens);
    }
}
