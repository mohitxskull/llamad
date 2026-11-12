use llama_cpp_4::prelude::*;

use super::slot::Slot;
use crate::protocol::LlamaError;

/// Length of the longest common prefix of two token slices.
pub(crate) fn longest_common_prefix(a: &[LlamaToken], b: &[LlamaToken]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Largest index `<= i` that lies on a `char` boundary of `s`.
///
/// Stop-sequence bookkeeping slices `text` at offsets derived from byte
/// lengths, which can land inside a multi-byte character; `str` indexing
/// panics there. (`str::floor_char_boundary` is still unstable.)
pub(crate) fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Probe once at startup into scratch seq id 0 whether the loaded model and context
/// support partial KV cache rewinding.
pub(crate) fn probe_kv_rewind(model: &LlamaModel, ctx: &mut LlamaContext) -> bool {
    let mut batch = LlamaBatch::new(8, 2);
    let probe_seq: u32 = 0; // scratch seq id; full-cleared after the probe
    // BOS-less vocabularies report LLAMA_TOKEN_NULL (-1), which the batch
    // validator rejects; fall back to token 0.
    let bos = model.token_bos();
    let tok = if bos.0 == -1 { LlamaToken::new(0) } else { bos };
    let ok_add = batch.add(tok, 0, &[probe_seq as i32], false).is_ok()
        && batch.add(tok, 1, &[probe_seq as i32], true).is_ok();
    if !ok_add || ctx.decode(&mut batch).is_err() {
        let _ = ctx.clear_kv_cache_seq(Some(probe_seq), None, None);
        return false; // decode itself failed — degrade to be safe
    }
    // Attempt to remove [1, ...): works only if partial rewind is supported.
    let rewinds = ctx
        .clear_kv_cache_seq(Some(probe_seq), Some(1), None)
        .unwrap_or(false);
    // Always leave the scratch seq clean.
    let _ = ctx.clear_kv_cache_seq(Some(probe_seq), None, None);
    rewinds
}

/// Whether a completed sequence should be retained for prefix reuse on the
/// next request into its slot. Reuse engages only when the env knob is on
/// (`LLAMAD_KV_CACHE`) AND the startup probe found the model supports
/// partial KV rewind.
pub(crate) fn reuse_allowed(kv_cache: bool, kv_rewind: bool) -> bool {
    kv_cache && kv_rewind
}

/// Handle a refused partial KV rewind: warn loudly and reset the slot to a
/// full prefill.
pub(crate) fn fallback_to_full_prefill(slot: &mut Slot, lcp: usize, idx: usize) {
    tracing::warn!("slot {idx}: partial KV rewind refused ({lcp}), falling back to full prefill");
    slot.cache_tokens.clear();
    slot.kv_pos = 0;
}

/// Choose which free slot should serve `tokens`.
///
/// Prefers the free slot whose retained KV prefix shares the most tokens with
/// the incoming prompt. Ties resolve to the lowest free index.
pub(crate) fn select_slot(
    slots: &[Option<Slot>],
    slot_cache: &[Vec<LlamaToken>],
    tokens: &[LlamaToken],
) -> Option<usize> {
    slots
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_none())
        .map(|(i, _)| (i, longest_common_prefix(&slot_cache[i], tokens)))
        .max_by_key(|&(i, lcp)| (lcp, std::cmp::Reverse(i)))
        .map(|(i, _)| i)
}

/// Reconcile a just-finished slot's KV cache with its mirror.
pub(crate) fn retire_slot(
    ctx: &mut LlamaContext,
    slot: &mut Slot,
    slot_cache: &mut [Vec<LlamaToken>],
    seq_id: usize,
    reuse: bool,
    per_slot_budget: u32,
) {
    if reuse && (slot.kv_pos as u32) < per_slot_budget {
        slot_cache[seq_id] = std::mem::take(&mut slot.cache_tokens);
        let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), Some(slot.n_prompt as u32), None);
    } else {
        slot_cache[seq_id].clear();
        let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
    }
}

/// Fail a slot and return its sequence to a known-empty state.
pub(crate) fn abandon_slot(
    ctx: &mut LlamaContext,
    slot: &mut Slot,
    slot_cache: &mut [Vec<LlamaToken>],
    seq_id: usize,
    err: LlamaError,
) {
    slot.cancel(err);
    slot_cache[seq_id].clear();
    let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
}

/// Cancel all active slots.
pub(crate) fn cancel_all(
    slots: &mut [Option<Slot>],
    slot_cache: &mut [Vec<LlamaToken>],
    err: LlamaError,
) {
    for (i, slot) in slots.iter_mut().enumerate() {
        if let Some(mut s) = slot.take() {
            s.cancel(err.clone());
            slot_cache[i].clear();
        }
    }
}
