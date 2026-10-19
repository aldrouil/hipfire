// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Dory one-loop forward pass (single-token decode, F32 activations, packed KV).
//!
//! Exact per-sublayer math (upstream `vllm/model_executor/models/dory.py`):
//!
//! - `justnorm(x) = x * rsqrt(sum(x²) + 1e-6)`, F32 accumulation, in place.
//!   Implemented by composing the audited `l2_norm.hip` kernel
//!   (`x *= rsqrt(sum(x²) + eps)`, eps passed as 1e-6) invoked with
//!   `n_heads = 1` for whole-vector norms or the head count for headwise
//!   norms — one launch per norm, no new kernels.
//! - Attention sublayer on normalized input `u`:
//!   `q = Wq u; k = Wk u; v = Wv u; g = Wg u` (weight codec × F32 input).
//!   RoPE (neox halfsplit, in place, F32, ABSOLUTE position) with the
//!   per-layer theta (sliding 10000 / full 10000000). Then headwise qk-norm
//!   **after** RoPE: `q[h] = justnorm(q[h]) * (sqk[h]*sqk_factor)²`,
//!   `k[h] = justnorm(k[h])` (K plain). Packed KV write at `next_pos` via
//!   `dory_kv_write` (Q8 / FWHT2/3/4 codecs); GQA flash over `[0, next_pos]`
//!   (full layers) or the trailing `min(window, next_pos+1)` rows (sliding
//!   layers, via stride-exact `sub_offset` views + a `len-1` position view)
//!   via `dory_kv_attend`.
//!   Output gate: `o = attn * sigmoid(g * (s_gate*s_gate_factor)²)`,
//!   `o = justnorm(o)`, `o = Wo o` (F32).
//! - MLP sublayer on `u`: `gt = Wgate u * (suv_gate*suv)`,
//!   `up = Wup u * (suv_up*suv)`, `m = silu(gt)*up`, `m = justnorm(m)`,
//!   `o = Wdown m`.
//! - Residual: `h' = slerp_norm(h, o, |alpha*alpha_scale|)`
//!   `= justnorm(justnorm(h) + lr*(justnorm(o) - justnorm(h)))`.
//! - Loop: input block → snapshot `input_h`; one recurrent pass with
//!   `combined = justnorm(state + input_h)` reinjection at the recurrent
//!   start and `state = justnorm(h)` at the recurrent end; output block.
//! - Head: `logits = (Wlm h_last) * (logit_scale / initializer_range)`
//!   (no final norm after the output block).
//!
//! Decode-only bring-up: one token per call. Prefill is N sequential
//! `forward_token` calls (non-final tokens with `compute_logits=false`
use crate::dory::{DoryAttnWeights, DoryConfig, DoryMlpWeights, DoryState, DoryWeights};
use crate::kv::{dory_kv_attend, dory_kv_prepare, dory_kv_write};
use crate::spec_impl::DoryBundle;
use hipfire_runtime::llama::{
    embedding_lookup_dispatch, weight_gemv, EmbeddingFormat, WeightTensor,
};
use rdna_compute::{Gpu, GpuTensor};

const EPS: f32 = 1e-6;

fn hip<T>(r: Result<T, hip_bridge::HipError>) -> Result<T, String> {
    r.map_err(|e| format!("dory: forward: {e}"))
}

fn gemv(gpu: &mut Gpu, w: &WeightTensor, x: &GpuTensor, y: &GpuTensor) -> Result<(), String> {
    weight_gemv(gpu, w, x, y).map_err(|e| format!("dory: forward gemv: {e}"))
}

/// `justnorm` over the whole vector (one row): `x *= rsqrt(sum(x²)+eps)`.
/// The `l2_norm_f32` kernel takes (n_heads, head_dim): one head of the
/// full width is exactly the whole-vector norm.
fn justnorm_all(gpu: &mut Gpu, x: &GpuTensor) -> Result<(), String> {
    // `l2_norm_f32` is deltanet-gated in rdna-compute (same gate as every
    // other arch crate: this crate builds with `default = ["deltanet"]`
    // forwarding the flag, mirroring minimax/cohere2/gemma4).
    hip(gpu.l2_norm_f32(x, 1, x.numel(), EPS))
}

/// Headwise `justnorm` over `[heads × dim]`, in place.
fn justnorm_heads(gpu: &mut Gpu, x: &GpuTensor, heads: usize, dim: usize) -> Result<(), String> {
    debug_assert_eq!(x.numel(), heads * dim);
    hip(gpu.l2_norm_f32(x, heads, dim, EPS))
}

/// Slerp residual: `h = justnorm(ya + s*(yb - ya))` with `ya=justnorm(h)`,
/// `yb=justnorm(o)`, `s = |alpha*alpha_scale|` elementwise.
/// `resid`/`tmp` are hidden-width scratch (both clobbered). `h` is dead
/// after `ya` lands in `resid`, so the final result is written back to `h`.
///
/// `alpha_vec` holds the precomputed `|alpha*alpha_scale|` vector (folded
/// once at load in `load_vec_f32_scaled_abs`); `tmp *= alpha_vec` applies
/// the reference's elementwise scaled interpolation (parity pending the
/// oracle compare — no bitwise claim until proven).
fn slerp_residual(
    gpu: &mut Gpu,
    h: &GpuTensor,
    o: &GpuTensor,
    alpha_vec: &GpuTensor,
    resid: &GpuTensor,
    tmp: &GpuTensor,
) -> Result<(), String> {
    debug_assert_eq!(h.numel(), o.numel());
    debug_assert_eq!(h.numel(), resid.numel());
    debug_assert_eq!(h.numel(), tmp.numel());
    debug_assert_eq!(h.numel(), alpha_vec.numel());
    let n = h.numel();
    hip(gpu.copy_d2d(h, resid, n * 4))?;
    justnorm_all(gpu, resid)?;
    hip(gpu.copy_d2d(o, tmp, n * 4))?;
    justnorm_all(gpu, tmp)?;
    // d = yb - ya into tmp; tmp *= s; resid += tmp; norm; write back.
    hip(gpu.scaled_add_inplace_cpu_scalar_f32(tmp, resid, -1.0))?;
    hip(gpu.mul_f32(tmp, alpha_vec, tmp))?;
    hip(gpu.add_inplace_f32(resid, tmp))?;
    justnorm_all(gpu, resid)?;
    hip(gpu.copy_d2d(resid, h, n * 4))?;
    Ok(())
}

/// Single attention sublayer + slerp residual, `h` updated in place.
/// `layer_idx` is the PHYSICAL layer index (the packed `KvCache` carries a
/// 72-slot mask; MLP slots hold placeholders). `pos` is the absolute
/// position (`next_pos`); sliding reads trail it via `kv.rs` views.
fn attn_sublayer(
    gpu: &mut Gpu,
    cfg: &DoryConfig,
    w: &DoryAttnWeights,
    st: &mut DoryState,
    layer_idx: usize,
    pos: usize,
    is_full: bool,
) -> Result<(), String> {
    let n_heads = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let hd = cfg.head_dim;
    let theta = if is_full {
        cfg.full_theta
    } else {
        cfg.sliding_theta
    };
    // 1. Projections from h (weight codec × F32 → F32).
    gemv(gpu, &w.wq, &st.h, &st.q)?;
    gemv(gpu, &w.wk, &st.h, &st.k)?;
    gemv(gpu, &w.wv, &st.h, &st.v)?;
    gemv(gpu, &w.wg, &st.h, &st.gate)?;
    // 2. RoPE after projections (neox halfsplit, in place), absolute
    // position. `dory_kv_prepare` (called once per `forward_token`) already
    // uploaded `pos` to `pos_buf` and `len-1` to `swa_pos_buf` — reuse the
    // prepared buffer here; no per-layer copy.
    hip(gpu.rope_f32(&st.q, &st.k, &st.pos_buf, n_heads, n_kv, hd, theta))?;
    // 3. Headwise qk-norm after RoPE.
    justnorm_heads(gpu, &st.q, n_heads, hd)?;
    // q *= (sqk*sqk_factor)² — folded once at load (reference order:
    // justnorm then scale; square folded into the stored vector).
    hip(gpu.mul_f32(&st.q, &w.sqk, &st.q))?;
    justnorm_heads(gpu, &st.k, n_kv, hd)?;
    // LAB-ONLY trace: capture exact pre-cache q/k/v BEFORE the write and raw
    // attention AFTER the read. Compiled out without `lab` (zero production
    // math/branch change); downloads happen ONLY when `kv_trace` is `Some`.
    #[cfg(feature = "lab")]
    let (trace_q, trace_k, trace_v) = if st.kv_trace.is_some() {
        (
            Some(hip(gpu.download_f32(&st.q))?),
            Some(hip(gpu.download_f32(&st.k))?),
            Some(hip(gpu.download_f32(&st.v))?),
        )
    } else {
        (None, None, None)
    };
    // 4. KV write + flash attention through the ONE production helper pair
    // (packed Q8 / FWHT2/3/4 codecs; F32 reference when built as such).
    // Buffers are already prepared for this `pos` — no mapping/uploads here.
    dory_kv_write(gpu, &st.kv, layer_idx, &st.k, &st.v, &st.pos_buf)?;
    dory_kv_attend(
        gpu,
        &mut st.kv,
        layer_idx,
        &st.q,
        &st.attn_out,
        &st.attn_partials,
        &st.pos_buf,
        &st.swa_pos_buf,
        pos,
        is_full,
        cfg.sliding_window,
        n_heads,
    )?;
    #[cfg(feature = "lab")]
    if let Some(rows) = st.kv_trace.as_mut() {
        rows.push(crate::kv::KvTraceRow {
            position: pos,
            layer: layer_idx,
            q: trace_q.expect("dory: kv_trace q missing"),
            k: trace_k.expect("dory: kv_trace k missing"),
            v: trace_v.expect("dory: kv_trace v missing"),
            attention: hip(gpu.download_f32(&st.attn_out))?,
        });
    }
    // gate *= (s_gate*s_gate_factor)² — folded at load; then
    // attn_out *= sigmoid(gate) via the fused epilogue (reference order:
    // sigmoid applied to the scaled gate, identical since stored IS s²).
    hip(gpu.mul_f32(&st.gate, &w.s_gate, &st.gate))?;
    hip(gpu.sigmoid_mul_f32(&st.attn_out, &st.gate))?;
    justnorm_all(gpu, &st.attn_out)?;
    gemv(gpu, &w.wo, &st.attn_out, &st.out)?;
    slerp_residual(gpu, &st.h, &st.out, &w.alpha, &st.resid, &st.tmp)?;
    Ok(())
}

/// Single MLP sublayer + slerp residual, `h` updated in place.
fn mlp_sublayer(
    gpu: &mut Gpu,
    _cfg: &DoryConfig,
    w: &DoryMlpWeights,
    st: &mut DoryState,
) -> Result<(), String> {
    gemv(gpu, &w.w_gate, &st.h, &st.gate_mlp)?;
    gemv(gpu, &w.w_up, &st.h, &st.up)?;
    // Reference order: g*(suv_gate*√H) per element (then silu), u likewise —
    // the stored vectors already hold (suv*√H), so one mul each.
    hip(gpu.mul_f32(&st.gate_mlp, &w.suv_gate, &st.gate_mlp))?;
    hip(gpu.mul_f32(&st.up, &w.suv_up, &st.up))?;
    hip(gpu.silu_mul_f32(&st.gate_mlp, &st.up, &st.ffn_mid))?;
    justnorm_all(gpu, &st.ffn_mid)?;
    gemv(gpu, &w.w_down, &st.ffn_mid, &st.out)?;
    slerp_residual(gpu, &st.h, &st.out, &w.alpha, &st.resid, &st.tmp)?;
    Ok(())
}

fn run_blocks(
    gpu: &mut Gpu,
    cfg: &DoryConfig,
    weights: &DoryWeights,
    st: &mut DoryState,
    range: std::ops::Range<usize>,
    pos: usize,
) -> Result<(), String> {
    for i in range {
        if cfg.is_attention[i] {
            let is_full = cfg.is_full_attention[i];
            let w = weights.attn[i]
                .as_ref()
                .ok_or_else(|| format!("dory: attention weights missing at layer {i}"))?;
            attn_sublayer(gpu, cfg, w, st, i, pos, is_full)?;
        } else {
            let w = weights.mlp[i]
                .as_ref()
                .ok_or_else(|| format!("dory: MLP weights missing at layer {i}"))?;
            mlp_sublayer(gpu, cfg, w, st)?;
        }
    }
    Ok(())
}

/// One decode step at `st.next_pos` for `token`. When `compute_logits` is
/// false the lm head is skipped and `None` is returned (prefill fast path);
/// otherwise the full vocab logits. Always bumps `st.next_pos`.
pub fn forward_token(
    gpu: &mut Gpu,
    bundle: &mut DoryBundle,
    token: u32,
    compute_logits: bool,
) -> Result<Option<Vec<f32>>, String> {
    let pos = bundle.state.next_pos;
    if pos >= bundle.state.max_seq {
        return Err(format!(
            "dory: forward pos={pos} >= max_seq={}; rebuild DoryState with a larger budget",
            bundle.state.max_seq
        ));
    }
    if token >= bundle.config.vocab_size as u32 {
        return Err(format!(
            "dory: token {token} >= vocab {}",
            bundle.config.vocab_size
        ));
    }
    let DoryBundle {
        config: cfg,
        weights,
        state: st,
        eos_tok: _,
    } = bundle;
    let h_dim = cfg.hidden_size;
    let n_in = cfg.block_split.0;
    let n_rec = cfg.block_split.1;
    let n_layers = cfg.num_hidden_layers;
    // Embedding lookup: BF16 table rows widen host-exact (upper F32 bits);
    // Q8 tables stay resident and dequantize on-GPU via the shared lookup
    // dispatch. No single-token BF16 lookup kernel exists; the BF16 row is
    // exactly h_dim*2 contiguous bytes.
    match weights.embd_format {
        Some(EmbeddingFormat::Q8_0) => {
            embedding_lookup_dispatch(
                gpu,
                EmbeddingFormat::Q8_0,
                &weights.token_embd,
                &st.h,
                token,
                h_dim,
            )
            .map_err(|e| format!("dory: forward embed q8: {e:?}"))?;
        }
        None => {
            let row = weights.token_embd.sub_offset(token as usize * h_dim, h_dim);
            let bytes = hip(gpu.download_raw_bytes(&row))?;
            debug_assert_eq!(bytes.len(), h_dim * 2);
            let vals: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect();
            let hb = hip(gpu.upload_f32(&vals, &[h_dim]))?;
            hip(gpu.copy_d2d(&hb, &st.h, h_dim * 4))?;
            hip(gpu.free_tensor(hb))?;
        }
        Some(other) => {
            return Err(format!(
                "dory: unsupported resident embedding format {other:?} (native BF16 or Q8_0 only)"
            ));
        }
    }
    // ONE per-token KV prepare: VMM map + absolute pos + sliding len-1
    // uploads. Every attention layer below consumes these buffers — no
    // per-layer mapping scans or position copies.
    let window = cfg.sliding_window;
    dory_kv_prepare(gpu, &mut st.kv, &st.pos_buf, &st.swa_pos_buf, pos, window)?;
    run_blocks(gpu, cfg, weights, st, 0..n_in, pos)?;
    hip(gpu.copy_d2d(&st.h, &st.input_h, h_dim * 4))?;
    // Single recurrent loop with reinjection:
    // combined = justnorm(state + input_h); end with state = justnorm(h).
    hip(gpu.copy_d2d(&st.h, &st.state_h, h_dim * 4))?;
    hip(gpu.copy_d2d(&st.state_h, &st.h, h_dim * 4))?;
    hip(gpu.add_inplace_f32(&st.h, &st.input_h))?;
    justnorm_all(gpu, &st.h)?;
    run_blocks(gpu, cfg, weights, st, n_in..n_in + n_rec, pos)?;
    hip(gpu.copy_d2d(&st.h, &st.state_h, h_dim * 4))?;
    justnorm_all(gpu, &st.state_h)?;
    hip(gpu.copy_d2d(&st.state_h, &st.h, h_dim * 4))?;
    run_blocks(gpu, cfg, weights, st, n_in + n_rec..n_layers, pos)?;
    if !compute_logits {
        st.next_pos = pos + 1;
        return Ok(None);
    }
    // lm_head (BF16 or quant codec × F32) then logits *= logit_scale / initializer_range.
    gemv(gpu, &weights.lm_head, &st.h, &st.logits)?;
    hip(gpu.mul_f32(&st.logits, &weights.logit_scale, &st.logits))?;
    hip(gpu.scale_f32(&st.logits, 1.0 / cfg.initializer_range))?;
    let out = hip(gpu.download_f32(&st.logits))?;
    st.next_pos = pos + 1;
    Ok(Some(out))
}

/// One decode step returning full vocab logits (calls `forward_token`
/// with `compute_logits=true`).
pub fn forward_step(
    gpu: &mut Gpu,
    bundle: &mut DoryBundle,
    token: u32,
) -> Result<Vec<f32>, String> {
    forward_token(gpu, bundle, token, true)
        .map(|o| o.expect("dory: forward_token(true) returns Some"))
}

/// Download the current residual stream (post output block) for oracle
/// cosine checks. F32, `hidden_size` elements.
pub fn download_last_hidden(gpu: &mut Gpu, bundle: &DoryBundle) -> Result<Vec<f32>, String> {
    hip(gpu.download_f32(&bundle.state.h))
}

/// Pure-host reference for the config/math boundary tests: slerp scale and
/// whole-vector justnorm on f32 slices.
pub fn slerp_lr_host(alpha: f32, alpha_scale: f32) -> f32 {
    (alpha * alpha_scale).abs()
}

pub fn justnorm_host(xs: &mut [f32]) {
    let sum: f32 = xs.iter().map(|x| x * x).sum();
    let inv = 1.0 / (sum + 1e-6).sqrt();
    for x in xs.iter_mut() {
        *x *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slerp_lr_matches_reference_abs_scale() {
        // Reference: lr = |alpha * alpha_scale| elementwise.
        assert_eq!(slerp_lr_host(0.5, 2.0), 1.0);
        assert_eq!(slerp_lr_host(-0.5, 2.0), 1.0);
        assert_eq!(slerp_lr_host(0.0, 2.0), 0.0);
    }

    #[test]
    fn justnorm_host_matches_unit_norm() {
        let mut v = vec![3.0f32, 4.0];
        justnorm_host(&mut v);
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-5, "norm {n}");
        assert!((v[0] - 0.6).abs() < 1e-4, "v0 {}", v[0]);
    }

    #[test]
    fn justnorm_host_eps_guards_zero_vector() {
        let mut v = vec![0.0f32; 8];
        justnorm_host(&mut v);
        assert!(v.iter().all(|x| *x == 0.0));
    }
}
