// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Dory KV ownership: `auto`/`q8`/`fwht2`/`fwht3`/`fwht4` over the shared
//! `KvCache` codecs. Production default is `auto` → Q8.
//!
//! Selection reuses the shared policy: [`resolve_dory_kv`] calls
//! `hipfire_runtime::kv_mode::resolve_kv_pair` with `DORY_POLICY`
//! (accepted `q8|fwht2|fwht3|fwht4`, default Q8) plus a Dory-only
//! deprecation precheck. Anything outside the accepted set — deprecated
//! Qwen aliases, native presets, unknown names — hard-errors through the
//! one shared parser; no parallel name/default tables live here.
//!
//! Per-token position work happens ONCE in [`dory_kv_prepare`] (VMM map +
//! absolute `pos_buf` upload + sliding `len-1` upload). [`dory_kv_write`] /
//! [`dory_kv_attend`] consume the already-prepared buffers — no per-layer
//! mapping scans, no per-layer position copies. The F32 reference
//! (`quantized == false`, built only by `DoryState::new_f32_reference`)
//! rides the same split: never a silent mode fallback.
//!
//! Sliding layers attend the trailing `min(window, pos+1)` rows via
//! `sub_offset` views over the SAME arenas (no GPU allocation, no copy —
//! `sub_offset` builds a host-side view struct over the arena pointer; only
//! the `shape` Vec allocs) with `swa_pos_buf` holding `len-1`, since tiles
//! derive `seq_len` from the position buffer they are given. RoPE stays
//! absolute (baked at write). Full layers attend `[0, pos]`.
//!
//! The ONE production helper set the Dory forward calls (and the lab probe
//! calls — no forked logic): `dory_kv_prepare` once per token, then
//! `dory_kv_write` / `dory_kv_attend` per layer. Q8 rides two
//! `kv_cache_write_q8_0` launches; FWHT tiers ride the fused
//! `kv_cache_write_fwhtN_fused` writers (K rotation + Q8_0 V, signs from the
//! cache's OWN `givens_cos`/`givens_sin`, never regenerated) and the
//! matching `attention_flash_fwhtN` readers.

use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::kv_mode::{resolve_kv_pair, KvPair};
use hipfire_runtime::kv_mode::{KvMode, SlotKvTierPlan, VMode, DEPRECATED_KV_NAMES, DORY_POLICY};
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::loader_api::LoadCtx;
use rdna_compute::{Gpu, GpuTensor};

fn hip<T>(r: Result<T, hip_bridge::HipError>) -> Result<(), String> {
    r.map(|_| ()).map_err(|e| format!("dory: kv: {e}"))
}
/// LAB-ONLY native KV trace row: exact pre-cache intermediates per
/// (position, attention layer). Compiled out of production (no `lab`
/// feature → no type, no alloc, no branch in the forward path).
///
/// Contract (agreed with the CPU trace owner):
/// - `q`: `n_heads × head_dim` flattened (Dory: 16 × 256 = 4096),
///   captured BEFORE [`dory_kv_write`] (post-RoPE, post-qk-norm, post-sqk).
/// - `k`/`v`: `n_kv_heads × head_dim` flattened (Dory: 4 × 256 = 1024),
///   captured BEFORE [`dory_kv_write`] (post-RoPE; k post-qk-norm).
/// - `attention`: `n_heads × head_dim` flattened, captured AFTER
///   [`dory_kv_attend`] and BEFORE the sigmoid gate / justnorm / `Wo`.
/// - `layer` is the PHYSICAL layer index; `position` is absolute `next_pos`.
#[cfg(feature = "lab")]
#[derive(Debug, Clone, serde::Serialize)]
pub struct KvTraceRow {
    pub position: usize,
    pub layer: usize,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attention: Vec<f32>,
}
/// Pure load-time selection over the shared policy. `mode_raw` is the
/// whole-cache string (`ctx.kv_mode_override` or the global `kv_mode`
/// default); `k_raw`/`v_raw` are the existing K/V override axes (absent
/// under the daemon — admission gates them to Qwen — but honored here,
/// never silently ignored). `auto`/empty with no axes is Q8 on every arch
/// (Dory's H16 geometry never takes the gfx1201 H24 native auto path).
/// Deprecated Qwen aliases refuse explicitly with the canonical spelling;
/// the resolved V must be Q8. `gpu_arch` feeds the shared arch-aware auto
/// default (always inert for Dory, but the real arch — never a placeholder).
pub fn resolve_dory_kv(
    mode_raw: &str,
    k_raw: Option<&str>,
    v_raw: Option<&str>,
    gpu_arch: &str,
    qwen_default_q8: bool,
) -> Result<KvMode, String> {
    // Dory-only deprecation precheck: `resolve_kv_pair` would fold these
    // through the shared name table (bare asymN/turboN → FwhtN;
    // legacy-asymN → AsymN → rejected by `accepted`), but Dory names the
    // canonical spelling explicitly instead of leaning on the generic refusal.
    for raw in [Some(mode_raw), k_raw].into_iter().flatten() {
        let name = raw.trim();
        if DEPRECATED_KV_NAMES.contains(&name) {
            let canonical = match name {
                "asym2" | "turbo2" | "legacy-asym2" => "fwht2",
                "asym4" | "turbo4" | "legacy-asym4" => "fwht4",
                _ => "fwht3",
            };
            return Err(format!(
                "dory: KV name '{name}' is deprecated (use {canonical} or q8)"
            ));
        }
    }
    let pair = resolve_kv_pair(
        mode_raw,
        k_raw,
        v_raw,
        &DORY_POLICY,
        gpu_arch,
        qwen_default_q8,
    )
    .map_err(|e| format!("dory: {e}"))?;
    match pair {
        KvPair::Native(k) => Err(format!(
            "dory: kv_mode {k:?} is a native K/V preset and unsupported on Dory \
             (native FP8 needs H24/Hkv4/D256; Dory is H16/Hkv4/D256); \
             use --kv-mode auto|q8|fwht2|fwht3|fwht4"
        )),
        KvPair::Split(k, VMode::Q8) => Ok(k),
        KvPair::Split(k, v) => Err(format!(
            "dory: V mode {v:?} unsupported with K={k:?} (Dory V is Q8_0; use --kv-v q8)"
        )),
    }
}

/// Backend the carrier passes through: honored, never silently remapped.
/// Legacy uses the contiguous `_filtered` family; VMM uses the capped VMM
/// family at `physical_cap == max_seq` (no eviction on Dory, so reserve ==
/// logical budget). Anything else is refused at admission.
pub fn dory_kv_backend(ctx: &LoadCtx) -> KvBackend {
    ctx.kv_backend
}

/// ONE per-token prepare step: map VMM capacity for `pos+1` (no-op on
/// legacy), upload absolute `pos` to `pos_buf` and sliding `len-1` to
/// `swa_pos_buf`, where `len = min(window, pos+1)`. Called ONCE per
/// `forward_token` (and once per probe token) — never per layer. The
/// write/attend helpers below consume these buffers and do NO mapping or
/// position uploads of their own.
pub fn dory_kv_prepare(
    gpu: &mut Gpu,
    kv: &mut KvCache,
    pos_buf: &hip_bridge::DeviceBuffer,
    swa_pos_buf: &hip_bridge::DeviceBuffer,
    pos: usize,
    window: usize,
) -> Result<(), String> {
    hip(kv.ensure_mapped_capacity(gpu, pos + 1))?;
    let pos_i32 = pos as i32;
    hip(gpu.hip.memcpy_htod(pos_buf, &pos_i32.to_ne_bytes()))?;
    let len = (pos + 1).min(window.max(1));
    let len_i32 = (len - 1) as i32;
    hip(gpu.hip.memcpy_htod(swa_pos_buf, &len_i32.to_ne_bytes()))?;
    Ok(())
}

/// One production KV write for physical `layer_idx`. Consumes the buffers
/// [`dory_kv_prepare`] uploaded (absolute `pos_buf`; NO mapping or uploads
/// here) and dispatches on the cache's live tier flags — never on a
/// caller-supplied mode (selection and storage cannot diverge).
pub fn dory_kv_write(
    gpu: &mut Gpu,
    kv: &KvCache,
    layer_idx: usize,
    k_src: &GpuTensor,
    v_src: &GpuTensor,
    pos_buf: &hip_bridge::DeviceBuffer,
) -> Result<(), String> {
    if layer_idx >= kv.k_gpu.len() || layer_idx >= kv.v_gpu.len() {
        return Err(format!(
            "dory: kv_write layer {layer_idx} out of range ({} layers)",
            kv.k_gpu.len()
        ));
    }
    if !kv.quantized {
        // F32 internal numeric reference (lab/oracle only).
        hip(gpu.kv_cache_write(&kv.k_gpu[layer_idx], k_src, pos_buf, kv.kv_dim))?;
        hip(gpu.kv_cache_write(&kv.v_gpu[layer_idx], v_src, pos_buf, kv.kv_dim))?;
        return Ok(());
    }
    if kv.quant_q8
        && !kv.quant_fwht
        && !kv.quant_asym2
        && !kv.quant_asym3
        && !kv.quant_asym4
        && !kv.quant_fp8
        && !kv.quant_bf16
    {
        hip(gpu.kv_cache_write_q8_0(
            &kv.k_gpu[layer_idx],
            k_src,
            pos_buf,
            kv.n_kv_heads,
            kv.head_dim,
        ))?;
        hip(gpu.kv_cache_write_q8_0(
            &kv.v_gpu[layer_idx],
            v_src,
            pos_buf,
            kv.n_kv_heads,
            kv.head_dim,
        ))?;
        return Ok(());
    }
    // FWHT tiers: K rotation + Q8_0 V in one fused writer per tier. Signs are
    // the cache's OWN tables (128-wide for fwht2/4, 256-wide for fwht3 —
    // matching the `new_gpu_fwhtN_filtered` constructors); `v_mode_bits` is 8
    // on every Dory cache (static Q8 V, no Lloyd-V).
    if kv.quant_fp8 || kv.quant_bf16 {
        return Err(format!(
            "dory: kv_write: unexpected native tier flags on cache (mode={:?}); \
             Dory builds Q8/FWHT only",
            kv.current_kv_mode()
        ));
    }
    let s1 = kv
        .givens_cos
        .as_ref()
        .ok_or_else(|| "dory: kv_write: FWHT cache missing signs1".to_string())?;
    let s2 = kv
        .givens_sin
        .as_ref()
        .ok_or_else(|| "dory: kv_write: FWHT cache missing signs2".to_string())?;
    let vm = kv.v_mode_bits();
    let (k_dst, v_dst) = (&kv.k_gpu[layer_idx], &kv.v_gpu[layer_idx]);
    if kv.quant_fwht && kv.quant_asym2 && !kv.quant_asym3 && !kv.quant_asym4 {
        hip(gpu.kv_cache_write_fwht2_fused(
            k_dst,
            v_dst,
            k_src,
            v_src,
            pos_buf,
            s1,
            s2,
            kv.n_kv_heads,
            kv.head_dim,
            vm,
        ))?;
        return Ok(());
    }
    if kv.quant_fwht && kv.quant_asym3 && !kv.quant_asym2 && !kv.quant_asym4 {
        hip(gpu.kv_cache_write_fwht3_fused(
            k_dst,
            v_dst,
            k_src,
            v_src,
            pos_buf,
            s1,
            s2,
            kv.n_kv_heads,
            kv.head_dim,
            vm,
        ))?;
        return Ok(());
    }
    if kv.quant_fwht && kv.quant_asym4 && !kv.quant_asym2 && !kv.quant_asym3 {
        hip(gpu.kv_cache_write_fwht4_fused(
            k_dst,
            v_dst,
            k_src,
            v_src,
            pos_buf,
            s1,
            s2,
            kv.n_kv_heads,
            kv.head_dim,
            vm,
        ))?;
        return Ok(());
    }
    Err(format!(
        "dory: kv_write: unexpected tier flags (mode={:?}); Dory builds Q8/FWHT only",
        kv.current_kv_mode()
    ))
}

/// One production flash-attention read for physical `layer_idx`. Consumes
/// the buffers [`dory_kv_prepare`] uploaded — NO mapping or uploads here.
/// Full layers read `[0, pos]` via `pos_buf` (absolute); sliding layers read
/// the trailing `min(window, pos+1)` rows via `sub_offset` views (below)
/// with `swa_pos_buf` holding `len-1`. Exactly ONE full-vs-SWA view/len
/// selection happens up front, then each format calls its reader ONCE.
/// `partials` must cover the worst case.
#[allow(clippy::too_many_arguments)]
pub fn dory_kv_attend(
    gpu: &mut Gpu,
    kv: &KvCache,
    layer_idx: usize,
    q: &GpuTensor,
    out: &GpuTensor,
    partials: &GpuTensor,
    pos_buf: &hip_bridge::DeviceBuffer,
    swa_pos_buf: &hip_bridge::DeviceBuffer,
    pos: usize,
    is_full: bool,
    window: usize,
    n_heads: usize,
) -> Result<(), String> {
    if layer_idx >= kv.k_gpu.len() || layer_idx >= kv.v_gpu.len() {
        return Err(format!(
            "dory: kv_attend layer {layer_idx} out of range ({} layers)",
            kv.k_gpu.len()
        ));
    }
    let n_kv = kv.n_kv_heads;
    let hd = kv.head_dim;
    let cap = kv.physical_cap;
    // Single view/len selection: full reads the whole arenas with absolute
    // `pos_buf`; sliding reads trailing views with `swa_pos_buf`. Kept
    // borrows, not clones: the views alias the arenas (host-side view
    // structs; must NOT be freed — see `swa_views`).
    enum View<'a> {
        Full(&'a GpuTensor, &'a GpuTensor),
        Windowed(GpuTensor, GpuTensor),
    }
    let view: View<'_> = if is_full {
        View::Full(&kv.k_gpu[layer_idx], &kv.v_gpu[layer_idx])
    } else {
        let (k_view, v_view, _) = swa_views(kv, layer_idx, pos, window)?;
        View::Windowed(k_view, v_view)
    };
    let (k_cache, v_cache, seq_pos, seq_len) = match &view {
        View::Full(k, v) => (*k, *v, pos_buf, pos + 1),
        View::Windowed(k, v) => (k, v, swa_pos_buf, (pos + 1).min(window.max(1))),
    };
    if !kv.quantized {
        // F32 internal numeric reference: matched views + explicit length.
        hip(Gpu::attention_flash(
            gpu, q, k_cache, v_cache, out, partials, seq_len, n_heads, n_kv, hd, kv.max_seq,
        ))?;
        return Ok(());
    }
    if kv.quant_q8
        && !kv.quant_fwht
        && !kv.quant_asym2
        && !kv.quant_asym3
        && !kv.quant_asym4
        && !kv.quant_fp8
        && !kv.quant_bf16
    {
        hip(gpu.attention_flash_q8_0(
            q, k_cache, v_cache, out, seq_pos, seq_len, n_heads, n_kv, hd, cap, partials,
        ))?;
        return Ok(());
    }
    if kv.quant_fp8 || kv.quant_bf16 {
        return Err(format!(
            "dory: kv_attend: unexpected native tier flags on cache (mode={:?}); \
             Dory builds Q8/FWHT only",
            kv.current_kv_mode()
        ));
    }
    let s1 = kv
        .givens_cos
        .as_ref()
        .ok_or_else(|| "dory: kv_attend: FWHT cache missing signs1".to_string())?;
    let s2 = kv
        .givens_sin
        .as_ref()
        .ok_or_else(|| "dory: kv_attend: FWHT cache missing signs2".to_string())?;
    let vm = kv.v_mode_bits();
    if kv.quant_fwht && kv.quant_asym2 && !kv.quant_asym3 && !kv.quant_asym4 {
        hip(gpu.attention_flash_fwht2(
            q, k_cache, v_cache, out, seq_pos, s1, s2, seq_len, n_heads, n_kv, hd, cap, partials,
            vm,
        ))?;
        return Ok(());
    }
    if kv.quant_fwht && kv.quant_asym3 && !kv.quant_asym2 && !kv.quant_asym4 {
        hip(gpu.attention_flash_fwht3(
            q, k_cache, v_cache, out, seq_pos, s1, s2, seq_len, n_heads, n_kv, hd, cap, partials,
            vm,
        ))?;
        return Ok(());
    }
    if kv.quant_fwht && kv.quant_asym4 && !kv.quant_asym2 && !kv.quant_asym3 {
        hip(gpu.attention_flash_fwht4(
            q, k_cache, v_cache, out, seq_pos, s1, s2, seq_len, n_heads, n_kv, hd, cap, partials,
            vm,
        ))?;
        return Ok(());
    }
    Err(format!(
        "dory: kv_attend: unexpected tier flags (mode={:?}); Dory builds Q8/FWHT only",
        kv.current_kv_mode()
    ))
}

/// Sliding-window views over cache arenas without a GPU allocation or copy.
/// Packed byte strides come from [`SlotKvTierPlan`]; the internal F32
/// reference uses `kv_dim` elements per row. `sub_offset` creates host-side
/// view metadata; its aliased buffers must not be freed.
fn swa_views(
    kv: &KvCache,
    layer_idx: usize,
    pos: usize,
    window: usize,
) -> Result<(GpuTensor, GpuTensor, usize), String> {
    let (k_row, v_row) = if kv.quantized {
        let mode = kv
            .current_kv_mode()
            .map_err(|e| format!("dory: swa view: {e}"))?;
        let plan = SlotKvTierPlan::resolve(mode, kv.n_kv_heads, kv.head_dim)
            .map_err(|e| format!("dory: swa view: {e}"))?;
        if plan.k_bytes_per_pos % 4 != 0 || plan.v_bytes_per_pos % 4 != 0 {
            return Err(format!(
                "dory: swa view: packed row stride not 4 B-aligned ({}/{})",
                plan.k_bytes_per_pos, plan.v_bytes_per_pos
            ));
        }
        (plan.k_bytes_per_pos / 4, plan.v_bytes_per_pos / 4)
    } else {
        (kv.kv_dim, kv.kv_dim)
    };
    let start = (pos + 1).saturating_sub(window);
    let len = pos + 1 - start;
    let k_view = kv.k_gpu[layer_idx].sub_offset(start * k_row, len * k_row);
    let v_view = kv.v_gpu[layer_idx].sub_offset(start * v_row, len * v_row);
    Ok((k_view, v_view, len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_and_empty_default_q8() {
        for raw in ["", "auto", "  auto  "] {
            assert_eq!(
                resolve_dory_kv(raw, None, None, "gfx1201", false),
                Ok(KvMode::Q8)
            );
        }
    }

    #[test]
    fn canonical_modes_accept() {
        for (raw, want) in [
            ("q8", KvMode::Q8),
            ("fwht2", KvMode::Fwht2),
            ("fwht3", KvMode::Fwht3),
            ("fwht4", KvMode::Fwht4),
        ] {
            assert_eq!(
                resolve_dory_kv(raw, None, None, "gfx1201", false),
                Ok(want),
                "{raw}"
            );
        }
    }

    #[test]
    fn deprecated_aliases_refuse_with_canonical_hint() {
        for raw in [
            "asym2",
            "asym3",
            "asym4",
            "turbo",
            "turbo2",
            "turbo3",
            "turbo4",
            "legacy-asym2",
            "legacy-asym3",
            "legacy-asym4",
        ] {
            let err = resolve_dory_kv(raw, None, None, "gfx1201", false).expect_err(raw);
            assert!(err.contains("deprecated"), "{raw}: {err}");
            // Same refusal through the K-override axis.
            let err = resolve_dory_kv("auto", Some(raw), None, "gfx1201", false).expect_err(raw);
            assert!(err.contains("deprecated"), "{raw}: {err}");
        }
    }

    #[test]
    fn native_presets_refuse() {
        for raw in ["fp8", "bf16", "f16"] {
            assert!(
                resolve_dory_kv(raw, None, None, "gfx1201", false).is_err(),
                "{raw}"
            );
            // Native through the V axis refuses too.
            assert!(
                resolve_dory_kv("auto", None, Some(raw), "gfx1201", false).is_err(),
                "{raw}"
            );
        }
    }

    #[test]
    fn unknown_names_fail_closed() {
        assert!(resolve_dory_kv("mq4", None, None, "gfx1201", false).is_err());
        assert!(resolve_dory_kv("auto", Some("q4"), None, "gfx1201", false).is_err());
        assert!(resolve_dory_kv("auto", None, Some("lloyd3"), "gfx1201", false).is_err());
    }

    #[test]
    fn k_v_axes_combine_canonically() {
        // FWHT K + Q8 V is the canonical split meaning.
        assert_eq!(
            resolve_dory_kv("auto", Some("fwht3"), Some("q8"), "gfx1201", false),
            Ok(KvMode::Fwht3)
        );
        // A K axis overrides an explicit whole mode; V must stay q8.
        assert_eq!(
            resolve_dory_kv("q8", Some("fwht2"), None, "gfx1201", false),
            Ok(KvMode::Fwht2)
        );
        assert_eq!(
            resolve_dory_kv("fwht4", None, Some("q8"), "gfx1201", false),
            Ok(KvMode::Fwht4)
        );
        // Auto stays Q8 on every arch (Dory never takes the H24 native path).
        assert_eq!(
            resolve_dory_kv("auto", None, None, "gfx942", false),
            Ok(KvMode::Q8)
        );
    }
}
