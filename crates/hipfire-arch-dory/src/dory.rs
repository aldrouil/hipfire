// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Dory config, weights, and decode state.
//!
//! Model: `DoryForCausalLM` (`model_type: "dory"`, upstream reference
//! `vllm/model_executor/models/dory.py` at
//! `https://github.com/vllm-project/vllm/blob/55076950b6c155f384d6766d63563c3bc20a0e3f/vllm/model_executor/models/dory.py`).
//! Physical sublayers strictly alternate attention / MLP (config
//! `layer_types`: every even index is `sliding_attention`/`full_attention`,
//! every odd index `mlp`). Each layer holds exactly ONE mixer plus its
//! `*_alpha` interpolation scale — not the standard attention+FFN block.
//! The shipped checkpoint has 72 sublayers; native expanded layouts (76 in
//! the expanded 44-recurrent model) are admitted on the same alternation,
//! with the declared `layer_types` carrying the full/sliding placement.
//!
//! Multi-loop recurrence policy: the forward runs exactly
//! `n_recurrent_loops` (1..=4) passes over the shared physical recurrent
//! block `[n_input, n_input + n_recurrent)`, with per-loop or last-loop KV
//! semantics selected by `recurrent_kv_cache_mode`.
//!
//! Weight residency: native BF16 bytes upload verbatim and execute through
//! `gemv_bf16_xf32` (F32 activations); HFQ packs ride their common quant
//! codecs (`dequant_weight_raw`, Q8 embedding via `load_embedding`) with
//! the same F32 activations. No 16 GB F32 weight expansion: the full
//! checkpoint stays compressed on device (BF16 ~7.95 GB; quants smaller).
//! Per-vector scales (`attn_alpha`, `mlp_alpha`, `sqk`, `s_gate`, `suv_*`,
//! `logit_scale`) are widened BF16 → F32 once at load (native) or consumed
//! as exact-F32 HFQ scales. Activations are F32; KV is packed (see
//! `crate::kv`: `auto`/Q8 default, FWHT2/3/4, F32 reference variant only).
//! Load-time developer diagnostics (`[developer]` namespace; unset = stock,
//! see `docs/env-vars.md`): `HIPFIRE_DORY_GATE_SCALE` multiplies the
//! effective residual gate `s = |alpha * alpha_scale|` AFTER the abs on the
//! Stage4-inserted sublayers only (56..64; never the raw alpha, `s_gate` or
//! `suv_*`), and `HIPFIRE_DORY_GATE_SOURCE` swaps
//! the `attn_alpha`/`mlp_alpha` vectors of only the sublayers the Stage4
//! expansion inserted (physical 56..64) for an untrained pack's. Both resolve
//! once before the first GPU allocation, keep every matrix from the loaded
//! model, and reverse at the next process.
//!
//! Naming: local safetensors tensors are `backbone.*`; the vLLM reference
//! packs QKV (`qkv_proj`) and gate/up (`gate_up_proj`) where the checkpoint
//! stores split `q/k/v_proj` and `gate/up_proj`. The loader maps exact local
//! names (never the packed names) and fails closed on any shape/dtype miss.

use hipfire_config::DoryRecurrentKvCacheMode;
use hipfire_runtime::llama::{EmbeddingFormat, WeightTensor};
use hipfire_runtime::model_source::{ModelSource, TensorInfo};
use hipfire_runtime::weight_backend::{dequant_weight_raw, load_embedding, MemoryTarget};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Exact BF16 bytes for one row-major `[m, k]` matrix on GPU.
fn upload_bf16_matrix(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<GpuTensor, String> {
    let (info, data) = source
        .tensor_data(name)
        .ok_or_else(|| format!("dory: tensor not found: {name}"))?;
    if info.dtype != "BF16" {
        return Err(format!(
            "dory: tensor {name} has dtype {} (expected BF16)",
            info.dtype
        ));
    }
    if info.shape != vec![m, k] {
        return Err(format!(
            "dory: tensor {name} has shape {:?} (expected [{m}, {k}])",
            info.shape
        ));
    }
    if data.len() != m * k * 2 {
        return Err(format!(
            "dory: tensor {name} has {} bytes (expected {})",
            data.len(),
            m * k * 2
        ));
    }
    let mut buf = gpu
        .upload_raw(data, &[m * k])
        .map_err(|e| format!("dory: upload {name}: {e}"))?;
    buf.dtype = DType::BF16;
    buf.shape = vec![m, k];
    Ok(buf)
}

/// Widen one per-vector scale to F32, applying a host-exact transform.
/// Admission (dtype/shape/bytes) is checked once here; `f` maps each decoded
/// value to what the forward consumes, so per-token work stays one `mul_f32`.
/// Sources: native BF16 safetensors bytes, or HFQ exact-F32 scales (the
/// quantizer widens every nGPT vector BF16 → F32 losslessly, qt=2).
/// Plain widen (`load_vec_f32`) is for `logit_scale` (consumed raw); the
/// scaled variants fold the reference's per-element factors at load.
fn load_vec_f32_mapped(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    n: usize,
    f: impl Fn(f32) -> f32,
) -> Result<GpuTensor, String> {
    let raw = decode_vec_f32(source, name, n)?;
    let vals: Vec<f32> = raw.into_iter().map(f).collect();
    gpu.upload_f32(&vals, &[n])
        .map_err(|e| format!("dory: upload {name}: {e}"))
}

/// Metadata admission shared by every length-`n` nGPT per-vector scale: 1-D
/// `[n]`, BF16 or F32. The gate-source substitution calls this for the
/// loaded model's own vector too, so a replacement never hides a missing or
/// malformed model tensor.
fn vec_f32_meta<'a>(
    source: &'a dyn ModelSource,
    name: &str,
    n: usize,
) -> Result<&'a TensorInfo, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("dory: tensor not found: {name}"))?;
    if info.shape != vec![n] {
        return Err(format!(
            "dory: tensor {name} has shape {:?} (expected [{n}])",
            info.shape
        ));
    }
    if info.dtype != "BF16" && info.dtype != "F32" {
        return Err(format!(
            "dory: tensor {name} has dtype {} (expected BF16 or F32)",
            info.dtype
        ));
    }
    Ok(info)
}

/// Host-side widening of one per-vector scale: native BF16 dir bytes widen
/// exactly (BF16 = upper F32 bits); HFQ nGPT scales ride exact F32 (qt=2) and
/// are consumed verbatim.
fn decode_vec_f32(source: &dyn ModelSource, name: &str, n: usize) -> Result<Vec<f32>, String> {
    let info = vec_f32_meta(source, name, n)?;
    let (_, data) = source
        .tensor_data(name)
        .ok_or_else(|| format!("dory: tensor not found: {name}"))?;
    let raw: Vec<f32> = if info.dtype == "BF16" {
        if data.len() != n * 2 {
            return Err(format!(
                "dory: tensor {name} has {} bytes (expected {})",
                data.len(),
                n * 2
            ));
        }
        data.chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()
    } else {
        if data.len() != n * 4 {
            return Err(format!(
                "dory: tensor {name} has {} bytes (expected {})",
                data.len(),
                n * 4
            ));
        }
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    debug_assert_eq!(raw.len(), n);
    Ok(raw)
}

/// One BF16 per-vector scale widened to F32 on GPU (host-exact widening).
fn load_vec_f32(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    n: usize,
) -> Result<GpuTensor, String> {
    load_vec_f32_mapped(source, gpu, name, n, |v| v)
}

/// Stores `|v * scale|` elementwise (the interpolation coefficient the
/// reference's `slerp_alpha` broadcasts). Used for `attn_alpha`/`mlp_alpha`
/// so the forward needs no abs/scale kernels.
fn load_vec_f32_scaled_abs(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    n: usize,
    scale: f32,
) -> Result<GpuTensor, String> {
    load_vec_f32_mapped(source, gpu, name, n, |v| (v * scale).abs())
}

/// Stores `(v * factor)^2` elementwise (the squared qk/gate scales the
/// reference applies after justnorm: `Q *= (sqk*f)^2`,
/// `sigmoid(g*(s_gate*f)^2)`). Used for `sqk` and `s_gate` only, so the
/// forward does one `mul_f32`.
fn load_vec_f32_scaled_square(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    n: usize,
    factor: f32,
) -> Result<GpuTensor, String> {
    load_vec_f32_mapped(source, gpu, name, n, |v| {
        let s = v * factor;
        s * s
    })
}

/// Stores `v * factor` elementwise (the UNSQUARED MLP scalings the reference
/// applies: `g*(suv_gate*√H)`, `u*(suv_up*√H)`). Used for `suv_gate`/`suv_up`
/// so the forward does one `mul_f32` in reference order.
fn load_vec_f32_scaled(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    n: usize,
    factor: f32,
) -> Result<GpuTensor, String> {
    load_vec_f32_mapped(source, gpu, name, n, |v| v * factor)
}

// ─── Developer load-time alpha-gate diagnostics ──────────────────────────
//
// Two `[developer]` knobs (unset = stock) for ablation experiments on the
// residual interpolation gates: they are resolved ONCE at load, before the
// first GPU allocation, and are reversible at the next process. Nothing is
// added to the forward, the reload identity or a user config surface.

/// Post-`abs` multiplier on the effective residual gate
/// `s = |alpha * alpha_scale|`, applied only on the sublayers the Stage4
/// expansion inserted (physical 56..64, [`stage4_inserted_layers`]); every
/// other sublayer keeps `|alpha * alpha_scale|` exactly. Nonnegative and
/// finite; `1` is the stock expression bit-for-bit, `0` zeroes those
/// sublayers' residual contribution while their own residual normalization
/// and KV behavior still run (it is not a skipped layer).
pub const DORY_GATE_SCALE_ENV: &str = "HIPFIRE_DORY_GATE_SCALE";

/// Untrained Stage4 HFQ whose `attn_alpha`/`mlp_alpha` vectors replace the
/// loaded model's for the sublayers the Stage4 expansion inserted. Only those
/// scale vectors are read: every matrix, embedding, output table and KV
/// parameter stays the loaded model's.
pub const DORY_GATE_SOURCE_ENV: &str = "HIPFIRE_DORY_GATE_SOURCE";

/// Stock (pre-expansion) Dory: 72 sublayers split 16 input / 40 recurrent /
/// 16 output — the layout every published dory checkpoint ships.
const STOCK_DORY_LAYERS: usize = 72;
const STOCK_DORY_SPLIT: (usize, usize, usize) = (16, 40, 16);
/// Stage4 Dory: the recurrent block carries four extra attention/MLP pairs.
const STAGE4_DORY_LAYERS: usize = 80;
const STAGE4_DORY_SPLIT: (usize, usize, usize) = (16, 48, 16);
const _: () =
    assert!(STOCK_DORY_SPLIT.0 + STOCK_DORY_SPLIT.1 + STOCK_DORY_SPLIT.2 == STOCK_DORY_LAYERS);
const _: () =
    assert!(STAGE4_DORY_SPLIT.0 + STAGE4_DORY_SPLIT.1 + STAGE4_DORY_SPLIT.2 == STAGE4_DORY_LAYERS);
// The expansion inserts whole attention/MLP pairs, so the substituted region
// must hold an even number of sublayers.
const _: () = assert!(
    (STAGE4_DORY_SPLIT.0 + STAGE4_DORY_SPLIT.1 - STOCK_DORY_SPLIT.0 - STOCK_DORY_SPLIT.1) % 2 == 0
);

/// Physical sublayers that exist only in the Stage4 expansion: the stock
/// recurrent block ends at `16 + 40 = 56`, the Stage4 block at `16 + 48 = 64`,
/// and the extra pairs were inserted at the end of that block, so `56..64` is
/// the checkpoint's new region (four attention + four MLP sublayers). Only
/// these carry a gate-source substitution; the stock sublayers keep theirs.
fn stage4_inserted_layers() -> std::ops::Range<usize> {
    (STOCK_DORY_SPLIT.0 + STOCK_DORY_SPLIT.1)..(STAGE4_DORY_SPLIT.0 + STAGE4_DORY_SPLIT.1)
}

/// Resolved `HIPFIRE_DORY_GATE_*` state. [`DoryGatePlan::stock`] (scale `1`,
/// no replacements) is the pre-ablation path: the loader calls the same
/// expressions as before and opens nothing extra.
struct DoryGatePlan {
    scale: f32,
    /// Raw (pre-`alpha_scale`, pre-gate-scale) alpha per physical sublayer;
    /// `None` everywhere outside [`stage4_inserted_layers`].
    replacements: Vec<Option<Vec<f32>>>,
}

impl DoryGatePlan {
    fn stock() -> Self {
        Self {
            scale: 1.0,
            replacements: Vec::new(),
        }
    }

    fn replacement(&self, physical: usize) -> Option<&[f32]> {
        self.replacements.get(physical).and_then(|v| v.as_deref())
    }

    /// Gate multiplier in effect for `physical`: the diagnostic scale only
    /// inside the Stage4-inserted region, exactly `1` everywhere else, so a
    /// stock sublayer's effective gate is never touched.
    fn scale_for(&self, physical: usize) -> f32 {
        if stage4_inserted_layers().contains(&physical) {
            self.scale
        } else {
            1.0
        }
    }
}

/// Effective residual interpolation coefficient the forward consumes:
/// `|alpha * alpha_scale|` and then the developer gate multiplier. `scale ==
/// 1` returns the pre-ablation expression untouched (no multiply), so the
/// default stays bit-identical.
fn scale_alpha(v: f32, alpha_scale: f32, gate_scale: f32) -> f32 {
    let effective = (v * alpha_scale).abs();
    if gate_scale == 1.0 {
        effective
    } else {
        effective * gate_scale
    }
}

/// Parse [`DORY_GATE_SCALE_ENV`]: a nonnegative finite float. Anything else
/// fails the load instead of silently falling back to stock, because a
/// mistyped ablation knob would otherwise look like a null result.
fn parse_gate_scale(raw: &str) -> Result<f32, String> {
    let value: f32 = raw.trim().parse().map_err(|_| {
        format!(
            "dory: {DORY_GATE_SCALE_ENV}=`{raw}` is not a float (admit a finite value >= 0; unset for stock)"
        )
    })?;
    if !value.is_finite() || value < 0.0 {
        return Err(format!(
            "dory: {DORY_GATE_SCALE_ENV}=`{raw}` must be finite and >= 0 (0 zeroes the residual gate; unset for stock)"
        ));
    }
    Ok(value)
}

/// Parse [`DORY_GATE_SOURCE_ENV`]: a whitespace-trimmed path, where blank
/// counts as unset so a harness can clear the knob without unsetting it.
fn parse_gate_source(raw: &str) -> Option<std::path::PathBuf> {
    let trimmed = raw.trim();
    (!trimmed.is_empty()).then(|| std::path::PathBuf::from(trimmed))
}

fn gate_scale_from_env() -> Result<Option<f32>, String> {
    match hipfire_config::developer_var(DORY_GATE_SCALE_ENV) {
        Ok(raw) => parse_gate_scale(&raw).map(Some),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(format!(
            "dory: {DORY_GATE_SCALE_ENV} is not valid UTF-8: {e}"
        )),
    }
}

fn gate_source_from_env() -> Result<Option<std::path::PathBuf>, String> {
    match hipfire_config::developer_var(DORY_GATE_SOURCE_ENV) {
        Ok(raw) => Ok(parse_gate_source(&raw)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(format!(
            "dory: {DORY_GATE_SOURCE_ENV} is not valid UTF-8: {e}"
        )),
    }
}

/// The substitution is validated only against the Stage4 geometry it was
/// built for: any other layout has different inserted indices, so it fails
/// closed instead of guessing which layers are new.
fn validate_stage4_gate_layout(cfg: &DoryConfig) -> Result<(), String> {
    if cfg.num_hidden_layers != STAGE4_DORY_LAYERS || cfg.block_split != STAGE4_DORY_SPLIT {
        return Err(format!(
            "dory: {DORY_GATE_SOURCE_ENV} is admitted only on the Stage4 layout \
             ({STAGE4_DORY_LAYERS} sublayers split {STAGE4_DORY_SPLIT:?}); this model has \
             {} sublayers split {:?}",
            cfg.num_hidden_layers, cfg.block_split
        ));
    }
    Ok(())
}

/// The gate source must describe the same net as the loaded model, or the
/// substituted vectors would not mean what the model's effective gate means.
fn validate_gate_source_geometry(main: &DoryConfig, src: &DoryConfig) -> Result<(), String> {
    let mismatch = |field: &str, src_value: String, main_value: String| {
        format!("dory: {DORY_GATE_SOURCE_ENV} {field}={src_value} != model {main_value}")
    };
    for (field, a, b) in [
        ("hidden_size", main.hidden_size, src.hidden_size),
        (
            "num_hidden_layers",
            main.num_hidden_layers,
            src.num_hidden_layers,
        ),
        (
            "intermediate_size",
            main.intermediate_size,
            src.intermediate_size,
        ),
        (
            "num_attention_heads",
            main.num_attention_heads,
            src.num_attention_heads,
        ),
        (
            "num_key_value_heads",
            main.num_key_value_heads,
            src.num_key_value_heads,
        ),
        ("head_dim", main.head_dim, src.head_dim),
        ("vocab_size", main.vocab_size, src.vocab_size),
        ("sliding_window", main.sliding_window, src.sliding_window),
    ] {
        if a != b {
            return Err(mismatch(field, b.to_string(), a.to_string()));
        }
    }
    if main.block_split != src.block_split {
        return Err(mismatch(
            "block_split",
            format!("{:?}", src.block_split),
            format!("{:?}", main.block_split),
        ));
    }
    if main.is_attention != src.is_attention {
        return Err(mismatch(
            "layer_types (attention/MLP placement)",
            "different".into(),
            "this model's".into(),
        ));
    }
    if main.is_full_attention != src.is_full_attention {
        return Err(mismatch(
            "layer_types (full/sliding attention)",
            "different".into(),
            "this model's".into(),
        ));
    }
    // Alpha semantics: the reference gate is `|alpha * ngpt_alpha_init /
    // initializer_range|`, so both checkpoints must fold the same constant.
    if main.alpha_scale.to_bits() != src.alpha_scale.to_bits() {
        return Err(mismatch(
            "alpha semantics (ngpt_alpha_init/initializer_range)",
            src.alpha_scale.to_string(),
            main.alpha_scale.to_string(),
        ));
    }
    Ok(())
}

/// Read the untrained gate source's alpha vectors for the inserted sublayers.
/// Only `attn_alpha`/`mlp_alpha` are read; the pack's matrices are never
/// touched, so this is a read-only, single-file operation.
fn load_gate_source(
    cfg: &DoryConfig,
    path: &std::path::Path,
) -> Result<Vec<Option<Vec<f32>>>, String> {
    let context = |detail: String| format!("dory: {DORY_GATE_SOURCE_ENV}={path:?}: {detail}");
    let hfq = hipfire_runtime::hfq::HfqFile::open(path)
        .map_err(|e| context(format!("open failed: {e}")))?;
    if hfq.arch_id != 19 {
        return Err(context(format!(
            "arch_id={} (expected 19=dory)",
            hfq.arch_id
        )));
    }
    let bridge = hipfire_runtime::hfq::HfqModelSource::from_hfq(hfq);
    let src_cfg = config_from_source(&bridge).map_err(|e| context(e))?;
    validate_gate_source_geometry(cfg, &src_cfg)?;
    let mut replacements: Vec<Option<Vec<f32>>> = vec![None; cfg.num_hidden_layers];
    for physical in stage4_inserted_layers() {
        let name = if cfg.is_attention[physical] {
            format!("backbone.layers.{physical}.attn_alpha")
        } else {
            format!("backbone.layers.{physical}.mlp_alpha")
        };
        let raw = decode_vec_f32(&bridge, &name, cfg.hidden_size).map_err(|e| context(e))?;
        if let Some(bad) = raw.iter().position(|v| !v.is_finite()) {
            return Err(context(format!(
                "{name}[{bad}] is not finite: a non-finite gate would poison every residual"
            )));
        }
        replacements[physical] = Some(raw);
    }
    Ok(replacements)
}

/// Resolve both developer knobs once, before the first GPU allocation.
fn resolve_gate_plan(cfg: &DoryConfig) -> Result<DoryGatePlan, String> {
    let scale = gate_scale_from_env()?;
    let source = gate_source_from_env()?;
    if scale.is_none() && source.is_none() {
        return Ok(DoryGatePlan::stock());
    }
    // Any explicit diagnostic — including an explicit `1` — is admitted only
    // on the Stage4 layout, because that is the only layout whose inserted
    // region this ablation was validated against.
    validate_stage4_gate_layout(cfg)?;
    let scale = scale.unwrap_or(1.0);
    let Some(path) = source else {
        eprintln!(
            "dory: developer diagnostic: {DORY_GATE_SCALE_ENV}={scale} multiplies the \
             effective residual gate |alpha*alpha_scale| after the abs on physical sublayers \
             {}..={} only; other sublayers, matrices, residual normalization and KV unchanged",
            stage4_inserted_layers().start,
            stage4_inserted_layers().end - 1
        );
        return Ok(DoryGatePlan {
            scale,
            replacements: Vec::new(),
        });
    };
    let replacements = load_gate_source(cfg, &path)?;
    let inserted = stage4_inserted_layers();
    eprintln!(
        "dory: developer diagnostic: {DORY_GATE_SOURCE_ENV}={path:?} replaces \
         attn_alpha/mlp_alpha for physical sublayers {}..={} only (every matrix stays the \
         loaded model's); {DORY_GATE_SCALE_ENV}={scale}",
        inserted.start,
        inserted.end - 1
    );
    Ok(DoryGatePlan {
        scale,
        replacements,
    })
}

/// Upload one residual alpha vector: the gate source's raw vector for an
/// inserted Stage4 sublayer when configured, else `name` from the loaded
/// model, then the effective-gate transform. The diagnostic scale is folded
/// only where [`DoryGatePlan::scale_for`] returns it; with a gate multiplier
/// of `1` this is exactly the pre-ablation `load_vec_f32_scaled_abs` call.
fn load_alpha_vec(
    source: &dyn ModelSource,
    gate: &DoryGatePlan,
    physical: usize,
    name: &str,
    n: usize,
    alpha_scale: f32,
    gpu: &mut Gpu,
) -> Result<GpuTensor, String> {
    let gate_scale = gate.scale_for(physical);
    let Some(raw) = gate.replacement(physical) else {
        return if gate_scale == 1.0 {
            load_vec_f32_scaled_abs(source, gpu, name, n, alpha_scale)
        } else {
            load_vec_f32_mapped(source, gpu, name, n, |v| {
                scale_alpha(v, alpha_scale, gate_scale)
            })
        };
    };
    // Even when the vector is replaced, the loaded model's own tensor must be
    // present and well formed: the loader's fail-closed contract covers every
    // tensor it names, so a corrupt pack never passes silently.
    vec_f32_meta(source, name, n)?;
    debug_assert_eq!(raw.len(), n);
    let vals: Vec<f32> = raw
        .iter()
        .map(|&v| scale_alpha(v, alpha_scale, gate_scale))
        .collect();
    gpu.upload_f32(&vals, &[n])
        .map_err(|e| format!("dory: upload {name}: {e}"))
}
fn load_linear_bf16(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    Ok(WeightTensor {
        buf: upload_bf16_matrix(source, gpu, name, m, k)?,
        gpu_dtype: DType::BF16,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    })
}
/// Load one `[m, k]` matrix through the shared [`dequant_weight_raw`] codec:
/// BF16 bytes (native dir or HFQ qt16) upload verbatim (`DType::BF16`); every
/// other HFQ quant in the Dory set rides its common passthrough (`MQ3V2`
/// qt49, `MQ4V2` qt44, `MQ5V2` qt48, `MQ6V2` qt47, `MQ8` qt14,
/// `Q8_0` qt3). MQ2V2 (qt50) never reaches here: the
/// `load_weights_from_source` preflight refuses it first (catastrophic on
/// Dory). Anything else reaches the common decoder, whose `K % 256`
/// and payload-length guards fail closed on unlisted layouts.
fn load_linear_common(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (info, data) = source
        .tensor_data(name)
        .ok_or_else(|| format!("dory: tensor not found: {name}"))?;
    if info.shape != vec![m, k] {
        return Err(format!(
            "dory: tensor {name} has shape {:?} (expected [{m}, {k}])",
            info.shape
        ));
    }
    // BF16 bytes upload verbatim regardless of container: native dirs and
    // HFQ qt16 alike (qt16 payloads are BF16 bytes; only the common
    // decoder's host path widens them — never taken here).
    if info.dtype == "BF16" {
        return load_linear_bf16(source, gpu, name, m, k);
    }
    dequant_weight_raw(gpu, info.quant_type, data, m, k, MemoryTarget::Device)
        .map_err(|e| format!("dory: tensor {name}: {e:?}"))
}

/// Load the embedding table for an HFQ source. Tables arrive Q8 (qt=3, every
/// shipped tier) via the shared [`load_embedding`] helper: raw upload + Q8
/// GPU lookup, no giant dequantized float matrix. Any other codec (fixed-tier
/// overrides, BF16 widenings) fails closed — never silently
/// misread. Native BF16 dirs never reach here (verbatim path inline).
fn load_embed_table(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    v: usize,
    h: usize,
) -> Result<(GpuTensor, EmbeddingFormat), String> {
    let (info, data) = source
        .tensor_data(name)
        .ok_or_else(|| format!("dory: tensor not found: {name}"))?;
    if info.shape != vec![v, h] {
        return Err(format!(
            "dory: tensor {name} has shape {:?} (expected [{v}, {h}])",
            info.shape
        ));
    }
    if info.quant_type != 3 {
        return Err(format!(
            "dory: tensor {name} has quant_type {} (expected 3=Q8_0); non-Q8 embedding tables unsupported",
            info.quant_type
        ));
    }
    load_embedding(gpu, info.quant_type, data, v, h)
        .map_err(|e| format!("dory: upload {name}: {e:?}"))
}

/// Dory shape constants parsed from the safetensors `config.json`.
#[derive(Debug, Clone)]
pub struct DoryConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub sliding_window: usize,
    pub sliding_theta: f32,
    pub full_theta: f32,
    pub initializer_range: f32,
    /// `layer_types[i] == true` iff layer `i` is attention (false = MLP).
    pub is_attention: Vec<bool>,
    /// `true` iff attention layer `i` is full (false = sliding window).
    /// Meaningful only where `is_attention[i]` is true.
    pub is_full_attention: Vec<bool>,
    /// `n_input_layers` / `n_recurrent_layers` / `n_output_layers`.
    pub block_split: (usize, usize, usize),
    /// Declared recurrent passes per token (checkpoint `n_recurrent_loops`,
    /// 1..=4; runtime override via [`DoryConfig::apply_recurrence_overrides`]).
    pub n_recurrent_loops: usize,
    /// KV semantics across recurrent passes (checkpoint
    /// `recurrent_kv_cache_mode`, default `per_loop`).
    pub recurrent_kv_cache_mode: DoryRecurrentKvCacheMode,
    /// `ngpt_alpha_init / initializer_range` (host precompute of the
    /// per-step slerp scale in `slerp_alpha`).
    pub alpha_scale: f32,
    /// `sqrt(g0) / std` with `g0 = hidden_size` when the checkpoint's
    /// `attention_output_gate_logit_scale_init` is null.
    pub s_gate_factor: f32,
    /// `sqrt(head_dim) / std`.
    pub sqk_factor: f32,
    /// `sqrt(hidden_size)` (MLP suv scaling).
    pub suv_factor: f32,
    pub eos_token_ids: Vec<u32>,
}

fn parse_eos(val: &serde_json::Value) -> Vec<u32> {
    match val {
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|e| e.as_u64().map(|n| n as u32))
            .collect(),
        serde_json::Value::Number(n) => n.as_u64().map(|n| vec![n as u32]).unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// Parse from the raw `config.json` value (safetensors dir metadata embeds
/// the file verbatim under `config` — see `build_metadata_json`).
pub fn config_from_value(inner: &serde_json::Value) -> Result<DoryConfig, String> {
    let req = |key: &str| {
        inner
            .get(key)
            .ok_or_else(|| format!("dory: config.json missing `{key}`"))
    };
    let usize_req = |key: &str| {
        req(key)?
            .as_u64()
            .ok_or_else(|| format!("dory: config `{key}` is not a number"))
            .map(|v| v as usize)
    };
    let hidden_size = usize_req("hidden_size")?;
    let num_hidden_layers = usize_req("num_hidden_layers")?;
    let vocab_size = usize_req("vocab_size")?;
    let intermediate_size = usize_req("intermediate_size")?;
    let num_attention_heads = usize_req("num_attention_heads")?;
    let num_key_value_heads = usize_req("num_key_value_heads")?;
    let head_dim = usize_req("head_dim")?;
    let sliding_window = usize_req("sliding_window")?;
    if hidden_size == 0 || intermediate_size == 0 || sliding_window == 0 {
        return Err("dory: hidden_size/intermediate_size/sliding_window must be > 0".into());
    }
    if num_attention_heads == 0 || num_key_value_heads == 0 {
        return Err("dory: attention head counts must be > 0".into());
    }
    if num_attention_heads % num_key_value_heads != 0 {
        return Err(format!(
            "dory: num_attention_heads {} is not a multiple of num_key_value_heads {}",
            num_attention_heads, num_key_value_heads
        ));
    }
    let initializer_range = req("initializer_range")?
        .as_f64()
        .ok_or_else(|| "dory: config `initializer_range` is not a number".to_string())?
        as f32;
    if !(initializer_range > 0.0) || !initializer_range.is_finite() {
        return Err("dory: config `initializer_range` must be finite and > 0".into());
    }
    let n_input = usize_req("n_input_layers")?;
    let n_recurrent = usize_req("n_recurrent_layers")?;
    let n_output = usize_req("n_output_layers")?;
    if n_input + n_recurrent + n_output != num_hidden_layers {
        return Err(format!(
            "dory: block split {n_input}+{n_recurrent}+{n_output} != num_hidden_layers {num_hidden_layers}"
        ));
    }
    let n_recurrent_loops = usize_req("n_recurrent_loops")?;
    // Genuine multi-loop schedule: the checkpoint declares its trained
    // recurrent depth and the forward runs exactly that many passes (llama.cpp
    // PR30306 admits 1..=4). Anything else fails closed rather than silently
    // running a mismatched schedule.
    if !(1..=4).contains(&n_recurrent_loops) {
        return Err(format!(
            "dory: unsupported n_recurrent_loops={n_recurrent_loops} (admit 1..=4)"
        ));
    }
    let mt = inner
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if mt != "dory" {
        return Err(format!("dory: model_type is `{mt}` (expected `dory`)"));
    }
    let rope = req("rope_parameters")?;
    for scope in ["sliding_attention", "full_attention"] {
        let node = rope
            .get(scope)
            .ok_or_else(|| format!("dory: config `rope_parameters.{scope}` missing"))?;
        let rtype = node.get("rope_type").and_then(|v| v.as_str()).unwrap_or("");
        if rtype != "default" {
            return Err(format!(
                "dory: config `rope_parameters.{scope}.rope_type` is `{rtype}` (expected `default`)"
            ));
        }
        if let Some(prf) = node.get("partial_rotary_factor") {
            let ok = prf
                .as_f64()
                .map(|v| (v - 1.0).abs() < 1e-9)
                .unwrap_or(false);
            if !ok {
                return Err(format!(
                    "dory: config `rope_parameters.{scope}.partial_rotary_factor` must be 1.0"
                ));
            }
        }
    }
    let theta = |scope: &str| {
        rope.get(scope)
            .and_then(|v| v.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .ok_or_else(|| format!("dory: config `rope_parameters.{scope}.rope_theta` missing"))
            .and_then(|v| {
                if v.is_finite() && v > 0.0 {
                    Ok(v as f32)
                } else {
                    Err(format!(
                        "dory: config `rope_parameters.{scope}.rope_theta` must be finite and > 0"
                    ))
                }
            })
    };
    let sliding_theta = theta("sliding_attention")?;
    let full_theta = theta("full_attention")?;
    let layer_types: Vec<String> = req("layer_types")?
        .as_array()
        .ok_or_else(|| "dory: config `layer_types` is not an array".to_string())?
        .iter()
        .map(|v| {
            v.as_str()
                .ok_or_else(|| "dory: config `layer_types` entry is not a string".to_string())
                .map(str::to_string)
        })
        .collect::<Result<_, _>>()?;
    if layer_types.len() != num_hidden_layers {
        return Err(format!(
            "dory: layer_types len {} != num_hidden_layers {num_hidden_layers}",
            layer_types.len()
        ));
    }
    let mut is_attention = Vec::with_capacity(num_hidden_layers);
    let mut is_full_attention = Vec::with_capacity(num_hidden_layers);
    for (i, t) in layer_types.iter().enumerate() {
        match t.as_str() {
            "mlp" => {
                is_attention.push(false);
                is_full_attention.push(false);
            }
            "sliding_attention" => {
                is_attention.push(true);
                is_full_attention.push(false);
            }
            "full_attention" => {
                is_attention.push(true);
                is_full_attention.push(true);
            }
            other => {
                return Err(format!("dory: layer {i} has unknown layer_type `{other}`"));
            }
        }
    }
    // Admission is the checkpoint's own declared `layer_types`: a native
    // Dory layout may place full/sliding attention on any even sublayer the
    // checkpoint declares (the expanded 76-layer layout inserts pairs after
    // layer 55 and shifts the output block, so a derived i%8 cadence would
    // reject a valid model). The structural contract stays fail-closed:
    // sublayers strictly alternate attention (even) / MLP (odd), and the
    // three sections tile the layer axis in whole pairs with a nonempty
    // shared recurrent block.
    for i in 0..num_hidden_layers {
        let expect_attention = i % 2 == 0;
        if is_attention[i] != expect_attention {
            return Err(format!(
                "dory: layer {i} breaks attention/MLP alternation (`{}`, expected `{}`)",
                layer_types[i],
                if expect_attention { "attention" } else { "mlp" }
            ));
        }
    }
    if n_recurrent == 0 {
        return Err("dory: n_recurrent_layers must be > 0".into());
    }
    for (key, count) in [
        ("n_input_layers", n_input),
        ("n_recurrent_layers", n_recurrent),
        ("n_output_layers", n_output),
    ] {
        if count % 2 != 0 {
            return Err(format!(
                "dory: {key}={count} splits an attention/MLP pair; every block section must hold whole pairs"
            ));
        }
    }
    // GQA geometry of the shipped checkpoint: q = 16×256 = 4096,
    // kv = 4×256 = 1024. head_dim 256 is baked into the attention kernel
    // choice, so any other geometry fails closed here (explicit dim gate,
    // not just product checks).
    if head_dim != 256 {
        return Err(format!(
            "dory: head_dim={head_dim} unsupported (packed KV codecs need D256)"
        ));
    }
    let ngpt_alpha_init = req("ngpt_alpha_init")?
        .as_f64()
        .ok_or_else(|| "dory: config `ngpt_alpha_init` is not a number".to_string())?
        as f32;
    if !ngpt_alpha_init.is_finite() || ngpt_alpha_init <= 0.0 {
        return Err("dory: config `ngpt_alpha_init` must be finite and > 0".into());
    }
    let std = initializer_range;
    let alpha_scale = ngpt_alpha_init / std;
    let sqk_factor = (head_dim as f32).sqrt() / std;
    let g0: f32 = match inner.get("attention_output_gate_logit_scale_init") {
        None | Some(serde_json::Value::Null) => (hidden_size as f32).sqrt(),
        Some(v) => {
            let g = v.as_f64().ok_or_else(|| {
                "dory: `attention_output_gate_logit_scale_init` is not a number".to_string()
            })? as f32;
            if !g.is_finite() || g <= 0.0 {
                return Err(
                    "dory: `attention_output_gate_logit_scale_init` must be finite and > 0".into(),
                );
            }
            g
        }
    };
    let s_gate_factor = g0.sqrt() / std;
    let suv_factor = (hidden_size as f32).sqrt();
    // Bias and activation contract: the reference forward has no Q/K/V/O
    // or gate/up/down biases and uses SiLU. A checkpoint declaring
    // otherwise would silently compute different math, so refuse it.
    for key in ["attention_bias", "mlp_bias"] {
        if let Some(v) = inner.get(key) {
            let is_false = v.as_bool() == Some(false);
            if !is_false {
                return Err(format!(
                    "dory: config `{key}` must be false/absent (reference has no biases), got {v}"
                ));
            }
        }
    }
    if let Some(act) = inner.get("hidden_act").and_then(|v| v.as_str()) {
        if act != "silu" {
            return Err(format!("dory: hidden_act is `{act}` (expected `silu`)"));
        }
    }
    let tie = inner
        .get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if tie {
        return Err(
            "dory: tie_word_embeddings=true unsupported (checkpoint ships untied lm_head)".into(),
        );
    }
    // Recurrent KV semantics: upstream `recurrent_kv_cache_mode`, absent =
    // `per_loop` (upstream default). Present-but-unknown fails closed —
    // never silently run a mismatched cache schedule.
    let recurrent_kv_cache_mode = match inner.get("recurrent_kv_cache_mode") {
        None => DoryRecurrentKvCacheMode::PerLoop,
        Some(v) => {
            let s = v.as_str().ok_or_else(|| {
                "dory: config `recurrent_kv_cache_mode` must be a string (per_loop|last_loop)"
                    .to_string()
            })?;
            match s {
                "per_loop" => DoryRecurrentKvCacheMode::PerLoop,
                "last_loop" => DoryRecurrentKvCacheMode::LastLoop,
                other => {
                    return Err(format!(
                        "dory: unsupported recurrent_kv_cache_mode=`{other}` (admit per_loop|last_loop)"
                    ));
                }
            }
        }
    };
    let mut eos_token_ids: Vec<u32> = inner.get("eos_token_id").map(parse_eos).unwrap_or_default();
    if eos_token_ids.is_empty() {
        eos_token_ids = vec![2, 11];
    }
    Ok(DoryConfig {
        hidden_size,
        num_hidden_layers,
        vocab_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        sliding_window,
        sliding_theta,
        full_theta,
        initializer_range,
        is_attention,
        is_full_attention,
        block_split: (n_input, n_recurrent, n_output),
        n_recurrent_loops,
        recurrent_kv_cache_mode,
        alpha_scale,
        s_gate_factor,
        sqk_factor,
        suv_factor,
        eos_token_ids,
    })
}

/// Parse from a `ModelSource` (safetensors dir: `metadata_json` embeds
/// `config.json` under `config`).
pub fn config_from_source(source: &dyn ModelSource) -> Result<DoryConfig, String> {
    let meta: serde_json::Value = serde_json::from_str(source.metadata_json())
        .map_err(|e| format!("dory: metadata_json not valid JSON: {e}"))?;
    let inner = meta
        .get("config")
        .ok_or_else(|| "dory: metadata_json missing `config` wrapper".to_string())?;
    config_from_value(inner)
}

impl DoryConfig {
    /// Apply runtime recurrence overrides (`None` = keep checkpoint value).
    /// Counts admit 1..=4; anything else fails closed. Validates after
    /// applying so a bad checkpoint value also fails before GPU allocation.
    pub fn apply_recurrence_overrides(
        &mut self,
        loops: Option<usize>,
        mode: Option<DoryRecurrentKvCacheMode>,
    ) -> Result<(), String> {
        if let Some(n) = loops {
            if !(1..=4).contains(&n) {
                return Err(format!("dory: unsupported dory_loops={n} (admit 1..=4)"));
            }
            self.n_recurrent_loops = n;
        }
        if let Some(m) = mode {
            self.recurrent_kv_cache_mode = m;
        }
        self.validate_recurrence()
    }

    /// Fail closed on out-of-range loop counts (checkpoint or override).
    /// Call before any GPU allocation.
    pub fn validate_recurrence(&self) -> Result<(), String> {
        if !(1..=4).contains(&self.n_recurrent_loops) {
            return Err(format!(
                "dory: unsupported n_recurrent_loops={} (admit 1..=4)",
                self.n_recurrent_loops
            ));
        }
        Ok(())
    }

    /// Cache-layer index for physical sublayer `physical` on recurrent pass
    /// `loop_index` (0-based; input/output passes use `loop_index == 0`).
    ///
    /// `per_loop` uses dense virtual indexing (equivalent to upstream sparse
    /// `loop * 72` IDs): input physical `i` maps to `i`; recurrent `i` on
    /// pass `l` maps to `i + l * n_rec` (the shared physical block repeats in
    /// cache once per pass); output `i` maps to `i + (N-1) * n_rec` (shifted
    /// past every recurrent repeat). `last_loop` is the identity map: every
    /// pass overwrites the same physical slot at the current absolute pos;
    /// only the final pass's rows persist for earlier tokens.
    pub fn kv_layer_index(&self, physical: usize, loop_index: usize) -> usize {
        debug_assert!(physical < self.num_hidden_layers);
        match self.recurrent_kv_cache_mode {
            DoryRecurrentKvCacheMode::LastLoop => physical,
            DoryRecurrentKvCacheMode::PerLoop => {
                let (n_in, n_rec, _) = self.block_split;
                if physical < n_in {
                    physical
                } else if physical < n_in + n_rec {
                    physical + loop_index * n_rec
                } else {
                    physical + (self.n_recurrent_loops - 1) * n_rec
                }
            }
        }
    }

    /// Per-cache-layer attention mask over [`DoryConfig::kv_layer_count`]
    /// slots: a cache slot carries KV iff its physical sublayer is attention
    /// (MLP slots hold 1-element placeholders). Input slots map from their
    /// physical index; recurrent repeat `l` maps from physical
    /// `slot - l * n_rec`; output slots map from `slot - (N-1) * n_rec`.
    pub fn kv_layer_mask(&self) -> Vec<bool> {
        match self.recurrent_kv_cache_mode {
            DoryRecurrentKvCacheMode::LastLoop => self.is_attention.clone(),
            DoryRecurrentKvCacheMode::PerLoop => {
                let (n_in, n_rec, _) = self.block_split;
                let n = self.n_recurrent_loops;
                let count = self.kv_layer_count();
                let mut mask = Vec::with_capacity(count);
                for slot in 0..count {
                    let physical = if slot < n_in {
                        slot
                    } else if slot < n_in + n * n_rec {
                        n_in + (slot - n_in) % n_rec
                    } else {
                        slot - (n - 1) * n_rec
                    };
                    debug_assert!(physical < self.is_attention.len());
                    mask.push(self.is_attention[physical]);
                }
                mask
            }
        }
    }

    /// Total virtual cache layers: `72 + 40 * (N-1)` under `per_loop`
    /// (input 16 + N*40 recurrent + output 16), `72` under `last_loop`.
    pub fn kv_layer_count(&self) -> usize {
        match self.recurrent_kv_cache_mode {
            DoryRecurrentKvCacheMode::LastLoop => self.num_hidden_layers,
            DoryRecurrentKvCacheMode::PerLoop => {
                self.num_hidden_layers + (self.n_recurrent_loops - 1) * self.block_split.1
            }
        }
    }
}

// ─── Weights ─────────────────────────────────────────────────────────────

/// Per-layer GPU-resident Dory weights. Attention layers populate the `attn`
/// half, MLP layers the `mlp` half; the other half is `None`. Exactly one
/// half is `Some` per layer, matching `config.is_attention[i]`.
pub struct DoryAttnWeights {
    pub wq: WeightTensor,  // backbone.layers.{i}.mixer.q_proj.weight [4096, 2560]
    pub wk: WeightTensor,  // ...k_proj.weight [1024, 2560]
    pub wv: WeightTensor,  // ...v_proj.weight [1024, 2560]
    pub wo: WeightTensor,  // ...o_proj.weight [2560, 4096]
    pub wg: WeightTensor,  // ...g_proj.weight [4096, 2560]
    pub sqk: GpuTensor,    // ...sqk [4096] F32 (host-widened)
    pub s_gate: GpuTensor, // ...s_gate [4096] F32 (host-widened)
    pub alpha: GpuTensor,  // ...attn_alpha [2560] F32 (host-widened)
}

pub struct DoryMlpWeights {
    pub w_gate: WeightTensor, // ...mixer.gate_proj.weight [7168, 2560]
    pub w_up: WeightTensor,   // ...mixer.up_proj.weight   [7168, 2560]
    pub w_down: WeightTensor, // ...mixer.down_proj.weight [2560, 7168]
    pub suv_gate: GpuTensor,  // ...suv_gate [7168] F32
    pub suv_up: GpuTensor,    // ...suv_up   [7168] F32
    pub alpha: GpuTensor,     // ...mlp_alpha [2560] F32
}

pub struct DoryWeights {
    pub token_embd: GpuTensor, // backbone.embeddings.weight [131072, 2560] BF16 or Q8
    /// On-GPU embedding layout: `None` = native verbatim BF16 (forward
    /// widens the row at use); `Some(Q8_0)` = HFQ Q8 table via
    /// `embedding_lookup_dispatch`. No other variant is admitted.
    pub embd_format: Option<EmbeddingFormat>,
    pub lm_head: WeightTensor, // lm_head.weight [131072, 2560] BF16 or quant
    pub logit_scale: GpuTensor, // lm_head.logit_scale [131072] F32
    pub attn: Vec<Option<DoryAttnWeights>>,
    pub mlp: Vec<Option<DoryMlpWeights>>,
}

impl DoryWeights {
    /// Release every GPU buffer. Exhaustive field-by-field frees: a future
    /// added GPU-bearing field is a compile error here, not a silent leak.
    /// `WeightTensor`s own plain buffers here (BF16 verbatim or quant
    /// passthrough — no Paro/AWQ/Lloyd sidecars), so `free_all` frees
    /// exactly the buffer. `embd_format` is a layout tag, not an allocation.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let DoryWeights {
            token_embd,
            embd_format: _,
            lm_head,
            logit_scale,
            attn,
            mlp,
        } = self;
        let _ = gpu.free_tensor(token_embd);
        lm_head.free_all(gpu);
        let _ = gpu.free_tensor(logit_scale);
        for (a, m) in attn.into_iter().zip(mlp.into_iter()) {
            if let Some(a) = a {
                let DoryAttnWeights {
                    wq,
                    wk,
                    wv,
                    wo,
                    wg,
                    sqk,
                    s_gate,
                    alpha,
                } = a;
                wq.free_all(gpu);
                wk.free_all(gpu);
                wv.free_all(gpu);
                wo.free_all(gpu);
                wg.free_all(gpu);
                let _ = gpu.free_tensor(sqk);
                let _ = gpu.free_tensor(s_gate);
                let _ = gpu.free_tensor(alpha);
            }
            if let Some(m) = m {
                let DoryMlpWeights {
                    w_gate,
                    w_up,
                    w_down,
                    suv_gate,
                    suv_up,
                    alpha,
                } = m;
                w_gate.free_all(gpu);
                w_up.free_all(gpu);
                w_down.free_all(gpu);
                let _ = gpu.free_tensor(suv_gate);
                let _ = gpu.free_tensor(suv_up);
                let _ = gpu.free_tensor(alpha);
            }
        }
    }
}

/// Load-time ownership guard: successfully loaded pieces move into the
/// `parts` set as they land; on the first error the guard frees everything
/// loaded so far, so a failed load leaks no VRAM. Each GPU buffer has
/// exactly one owner at all times (no alias double-free).
struct LoadParts {
    token_embd: Option<GpuTensor>,
    lm_head: Option<WeightTensor>,
    logit_scale: Option<GpuTensor>,
    attn: Vec<Option<DoryAttnWeights>>,
    mlp: Vec<Option<DoryMlpWeights>>,
}

impl LoadParts {
    fn abort(self, gpu: &mut Gpu) {
        let Self {
            token_embd,
            lm_head,
            logit_scale,
            attn,
            mlp,
        } = self;
        if let Some(t) = token_embd {
            let _ = gpu.free_tensor(t);
        }
        if let Some(w) = lm_head {
            w.free_all(gpu);
        }
        if let Some(t) = logit_scale {
            let _ = gpu.free_tensor(t);
        }
        for (a, m) in attn.into_iter().zip(mlp.into_iter()) {
            if let Some(a) = a {
                let DoryAttnWeights {
                    wq,
                    wk,
                    wv,
                    wo,
                    wg,
                    sqk,
                    s_gate,
                    alpha,
                } = a;
                wq.free_all(gpu);
                wk.free_all(gpu);
                wv.free_all(gpu);
                wo.free_all(gpu);
                wg.free_all(gpu);
                let _ = gpu.free_tensor(sqk);
                let _ = gpu.free_tensor(s_gate);
                let _ = gpu.free_tensor(alpha);
            }
            if let Some(m) = m {
                let DoryMlpWeights {
                    w_gate,
                    w_up,
                    w_down,
                    suv_gate,
                    suv_up,
                    alpha,
                } = m;
                w_gate.free_all(gpu);
                w_up.free_all(gpu);
                w_down.free_all(gpu);
                let _ = gpu.free_tensor(suv_gate);
                let _ = gpu.free_tensor(suv_up);
                let _ = gpu.free_tensor(alpha);
            }
        }
    }
}

/// Load full Dory weights from a `ModelSource`. Fails closed on any missing
/// tensor, dtype, or shape.
pub fn load_weights_from_source(
    source: &dyn ModelSource,
    cfg: &DoryConfig,
    gpu: &mut Gpu,
) -> Result<DoryWeights, String> {
    // MQ2V2 (qt50) preflight: refused before any GPU allocation. The
    // quantizer no longer emits MQ2V2 Dory artifacts, but an existing
    // arch-19 qt50 pack must fail closed with an explicit message — never
    // fall through to the common decoder. `tensor_info` exposes the
    // per-tensor `quant_type` on the shared trait, so the check stays in
    // the generic `ModelSource` contract (no HFQ downcast). Only qt50 is
    // refused; native safetensors dirs carry sentinel 0xFF (never 50) and
    // all other quants flow unchanged to their codec arms below.
    for name in source.tensor_names() {
        if let Some(info) = source.tensor_info(name) {
            if info.quant_type == 50 {
                return Err(format!(
                    "dory: tensor {name} has quant_type 50=MQ2V2 (unsupported on Dory: catastrophic quality); re-export with --format mq3v2 or higher"
                ));
            }
        }
    }
    // Developer gate diagnostics (`HIPFIRE_DORY_GATE_*`) resolve here, before
    // the first allocation: an out-of-scope layout, a mismatched gate source
    // or a bad scale must abort with nothing resident. Unset knobs return the
    // stock plan and open no extra file.
    let gate = resolve_gate_plan(cfg)?;
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;
    let mut parts = LoadParts {
        token_embd: None,
        lm_head: None,
        logit_scale: None,
        attn: Vec::with_capacity(cfg.num_hidden_layers),
        mlp: Vec::with_capacity(cfg.num_hidden_layers),
    };
    // `step` runs one fallible load; on error it frees everything in
    // `parts` (moved out via `replace`) and returns the error.
    macro_rules! step {
        ($expr:expr, $assign:expr) => {{
            match $expr {
                Ok(val) => {
                    $assign = Some(val);
                }
                Err(e) => {
                    let dead = std::mem::replace(
                        &mut parts,
                        LoadParts {
                            token_embd: None,
                            lm_head: None,
                            logit_scale: None,
                            attn: Vec::new(),
                            mlp: Vec::new(),
                        },
                    );
                    dead.abort(gpu);
                    return Err(e);
                }
            }
        }};
    }
    // Embeddings: native BF16 uploads verbatim (`embd_format=None`, the
    // forward widens the row at use); HFQ tables arrive Q8 (qt=3, every
    // shipped tier) and stay resident Q8 (`Some(Q8_0)` consumed via
    // `embedding_lookup_dispatch`) — no giant dequantized float matrix.
    let mut embd_format: Option<EmbeddingFormat> = None;
    {
        let is_bf16 = source
            .tensor_info("backbone.embeddings.weight")
            .map(|i| i.dtype == "BF16")
            .unwrap_or(false);
        if is_bf16 {
            step!(
                upload_bf16_matrix(source, gpu, "backbone.embeddings.weight", v, h),
                parts.token_embd
            );
        } else {
            // First load: nothing in `parts` yet, so a failure returns
            // directly (no VRAM to reclaim).
            match load_embed_table(source, gpu, "backbone.embeddings.weight", v, h) {
                Ok((t, f)) => {
                    parts.token_embd = Some(t);
                    embd_format = Some(f);
                }
                Err(e) => return Err(e),
            }
        }
    }
    step!(
        load_linear_common(source, gpu, "lm_head.weight", v, h),
        parts.lm_head
    );
    step!(
        load_vec_f32(source, gpu, "lm_head.logit_scale", v),
        parts.logit_scale
    );
    for i in 0..cfg.num_hidden_layers {
        if cfg.is_attention[i] {
            let p = format!("backbone.layers.{i}.mixer");
            let q = cfg.num_attention_heads * cfg.head_dim;
            let kv = cfg.num_key_value_heads * cfg.head_dim;
            let wq = match load_linear_common(source, gpu, &format!("{p}.q_proj.weight"), q, h) {
                Ok(w) => w,
                Err(e) => {
                    parts.abort(gpu);
                    return Err(e);
                }
            };
            macro_rules! attn_fail {
                ($e:expr, $($free:stmt)*) => {{
                    $($free)*
                    parts.abort(gpu);
                    return Err($e);
                }};
            }
            let wk = match load_linear_common(source, gpu, &format!("{p}.k_proj.weight"), kv, h) {
                Ok(w) => w,
                Err(e) => attn_fail!(e, wq.free_all(gpu);),
            };
            let wv = match load_linear_common(source, gpu, &format!("{p}.v_proj.weight"), kv, h) {
                Ok(w) => w,
                Err(e) => attn_fail!(e, wq.free_all(gpu); wk.free_all(gpu);),
            };
            let wo = match load_linear_common(source, gpu, &format!("{p}.o_proj.weight"), h, q) {
                Ok(w) => w,
                Err(e) => attn_fail!(e, wq.free_all(gpu); wk.free_all(gpu); wv.free_all(gpu);),
            };
            let wg = match load_linear_common(source, gpu, &format!("{p}.g_proj.weight"), q, h) {
                Ok(w) => w,
                Err(e) => {
                    attn_fail!(e, wq.free_all(gpu); wk.free_all(gpu); wv.free_all(gpu); wo.free_all(gpu);)
                }
            };
            let sqk = match load_vec_f32_scaled_square(
                source,
                gpu,
                &format!("{p}.sqk"),
                q,
                cfg.sqk_factor,
            ) {
                Ok(t) => t,
                Err(e) => {
                    attn_fail!(e, wq.free_all(gpu); wk.free_all(gpu); wv.free_all(gpu); wo.free_all(gpu); wg.free_all(gpu);)
                }
            };
            let s_gate = match load_vec_f32_scaled_square(
                source,
                gpu,
                &format!("{p}.s_gate"),
                q,
                cfg.s_gate_factor,
            ) {
                Ok(t) => t,
                Err(e) => {
                    attn_fail!(e, wq.free_all(gpu); wk.free_all(gpu); wv.free_all(gpu); wo.free_all(gpu); wg.free_all(gpu); let _ = gpu.free_tensor(sqk);)
                }
            };
            let alpha = match load_alpha_vec(
                source,
                &gate,
                i,
                &format!("backbone.layers.{i}.attn_alpha"),
                h,
                cfg.alpha_scale,
                gpu,
            ) {
                Ok(t) => t,
                Err(e) => {
                    attn_fail!(e, wq.free_all(gpu); wk.free_all(gpu); wv.free_all(gpu); wo.free_all(gpu); wg.free_all(gpu); let _ = gpu.free_tensor(sqk); let _ = gpu.free_tensor(s_gate);)
                }
            };
            parts.attn.push(Some(DoryAttnWeights {
                wq,
                wk,
                wv,
                wo,
                wg,
                sqk,
                s_gate,
                alpha,
            }));
            parts.mlp.push(None);
        } else {
            let p = format!("backbone.layers.{i}.mixer");
            let f = cfg.intermediate_size;
            macro_rules! mlp_fail {
                ($e:expr, $($free:stmt)*) => {{
                    $($free)*
                    parts.abort(gpu);
                    return Err($e);
                }};
            }
            let w_gate =
                match load_linear_common(source, gpu, &format!("{p}.gate_proj.weight"), f, h) {
                    Ok(w) => w,
                    Err(e) => mlp_fail!(e,),
                };
            let w_up = match load_linear_common(source, gpu, &format!("{p}.up_proj.weight"), f, h) {
                Ok(w) => w,
                Err(e) => mlp_fail!(e, w_gate.free_all(gpu);),
            };
            let w_down =
                match load_linear_common(source, gpu, &format!("{p}.down_proj.weight"), h, f) {
                    Ok(w) => w,
                    Err(e) => mlp_fail!(e, w_gate.free_all(gpu); w_up.free_all(gpu);),
                };
            let suv_gate = match load_vec_f32_scaled(
                source,
                gpu,
                &format!("{p}.suv_gate"),
                f,
                cfg.suv_factor,
            ) {
                Ok(t) => t,
                Err(e) => {
                    mlp_fail!(e, w_gate.free_all(gpu); w_up.free_all(gpu); w_down.free_all(gpu);)
                }
            };
            let suv_up = match load_vec_f32_scaled(
                source,
                gpu,
                &format!("{p}.suv_up"),
                f,
                cfg.suv_factor,
            ) {
                Ok(t) => t,
                Err(e) => {
                    mlp_fail!(e, w_gate.free_all(gpu); w_up.free_all(gpu); w_down.free_all(gpu); let _ = gpu.free_tensor(suv_gate);)
                }
            };
            let alpha = match load_alpha_vec(
                source,
                &gate,
                i,
                &format!("backbone.layers.{i}.mlp_alpha"),
                h,
                cfg.alpha_scale,
                gpu,
            ) {
                Ok(t) => t,
                Err(e) => {
                    mlp_fail!(e, w_gate.free_all(gpu); w_up.free_all(gpu); w_down.free_all(gpu); let _ = gpu.free_tensor(suv_gate); let _ = gpu.free_tensor(suv_up);)
                }
            };
            parts.attn.push(None);
            parts.mlp.push(Some(DoryMlpWeights {
                w_gate,
                w_up,
                w_down,
                suv_gate,
                suv_up,
                alpha,
            }));
        }
    }
    let LoadParts {
        token_embd,
        lm_head,
        logit_scale,
        attn,
        mlp,
    } = parts;
    Ok(DoryWeights {
        token_embd: token_embd.expect("dory: embeddings loaded"),
        embd_format,
        lm_head: lm_head.expect("dory: lm_head loaded"),
        logit_scale: logit_scale.expect("dory: logit_scale loaded"),
        attn,
        mlp,
    })
}

/// Dory decode state: per-token GPU scratch + one packed [`KvCache`].
///
/// Buffers: residual stream `h`, `input_h` (post-input-block snapshot),
/// `state_h` (recurrent accumulator), `resid`, `out`, `tmp` (hidden),
/// `q`/`k`/`v`, `gate`, `attn_out`, `gate_mlp`/`up`/`ffn_mid`, `logits`,
/// `attn_partials` (flash reduce scratch), `pos_buf` (absolute position for
/// RoPE + KV write + full-attention reads), `swa_pos_buf` (sliding `len-1`
/// for windowed reads — tiles derive `seq_len` from the buffer given), plus
/// one packed `KvCache` over [`DoryConfig::kv_layer_count`] virtual cache
/// slots (MLP slots hold 1-element placeholders; address by
/// [`DoryConfig::kv_layer_index`], not physical `layer_idx` — the virtual
/// slot repeats the shared recurrent block once per pass under `per_loop`,
/// and aliases physical layers under `last_loop`).
///
/// `max_seq` is the logical KV budget at allocation. Stale KV tails are
/// ignored: `reset()` only rewinds `next_pos`; slots are overwritten in place.
/// `kv_mode` is the resolved selection (reported at load); `kv_backend`
/// names the honored storage backend (legacy contiguous or VMM).
pub struct DoryState {
    pub h: GpuTensor,
    pub input_h: GpuTensor,
    pub state_h: GpuTensor,
    pub resid: GpuTensor,
    pub out: GpuTensor,
    pub tmp: GpuTensor,
    pub q: GpuTensor,
    pub k: GpuTensor,
    pub v: GpuTensor,
    pub gate: GpuTensor,
    pub attn_out: GpuTensor,
    pub gate_mlp: GpuTensor,
    pub up: GpuTensor,
    pub ffn_mid: GpuTensor,
    pub logits: GpuTensor,
    pub attn_partials: GpuTensor,
    pub pos_buf: hip_bridge::DeviceBuffer,
    pub swa_pos_buf: hip_bridge::DeviceBuffer,
    pub kv: saddle_core::kv::KvCache,
    pub kv_mode: saddle_core::kv::KvMode,
    pub kv_backend: hipfire_runtime::kv_backend::KvBackend,
    pub max_seq: usize,
    /// Next free KV slot / absolute position of the next forward step.
    /// Bumped by the forward pass; rewound by `reset()`.
    pub next_pos: usize,
    /// LAB-ONLY native KV trace sink. `None` (default) = no capture, no
    /// download, no branch cost beyond one `is_some` check per attention
    /// layer (field absent entirely without the `lab` feature). The parity
    /// driver sets `Some(Vec::new())` to capture exact pre-cache `q`/`k`/`v`
    /// and post-attend `attention` per (position, layer); see
    /// [`crate::kv::KvTraceRow`]. Never read by production code.
    #[cfg(feature = "lab")]
    pub kv_trace: Option<Vec<crate::kv::KvTraceRow>>,
}

/// Default KV budget: 2048 positions × 36 KV-carrying slots at N=1 (more under
/// `per_loop`: input 8 + 20*N recurrent + 8 output attention caches);
/// packed Q8 ≈ 1.4 MB/layer (1088 B K+V/pos), FWHT tiers smaller.
pub const DEFAULT_MAX_SEQ: usize = 2048;

impl DoryState {
    pub fn new(gpu: &mut Gpu, cfg: &DoryConfig) -> Result<Self, String> {
        Self::new_with_max_seq(gpu, cfg, DEFAULT_MAX_SEQ)
    }

    /// Plain-F32 reference constructor for the internal numeric oracle
    /// (lab-gated). Same geometry as production, `quantized == false`;
    /// production never builds through this path.
    pub fn new_f32_reference(
        gpu: &mut Gpu,
        cfg: &DoryConfig,
        max_seq: usize,
    ) -> Result<Self, String> {
        Self::new_inner(
            gpu,
            cfg,
            max_seq,
            saddle_core::kv::KvMode::Q8,
            hipfire_runtime::kv_backend::KvBackend::Legacy,
            true,
        )
    }

    pub fn new_with_max_seq(
        gpu: &mut Gpu,
        cfg: &DoryConfig,
        max_seq: usize,
    ) -> Result<Self, String> {
        Self::new_with_kv(
            gpu,
            cfg,
            max_seq,
            saddle_core::kv::KvMode::Q8,
            hipfire_runtime::kv_backend::KvBackend::Legacy,
        )
    }

    /// Production constructor: build the packed `KvCache` for `kv_mode` on
    /// `backend` over the resolved [`DoryConfig::kv_layer_mask`] (virtual
    /// cache slots, not physical layers), with failure-safe teardown of
    /// every owner on the first error.
    pub fn new_with_kv(
        gpu: &mut Gpu,
        cfg: &DoryConfig,
        max_seq: usize,
        kv_mode: saddle_core::kv::KvMode,
        kv_backend: hipfire_runtime::kv_backend::KvBackend,
    ) -> Result<Self, String> {
        Self::new_inner(gpu, cfg, max_seq, kv_mode, kv_backend, false)
    }

    fn new_inner(
        gpu: &mut Gpu,
        cfg: &DoryConfig,
        max_seq: usize,
        kv_mode: saddle_core::kv::KvMode,
        kv_backend: hipfire_runtime::kv_backend::KvBackend,
        f32_reference: bool,
    ) -> Result<Self, String> {
        cfg.validate_recurrence()?;
        if max_seq == 0 {
            return Err("dory: max_seq must be > 0".into());
        }
        if cfg.head_dim != 256 {
            return Err(format!(
                "dory: head_dim={} unsupported for KV (packed codecs need D256)",
                cfg.head_dim
            ));
        }
        let h = cfg.hidden_size;
        let f = cfg.intermediate_size;
        let q = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        // Virtual cache mask over `kv_layer_count()` slots (input once,
        // recurrent block repeated per pass under per_loop, output once;
        // identity under last_loop). The F32 reference rides the same
        // resolved mask so its slots agree with production indexing.
        let is_kv_layer: Vec<bool> = cfg.kv_layer_mask();
        // Worst-case flash scratch: the FWHT/Q8 tile kernels read up to the
        // smallest certified tile (16) in the size ladder, so size partials
        // for tile 16 regardless of the arch default (oversize is harmless).
        let n_chunks_max = max_seq.div_ceil(16);
        let attn_partials_len = cfg.num_attention_heads * n_chunks_max * (2 + cfg.head_dim);
        // Build the packed KV owner FIRST: every later allocation rolls back
        // through it on failure (legacy frees, VMM tears down arenas).
        let mut kv = if f32_reference {
            saddle_core::kv::KvCache::new_gpu(
                gpu,
                cfg.kv_layer_count(),
                cfg.num_key_value_heads,
                cfg.head_dim,
                max_seq,
            )
            .map_err(|e| format!("dory: F32 reference KV alloc failed: {e}"))?
        } else {
            let dims = saddle_core::kv::KvDims {
                layers: saddle_core::kv::KvLayers::Mask(is_kv_layer.clone()),
                n_kv_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
                max_seq,
                physical_cap: Some(max_seq),
            };
            saddle_core::kv::KvCache::from_mode_with_backend(kv_mode, kv_backend, gpu, &dims)
                .map_err(|e| {
                    format!("dory: packed KV alloc failed ({kv_mode:?}/{kv_backend:?}): {e}")
                })?
        };
        let pos_buf = match gpu.hip.malloc(4) {
            Ok(b) => b,
            Err(e) => {
                let _ = kv.free_gpu(gpu);
                return Err(format!("dory: state pos_buf alloc failed: {e}"));
            }
        };
        let swa_pos_buf = match gpu.hip.malloc(4) {
            Ok(b) => b,
            Err(e) => {
                let _ = kv.free_gpu(gpu);
                let _ = gpu.hip.free(pos_buf);
                return Err(format!("dory: state swa_pos_buf alloc failed: {e}"));
            }
        };
        // Scratch allocations: collect into a vec so a late failure frees the
        // KV owner + pos bufs + every scratch buffer so far.
        let mut scratch: Vec<GpuTensor> = Vec::with_capacity(16);
        macro_rules! salloc {
            ($alloc:expr) => {
                match $alloc {
                    Ok(t) => scratch.push(t),
                    Err(e) => {
                        let _ = kv.free_gpu(gpu);
                        let _ = gpu.hip.free(pos_buf);
                        let _ = gpu.hip.free(swa_pos_buf);
                        for t in scratch {
                            let _ = gpu.free_tensor(t);
                        }
                        return Err(format!("dory: state alloc failed: {e}"));
                    }
                }
            };
        }
        salloc!(gpu.alloc_tensor(&[h], DType::F32)); // h
        salloc!(gpu.alloc_tensor(&[h], DType::F32)); // input_h
        salloc!(gpu.alloc_tensor(&[h], DType::F32)); // state_h
        salloc!(gpu.alloc_tensor(&[h], DType::F32)); // resid
        salloc!(gpu.alloc_tensor(&[h], DType::F32)); // out
        salloc!(gpu.alloc_tensor(&[h], DType::F32)); // tmp
        salloc!(gpu.alloc_tensor(&[q], DType::F32)); // q
        salloc!(gpu.alloc_tensor(&[kv_dim], DType::F32)); // k
        salloc!(gpu.alloc_tensor(&[kv_dim], DType::F32)); // v
        salloc!(gpu.alloc_tensor(&[q], DType::F32)); // gate
        salloc!(gpu.alloc_tensor(&[q], DType::F32)); // attn_out
        salloc!(gpu.alloc_tensor(&[f], DType::F32)); // gate_mlp
        salloc!(gpu.alloc_tensor(&[f], DType::F32)); // up
        salloc!(gpu.alloc_tensor(&[f], DType::F32)); // ffn_mid
        salloc!(gpu.alloc_tensor(&[cfg.vocab_size], DType::F32)); // logits
        salloc!(gpu.alloc_tensor(&[attn_partials_len], DType::F32)); // attn_partials
        let mut it = scratch.into_iter();
        let h_b = it.next().expect("16 scratch");
        let input_h = it.next().expect("16 scratch");
        let state_h = it.next().expect("16 scratch");
        let resid = it.next().expect("16 scratch");
        let out = it.next().expect("16 scratch");
        let tmp = it.next().expect("16 scratch");
        let qb = it.next().expect("16 scratch");
        let kb = it.next().expect("16 scratch");
        let vb = it.next().expect("16 scratch");
        let gate = it.next().expect("16 scratch");
        let attn_out = it.next().expect("16 scratch");
        let gate_mlp = it.next().expect("16 scratch");
        let up = it.next().expect("16 scratch");
        let ffn_mid = it.next().expect("16 scratch");
        let logits = it.next().expect("16 scratch");
        let attn_partials = it.next().expect("16 scratch");
        // Zero the sliding position buffer to a valid `len-1 = 0` view: a fresh
        // state must never hand an uninitialized buffer to a windowed read.
        let zero: i32 = 0;
        if let Err(e) = gpu.hip.memcpy_htod(&swa_pos_buf, &zero.to_ne_bytes()) {
            let _ = kv.free_gpu(gpu);
            let _ = gpu.hip.free(pos_buf);
            let _ = gpu.hip.free(swa_pos_buf);
            for t in [
                h_b,
                input_h,
                state_h,
                resid,
                out,
                tmp,
                qb,
                kb,
                vb,
                gate,
                attn_out,
                gate_mlp,
                up,
                ffn_mid,
                logits,
                attn_partials,
            ] {
                let _ = gpu.free_tensor(t);
            }
            return Err(format!("dory: state swa_pos_buf init failed: {e}"));
        }
        Ok(Self {
            h: h_b,
            input_h,
            state_h,
            resid,
            out,
            tmp,
            q: qb,
            k: kb,
            v: vb,
            gate,
            attn_out,
            gate_mlp,
            up,
            ffn_mid,
            logits,
            attn_partials,
            pos_buf,
            swa_pos_buf,
            kv,
            kv_mode,
            kv_backend,
            max_seq,
            next_pos: 0,
            #[cfg(feature = "lab")]
            kv_trace: None,
        })
    }

    /// Fresh-prompt reset: rewind `next_pos` to 0. KV buffers are not
    /// zeroed — slots are overwritten in place as the next prompt fills
    /// positions from 0, and the forward only reads `[0, next_pos)` (full)
    /// or the trailing window (sliding). `compact_offset` is always 0 on
    /// Dory (no eviction), so no stale-tail mapping can leak across resets.
    pub fn reset(&mut self) {
        self.next_pos = 0;
        self.kv.compact_offset = 0;
    }

    /// Release every GPU buffer. Consumes self. The packed `KvCache` frees
    /// its arenas (legacy) or tears down VMM mappings; pos buffers and all
    /// scratch follow — exhaustive field-by-field frees.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let Self {
            h,
            input_h,
            state_h,
            resid,
            out,
            tmp,
            q,
            k,
            v,
            gate,
            attn_out,
            gate_mlp,
            up,
            ffn_mid,
            logits,
            attn_partials,
            pos_buf,
            swa_pos_buf,
            kv,
            kv_mode: _,
            kv_backend: _,
            max_seq: _,
            next_pos: _,
            #[cfg(feature = "lab")]
                kv_trace: _,
        } = self;
        for t in [
            h,
            input_h,
            state_h,
            resid,
            out,
            tmp,
            q,
            k,
            v,
            gate,
            attn_out,
            gate_mlp,
            up,
            ffn_mid,
            logits,
            attn_partials,
        ] {
            let _ = gpu.free_tensor(t);
        }
        let _ = kv.free_gpu(gpu);
        let _ = gpu.hip.free(pos_buf);
        let _ = gpu.hip.free(swa_pos_buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_config_json() -> serde_json::Value {
        let mut layer_types = Vec::new();
        for i in 0..72usize {
            if i % 2 == 1 {
                layer_types.push(serde_json::Value::String("mlp".into()));
            } else if i % 8 == 6 || i == 70 {
                layer_types.push(serde_json::Value::String("full_attention".into()));
            } else {
                layer_types.push(serde_json::Value::String("sliding_attention".into()));
            }
        }
        serde_json::json!({
            "model_type": "dory",
            "vocab_size": 131072,
            "hidden_size": 2560,
            "num_hidden_layers": 72,
            "intermediate_size": 7168,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "sliding_window": 1024,
            "initializer_range": 0.0177,
            "ngpt_alpha_init": 0.05,
            "n_input_layers": 16,
            "n_recurrent_layers": 40,
            "n_output_layers": 16,
            "n_recurrent_loops": 2,
            "tie_word_embeddings": false,
            "rope_parameters": {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {"rope_type": "default", "rope_theta": 10000000.0}
            },
            "layer_types": layer_types,
        })
    }

    /// Stock 72-layer config with two native attention/MLP pairs inserted
    /// after physical 55 (new 56..59, copies of 52..55) and the former
    /// 56..71 shifted to 60..75 — the expanded 76-layer / 44-recurrent
    /// layout. Full attention lands at 6..54 step 8, 58, 66 and 74.
    fn expanded_76_config_json() -> serde_json::Value {
        let mut v = minimal_config_json();
        let mut lt = v["layer_types"].as_array().unwrap().clone();
        let tail = lt.split_off(56);
        for t in ["sliding_attention", "mlp", "full_attention", "mlp"] {
            lt.push(serde_json::Value::String(t.into()));
        }
        lt.extend(tail);
        v["num_hidden_layers"] = serde_json::json!(76);
        v["n_recurrent_layers"] = serde_json::json!(44);
        v["layer_types"] = serde_json::Value::Array(lt);
        v
    }

    #[test]
    fn shipped_topology_admits_with_checkpoint_loops_and_default_per_loop() {
        let cfg = config_from_value(&minimal_config_json()).expect("admits");
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_hidden_layers, 72);
        assert_eq!(cfg.vocab_size, 131072);
        assert_eq!(cfg.block_split, (16, 40, 16));
        assert_eq!(cfg.n_recurrent_loops, 2);
        assert_eq!(
            cfg.recurrent_kv_cache_mode,
            DoryRecurrentKvCacheMode::PerLoop
        );
        assert_eq!(cfg.sliding_window, 1024);
        assert!((cfg.alpha_scale - 0.05 / 0.0177).abs() < 1e-6);
        assert_eq!(cfg.is_attention.iter().filter(|&&b| b).count(), 36);
    }

    #[test]
    fn loop_counts_one_through_four_admit() {
        for loops in [1, 2, 3, 4] {
            let mut v = minimal_config_json();
            v["n_recurrent_loops"] = serde_json::json!(loops);
            let cfg = config_from_value(&v).expect("admits");
            assert_eq!(cfg.n_recurrent_loops, loops);
        }
        for bad in [0, 5, 8] {
            let mut v = minimal_config_json();
            v["n_recurrent_loops"] = serde_json::json!(bad);
            assert!(config_from_value(&v).is_err(), "loops={bad}");
        }
    }

    #[test]
    fn nonzero_dims_and_gqa_multiple_required() {
        let mut v = minimal_config_json();
        v["hidden_size"] = serde_json::json!(0);
        assert!(config_from_value(&v).is_err());
        let mut v = minimal_config_json();
        v["num_attention_heads"] = serde_json::json!(15);
        assert!(config_from_value(&v).is_err());
        let mut v = minimal_config_json();
        v["head_dim"] = serde_json::json!(128);
        assert!(config_from_value(&v).is_err());
    }

    #[test]
    fn bad_thetas_bias_act_rejected() {
        let mut v = minimal_config_json();
        v["rope_parameters"]["sliding_attention"]["rope_theta"] = serde_json::json!(-1.0);
        assert!(config_from_value(&v).is_err());
        let mut v = minimal_config_json();
        v["rope_parameters"]["full_attention"]["rope_type"] =
            serde_json::Value::String("yarn".into());
        assert!(config_from_value(&v).is_err());
        let mut v = minimal_config_json();
        v["attention_bias"] = serde_json::json!(true);
        assert!(config_from_value(&v).is_err());
        let mut v = minimal_config_json();
        v["hidden_act"] = serde_json::Value::String("gelu".into());
        assert!(config_from_value(&v).is_err());
        let mut v = minimal_config_json();
        v["initializer_range"] = serde_json::json!(0.0);
        assert!(config_from_value(&v).is_err());
    }

    #[test]
    fn declared_sliding_full_kinds_are_authoritative() {
        // The checkpoint's declared `layer_types` decide sliding vs full;
        // no index-derived cadence may override or reject them.
        let mut v = minimal_config_json();
        v["layer_types"][6] = serde_json::Value::String("sliding_attention".into());
        let cfg = config_from_value(&v).expect("admits declared sliding");
        assert!(!cfg.is_full_attention[6]);
        let mut v = minimal_config_json();
        v["layer_types"][8] = serde_json::Value::String("full_attention".into());
        let cfg = config_from_value(&v).expect("admits declared full");
        assert!(cfg.is_full_attention[8]);
    }

    #[test]
    fn expanded_76_layer_layout_admits_and_keeps_declared_kinds() {
        let v = expanded_76_config_json();
        let cfg = config_from_value(&v).expect("expanded layout admits");
        assert_eq!(cfg.num_hidden_layers, 76);
        assert_eq!(cfg.block_split, (16, 44, 16));
        // Exact declared kinds survive at the inserted pairs and the shifted
        // output block.
        let declared: Vec<bool> = v["layer_types"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str() == Some("full_attention"))
            .collect();
        assert_eq!(cfg.is_full_attention, declared);
        assert!(!cfg.is_full_attention[56]);
        assert!(cfg.is_full_attention[58]);
        assert!(!cfg.is_full_attention[62]);
        assert!(cfg.is_full_attention[66] && cfg.is_full_attention[74]);
        assert_eq!(
            cfg.is_attention.iter().filter(|&&b| b).count(),
            38,
            "even sublayers are the 38 attention mixers"
        );
        // Cache sizing and indexing follow the declared counts: 38 + 22*N-1
        // attention slots over 76 + 44*N-1 virtual layers.
        for loops in 1..=4usize {
            let mut v = expanded_76_config_json();
            v["n_recurrent_loops"] = serde_json::json!(loops);
            let c = config_from_value(&v).expect("admits");
            assert_eq!(c.kv_layer_count(), 76 + 44 * (loops - 1));
            let mask = c.kv_layer_mask();
            assert_eq!(mask.len(), c.kv_layer_count());
            assert_eq!(mask.iter().filter(|&&b| b).count(), 38 + 22 * (loops - 1));
            for i in 0..16usize {
                assert_eq!(c.kv_layer_index(i, 0), i);
                assert_eq!(c.kv_layer_index(60 + i, 0), 60 + i + (loops - 1) * 44);
            }
            for l in 0..loops {
                for i in 16..60usize {
                    assert_eq!(c.kv_layer_index(i, l), i + l * 44);
                }
            }
        }
    }

    #[test]
    fn malformed_expanded_layouts_fail_closed() {
        // Odd section counts split an attention/MLP pair.
        let mut v = expanded_76_config_json();
        v["n_input_layers"] = serde_json::json!(15);
        v["n_output_layers"] = serde_json::json!(17);
        let e = config_from_value(&v).expect_err("odd input section");
        assert!(e.contains("n_input_layers=15"), "{e}");
        let mut v = expanded_76_config_json();
        v["n_recurrent_layers"] = serde_json::json!(43);
        v["n_output_layers"] = serde_json::json!(17);
        let e = config_from_value(&v).expect_err("odd recurrent section");
        assert!(e.contains("n_recurrent_layers=43"), "{e}");
        let mut v = expanded_76_config_json();
        v["n_input_layers"] = serde_json::json!(17);
        v["n_output_layers"] = serde_json::json!(15);
        let e = config_from_value(&v).expect_err("odd output section");
        assert!(e.contains("splits an attention/MLP pair"), "{e}");
        // The shared recurrent block must not be empty.
        let mut v = expanded_76_config_json();
        v["n_recurrent_layers"] = serde_json::json!(0);
        v["n_output_layers"] = serde_json::json!(60);
        let e = config_from_value(&v).expect_err("empty recurrent block");
        assert!(e.contains("n_recurrent_layers must be > 0"), "{e}");
        // Alternation holds in both directions.
        let mut v = expanded_76_config_json();
        v["layer_types"][57] = serde_json::Value::String("sliding_attention".into());
        let e = config_from_value(&v).expect_err("attention at odd index");
        assert!(e.contains("alternation"), "{e}");
        let mut v = expanded_76_config_json();
        v["layer_types"][58] = serde_json::Value::String("mlp".into());
        let e = config_from_value(&v).expect_err("mlp at even index");
        assert!(e.contains("alternation"), "{e}");
        // Declared length and kinds stay checked.
        let mut v = expanded_76_config_json();
        v["layer_types"].as_array_mut().unwrap().pop();
        let e = config_from_value(&v).expect_err("short layer_types");
        assert!(e.contains("layer_types len 75"), "{e}");
        let mut v = expanded_76_config_json();
        v["layer_types"][56] = serde_json::Value::String("mamba".into());
        let e = config_from_value(&v).expect_err("unknown kind");
        assert!(e.contains("unknown layer_type"), "{e}");
    }

    #[test]
    fn tied_embeddings_fail_closed() {
        let mut v = minimal_config_json();
        v["tie_word_embeddings"] = serde_json::json!(true);
        assert!(config_from_value(&v).is_err());
    }

    #[test]
    fn bad_recurrent_kv_cache_mode_fails_closed() {
        let mut v = minimal_config_json();
        v["recurrent_kv_cache_mode"] = serde_json::Value::String("sparse".into());
        assert!(config_from_value(&v).is_err());
        let mut v = minimal_config_json();
        v["recurrent_kv_cache_mode"] = serde_json::json!(2);
        assert!(config_from_value(&v).is_err());
    }

    #[test]
    fn last_loop_mode_parses() {
        let mut v = minimal_config_json();
        v["recurrent_kv_cache_mode"] = serde_json::Value::String("last_loop".into());
        let cfg = config_from_value(&v).expect("admits");
        assert_eq!(
            cfg.recurrent_kv_cache_mode,
            DoryRecurrentKvCacheMode::LastLoop
        );
    }

    fn per_loop_cfg(loops: usize) -> DoryConfig {
        let mut v = minimal_config_json();
        v["n_recurrent_loops"] = serde_json::json!(loops);
        config_from_value(&v).expect("admits")
    }

    fn last_loop_cfg(loops: usize) -> DoryConfig {
        let mut v = minimal_config_json();
        v["n_recurrent_loops"] = serde_json::json!(loops);
        v["recurrent_kv_cache_mode"] = serde_json::Value::String("last_loop".into());
        config_from_value(&v).expect("admits")
    }

    #[test]
    fn per_loop_recurrent_slots_unique_input_output_noalias() {
        // Shipped split: 16 input / 40 recurrent / 16 output physical, 20
        // attention sublayers per recurrent repeat.
        for loops in [1usize, 2, 3, 4] {
            let cfg = per_loop_cfg(loops);
            assert_eq!(cfg.kv_layer_count(), 72 + 40 * (loops - 1));
            let mask = cfg.kv_layer_mask();
            assert_eq!(mask.len(), cfg.kv_layer_count());
            assert_eq!(mask.iter().filter(|&&b| b).count(), 36 + 20 * (loops - 1));
            // Input/output cache slots never alias recurrent slots.
            for i in 0..16usize {
                assert_eq!(cfg.kv_layer_index(i, 0), i);
                assert_eq!(cfg.kv_layer_index(56 + i, 0), 56 + i + (loops - 1) * 40);
            }
            // Recurrent repeats are pairwise disjoint across passes.
            let mut seen = std::collections::HashSet::new();
            for l in 0..loops {
                for i in 16..56usize {
                    let slot = cfg.kv_layer_index(i, l);
                    assert_eq!(slot, i + l * 40);
                    assert!(seen.insert(slot), "loops={loops} slot={slot}");
                }
            }
        }
    }

    #[test]
    fn per_loop_mask_slots_track_physical_attention() {
        let cfg = per_loop_cfg(3);
        let mask = cfg.kv_layer_mask();
        // Input region mirrors physical 0..16.
        for i in 0..16usize {
            assert_eq!(mask[i], cfg.is_attention[i]);
        }
        // Each recurrent repeat mirrors physical 16..56: even slot offsets
        // (physical even sublayers) carry attention.
        for l in 0..3usize {
            for o in 0..40usize {
                assert_eq!(mask[16 + l * 40 + o], cfg.is_attention[16 + o]);
            }
        }
        // Output region mirrors physical 56..72 shifted past the repeats.
        for i in 0..16usize {
            assert_eq!(mask[16 + 3 * 40 + i], cfg.is_attention[56 + i]);
        }
    }

    #[test]
    fn last_loop_shares_only_recurrent_slots() {
        for loops in [1usize, 2, 3, 4] {
            let cfg = last_loop_cfg(loops);
            assert_eq!(cfg.kv_layer_count(), 72);
            assert_eq!(cfg.kv_layer_mask(), cfg.is_attention);
            // Identity map for every layer and pass: every recurrent pass
            // overwrites the same physical slot at the current position.
            for l in 0..loops {
                for i in 0..72usize {
                    assert_eq!(cfg.kv_layer_index(i, l), i);
                }
            }
        }
    }

    #[test]
    fn kv_layer_counts_match_36_56_76_96_attention_slots() {
        // Attention slots: 8 input + 20*N recurrent + 8 output.
        for (loops, slots) in [(1usize, 36usize), (2, 56), (3, 76), (4, 96)] {
            let per = per_loop_cfg(loops);
            assert_eq!(per.kv_layer_count(), 72 + 40 * (loops - 1));
            assert_eq!(per.kv_layer_mask().iter().filter(|&&b| b).count(), slots);
            let last = last_loop_cfg(loops);
            assert_eq!(last.kv_layer_count(), 72);
            assert_eq!(last.kv_layer_mask().iter().filter(|&&b| b).count(), 36);
        }
    }

    #[test]
    fn recurrence_override_boundaries() {
        let mut cfg = per_loop_cfg(2);
        cfg.apply_recurrence_overrides(Some(4), None)
            .expect("admits");
        assert_eq!(cfg.n_recurrent_loops, 4);
        assert_eq!(cfg.kv_layer_count(), 72 + 40 * 3);
        cfg.apply_recurrence_overrides(Some(1), Some(DoryRecurrentKvCacheMode::LastLoop))
            .expect("admits");
        assert_eq!(cfg.n_recurrent_loops, 1);
        assert_eq!(
            cfg.recurrent_kv_cache_mode,
            DoryRecurrentKvCacheMode::LastLoop
        );
        assert_eq!(cfg.kv_layer_count(), 72);
        for bad in [0, 5] {
            assert!(
                cfg.apply_recurrence_overrides(Some(bad), None).is_err(),
                "loops={bad}"
            );
        }
        // None/None keeps checkpoint values.
        cfg.apply_recurrence_overrides(None, None).expect("admits");
        assert_eq!(cfg.n_recurrent_loops, 1);
    }

    #[test]
    fn mask_len_always_matches_count_across_modes_and_overrides() {
        // Sizing is a pure function of (split, loops, mode): `reset()` only
        // rewinds `next_pos`/`compact_offset` and never reallocates, so the
        // mask length must equal the count for every reachable config —
        // otherwise a post-reset forward would address freed slots.
        for loops in [1usize, 2, 3, 4] {
            for mode in [
                DoryRecurrentKvCacheMode::PerLoop,
                DoryRecurrentKvCacheMode::LastLoop,
            ] {
                let mut cfg = per_loop_cfg(loops);
                cfg.apply_recurrence_overrides(None, Some(mode))
                    .expect("admits");
                assert_eq!(cfg.kv_layer_mask().len(), cfg.kv_layer_count());
            }
        }
    }

    #[test]
    fn non_recurrent_slots_ignore_loop_index_recurrent_do_not() {
        // `kv_layer_index` takes `loop_index` only for the shared recurrent
        // block: input/output slots are loop-independent under `per_loop`
        // (and everything is under `last_loop`). A recurrent slot that
        // ignored its pass would alias histories across passes; a
        // non-recurrent slot that moved with the pass would strand writes
        // outside the input/output caches.
        for loops in [1usize, 2, 3, 4] {
            let per = per_loop_cfg(loops);
            for l in 0..loops {
                for i in 0..16usize {
                    assert_eq!(per.kv_layer_index(i, l), i);
                    assert_eq!(per.kv_layer_index(56 + i, l), 56 + i + (loops - 1) * 40);
                }
                for i in 16..56usize {
                    assert_eq!(per.kv_layer_index(i, l), i + l * 40);
                }
            }
            let last = last_loop_cfg(loops);
            for l in 0..loops {
                for i in 0..72usize {
                    assert_eq!(last.kv_layer_index(i, l), i);
                }
            }
        }
    }

    // ─── Developer alpha-gate diagnostics ────────────────────────────────

    /// Stage4 layout: four attention/MLP pairs inserted after physical 55
    /// (the new 56..63 region `HIPFIRE_DORY_GATE_SOURCE` substitutes), the
    /// stock 56..71 tail shifted to 64..79.
    fn stage4_80_config_json() -> serde_json::Value {
        let mut v = minimal_config_json();
        let mut lt = v["layer_types"].as_array().unwrap().clone();
        let tail = lt.split_off(56);
        for t in [
            "sliding_attention",
            "mlp",
            "full_attention",
            "mlp",
            "sliding_attention",
            "mlp",
            "sliding_attention",
            "mlp",
        ] {
            lt.push(serde_json::Value::String(t.into()));
        }
        lt.extend(tail);
        v["num_hidden_layers"] = serde_json::json!(80);
        v["n_recurrent_layers"] = serde_json::json!(48);
        v["layer_types"] = serde_json::Value::Array(lt);
        v
    }

    #[test]
    fn stage4_inserted_layers_are_the_expansion_pairs_only() {
        assert_eq!(stage4_inserted_layers(), 56..64);
    }

    #[test]
    fn gate_scale_scope_is_the_inserted_region_only() {
        let plan = DoryGatePlan {
            scale: 2.0,
            replacements: Vec::new(),
        };
        // Inside the inserted region the diagnostic multiplies the effective
        // gate; outside it the multiplier is exactly 1, so the stock
        // expression runs unchanged — including the adjacent stock
        // sublayers at 55 (last stock recurrent) and 64 (first shifted
        // output sublayer).
        assert_eq!(plan.scale_for(56), 2.0);
        assert_eq!(plan.scale_for(63), 2.0);
        for physical in [0, 16, 55, 64, 79, 80] {
            assert_eq!(plan.scale_for(physical), 1.0, "physical {physical}");
        }
        // Composed through the fold: 55 is bit-identical to the pre-ablation
        // expression, 56 takes the scaled coefficient.
        let alpha_scale = 0.05f32 / 0.0177f32;
        let raw = 0.0006681768f32;
        assert_eq!(
            scale_alpha(raw, alpha_scale, plan.scale_for(55)).to_bits(),
            (raw * alpha_scale).abs().to_bits()
        );
        assert_eq!(
            scale_alpha(raw, alpha_scale, plan.scale_for(56)),
            (raw * alpha_scale).abs() * 2.0
        );
        // An explicit scale of 1 (still a validated diagnostic) touches
        // nothing anywhere.
        let neutral = DoryGatePlan {
            scale: 1.0,
            replacements: Vec::new(),
        };
        for physical in [55, 56, 63, 64] {
            assert_eq!(neutral.scale_for(physical), 1.0, "physical {physical}");
        }
        assert_eq!(DoryGatePlan::stock().scale_for(56), 1.0);
    }

    #[test]
    fn gate_scale_parse_admits_nonnegative_finite_only() {
        for (raw, want) in [("0", 0.0f32), ("1", 1.0), ("2.5", 2.5), (" 0.125 ", 0.125)] {
            assert_eq!(
                parse_gate_scale(raw).expect(raw).to_bits(),
                want.to_bits(),
                "raw={raw:?}"
            );
        }
        for bad in [
            "", "-1", "-0.5", "nan", "NaN", "inf", "-inf", "1e999", "abc", "1,0",
        ] {
            let err = parse_gate_scale(bad).expect_err(&format!("raw={bad:?}"));
            assert!(err.contains(DORY_GATE_SCALE_ENV), "{err}");
        }
    }

    #[test]
    fn gate_source_parse_trims_and_treats_blank_as_unset() {
        assert_eq!(parse_gate_source(""), None);
        assert_eq!(parse_gate_source("  \t "), None);
        assert_eq!(
            parse_gate_source(" /tmp/gates.mq8 "),
            Some(std::path::PathBuf::from("/tmp/gates.mq8"))
        );
    }

    #[test]
    fn effective_gate_transform_preserves_stock_expression_and_boundaries() {
        // Shipped alpha semantics: ngpt_alpha_init 0.05 / initializer_range
        // 0.0177, folded to f32 at config parse.
        let alpha_scale = 0.05f32 / 0.0177f32;
        // The untrained Stage4 gate vector: raw 3.54e-06 (exact-F32 HFQ
        // scale), effective |raw * alpha_scale| = 1.0000000656873453e-05.
        assert_eq!(
            scale_alpha(3.54e-06, alpha_scale, 1.0).to_bits(),
            0x3727_c5ad
        );
        // scale == 1 is the pre-ablation expression bit-for-bit, both signs
        // and the signed-zero input included.
        for v in [-0.0029907227f32, -3.54e-06, 0.0, 0.12695312] {
            assert_eq!(
                scale_alpha(v, alpha_scale, 1.0).to_bits(),
                (v * alpha_scale).abs().to_bits(),
                "v={v}"
            );
        }
        // scale == 0 zeroes the coefficient exactly (the forward still runs
        // residual normalization and KV; the layer is not skipped).
        assert_eq!(
            scale_alpha(-0.12695312, alpha_scale, 0.0).to_bits(),
            0.0f32.to_bits()
        );
        // Any other scale multiplies AFTER the abs, never the raw alpha.
        let v = -0.12695312f32;
        assert_eq!(
            scale_alpha(v, alpha_scale, 2.0),
            (v * alpha_scale).abs() * 2.0
        );
        assert_eq!(
            scale_alpha(v, alpha_scale, 0.25),
            (v * alpha_scale).abs() * 0.25
        );
    }

    #[test]
    fn gate_source_layout_gate_admits_only_stage4_expansion() {
        let stock = config_from_value(&minimal_config_json()).expect("stock admits");
        let err =
            validate_stage4_gate_layout(&stock).expect_err("72-sublayer stock is out of scope");
        assert!(err.contains(DORY_GATE_SOURCE_ENV), "{err}");
        assert!(err.contains("Stage4"), "{err}");
        let wider = config_from_value(&expanded_76_config_json()).expect("76 admits");
        assert!(validate_stage4_gate_layout(&wider).is_err());
        let stage4 = config_from_value(&stage4_80_config_json()).expect("stage4 admits");
        assert_eq!(stage4.block_split, (16, 48, 16));
        validate_stage4_gate_layout(&stage4).expect("the Stage4 layout is the admitted one");
    }

    #[test]
    fn gate_source_must_match_model_geometry_and_alpha_semantics() {
        let main = config_from_value(&stage4_80_config_json()).expect("stage4 admits");
        let same = config_from_value(&stage4_80_config_json()).expect("stage4 admits");
        validate_gate_source_geometry(&main, &same).expect("identical geometry admits");

        let mut v = stage4_80_config_json();
        v["intermediate_size"] = serde_json::json!(6144);
        let bad = config_from_value(&v).expect("a geometry-only change still parses");
        let err =
            validate_gate_source_geometry(&main, &bad).expect_err("intermediate_size differs");
        assert!(err.contains("intermediate_size"), "{err}");

        let mut v = stage4_80_config_json();
        v["ngpt_alpha_init"] = serde_json::json!(0.04);
        let bad = config_from_value(&v).expect("an alpha-init change still parses");
        let err = validate_gate_source_geometry(&main, &bad).expect_err("alpha_scale differs");
        assert!(err.contains("alpha semantics"), "{err}");
    }

    #[test]
    fn gate_replacement_scope_is_the_inserted_region_only() {
        let mut replacements: Vec<Option<Vec<f32>>> = vec![None; 80];
        for physical in stage4_inserted_layers() {
            replacements[physical] = Some(vec![3.54e-06; 3]);
        }
        let plan = DoryGatePlan {
            scale: 1.0,
            replacements,
        };
        for physical in stage4_inserted_layers() {
            assert_eq!(
                plan.replacement(physical).map(<[f32]>::len),
                Some(3),
                "physical {physical}"
            );
        }
        // Stock sublayers and out-of-range indices never pick up a
        // substitution: 55 is the last stock recurrent sublayer, 64 the
        // first shifted stock output sublayer.
        for physical in [0, 16, 55, 64, 79, 80] {
            assert!(plan.replacement(physical).is_none(), "physical {physical}");
        }
        let stock = DoryGatePlan::stock();
        assert_eq!(stock.scale, 1.0);
        assert!(stock.replacement(56).is_none());
    }

    /// Minimal in-memory `ModelSource` for the per-vector admission checks.
    struct VecSource {
        tensors: Vec<(TensorInfo, Vec<u8>)>,
    }

    impl ModelSource for VecSource {
        fn metadata_json(&self) -> &str {
            "{}"
        }
        fn arch_id(&self) -> u32 {
            19
        }
        fn quant_config(&self) -> Option<&hipfire_runtime::model_source::QuantConfig> {
            None
        }
        fn tensor_data(&self, name: &str) -> Option<(&TensorInfo, &[u8])> {
            self.tensors
                .iter()
                .find(|(info, _)| info.name == name)
                .map(|(info, bytes)| (info, bytes.as_slice()))
        }
        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.tensors
                .iter()
                .find(|(info, _)| info.name == name)
                .map(|(info, _)| info)
        }
        fn tensor_names(&self) -> Vec<&str> {
            self.tensors
                .iter()
                .map(|(info, _)| info.name.as_str())
                .collect()
        }
        fn path(&self) -> &std::path::Path {
            std::path::Path::new("<memory>")
        }
    }

    fn vec_source(name: &str, dtype: &str, shape: Vec<usize>, bytes: Vec<u8>) -> VecSource {
        VecSource {
            tensors: vec![(
                TensorInfo {
                    name: name.to_string(),
                    dtype: dtype.to_string(),
                    shape,
                    quant_type: 2,
                    data_offset: 0,
                    data_size: bytes.len(),
                },
                bytes,
            )],
        }
    }

    #[test]
    fn alpha_vector_admission_requires_1d_bf16_or_f32() {
        let f32_bytes: Vec<u8> = [0.5f32, -3.54e-06, 0.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let src = vec_source("a.alpha", "F32", vec![3], f32_bytes);
        assert_eq!(
            decode_vec_f32(&src, "a.alpha", 3).expect("admits"),
            vec![0.5, -3.54e-06, 0.0]
        );
        // A 2-D matrix is not an nGPT per-vector scale.
        let src = vec_source("a.alpha", "F32", vec![1, 3], vec![0u8; 12]);
        assert!(decode_vec_f32(&src, "a.alpha", 3).is_err());
        // Wrong length for the declared hidden width.
        let src = vec_source("a.alpha", "F32", vec![2], vec![0u8; 12]);
        assert!(decode_vec_f32(&src, "a.alpha", 3).is_err());
        // Only the two exactly-widenable dtypes.
        let src = vec_source("a.alpha", "F16", vec![3], vec![0u8; 6]);
        assert!(decode_vec_f32(&src, "a.alpha", 3).is_err());
        // Payload length must match the dtype width.
        let src = vec_source("a.alpha", "BF16", vec![3], vec![0u8; 4]);
        assert!(decode_vec_f32(&src, "a.alpha", 3).is_err());
        // Missing tensor.
        let src = vec_source("a.alpha", "F32", vec![3], vec![0u8; 12]);
        assert!(decode_vec_f32(&src, "b.alpha", 3).is_err());
        // BF16 widens to the upper F32 bits (exact, no rounding).
        let src = vec_source("a.alpha", "BF16", vec![1], 0x3f80u16.to_le_bytes().to_vec());
        assert_eq!(
            decode_vec_f32(&src, "a.alpha", 1).expect("admits"),
            vec![1.0]
        );
    }
}
