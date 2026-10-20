// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Dory config, weights, and decode state.
//!
//! Model: `DoryForCausalLM` (`model_type: "dory"`, upstream reference
//! `vllm/model_executor/models/dory.py` at
//! `https://github.com/vllm-project/vllm/blob/55076950b6c155f384d6766d63563c3bc20a0e3f/vllm/model_executor/models/dory.py`).
//! 72 physical sublayers strictly alternate attention / MLP (config
//! `layer_types`: every even index is `sliding_attention`/`full_attention`,
//! every odd index `mlp`). Each layer holds exactly ONE mixer plus its
//! `*_alpha` interpolation scale — not the standard attention+FFN block.
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
//! Naming: local safetensors tensors are `backbone.*`; the vLLM reference
//! packs QKV (`qkv_proj`) and gate/up (`gate_up_proj`) where the checkpoint
//! stores split `q/k/v_proj` and `gate/up_proj`. The loader maps exact local
//! names (never the packed names) and fails closed on any shape/dtype miss.

use hipfire_config::DoryRecurrentKvCacheMode;
use hipfire_runtime::llama::{EmbeddingFormat, WeightTensor};
use hipfire_runtime::model_source::ModelSource;
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
    let (info, data) = source
        .tensor_data(name)
        .ok_or_else(|| format!("dory: tensor not found: {name}"))?;
    if info.shape != vec![n] {
        return Err(format!(
            "dory: tensor {name} has shape {:?} (expected [{n}])",
            info.shape
        ));
    }
    // Native BF16 dir bytes widen exactly (BF16 = upper F32 bits); HFQ
    // nGPT scales ride exact F32 (qt=2) and are consumed verbatim.
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
    } else if info.dtype == "F32" {
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
    } else {
        return Err(format!(
            "dory: tensor {name} has dtype {} (expected BF16 or F32)",
            info.dtype
        ));
    };
    debug_assert_eq!(raw.len(), n);
    let vals: Vec<f32> = raw.into_iter().map(f).collect();
    gpu.upload_f32(&vals, &[n])
        .map_err(|e| format!("dory: upload {name}: {e}"))
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
    // Admission: exact shipped topology. Full attention sits at every
    // i%8==6 (6..62 step 8, spanning all three blocks) plus layer 70.
    for i in 0..num_hidden_layers {
        let expect_full = is_attention[i] && (i % 8 == 6 || i == 70);
        if is_attention[i] && is_full_attention[i] != expect_full {
            return Err(format!(
                "dory: layer {i} full/sliding kind differs from shipped topology"
            ));
        }
        if !is_attention[i] && (i % 2 != 1) {
            return Err(format!("dory: layer {i} breaks attention/MLP alternation"));
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
            let alpha = match load_vec_f32_scaled_abs(
                source,
                gpu,
                &format!("backbone.layers.{i}.attn_alpha"),
                h,
                cfg.alpha_scale,
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
            let alpha = match load_vec_f32_scaled_abs(
                source,
                gpu,
                &format!("backbone.layers.{i}.mlp_alpha"),
                h,
                cfg.alpha_scale,
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
    fn swapped_sliding_full_fails_closed() {
        let mut v = minimal_config_json();
        v["layer_types"][6] = serde_json::Value::String("sliding_attention".into());
        assert!(config_from_value(&v).is_err());
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
}
