// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `load_dory_bundle` — the exact-signature entry point the loader's
//! `Carrier` calls into. Native BF16 safetensors dirs and native HFQM
//! quantized packs (`I->R^N->O`: one input block, N shared recurrent passes
//! with input reinjection, one output block; `next_pos` advances once per
//! token).
//! Recurrence policy is explicit: the checkpoint default is 2 loops with
//! `per_loop` KV; `LoadCtx::{dory_loops, dory_recurrent_kv_cache_mode}`
//! (`model.dory_loops` 1..=4, `model.dory_recurrent_kv_cache_mode`
//! auto/per_loop/last_loop) resolve against the checkpoint config before
//! weights/state, and the effective loops/mode are logged at load.
use crate::dory::{config_from_source, load_weights_from_source, DoryState};
use crate::kv::dory_kv_backend;
use crate::kv::resolve_dory_kv;
use crate::spec_impl::DoryBundle;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};
use hipfire_runtime::model_source::ModelSource as _;

/// Resolve the primary EOS token id for Dory.
///
/// Priority (Dory-specific, no Qwen hardcoded ids):
/// 1. `tokenizer_config.json` `eos_token` string encoded through
///    `tokenizer.json` (`"<|im_end|>"` → single id 11 on this checkpoint).
/// 2. `generation_config.json` `eos_token_id`, preferring 11 when listed
///    (shipped `[2, 11]`), else the head.
/// 3. `tokenizer.json` `eos_id` heuristic.
fn resolve_eos_tok(
    dir: &std::path::Path,
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
) -> u32 {
    if let Ok(s) = std::fs::read_to_string(dir.join("tokenizer_config.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
            if let Some(marker) = v.get("eos_token").and_then(|v| v.as_str()) {
                let ids = tokenizer.encode(marker);
                if ids.len() == 1 {
                    return ids[0];
                }
            }
        }
    }
    if let Ok(s) = std::fs::read_to_string(dir.join("generation_config.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
            if let Some(eos) = v.get("eos_token_id") {
                let ids: Vec<u32> = match eos {
                    serde_json::Value::Array(arr) => arr
                        .iter()
                        .filter_map(|e| e.as_u64().map(|n| n as u32))
                        .collect(),
                    serde_json::Value::Number(n) => {
                        n.as_u64().map(|n| vec![n as u32]).unwrap_or_default()
                    }
                    _ => Vec::new(),
                };
                if ids.contains(&11) {
                    return 11;
                }
                if let Some(first) = ids.first() {
                    return *first;
                }
            }
        }
    }
    tokenizer.eos_id
}
/// Resolve EOS from HFQ metadata: same priority as the Dir path, but the
/// `tokenizer_config` / `generation_config` JSON objects come from the
/// pack's metadata envelope instead of sibling files.
fn resolve_eos_tok_hfq(
    bridge: &hipfire_runtime::hfq::HfqModelSource,
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
) -> u32 {
    let meta: serde_json::Value =
        serde_json::from_str(bridge.metadata_json()).unwrap_or(serde_json::Value::Null);
    if let Some(marker) = meta
        .get("tokenizer_config")
        .and_then(|v| v.get("eos_token"))
        .and_then(|v| v.as_str())
    {
        let ids = tokenizer.encode(marker);
        if ids.len() == 1 {
            return ids[0];
        }
    }
    if let Some(eos) = meta
        .get("generation_config")
        .and_then(|v| v.get("eos_token_id"))
    {
        let ids: Vec<u32> = match eos {
            serde_json::Value::Array(arr) => arr
                .iter()
                .filter_map(|e| e.as_u64().map(|n| n as u32))
                .collect(),
            serde_json::Value::Number(n) => n.as_u64().map(|n| vec![n as u32]).unwrap_or_default(),
            _ => Vec::new(),
        };
        if ids.contains(&11) {
            return 11;
        }
        if let Some(first) = ids.first() {
            return *first;
        }
    }
    tokenizer.eos_id
}

/// Build the Dory GPU bundle from a model source.
///
/// Dir: parses `config.json` via the source metadata, uploads native BF16
/// weights. HFQ: parses the `config` wrapper out of the HFQM metadata
/// (self-contained: config + tokenizer + chat template + generation EOS
/// ride the pack, no neighboring BF16 files), uploads the quant codecs
/// through the shared weight backends. Both allocate `DoryState` at
/// `ctx.max_seq`. Recurrence (`LoadCtx::{dory_loops,
/// dory_recurrent_kv_cache_mode}`) applies to the parsed checkpoint config
/// BEFORE weights/state — an invalid override fails with no GPU allocation.
/// Refuses pipeline-parallel, CASK, and draft/speculator sources — none
/// apply to the dense multi-loop forward.
pub fn load_dory_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<DoryBundle, String> {
    if ctx.pp > 1 {
        return Err("dory: pipeline-parallel (pp>1) unsupported".into());
    }
    if ctx.draft_path.is_some() {
        return Err("dory: DFlash draft unsupported on arch_id=19 (dory bring-up)".into());
    }
    if ctx.cask.sidecar.is_some() {
        return Err("dory: CASK eviction unsupported on arch_id=19 (dory bring-up)".into());
    }
    // KV surface: whole `--kv-mode` string plus the existing axis overrides,
    // resolved over the shared `DORY_POLICY` (`auto`/empty → Q8 on every
    // arch; FWHT K always pairs with Q8 V; deprecated aliases and native
    // presets refuse — never silently folded). VMM is REFUSED for Dory
    // (admission: `carrier dory has no VMM KV owner`; explicit `--kv-backend
    // vmm` errors there, `auto` falls back to legacy) — legacy contiguous
    // `*_filtered` allocation below is the only path; there is no reserve
    // vs budget distinction and no eviction on Dory.
    let mode_raw = ctx
        .kv_mode_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| hipfire_runtime::config::get().kv_mode.clone());
    let kv_mode = resolve_dory_kv(
        &mode_raw,
        ctx.kv_k_override,
        ctx.kv_v_override,
        ctx.gpu.arch.as_str(),
        ctx.qwen_default_q8,
    )?;
    let kv_backend = dory_kv_backend(ctx);
    // Snapshot the load-time recurrence request exactly once: no per-token
    // config/env reads. Threads through to BOTH paths below; each path
    // applies it to its own parsed config before weights/state, so an
    // invalid override fails before any GPU allocation.
    let dory_loops = ctx.dory_loops;
    let dory_kv_cache_mode = ctx.dory_recurrent_kv_cache_mode;
    match src {
        ModelSource::Hfq(hfq) => {
            // HFQ path: the pack is self-contained (config + tokenizer +
            // chat template + generation EOS in its metadata; weights +
            // exact-F32 nGPT scales as tensors). Wrap the owned file in the
            // `HfqModelSource` bridge — bare `HfqFile::tensor_data`
            // returns `None` through the generic trait — then share the
            // exact Dir load path (config admission, codec uploads, packed
            // KV state). `arch_id` is checked before any GPU allocation.
            if hfq.arch_id != crate::ARCH_ID {
                return Err(format!(
                    "dory: HFQ arch_id={} (expected {})",
                    hfq.arch_id,
                    crate::ARCH_ID
                ));
            }
            let bridge = hipfire_runtime::hfq::HfqModelSource::from_hfq(hfq);
            // Parse everything fallible BEFORE any GPU allocation: config,
            // then tokenizer + EOS. Recurrence overrides resolve against the
            // checkpoint config here; weights allocate next; if state
            // allocation then fails, weights are freed — no leak on any
            // post-weights error path.
            let mut config = config_from_source(&bridge)
                .map_err(|e| format!("dory: failed to parse Dory config from HFQ metadata: {e}"))?;
            config
                .apply_recurrence_overrides(dory_loops, dory_kv_cache_mode)
                .map_err(|e| format!("dory: {e}"))?;
            eprintln!(
                "dory: recurrence: loops={} kv_cache_mode={} (I->R^N->O: one input block, N shared recurrent passes, one output block; 36 transformer blocks = 72 physical sublayers)",
                config.n_recurrent_loops, config.recurrent_kv_cache_mode,
            );
            let tokenizer =
                hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(bridge.metadata_json())
                    .map_err(|e| format!("dory: failed to parse HFQ tokenizer: {e}"))?;
            let eos_tok = resolve_eos_tok_hfq(&bridge, &tokenizer);
            let weights = load_weights_from_source(&bridge, &config, ctx.gpu)?;
            let state =
                match DoryState::new_with_kv(ctx.gpu, &config, ctx.max_seq, kv_mode, kv_backend) {
                    Ok(s) => s,
                    Err(e) => {
                        weights.free_gpu(ctx.gpu);
                        return Err(format!("dory: DoryState::new_with_kv failed: {e}"));
                    }
                };
            eprintln!(
                "dory: KV cache: requested mode={mode_raw}, effective KV={kv_mode:?} backend={kv_backend:?} kv_layers={}",
                config.kv_layer_count(),
            );
            Ok(DoryBundle {
                config,
                weights,
                state,
                eos_tok,
            })
        }
        ModelSource::Dir(source) => {
            // Parse everything fallible BEFORE any GPU allocation: config,
            // then tokenizer + EOS. Recurrence overrides resolve against the
            // checkpoint config here; weights allocate next; if state
            // allocation then fails, weights are freed — no leak on any
            // post-weights error path.
            let mut config = config_from_source(&source)
                .map_err(|e| format!("dory: failed to parse Dory config from config.json: {e}"))?;
            config
                .apply_recurrence_overrides(dory_loops, dory_kv_cache_mode)
                .map_err(|e| format!("dory: {e}"))?;
            eprintln!(
                "dory: recurrence: loops={} kv_cache_mode={} (I->R^N->O: one input block, N shared recurrent passes, one output block; 36 transformer blocks = 72 physical sublayers)",
                config.n_recurrent_loops, config.recurrent_kv_cache_mode,
            );
            let tok_path = source.tokenizer_json_path().ok_or_else(|| {
                format!(
                    "dory: no tokenizer.json found in {}",
                    source.path().display()
                )
            })?;
            let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_tokenizer_json(&tok_path)
                .map_err(|e| format!("dory: failed to parse tokenizer: {e}"))?
                .ok_or_else(|| {
                    format!("dory: failed to load tokenizer from {}", tok_path.display())
                })?;
            let eos_tok = resolve_eos_tok(source.path(), &tokenizer);
            let weights = load_weights_from_source(&source, &config, ctx.gpu)?;
            let state =
                match DoryState::new_with_kv(ctx.gpu, &config, ctx.max_seq, kv_mode, kv_backend) {
                    Ok(s) => s,
                    Err(e) => {
                        weights.free_gpu(ctx.gpu);
                        return Err(format!("dory: DoryState::new_with_kv failed: {e}"));
                    }
                };
            for name in source.tensor_names() {
                source.release_tensor_pages(name);
            }
            eprintln!(
                "dory: KV cache: requested mode={mode_raw}, effective KV={kv_mode:?} backend={kv_backend:?} kv_layers={}",
                config.kv_layer_count(),
            );
            Ok(DoryBundle {
                config,
                weights,
                state,
                eos_tok,
            })
        }
    }
}
