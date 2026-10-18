// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `load_dory_bundle` — the exact-signature entry point the loader's
//! `Carrier` calls into. Dir (native BF16 safetensors) only; HFQ refused.
//! Single-loop policy is explicit: the loader logs that the shipped
//! `n_recurrent_loops=2` checkpoint is intentionally run once.

use crate::dory::{config_from_source, load_weights_from_source, DoryState};
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

/// Build the Dory GPU bundle from a model source.
///
/// Dir only: parses `config.json` via the source metadata, uploads native
/// BF16 weights, allocates `DoryState` at `ctx.max_seq`. Refuses HFQ,
/// pipeline-parallel, CASK, and draft/speculator sources — none apply to
/// the dense single-loop bring-up.
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
    match src {
        ModelSource::Hfq(_) => {
            Err("dory: HFQ unsupported (native BF16 safetensors dir only)".into())
        }
        ModelSource::Dir(source) => {
            // Parse everything fallible BEFORE any GPU allocation: config,
            // then tokenizer + EOS. Weights allocate next; if state
            // allocation then fails, weights are freed — no leak on any
            // post-weights error path.
            let config = config_from_source(&source)
                .map_err(|e| format!("dory: failed to parse Dory config from config.json: {e}"))?;
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
            eprintln!(
                "dory: one-loop bring-up: running exactly ONE recurrent pass (declared n_recurrent_loops admitted: 1 or 2)"
            );
            let weights = load_weights_from_source(&source, &config, ctx.gpu)?;
            let state = match DoryState::new_with_max_seq(ctx.gpu, &config, ctx.max_seq) {
                Ok(s) => s,
                Err(e) => {
                    weights.free_gpu(ctx.gpu);
                    return Err(format!("dory: DoryState::new_with_max_seq failed: {e}"));
                }
            };
            for name in source.tensor_names() {
                source.release_tensor_pages(name);
            }
            Ok(DoryBundle {
                config,
                weights,
                state,
                eos_tok,
            })
        }
    }
}
