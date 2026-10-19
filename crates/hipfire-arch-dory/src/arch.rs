// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Architecture` trait marker for Dory (arch_id = 19). Thin delegation to
//! the real config/weight/state constructors in [`crate::dory`]; the hot
//! forward path is free functions in [`crate::forward`] (static dispatch).

use crate::dory::{DoryConfig, DoryState, DoryWeights};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

/// Zero-sized marker for the Dory arch.
pub struct Dory;

impl Architecture for Dory {
    type Weights = DoryWeights;
    type State = DoryState;
    type Config = DoryConfig;

    /// Canonical family marker: arch_id = 19 (`docs/architecture-ids.md`
    /// reserves 19 for `dory`).
    fn arch_id() -> u32 {
        19
    }

    fn name() -> &'static str {
        "dory"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        if hfq.arch_id != 19 {
            return Err(format!("dory: HFQ arch_id={} (expected 19)", hfq.arch_id));
        }
        // Config rides the already-open metadata JSON — parse it in place
        // (same `config` wrapper `config_from_source` reads). No reopen,
        // no mmap, no index materialization just to read config.
        let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
            .map_err(|e| format!("dory: HFQ metadata_json not valid JSON: {e}"))?;
        let inner = meta
            .get("config")
            .ok_or_else(|| "dory: HFQ metadata_json missing `config` wrapper".to_string())?;
        crate::dory::config_from_value(inner)
            .map_err(|e| format!("dory: failed to parse Dory config from HFQ metadata: {e}"))
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        // The owned bridge needs the file by value (`HfqFile` is not
        // `Clone`), so reopen the pack at its path and hand that to the
        // shared source loader. Same bytes, same index.
        let owned = HfqFile::open(hfq.path()).map_err(|e| format!("dory: reopen HFQ: {e:?}"))?;
        let bridge = hipfire_runtime::hfq::HfqModelSource::from_hfq(owned);
        crate::dory::load_weights_from_source(&bridge, cfg, gpu)
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        DoryState::new_with_kv(
            gpu,
            cfg,
            crate::dory::DEFAULT_MAX_SEQ,
            saddle_core::kv::KvMode::Q8,
            hipfire_runtime::kv_backend::KvBackend::Legacy,
        )
    }
}
