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

    fn config_from_hfq(_hfq: &HfqFile) -> Result<Self::Config, String> {
        Err("dory: HFQ unsupported (native BF16 safetensors dir only)".into())
    }

    fn load_weights(
        _hfq: &mut HfqFile,
        _cfg: &Self::Config,
        _gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        Err("dory: HFQ unsupported (native BF16 safetensors dir only)".into())
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        DoryState::new(gpu, cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dory_arch_id_and_name() {
        assert_eq!(Dory::arch_id(), 19);
        assert_eq!(Dory::name(), "dory");
    }
}
