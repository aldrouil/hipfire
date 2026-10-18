// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Owned Dory model bundle: config + weights + decode state + eos.
//! Defined here (not in `dory.rs`) for the same reason as MiniMax: the
//! `SpecTarget` seam pattern keeps bundle ownership in the arch crate.

use crate::dory::{DoryConfig, DoryState, DoryWeights};

/// Owned Dory bundle — config + weights + state + eos.
pub struct DoryBundle {
    pub config: DoryConfig,
    pub weights: DoryWeights,
    pub state: DoryState,
    pub eos_tok: u32,
}
