// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `ArchModel` impl for the Dory bundle (the loader's arch-agnostic view).

use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::llama::KvCache;
use rdna_compute::Gpu;

use crate::spec_impl::DoryBundle;

impl ArchModel for DoryBundle {
    fn dim(&self) -> usize {
        self.config.hidden_size
    }

    fn n_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn arch_key(&self) -> &'static str {
        "dory"
    }

    /// Dory owns its KV inside `DoryState` (not a `KvCache` field of its own):
    /// `None` — same pattern as `MuseGlimmerBundle`. The loader admission
    /// (`admission.rs`) routes `--kv-k/--kv-v` to Qwen only; the Dory carrier
    /// honors them only when present in `LoadCtx` (never re-reads env).
    fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        None
    }

    /// Fresh-prompt reset: rewind `next_pos`; stale KV tail is overwritten
    /// in place by the next prompt (never read past `next_pos`).
    fn reset_session_state(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        self.state.reset();
        Ok(())
    }

    /// Release every GPU allocation owned by weights and state.
    fn free_gpu(self: Box<Self>, gpu: &mut Gpu) {
        let DoryBundle {
            config: _,
            weights,
            state,
            eos_tok: _,
        } = *self;
        state.free_gpu(gpu);
        weights.free_gpu(gpu);
    }
}
