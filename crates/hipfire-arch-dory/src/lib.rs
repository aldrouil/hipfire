// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-arch-dory: Dory dense recurrent text decoder.
//!
//! Model: `DoryForCausalLM` (`model_type: "dory"`), 72 physical sublayers
//! strictly alternating attention / MLP (config `layer_types`: every even
//! index is `sliding_attention` or `full_attention`, every odd index `mlp`).
//! Each layer holds exactly ONE mixer plus its `*_alpha` nGPT interpolation
//! scale — not the standard transformer attention+FFN block.
//!
//! Math (upstream `vllm/model_executor/models/dory.py` at
//! `https://github.com/vllm-project/vllm/blob/55076950b6c155f384d6766d63563c3bc20a0e3f/vllm/model_executor/models/dory.py`):
//! nGPT `justnorm` (`x*rsqrt(sum(x²)+1e-6)`), headwise qk-norm AFTER halfsplit
//! RoPE with `Q *= (sqk*sqrt(hd)/std)²`, K plain; sigmoid output gate with
//! `(s_gate*sqrt(g0)/std)²` then whole-vector `justnorm`; SwiGLU MLP with
//! `suv*sqrt(hidden)` scalings then `justnorm`; slerp residual
//! `justnorm(norm(a)+|alpha*alpha_init/std|*(norm(b)-norm(a)))`; input
//! reinjection (`combined = justnorm(state+input_h)`) at the recurrent
//! start and `state = justnorm(h)` at the recurrent end; untied lm head
//! with `logits * logit_scale/initializer_range` and no final norm.
//!
//! Bring-up policy: exactly ONE recurrent pass over
//! `[n_input, n_input+n_recurrent)` (declared `n_recurrent_loops` admits 1
//! or 2). Native BF16 weights stay BF16 on device (`gemv_bf16_xf32`);
//! HFQM quants ride the shared quant codecs; embeddings stay resident
//! (BF16 verbatim / Q8 lookup). Activations are F32; KV is packed (see
//! `crate::kv`: `auto`/Q8 default, FWHT2/3/4, F32 reference variant only).
//!
//! arch_id = 19, arch_key = "dory". See `docs/architecture-ids.md`.

pub mod arch;
pub mod arch_model;
pub mod carrier;
pub mod dory;
pub mod forward;
pub mod kv;
pub mod spec_impl;

pub use arch::Dory;
pub use carrier::load_dory_bundle;
pub use forward::{download_last_hidden, forward_step, forward_token};
#[cfg(feature = "lab")]
pub use kv::KvTraceRow;
pub use kv::{dory_kv_attend, dory_kv_prepare, dory_kv_write, resolve_dory_kv};
pub use spec_impl::DoryBundle;

pub const ARCH_ID: u32 = 19;
