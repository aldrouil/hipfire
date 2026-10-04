// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! CPU execution of host-mapped weight steps (`memory.offload_exec=cpu`).
//!
//! Partial offload ([`hipfire_config::memory::gpu_layer_budget`]) places whole
//! layers in host-mapped system RAM, but by default the GPU kernels still
//! dereference those bytes — once per token, across PCIe. With
//! `memory.offload_exec=cpu` the steps whose weight tensor is host-mapped
//! execute on the CPU instead, so a spilled layer's per-token cost is bound by
//! the link (27.1 GB/s measured on the host in
//! `docs/perf-checkpoints/2026-09-26-llamacpp-offload-scaling-baseline.md`)
//! rather than by device DRAM — which is what llama.cpp's CPU backend buys with
//! its `-ngl` spill.
//!
//! Scope is deliberately the *weight-reading* ops only — [`Step::Gemv`] and
//! [`Step::GemvResidual`], plus the one op family that fused them
//! (`hipfire_runtime::llama::weight_gemv_swiglu_residual`, which splits itself
//! and calls [`run_host_mapped_gemv_residual`]). Attention, softmax, rmsnorm,
//! RoPE, qk-norm, the KV write, the flash-attention families and the DeltaNet
//! recurrence stay on the GPU, and so does the KV cache's residency: this
//! changes *who multiplies*, never *what is spilled*.
//!
//! The numerical contract is llama.cpp-level, not bit-identity — the reference
//! behaviour for this feature is llama.cpp's own partial offload, which is not
//! numerically transparent either (same record, "Correctness gate"). Acceptance
//! is coherence plus task-correct output plus a *measured* divergence; the
//! per-format device-vs-CPU distances are recorded in
//! `crates/hipfire-arch-qwen35/tests/gpu_gemv_parity.rs`.
//!
//! Two properties of the launcher this path has to reproduce exactly, because
//! getting either wrong produces plausible-looking wrong numbers rather than an
//! error:
//!
//! * **Rotation.** `Raw` inputs are FWHT-rotated exactly when the dtype's
//!   `dtype_rotation_plan` says so; the weights are stored post-rotation.
//! * **AWQ.** A weight carrying an `awq_scale` sidecar was pre-scaled by `s` at
//!   quantize time, and the *rotate* step (not the GEMV) divides the activation
//!   by it. That applies to a `Raw` input we rotate ourselves, and equally to
//!   the fused down-projection this path splits. It must **not** be applied
//!   again to a `Prerotated` input, whose producer already did it.
//!
//! Known gap, stated rather than papered over: the launcher *verifies* a
//! `Prerotated` buffer's rotation tag (`plan` + `awq`) and errors on a mismatch,
//! while a bare `GpuTensor` carries no tag, so a CPU step cannot make that check
//! for a pre-rotated input. A mismatch is a producer bug that fails loudly on
//! the GPU arms of the same model, which is why the check was not duplicated
//! into the step representation.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

use rdna_compute::{DType, Gpu, GpuTensor};

use hipfire_cpu::block_i8::BlockI8_128;
use hipfire_cpu::epilogue::{residual_add, silu_mul};
use hipfire_cpu::gemv::gemv;
use hipfire_cpu::quant::{divide_by_awq_scale, rotate_x, CpuQuant};

use crate::families::gemv::WeightRef;
use crate::pipeline::steps::{GemvInput, Step};
use crate::types::{dtype_rotation_plan, DispatchError, RotationPlan};

/// Cached `memory.offload_exec == Cpu`, mirroring the `forward_lowered_enabled`
/// precedent: resolved once from the process snapshot, never re-read per step.
pub fn cpu_exec_enabled() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        hipfire_config::memory::offload_exec() == hipfire_config::memory::OffloadExec::Cpu
    });
    *ENABLED
}

/// Whether the CPU expert splice is armed: `memory.offload_exec=cpu` and the
/// `HIPFIRE_MOE_CPU_EXPERTS=0` kill-switch unset. This is the *model-level*
/// decision; the loader additionally requires a host-placed packed layer with no
/// AWQ sidecar, and records the armed sink on the layer weights.
pub fn moe_cpu_experts_enabled() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        cpu_exec_enabled()
            && hipfire_config::developer_var("HIPFIRE_MOE_CPU_EXPERTS")
                .ok()
                .as_deref()
                != Some("0")
    });
    *ENABLED
}

/// (steps executed on the CPU, host-mapped steps that stayed on the GPU).
///
/// The second number is the honest coverage failure signal: it counts steps whose
/// weight tensor *was* host-mapped while CPU execution was enabled but whose
/// shape never reached the CPU — either an unsupported quant format or a fused
/// launch the seam did not split. Zero means every host-mapped step this process
/// executed went to the CPU.
static CPU_STEPS: AtomicUsize = AtomicUsize::new(0);
static HOST_MAPPED_GPU_STEPS: AtomicUsize = AtomicUsize::new(0);

pub fn cpu_exec_counters() -> (usize, usize) {
    (
        CPU_STEPS.load(Ordering::Relaxed),
        HOST_MAPPED_GPU_STEPS.load(Ordering::Relaxed),
    )
}

/// Per-step wall-time split, in nanoseconds: device→host, the GEMV itself,
/// host→device — keyed by step *shape*, not summed across the process.
///
/// The structural risk of this feature is that a CPU step is a host sync point,
/// so it serializes against the surrounding GPU work; these three numbers say
/// whether a slow `cpu` arm is paying for copies or for arithmetic. Keying them
/// by shape is the whole point: a process-wide sum divided by the total step
/// count is a *cumulative mean* that every shape reports identically (and that
/// early cold steps inflate), which reads as per-shape attribution and is not.
#[derive(Clone, Copy, Default)]
struct StepStats {
    calls: u64,
    d2h_ns: u64,
    gemv_ns: u64,
    h2d_ns: u64,
}

/// `(quant, m, k, rotated, residual, awq)` → running per-shape totals.
static SHAPES: LazyLock<Mutex<BTreeMap<(u8, usize, usize, bool, bool, bool), StepStats>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// Per-shape totals for the decode MoE expert splice: the d2h / gemv / h2d split
/// of one `(quant, dim, mi, k)` shape's calls.
#[derive(Clone, Copy, Default)]
struct MoeStepStats {
    calls: usize,
    /// The `x_rot` download, which starts before the sealed MoE step's writes to
    /// it have drained — so this carries the GPU wait, not just copy cost.
    d2h_first_ns: u64,
    /// The other three downloads (`ti`, `tw`, `residual`), pure copy latency.
    d2h_rest_ns: u64,
    /// All experts' gate_up GEMV.
    gu_ns: u64,
    /// All experts' down GEMV.
    dn_ns: u64,
    h2d_ns: u64,
}

/// `(quant, dim, mi, k)` → running per-shape totals.
static MOE_SHAPES: LazyLock<Mutex<BTreeMap<(u8, usize, usize, usize, bool), MoeStepStats>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// `DType` → the decoder for it, for exactly the formats `hipfire_cpu` can
/// decode. `None` means the step stays on the GPU (over PCIe): not a correctness
/// problem, only a smaller bandwidth win, and the load-time coverage line names
/// the format.
///
/// Invariant, pinned by `cpu_quant_rotation_agrees_with_plan`: for every dtype
/// here, [`CpuQuant::is_fwht_g256`] agrees with [`dtype_rotation_plan`], so the
/// seam and the launcher cannot disagree about whether a `Raw` activation
/// arrives rotated.
pub fn cpu_quant_for(dtype: DType) -> Option<CpuQuant> {
    match dtype {
        DType::MQ4G256 => Some(CpuQuant::Mq4G256),
        DType::MQ4G256V2 => Some(CpuQuant::Mq4G256V2),
        DType::MQ6G256 => Some(CpuQuant::Mq6G256),
        DType::MQ6G256V2 => Some(CpuQuant::Mq6G256V2),
        DType::MQ5G256 => Some(CpuQuant::Mq5G256),
        DType::MQ5G256V2 => Some(CpuQuant::Mq5G256V2),
        DType::MQ4CG256 => Some(CpuQuant::Mq4CG256),
        DType::MQ3G256 => Some(CpuQuant::Mq3G256),
        DType::MQ3G256V2 => Some(CpuQuant::Mq3G256V2),
        DType::MQ3G256Lloyd => Some(CpuQuant::Mq3G256Lloyd),
        DType::MQ2G256 => Some(CpuQuant::Mq2G256),
        DType::MQ2G256V2 => Some(CpuQuant::Mq2G256V2),
        DType::MQ2G256Lloyd => Some(CpuQuant::Mq2G256Lloyd),
        DType::MQ2G256LloydU => Some(CpuQuant::Mq2G256LloydU),
        DType::MQ4G256Lloyd => Some(CpuQuant::Mq4G256Lloyd),
        DType::HFQ6G256 => Some(CpuQuant::Hfq6G256),
        DType::HFQ4G256 => Some(CpuQuant::Hfq4G256),
        DType::HFQ4G128 => Some(CpuQuant::Hfq4G128),
        DType::HFQ3G256 => Some(CpuQuant::Hfq3G256),
        DType::HFQ3G128 => Some(CpuQuant::Hfq3G128),
        DType::HFQ2G256 => Some(CpuQuant::Hfq2G256),
        DType::HFQ2G128 => Some(CpuQuant::Hfq2G128),
        DType::TQ2G128 => Some(CpuQuant::Tq2G128),
        DType::BQ1G128 => Some(CpuQuant::Bq1G128),
        DType::Q8_0 => Some(CpuQuant::Q8F16),
        DType::F16 => Some(CpuQuant::F16),
        DType::F32 => Some(CpuQuant::F32),
        DType::BF16 => Some(CpuQuant::Bf16),
        _ => None,
    }
}

/// Whether a model with this many host-placed weight layers has CPU-executed
/// steps at all: CPU execution selected *and* a non-empty host placement. With
/// no spill every weight stays device-resident, [`Gpu::host_located`] is false
/// everywhere, and not a single step can move to the CPU — `memory.offload_exec=cpu`
/// must not change a number there.
///
/// `host_weight_layers` counts every layer with a host-mapped weight the CPU may
/// multiply: whole-layer spills (the step seam) *and* routed-expert layers (the
/// MoE CPU-down splice). Conservative by construction: a spill whose every weight
/// is an unsupported format also reports `true`, which costs a hipGraph but never
/// correctness.
pub fn cpu_offload_active(host_weight_layers: usize) -> bool {
    cpu_exec_enabled() && host_weight_layers > 0
}

/// Whether `memory.offload_exec=cpu` conflicts with a retained-replay backend.
///
/// Pure truth table (the CPU-exec decision, the resolved split, and whether a
/// replay controller is in play) so it is testable without a process snapshot.
pub fn cpu_exec_redline_conflict(cpu_exec: bool, i_gpu_start: usize, replay_enabled: bool) -> bool {
    cpu_exec && i_gpu_start > 0 && replay_enabled
}

/// Refuse `memory.offload_exec=cpu` together with a retained-replay (Redline)
/// backend, naming both keys.
///
/// Redline does not go through `execute_steps`' launch funnel: its tape records
/// GPU launches and replays them, so a CPU-executed step would run while the
/// tape is built and then be *absent* from every replay — the replayed route
/// would keep computing from the activations that step should have refreshed.
/// Nothing in the tape can express that, so the load fails instead of running a
/// route whose output is silently stale.
///
/// `replay_enabled` is the controller's own predicate
/// (`ReplayController::is_enabled`), not a config string: whether Redline is in
/// play depends on the runtime route decision and certification state, so the
/// caller with the `Gpu` supplies the fact. Shadow controllers are refused too
/// (conservatively — shadow does not change the launch route, so the refusal is
/// stricter than strictly necessary there).
pub fn reject_cpu_exec_under_redline(
    i_gpu_start: usize,
    replay_enabled: bool,
) -> Result<(), String> {
    if cpu_exec_redline_conflict(cpu_exec_enabled(), i_gpu_start, replay_enabled) {
        return Err(
            "memory.offload_exec=cpu conflicts with the retained-replay (Redline) backend: the \
             replay tape records GPU launches and does not execute the CPU-executed steps, so a \
             replayed route would compute from stale activations. Set memory.offload_exec=pcie \
             (HIPFIRE_OFFLOAD_EXEC=pcie) or replay.backend=hip"
                .to_string(),
        );
    }
    Ok(())
}

/// Log the capture-disable decision once per process. A CPU-executed step is a
/// host sync point (a D2H and an H2D around the multiplication), so a graph that
/// contained one could neither be recorded nor replayed correctly.
pub fn log_capture_disabled_once() {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        eprintln!("cpu exec: hipGraph capture disabled (CPU-executed steps present)");
    });
}

/// `out[0..m] = W · x[0..k]` on the CPU, over a host-mapped weight.
///
/// `rotate_x` must be `true` exactly when the launcher would self-rotate a `Raw`
/// input for this dtype (i.e. `dtype_rotation_plan(w.dtype) == FwhtG256`), which
/// [`RotationPlan`] is the single source of truth for; [`run_gemv`] derives it
/// so no caller has to.
pub fn run_host_mapped_gemv(
    gpu: &Gpu,
    w: &WeightRef,
    x: &GpuTensor,
    rotate_input: bool,
    out: &GpuTensor,
) -> Result<(), DispatchError> {
    let q = host_mapped_quant(gpu, w)?;
    let (m, k) = (w.m, w.k);
    let bytes = gpu
        .host_bytes(w.buf)
        .ok_or_else(|| cpu_err("weight tensor is host-mapped but has no host pointer"))?;
    let (x_host, d2h_ns) = prepare_activation(gpu, w, x, rotate_input)?;
    let mut y = vec![0.0f32; m];
    let t1 = Instant::now();
    gemv(q, bytes, m, k, &x_host, &mut y);
    let gemv_ns = t1.elapsed().as_nanos() as u64;
    let t2 = Instant::now();
    upload_f32(gpu, out, &y)?;
    let h2d_ns = t2.elapsed().as_nanos() as u64;
    CPU_STEPS.fetch_add(1, Ordering::Relaxed);
    trace_step(
        q,
        w,
        rotate_input,
        false,
        StepTiming {
            d2h_ns,
            gemv_ns,
            h2d_ns,
        },
    );
    Ok(())
}

/// Read the activation for a CPU step, applying the same pre-rotation
/// transforms the launcher's rotate step would, and return it with the elapsed
/// nanoseconds — the trace's `d2h` figure, which therefore covers the AWQ divide
/// and the FWHT whenever they apply, not just the copy.
///
/// * AWQ (`w.awq_scale`): the quantizer pre-scaled the weights by `s` and the
///   rotate kernel divides the activation by it (`(W·s)·(x/s) = W·x`). This
///   happens *inside the rotation*, so it applies only when we rotate — a
///   `Prerotated` input already had it applied by whoever produced it (the
///   launcher *checks* that via the rotation tag; a CPU step cannot, which is
///   the one semantic gap this path has — see the module docs).
/// * FWHT (`rotate_input`): unless the input arrived pre-rotated.
fn prepare_activation(
    gpu: &Gpu,
    w: &WeightRef,
    x: &GpuTensor,
    rotate_input: bool,
) -> Result<(Vec<f32>, u64), DispatchError> {
    let t0 = Instant::now();
    let mut host = download_f32(gpu, x, w.k)?;
    if rotate_input {
        if let Some(scale) = w.awq_scale {
            let scale = download_f32(gpu, scale, w.k)?;
            divide_by_awq_scale(&mut host, &scale);
        }
        rotate_x(&mut host);
    }
    Ok((host, t0.elapsed().as_nanos() as u64))
}

/// `acc += W · x` on the CPU, over a host-mapped weight — the residual form the
/// fused down-projection kernels use (`WithSwiGLUResidual`, `WithResidual`).
///
/// The accumulation is in place in `acc`, matching both fused arms: a residual
/// step never writes its `out` scratch, so writing only `out` when the two do not
/// alias would leave every downstream activation stale.
pub fn run_host_mapped_gemv_residual(
    gpu: &Gpu,
    w: &WeightRef,
    x: &GpuTensor,
    acc: &GpuTensor,
) -> Result<(), DispatchError> {
    let q = host_mapped_quant(gpu, w)?;
    let (m, k) = (w.m, w.k);
    let rotate_input = dtype_rotation_plan(w.dtype) == RotationPlan::FwhtG256;
    let bytes = gpu
        .host_bytes(w.buf)
        .ok_or_else(|| cpu_err("weight tensor is host-mapped but has no host pointer"))?;
    let (x_host, d2h_ns) = prepare_activation(gpu, w, x, rotate_input)?;
    let mut y = vec![0.0f32; m];
    let t1 = Instant::now();
    gemv(q, bytes, m, k, &x_host, &mut y);
    let gemv_ns = t1.elapsed().as_nanos() as u64;
    let t2 = Instant::now();
    let mut acc_host = download_f32(gpu, acc, m)?;
    residual_add(&mut acc_host, &y);
    upload_f32(gpu, acc, &acc_host)?;
    let h2d_ns = t2.elapsed().as_nanos() as u64;
    CPU_STEPS.fetch_add(1, Ordering::Relaxed);
    trace_step(
        q,
        w,
        rotate_input,
        true,
        StepTiming {
            d2h_ns,
            gemv_ns,
            h2d_ns,
        },
    );
    Ok(())
}

/// CPU FFN for one decode token's routed experts: recompute the full expert
/// (gate_up GEMV + SiLU + hidden-rotate + down GEMV) on the CPU from the intact
/// host blobs and accumulate the routing-weighted result into the residual.
///
/// `x_rot` is the already-rotated activation the gate_up projection consumes; the
/// post-SiLU hidden is rotated here exactly as the fused kernel would. The decode
/// MoE params bound the loader's zeroed sink twins, so the sealed MoE step
/// contributed 0 for the routed experts; this supplies the whole contribution.
/// Batched forwards (prefill, MTP verify) bind the real tables and keep the GPU
/// grouped PCIe read, so this runs for single-token decode only.
///
/// Fail-closed by construction: unavailable host bytes, an undecodable dtype, an
/// AWQ sidecar (a bare `gemv` would ignore the per-expert scale), graph capture /
/// replay recording (a CPU step is a host sync point), or short scratch are all
/// errors, never silent zeros.
#[allow(clippy::too_many_arguments)]
pub fn moe_cpu_experts(
    gpu: &Gpu,
    quant: CpuQuant,
    dim: usize,
    mi: usize,
    gate_up_owner: &GpuTensor,
    gate_up_stride: usize,
    down_owner: &GpuTensor,
    down_stride: usize,
    x_rot: &GpuTensor,
    topk_indices: &GpuTensor,
    topk_weights: &GpuTensor,
    residual: &GpuTensor,
    awq: bool,
) -> Result<(), DispatchError> {
    if gpu.graphs.capture_mode || gpu.replay.is_recording() {
        return Err(cpu_err(
            "moe cpu expert splice refuses graph capture / replay recording: a CPU step is a host sync point",
        ));
    }
    if awq {
        return Err(cpu_err(
            "moe cpu expert splice refuses AWQ weights: a bare gemv would ignore the per-expert scale",
        ));
    }
    let gu_bytes = gpu
        .host_bytes(gate_up_owner)
        .ok_or_else(|| cpu_err("moe cpu expert splice: gate_up blob has no host bytes"))?;
    let dn_bytes = gpu
        .host_bytes(down_owner)
        .ok_or_else(|| cpu_err("moe cpu expert splice: down blob has no host bytes"))?;
    let k = topk_indices.numel().min(topk_weights.numel());
    if k == 0 {
        return Err(cpu_err("moe cpu expert splice: empty top-k scratch"));
    }
    let t_first = Instant::now();
    let x = download_f32(gpu, x_rot, dim)?;
    let d2h_first_ns = t_first.elapsed().as_nanos() as u64;
    let t_rest = Instant::now();
    let ti = download_f32(gpu, topk_indices, k)?;
    let tw = download_f32(gpu, topk_weights, k)?;
    let gu_need = hipfire_cpu::gemv::row_bytes(quant, dim)
        .checked_mul(2 * mi)
        .ok_or_else(|| cpu_err("moe cpu expert splice: gate_up extent overflows"))?;
    let dn_need = hipfire_cpu::gemv::row_bytes(quant, mi)
        .checked_mul(dim)
        .ok_or_else(|| cpu_err("moe cpu expert splice: down extent overflows"))?;
    if gate_up_stride < gu_need {
        return Err(cpu_err(&format!(
            "moe cpu expert splice: gate_up_stride {gate_up_stride} < expert bytes {gu_need}"
        )));
    }
    if down_stride < dn_need {
        return Err(cpu_err(&format!(
            "moe cpu expert splice: down_stride {down_stride} < expert bytes {dn_need}"
        )));
    }
    let mut acc = download_f32(gpu, residual, dim)?;
    let d2h_rest_ns = t_rest.elapsed().as_nanos() as u64;
    // Two rayon regions per layer (all experts' gate_up, then all experts' down)
    // instead of 2k small ones: the per-call region entry dominates when each
    // expert's GEMV is only a few thousand rows. Per output element the work is
    // the same `dot_row_simd`, and the residual is still accumulated in rank
    // order, so the result is unchanged.
    let mut gu_all = vec![0.0f32; k * 2 * mi];
    let mut hidden = vec![0.0f32; k * mi];
    let mut dn_all = vec![0.0f32; k * dim];
    let mut slots = vec![0usize; k];
    for (krank, slot_out) in slots.iter_mut().enumerate() {
        let slot = (ti[krank].to_bits() as i32) as u16 as usize;
        *slot_out = slot;
        let gu0 = slot
            .checked_mul(gate_up_stride)
            .ok_or_else(|| cpu_err("moe cpu expert splice: gate_up offset overflows"))?;
        if gu0.checked_add(gu_need).is_none_or(|end| end > gu_bytes.len()) {
            return Err(cpu_err(&format!("moe cpu expert splice: gate_up slot {slot} out of range")));
        }
        let dn0 = slot
            .checked_mul(down_stride)
            .ok_or_else(|| cpu_err("moe cpu expert splice: down offset overflows"))?;
        if dn0.checked_add(dn_need).is_none_or(|end| end > dn_bytes.len()) {
            return Err(cpu_err(&format!("moe cpu expert splice: down slot {slot} out of range")));
        }
    }
    // The int8-activation path (llama.cpp's arithmetic: maddubs over int8
    // activations instead of decoding every code to f32) when the format and the
    // CPU both allow it; the f32 path is the fallback and the reference.
    // `HIPFIRE_MOE_CPU_I8=0` forces the f32 path (A/B).
    let i8_kill = hipfire_config::developer_var("HIPFIRE_MOE_CPU_I8").ok().as_deref() == Some("0");
    let use_i8 = !i8_kill
        && matches!(quant, CpuQuant::Mq4G256V2)
        && hipfire_cpu::simd::int8_dot_available()
        && dim % 128 == 0
        && mi % 128 == 0;
    let t_gu = Instant::now();
    if use_i8 {
        let act_dim: Vec<BlockI8_128> = (0..dim / 128)
            .map(|b| BlockI8_128::quantize(&x[b * 128..]))
            .collect();
        let gu_blocks: Vec<&[u8]> = slots
            .iter()
            .map(|&s| {
                let gu0 = s * gate_up_stride;
                &gu_bytes[gu0..gu0 + gu_need]
            })
            .collect();
        let acts: Vec<&[BlockI8_128]> = vec![act_dim.as_slice(); k];
        hipfire_cpu::gemv::gemv_experts_i8_mq4v2(&gu_blocks, 2 * mi, dim, &acts, &mut gu_all);
    } else {
        let gu_pairs: Vec<(&[u8], &[f32])> = slots
            .iter()
            .map(|&s| {
                let gu0 = s * gate_up_stride;
                (&gu_bytes[gu0..gu0 + gu_need], x.as_slice())
            })
            .collect();
        hipfire_cpu::gemv::gemv_experts(quant, 2 * mi, dim, &gu_pairs, &mut gu_all, None);
    }
    let gu_ns = t_gu.elapsed().as_nanos() as u64;
    for e in 0..k {
        silu_mul(
            &gu_all[e * 2 * mi..(e + 1) * 2 * mi],
            &mut hidden[e * mi..(e + 1) * mi],
        );
        rotate_x(&mut hidden[e * mi..(e + 1) * mi]);
    }
    let t_dn = Instant::now();
    if use_i8 {
        // Each expert's down projection has its own activation (its post-SiLU
        // hidden), so it needs its own int8 blocks.
        let per = mi / 128;
        let mut act_hid: Vec<BlockI8_128> = Vec::with_capacity(k * per);
        for e in 0..k {
            for b in 0..per {
                act_hid.push(BlockI8_128::quantize(&hidden[e * mi + b * 128..]));
            }
        }
        let acts: Vec<&[BlockI8_128]> = (0..k).map(|e| &act_hid[e * per..(e + 1) * per]).collect();
        let dn_blocks: Vec<&[u8]> = slots
            .iter()
            .map(|&s| {
                let dn0 = s * down_stride;
                &dn_bytes[dn0..dn0 + dn_need]
            })
            .collect();
        hipfire_cpu::gemv::gemv_experts_i8_mq4v2(&dn_blocks, dim, mi, &acts, &mut dn_all);
    } else {
        let dn_pairs: Vec<(&[u8], &[f32])> = slots
            .iter()
            .enumerate()
            .map(|(krank, &s)| {
                let dn0 = s * down_stride;
                (
                    &dn_bytes[dn0..dn0 + dn_need],
                    &hidden[krank * mi..(krank + 1) * mi],
                )
            })
            .collect();
        hipfire_cpu::gemv::gemv_experts(quant, dim, mi, &dn_pairs, &mut dn_all, None);
    }
    let dn_ns = t_dn.elapsed().as_nanos() as u64;
    for krank in 0..k {
        let w = tw[krank];
        let dn_out = &dn_all[krank * dim..(krank + 1) * dim];
        for (a, v) in acc.iter_mut().zip(dn_out.iter()) {
            *a += w * v;
        }
    }
    let t_h2d = Instant::now();
    upload_f32(gpu, residual, &acc)?;
    let h2d_ns = t_h2d.elapsed().as_nanos() as u64;
    CPU_STEPS.fetch_add(1, Ordering::Relaxed);
    trace_moe_step(
        quant,
        dim,
        mi,
        k,
        use_i8,
        MoeStepTiming {
            d2h_first_ns,
            d2h_rest_ns,
            gu_ns,
            dn_ns,
            h2d_ns,
        },
    );
    Ok(())
}

/// Per-shape accounting for the decode MoE splice under
/// `HIPFIRE_CPU_EXEC_TRACE=1`: one line per `(quant, dim, mi, k)` at its first
/// call and at every doubling, so the `calls=1` line is the cold first step and
/// later lines are steady state. Keyed by shape rather than summed process-wide,
/// for the same reason [`trace_step`] is.
fn trace_moe_step(q: CpuQuant, dim: usize, mi: usize, k: usize, i8: bool, timing: MoeStepTiming) {
    if hipfire_config::developer_var("HIPFIRE_CPU_EXEC_TRACE").is_err() {
        return;
    }
    let Ok(mut shapes) = MOE_SHAPES.lock() else {
        return;
    };
    let stats = shapes.entry((q as u8, dim, mi, k, i8)).or_default();
    stats.calls += 1;
    stats.d2h_first_ns += timing.d2h_first_ns;
    stats.d2h_rest_ns += timing.d2h_rest_ns;
    stats.gu_ns += timing.gu_ns;
    stats.dn_ns += timing.dn_ns;
    stats.h2d_ns += timing.h2d_ns;
    let stats = *stats;
    if !stats.calls.is_power_of_two() {
        return;
    }
    let (on_cpu, on_gpu) = cpu_exec_counters();
    let per_ms = |ns: u64| ns as f64 / 1e6 / stats.calls as f64;
    eprintln!(
        "cpu exec: moe expert splice dim={dim} mi={mi} k={k} quant={q:?} i8={i8} | {} calls | \
         {on_cpu} steps on CPU, {on_gpu} host-mapped steps still on GPU | mean per call: \
         d2h_first={:.2}ms d2h_rest={:.2}ms gu={:.2}ms dn={:.2}ms h2d={:.2}ms",
        stats.calls,
        per_ms(stats.d2h_first_ns),
        per_ms(stats.d2h_rest_ns),
        per_ms(stats.gu_ns),
        per_ms(stats.dn_ns),
        per_ms(stats.h2d_ns)
    );
}

/// One splice call's wall-time split, handed to [`trace_moe_step`].
struct MoeStepTiming {
    d2h_first_ns: u64,
    d2h_rest_ns: u64,
    gu_ns: u64,
    dn_ns: u64,
    h2d_ns: u64,
}

/// Whether [`run_host_mapped_gemv`] / [`run_host_mapped_gemv_residual`] can drive
/// this weight: host-mapped *and* a decodable format.
pub fn host_mapped_cpu_capable(gpu: &Gpu, w: &WeightRef) -> bool {
    cpu_exec_enabled() && gpu.host_located(w.buf) && cpu_quant_for(w.dtype).is_some()
}

/// `HIPFIRE_MOE_CPU_ORACLE=1`: run the same host-mapped expert bytes that fed the
/// GPU's routed-expert kernels through the CPU SIMD expert FFN
///
/// Attribution, not a fallback. If the CPU path over the same blobs reproduces the
/// GPU's own per-expert down outputs, the packed layout and the GPU's read of
/// host-mapped memory are sound — a wrong token then cannot be explained by either
/// and must come from the routing, the dtype/arm selection, or a non-expert tensor.
/// If they disagree, the printed rank and element localize it in one run, which a
/// resident control arm cannot do on a card too small to hold the model.
///
/// Lives here rather than in the arch because this crate owns the CPU seam and the
/// arch intentionally has no `hipfire-cpu` dependency. Both candidate inputs are
/// tried (activation as stored, and freshly rotated), so the report is evidence
/// about which one the kernels consume instead of assuming it.
#[allow(clippy::too_many_arguments)]
pub fn moe_cpu_oracle_report(
    gpu: &Gpu,
    quant: CpuQuant,
    dim: usize,
    mi: usize,
    k: usize,
    gate_up_owner: &GpuTensor,
    down_owner: &GpuTensor,
    x_norm: &GpuTensor,
    topk_indices: &GpuTensor,
    topk_weights: &GpuTensor,
    down_expanded: &GpuTensor,
    layer_idx: u16,
    // `ffn.expert_down_awq_ptrs.is_some()`. Printed because it is a hard
    // precondition of any CPU down projection: a bare `gemv` ignores a
    // per-expert AWQ scale, so a fixture that gains sidecars later must not
    // silently keep taking the CPU path. The index scan that established its
    // absence here is a 60 MB window, not a guarantee.
    down_awq: bool,
) {
    let (x, ti, tw, dn_gpu) = match (
        download_f32(gpu, x_norm, dim),
        download_f32(gpu, topk_indices, k),
        download_f32(gpu, topk_weights, k),
        download_f32(gpu, down_expanded, k * dim),
    ) {
        (Ok(a), Ok(b), Ok(c), Ok(d)) => (a, b, c, d),
        _ => {
            eprintln!("[moe-cpu-oracle] layer {layer_idx}: scratch download failed");
            return;
        }
    };
    let (gu, dn) = match (gpu.host_bytes(gate_up_owner), gpu.host_bytes(down_owner)) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            eprintln!("[moe-cpu-oracle] layer {layer_idx}: expert blobs are not host-mapped");
            return;
        }
    };
    let blobs = hipfire_cpu::moe::ExpertBlobs {
        gate_up: gu,
        down: dn,
        gate_up_stride: hipfire_cpu::gemv::row_bytes(quant, dim) * 2 * mi,
        down_stride: hipfire_cpu::gemv::row_bytes(quant, mi) * dim,
        quant,
        dim,
        mi,
    };
    // Which side is empty is the question, so report magnitudes, not just the
    // worst relative difference: a layer whose GPU expert output is identically
    // zero and whose CPU output is not is a *missing contribution*, a different
    // fault from a layer that computes a similar-but-different number.
    let sum_abs = |v: &[f32]| v.iter().map(|x| x.abs() as f64).sum::<f64>();
    let zeros = |v: &[f32]| v.iter().filter(|x| **x == 0.0).count();
    let mut xin = x[..dim.min(x.len())].to_vec();
    if dim % 256 == 0 {
        rotate_x(&mut xin);
    }
    let mut cpu_all = vec![0.0f32; k * dim];
    let mut routed_ranks = 0usize;
    for krank in 0..k.min(ti.len()).min(tw.len()) {
        let expert = (ti[krank].to_bits() as i32) as u16 as usize;
        if hipfire_cpu::moe::run_experts(
            &blobs,
            &xin,
            &[(expert, 1.0)],
            &mut cpu_all[krank * dim..(krank + 1) * dim],
        )
        .is_ok()
        {
            routed_ranks += 1;
        }
    }
    eprintln!(
        "[moe-cpu-oracle] layer {layer_idx} ranks {routed_ranks}/{k} down_awq={down_awq} |x| {:.4e} (zeros {}) |gpu-down| {:.4e} (zeros {}/{}) |cpu-down| {:.4e} (zeros {})",
        sum_abs(&x[..dim.min(x.len())]),
        zeros(&x[..dim.min(x.len())]),
        sum_abs(&dn_gpu),
        zeros(&dn_gpu),
        dn_gpu.len(),
        sum_abs(&cpu_all),
        zeros(&cpu_all),
    );
    for rotated in [false, true] {
        if x.len() < dim || (rotated && dim % 256 != 0) {
            break;
        }
        let mut xin = x[..dim].to_vec();
        if rotated {
            rotate_x(&mut xin);
        }
        let mut worst_rel = 0.0f64;
        let mut worst_at = (usize::MAX, usize::MAX);
        let mut worst_vals = (0.0f32, 0.0f32);
        let mut ranks = 0usize;
        for krank in 0..k.min(ti.len()).min(tw.len()) {
            let expert = (ti[krank].to_bits() as i32) as u16 as usize;
            let base = krank * dim;
            if base + dim > dn_gpu.len() {
                break;
            }
            let mut out = vec![0.0f32; dim];
            if hipfire_cpu::moe::run_experts(&blobs, &xin, &[(expert, 1.0)], &mut out).is_err() {
                continue;
            }
            ranks += 1;
            for j in 0..dim {
                let (a, b) = (out[j] as f64, dn_gpu[base + j] as f64);
                let rel = (a - b).abs() / a.abs().max(b.abs()).max(1e-6);
                if rel > worst_rel {
                    worst_rel = rel;
                    worst_at = (krank, j);
                    worst_vals = (a as f32, b as f32);
                }
            }
        }
        eprintln!(
            "[moe-cpu-oracle] layer {layer_idx} quant {quant:?} ranks {ranks} input {}: max |cpu-gpu|/max|.| = {worst_rel:.3e} at (rank, elem) {worst_at:?} cpu={} gpu={}",
            if rotated { "fwht-rotated" } else { "as-stored" },
            worst_vals.0,
            worst_vals.1,
        );
    }
}

fn host_mapped_quant(gpu: &Gpu, w: &WeightRef) -> Result<CpuQuant, DispatchError> {
    if !cpu_exec_enabled() {
        return Err(cpu_err("cpu exec is not enabled (memory.offload_exec)"));
    }
    if !gpu.host_located(w.buf) {
        return Err(cpu_err("weight tensor is not host-mapped"));
    }
    cpu_quant_for(w.dtype).ok_or_else(|| {
        cpu_err(&format!(
            "no CPU decoder for dtype {:?}; this step must stay on the GPU",
            w.dtype
        ))
    })
}

/// Rotation disposition of a step's input. The distinction is load-bearing:
/// `Raw` means "the launcher would rotate this", `Prerotated` means someone
/// already did — rotating a `Prerotated` input again is a silent
/// $\mathcal{R}^2$ error that still looks like plausible activations.
enum CpuInput<'a> {
    Raw(&'a GpuTensor),
    Prerotated(&'a GpuTensor),
}

impl<'a> CpuInput<'a> {
    fn tensor(&self) -> &'a GpuTensor {
        match self {
            CpuInput::Raw(t) | CpuInput::Prerotated(t) => t,
        }
    }
}

/// A host-mapped weight step that the CPU can execute.
pub(crate) struct CpuStep<'a> {
    w: &'a WeightRef<'a>,
    input: CpuInput<'a>,
    out: &'a GpuTensor,
    /// `Some` for `Step::GemvResidual`, which accumulates in place into the
    /// residual (never into `out` — see the GemvResidual arms in `steps.rs`).
    residual: Option<&'a GpuTensor>,
}

/// Plan a CPU execution for `step`, or `None` when it must stay on the GPU:
/// CPU execution disabled, weight not host-mapped, or a format
/// [`cpu_quant_for`] does not cover.
pub(crate) fn plan_step<'a>(gpu: &Gpu, step: &'a Step<'a>) -> Option<CpuStep<'a>> {
    if !cpu_exec_enabled() {
        return None;
    }
    let (w, input, out, residual) = match step {
        Step::Gemv { w, input, out } => (*w, input, *out, None),
        Step::GemvResidual {
            w,
            input,
            residual,
            out,
        } => (*w, input, *out, Some(*residual)),
        _ => return None,
    };
    if !gpu.host_located(w.buf) || cpu_quant_for(w.dtype).is_none() {
        return None;
    }
    Some(CpuStep {
        w,
        input: match input {
            GemvInput::Raw(t) => CpuInput::Raw(t),
            GemvInput::Prerotated(t) => CpuInput::Prerotated(t),
        },
        out,
        residual,
    })
}

/// Whether `step` reads a host-mapped weight, regardless of whether the CPU can
/// decode it — the coverage counter's question.
pub(crate) fn reads_host_mapped_weight(gpu: &Gpu, step: &Step) -> bool {
    match step {
        Step::Gemv { w, .. } | Step::GemvResidual { w, .. } => gpu.host_located(w.buf),
        _ => false,
    }
}

/// Execute a planned CPU step.
pub(crate) fn run_step(gpu: &Gpu, plan: &CpuStep) -> Result<(), DispatchError> {
    match plan.residual {
        // The fused down-projection this splits always feeds the CPU path an
        // unrotated activation (the SiLU output), so it rotates when the dtype
        // says the weights were encoded post-rotation.
        Some(residual) => run_host_mapped_gemv_residual(gpu, plan.w, plan.input.tensor(), residual),
        // Mirror `GemvFamily::run_input`: a `Raw` input is rotated by the
        // launcher *and only when* the dtype's plan is FwhtG256; a `Prerotated`
        // input is never touched again.
        None => run_host_mapped_gemv(
            gpu,
            plan.w,
            plan.input.tensor(),
            matches!(plan.input, CpuInput::Raw(_))
                && dtype_rotation_plan(plan.w.dtype) == RotationPlan::FwhtG256,
            plan.out,
        ),
    }
}

/// Count a host-mapped step that is about to be launched on the GPU anyway.
pub(crate) fn count_host_mapped_gpu_step() {
    HOST_MAPPED_GPU_STEPS.fetch_add(1, Ordering::Relaxed);
}

fn cpu_err(msg: &str) -> DispatchError {
    DispatchError::Hip(format!("cpu exec: {msg}"))
}

fn download_f32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Result<Vec<f32>, DispatchError> {
    assert!(
        t.numel() >= n,
        "cpu exec: tensor has {} elements, need {n}",
        t.numel()
    );
    let mut out = vec![0.0f32; n];
    // Safety: `out` is a live `n`-element f32 buffer; the slice covers exactly
    // those bytes and does not outlive the call.
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    gpu.memcpy_dtoh_auto(bytes, &t.buf)
        .map_err(|e| cpu_err(&format!("D2H: {e}")))?;
    Ok(out)
}

fn upload_f32(gpu: &Gpu, t: &GpuTensor, v: &[f32]) -> Result<(), DispatchError> {
    assert!(
        t.numel() >= v.len(),
        "cpu exec: destination has {} elements, source has {}",
        t.numel(),
        v.len()
    );
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    gpu.memcpy_htod_auto(&t.buf, bytes)
        .map_err(|e| cpu_err(&format!("H2D: {e}")))
}

/// One step's own wall-time split, handed to [`trace_step`].
struct StepTiming {
    d2h_ns: u64,
    gemv_ns: u64,
    h2d_ns: u64,
}

/// Per-shape step accounting under `HIPFIRE_CPU_EXEC_TRACE=1`: one line per
/// distinct step shape at its first call and then at every doubling of that
/// shape's call count, followed by the running counters.
///
/// Both halves of that schedule matter. The `calls=1` line is the *cold* first
/// step of the shape (host-mapped page first touch, rayon pool wake-up); the
/// later lines are the shape's own steady state, which is the only number worth
/// quoting. Printing the whole process-wide mean per shape — as this did before
/// — gives every shape the same figure and inflates it with the cold steps of
/// whatever ran first, which reads as attribution and is not.
///
/// The counters are the coverage signal: `host-mapped steps still on GPU` must be
/// 0 for a model the CPU covers, so a shape that silently never reaches the seam
/// shows up as a number rather than as a mystery.
fn trace_step(q: CpuQuant, w: &WeightRef, rotated: bool, residual: bool, timing: StepTiming) {
    if hipfire_config::developer_var("HIPFIRE_CPU_EXEC_TRACE").is_err() {
        return;
    }
    let key = (q as u8, w.m, w.k, rotated, residual, w.awq_scale.is_some());
    let Ok(mut shapes) = SHAPES.lock() else {
        return;
    };
    let stats = shapes.entry(key).or_default();
    stats.calls += 1;
    stats.d2h_ns += timing.d2h_ns;
    stats.gemv_ns += timing.gemv_ns;
    stats.h2d_ns += timing.h2d_ns;
    let stats = *stats;
    if !stats.calls.is_power_of_two() {
        return;
    }
    let (m, k) = (w.m, w.k);
    let (on_cpu, on_gpu) = cpu_exec_counters();
    // ns -> ms, averaged over *this shape's* calls so far.
    let per_ms = |ns: u64| ns as f64 / 1e6 / stats.calls as f64;
    eprintln!(
        "cpu exec: step gemv m={m} k={k} quant={q:?} rotated={rotated} residual={residual} \
         awq={} | {} calls | {on_cpu} steps on CPU, {on_gpu} host-mapped steps still on GPU | \
         mean per step: d2h={:.2}ms gemv={:.2}ms h2d={:.2}ms",
        w.awq_scale.is_some(),
        stats.calls,
        per_ms(stats.d2h_ns),
        per_ms(stats.gemv_ns),
        per_ms(stats.h2d_ns)
    );
}
