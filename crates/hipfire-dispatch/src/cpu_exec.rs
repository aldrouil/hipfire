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

/// Whether the expanded MoE timing trace is armed: the existing
/// `HIPFIRE_CPU_EXEC_TRACE` diagnostic. Resolved per call (not cached) so the
/// debug script's trace-off control run in the same process snapshot shape
/// still works; the lookup is a snapshot-map read, off the hot arithmetic.
pub fn moe_trace_enabled() -> bool {
    hipfire_config::developer_var("HIPFIRE_CPU_EXEC_TRACE").is_ok()
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

/// Slots per `CpuQuant` discriminant (`q as usize`) in one splice call's
/// selected experts. Length 32 covers the 28 current variants with headroom;
/// indices at/above the variant count stay zero. Printed as `NamexCount`
/// pairs via [`cpu_quant_mix_name`].
const MOE_QUANT_MIX_LEN: usize = 32;

/// Per-shape totals for the MoE expert splice: the d2h / gemv / h2d split plus
/// the coarse setup/middle/combine phases and total wall of one
/// `(quant, dim, mi, top_k, rows, graded)` shape's calls. `quant` is the first
/// selected expert's gate_up format (a representative); `graded` says the call's
/// experts were not all one tier.
#[derive(Clone, Copy, Default)]
struct MoeStepStats {
    calls: usize,
    /// Four D2H submissions, including HIP API overhead.
    d2h_enqueue_ns: u64,
    /// Single completion barrier for all four downloads.
    d2h_complete_ns: u64,
    /// All experts' gate_up GEMV.
    gu_ns: u64,
    /// All experts' down GEMV.
    dn_ns: u64,
    h2d_ns: u64,
    /// Resolving the selected slots to host experts (ids/spans), before any
    /// arithmetic.
    setup_ns: u64,
    /// All experts' SiLU + optional hidden rotation, the whole per-token loop.
    middle_ns: u64,
    /// All experts' routing-weighted accumulate into the residual.
    combine_ns: u64,
    /// The whole splice call's wall time, exclusive of the trace print itself.
    total_ns: u64,
    /// GU activation-prep/metadata vs Rayon decode-and-dot execution split.
    /// `gu_ns` stays the whole region (`prep + exec`); `prep` is 0 until the
    /// traced `gemv_shared_sourced` sibling lands.
    gu_prep_ns: u64,
    gu_exec_ns: u64,
    /// DN activation-prep/metadata vs Rayon decode-and-dot execution split.
    dn_prep_ns: u64,
    dn_exec_ns: u64,
    /// CPU middle split: gather vs SiLU vs FWHT across the per-token loop.
    middle_gather_ns: u64,
    middle_silu_ns: u64,
    middle_fwht_ns: u64,
    /// Host time to drain the producers before downloads (trace only). Includes
    /// the synchronization API cost, not physical copy time.
    d2h_wait_ns: u64,
    /// Selected-slot GU / DN quant mixture summed across this shape's calls.
    gu_mix: [u64; MOE_QUANT_MIX_LEN],
    dn_mix: [u64; MOE_QUANT_MIX_LEN],
}

/// `(quant, dim, mi, top_k, rows, graded)` → running per-shape totals.
static MOE_SHAPES: LazyLock<Mutex<BTreeMap<(u8, usize, usize, usize, usize, bool), MoeStepStats>>> =
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

/// One selected expert's host-resident weight bytes for the CPU splice.
///
/// Each projection carries its own format, so a graded layer's experts (MQ6 hot
/// / MQ4 mid / MQ3-Lloyd cold) resolve per expert, and an expert whose two
/// projections differ is expressible too. The slices are the expert's *own*
/// extents — no stride padding: `gate_up` is exactly
/// `row_bytes(gate_up_quant, dim) * 2 * mi` bytes and `down` exactly
/// `row_bytes(down_quant, mi) * dim`. The engine validates and refuses a short
/// slice rather than reading past it.
///
/// Owns nothing, so the splice can hold a whole window's resolved slots in a
/// fixed-capacity stack array instead of a heap `Vec` (`Copy` is what makes
/// `[CpuMoeExpert; N]` initializable without an allocation).
#[derive(Clone, Copy)]
pub struct CpuMoeExpert<'a> {
    pub gate_up_quant: CpuQuant,
    pub gate_up: &'a [u8],
    pub down_quant: CpuQuant,
    pub down: &'a [u8],
}

/// Reusable host workspaces for the MoE CPU splice, kept in a thread-local so a
/// per-token decode does not re-allocate them every layer. Resized on demand,
/// never shrunk — the buffers are the splice's, not the caller's.
///
/// The first four are the arithmetic `f32` workspaces. The rest is index metadata
/// (counts and offsets, no borrows), rebuilt each call in reusable storage so
/// the warm path makes no heap allocation: the selected slots are grouped by
/// expert id by a counting sort, which needs no per-group `Vec`. The
/// borrow-carrying job descriptors cannot live here at all — they borrow these
/// very buffers — so they are answered on demand through
/// [`MoeJobSource`] instead.
#[derive(Default)]
struct MoeScratch {
    gu: Vec<f32>,
    hidden: Vec<f32>,
    dn: Vec<f32>,
    gather: Vec<f32>,
    /// Slots in first-seen group order: group `g` owns
    /// `order[group_start[g]..group_start[g+1]]`.
    order: Vec<u32>,
    /// `groups + 1` start offsets into [`Self::order`].
    group_start: Vec<u32>,
    /// The global expert id of each group.
    group_id: Vec<u16>,
    /// Group index of each slot.
    slot_group: Vec<u32>,
    /// Position of each slot within its group (its column in the group's
    /// row-major output run).
    slot_si: Vec<u32>,
    /// Element offset of each group's output run, plus a trailing total
    /// (`groups + 1`); refilled per projection with that projection's row count.
    group_off: Vec<usize>,
}

thread_local! {
    static MOE_SCRATCH: std::cell::RefCell<MoeScratch> =
        const { std::cell::RefCell::new(MoeScratch {
            gu: Vec::new(),
            hidden: Vec::new(),
            dn: Vec::new(),
            gather: Vec::new(),
            order: Vec::new(),
            group_start: Vec::new(),
            group_id: Vec::new(),
            slot_group: Vec::new(),
            slot_si: Vec::new(),
            group_off: Vec::new(),
        }) };
    static MOE_TRANSFER: std::cell::RefCell<Option<hip_bridge::PinnedHostBuffer>> =
        const { std::cell::RefCell::new(None) };
}

/// The top-k scratch's expert id, decoded exactly as the loader packed it: the
/// router writes a `u16` id as the low bits of an `f32` bit pattern.
fn expert_id(raw: f32) -> usize {
    (raw.to_bits() as i32) as u16 as usize
}

/// `gate_up` extent of one expert: `row_bytes * 2 * mi`, checked.
fn expert_gu_bytes(q: CpuQuant, dim: usize, mi: usize) -> Result<usize, DispatchError> {
    hipfire_cpu::gemv::row_bytes(q, dim)
        .checked_mul(2 * mi)
        .ok_or_else(|| cpu_err("moe cpu expert splice: gate_up extent overflows"))
}

/// `down` extent of one expert: `row_bytes * dim`, checked.
fn expert_down_bytes(q: CpuQuant, dim: usize, mi: usize) -> Result<usize, DispatchError> {
    hipfire_cpu::gemv::row_bytes(q, mi)
        .checked_mul(dim)
        .ok_or_else(|| cpu_err("moe cpu expert splice: down extent overflows"))
}

/// Cap on the slots (`rows × top_k`) [`moe_cpu_experts`] resolves into a stack
/// array instead of a heap `Vec`. Decode (1 row) and the ≤4-row verify window
/// keep `rows × top_k` far below this for every real `num_experts_per_tok`; a
/// wider window still works through the `Vec` fallback.
const MOE_STACK_SLOTS: usize = 256;

/// CPU FFN for a window of `rows` tokens' routed experts: recompute every
/// selected expert (gate_up GEMV + SiLU + hidden-rotate + down GEMV) on the CPU
/// from the intact host blobs and accumulate the routing-weighted result into
/// the residual, one row at a time.
///
/// `x_rot` is the already-rotated activation the gate_up projection consumes; the
/// post-SiLU hidden is rotated here exactly as the fused kernel would (per the
/// *down* projection's format). The sealed MoE program of a CPU-deferred call
/// ([`MoeParams::cpu_deferred_experts`](crate::families::moe::MoeParams::cpu_deferred_experts))
/// omits every routed GPU stage, so this supplies the whole routed contribution.
/// `rows` is the window width (1 for decode, up
/// to the narrow MTP verify width) and `top_k` is `num_experts_per_tok`; both are
/// explicit so a tensor's spare capacity can never silently become the batch
/// size.
///
/// `resolve` maps a selected expert's global id to its host bytes. It is called
/// once per selected slot (`rows * top_k` calls, row-major then rank order) up
/// front, may be asked for the same id more than once, and must be a pure
/// function of the id.
///
/// Fail-closed by construction: an unknown expert id, unavailable host bytes, an
/// undecodable dtype, an AWQ sidecar (a bare `gemv` would ignore the per-expert
/// scale), graph capture / replay recording (a CPU step is a host sync point), a
/// `dim`/`mi` that is not a multiple of 256, or short scratch are all errors,
/// never silent zeros.
#[allow(clippy::too_many_arguments)]
pub fn moe_cpu_experts<'a>(
    gpu: &Gpu,
    dim: usize,
    mi: usize,
    rows: usize,
    top_k: usize,
    x_rot: &GpuTensor,
    topk_indices: &GpuTensor,
    topk_weights: &GpuTensor,
    residual: &GpuTensor,
    awq: bool,
    resolve: impl Fn(usize) -> Result<CpuMoeExpert<'a>, DispatchError>,
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
    if rows == 0 || top_k == 0 {
        return Err(cpu_err(
            "moe cpu expert splice: rows and top_k must be non-zero",
        ));
    }
    if dim % 256 != 0 || mi % 256 != 0 {
        return Err(cpu_err(&format!(
            "moe cpu expert splice: dim={dim} mi={mi} must be multiples of 256"
        )));
    }
    let slots = rows
        .checked_mul(top_k)
        .ok_or_else(|| cpu_err("moe cpu expert splice: rows × top_k overflows"))?;
    let n_x = rows
        .checked_mul(dim)
        .ok_or_else(|| cpu_err("moe cpu expert splice: rows × dim overflows"))?;
    if x_rot.numel() < n_x || residual.numel() < n_x {
        return Err(cpu_err(&format!(
            "moe cpu expert splice: activation/residual need {n_x} elements"
        )));
    }
    if topk_indices.numel() < slots || topk_weights.numel() < slots {
        return Err(cpu_err(&format!(
            "moe cpu expert splice: top-k scratch needs {slots} elements"
        )));
    }
    // The splice's whole wall time, started after validation and stopped just
    // before the trace print (which is not part of the call's cost).
    let t_total = Instant::now();
    gpu.bind_thread().map_err(|e| cpu_err(&format!("bind: {e}")))?;
    let stage_bytes = n_x
        .checked_add(slots)
        .and_then(|n| n.checked_mul(8))
        .ok_or_else(|| cpu_err("moe cpu expert splice: transfer extent overflows"))?;
    // Check every span before the first async enqueue: a later HIP size assert
    // must never unwind while a download still writes the reusable stage.
    for (tensor, n) in [
        (x_rot, n_x),
        (topk_indices, slots),
        (topk_weights, slots),
        (residual, n_x),
    ] {
        if tensor.buf.size() < n * 4 {
            return Err(cpu_err("moe cpu expert splice: transfer buffer too small"));
        }
    }
    MOE_TRANSFER.with(|transfer_cell| {
        let mut transfer = transfer_cell.borrow_mut();
        if transfer.as_ref().is_none_or(|b| b.size() < stage_bytes) {
            *transfer = Some(
                gpu.hip.host_buffer(stage_bytes)
                    .map_err(|e| cpu_err(&format!("pinned transfer stage: {e}")))?,
            );
        }
        let result = MOE_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let bytes = transfer
                .as_mut()
                .expect("stage allocated above")
                .as_bytes_mut();
            // SAFETY: hipHostMalloc is suitably aligned for f32, every byte was
            // initialized on allocation, and the four nonoverlapping slices stay
            // alive until both download completion and the blocking upload.
            let values = unsafe {
                std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<f32>(), stage_bytes / 4)
            };
            let (x, rest) = values.split_at_mut(n_x);
            let (ti, rest) = rest.split_at_mut(slots);
            let (tw, acc) = rest.split_at_mut(slots);
            let mut timing = MoeStepTiming::default();
            // An event inserted here is after the producers and cannot measure
            // the host wait for them. Trace only: drain before copying to
            // separate readiness from copy/API cost. Controls wait once after
            // enqueueing all four downloads.
            if moe_trace_enabled() {
                let t_wait = Instant::now();
                match gpu.active_stream.as_ref() {
                    Some(stream) => gpu.hip.stream_synchronize(stream),
                    None => gpu.hip.stream_synchronize_default(),
                }
                .map_err(|e| cpu_err(&format!("D2H producer drain: {e}")))?;
                timing.d2h_wait_ns = t_wait.elapsed().as_nanos() as u64;
            }
            let t_enqueue = Instant::now();
            download_into_f32_async(gpu, x_rot, x)?;
            download_into_f32_async(gpu, topk_indices, ti)?;
            download_into_f32_async(gpu, topk_weights, tw)?;
            download_into_f32_async(gpu, residual, acc)?;
            timing.d2h_enqueue_ns = t_enqueue.elapsed().as_nanos() as u64;
            let t_complete = Instant::now();
            match gpu.active_stream.as_ref() {
                Some(stream) => gpu.hip.stream_synchronize(stream),
                None => gpu.hip.stream_synchronize_default(),
            }
            .map_err(|e| cpu_err(&format!("D2H completion: {e}")))?;
            timing.d2h_complete_ns = t_complete.elapsed().as_nanos() as u64;
            // Resolve every selected slot up front: a bad id or a short span is
            // an error before arithmetic. Warm decode/verify resolves into a
            // fixed stack array; wider windows retain the heap fallback.
            let t_setup = Instant::now();
            let dummy: CpuMoeExpert<'a> = CpuMoeExpert {
                gate_up_quant: CpuQuant::F32,
                gate_up: &[],
                down_quant: CpuQuant::F32,
                down: &[],
            };
            let mut stack = [dummy; MOE_STACK_SLOTS];
            let heap: Vec<CpuMoeExpert<'a>>;
            let experts: &[CpuMoeExpert<'a>] = if slots <= MOE_STACK_SLOTS {
                for s in 0..slots {
                    stack[s] = resolve(expert_id(ti[s]))?;
                }
                &stack[..slots]
            } else {
                heap = (0..slots)
                    .map(|s| resolve(expert_id(ti[s])))
                    .collect::<Result<Vec<_>, _>>()?;
                &heap
            };
            timing.setup_ns = t_setup.elapsed().as_nanos() as u64;
            // Actual GU/DN tier mixture, independent of the first expert label.
            for e in experts.iter() {
                let gi = (e.gate_up_quant as usize).min(MOE_QUANT_MIX_LEN - 1);
                let di = (e.down_quant as usize).min(MOE_QUANT_MIX_LEN - 1);
                timing.gu_mix[gi] = timing.gu_mix[gi].saturating_add(1);
                timing.dn_mix[di] = timing.dn_mix[di].saturating_add(1);
            }
            moe_cpu_experts_host(
                dim, mi, rows, top_k, experts, x, ti, tw, acc, &mut scratch, &mut timing,
            )?;
            let t_h2d = Instant::now();
            upload_f32(gpu, residual, acc)?;
            timing.h2d_ns = t_h2d.elapsed().as_nanos() as u64;
            CPU_STEPS.fetch_add(1, Ordering::Relaxed);
            let mixed = experts.iter().any(|e| {
                e.gate_up_quant != experts[0].gate_up_quant || e.down_quant != experts[0].down_quant
            });
            timing.total_ns = t_total.elapsed().as_nanos() as u64;
            trace_moe_step(
                experts[0].gate_up_quant,
                dim,
                mi,
                top_k,
                rows,
                mixed,
                timing,
            );
            Ok(())
        });
        // Even a failed enqueue may leave earlier copies in flight. Retire
        // them before any reuse/free; if the device cannot drain, retain the
        // allocation forever instead of allowing a GPU use-after-free.
        if result.is_err() && gpu.hip.device_synchronize().is_err() {
            std::mem::forget(transfer.take());
        }
        result
    })
}

/// The splice's arithmetic over already-resolved host slices — split out from
/// [`moe_cpu_experts`] so it is testable without a device. It accumulates into
/// `acc` and records its `gate_up`/SiLU+rotate/down/combine region
/// timings into `timing`'s matching fields (the caller owns the D2H/H2D/total
/// fields).
///
/// Slots are grouped by expert id: one weight, one shared-weight projection,
/// every token that selected it. A weight row's bytes are therefore read once
/// for all of that expert's tokens (row-major, see
/// [`hipfire_cpu::gemv::gemv_shared_sourced`]), and all groups of a projection go
/// to the CPU in one rayon region.
///
/// Every working set is a reusable field of `scratch` — the `f32` buffers and the
/// index metadata — and the grouped window is described to the kernel through
/// [`MoeJobSource`] instead of a materialized job slice, so a warm call performs
/// no heap allocation.
#[allow(clippy::too_many_arguments)]
fn moe_cpu_experts_host(
    dim: usize,
    mi: usize,
    rows: usize,
    top_k: usize,
    experts: &[CpuMoeExpert<'_>],
    x: &[f32],
    ti: &[f32],
    tw: &[f32],
    residual_acc: &mut [f32],
    scratch: &mut MoeScratch,
    timing: &mut MoeStepTiming,
) -> Result<(), DispatchError> {
    let slots = rows * top_k;
    if experts.len() != slots {
        return Err(cpu_err(
            "moe cpu expert splice: expert count != rows × top_k",
        ));
    }
    let n_x = rows * dim;
    if x.len() < n_x || residual_acc.len() < n_x || ti.len() < slots || tw.len() < slots {
        return Err(cpu_err("moe cpu expert splice: scratch too small"));
    }

    // Group the selected slots by expert id, in first-seen order, validating each
    // expert's spans as we go. A counting sort into the reusable index vectors
    // keeps each group's members contiguous in `order` without a per-group `Vec`:
    // assign every slot its group index and collect the group ids, then
    // prefix-sum the sizes into `group_start` and place the slots. (`group_off`
    // doubles as the fill cursor here, before it holds output offsets.)
    scratch.group_id.clear();
    scratch.slot_group.resize(slots, 0);
    for s in 0..slots {
        let e = &experts[s];
        let gu_need = expert_gu_bytes(e.gate_up_quant, dim, mi)?;
        let dn_need = expert_down_bytes(e.down_quant, dim, mi)?;
        if e.gate_up.len() < gu_need {
            return Err(cpu_err(&format!(
                "moe cpu expert splice: expert gate_up span {} < {gu_need} bytes",
                e.gate_up.len()
            )));
        }
        if e.down.len() < dn_need {
            return Err(cpu_err(&format!(
                "moe cpu expert splice: expert down span {} < {dn_need} bytes",
                e.down.len()
            )));
        }
        let eid = expert_id(ti[s]) as u16;
        let gi = match scratch.group_id.iter().position(|&g| g == eid) {
            Some(gi) => gi,
            None => {
                scratch.group_id.push(eid);
                scratch.group_id.len() - 1
            }
        };
        scratch.slot_group[s] = gi as u32;
    }
    let groups = scratch.group_id.len();
    scratch.group_off.clear();
    scratch.group_off.resize(groups, 0);
    for s in 0..slots {
        scratch.group_off[scratch.slot_group[s] as usize] += 1;
    }
    scratch.group_start.clear();
    scratch.group_start.resize(groups + 1, 0);
    let mut acc = 0u32;
    for gi in 0..groups {
        scratch.group_start[gi] = acc;
        acc += scratch.group_off[gi] as u32;
    }
    scratch.group_start[groups] = acc;
    for gi in 0..groups {
        scratch.group_off[gi] = scratch.group_start[gi] as usize;
    }
    scratch.order.resize(slots, 0);
    scratch.slot_si.resize(slots, 0);
    for s in 0..slots {
        let gi = scratch.slot_group[s] as usize;
        let at = scratch.group_off[gi];
        scratch.order[at] = s as u32;
        scratch.slot_si[s] = at as u32 - scratch.group_start[gi];
        scratch.group_off[gi] += 1;
    }

    // gate_up: one shared-weight projection per group, every group in one rayon
    // region. Output is row-major per group (`out[r * n + t]`), so a weight row's
    // bytes are read once for all of that expert's tokens.
    scratch.gu.resize(2 * mi * slots, 0.0);
    scratch.hidden.resize(mi * slots, 0.0);
    scratch.gather.resize(2 * mi, 0.0);
    scratch.dn.resize(dim * slots, 0.0);
    scratch.group_off.resize(groups + 1, 0);
    fill_group_off(&scratch.group_start, &mut scratch.group_off, 2 * mi);
    let gu_split = {
        let src = MoeJobSource {
            experts,
            order: &scratch.order,
            group_start: &scratch.group_start,
            act: x,
            row_len: dim,
            top_k,
            proj: MoeProj::GateUp,
        };
        hipfire_cpu::gemv::gemv_shared_sourced_traced(2 * mi, dim, &mut scratch.gu, &src, None)
    };
    timing.gu_ns = gu_split.total_ns;
    timing.gu_prep_ns = gu_split.prep_ns;
    timing.gu_exec_ns = gu_split.exec_ns;
    // SwiGLU + hidden rotation, per token. The gate/up pair sits at strided
    // indices in the row-major output, so gather it; the rotation is the *down*
    // projection's contract (its weights are stored post-rotation).
    let t_middle = Instant::now();
    let mut gather_ns: u64 = 0;
    let mut silu_ns: u64 = 0;
    let mut fwht_ns: u64 = 0;
    {
        let gu = &scratch.gu[..2 * mi * slots];
        let hidden = &mut scratch.hidden[..mi * slots];
        let gather = &mut scratch.gather[..2 * mi];
        for gi in 0..groups {
            let n = group_n(&scratch.group_start, gi);
            let base = scratch.group_off[gi];
            let first = scratch.order[scratch.group_start[gi] as usize] as usize;
            let rotate = experts[first].down_quant.is_fwht_g256();
            for si in 0..n {
                let s = scratch.order[scratch.group_start[gi] as usize + si] as usize;
                let t_gather = Instant::now();
                for (r, g) in gather.iter_mut().enumerate() {
                    *g = gu[base + r * n + si];
                }
                gather_ns += t_gather.elapsed().as_nanos() as u64;
                let hid = &mut hidden[s * mi..(s + 1) * mi];
                let t_silu = Instant::now();
                silu_mul(gather, hid);
                silu_ns += t_silu.elapsed().as_nanos() as u64;
                if rotate {
                    let t_fwht = Instant::now();
                    rotate_x(hid);
                    fwht_ns += t_fwht.elapsed().as_nanos() as u64;
                }
            }
        }
    }
    timing.middle_ns = t_middle.elapsed().as_nanos() as u64;
    timing.middle_gather_ns = gather_ns;
    timing.middle_silu_ns = silu_ns;
    timing.middle_fwht_ns = fwht_ns;
    scratch.group_off.resize(groups + 1, 0);
    fill_group_off(&scratch.group_start, &mut scratch.group_off, dim);
    let dn_split = {
        let src = MoeJobSource {
            experts,
            order: &scratch.order,
            group_start: &scratch.group_start,
            act: &scratch.hidden[..mi * slots],
            row_len: mi,
            top_k,
            proj: MoeProj::Down,
        };
        hipfire_cpu::gemv::gemv_shared_sourced_traced(dim, mi, &mut scratch.dn, &src, None)
    };
    timing.dn_ns = dn_split.total_ns;
    timing.dn_prep_ns = dn_split.prep_ns;
    timing.dn_exec_ns = dn_split.exec_ns;
    // The down outputs are row-major per group, so a token's vector is strided:
    // `slot_group`/`slot_si` place each slot in its group's run, and the residual
    // accumulates in the original row-major, rank-minor order.
    let t_combine = Instant::now();
    {
        let dn = &scratch.dn[..dim * slots];
        for s in 0..slots {
            let t = s / top_k;
            let gi = scratch.slot_group[s] as usize;
            let si = scratch.slot_si[s] as usize;
            let n = group_n(&scratch.group_start, gi);
            let base = scratch.group_off[gi];
            let w = tw[s];
            let accrow = &mut residual_acc[t * dim..(t + 1) * dim];
            for (r, a) in accrow.iter_mut().enumerate() {
                *a += w * dn[base + r * n + si];
            }
        }
    }
    timing.combine_ns = t_combine.elapsed().as_nanos() as u64;
    Ok(())
}

/// Which projection of a selected expert a [`MoeJobSource`] job reads.
#[derive(Clone, Copy)]
enum MoeProj {
    GateUp,
    Down,
}

/// Group `gi`'s member count from the prefix-summed `group_start`.
fn group_n(group_start: &[u32], gi: usize) -> usize {
    (group_start[gi + 1] - group_start[gi]) as usize
}

/// Fill `group_off` (length `groups + 1`) with each group's element offset into a
/// projection buffer whose per-group run is `per_row * n(g)` elements — `2 * mi`
/// for gate_up, `dim` for down — plus a trailing total.
fn fill_group_off(group_start: &[u32], group_off: &mut [usize], per_row: usize) {
    let groups = group_start.len() - 1;
    let mut acc = 0usize;
    for gi in 0..groups {
        group_off[gi] = acc;
        acc += per_row * group_n(group_start, gi);
    }
    group_off[groups] = acc;
}

/// The splice's grouped window presented to
/// [`hipfire_cpu::gemv::gemv_shared_sourced`] without materializing a job slice:
/// every read is answered from the reusable index metadata (`order`,
/// `group_start`) plus the activation buffer and the resolved experts, so no
/// per-call descriptor allocation is needed and nothing borrows out of the
/// thread-local scratch after the call.
struct MoeJobSource<'a, 's> {
    experts: &'s [CpuMoeExpert<'a>],
    order: &'s [u32],
    group_start: &'s [u32],
    act: &'s [f32],
    /// Row width of `act`: `dim` for gate_up, `mi` for down.
    row_len: usize,
    top_k: usize,
    proj: MoeProj,
}

impl<'a, 's> MoeJobSource<'a, 's> {
    /// A group's expert — its first-seen slot; every slot in the group resolved to
    /// the same expert id, hence the same bytes.
    fn first_expert(&self, j: usize) -> &'s CpuMoeExpert<'a> {
        &self.experts[self.order[self.group_start[j] as usize] as usize]
    }
}

impl<'a, 's> hipfire_cpu::gemv::SharedJobSource for MoeJobSource<'a, 's> {
    fn jobs(&self) -> usize {
        self.group_start.len() - 1
    }
    fn n(&self, j: usize) -> usize {
        group_n(self.group_start, j)
    }
    fn quant(&self, j: usize) -> CpuQuant {
        let e = self.first_expert(j);
        match self.proj {
            MoeProj::GateUp => e.gate_up_quant,
            MoeProj::Down => e.down_quant,
        }
    }
    fn packed(&self, j: usize) -> &[u8] {
        let e = self.first_expert(j);
        match self.proj {
            MoeProj::GateUp => e.gate_up,
            MoeProj::Down => e.down,
        }
    }
    fn xs_row(&self, j: usize, t: usize) -> &[f32] {
        let slot = self.order[self.group_start[j] as usize + t] as usize;
        // gate_up consumes token `slot / top_k`'s activation row; down consumes
        // the slot's own post-SiLU hidden row.
        let row = match self.proj {
            MoeProj::GateUp => slot / self.top_k,
            MoeProj::Down => slot,
        };
        &self.act[row * self.row_len..(row + 1) * self.row_len]
    }
}

/// Per-shape accounting for the MoE splice under
/// `HIPFIRE_CPU_EXEC_TRACE=1`: one line per `(quant, dim, mi, top_k, rows,
/// graded)` at its first call and at every doubling, so the `calls=1` line is
/// the cold first step and later lines are steady state. Keyed by shape rather
/// than summed process-wide, for the same reason [`trace_step`] is.
#[allow(clippy::too_many_arguments)]
fn trace_moe_step(
    q: CpuQuant,
    dim: usize,
    mi: usize,
    k: usize,
    rows: usize,
    graded: bool,
    timing: MoeStepTiming,
) {
    if hipfire_config::developer_var("HIPFIRE_CPU_EXEC_TRACE").is_err() {
        return;
    }
    let Ok(mut shapes) = MOE_SHAPES.lock() else {
        return;
    };
    let stats = shapes.entry((q as u8, dim, mi, k, rows, graded)).or_default();
    stats.calls += 1;
    stats.d2h_enqueue_ns += timing.d2h_enqueue_ns;
    stats.d2h_complete_ns += timing.d2h_complete_ns;
    stats.gu_ns += timing.gu_ns;
    stats.dn_ns += timing.dn_ns;
    stats.h2d_ns += timing.h2d_ns;
    stats.setup_ns += timing.setup_ns;
    stats.middle_ns += timing.middle_ns;
    stats.combine_ns += timing.combine_ns;
    stats.total_ns += timing.total_ns;
    stats.gu_prep_ns += timing.gu_prep_ns;
    stats.gu_exec_ns += timing.gu_exec_ns;
    stats.dn_prep_ns += timing.dn_prep_ns;
    stats.dn_exec_ns += timing.dn_exec_ns;
    stats.middle_gather_ns += timing.middle_gather_ns;
    stats.middle_silu_ns += timing.middle_silu_ns;
    stats.middle_fwht_ns += timing.middle_fwht_ns;
    stats.d2h_wait_ns += timing.d2h_wait_ns;
    for i in 0..MOE_QUANT_MIX_LEN {
        stats.gu_mix[i] += timing.gu_mix[i] as u64;
        stats.dn_mix[i] += timing.dn_mix[i] as u64;
    }
    let stats = *stats;
    if !stats.calls.is_power_of_two() {
        return;
    }
    let (on_cpu, on_gpu) = cpu_exec_counters();
    let per_ms = |ns: u64| ns as f64 / 1e6 / stats.calls as f64;
    // Enqueue + completion is copy/API wall after the diagnostic producer drain;
    // neither region is a physical PCIe transfer measurement.
    let d2h_copy_ns = stats.d2h_enqueue_ns + stats.d2h_complete_ns;
    eprintln!(
        "cpu exec: moe expert splice dim={dim} mi={mi} k={k} rows={rows} quant={q:?} \
         graded={graded} | {} calls | {on_cpu} steps on CPU, {on_gpu} host-mapped steps still \
         on GPU | mean per call: d2h_enqueue={:.2}ms d2h_complete={:.2}ms gu={:.2}ms dn={:.2}ms \
         h2d={:.2}ms setup={:.2}ms middle={:.2}ms combine={:.2}ms total={:.2}ms \
         | gu_prep={:.2}ms gu_exec={:.2}ms dn_prep={:.2}ms dn_exec={:.2}ms \
         gather={:.2}ms silu={:.2}ms fwht={:.2}ms d2h_wait={:.2}ms d2h_copy_api={:.2}ms \
         | gu_mix=[{}] dn_mix=[{}]",
        stats.calls,
        per_ms(stats.d2h_enqueue_ns),
        per_ms(stats.d2h_complete_ns),
        per_ms(stats.gu_ns),
        per_ms(stats.dn_ns),
        per_ms(stats.h2d_ns),
        per_ms(stats.setup_ns),
        per_ms(stats.middle_ns),
        per_ms(stats.combine_ns),
        per_ms(stats.total_ns),
        per_ms(stats.gu_prep_ns),
        per_ms(stats.gu_exec_ns),
        per_ms(stats.dn_prep_ns),
        per_ms(stats.dn_exec_ns),
        per_ms(stats.middle_gather_ns),
        per_ms(stats.middle_silu_ns),
        per_ms(stats.middle_fwht_ns),
        per_ms(stats.d2h_wait_ns),
        per_ms(d2h_copy_ns),
        format_quant_mix(&stats.gu_mix, stats.calls),
        format_quant_mix(&stats.dn_mix, stats.calls),
    );
}

/// `CpuQuant` discriminant (`q as usize`) → short variant name for the
/// per-call quant-mixture print. Order matches the enum declaration; unknown
/// indices (past the current variant count) print as `q{N}`.
fn cpu_quant_mix_name(idx: usize) -> &'static str {
    match idx {
        0 => "Mq4G256",
        1 => "Mq4G256V2",
        2 => "Mq6G256",
        3 => "Mq3G256",
        4 => "Mq3G256Lloyd",
        5 => "Hfq6G256",
        6 => "Hfq3G256",
        7 => "Hfq2G256",
        8 => "Hfq4G128",
        9 => "Hfq3G128",
        10 => "Hfq2G128",
        11 => "Mq5G256",
        12 => "Mq2G256",
        13 => "Mq2G256Lloyd",
        14 => "Mq2G256LloydU",
        15 => "Mq4G256Lloyd",
        16 => "Mq4CG256",
        17 => "Mq6G256V2",
        18 => "Mq5G256V2",
        19 => "Mq3G256V2",
        20 => "Mq2G256V2",
        21 => "Tq2G128",
        22 => "Bq1G128",
        23 => "Hfq4G256",
        24 => "F16",
        25 => "F32",
        26 => "Bf16",
        27 => "Q8F16",
        _ => "q?",
    }
}

/// Mean slots per call per quant (`NamexMean` pairs, one decimal) from summed
/// mixture counts. Empty (trace shapes that predate the mix) prints `-`.
fn format_quant_mix(mix: &[u64; MOE_QUANT_MIX_LEN], calls: usize) -> String {
    let mut out = String::new();
    for (i, &sum) in mix.iter().enumerate() {
        if sum == 0 {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        let mean = sum as f64 / calls.max(1) as f64;
        out.push_str(&format!("{}x{mean:.1}", cpu_quant_mix_name(i)));
    }
    if out.is_empty() {
        out.push('-');
    }
    out
}

/// One splice call's wall-time split, handed to [`trace_moe_step`].
#[derive(Clone, Copy, Default)]
struct MoeStepTiming {
    d2h_enqueue_ns: u64,
    d2h_complete_ns: u64,
    gu_ns: u64,
    dn_ns: u64,
    h2d_ns: u64,
    /// Resolving the selected slots to host experts.
    setup_ns: u64,
    /// SiLU + optional hidden rotation, the whole per-token loop.
    middle_ns: u64,
    /// The routing-weighted accumulate into the residual.
    combine_ns: u64,
    /// The whole splice call's wall time.
    total_ns: u64,
    /// GU prep (activation-sum/metadata) vs Rayon decode-and-dot execution.
    /// `gu_ns` stays the whole region (`prep + exec`); `prep_ns` is 0 until
    /// the traced `gemv_shared_sourced` sibling lands.
    gu_prep_ns: u64,
    gu_exec_ns: u64,
    /// DN prep (activation-sum/metadata) vs Rayon decode-and-dot execution.
    dn_prep_ns: u64,
    dn_exec_ns: u64,
    /// CPU middle split: gather vs SiLU vs FWHT sums across the per-token loop.
    middle_gather_ns: u64,
    middle_silu_ns: u64,
    middle_fwht_ns: u64,
    /// Trace-only host producer drain, including synchronization API overhead.
    d2h_wait_ns: u64,
    /// Selected-slot GU / DN quant mixture of this call, indexed by
    /// `CpuQuant as usize`; summed into the shape stats by [`trace_moe_step`].
    gu_mix: [u16; MOE_QUANT_MIX_LEN],
    dn_mix: [u16; MOE_QUANT_MIX_LEN],
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
    let mut out = vec![0.0f32; n];
    download_into_f32(gpu, t, n, &mut out)?;
    Ok(out)
}

/// [`download_f32`] into a caller-owned buffer, so the MoE splice's thread-local
/// workspaces are reused instead of re-allocated every layer.
fn download_into_f32(
    gpu: &Gpu,
    t: &GpuTensor,
    n: usize,
    out: &mut [f32],
) -> Result<(), DispatchError> {
    assert!(
        t.numel() >= n,
        "cpu exec: tensor has {} elements, need {n}",
        t.numel()
    );
    assert!(
        out.len() >= n,
        "cpu exec: buffer has {} elements, need {n}",
        out.len()
    );
    // Safety: `out[..n]` is a live `n`-element f32 buffer; the slice covers
    // exactly those bytes and does not outlive the call.
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    gpu.memcpy_dtoh_auto(bytes, &t.buf)
        .map_err(|e| cpu_err(&format!("D2H: {e}")))
}

/// Enqueue into the pinned MoE stage on the producers' own stream. The caller
/// owns the stage and must drain on success, error, and before buffer reuse.
fn download_into_f32_async(
    gpu: &Gpu,
    t: &GpuTensor,
    out: &mut [f32],
) -> Result<(), DispatchError> {
    // SAFETY: the initialized f32 slice covers exactly these bytes. The
    // caller retains the allocation until the batch's completion barrier.
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(out))
    };
    gpu.hip.memcpy_dtoh_async(bytes, &t.buf, gpu.active_stream.as_ref())
        .map_err(|e| cpu_err(&format!("D2H enqueue: {e}")))
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

/// One MoE layer window's wall-time split on the CPU-deferred route, handed to
/// [`moe_window_trace`]. All spans are host wall (`Instant`), adjacent not
/// nested: `gpu_sealed_ns` covers the sealed GPU program (`execute_steps`),
/// `cpu_splice_ns` the CPU splice (`moe_cpu_experts`), and `other_ns` the
/// caller-side gap between/around them (scratch access, flag checks). The
/// caller records all three from one outer span so they reconcile against
/// `total_ns`; nothing here launches, syncs, or allocates on the hot path
/// beyond the guide struct itself.
#[derive(Clone, Copy, Default, Debug)]
pub struct MoeWindowTiming {
    /// Whole window: sealed GPU program + CPU splice + caller-side gap.
    pub total_ns: u64,
    /// Sealed GPU program (`execute_steps`): router/norm/shared GPU stages.
    pub gpu_sealed_ns: u64,
    /// CPU splice (`moe_cpu_experts`): whole routed FFN on the CPU.
    pub cpu_splice_ns: u64,
    /// Caller-side gap: `total - sealed - splice` (scratch, flags, hooks).
    pub other_ns: u64,
    /// Forward width that selected this route (1 = decode, ≤4 = narrow).
    pub rows: usize,
    /// Routed-expert count per token (`num_experts_per_tok`).
    pub top_k: usize,
}

/// Per-shape totals for [`moe_window_trace`]: one line per `(layer, rows,
/// top_k)` at its first window and at every doubling — the same power-of-two
/// schedule as [`trace_moe_step`], so the `calls=1` line is the cold first
/// window and later lines are steady state. Keyed per layer (not just shape)
/// so the printed `layer=` owns its `calls` count.
#[derive(Clone, Copy, Default)]
struct MoeWindowStats {
    calls: u64,
    total_ns: u64,
    gpu_sealed_ns: u64,
    cpu_splice_ns: u64,
    other_ns: u64,
}

/// `(layer, rows, top_k)` → running per-layer window totals.
static MOE_WINDOWS: LazyLock<Mutex<BTreeMap<(usize, usize, usize), MoeWindowStats>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// Opt-in aggregated per-layer window timing on the CPU-deferred MoE route,
/// under the same `HIPFIRE_CPU_EXEC_TRACE=1` gate as [`trace_moe_step`].
///
/// The arch caller records one window per MoE layer: the sealed GPU program
/// span, the CPU splice span, and the caller-side gap, all from host wall
/// timestamps around calls it already makes. No GPU events, no new device
/// sync, no execution-order change. Summaries print outside the measured
/// region (power-of-two schedule), and tracing is disabled by default with a
/// single snapshot-map read on the hot path.
pub fn moe_window_trace(layer_idx: usize, timing: MoeWindowTiming) {
    if moe_trace_enabled() {
        moe_window_trace_inner(layer_idx, timing);
    }
}

fn moe_window_trace_inner(layer_idx: usize, timing: MoeWindowTiming) {
    let Ok(mut windows) = MOE_WINDOWS.lock() else {
        return;
    };
    let stats = windows
        .entry((layer_idx, timing.rows, timing.top_k))
        .or_default();
    stats.calls += 1;
    stats.total_ns += timing.total_ns;
    stats.gpu_sealed_ns += timing.gpu_sealed_ns;
    stats.cpu_splice_ns += timing.cpu_splice_ns;
    stats.other_ns += timing.other_ns;
    let stats = *stats;
    if !stats.calls.is_power_of_two() {
        return;
    }
    let per_ms = |ns: u64| ns as f64 / 1e6 / stats.calls as f64;
    eprintln!(
        "cpu exec: moe window layer={layer_idx} rows={} k={} | {} calls | \
         mean per window: sealed_gpu={:.2}ms cpu_splice={:.2}ms other={:.2}ms total={:.2}ms",
        timing.rows,
        timing.top_k,
        stats.calls,
        per_ms(stats.gpu_sealed_ns),
        per_ms(stats.cpu_splice_ns),
        per_ms(stats.other_ns),
        per_ms(stats.total_ns),
    );
}

/// One MoE layer's GPU-vs-CPU interleave on the batched-slots route, handed to
/// [`moe_layer_trace`]. Host wall around the two halves the layer already
/// executes: `attn_ns` covers the attention half (QKVZA/recurrence or
/// QKV/attention + output projection, host wall including queued GPU work, not
/// device-only time), `moe_ns` covers the MoE FFN body (`prefill_moe_ffn_body`
/// including any CPU splice). Adjacent, not nested; `total_ns` is the outer
/// span so the parts reconcile. No GPU events, no new sync, no order change.
#[derive(Clone, Copy, Default, Debug)]
pub struct MoeLayerTiming {
    pub total_ns: u64,
    pub attn_ns: u64,
    pub moe_ns: u64,
    pub rows: usize,
}

/// Per-layer totals for [`moe_layer_trace`]: one line per `(layer, attn-kind,
/// rows)` at its first window and at every doubling — same power-of-two
/// schedule as the splice. Keyed per layer (not just shape) so the printed
/// `layer=` owns its `calls` count.
#[derive(Clone, Copy, Default)]
struct MoeLayerStats {
    calls: u64,
    total_ns: u64,
    attn_ns: u64,
    moe_ns: u64,
}

/// `(layer, full_attn, rows)` → running per-layer totals.
static MOE_LAYERS: LazyLock<Mutex<BTreeMap<(usize, bool, usize), MoeLayerStats>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// Opt-in per-layer GPU/CPU interleave timing on the batched-slots MoE route,
/// under the same `HIPFIRE_CPU_EXEC_TRACE=1` gate. The layer caller records
/// host wall around the attention half and the MoE FFN body it already runs;
/// summaries print outside the measured region. Disabled by default with a
/// single snapshot-map read on the hot path.
pub fn moe_layer_trace(layer_idx: usize, is_full_attn: bool, timing: MoeLayerTiming) {
    if moe_trace_enabled() {
        moe_layer_trace_inner(layer_idx, is_full_attn, timing);
    }
}

fn moe_layer_trace_inner(layer_idx: usize, is_full_attn: bool, timing: MoeLayerTiming) {
    let Ok(mut layers) = MOE_LAYERS.lock() else {
        return;
    };
    let stats = layers.entry((layer_idx, is_full_attn, timing.rows)).or_default();
    stats.calls += 1;
    stats.total_ns += timing.total_ns;
    stats.attn_ns += timing.attn_ns;
    stats.moe_ns += timing.moe_ns;
    let stats = *stats;
    if !stats.calls.is_power_of_two() {
        return;
    }
    let per_ms = |ns: u64| ns as f64 / 1e6 / stats.calls as f64;
    eprintln!(
        "cpu exec: moe layer layer={layer_idx} {} rows={} | {} calls | \
         mean per layer: attn={:.2}ms moe_ffn={:.2}ms total={:.2}ms",
        if is_full_attn { "fullattn" } else { "deltanet" },
        timing.rows,
        stats.calls,
        per_ms(stats.attn_ns),
        per_ms(stats.moe_ns),
        per_ms(stats.total_ns),
    );
}

/// One batched serve-engine step's wall-time split on the MoE route, handed to
/// [`moe_step_trace`]. Host wall around work the step already does:
/// `forward_ns` covers the batched forward (`ModelRig::forward_step` →
/// `forward_batch_slots_graphed_opts`, all layers), `sample_ns` the sampler
/// (`sample_per_slot`, GPU kernels + launch), `sync_readback_ns` the device
/// synchronize + token-id D2H, and `emit_ns` the accept/commit/emit loop.
/// Adjacent, not nested; `total_ns` is the outer step span so the parts
/// reconcile. No GPU events, no new sync, no order change.
#[derive(Clone, Copy, Default, Debug)]
pub struct MoeStepTime {
    pub total_ns: u64,
    pub forward_ns: u64,
    pub sample_ns: u64,
    pub sync_readback_ns: u64,
    pub emit_ns: u64,
    pub rows: usize,
}

/// Per-shape totals for [`moe_step_trace`]: one line per `rows` at its first
/// step and at every doubling — same power-of-two schedule as the splice.
#[derive(Clone, Copy, Default)]
struct MoeServeStepStats {
    calls: u64,
    total_ns: u64,
    forward_ns: u64,
    sample_ns: u64,
    sync_readback_ns: u64,
    emit_ns: u64,
}

/// `rows` → running per-shape step totals.
static MOE_STEPS: LazyLock<Mutex<BTreeMap<usize, MoeServeStepStats>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// Opt-in whole-step timing on the batched MoE route, under the same
/// `HIPFIRE_CPU_EXEC_TRACE=1` gate. The serve engine records host wall around
/// the forward, the sampler, the sync/readback, and the emit loop it already
/// runs; summaries print outside the measured region. Disabled by default
/// with a single snapshot-map read on the hot path.
pub fn moe_step_trace(timing: MoeStepTime) {
    if moe_trace_enabled() {
        moe_step_trace_inner(timing);
    }
}

fn moe_step_trace_inner(timing: MoeStepTime) {
    let Ok(mut steps) = MOE_STEPS.lock() else {
        return;
    };
    let stats = steps.entry(timing.rows).or_default();
    stats.calls += 1;
    stats.total_ns += timing.total_ns;
    stats.forward_ns += timing.forward_ns;
    stats.sample_ns += timing.sample_ns;
    stats.sync_readback_ns += timing.sync_readback_ns;
    stats.emit_ns += timing.emit_ns;
    let stats = *stats;
    if !stats.calls.is_power_of_two() {
        return;
    }
    let per_ms = |ns: u64| ns as f64 / 1e6 / stats.calls as f64;
    eprintln!(
        "cpu exec: moe step rows={} | {} calls | \
         mean per step: forward={:.2}ms sample={:.2}ms sync_readback={:.2}ms emit={:.2}ms total={:.2}ms",
        timing.rows,
        stats.calls,
        per_ms(stats.forward_ns),
        per_ms(stats.sample_ns),
        per_ms(stats.sync_readback_ns),
        per_ms(stats.emit_ns),
        per_ms(stats.total_ns),
    );
}
/// One AR whole-token wall-time split on the single-token decode route
/// (`hipfire-generate/ar.rs`), handed to [`moe_token_trace`]. Host wall around
/// work the loop already does: `forward_ns` covers the whole forward
/// (`qwen35::forward_scratch`, all layers including CPU splices), `sample_ns`
/// the sampler (GPU or grammar-gated CPU path), and `emit_ns` the
/// decode/commit/emit tail. Adjacent, not nested; `total_ns` is the outer
/// loop-iteration span so the parts reconcile. The critical-path caveat
/// stands: this is elapsed wall per token, not a proof that the parts are
/// serial — the forward's own CPU splices already serialize on the host, but
/// GPU work queued across the boundary still overlaps. No GPU events, no new
/// sync, no order change.
#[derive(Clone, Copy, Default, Debug)]
pub struct MoeTokenTiming {
    pub total_ns: u64,
    pub forward_ns: u64,
    pub sample_ns: u64,
    pub emit_ns: u64,
}

/// Running totals for [`moe_token_trace`]: one line at the first token and at
/// every doubling — same power-of-two schedule as the splice.
#[derive(Clone, Copy, Default)]
struct MoeTokenStats {
    calls: u64,
    total_ns: u64,
    forward_ns: u64,
    sample_ns: u64,
    emit_ns: u64,
}

/// Process-wide AR token totals (single shape — no key).
static MOE_TOKENS: LazyLock<Mutex<MoeTokenStats>> =
    LazyLock::new(|| Mutex::new(MoeTokenStats::default()));

/// Opt-in AR whole-token timing, under the same `HIPFIRE_CPU_EXEC_TRACE=1`
/// gate. The generate loop records host wall around the forward, the sample,
/// and the emit tail it already runs; summaries print outside the measured
/// region. Disabled by default with a single snapshot-map read on the hot
/// path. Against this the per-layer/per-splice critical path must reconcile
/// (16 splices + sealed GPU work + sample/emit ≈ token wall).
pub fn moe_token_trace(timing: MoeTokenTiming) {
    if moe_trace_enabled() {
        moe_token_trace_inner(timing);
    }
}

fn moe_token_trace_inner(timing: MoeTokenTiming) {
    let Ok(mut guard) = MOE_TOKENS.lock() else {
        return;
    };
    guard.calls += 1;
    guard.total_ns += timing.total_ns;
    guard.forward_ns += timing.forward_ns;
    guard.sample_ns += timing.sample_ns;
    guard.emit_ns += timing.emit_ns;
    let stats = *guard;
    drop(guard);
    if !stats.calls.is_power_of_two() {
        return;
    }
    let per_ms = |ns: u64| ns as f64 / 1e6 / stats.calls as f64;
    eprintln!(
        "cpu exec: moe token | {} calls | \
         mean per token: forward={:.2}ms sample={:.2}ms emit={:.2}ms total={:.2}ms",
        stats.calls,
        per_ms(stats.forward_ns),
        per_ms(stats.sample_ns),
        per_ms(stats.emit_ns),
        per_ms(stats.total_ns),
    );
}

/// Per-shape step accounting under `HIPFIRE_CPU_EXEC_TRACE=1`: one line per
/// distinct step shape at its first call and then at every doubling of that
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

#[cfg(test)]
mod moe_splice_tests {
    use super::*;

    /// One packed group of weights for `q`, mirroring `hipfire_cpu`'s own test
    /// fixture: a valid, finite header plus a bijective payload, so a wrong
    /// offset or format fails rather than matching by symmetry.
    fn group_bytes(q: CpuQuant, salt: usize) -> Vec<u8> {
        let mut b: Vec<u8> = (0..q.group_bytes())
            .map(|i| (((i + salt) * 37 + 11) & 0xFF) as u8)
            .collect();
        let f32x2 = |a: f32, c: f32| {
            let mut o = [0u8; 8];
            o[..4].copy_from_slice(&a.to_le_bytes());
            o[4..].copy_from_slice(&c.to_le_bytes());
            o
        };
        match q {
            CpuQuant::Mq4G256 => b[..8].copy_from_slice(&f32x2(0.03125, -0.5)),
            CpuQuant::Mq6G256 => b[..8].copy_from_slice(&f32x2(0.0078125, -0.125)),
            CpuQuant::Mq3G256Lloyd => {
                const CB: [u16; 8] =
                    [0xbc00, 0xb800, 0xb400, 0xb000, 0x3000, 0x3400, 0x3800, 0x3c00];
                for (k, v) in CB.iter().enumerate() {
                    b[2 * k..2 * k + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
            _ => {}
        }
        b
    }

    /// A `[m, k]` weight tensor's bytes, group by group.
    fn weight_bytes(q: CpuQuant, m: usize, k: usize, salt: usize) -> Vec<u8> {
        let groups = k / q.group_elems();
        let mut out = Vec::with_capacity(m * groups * q.group_bytes());
        for row in 0..m {
            for g in 0..groups {
                out.extend_from_slice(&group_bytes(q, salt + row * groups + g));
            }
        }
        out
    }

    /// The top-k scratch stores an id as the low bits of an `f32` bit pattern
    /// (`expert_id` inverts this).
    fn id_bits(e: usize) -> f32 {
        f32::from_bits(e as u32)
    }

    fn x_of(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.017).sin() * 0.5).collect()
    }

    /// A uniform layer: the multirow engine must reproduce repeated single-row
    /// CPU expert evaluation, measured against the existing per-expert reference
    /// ([`hipfire_cpu::moe::run_experts`]) over the same packed blobs.
    #[test]
    fn uniform_multirow_matches_run_experts() {
        let (dim, mi) = (256usize, 256usize);
        let (rows, top_k, n_exp) = (4usize, 3usize, 4usize);
        let q = CpuQuant::Mq4G256;
        let gu_stride = hipfire_cpu::gemv::row_bytes(q, dim) * 2 * mi;
        let dn_stride = hipfire_cpu::gemv::row_bytes(q, mi) * dim;
        let mut gu_all = Vec::with_capacity(n_exp * gu_stride);
        let mut dn_all = Vec::with_capacity(n_exp * dn_stride);
        for e in 0..n_exp {
            gu_all.extend_from_slice(&weight_bytes(q, 2 * mi, dim, 100 + e));
            dn_all.extend_from_slice(&weight_bytes(q, dim, mi, 200 + e));
        }
        // Repeated experts across rows, and the higher-id experts selected too.
        let idx: [usize; 12] = [0, 2, 3, 2, 2, 0, 3, 0, 1, 1, 3, 3];
        let wts: [f32; 12] = [0.5, 0.3, 0.2, 0.4, 0.4, 0.2, 0.25, 0.35, 0.4, 0.6, 0.2, 0.2];
        let x = x_of(rows * dim);
        let seed: Vec<f32> = (0..rows * dim).map(|i| 0.1 * (i % 7) as f32).collect();
        let ti: Vec<f32> = idx.iter().map(|&e| id_bits(e)).collect();

        let experts: Vec<CpuMoeExpert> = (0..rows * top_k)
            .map(|s| {
                let e = idx[s];
                CpuMoeExpert {
                    gate_up_quant: q,
                    gate_up: &gu_all[e * gu_stride..(e + 1) * gu_stride],
                    down_quant: q,
                    down: &dn_all[e * dn_stride..(e + 1) * dn_stride],
                }
            })
            .collect();
        let mut scratch = MoeScratch::default();
        let mut acc = seed.clone();
        moe_cpu_experts_host(
            dim,
            mi,
            rows,
            top_k,
            &experts,
            &x,
            &ti,
            &wts,
            &mut acc,
            &mut scratch,
            &mut MoeStepTiming::default(),
        )
        .unwrap();

        let blobs = hipfire_cpu::moe::ExpertBlobs {
            gate_up: &gu_all,
            down: &dn_all,
            gate_up_stride: gu_stride,
            down_stride: dn_stride,
            quant: q,
            dim,
            mi,
        };
        let mut reference = seed.clone();
        for t in 0..rows {
            let routing: Vec<(usize, f32)> = (0..top_k)
                .map(|j| (idx[t * top_k + j], wts[t * top_k + j]))
                .collect();
            hipfire_cpu::moe::run_experts(
                &blobs,
                &x[t * dim..(t + 1) * dim],
                &routing,
                &mut reference[t * dim..(t + 1) * dim],
            )
            .unwrap();
        }
        assert_eq!(
            acc, reference,
            "multirow CPU output != repeated single-row run_experts"
        );
    }

    /// A graded layer: experts of different tiers, including one whose gate_up
    /// and down formats differ, and a higher-id cold (MQ3-Lloyd) expert. The
    /// reference is the per-(row, rank) single-expert evaluation built from the
    /// crate's own `gemv` primitive.
    #[test]
    fn graded_multirow_matches_per_expert_gemv() {
        let (dim, mi) = (256usize, 256usize);
        let (rows, top_k, n_exp) = (3usize, 4usize, 4usize);
        let gu_q = [
            CpuQuant::Mq6G256,
            CpuQuant::Mq4G256,
            CpuQuant::Mq3G256Lloyd,
            CpuQuant::Mq6G256,
        ];
        let dn_q = [
            CpuQuant::Mq6G256,
            CpuQuant::Mq4G256,
            CpuQuant::Mq3G256Lloyd,
            CpuQuant::Mq3G256Lloyd,
        ];
        let gu: Vec<Vec<u8>> = (0..n_exp)
            .map(|e| weight_bytes(gu_q[e], 2 * mi, dim, 300 + e))
            .collect();
        let dn: Vec<Vec<u8>> = (0..n_exp)
            .map(|e| weight_bytes(dn_q[e], dim, mi, 400 + e))
            .collect();
        let idx: [usize; 12] = [0, 1, 2, 3, 2, 2, 3, 0, 3, 1, 1, 2];
        let wts: [f32; 12] = [0.4, 0.3, 0.2, 0.1, 0.5, 0.3, 0.2, 0.4, 0.6, 0.25, 0.35, 0.4];
        let x = x_of(rows * dim);
        let seed: Vec<f32> = (0..rows * dim).map(|i| 0.05 * (i % 5) as f32).collect();
        let ti: Vec<f32> = idx.iter().map(|&e| id_bits(e)).collect();

        let experts: Vec<CpuMoeExpert> = (0..rows * top_k)
            .map(|s| {
                let e = idx[s];
                CpuMoeExpert {
                    gate_up_quant: gu_q[e],
                    gate_up: &gu[e],
                    down_quant: dn_q[e],
                    down: &dn[e],
                }
            })
            .collect();
        let mut scratch = MoeScratch::default();
        let mut acc = seed.clone();
        moe_cpu_experts_host(
            dim,
            mi,
            rows,
            top_k,
            &experts,
            &x,
            &ti,
            &wts,
            &mut acc,
            &mut scratch,
            &mut MoeStepTiming::default(),
        )
        .unwrap();

        let mut reference = seed.clone();
        for t in 0..rows {
            for j in 0..top_k {
                let s = t * top_k + j;
                let e = idx[s];
                let mut gv = vec![0.0f32; 2 * mi];
                gemv(gu_q[e], &gu[e], 2 * mi, dim, &x[t * dim..(t + 1) * dim], &mut gv);
                let mut hid = vec![0.0f32; mi];
                silu_mul(&gv, &mut hid);
                if dn_q[e].is_fwht_g256() {
                    rotate_x(&mut hid);
                }
                let mut dout = vec![0.0f32; dim];
                gemv(dn_q[e], &dn[e], dim, mi, &hid, &mut dout);
                for (a, v) in reference[t * dim..(t + 1) * dim].iter_mut().zip(dout) {
                    *a += wts[s] * v;
                }
            }
        }
        assert_eq!(
            acc, reference,
            "graded multirow CPU output != per-expert gemv reference"
        );
    }

    /// A short expert span is an error, never a silent read past it.
    #[test]
    fn short_expert_span_is_an_error() {
        let (dim, mi) = (256usize, 256usize);
        let q = CpuQuant::Mq4G256;
        let short = vec![0u8; 8];
        let experts = vec![CpuMoeExpert {
            gate_up_quant: q,
            gate_up: &short,
            down_quant: q,
            down: &short,
        }];
        let mut scratch = MoeScratch::default();
        let err = moe_cpu_experts_host(
            dim,
            mi,
            1,
            1,
            &experts,
            &vec![0.0f32; dim],
            &[id_bits(0)],
            &[1.0],
            &mut vec![0.0f32; dim],
            &mut scratch,
            &mut MoeStepTiming::default(),
        )
        .expect_err("a short expert span must fail");
        assert!(format!("{err}").contains("span"), "unexpected error: {err}");
    }

    /// An expert count that disagrees with `rows × top_k` is an error.
    #[test]
    fn expert_count_mismatch_is_an_error() {
        let (dim, mi) = (256usize, 256usize);
        let q = CpuQuant::Mq4G256;
        let gu = weight_bytes(q, 2 * mi, dim, 1);
        let dn = weight_bytes(q, dim, mi, 2);
        let experts = vec![CpuMoeExpert {
            gate_up_quant: q,
            gate_up: &gu,
            down_quant: q,
            down: &dn,
        }];
        let mut scratch = MoeScratch::default();
        // rows=2, top_k=1 needs 2 experts; only 1 supplied.
        assert!(
            moe_cpu_experts_host(
                dim,
                mi,
                2,
                1,
                &experts,
                &vec![0.0f32; 2 * dim],
                &[id_bits(0), id_bits(0)],
                &[1.0, 1.0],
                &mut vec![0.0f32; 2 * dim],
                &mut scratch,
                &mut MoeStepTiming::default()
            )
            .is_err()
        );
    }
}
