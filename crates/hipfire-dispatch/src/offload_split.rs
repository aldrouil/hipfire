// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Pass-back: **scheduled co-inference of the layers spilled into system RAM**
//! (`memory.offload_exec=passback`).
//!
//! The partial-offload feature spills a contiguous prefix of layers to pinned
//! host RAM (`memory.gpu_layer_budget`) and executes their weight-reading GEMVs
//! through one of two engines: `pcie` — the GPU reading the host-mapped weights
//! over the link — or `cpu` — the CPU SIMD kernels reading host RAM directly.
//! Pass-back **runs both engines on the same spilled step, concurrently**. It is a
//! mixture of those two existing paths working in concert, not a third engine and
//! not a fallback: the GPU arm *is* the `pcie` path restricted to a row range, the
//! CPU arm *is* the `cpu` path restricted to the rest, and `share = 0` is
//! byte-identical to `memory.offload_exec=cpu`.
//!
//! Why it pays: a CPU-only step is a host sync point — a blocking D2H, a host
//! GEMV, a blocking H2D — so the GPU sits idle for most of its wall
//! (`docs/perf-checkpoints/2026-10-01-offload-passback-headroom-idle.md` measured
//! 77.3–94.6 % on the 9B at 8–24 of 32 layers spilled), while a CPU GEMV stream
//! and a GPU PCIe read of the *same* host-mapped pages are near-additive up to
//! 75–78 GB/s combined against ~52 GB/s for the CPU alone. Co-inference spends
//! that idle window on the same host bytes.
//!
//! **The schedule is per step, not per layer.** Consecutive layers are serially
//! dependent through the residual stream, so moving a whole *layer* between the
//! engines only swaps who idles; the mixture has to be *within* a step. A spilled
//! step's output rows are independent, so the step is divided by output rows: rows
//! `[0, g)` go to the GPU (the `pcie` path) and rows `[g, m)` stay on the CPU (the
//! `cpu` path), concurrently. A step that ends up wholly on one engine is a
//! **degenerate point of this schedule** (`g → 0` for an all-CPU step), not a
//! validation failure and not a fallback to a worse path — the eligibility gates
//! below only bound which steps can be co-inferenced at all.
//!
//! Two properties make it correct and cheap:
//!
//! * **Both arms are the existing paths, over a row range.** The GPU arm is
//!   `pcie` mode's launch restricted to rows `[0, g)`
//!   ([`crate::pipeline::steps::launch_op_rows`]); the CPU arm is `cpu` mode's
//!   step restricted to rows `[g, m)` ([`crate::cpu_exec::cpu_arm_prepare`] /
//!   [`crate::cpu_exec::cpu_arm_finish`]). No third engine and no new numerics,
//!   which is also why a share of `0` is byte-identical to
//!   `memory.offload_exec=cpu`.
//! * **No second stream and no events.** The overlap is an ordering result:
//!   issue the blocking D2H *before* enqueueing the GPU arm (so it drains only
//!   the step's producer), enqueue the GPU arm (async), run the CPU multiply
//!   while that kernel executes, then do the blocking H2D of the CPU's rows —
//!   which is stream-ordered after the GPU arm because both are on the same
//!   (default) stream. The copies are `k*4` bytes down and `(m-g)*4` up (~30 KB
//!   at 9B shapes) against tens of MB of weight bytes, so a two-stream design
//!   would buy the overlap of a ~2 µs copy.
//!
//! The **share** `g` is scheduled, not hardcoded: the optimum is a property of
//! the host (link width, DRAM peak, core count, AVX2), so the first
//! split-eligible step of each `(dtype, k)` seeds itself by timing both engines
//! on that step's own weight buffer, and every split step then refines the share
//! from its own arm timings. No host constant is load-bearing: the probe
//! accelerates convergence and the online controller corrects it.
//!
//! The controller reads the blocking H2D (`join`) against the shape's own copy
//! floor — the smallest join it has seen, i.e. what the copy costs when the GPU
//! finished first — plus a hysteresis margin. Only the excess over
//! `floor + margin` can be the GPU arm overrunning the CPU multiply, and then
//! `gemv_ns + excess` *is* the arm's duration. Below the balance point the GPU
//! finishes first on every step, so its duration is only *bounded*; those steps
//! are kept as censored observations, and the arm's time-per-byte is a censored
//! (Tobit) EM estimate over both exact and censored observations, anchored by the
//! one-time probe's directly measured rate — the estimator's identifiability
//! anchor and its guard against a floor-noise runaway. The setpoint is the DLT
//! balance point `r_gpu/(r_gpu+r_cpu)`. Reading the raw join as a GPU duration in
//! the CPU-straggler regime is how a split ends up pinned to its own floor with
//! both engines idle in turn. The estimator's sources and its known limits are
//! documented above [`estimate_tau_gpu`].
//!
//! Scope is the dense qwen3.5 seam only (the step lists that carry
//! `qkv_via_execute_steps`, `qkvza_via_execute_steps`,
//! `gate_up_via_execute_steps` and the dense `Step::GemvResidual` sites). MoE /
//! routed-expert paths never reach it and get no arms here.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

use hipfire_config::memory::PassbackShare;
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::context::DispatchCtx;
use crate::cpu_exec::{self, HostExec, StepTiming};
use crate::families::gemv::{weight_row_bytes, WeightRef};
use crate::pipeline::steps::{launch_op_rows, GemvInput, Step};
use crate::types::{dtype_rotation_plan, DispatchError, KernelKey, RotationPlan};

/// Split offsets are rounded down to a multiple of this many rows: 2 covers the
/// residual kernels' `row0 = blockIdx.x << 1` + `float2` store at `y + row0` (an
/// odd offset would misalign that store) and 8 keeps every covered format's
/// weight byte offset 4-byte aligned (their row strides are multiples of 8).
const ROW_ALIGN: usize = 8;

/// Below this many weight bytes the extra launch + join outweighs the overlap;
/// the step runs wholly on the CPU. At 9B shapes the covered projections are
/// 4–50 MB, so this only catches the tiny ones (the DeltaNet beta/alpha rows,
/// ~64 KB).
pub const MIN_SPLIT_BYTES: usize = 2 << 20;

/// First-trial share before either engine has been timed, and the floor/ceiling
/// of every scheduled share. 0.375 is the 2026-10-01 probe's balance point on
/// gfx1201 (27.3 GB/s link vs 46 GB/s contended CPU) — a starting guess the
/// controller corrects, not a hardware assumption the feature depends on.
const DEFAULT_GPU_SHARE: f64 = 0.375;
const SHARE_MIN: f64 = 0.05;
const SHARE_MAX: f64 = 0.50;

/// Reps per engine in the seeding probe: the median of three is enough to reject
/// a cold first touch, and each rep is one full pass over the step's weight
/// buffer.
const PROBE_REPS: usize = 3;

/// How much of the controller's proposal to apply per adjustment, and how often
/// to adjust. The proposal is the DLT balance point (see [`next_share`]); the
/// gain only paces how fast the share walks to it. No integral term is needed —
/// the target *is* the estimate's own balance point, so there is no offset to
/// remove.
const ALPHA: f64 = 0.3;
const APPLY_EVERY: u64 = 4;

/// EWMA weight for the CPU arm's time-per-byte sample.
const EWMA_ALPHA: f64 = 0.25;

/// The GPU estimator: the number of online observations it keeps (a rolling
/// window), how often it re-estimates, the assumed coefficient of variation of
/// the GPU arm's per-byte time, the weight — in online pseudo-observations — of
/// the probe's directly measured GPU rate, and the EM iteration cap.
///
/// `ESTIMATE_EVERY` is the estimator's update cadence and is set equal to
/// `ESTIMATE_WINDOW`, so each update sees one full, disjoint window (the rolling
/// buffer degenerates to the batch it replaced). A shorter cadence — a sliding
/// window — was tried against the investigation doc's §1.2 "lag, not offset" lever
/// and measured as a **regression** on the 9B fixture: 31.60 vs 32.50 tok/s
/// (median of five interleaved fresh-process pairs, 5/5 lost), because the lag was
/// low-pass-filtering the anchor bias rather than costing throughput. Do not
/// shorten it again without a paired measurement that says otherwise.
///
/// `TAU_PRIOR_WEIGHT` and `TAU_CV` are the two knobs a reader should scrutinise:
/// * `TAU_PRIOR_WEIGHT` is the identifiability anchor, not a tuning knob. If
///   every observation is censored (the GPU always finishing first) the GPU mean
///   is not identified by the data at all — the likelihood is flat in the mean —
///   so the estimator must lean on the one direct measurement. It is stated as a
///   number of online samples, and the estimate moves monotonically with it; a
///   value near 16 keeps it a strong prior without freezing out a sustained run
///   of genuine overruns.
/// * `TAU_CV` is an *assumed* shape parameter, not fitted — a normal model with a
///   plausible spread is all the EM needs for the mean, and the mean is what the
///   share uses. [`estimate_tau_gpu_is_insensitive_to_the_iteration_cap`] pins the
///   cap; the CV is deliberately left as an assumption rather than fitted to a
///   placement.
const ESTIMATE_WINDOW: usize = 64;
const ESTIMATE_EVERY: u64 = 64;
const TAU_CV: f64 = 0.5;
const TAU_PRIOR_WEIGHT: f64 = 16.0;
const ESTIMATE_ITERS: usize = 8;

/// `enabled` + `share` supplied by the caller, so the parity test and the
/// planner tests need no process snapshot.
#[derive(Clone, Copy, Debug)]
pub struct PassbackOptions {
    pub enabled: bool,
    pub share: PassbackShare,
}

impl PassbackOptions {
    /// The configuration for this process: `memory.offload_exec=passback` and
    /// `memory.offload_passback_share`.
    pub fn from_process() -> Self {
        PassbackOptions {
            enabled: enabled(),
            share: hipfire_config::memory::offload_passback_share(),
        }
    }
}

/// The seam's entry point: reads the process config snapshot.
pub fn run(gpu: &mut Gpu, ctx: &DispatchCtx, step: &Step) -> Result<bool, DispatchError> {
    run_with(gpu, ctx, step, &PassbackOptions::from_process())
}

/// `memory.offload_exec == passback` (via the cached CPU-exec predicate, which
/// covers both modes that execute spilled steps host-side).
pub fn enabled() -> bool {
    cpu_exec::passback_enabled()
}

/// Whether `dtype` can be *split*: a CPU decoder exists **and** its vector row dot
/// is available. With the scalar decoder the CPU rate (~10 GMAC/s) is well below
/// the GPU's PCIe read rate, so the best share is all-GPU — which plain `pcie`
/// does better. The load-time coverage line counts a splittable layer with this.
pub fn passback_capable_format(dtype: DType) -> bool {
    cpu_exec::cpu_quant_for(dtype).is_some_and(|q| hipfire_cpu::simd::row_dot_enabled(q, None))
}

/// Bytes per weight row for `dtype` at `k`, or `None` when there is no CPU
/// decoder for it.
///
/// Re-exported for callers that size a probe without depending on `hipfire-cpu`
/// (the CLI's `hipfire offload-bench` route reaches this through
/// `hipfire-runtime`): the CPU decoder is the single source of truth for the row
/// stride, which is also the invariant the feasibility gate checks the weight's
/// byte length against.
pub fn row_bytes_for(dtype: DType, k: usize) -> Option<usize> {
    cpu_exec::cpu_quant_for(dtype).map(|q| hipfire_cpu::gemv::row_bytes(q, k))
}

/// The share of a step's rows the GPU takes, chosen from `(m, row_bytes, share)`.
///
/// `None` when the step must run wholly on the CPU: a share that is not strictly
/// inside `(0, 1)`, fewer than two alignment quanta of rows, or a weight below
/// [`MIN_SPLIT_BYTES`]. `share = 0` therefore lands here, which is what makes
/// `memory.offload_passback_share = 0` byte-identical to
/// `memory.offload_exec=cpu`.
pub(crate) fn plan_rows(m: usize, row_bytes: usize, share: f64) -> Option<usize> {
    if !(share > 0.0 && share < 1.0) {
        return None;
    }
    if m < 2 * ROW_ALIGN {
        return None;
    }
    if row_bytes.checked_mul(m)? < MIN_SPLIT_BYTES {
        return None;
    }
    let g = align_down((m as f64 * share) as usize, ROW_ALIGN);
    Some(g.clamp(ROW_ALIGN, m - ROW_ALIGN))
}

fn align_down(v: usize, align: usize) -> usize {
    v / align * align
}

fn clamp_share(share: f64) -> f64 {
    share.clamp(SHARE_MIN, SHARE_MAX)
}

// ── The scheduler ──────────────────────────────────────
//
// Sources. The scheduler is the standard two-processor result plus a censored
// estimator, not a bespoke controller. Entries are marked [read] where the cited
// text was actually read in this worktree, [index] where only a bibliographic
// record was available — do not treat an [index] entry as verified.
//
// * Setpoint: divisible load theory's optimality principle — "All the processors
//   should finish computing at the same moment to achieve the smallest T_f" — and
//   the closed form T_f = α_i·W_i·T_cp, so the optimal partition α_i ∝ 1/W_i, the
//   processor's rate: Wu, Cao & Robertazzi, "Optimal Divisible Load Scheduling
//   for Resource-Sharing Network," arXiv:1902.01898 §II-A (2019) [read]. That
//   paper attributes the optimality proof to Cheng & Robertazzi, *IEEE Trans.
//   Computers* 1994 [index] — the specific closed-form result was not read here.
// * Processors whose speed drifts with background load (this host's contention)
//   are the subject of the same paper [read].
// * Censored estimation is the Tobit model: Tobin, "Estimation of Relationships
//   for Limited Dependent Variables," *Econometrica* 26(1), 1958 [index]. The
//   identifiability caveat that motivates the probe anchor — a censored model can
//   be non-identifiable without a priori information — is Wang, "Identifiability
//   and Estimation of Censored Errors-in-Variables Models," ASA 1994 Proceedings
//   §2 [read].
// * Rejected on the doc's own terms. Gradient-free perturbation methods
//   (SPSA: Spall, "An Overview of the Simultaneous Perturbation Method," *JHU APL
//   Technical Digest* 19(4), 1998 [read], which requires "L(u) sufficiently smooth
//   (several times differentiable) near u*"; extremum seeking: Krstić & Wang,
//   *Automatica*, doi:10.1016/S0005-1098(99)00183-1 [index]) inject a probing
//   signal — the deliberate excitation `docs/investigations/2026-10-01-offload-passback-scheduling-algorithms-revised.md`
//   §0.3 rules out for a production path. Classical PID tuning (ZN, Cohen–Coon,
//   IMC) is rejected there §2, as targeting a steady-state offset this plant does
//   not have (§1.1).
//
// Portability. The next host may have a different link width, DRAM peak, core
// count, ISA and contention. Nothing in the estimator encodes this host's rates:
// * every rate it uses is *measured on the target host* — the CPU arm's online,
//   and the GPU arm's from the one-time probe plus the online censored samples —
//   so a different machine is a different measurement, not a different constant;
// * its parameters are dimensionless (EWMA weights, an EM iteration cap, a
//   prior described in "online samples", an assumed shape) or timer-resolution
//   (`JOIN_MARGIN_NS`, bounded by the clock, not the hardware);
// * the only value that even *mentions* this host, `DEFAULT_GPU_SHARE`, is a
//   starting guess the probe overrides on the first split step.
// Two host-scaling hazards remain and are recorded here rather than hidden: the
// fixed equilibrium term `margin/((τ0+τ_cpu)·B)` grows for small shapes and fast
// hosts (the margin is absolute; `B` and the rates are the host's), and
// `SHARE_MAX` caps the optimum outright on a host whose GPU is far faster than
// its CPU. Both are pre-existing bounds, not properties of the estimator.
//
// A third failure mode is the automatic probe not running or failing on an
// unknown host. Then there is no anchor, and below the balance point the
// estimator is unidentifiable (every observation censored), so left alone it
// would sit at `DEFAULT_GPU_SHARE`. It does not: [`observe`] escalates the share
// upward while unanchored and unobserved-by-overrun, until an overrun yields the
// first exact sample and the estimate becomes identifiable. The trace marks the
// state (`anchor=none`), and `hipfire offload-bench`'s `seed()`/`fallback`
// installs a measured anchor outright, so the escape is a floor, not the plan.

/// Φ(x), the standard normal CDF, via the Abramowitz & Stegun 7.1.26 erf
/// approximation (|ε| ≤ 1.5e-7) — accurate enough for the EM below and free of a
/// dependency.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// φ(x), the standard normal density.
fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn erf(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26, valid for x >= 0; odd-symmetric otherwise.
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    if x >= 0.0 {
        y
    } else {
        -y
    }
}

/// Estimate the GPU arm's **time per weight byte**, from right-censored
/// observations, by EM under a fixed-coefficient-of-variation normal model.
///
/// A split step yields one of two observations of the GPU arm's duration `D_g`:
/// * the arm **overran** the CPU multiply, and `D_g = gemv_ns + excess` is exact
///   (`exact`, in ns/byte); or
/// * the arm **finished first**, and all the data say is `D_g ≤ gemv_ns + margin`
///   (`censored`, a threshold in ns/byte).
///
/// The shipped estimator averaged only the exact samples, which conditions the
/// estimate on the GPU's slow tail and drives the share below the balance point.
/// This is the regular censored-regression (Tobit) likelihood; `prior` is the
/// probe's directly measured time per byte, entering as [`TAU_PRIOR_WEIGHT`]
/// pseudo-observations, and it is what makes the estimate identifiable when every
/// online observation is censored.
///
/// **Known limit of this estimator.** Its fixed point is where the anchor `τ0` and
/// the censoring bound disagree no more, `τ0·s·B = (1−s)·τ_cpu·B + margin`. The
/// anchor is a *solo* measurement while the optimum is defined on *contended*
/// rates, so that fixed point is not the DLT balance: on the reference fixture it
/// lands near 0.46 against a measured optimum of ~0.42. No value of
/// [`TAU_PRIOR_WEIGHT`] removes it — above the fixed point the bound votes against
/// the anchor — so closing the gap needs a contended GPU-rate measurement, which
/// the censored data cannot supply below the balance point (the identifiability
/// limit noted above).
///
/// It also means the estimator does **not** converge the GPU level over a long
/// horizon. The anchor is fixed and load-bearing, not only for identifiability:
/// the exact (overrun) samples are contaminated by floor-noise overruns whose
/// reconstructed `τ = (gemv + excess)/(s·B)` grows as the share falls, so fading
/// the anchor lets the estimate run away downward — measured on the reference
/// fixture as `est_gpu` collapsing to 2–9 GB/s and the share to 0.06–0.21. The
/// CPU side tracks fully (online EWMA); the GPU side is pinned to the one-time
/// probe. Removing that pin needs a floor that separates genuine overruns from
/// copy noise, which is the place to work next.
fn estimate_tau_gpu(
    exact: &[f64],
    censored: &[f64],
    prior: Option<f64>,
    prior_weight: f64,
) -> Option<f64> {
    estimate_tau_gpu_iters(exact, censored, prior, prior_weight, ESTIMATE_ITERS)
}

/// [`estimate_tau_gpu`] with an explicit iteration cap, so a test can show the cap
/// is not load-bearing.
fn estimate_tau_gpu_iters(
    exact: &[f64],
    censored: &[f64],
    prior: Option<f64>,
    prior_weight: f64,
    iters: usize,
) -> Option<f64> {
    // A starting value that cannot be a local artifact: the anchor, else the exact
    // mean, else the largest censoring threshold (an upper bound on the mean).
    let mut mu = prior
        .or_else(|| (!exact.is_empty()).then(|| exact.iter().sum::<f64>() / exact.len() as f64))
        .or_else(|| {
            censored
                .iter()
                .copied()
                .fold(None, |acc: Option<f64>, u| Some(acc.map_or(u, |a| a.max(u))))
        })?;
    if !(mu.is_finite() && mu > 0.0) {
        return None;
    }
    for _ in 0..iters {
        let sigma = TAU_CV * mu;
        let mut sum = 0.0f64;
        let mut n = 0.0f64;
        for t in exact {
            sum += *t;
            n += 1.0;
        }
        for u in censored {
            let alpha = (u - mu) / sigma;
            let cdf = normal_cdf(alpha);
            // E[tau | tau <= u] = mu - sigma * phi(alpha)/Phi(alpha). The Mills
            // ratio tends to `u` as `u` falls below `mu` (a hard bound pulls the
            // estimate down to it) and to 0 as `u` rises above it (no information).
            let imputed = if cdf > 1e-9 {
                mu - sigma * normal_pdf(alpha) / cdf
            } else {
                *u
            };
            sum += imputed.max(0.0);
            n += 1.0;
        }
        if let Some(p) = prior {
            sum += prior_weight * p;
            n += prior_weight;
        }
        if n <= 0.0 {
            break;
        }
        let next = sum / n;
        if !(next.is_finite() && next > 0.0) {
            break;
        }
        let converged = (next - mu).abs() <= 1e-4 * mu;
        mu = next;
        if converged {
            break;
        }
    }
    Some(mu)
}

/// The probe's directly measured GPU rate as a time per weight byte.
fn tau_from_rate(gpu_bytes_per_s: f64) -> Option<f64> {
    (gpu_bytes_per_s > 0.0 && gpu_bytes_per_s.is_finite()).then(|| 1e9 / gpu_bytes_per_s)
}

/// A shape's split state: the share in force, the two engines' estimated
/// time-per-byte, and the copy floor the join is read against.
struct ShapeState {
    share: f64,
    /// The CPU arm's time per weight byte (ns/byte), from the multiply alone —
    /// never the preceding D2H, which is not part of the arm's throughput.
    tau_cpu: Option<f64>,
    /// The GPU arm's time per weight byte: the censored-MLE estimate, or the
    /// probe's direct measurement until the first update.
    tau_gpu: Option<f64>,
    /// The probe's directly measured GPU time per byte — the identifiability
    /// anchor the censored estimate leans on.
    tau_gpu_prior: Option<f64>,
    /// The smallest blocking-H2D time seen for this shape: the copy floor every
    /// join is read against. A running minimum, deliberately: it never rises onto
    /// a sustained overrun (which would let overrun detection wash out), and the
    /// margin in [`waited_extra_ns`] absorbs a single unusually fast copy.
    min_join_ns: Option<u64>,
    /// The rolling window of observations, in ns/byte: exact overrun durations and
    /// censoring thresholds. A rolling window rather than a fixed batch, so the
    /// estimate re-runs every [`ESTIMATE_EVERY`] steps (a short lag, doc §1.2
    /// weakness 1) over the last [`ESTIMATE_WINDOW`] observations.
    exact_tau: VecDeque<f64>,
    censored_tau: VecDeque<f64>,
    samples: u64,
    /// Steps the GPU arm overran the CPU multiply — the only ones that yield an
    /// exact arm duration, as opposed to a censoring bound. Printed with the shape
    /// so an operator can see whether the estimate rests on direct observations.
    gpu_samples: u64,
    /// Cumulative exact samples ever seen, which set the anchor's decayed weight
    /// (see [`anchor_weight`]).
    exact_total: u64,
    last_target: Option<f64>,
    last_waited: bool,
    applied: u32,
}

impl ShapeState {
    fn new(share: f64) -> Self {
        ShapeState {
            share,
            tau_cpu: None,
            tau_gpu: None,
            tau_gpu_prior: None,
            min_join_ns: None,
            exact_tau: VecDeque::new(),
            censored_tau: VecDeque::new(),
            samples: 0,
            gpu_samples: 0,
            exact_total: 0,
            last_target: None,
            last_waited: false,
            applied: 0,
        }
    }
}

/// Keyed by `(dtype, k)`: the CPU kernels differ per format and their throughput
/// per `k`.
struct SplitScheduler {
    shapes: BTreeMap<(DType, usize), ShapeState>,
    /// Shapes already seeded in this process, so the two-engine probe (a full
    /// pass on each engine) runs once per shape, not once per layer.
    probed: BTreeSet<(DType, usize)>,
    /// A calibration installed by [`seed`], which replaces the seeding probe for
    /// every shape that has no state yet.
    fallback: Option<SplitCalibration>,
}

impl Default for SplitScheduler {
    fn default() -> Self {
        SplitScheduler {
            shapes: BTreeMap::new(),
            probed: BTreeSet::new(),
            fallback: None,
        }
    }
}

static SCHEDULER: LazyLock<Mutex<SplitScheduler>> =
    LazyLock::new(|| Mutex::new(SplitScheduler::default()));

/// One shape's read-only scheduler state, for diagnostics and the bench.
#[derive(Clone, Copy, Debug)]
pub struct ShapeSnapshot {
    pub dtype: DType,
    pub k: usize,
    pub share: f64,
    /// Each arm's rate over its own bytes (`1/tau`); the CPU value is the
    /// multiply alone, the GPU value the censored estimate.
    pub cpu_bytes_per_s: Option<f64>,
    pub gpu_bytes_per_s: Option<f64>,
    pub samples: u64,
    /// Steps the GPU arm overran the CPU multiply — the only exact arm-duration
    /// observations behind `gpu_bytes_per_s`; the rest are censoring bounds.
    pub gpu_samples: u64,
    /// Whether the estimate has a probe anchor. `false` means the automatic probe
    /// did not run or failed — the operator-visible signal that the GPU level is
    /// resting on the online escape rather than a direct measurement.
    pub anchored: bool,
    /// The smallest blocking-H2D time seen for this shape: the copy floor the
    /// controller reads every join against, so an overrun needs no host constant.
    pub min_join_ns: Option<u64>,
    pub last_waited: bool,
    pub last_target: Option<f64>,
    pub applied: u32,
}

/// Every shape the scheduler has state for. Read-only; no device, no locks held
/// on return.
pub fn scheduler_snapshot() -> Vec<ShapeSnapshot> {
    let Ok(scheduler) = SCHEDULER.lock() else {
        return Vec::new();
    };
    scheduler.shapes.iter().map(shape_of).collect()
}

/// Install a calibration measured elsewhere (by [`probe_synthetic`], the
/// `hipfire offload-bench` path, or a test) as this process's starting point: a
/// shape with no state yet starts from this share and rate, and no seeding probe
/// runs for it. The online controller still refines it.
pub fn seed(calibration: &SplitCalibration) {
    let Ok(mut scheduler) = SCHEDULER.lock() else {
        return;
    };
    scheduler.fallback = Some(*calibration);
}

/// The share in force for `dtype`'s `k`, if the scheduler has state for it.
///
/// The split trace reads this so the printed share is the one actually used,
/// pinned or scheduled.
pub(crate) fn shape_state(dtype: DType, k: usize) -> Option<ShapeSnapshot> {
    let scheduler = SCHEDULER.lock().ok()?;
    let state = scheduler.shapes.get(&(dtype, k))?;
    Some(shape_of((&(dtype, k), state)))
}

fn shape_of(((dtype, k), state): (&(DType, usize), &ShapeState)) -> ShapeSnapshot {
    let rate = |tau: Option<f64>| tau.filter(|t| *t > 0.0).map(|t| 1e9 / t);
    ShapeSnapshot {
        dtype: *dtype,
        k: *k,
        share: state.share,
        cpu_bytes_per_s: rate(state.tau_cpu),
        gpu_bytes_per_s: rate(state.tau_gpu),
        samples: state.samples,
        gpu_samples: state.gpu_samples,
        anchored: state.tau_gpu_prior.is_some(),
        min_join_ns: state.min_join_ns,
        last_waited: state.last_waited,
        last_target: state.last_target,
        applied: state.applied,
    }
}

/// The scheduler's fallback share, when one was installed.
fn fallback_share() -> Option<f64> {
    SCHEDULER.lock().ok()?.fallback.map(|c| c.share)
}

/// The fallback calibration's GPU time per byte, when one was installed.
fn fallback_tau() -> Option<f64> {
    tau_from_rate(SCHEDULER.lock().ok()?.fallback.map(|c| c.gpu_bytes_per_s)?)
}

/// Whether this shape has already been probed (or seeded) in this process.
fn is_seeded(key: (DType, usize)) -> bool {
    SCHEDULER
        .lock()
        .map(|scheduler| scheduler.probed.contains(&key))
        .unwrap_or(true)
}

/// Insert a shape's state from a starting share and the probe's measured GPU time
/// per byte (the estimator's identifiability anchor), once. `probed` is marked so
/// a later step of the same shape neither probes nor re-seeds it.
fn seed_shape(key: (DType, usize), share: f64, tau_gpu_prior: Option<f64>) {
    let Ok(mut scheduler) = SCHEDULER.lock() else {
        return;
    };
    scheduler.probed.insert(key);
    scheduler.shapes.entry(key).or_insert_with(|| {
        let mut state = ShapeState::new(clamp_share(share));
        state.tau_gpu_prior = tau_gpu_prior;
        // The probe's measurement *is* an estimate; the censored MLE refines it.
        state.tau_gpu = tau_gpu_prior;
        state
    });
}

/// Record a pinned share so the trace and the snapshot can report it.
fn note_pinned(key: (DType, usize), share: f64) {
    let Ok(mut scheduler) = SCHEDULER.lock() else {
        return;
    };
    scheduler
        .shapes
        .entry(key)
        .or_insert_with(|| ShapeState::new(share));
}

/// The share to use for a shape whose state should already exist.
fn scheduled_share(key: (DType, usize)) -> f64 {
    SCHEDULER
        .lock()
        .ok()
        .and_then(|scheduler| {
            scheduler
                .shapes
                .get(&key)
                .map(|state| state.share)
                .or_else(|| scheduler.fallback.map(|c| c.share))
        })
        .unwrap_or(DEFAULT_GPU_SHARE)
}

/// Hysteresis on the join, in nanoseconds: an excess over the copy floor smaller
/// than this is host-timer noise around the copy, not a GPU overrun. A
/// timer-resolution constant, not a host performance one.
const JOIN_MARGIN_NS: u64 = 10_000;

/// The hysteresis margin for a given copy floor: the timer margin, or a quarter
/// of the floor for a host whose copy is itself slow (the copy's own jitter
/// scales with it).
fn copy_margin_ns(floor_ns: Option<u64>) -> u64 {
    JOIN_MARGIN_NS.max(floor_ns.unwrap_or(0) / 4)
}

/// The excess of `join_ns` over the copy floor that can be a GPU overrun: the
/// join past the floor and its hysteresis margin. Zero when no floor exists yet.
fn waited_extra_ns(join_ns: u64, floor_ns: Option<u64>) -> u64 {
    match floor_ns {
        Some(floor) => join_ns.saturating_sub(floor.saturating_add(copy_margin_ns(floor_ns))),
        None => 0,
    }
}

/// The controller's proposal after one step's measurements.
///
/// The setpoint is the DLT balance point `r_gpu / (r_gpu + r_cpu)` — the share at
/// which both arms finish together (Cheng & Robertazzi, *IEEE Trans. Computers*
/// 1994); the gain [`ALPHA`] only paces the walk to it. There is no integral term,
/// because the target *is* the estimate's own balance point and so no offset
/// exists to remove, and no blind ratchet, because the probe-anchored censored MLE
/// supplies a direction on its own — including below the balance point, where no
/// exact GPU sample exists. Until an estimate exists the share holds at the
/// probe's value.
fn next_share(cur: f64, r_cpu: Option<f64>, r_gpu: Option<f64>) -> f64 {
    match (r_cpu, r_gpu) {
        (Some(cpu), Some(gpu)) if cpu + gpu > 0.0 => {
            clamp_share(cur + ALPHA * (clamp_share(gpu / (gpu + cpu)) - cur))
        }
        _ => cur,
    }
}

/// One split step's measurements, in the censored-estimation frame.
///
/// `gemv_ns` is the CPU arm's multiply over `rows_cpu` rows — the arm's
/// weight-proportional work, never the blocking D2H that precedes it — and
/// `join_ns` is the blocking H2D, `copy + max(0, gpu_ns - gemv_ns)`.
///
/// The join is read against the shape's own copy floor ([`ShapeState::min_join_ns`]):
/// only its excess over that floor plus the timer margin can be the GPU arm
/// overrunning the CPU multiply, and then `gemv_ns + excess` *is* the arm's
/// duration. Below the balance point the GPU finishes first on every step, so its
/// duration is only *bounded* there; those steps are kept as censored
/// observations, not dropped, which is what keeps the estimate from being
/// conditioned on the GPU's slow tail (the shipped estimator's defect).
fn observe(
    key: (DType, usize),
    rows_cpu: usize,
    rows_gpu: usize,
    row_bytes: usize,
    gemv_ns: u64,
    join_ns: u64,
) {
    let Ok(mut scheduler) = SCHEDULER.lock() else {
        return;
    };
    let Some(state) = scheduler.shapes.get_mut(&key) else {
        return;
    };
    // CPU arm: the multiply alone, over the CPU arm's own rows.
    if gemv_ns > 0 && rows_cpu > 0 && row_bytes > 0 {
        let tau = gemv_ns as f64 / (rows_cpu as f64 * row_bytes as f64);
        state.tau_cpu = Some(match state.tau_cpu {
            Some(v) => v + EWMA_ALPHA * (tau - v),
            None => tau,
        });
    }
    // Copy floor: the smallest join ever seen (see the field's note).
    let floor = state.min_join_ns;
    state.min_join_ns = Some(floor.map_or(join_ns, |f| f.min(join_ns)));
    let margin = copy_margin_ns(floor);
    let excess = waited_extra_ns(join_ns, floor);
    let gpu_waited = excess > 0;
    let bytes_gpu = (rows_gpu as f64 * row_bytes as f64).max(1.0);
    if gpu_waited {
        state.exact_tau.push_back((gemv_ns + excess) as f64 / bytes_gpu);
        while state.exact_tau.len() > ESTIMATE_WINDOW {
            state.exact_tau.pop_front();
        }
        state.gpu_samples += 1;
        state.exact_total += 1;
    } else {
        // The GPU finished first: its duration is at most the CPU multiply plus
        // everything the join could be hiding (the margin).
        state
            .censored_tau
            .push_back((gemv_ns as f64 + margin as f64) / bytes_gpu);
        while state.censored_tau.len() > ESTIMATE_WINDOW {
            state.censored_tau.pop_front();
        }
    }
    state.samples += 1;
    state.last_waited = gpu_waited;
    if state.samples % ESTIMATE_EVERY == 0 {
        let mu = {
            let exact = state.exact_tau.make_contiguous();
            let censored = state.censored_tau.make_contiguous();
            estimate_tau_gpu(exact, censored, state.tau_gpu_prior, TAU_PRIOR_WEIGHT)
        };
        if let Some(mu) = mu {
            state.tau_gpu = Some(mu);
        }
        // Unanchored (no probe) and never an overrun: every observation was
        // censored, so the estimate is degenerate — it equals the censoring bound,
        // which makes the setpoint equal the current share. There is neither
        // identifiability nor an anchor, so the only escape is to push the share
        // up until the GPU arm overruns and yields an exact sample. Bounded: it
        // fires only in this state and stops the moment an overrun appears.
        if state.tau_gpu_prior.is_none() && state.exact_total == 0 {
            state.share = clamp_share(state.share + 0.02);
            state.last_target = Some(state.share);
            state.applied += 1;
            return;
        }
    }
    if state.samples % APPLY_EVERY != 0 {
        return;
    }
    let rate = |tau: Option<f64>| tau.filter(|t| *t > 0.0).map(|t| 1.0 / t);
    let next = next_share(state.share, rate(state.tau_cpu), rate(state.tau_gpu));
    state.share = next;
    state.last_target = Some(next);
    state.applied += 1;
}

fn bytes_per_s(bytes: f64, ns: u64) -> f64 {
    bytes * 1e9 / ns.max(1) as f64
}

// ── The seeding probe ──────────────────────────────────

/// What one two-engine measurement of a `(format, m, k)` shape found.
#[derive(Clone, Copy, Debug)]
pub struct SplitCalibration {
    /// The CPU arm's rate over the shape's weight bytes.
    pub cpu_bytes_per_s: f64,
    /// The GPU arm's rate over the same bytes.
    pub gpu_bytes_per_s: f64,
    /// `clamp(gpu / (gpu + cpu), SHARE_MIN, SHARE_MAX)`: the starting share.
    pub share: f64,
}

fn hip_err(e: hip_bridge::HipError) -> DispatchError {
    DispatchError::Hip(e.to_string())
}

/// The activation a probe should read, and whether the CPU arm would rotate it.
fn probe_activation<'a>(step: &'a Step<'a>, w: &WeightRef) -> Option<(&'a GpuTensor, bool)> {
    match step {
        Step::Gemv { input, .. } | Step::GemvResidual { input, .. } => match input {
            GemvInput::Raw(t) => Some((*t, dtype_rotation_plan(w.dtype) == RotationPlan::FwhtG256)),
            GemvInput::Prerotated(t) => Some((*t, false)),
        },
        _ => None,
    }
}

/// `GemvInput` holds references, so a probe step borrows the *real* input rather
/// than owning a copy: a `Raw` probe therefore rotates exactly as the real arm
/// will.
fn clone_input<'a>(input: &GemvInput<'a>) -> GemvInput<'a> {
    match input {
        GemvInput::Raw(t) => GemvInput::Raw(t),
        GemvInput::Prerotated(t) => GemvInput::Prerotated(t),
    }
}

fn median(mut samples: Vec<f64>) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    samples[samples.len() / 2]
}

/// Time both engines on `probe`'s weight buffer, `reps` times each, and return
/// the median rates. `Ok(None)` when the format has no launchable GPU arm (the
/// scheduler then keeps [`DEFAULT_GPU_SHARE`] and the online controller converges
/// from there).
///
/// Both engines are timed *alone*, sequentially, so both rates are overestimates
/// and the derived share is only a starting point.
fn measure_both(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    probe: &Step,
    w: &WeightRef,
    q: hipfire_cpu::quant::CpuQuant,
    row_bytes: usize,
    x_host: &[f32],
    reps: usize,
) -> Result<Option<SplitCalibration>, DispatchError> {
    let reps = reps.max(1);
    let Some(bytes) = gpu.host_bytes(w.buf) else {
        return Ok(None);
    };
    let mut y = vec![0.0f32; w.m];
    let mut cpu_ns = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        hipfire_cpu::gemv::gemv(q, bytes, w.m, w.k, x_host, &mut y);
        cpu_ns.push(t.elapsed().as_nanos() as f64);
    }
    let mut gpu_ns = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        if launch_op_rows(gpu, ctx, probe, Some(0..w.m)).is_err() {
            return Ok(None);
        }
        // A real blocking sync, NOT `sync_with_deadline`: that one polls with
        // `SYNC_POLL_INTERVAL` (2 ms) of sleep granularity, which on a sub-
        // millisecond kernel *is* the measurement — it reports ~4 GB/s for an
        // 8 MiB weight whose true rate is ~32 GB/s. Deadline-bearing paths still
        // use it; a timed benchmark must not.
        gpu.hip.device_synchronize().map_err(hip_err)?;
        gpu_ns.push(t.elapsed().as_nanos() as f64);
    }
    let size = (w.m * row_bytes) as f64;
    let cpu_bytes_per_s = bytes_per_s(size, median(cpu_ns).max(1.0) as u64);
    let gpu_bytes_per_s = bytes_per_s(size, median(gpu_ns).max(1.0) as u64);
    let share = clamp_share(gpu_bytes_per_s / (gpu_bytes_per_s + cpu_bytes_per_s));
    Ok(Some(SplitCalibration {
        cpu_bytes_per_s,
        gpu_bytes_per_s,
        share,
    }))
}

/// The seeding probe (4.3): time both engines on **this step's own weight
/// buffer**, once per shape per process.
///
/// The GPU arm is measured through a probe `Step` built over a scratch output,
/// never through the real step: for the residual form a probe write into the real
/// accumulator would be double-counted by the following real arm.
pub fn probe_weight(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    step: &Step,
    w: &WeightRef,
    reps: usize,
) -> Result<Option<SplitCalibration>, DispatchError> {
    let Some(q) = cpu_exec::cpu_quant_for(w.dtype) else {
        return Ok(None);
    };
    let Some(row_bytes) = weight_row_bytes(w) else {
        return Ok(None);
    };
    if row_bytes == 0 || w.m == 0 || w.k == 0 {
        return Ok(None);
    }
    let Some((activation, rotate_input)) = probe_activation(step, w) else {
        return Ok(None);
    };
    // The activation as the CPU arm will see it, once: the real arm downloads (and
    // rotates) it once per step, so re-deriving it per rep would measure a
    // transform the step does not repeat.
    let (x_host, _) = cpu_exec::prepare_activation(gpu, w, activation, rotate_input)?;
    let scratch = gpu.alloc_tensor(&[w.m], DType::F32).map_err(hip_err)?;
    let acc_scratch = match step {
        Step::GemvResidual { .. } => match gpu.alloc_tensor(&[w.m], DType::F32) {
            Ok(acc) => Some(acc),
            Err(e) => {
                let _ = gpu.free_tensor(scratch);
                return Err(hip_err(e));
            }
        },
        _ => None,
    };
    let input = match step {
        Step::Gemv { input, .. } => input,
        Step::GemvResidual { input, .. } => input,
        _ => unreachable!("probe_activation admitted only the two GEMV shapes"),
    };
    let input = clone_input(input);
    let probe = match acc_scratch.as_ref() {
        Some(acc) => Step::GemvResidual {
            w,
            input,
            residual: acc,
            out: acc,
        },
        None => Step::Gemv {
            w,
            input,
            out: &scratch,
        },
    };
    let result = measure_both(gpu, ctx, &probe, w, q, row_bytes, &x_host, reps);
    let _ = gpu.free_tensor(scratch);
    if let Some(acc) = acc_scratch {
        let _ = gpu.free_tensor(acc);
    }
    result
}

/// The same two-engine measurement over a **synthetic** host-mapped weight, for
/// `hipfire offload-bench`: no model and no step list needed.
///
/// Allocates, uses and frees its own buffers (a host-mapped weight of `m` rows —
/// system RAM the GPU reads over the link — an f32 activation of `k`, an f32
/// output of `m`). The weight's *contents* are irrelevant: both engines decode
/// the same bytes and the measurement is memory traffic.
pub fn probe_synthetic(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    dtype: DType,
    m: usize,
    k: usize,
    reps: usize,
) -> Result<SplitCalibration, DispatchError> {
    let q = cpu_exec::cpu_quant_for(dtype).ok_or_else(|| {
        DispatchError::Hip(format!(
            "passback probe: no CPU decoder for {dtype:?}; nothing to measure"
        ))
    })?;
    let row_bytes = hipfire_cpu::gemv::row_bytes(q, k);
    if m == 0 || k == 0 || row_bytes == 0 {
        return Err(DispatchError::Hip(
            "passback probe: m, k and the format's row stride must all be non-zero".into(),
        ));
    }
    let weight = gpu
        .upload_raw_host(&vec![0u8; m * row_bytes], &[m, k])
        .map_err(hip_err)?;
    let x_act = match gpu.upload_f32(&vec![0.0f32; k], &[k]) {
        Ok(t) => t,
        Err(e) => {
            let _ = gpu.free_tensor(weight);
            return Err(hip_err(e));
        }
    };
    let scratch = match gpu.alloc_tensor(&[m], DType::F32) {
        Ok(t) => t,
        Err(e) => {
            let _ = gpu.free_tensor(x_act);
            let _ = gpu.free_tensor(weight);
            return Err(hip_err(e));
        }
    };
    let result = {
        let w_ref = WeightRef {
            buf: &weight,
            dtype,
            m,
            k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
        };
        // `Raw`, not `Prerotated`: the probe needs no producer to have warmed the
        // rotation scratch, and the extra rotate is `k` floats against MBs of
        // weight bytes. The real split of a `Raw` step does exactly this too.
        let probe = Step::Gemv {
            w: &w_ref,
            input: GemvInput::Raw(&x_act),
            out: &scratch,
        };
        let rotate = dtype_rotation_plan(dtype) == RotationPlan::FwhtG256;
        let (x_host, _) = cpu_exec::prepare_activation(gpu, &w_ref, &x_act, rotate)?;
        measure_both(gpu, ctx, &probe, &w_ref, q, row_bytes, &x_host, reps)?
    };
    let _ = gpu.free_tensor(scratch);
    let _ = gpu.free_tensor(x_act);
    let _ = gpu.free_tensor(weight);
    result.ok_or_else(|| {
        DispatchError::Hip(format!(
            "passback probe: {dtype:?} has no launchable GPU arm at m={m}, k={k}"
        ))
    })
}

// ── The seam ───────────────────────────────────────────

/// The weight of a row-splittable step, or `None` for any other kind.
fn step_weight<'a, 'b>(step: &'b Step<'a>) -> Option<&'a WeightRef<'a>> {
    match step {
        Step::Gemv { w, .. } | Step::GemvResidual { w, .. } => Some(*w),
        _ => None,
    }
}

/// The step's output rows that must be covered by the two arms, in order: the
/// output, plus the residual for the residual form.
fn output_extent(w: &WeightRef, step: &Step) -> bool {
    let out_ok = |t: &GpuTensor| t.dtype == DType::F32 && t.numel() >= w.m;
    match step {
        Step::Gemv { out, .. } => out_ok(out),
        Step::GemvResidual { residual, out, .. } => out_ok(residual) && out_ok(out),
        _ => false,
    }
}

/// Execute `step` as a pass-back split, or `Ok(false)` to leave it wholly to
/// `memory.offload_exec=cpu`.
///
/// Every refusal below is silent-with-a-fallback rather than an error: the mode
/// has a correct whole-CPU route for every step, so an unsplittable shape runs
/// exactly as `cpu` mode runs it (never as `pcie`).
pub fn run_with(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    step: &Step,
    opts: &PassbackOptions,
) -> Result<bool, DispatchError> {
    if !opts.enabled {
        return Ok(false);
    }
    // 1–2: the CPU must be able to take the step at all, and the step must be one
    // of the four row-splittable shapes.
    let Some(plan) = cpu_exec::plan_step(gpu, step) else {
        return Ok(false);
    };
    let Some(w) = step_weight(step) else {
        return Ok(false);
    };
    let residual = matches!(step, Step::GemvResidual { .. });
    if residual && KernelKey::for_gemv_residual(w.dtype).is_err() {
        // The GPU arm of a residual step is `dispatch_residual`, whose dtype set
        // is exactly `for_gemv_residual`'s. A `Raw` step outside it would take
        // `launch_op`'s multi-launch fallback (GEMV into a whole-tensor scratch,
        // then `residual += out`), and a `Prerotated` one would simply fail to
        // launch. Either way the step cannot be co-inferenced: it runs on the CPU
        // engine alone (the schedule's degenerate point), never `pcie` — this mode
        // is never allowed to be less robust than `memory.offload_exec=cpu`, which
        // handles every format the CPU decodes.
        return Ok(false);
    }
    // 3–6: host-mapped, unpadded, and consistent with the CPU decoder.
    if !gpu.host_located(w.buf) || w.row_stride != 0 || w.m == 0 {
        return Ok(false);
    }
    let Some(row_bytes) = weight_row_bytes(w) else {
        return Ok(false);
    };
    let Some(q) = cpu_exec::cpu_quant_for(w.dtype) else {
        return Ok(false);
    };
    // The correctness-critical invariant: it is what makes `&host_bytes[g *
    // row_bytes..]` the CPU arm's row `g`. A mismatch falls back rather than
    // reading the wrong rows.
    if row_bytes != hipfire_cpu::gemv::row_bytes(q, w.k) {
        return Ok(false);
    }
    if !hipfire_cpu::simd::row_dot_enabled(q, None) {
        return Ok(false);
    }
    // 7: both write targets must be F32 and cover every row.
    if !output_extent(w, step) {
        return Ok(false);
    }
    // 8: the share.
    let key = (w.dtype, w.k);
    let share = match opts.share {
        PassbackShare::Share(f) => {
            note_pinned(key, f);
            f
        }
        PassbackShare::Auto => {
            if !is_seeded(key) {
                match fallback_share() {
                    Some(f) => seed_shape(key, f, fallback_tau()),
                    None => {
                        // A failed or unavailable probe must not fail a step that
                        // plain `cpu` mode would have run: fall back to
                        // `DEFAULT_GPU_SHARE` with no prior and let the online
                        // censored estimator correct it from the arm timings.
                        match probe_weight(gpu, ctx, step, w, PROBE_REPS).ok().flatten() {
                            Some(c) => seed_shape(key, c.share, tau_from_rate(c.gpu_bytes_per_s)),
                            None => seed_shape(key, DEFAULT_GPU_SHARE, None),
                        }
                    }
                }
            }
            scheduled_share(key)
        }
    };
    // 9: is this shape splittable at this share?
    let Some(g) = plan_rows(w.m, row_bytes, share) else {
        return Ok(false);
    };

    // The ordering *is* the mechanism: prepare (blocking D2H) before the GPU arm,
    // GPU arm async, CPU multiply, then the blocking H2D of the CPU's rows as the
    // join.
    let step_start = Instant::now();
    let arm = cpu_exec::cpu_arm_prepare(gpu, &plan, g..w.m)?;
    launch_op_rows(gpu, ctx, step, Some(0..g))?;
    let (gemv_ns, join_ns) = cpu_exec::cpu_arm_finish(gpu, &arm)?;
    observe(key, w.m - g, g, row_bytes, gemv_ns, join_ns);
    cpu_exec::finish_step(
        arm.q,
        w,
        arm.rotate_input,
        residual,
        HostExec::Split { gpu_rows: g },
        step_start,
        StepTiming {
            d2h_ns: arm.d2h_ns,
            gemv_ns,
            h2d_ns: join_ns,
        },
    );
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_rows_balances_and_refuses() {
        // The plan's pinned example: 0.375 of 12288 rows at 2720 B/row.
        assert_eq!(plan_rows(12288, 2720, 0.375), Some(4608));
        // A share outside (0, 1) never splits — 0 is the cpu byte-identity twin.
        assert_eq!(plan_rows(12288, 2720, 0.0), None);
        assert_eq!(plan_rows(12288, 2720, 1.0), None);
        assert_eq!(plan_rows(12288, 2720, -0.1), None);
        assert_eq!(plan_rows(12288, 2720, f64::NAN), None);
        // Too few rows to align on.
        assert_eq!(plan_rows(8, 1 << 20, 0.5), None);
        // A weight below the minimum split size.
        assert_eq!(plan_rows(1024, 1024, 0.5), None);
        assert!(1024 * 1024 < MIN_SPLIT_BYTES);
    }

    #[test]
    fn plan_rows_is_aligned_and_inside_its_bounds() {
        for m in [16usize, 17, 63, 1000, 12288, 40961] {
            for share in [0.01f64, 0.05, 0.1, 0.375, 0.5, 0.9, 0.99] {
                let Some(g) = plan_rows(m, 2720, share) else {
                    continue;
                };
                assert_eq!(g % ROW_ALIGN, 0, "m={m} share={share} g={g}");
                assert!(g >= ROW_ALIGN && g <= m - ROW_ALIGN, "m={m} share={share} g={g}");
            }
        }
    }

    #[test]
    fn next_share_moves_toward_the_dlt_balance_point() {
        // CPU at 52 GB/s, GPU at 27.3 GB/s -> balance 0.344; from 0.375 it moves
        // down by ALPHA times the gap.
        let moved = next_share(0.375, Some(52e9), Some(27.3e9));
        let target = 27.3 / (27.3 + 52.0);
        assert!(moved < 0.375 && moved > target, "moved={moved}");
        assert!((moved - (0.375 + ALPHA * (target - 0.375))).abs() < 1e-12);
        // No GPU estimate yet: the share holds at the probe's value — there is no
        // blind ratchet any more.
        assert_eq!(next_share(0.3, Some(52e9), None), 0.3);
        assert_eq!(next_share(0.3, None, Some(52e9)), 0.3);
        // A GPU rate below the CPU's walks the share down to the floor and clamps.
        let down = next_share(SHARE_MIN, Some(1e12), Some(1.0));
        assert_eq!(down, SHARE_MIN);
        // A GPU rate above the CPU's walks it up, clamped at the ceiling.
        let up = next_share(SHARE_MIN, Some(1.0), Some(1e12));
        assert!(up > SHARE_MIN && up <= SHARE_MAX, "up={up}");
        assert!((up - (SHARE_MIN + ALPHA * (SHARE_MAX - SHARE_MIN))).abs() < 1e-12);
    }

    #[test]
    fn estimate_tau_gpu_uses_the_anchor_and_both_observation_kinds() {
        // Nothing to go on: no estimate.
        assert!(estimate_tau_gpu(&[], &[], None, TAU_PRIOR_WEIGHT).is_none());

        // Exact samples only, no anchor: the estimate is their mean.
        let exact = estimate_tau_gpu(&[2.0, 4.0, 4.0, 2.0], &[], None, TAU_PRIOR_WEIGHT).unwrap();
        assert!((exact - 3.0).abs() < 1e-6, "exact={exact}");

        // Exact samples with an anchor below them: pulled up, but only part way —
        // the anchor keeps its weight.
        let anchored =
            estimate_tau_gpu(&[2.0, 4.0, 4.0, 2.0], &[], Some(1.0), TAU_PRIOR_WEIGHT).unwrap();
        assert!(anchored > 1.0 && anchored < 3.0, "anchored={anchored}");

        // Censored observations *below* the anchor bind: pulled down, part way to
        // the bound and not past it.
        let pulled =
            estimate_tau_gpu(&[], &[0.4, 0.4, 0.4, 0.4], Some(1.0), TAU_PRIOR_WEIGHT).unwrap();
        assert!(pulled < 1.0 && pulled > 0.4, "pulled={pulled}");

        // Censored observations *above* the anchor carry nothing the anchor lacks,
        // so the estimate stays at the anchor (this is the identifiability limit:
        // a GPU that always finishes first is only known to be at least this fast).
        let inert = estimate_tau_gpu(&[], &[10.0; 8], Some(1.0), TAU_PRIOR_WEIGHT).unwrap();
        assert!((inert - 1.0).abs() < 0.1, "inert={inert}");
    }

    #[test]
    fn an_unanchored_shape_escapes_upward_rather_than_sticking_at_default() {
        let key = (DType::Q8_0, 7777);
        // No anchor: the automatic probe did not run or failed.
        seed_shape(key, 0.375, None);
        // A full window of floor joins is entirely censored. Without an anchor the
        // estimate is degenerate, so the share must climb instead of sticking.
        for _ in 0..ESTIMATE_EVERY {
            observe(key, 3072, 1024, 2176, 500_000, 40_000);
        }
        let s = shape_state(key.0, key.1).expect("seeded");
        assert_eq!(s.gpu_samples, 0, "floor joins are not overruns");
        assert!(!s.anchored, "no probe ran, so there is no anchor");
        assert!(
            s.share > 0.375,
            "unanchored and censored must climb, got {}",
            s.share
        );
        // And it is bounded — it never exceeds the ceiling.
        for _ in 0..(ESTIMATE_EVERY * 20) {
            observe(key, 3072, 1024, 2176, 500_000, 40_000);
        }
        let s = shape_state(key.0, key.1).expect("seeded");
        assert!(s.share <= SHARE_MAX, "share={}", s.share);
    }

    #[test]
    fn estimate_tau_gpu_is_insensitive_to_the_iteration_cap() {
        // The EM contracts quickly; the shipped cap must already be converged. If
        // this ever fails, the cap is load-bearing and the estimator is not at a
        // fixed point — the thing to fix, not the cap.
        let exact = [1.9, 2.05, 1.95, 2.1, 1.98];
        let censored = [8.0, 9.0, 7.5];
        let capped = estimate_tau_gpu_iters(
            &exact,
            &censored,
            Some(1.0),
            TAU_PRIOR_WEIGHT,
            ESTIMATE_ITERS,
        )
        .unwrap();
        let long =
            estimate_tau_gpu_iters(&exact, &censored, Some(1.0), TAU_PRIOR_WEIGHT, 30).unwrap();
        assert!(
            (capped - long).abs() <= 1e-6 * long.abs(),
            "cap {ESTIMATE_ITERS} vs 30 disagree: {capped} vs {long}"
        );
    }

    #[test]
    fn observe_separates_censored_from_exact_and_moves_toward_the_balance() {
        // A distinct key so the process-global scheduler cannot interfere.
        let key = (DType::Q8_0, 1234);
        // Anchor the GPU at 25 GB/s (0.04 ns/byte). The CPU measures far below
        // that, so the balance point is above the seed and the share must rise.
        seed_shape(key, 0.3, Some(1e9 / 25e9));

        // A join at the copy floor is censored: a bound on the arm, not an exact
        // sample, and the anchored (fast) GPU pulls the share up.
        for _ in 0..4 {
            observe(key, 3072, 1024, 2176, 500_000, 40_000);
        }
        let censored = shape_state(key.0, key.1).expect("seeded");
        assert_eq!(censored.gpu_samples, 0, "a floor join is not an overrun");
        assert!(censored.share > 0.3, "share must rise, got {}", censored.share);
        // CPU arm: 3072 rows * 2176 B over 500 µs, the multiply alone.
        let r_cpu = censored.cpu_bytes_per_s.expect("cpu sampled");
        assert!((r_cpu - 13.4e9).abs() / 13.4e9 < 0.05, "r_cpu={r_cpu:e}");

        // A join well past the floor overruns: exact samples, and after a full
        // window the slow GPU arm pulls the estimate's rate below the anchor.
        for _ in 0..ESTIMATE_EVERY {
            observe(key, 3072, 1024, 2176, 500_000, 900_000);
        }
        let overran = shape_state(key.0, key.1).expect("seeded");
        assert!(overran.gpu_samples > 0, "an overrun must be an exact sample");
        let r_gpu = overran.gpu_bytes_per_s.expect("estimated");
        assert!(r_gpu < 25e9, "slow overruns must pull the rate down: {r_gpu:e}");
    }

    #[test]
    fn share_always_lands_inside_the_ceiling_and_floor() {
        for start in [0.0f64, 0.05, 0.375, 0.5, 0.9] {
            for (cpu, gpu) in [(1e9, 1e12), (1e12, 1e9), (1.0, 1.0)] {
                let s = next_share(start, Some(cpu), Some(gpu));
                assert!((SHARE_MIN..=SHARE_MAX).contains(&s), "s={s}");
            }
        }
    }
}
