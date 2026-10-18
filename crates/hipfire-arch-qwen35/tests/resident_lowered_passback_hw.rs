// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//!
//! Hardware correctness oracle for **resident-layer lowered routing of a
//! partially spilled (pass-back) Qwen3.5 dense decode**: the production mixed
//! forward (resident layers on the lowered super-op executor, spilled layers on
//! the hand arms with the pass-back CPU splice) versus the all-hand reference.
//!
//! Both arms run the *same* public single-token decode entry
//! ([`forward_scratch`], teacher-forced tokens), on the *same* weights,
//! the *same* placement, the *same* pass-back share and the *same* zeroed
//! initial state, in **separate fresh processes** (the lowered toggle and the
//! pass-back share are process-cached `OnceLock`s, so one process can hold only
//! one arm). The comparison is byte-exact over every step's logits and the full
//! valid state, with numeric diagnostics on any deviation — never a tolerance.
//!
//! What is compared (all after a completed token, on the same stream):
//!   * `logits` of **every** step (all `vocab` f32), so a stale/queued CPU-return
//!     tail shows up at the step it corrupts, not only at the end;
//!   * the **full valid state at several post-token checkpoints** — a quarter, a
//!     half and the end of the run (3, 6 and 12 for the default 12 steps), each
//!     captured after that token completed:
//!       - FullAttention layers: the **valid** prefix of `KV` in the live
//!         encoding — Q8_0 key + value (the block scale inside each 34-byte
//!         block) by default, or the native E4M3FN token rows
//!         `[Hkv*D codes][Hkv f16 scales]` under `HIPFIRE_RESIDENT_ORACLE_KV_MODE=fp8`
//!         — `committed` tokens only, never the uninitialized capacity;
//!       - LinearAttention layers: the dense GDN state (`s_matrices` i8 codes,
//!         `s_scales` f32, `conv_states` f32) and `s_ef_residual` (f16) **when
//!         the state carries an EF plane**; its absence is recorded and must
//!         agree on both arms (no fake echo of a tensor that does not exist);
//!   * the **initial** state, dumped before the first token and asserted to be
//!     exactly zero on both arms — the identical starting point is a checked
//!     precondition, not an assumption.
//!
//! Route proof (no production counter added): the mixed arm must report
//! `owns_host_cpu_weights`, pass-back enabled and a live split with sampled and
//! unsampled steps (`split_counters` / `scheduler_snapshot`, the existing
//! scheduler stats), and its per-layer residency must equal the dense spill
//! prefix exactly. Beyond that, the per-step HIP launch tallies
//! (`hip_bridge::launch_counters`, thread-local, reset per step) and all exec
//! counters are dumped as **diagnostic metadata** — an incidental implementation
//! count is not a numerical invariant, so nothing asserts on it; the runner
//! checks the path out-of-band before accepting any performance claim.
//!
//! Determinism: the share is **pinned** (`HIPFIRE_OFFLOAD_PASSBACK_SHARE`), so
//! each step's row split is a pure function of the weight shape and is identical
//! in both arms — the reference comparison is only meaningful because of that.
//! The scheduler's online controller still measures every sampled step, but a
//! pinned share is read straight from the mode (`PassbackShare::Share`), never
//! from the drifting per-shape state, so host timing cannot move the split. An
//! auto share is NOT a valid oracle input for the same reason.
//!
//! Async coverage and its limits: a sampled split step (about one in four, per
//! `(dtype, k)` shape) completes its return H2D inline, while roughly three in
//! four queue it — that queued (unsampled) return path is what a stream-ordering
//! regression would corrupt. Per-step logits and the state checkpoints are
//! therefore compared at every token and the run asserts
//! `split_steps > sampled_steps`, which proves at least one queued return was
//! taken without depending on the cadence's exact draw. Both the default stream
//! and a non-blocking active stream are exercised; the consumer is always the next
//! step's kernels on that same stream, and the oracle never inserts a
//! between-layer synchronization of its own (one drain per completed token, and
//! the checkpoint read follows that drain).
//!
//! Cases (default `9b`): dense prefix spill cut at 16 (first resident layer is
//! DeltaNet) and at 15 (first resident layer is FullAttention), at pinned
//! pass-back shares 0.375 and 0.5, with the default stream and with a
//! non-blocking active stream. An opt-in `HIPFIRE_RESIDENT_ORACLE_TWIN=1` case
//! adds the share-0 twin: `memory.offload_exec=cpu` (lowered off) versus
//! `passback` + `share=0` **with the lowered toggle left on**, both of which must
//! take the same all-hand fallback and be byte-identical.
//!
//! Fixtures: `HIPFIRE_RESIDENT_ORACLE_FIXTURE=9b` (default) keeps the 9B matrix
//! above; `=27b` selects the **Qwen3.8-27B Phase-3** matrix, which spills **8 of
//! 64** layers: cut 8 (first resident DeltaNet) and the cut-7 boundary (first
//! resident FullAttention), at the same shares/streams, 12 teacher-forced steps.
//! Both fixtures use the `L,L,L,F` period; the worker checks the actual model's
//! layer type at each selected boundary rather than inferring it from a filename.
//!
//! KV encoding: `HIPFIRE_RESIDENT_ORACLE_KV_MODE=q8` (default) keeps the
//! historical Q8_0 KV cache and its dump. `=fp8` builds the **same** filtered,
//! absolute-layer-owned VMM cache in native E4M3FN
//! (`KvCache::new_gpu_fp8_vmm_capped_filtered`): one head-major token row
//! `[Hkv*D codes][Hkv f16 scales]`, `Hkv*(D+2)` bytes/row, **D=256 only** — the
//! 27B production KV (KV auto → FP8/VMM). The dumped KV prefix is then the raw
//! FP8 token rows, finite-checked per side (E4M3FN codes and inline f16 scales
//! decoded with their own formats — never a scale byte read as a code, never the
//! Q8_0 34-byte block math). The GDN state is untouched in both modes: it stays
//! `StateQuant::Q8` (its own, independent quantization).
//!
//! Model + KV mode are selected by environment (the model default follows the
//! fixture):
//!
//! ```text
//! # 9B / Q8 (default)
//! cargo test -p hipfire-arch-qwen35 --release --test resident_lowered_passback_hw \
//!     -- --ignored --nocapture
//!
//! # Qwen3.8-27B Phase-3: 8 spilled / 56 resident, native FP8 (=E4M3FN) VMM KV
//! HIPFIRE_RESIDENT_ORACLE_FIXTURE=27b \
//! HIPFIRE_RESIDENT_ORACLE_MODEL="$HOME/.hipfire/models/qwen3.8-27b.mq4" \
//! HIPFIRE_RESIDENT_ORACLE_KV_MODE=fp8 \
//! cargo test -p hipfire-arch-qwen35 --release --test resident_lowered_passback_hw \
//!     -- --ignored --nocapture
//! ```
//!
//! Both tests are ignored by default, so CI without a GPU stays green; an
//! explicitly requested run with a missing model FAILS (it never silently
//! skips). Dumps are retained under `HIPFIRE_RESIDENT_ORACLE_OUT` (or a fresh
//! temp dir) for localization.

use hipfire_arch_qwen35::qwen35::{
    config_from_hfq, forward_scratch, load_weights, DeltaNetState, HfqSource, LayerType,
    LayerWeights, Layout, Qwen35Config, Qwen35Scratch, StateQuant,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::offload::Placement;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{Gpu, GpuTensor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

// ── Child-process contract (env only; no generic CLI arguments) ──────────
const ARM_ENV: &str = "HIPFIRE_RESIDENT_ORACLE_ARM";
const CUT_ENV: &str = "HIPFIRE_RESIDENT_ORACLE_CUT";
const SHARE_ENV: &str = "HIPFIRE_RESIDENT_ORACLE_SHARE";
const OUT_ENV: &str = "HIPFIRE_RESIDENT_ORACLE_OUT";
const MODEL_ENV: &str = "HIPFIRE_RESIDENT_ORACLE_MODEL";
const STEPS_ENV: &str = "HIPFIRE_RESIDENT_ORACLE_STEPS";
const NONBLOCK_ENV: &str = "HIPFIRE_RESIDENT_ORACLE_NONBLOCKING";
/// Worker + orchestrator: KV encoding for the FullAttention cache. `q8`
/// (default) keeps the historical Q8_0 oracle; `fp8` builds the same filtered
/// VMM cache in native E4M3FN for the 27B fixture. Unknown values are refused.
const KV_MODE_ENV: &str = "HIPFIRE_RESIDENT_ORACLE_KV_MODE";
/// Orchestrator-only: explicit `9b` (default) or `27b` placement matrix.
const FIXTURE_ENV: &str = "HIPFIRE_RESIDENT_ORACLE_FIXTURE";
/// Orchestrator-only: comma list of case indices (default: every case).
const CASES_ENV: &str = "HIPFIRE_RESIDENT_ORACLE_CASES";
/// Orchestrator-only: add the share-0 twin case (`cpu` vs `passback` share 0).
const TWIN_ENV: &str = "HIPFIRE_RESIDENT_ORACLE_TWIN";

const WORKER_TEST: &str = "resident_lowered_passback_worker";
const DEFAULT_MODEL_9B: &str = "qwen3.5-9b.mq4";
/// The Qwen3.8-27B Phase-3 fixture (8 spilled / 56 resident, KV auto → FP8/VMM).
const DEFAULT_MODEL_27B: &str = "qwen3.8-27b.mq4";
/// Teacher-forced token seed; cycled/truncated to exactly `STEPS` tokens.
const PROMPT: &str = "The quick brown fox jumps over the lazy dog, twice over.";
const DEFAULT_STEPS: usize = 12;

// ── Case matrix ─────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// All-hand (lowered off) vs mixed (lowered on), same pass-back config.
    Mixed,
    /// `offload_exec=cpu` vs `offload_exec=passback, share=0`, both all-hand.
    Twin,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    cut: usize,
    share: f64,
    nonblocking: bool,
    kind: Kind,
}

/// First-resident layer kind the fixture must show for a spill cut. Both the 9B
/// and the 27B stacks use the same `L,L,L,F` period, so a cut pinned here means
/// the same layer kind on either fixture; the worker still checks it against the
/// model's actual `layer_types`, so an unexpected cut is refused rather than
/// silently accepted.
fn expected_first_resident(cut: usize) -> LayerType {
    match cut {
        // 9B: first resident DeltaNet at 16, FullAttention at 15.
        16 => LayerType::LinearAttention,
        15 => LayerType::FullAttention,
        // 27B (64 layers): index 8 is DeltaNet, index 7 is FullAttention.
        8 => LayerType::LinearAttention,
        7 => LayerType::FullAttention,
        other => panic!("no expected first-resident layer pinned for cut {other}"),
    }
}

/// KV encoding for the FullAttention cache.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum KvSel {
    /// Historical Q8_0 VMM cache and dump (default).
    Q8,
    /// Native E4M3FN VMM cache and dump (27B fixture; D=256 only).
    Fp8,
}

impl KvSel {
    /// Resolve [`KV_MODE_ENV`]: unset → `q8`; `q8`/`fp8` accepted; anything else
    /// (including an empty value) is refused.
    fn from_env() -> KvSel {
        match std::env::var(KV_MODE_ENV).ok().as_deref() {
            None | Some("q8") => KvSel::Q8,
            Some("fp8") => KvSel::Fp8,
            Some(other) => {
                panic!("{KV_MODE_ENV}={other:?} is not a known KV mode (want q8|fp8)")
            }
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            KvSel::Q8 => "q8",
            KvSel::Fp8 => "fp8",
        }
    }
}

/// Select the placement matrix explicitly; model filenames are not identities.
fn fixture_27b() -> bool {
    match std::env::var(FIXTURE_ENV).ok().as_deref() {
        None | Some("9b") => false,
        Some("27b") => true,
        Some(other) => panic!("{FIXTURE_ENV}={other:?} is not a known fixture (want 9b|27b)"),
    }
}

/// Default model path for the selected placement fixture.
fn default_model() -> String {
    let home = std::env::var("HOME").expect("HOME");
    let name = if fixture_27b() {
        DEFAULT_MODEL_27B
    } else {
        DEFAULT_MODEL_9B
    };
    format!("{home}/.hipfire/models/{name}")
}

/// The pinned mixed matrix for a fixture: `cut_la` spills so the first resident
/// layer is DeltaNet, `cut_fa` so it is FullAttention; the remaining two cases
/// reuse `cut_la` for the share-0.5 and non-blocking variants.
fn mixed_cases(cut_la: usize, cut_fa: usize) -> Vec<Case> {
    vec![
        Case {
            cut: cut_la,
            share: 0.375,
            nonblocking: false,
            kind: Kind::Mixed,
        },
        Case {
            cut: cut_fa,
            share: 0.375,
            nonblocking: false,
            kind: Kind::Mixed,
        },
        Case {
            cut: cut_la,
            share: 0.5,
            nonblocking: false,
            kind: Kind::Mixed,
        },
        Case {
            cut: cut_la,
            share: 0.375,
            nonblocking: true,
            kind: Kind::Mixed,
        },
    ]
}

fn default_cases() -> Vec<Case> {
    // 9B: DeltaNet-first at 16, FullAttention-first at 15. 27B (64 layers,
    // 8 spilled / 56 resident): DeltaNet-first at 8, FullAttention-first at 7.
    let (cut_la, cut_fa) = if fixture_27b() { (8, 7) } else { (16, 15) };
    let mut cases = mixed_cases(cut_la, cut_fa);
    if std::env::var(TWIN_ENV).ok().as_deref() == Some("1") {
        cases.push(Case {
            cut: cut_la,
            share: 0.0,
            nonblocking: false,
            kind: Kind::Twin,
        });
    }
    if let Ok(sel) = std::env::var(CASES_ENV) {
        let want: Vec<usize> = sel
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<usize>().expect("case index"))
            .collect();
        let mut kept = Vec::new();
        for (i, c) in cases.iter().enumerate() {
            if want.contains(&i) {
                kept.push(*c);
            }
        }
        assert!(
            !kept.is_empty(),
            "{CASES_ENV}={sel} selected no case out of {}",
            cases.len()
        );
        cases = kept;
    }
    cases
}

#[derive(Clone, Copy, Debug)]
struct ArmSpec {
    label: &'static str,
    /// `HIPFIRE_OFFLOAD_EXEC` for the child.
    exec: &'static str,
    /// Whether the child gets `HIPFIRE_OFFLOAD_PASSBACK_SHARE`.
    set_share: bool,
    lowered: bool,
}

fn arms_for(kind: Kind) -> [ArmSpec; 2] {
    match kind {
        Kind::Mixed => [
            ArmSpec {
                label: "hand",
                exec: "passback",
                set_share: true,
                lowered: false,
            },
            ArmSpec {
                label: "mixed",
                exec: "passback",
                set_share: true,
                lowered: true,
            },
        ],
        // The share-0 twin: the same spill on the CPU engine alone
        // (`memory.offload_exec=cpu`, lowered off) versus `passback` with a pinned
        // share of 0 and the lowered toggle left on. The second arm is the one that
        // exercises the share-0 gate: it must fall back to the same all-hand route
        // as `cpu` mode, so the two arms are byte-identical and neither splits.
        Kind::Twin => [
            ArmSpec {
                label: "cpu",
                exec: "cpu",
                set_share: false,
                lowered: false,
            },
            ArmSpec {
                label: "passback0",
                exec: "passback",
                set_share: true,
                lowered: true,
            },
        ],
    }
}

// ── Manifest ────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct Entry {
    name: String,
    kind: String,
    file: String,
    bytes: usize,
    sha256: String,
    nonfinite: usize,
    min: f32,
    max: f32,
}

#[derive(Serialize, Deserialize)]
struct ShapeRec {
    dtype: String,
    k: usize,
    share: f64,
    samples: u64,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    arm: String,
    model: String,
    arch: String,
    cut: usize,
    share: f64,
    steps: usize,
    nonblocking: bool,
    /// KV encoding the FullAttention cache was built with (`q8` | `fp8`).
    kv_mode: String,
    /// Bytes of one KV token row **per side** in that encoding (`q8_0`:
    /// `Hkv*(D/32)*34`; `fp8`: `Hkv*(D+2)`). The dumped KV prefix is
    /// `committed * kv_row_bytes`.
    kv_row_bytes: usize,
    tokens: Vec<u32>,
    vocab: usize,
    dim: usize,
    n_layers: usize,
    layer_types: Vec<String>,
    passback_enabled: bool,
    cpu_exec_enabled: bool,
    owns_host_cpu_weights: bool,
    host_layers: usize,
    resident_first_layer: usize,
    resident_first_kind: String,
    expected_resident_first_kind: String,
    residency_mismatches: usize,
    cpu_steps: usize,
    host_mapped_gpu_steps: usize,
    split_steps: usize,
    split_gpu_rows: u64,
    scheduler: Vec<ShapeRec>,
    sampled_steps: u64,
    state_checkpoints: Vec<usize>,
    launches_per_step: Vec<u64>,
    kernels_per_step: Vec<u64>,
    launches_total: u64,
    kernels_total: u64,
    entries: Vec<Entry>,
}

// ── Dump/decode helpers (mirrors examples/mq4_prefill_state_oracle.rs) ──

fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = (bits >> 10) & 31;
    let m = bits & 1023;
    match e {
        0 => sign * (m as f32) * 2f32.powi(-24),
        31 if m == 0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + m as f32 / 1024.0) * 2f32.powi(e as i32 - 15),
    }
}

/// OCP E4M3FN decode (4-bit exponent, bias 7; 3-bit mantissa, max finite 448).
/// The only NaN bit patterns are `0x7f`/`0xff` (exp all-ones, mantissa all-ones),
/// which a saturating encoder never emits — a NaN here is a real corruption.
fn e4m3fn(code: u8) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = (code >> 3) & 0x0f;
    let m = code & 0x07;
    match e {
        0 => sign * (m as f32) * 2f32.powi(-9),
        15 if m == 7 => f32::NAN,
        _ => sign * (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7),
    }
}

/// Parse the `fp8_h{Hkv}_d{D}` dump kind into `(Hkv, head_dim)`.
fn parse_fp8_kind(kind: &str) -> (usize, usize) {
    let (h, d) = kind
        .strip_prefix("fp8_h")
        .and_then(|rest| rest.split_once("_d"))
        .unwrap_or_else(|| panic!("malformed fp8 dump kind {kind:?}"));
    (
        h.parse().expect("fp8 dump kind Hkv"),
        d.parse().expect("fp8 dump kind head_dim"),
    )
}

fn decode(kind: &str, bytes: &[u8]) -> Vec<f32> {
    match kind {
        "f32" => {
            assert_eq!(bytes.len() % 4, 0);
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        }
        "f16" => {
            assert_eq!(bytes.len() % 2, 0);
            bytes
                .chunks_exact(2)
                .map(|b| half(u16::from_le_bytes(b.try_into().unwrap())))
                .collect()
        }
        "i8" => bytes.iter().map(|&b| (b as i8) as f32).collect(),
        "q8_0" => {
            assert_eq!(bytes.len() % 34, 0);
            bytes
                .chunks_exact(34)
                .flat_map(|b| {
                    let d = half(u16::from_le_bytes([b[0], b[1]]));
                    b[2..].iter().map(move |&q| (q as i8) as f32 * d)
                })
                .collect()
        }
        // Native fp8 token row: `[Hkv*D E4M3FN codes][Hkv inline f16 scales]`.
        // Codes and scales are decoded with their own formats — a scale byte is
        // never read as a code, and the Q8_0 34-byte block math never applies.
        // Every code and every scale of every dumped row is visited.
        kind if kind.starts_with("fp8_h") => {
            let (hkv, dim) = parse_fp8_kind(kind);
            let row = hkv * (dim + 2);
            assert_eq!(
                bytes.len() % row,
                0,
                "fp8 dump must be whole `Hkv*(D+2)`-byte token rows"
            );
            let mut values = Vec::with_capacity((bytes.len() / row) * hkv * (dim + 1));
            for r in bytes.chunks_exact(row) {
                let scales = &r[hkv * dim..];
                for k in 0..hkv {
                    let scale = half(u16::from_le_bytes([scales[2 * k], scales[2 * k + 1]]));
                    for i in 0..dim {
                        values.push(scale * e4m3fn(r[k * dim + i]));
                    }
                    // The inline f16 scale itself, checked for finiteness too.
                    values.push(scale);
                }
            }
            values
        }
        other => panic!("unsupported dump encoding {other}"),
    }
}

/// Bytes of one KV token row **per side** for the live cache encoding.
fn kv_row_bytes(kv: &KvCache) -> usize {
    if kv.quant_fp8 {
        kv.fp8_bytes_per_token().expect("fp8 row bytes")
    } else {
        assert!(
            kv.quant_q8 && kv.head_dim.is_multiple_of(32),
            "KV row bytes: unsupported encoding (quant_q8={}, quant_fp8={}, head_dim={})",
            kv.quant_q8,
            kv.quant_fp8,
            kv.head_dim
        );
        kv.n_kv_heads * (kv.head_dim / 32) * 34
    }
}

/// Dump/decode kind for the live cache encoding.
fn kv_dump_kind(kv: &KvCache) -> String {
    if kv.quant_fp8 {
        format!("fp8_h{}_d{}", kv.n_kv_heads, kv.head_dim)
    } else {
        "q8_0".into()
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for b in digest {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

fn save(
    gpu: &Gpu,
    dir: &Path,
    entries: &mut Vec<Entry>,
    name: String,
    kind: &str,
    tensor: &GpuTensor,
    bytes: usize,
    expect_zero: bool,
) -> Result<()> {
    assert!(
        bytes <= tensor.buf.size(),
        "{name}: download exceeds allocation"
    );
    let mut raw = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut raw, &tensor.buf)?;
    let values = decode(kind, &raw);
    let nonfinite = values.iter().filter(|v| !v.is_finite()).count();
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let file = format!("{:04}.bin", entries.len());
    std::fs::write(dir.join(&file), &raw)?;
    entries.push(Entry {
        name: name.clone(),
        kind: kind.into(),
        file,
        bytes,
        sha256: sha256_hex(&raw),
        nonfinite,
        min: if min.is_finite() { min } else { 0.0 },
        max: if max.is_finite() { max } else { 0.0 },
    });
    eprintln!(
        "  dump {name}: {} values ({kind}), nonfinite={nonfinite}, range=[{min:.6e},{max:.6e}]",
        values.len()
    );
    if nonfinite != 0 {
        return Err(format!("{name}: {nonfinite} non-finite values in dumped state/logits").into());
    }
    if expect_zero {
        let nonzero = values.iter().filter(|v| **v != 0.0).count();
        if nonzero != 0 {
            return Err(format!(
                "{name}: initial state is not the zero state ({nonzero}/{} values non-zero)",
                values.len()
            )
            .into());
        }
    }
    Ok(())
}

// ── Worker (one arm, one fresh process) ─────────────────────────────────

fn env_req(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| {
        panic!(
            "{key} must be set: this test is the child arm of a two-process oracle. \
             Run the ignored orchestrator `resident_lowered_passback_oracle` \
             (it sets every variable per fresh child), or set the full set yourself."
        )
    })
}

fn projection_tensors(layer: &LayerWeights) -> Vec<(&'static str, &GpuTensor)> {
    match layer {
        LayerWeights::DeltaNet(l) => vec![
            ("wqkv", &l.wqkv.buf),
            ("wz", &l.wz.buf),
            ("w_beta", &l.w_beta.buf),
            ("w_alpha", &l.w_alpha.buf),
            ("wo", &l.wo.buf),
            ("w_gate", &l.w_gate.buf),
            ("w_up", &l.w_up.buf),
            ("w_down", &l.w_down.buf),
        ],
        LayerWeights::FullAttn(l) => vec![
            ("wq", &l.wq.buf),
            ("wk", &l.wk.buf),
            ("wv", &l.wv.buf),
            ("wo", &l.wo.buf),
            ("w_gate", &l.w_gate.buf),
            ("w_up", &l.w_up.buf),
            ("w_down", &l.w_down.buf),
        ],
        _ => Vec::new(),
    }
}

fn layer_kind_name(ty: LayerType) -> &'static str {
    match ty {
        LayerType::LinearAttention => "LinearAttention",
        LayerType::FullAttention => "FullAttention",
    }
}

/// Post-token checkpoints (1-based committed-token counts) whose FULL valid state
/// is dumped: a quarter, a half and the end of the run, deduped and clamped into
/// `1..=steps`. `steps = 12` gives 3, 6 and 12, so the recurrent and KV state is
/// compared at several distinct points of the decode, not only at the end.
fn state_checkpoints(steps: usize) -> Vec<usize> {
    let mut marks = vec![steps / 4, steps / 2, steps];
    marks.retain(|n| *n >= 1 && *n <= steps);
    marks.sort_unstable();
    marks.dedup();
    marks
}

/// One step's logits row (all `vocab` f32) as its own entry and file. Written once
/// from the step that produced it, so no stage ever rewrites another's file.
fn dump_logits(dir: &Path, entries: &mut Vec<Entry>, step: usize, raw: &[u8]) -> Result<()> {
    let values = decode("f32", raw);
    let nonfinite = values.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        nonfinite, 0,
        "logits step {step}: {nonfinite} non-finite values"
    );
    let file = format!("logits_step_{step:03}.bin");
    std::fs::write(dir.join(&file), raw)?;
    entries.push(Entry {
        name: format!("logits/step_{step:03}"),
        kind: "f32".into(),
        file,
        bytes: raw.len(),
        sha256: sha256_hex(raw),
        nonfinite,
        min: values.iter().copied().fold(f32::INFINITY, f32::min),
        max: values.iter().copied().fold(f32::NEG_INFINITY, f32::max),
    });
    Ok(())
}

/// The full valid state after `committed_tokens` completed tokens: the dense GDN
/// state (codes/scales/conv, plus EF when the Q8 state carries it) for every
/// LinearAttention layer, and the **valid prefix** of the KV for every
/// FullAttention layer in the live encoding — Q8_0 key + value (the block scale
/// inside each 34-byte block) or, under `fp8`, the native E4M3FN token rows
/// `[Hkv*D codes][Hkv inline f16 scales]`.
///
/// `committed_tokens == 0` dumps the GDN state only — the freshly allocated
/// initial state — and `expect_zero` then asserts those planes really are the
/// zero state every arm must start from.
#[allow(clippy::too_many_arguments)]
fn dump_state(
    gpu: &Gpu,
    dir: &Path,
    entries: &mut Vec<Entry>,
    stage: &str,
    committed_tokens: usize,
    expect_zero: bool,
    config: &Qwen35Config,
    dn: &DeltaNetState,
    kv: &KvCache,
) -> Result<()> {
    gpu.hip.device_synchronize()?;
    // Live KV encoding is uniform across FullAttention layers; resolve it once.
    let kv_kind = kv_dump_kind(kv);
    let kv_prefix = committed_tokens * kv_row_bytes(kv);
    let mut la = 0usize;
    for (layer, ty) in config.layer_types.iter().enumerate() {
        let prefix = format!("{stage}/layer_{layer:02}");
        if *ty == LayerType::LinearAttention {
            let n = config.linear_num_value_heads * config.linear_value_head_dim.pow(2);
            save(
                gpu,
                dir,
                entries,
                format!("{prefix}/state_codes"),
                "i8",
                &dn.s_matrices[la],
                n,
                expect_zero,
            )?;
            save(
                gpu,
                dir,
                entries,
                format!("{prefix}/state_scales"),
                "f32",
                &dn.s_scales[la],
                dn.s_scales[la].numel() * 4,
                expect_zero,
            )?;
            if dn.s_ef_residual.len() == dn.s_matrices.len() {
                save(
                    gpu,
                    dir,
                    entries,
                    format!("{prefix}/ef"),
                    "f16",
                    &dn.s_ef_residual[la],
                    n * 2,
                    expect_zero,
                )?;
            }
            save(
                gpu,
                dir,
                entries,
                format!("{prefix}/conv"),
                "f32",
                &dn.conv_states[la],
                dn.conv_states[la].numel() * 4,
                expect_zero,
            )?;
            la += 1;
        } else if committed_tokens > 0 {
            // Raw valid prefix of the live KV encoding (`q8_0` blocks or fp8
            // token rows), committed tokens only — never the uninitialized
            // capacity. The row kind carries the fp8 head geometry.
            save(
                gpu,
                dir,
                entries,
                format!("{prefix}/key"),
                &kv_kind,
                &kv.k_gpu[layer],
                kv_prefix,
                false,
            )?;
            save(
                gpu,
                dir,
                entries,
                format!("{prefix}/value"),
                &kv_kind,
                &kv.v_gpu[layer],
                kv_prefix,
                false,
            )?;
        }
    }
    Ok(())
}

fn worker() -> Result<()> {
    let arm = env_req(ARM_ENV);
    let cut: usize = env_req(CUT_ENV).parse()?;
    let share: f64 = env_req(SHARE_ENV).parse()?;
    let out = PathBuf::from(env_req(OUT_ENV));
    let model = std::env::var(MODEL_ENV).unwrap_or_else(|_| default_model());
    let kv_sel = KvSel::from_env();
    let steps: usize = std::env::var(STEPS_ENV)
        .ok()
        .map(|v| v.parse().expect("steps"))
        .unwrap_or(DEFAULT_STEPS);
    let nonblocking = std::env::var(NONBLOCK_ENV).ok().as_deref() == Some("1");
    assert!(steps >= 2, "need at least two teacher-forced tokens");

    let expect_lowered = matches!(arm.as_str(), "mixed" | "passback0");
    let expect_exec = if arm == "cpu" { "cpu" } else { "passback" };
    let declared_lowered = std::env::var("HIPFIRE_FORWARD_LOWERED").unwrap_or_default();
    let declared_exec = std::env::var("HIPFIRE_OFFLOAD_EXEC").unwrap_or_default();
    if expect_lowered {
        // The mixed route is the production default: the toggle is absent (or an
        // explicit non-"0" value); `=0` would force the all-hand reference.
        assert!(
            matches!(declared_lowered.as_str(), "" | "1"),
            "arm {arm} must leave HIPFIRE_FORWARD_LOWERED unset (got {declared_lowered:?})"
        );
    } else {
        assert_eq!(
            declared_lowered, "0",
            "arm {arm} must set HIPFIRE_FORWARD_LOWERED=0"
        );
    }
    assert_eq!(
        declared_exec, expect_exec,
        "child env HIPFIRE_OFFLOAD_EXEC disagrees with arm {arm}"
    );
    assert_eq!(
        hipfire_dispatch::passback_enabled(),
        expect_exec == "passback",
        "arm {arm}: pass-back mode disagrees with HIPFIRE_OFFLOAD_EXEC={declared_exec}"
    );
    assert!(
        hipfire_dispatch::cpu_exec_enabled(),
        "arm {arm}: a host-executing mode (cpu|passback) is required"
    );
    if expect_exec == "passback" {
        // The oracle is only reproducible because the share is PINNED: `Share(f)`
        // is read directly for the row split, while `Auto` consults a scheduler
        // that follows measured host rates. Unset/empty/unknown all resolve to
        // Auto, so refuse that here instead of comparing two drifting runs.
        match hipfire_config::memory::offload_passback_share() {
            hipfire_config::memory::PassbackShare::Share(f) => assert_eq!(
                f, share,
                "arm {arm}: resolved pass-back share {f} != pinned share {share}"
            ),
            hipfire_config::memory::PassbackShare::Auto => panic!(
                "arm {arm}: HIPFIRE_OFFLOAD_PASSBACK_SHARE resolved to Auto; the oracle \
                 requires a pinned share (unset/empty/unknown all mean Auto)"
            ),
        }
    }

    if !Path::new(&model).is_file() {
        panic!(
            "oracle model {model} not found (set {MODEL_ENV}): an explicitly requested \
             hardware oracle must not pass without the fixture"
        );
    }

    eprintln!(
        "[oracle {arm}] model={model} cut={cut} share={share} steps={steps} \
         nonblocking={nonblocking} kv_mode={}",
        kv_sel.as_str()
    );
    let mut gpu = Gpu::init().expect("Gpu::init");
    // `Stream` is not `Copy`: it lives only in `gpu.active_stream` (as every other
    // caller does), and the drain/destroy below reach it through that field.
    if nonblocking {
        gpu.active_stream = Some(gpu.hip.stream_create_non_blocking()?);
        eprintln!("[oracle {arm}] active stream: non-blocking");
    } else {
        eprintln!("[oracle {arm}] active stream: default");
    }

    let mut hfq = HfqFile::open(Path::new(&model))?;
    let config = config_from_hfq(&hfq)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)?;
    assert_eq!(
        config.num_experts, 0,
        "the mixed resident-layer branch is dense-only"
    );
    assert!(
        config.n_layers > cut,
        "spill cut {cut} must leave resident layers"
    );
    let seed = tokenizer.encode(PROMPT);
    assert!(!seed.is_empty(), "prompt tokenized to empty");
    let tokens: Vec<u32> = seed.iter().copied().cycle().take(steps).collect();

    let first = config.layer_types[cut];
    assert_eq!(
        first,
        expected_first_resident(cut),
        "fixture layer_types[{cut}] is {first:?}, not the pinned cut kind"
    );

    let placement = Placement::dense_prefix(config.n_layers, cut);
    let host_layers = placement.host_layers() + placement.host_expert_layers();
    assert_eq!(
        host_layers, cut,
        "dense prefix placement must spill exactly `cut` layers"
    );
    let layout = Layout::single(config.n_layers).with_placement(placement);
    let mut source = HfqSource::new(&mut hfq, &config);
    let weights = load_weights(&mut source, std::slice::from_mut(&mut gpu), &layout)?;
    // What `Architecture::load_weights_with_placement` does for a resolved
    // placement: the model-level CPU-weight gate that selects the mixed route.
    gpu.set_host_cpu_weights(hipfire_dispatch::cpu_offload_active(host_layers));
    assert!(
        gpu.owns_host_cpu_weights(),
        "arm {arm}: loader did not set the host CPU-weight gate (no mixed route possible)"
    );
    assert!(
        hipfire_dispatch::cpu_offload_active(host_layers),
        "arm {arm}: host_layers={host_layers} must arm the CPU-offload predicate"
    );

    // Actual placement: every dense projection of a spilled layer is host-mapped
    // and every projection of a resident layer is device-resident.
    let mut residency_mismatches = 0usize;
    let mut first_mismatch = String::new();
    for (i, layer) in weights.layers.iter().enumerate() {
        for (name, t) in projection_tensors(layer) {
            let host = gpu.host_located(t);
            if host != (i < cut) {
                residency_mismatches += 1;
                if first_mismatch.is_empty() {
                    first_mismatch = format!(
                        "layer {i} {name}: host_located={host}, expected spill={}",
                        i < cut
                    );
                }
            }
        }
    }
    assert_eq!(
        residency_mismatches, 0,
        "arm {arm}: placement is not the dense spill prefix ({first_mismatch})"
    );

    let mask: Vec<bool> = config
        .layer_types
        .iter()
        .map(|t| *t == LayerType::FullAttention)
        .collect();
    let cap = (steps + 32).max(512);
    // Same filtered, absolute-layer-owned VMM cache on either encoding: only the
    // K/V codec changes (Q8_0 blocks vs native E4M3FN token rows).
    let mut kv = match kv_sel {
        KvSel::Q8 => KvCache::new_gpu_q8_vmm_capped_filtered(
            &mut gpu,
            &mask,
            config.n_kv_heads,
            config.head_dim,
            cap,
            cap,
        )?,
        KvSel::Fp8 => KvCache::new_gpu_fp8_vmm_capped_filtered(
            &mut gpu,
            &mask,
            config.n_kv_heads,
            config.head_dim,
            cap,
            cap,
        )?,
    };
    assert_eq!(kv.quant_fp8, kv_sel == KvSel::Fp8, "KV fp8 flag mismatch");
    let kv_stride = kv_row_bytes(&kv);
    eprintln!(
        "[oracle {arm}] KV cache: kind={} row_bytes={kv_stride} (per side)",
        kv_dump_kind(&kv)
    );
    let mut dn = DeltaNetState::new_with_quant(&mut gpu, &config, StateQuant::Q8)?;
    let scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 64, cap)?;
    let ef_present = dn.s_ef_residual.len() == dn.s_matrices.len();
    eprintln!(
        "[oracle {arm}] GDN EF plane: {ef_present}; vocab={}",
        config.vocab_size
    );

    std::fs::create_dir_all(&out)?;
    let mut entries: Vec<Entry> = Vec::new();
    let mut launches_per_step: Vec<u64> = Vec::with_capacity(steps);
    let mut kernels_per_step: Vec<u64> = Vec::with_capacity(steps);
    let checkpoints = state_checkpoints(steps);
    // The state the run starts from, dumped before any token: identical zero bytes
    // in both arms is a precondition of the comparison, not an assumption.
    dump_state(
        &gpu,
        &out,
        &mut entries,
        "state_initial",
        0,
        true,
        &config,
        &dn,
        &kv,
    )?;
    eprintln!("[oracle {arm}] state checkpoints after tokens {checkpoints:?} (+ initial state)");

    for (step, &tok) in tokens.iter().enumerate() {
        // Exact initial state on both arms: the per-step logits below are the
        // first thing either arm computes, from the same zeroed GDN state and
        // the same empty KV.
        hip_bridge::launch_counters::reset();
        forward_scratch(
            &mut gpu, &weights, &config, tok, step, &mut kv, &mut dn, &scratch,
        )?;
        // Read before the drain: launches/copies are tallied when issued, so the
        // per-step tally is this step's own issue count. Diagnostic metadata only
        // (an incidental implementation count, not a numerical invariant).
        launches_per_step.push(hip_bridge::launch_counters::count());
        kernels_per_step.push(hip_bridge::launch_counters::launch_kernel::count());
        // One drain per completed token; nothing is read between layers, and the
        // state download below happens only after the whole token.
        gpu.hip.device_synchronize()?;
        let mut logits = vec![0u8; config.vocab_size * 4];
        gpu.hip.memcpy_dtoh(&mut logits, &scratch.logits.buf)?;
        let values = decode("f32", &logits);
        let argmax = values
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            })
            .0;
        dump_logits(&out, &mut entries, step, &logits)?;
        eprintln!(
            "[oracle {arm}] step {step} pos={step} tok={tok} argmax={argmax} \
             launches={} kernels={}",
            launches_per_step[step], kernels_per_step[step]
        );
        let committed = step + 1;
        if checkpoints.contains(&committed) {
            dump_state(
                &gpu,
                &out,
                &mut entries,
                &format!("state_token_{committed:03}"),
                committed,
                false,
                &config,
                &dn,
                &kv,
            )?;
            eprintln!(
                "[oracle {arm}] state checkpoint after token {committed} ({} valid KV rows)",
                committed
            );
        }
    }

    // Drain the active stream before the last state readback when one is in use:
    // the last token's queued CPU-return tail is retired here rather than at
    // teardown (each checkpoint above drained its own token on the same stream).
    if let Some(s) = gpu.active_stream.as_ref() {
        gpu.hip.stream_synchronize(s)?;
    }

    let scheduler: Vec<ShapeRec> = hipfire_dispatch::scheduler_snapshot()
        .iter()
        .map(|s| ShapeRec {
            dtype: format!("{:?}", s.dtype),
            k: s.k,
            share: s.share,
            samples: s.samples,
        })
        .collect();
    let sampled_steps: u64 = scheduler.iter().map(|s| s.samples).sum();
    let (cpu_steps, host_mapped_gpu_steps) = hipfire_dispatch::cpu_exec_counters();
    let (split_steps, split_gpu_rows) = hipfire_dispatch::split_counters();
    let launches_total: u64 = launches_per_step.iter().sum();
    let kernels_total: u64 = kernels_per_step.iter().sum();

    let manifest = Manifest {
        arm: arm.clone(),
        model: model.clone(),
        arch: gpu.arch.clone(),
        cut,
        share,
        steps,
        nonblocking,
        kv_mode: kv_sel.as_str().into(),
        kv_row_bytes: kv_stride,
        tokens,
        vocab: config.vocab_size,
        dim: config.dim,
        n_layers: config.n_layers,
        layer_types: config
            .layer_types
            .iter()
            .map(|t| layer_kind_name(*t).into())
            .collect(),
        passback_enabled: hipfire_dispatch::passback_enabled(),
        cpu_exec_enabled: hipfire_dispatch::cpu_exec_enabled(),
        owns_host_cpu_weights: gpu.owns_host_cpu_weights(),
        host_layers,
        resident_first_layer: cut,
        resident_first_kind: layer_kind_name(first).into(),
        expected_resident_first_kind: layer_kind_name(expected_first_resident(cut)).into(),
        residency_mismatches,
        cpu_steps,
        host_mapped_gpu_steps,
        split_steps,
        split_gpu_rows,
        scheduler,
        sampled_steps,
        state_checkpoints: checkpoints,
        launches_per_step,
        kernels_per_step,
        launches_total,
        kernels_total,
        entries,
    };

    // Engagement is a property of the arm itself, so it fails here too when the
    // worker is driven by hand.
    match arm.as_str() {
        "mixed" | "hand" => {
            assert!(
                split_steps > 0 || share == 0.0,
                "arm {arm}: pass-back reported no row split on {} steps (share {share})",
                steps
            );
            if share > 0.0 {
                assert!(
                    sampled_steps > 0,
                    "arm {arm}: no sampled pass-back step; the async-return join never ran"
                );
                assert!(
                    split_steps as u64 > sampled_steps,
                    "arm {arm}: every split step was sampled ({} vs {}), so the queued \
                     unsampled return path was never taken",
                    split_steps,
                    sampled_steps
                );
            }
        }
        "cpu" | "passback0" => {
            assert_eq!(
                split_steps, 0,
                "arm {arm}: share-0 twin must not split any step"
            );
            eprintln!(
                "[oracle {arm}] share-0 twin: cpu_steps={cpu_steps} \
                 host_mapped_gpu_steps={host_mapped_gpu_steps} (host-mapped GEMVs with no \
                 CPU plan; the two twin arms must report the same number)"
            );
        }
        other => panic!("unknown arm {other}"),
    }

    std::fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    eprintln!(
        "[oracle {arm}] OK: {} entries; launches={launches_total} kernels={kernels_total} \
         cpu_steps={cpu_steps} split_steps={split_steps} split_gpu_rows={split_gpu_rows} \
         samples={sampled_steps} -> {}",
        manifest.entries.len(),
        out.display()
    );
    if let Some(s) = gpu.active_stream.take() {
        let _ = gpu.hip.stream_destroy(s);
    }
    // Process isolation is intentional; OS teardown releases the single-run GPU.
    Ok(())
}

// ── Orchestrator ────────────────────────────────────────────────────────

fn spawn_arm(
    exe: &Path,
    case: &Case,
    arm: &ArmSpec,
    kv: KvSel,
    root: &Path,
    model: &str,
    steps: usize,
) -> Result<PathBuf> {
    let out = root
        .join(format!(
            "cut{}_share{}_nb{}",
            case.cut, case.share, case.nonblocking as u8
        ))
        .join(arm.label);
    std::fs::create_dir_all(&out)?;
    let mut cmd = Command::new(exe);
    cmd.arg("--exact")
        .arg(WORKER_TEST)
        .arg("--ignored")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(ARM_ENV, arm.label)
        .env(CUT_ENV, case.cut.to_string())
        .env(SHARE_ENV, format!("{}", case.share))
        .env(OUT_ENV, &out)
        .env(MODEL_ENV, model)
        .env(STEPS_ENV, steps.to_string())
        .env(NONBLOCK_ENV, if case.nonblocking { "1" } else { "0" })
        .env(KV_MODE_ENV, kv.as_str())
        .env("HIPFIRE_OFFLOAD_EXEC", arm.exec);
    if arm.lowered {
        // The production mixed route is the DEFAULT (`HIPFIRE_FORWARD_LOWERED`
        // unset); leave it unset so the arm exercises what ships.
        cmd.env_remove("HIPFIRE_FORWARD_LOWERED");
    } else {
        cmd.env("HIPFIRE_FORWARD_LOWERED", "0");
    }
    if arm.set_share {
        cmd.env("HIPFIRE_OFFLOAD_PASSBACK_SHARE", format!("{}", case.share));
    } else {
        cmd.env_remove("HIPFIRE_OFFLOAD_PASSBACK_SHARE");
    }
    eprintln!(
        "[oracle] spawning arm {} (cut={} share={} exec={} lowered={} nonblocking={} kv={}) -> {}",
        arm.label,
        case.cut,
        case.share,
        arm.exec,
        arm.lowered as u8,
        case.nonblocking as u8,
        kv.as_str(),
        out.display()
    );
    let output = cmd.output()?;
    std::fs::write(out.join("child.log"), &output.stderr)?;
    if !output.status.success() {
        eprintln!(
            "[oracle] arm {} FAILED ({}):\n{}",
            arm.label,
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        return Err(format!("arm {} exited {}", arm.label, output.status).into());
    }
    if !output.stdout.is_empty() {
        std::fs::write(out.join("child.stdout.log"), &output.stdout)?;
    }
    let manifest = out.join("manifest.json");
    if !manifest.is_file() {
        return Err(format!(
            "arm {} produced no manifest at {} (child test filter/target mismatch?)",
            arm.label,
            manifest.display()
        )
        .into());
    }
    Ok(out)
}

fn load_manifest(dir: &Path) -> Result<Manifest> {
    let raw = std::fs::read(dir.join("manifest.json"))?;
    Ok(serde_json::from_slice(&raw)?)
}

fn numeric_diff(name: &str, kind: &str, a: &[u8], b: &[u8]) {
    let va = decode(kind, a);
    let vb = decode(kind, b);
    assert_eq!(va.len(), vb.len(), "{name}: decoded length drift");
    let mut first = None;
    let mut diffs = 0usize;
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    for (i, (x, y)) in va.iter().zip(vb.iter()).enumerate() {
        if x != y {
            if first.is_none() {
                first = Some(i);
            }
            diffs += 1;
            let d = (x - y).abs();
            max_abs = max_abs.max(d);
            sum_abs += d as f64;
        }
    }
    eprintln!(
        "[oracle] DIFF {name} ({kind}): {diffs}/{} elements differ, first={:?}, \
         max|Δ|={max_abs:.6e}, mean|Δ|={:.6e}",
        va.len(),
        first,
        if diffs == 0 {
            0.0
        } else {
            sum_abs / diffs as f64
        }
    );
    if let Some(i) = first {
        eprintln!(
            "[oracle]   first sample idx {i}: all-hand={:.9e} mixed={:.9e}",
            va[i], vb[i]
        );
    }
}

fn compare_case(case: &Case, a_dir: &Path, b_dir: &Path) -> Result<()> {
    let (a, b) = (load_manifest(a_dir)?, load_manifest(b_dir)?);
    let tag = format!(
        "cut{} share{} nonblocking{} [{} vs {}]",
        case.cut, case.share, case.nonblocking as u8, a.arm, b.arm
    );
    assert_eq!(a.tokens, b.tokens, "{tag}: token streams differ");
    assert_eq!(
        (a.cut, a.share, a.steps),
        (b.cut, b.share, b.steps),
        "{tag}: knobs differ"
    );
    assert_eq!(
        a.layer_types, b.layer_types,
        "{tag}: layer topology differs"
    );
    assert_eq!(
        (a.vocab, a.dim, a.n_layers, a.host_layers),
        (b.vocab, b.dim, b.n_layers, b.host_layers),
        "{tag}: model shape differs"
    );
    assert_eq!(a.nonblocking, b.nonblocking, "{tag}: stream mode differs");
    assert_eq!(a.kv_mode, b.kv_mode, "{tag}: KV encoding differs");
    assert_eq!(
        a.kv_row_bytes, b.kv_row_bytes,
        "{tag}: KV row stride differs"
    );
    assert!(a.kv_row_bytes > 0, "{tag}: KV row stride must be non-zero");
    assert_eq!(
        a.state_checkpoints, b.state_checkpoints,
        "{tag}: state checkpoint sets differ"
    );
    assert!(
        a.state_checkpoints.len() >= 2,
        "{tag}: only {} state checkpoint(s) dumped (want >= 2 post-token points)",
        a.state_checkpoints.len()
    );
    assert_eq!(
        *a.state_checkpoints.last().unwrap(),
        a.steps,
        "{tag}: the last state checkpoint must be the end of the run"
    );
    assert_eq!(a.resident_first_layer, case.cut);
    assert_eq!(
        a.resident_first_kind, a.expected_resident_first_kind,
        "{tag}"
    );
    assert_eq!(
        a.resident_first_kind, b.resident_first_kind,
        "{tag}: resident cut kind differs between arms"
    );
    for m in [&a, &b] {
        assert!(
            m.owns_host_cpu_weights,
            "{tag}: arm {} lacks the host CPU-weight gate",
            m.arm
        );
        assert!(
            m.cpu_exec_enabled,
            "{tag}: arm {} is not host-executing",
            m.arm
        );
        assert_eq!(
            m.residency_mismatches, 0,
            "{tag}: arm {} placement drift",
            m.arm
        );
    }
    match case.kind {
        Kind::Mixed => {
            for m in [&a, &b] {
                assert!(
                    m.passback_enabled,
                    "{tag}: arm {} is not in pass-back mode",
                    m.arm
                );
                assert!(
                    m.split_steps > 0,
                    "{tag}: arm {} reported no pass-back row split",
                    m.arm
                );
                assert!(
                    m.sampled_steps > 0 && m.split_steps as u64 > m.sampled_steps,
                    "{tag}: arm {} did not exercise both sampled ({}) and unsampled ({}) \
                     pass-back returns",
                    m.arm,
                    m.sampled_steps,
                    m.split_steps as u64 - m.sampled_steps
                );
            }
            // Launch tallies are DIAGNOSTIC METADATA only: an incidental implementation
            // count, not a numerical invariant of the mixed route. Published here so
            // the route proof can be checked out-of-band; never asserted.
            eprintln!(
                "[oracle] {tag}: kernels/step all-hand={:?}",
                a.kernels_per_step
            );
            eprintln!(
                "[oracle] {tag}: kernels/step mixed   ={:?}",
                b.kernels_per_step
            );
            eprintln!(
                "[oracle] {tag}: launches/step all-hand={:?}",
                a.launches_per_step
            );
            eprintln!(
                "[oracle] {tag}: launches/step mixed   ={:?}",
                b.launches_per_step
            );
            eprintln!(
                "[oracle] {tag}: totals all-hand launches={} kernels={}; mixed launches={} kernels={}",
                a.launches_total, a.kernels_total, b.launches_total, b.kernels_total
            );
        }
        Kind::Twin => {
            // The documented twin: share 0 is byte-identical to `exec=cpu`. Both
            // arms run all-hand over the same spill, so the split counters must be
            // empty and the two arms must agree on the host-mapped coverage they
            // produced (the load-time coverage line remains the authority).
            eprintln!(
                "[oracle] {tag}: cpu_steps {} vs {}; host_mapped_gpu_steps {} vs {}",
                a.cpu_steps, b.cpu_steps, a.host_mapped_gpu_steps, b.host_mapped_gpu_steps
            );
            for m in [&a, &b] {
                assert_eq!(m.split_steps, 0, "{tag}: arm {} split under share 0", m.arm);
            }
            assert_eq!(
                a.host_mapped_gpu_steps, b.host_mapped_gpu_steps,
                "{tag}: share-0 twin arms disagree on host-mapped GEMVs taken on the GPU"
            );
        }
    }

    assert_eq!(
        a.entries.len(),
        b.entries.len(),
        "{tag}: dump entry counts differ ({} vs {})",
        a.entries.len(),
        b.entries.len()
    );
    let mut mismatches = 0usize;
    for (ea, eb) in a.entries.iter().zip(b.entries.iter()) {
        assert_eq!(ea.name, eb.name, "{tag}: entry order/name drift");
        assert_eq!(ea.kind, eb.kind, "{tag}: {}: dump encoding drift", ea.name);
        assert_eq!(ea.bytes, eb.bytes, "{tag}: {}: extent drift", ea.name);
        let ra = std::fs::read(a_dir.join(&ea.file))?;
        let rb = std::fs::read(b_dir.join(&eb.file))?;
        if ra != rb {
            mismatches += 1;
            if mismatches <= 8 {
                numeric_diff(&ea.name, &ea.kind, &ra, &rb);
                eprintln!("[oracle]   sha all-hand={} mixed={}", ea.sha256, eb.sha256);
            }
        }
    }
    if mismatches != 0 {
        return Err(format!(
            "{tag}: {mismatches}/{} dumped tensors differ byte-wise between {} and {}; \
             dumps retained under {} and {}",
            a.entries.len(),
            a.arm,
            b.arm,
            a_dir.display(),
            b_dir.display()
        )
        .into());
    }
    eprintln!(
        "[oracle] {tag}: BYTE-EXACT across {} tensors ({} logits rows, full valid state)",
        a.entries.len(),
        a.steps
    );
    Ok(())
}

/// Ignored two-process orchestrator: for every case, spawn the all-hand and the
/// mixed arm in fresh processes and compare their dumps byte-for-byte.
#[test]
#[ignore = "requires a real HIP GPU + the selected mq4 fixture; spawns fresh child processes"]
fn resident_lowered_passback_oracle() {
    if std::env::var(ARM_ENV).is_ok() {
        // This process IS a child arm: the worker test below owns the work.
        return;
    }
    let exe = std::env::current_exe().expect("current_exe");
    let model = std::env::var(MODEL_ENV).unwrap_or_else(|_| default_model());
    if !Path::new(&model).is_file() {
        panic!(
            "oracle model {model} not found (set {MODEL_ENV}); this ignored hardware \
             oracle must fail rather than skip when explicitly requested"
        );
    }
    let kv_sel = KvSel::from_env();
    let steps: usize = std::env::var(STEPS_ENV)
        .ok()
        .map(|v| v.parse().expect("steps"))
        .unwrap_or(DEFAULT_STEPS);
    let cases = default_cases();
    let root = std::env::var(OUT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::temp_dir().join(format!(
                "hipfire-resident-lowered-passback-{}",
                std::process::id()
            ))
        });
    std::fs::create_dir_all(&root).expect("create oracle root");
    eprintln!(
        "[oracle] {} case(s), model={model}, steps={steps}, fixture={}, kv_mode={}, \
         artifacts under {}",
        cases.len(),
        if fixture_27b() { "27b" } else { "9b" },
        kv_sel.as_str(),
        root.display()
    );

    let mut failures: Vec<String> = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        eprintln!("[oracle] ── case {i}: {case:?}");
        let arms = arms_for(case.kind);
        let mut dirs = Vec::new();
        for arm in &arms {
            match spawn_arm(&exe, case, arm, kv_sel, &root, &model, steps) {
                Ok(dir) => dirs.push(dir),
                Err(e) => failures.push(format!("case {i} {:?} spawn: {e}", case.kind)),
            }
        }
        if dirs.len() == 2 {
            if let Err(e) = compare_case(case, &dirs[0], &dirs[1]) {
                failures.push(format!("case {i} {:?}: {e}", case.kind));
            }
        }
    }

    if !failures.is_empty() {
        for f in &failures {
            eprintln!("[oracle] FAIL: {f}");
        }
        panic!(
            "resident-lowered pass-back oracle: {} case failure(s); artifacts (with per-arm \
             child logs) retained under {}",
            failures.len(),
            root.display()
        );
    }
    eprintln!(
        "[oracle] PASS: every case byte-exact with the all-hand reference; artifacts under {}",
        root.display()
    );
}

/// Ignored child worker: runs exactly one arm from its environment. Driven by
/// [`spawn_arm`] in a fresh process; with no `HIPFIRE_RESIDENT_ORACLE_ARM` it is
/// a no-op (a plain `--ignored` run of this file exercises the orchestrator
/// only, and the orchestrator fails closed if its child produced no manifest).
#[test]
#[ignore = "child arm of resident_lowered_passback_oracle; driven by env in a fresh process"]
fn resident_lowered_passback_worker() {
    if std::env::var(ARM_ENV).is_err() {
        eprintln!(
            "[oracle worker] no {ARM_ENV} in this process; nothing to do (the \
             resident_lowered_passback_oracle test spawns fresh child arms with it set)"
        );
        return;
    }
    if let Err(e) = worker() {
        panic!(
            "oracle worker (arm {}) failed: {e}",
            std::env::var(ARM_ENV).unwrap()
        );
    }
}
