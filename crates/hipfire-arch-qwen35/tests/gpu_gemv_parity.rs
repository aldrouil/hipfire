// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU↔CPU GEMV parity for the host-mapped offload path, one row per quant
//! format.
//!
//! `hipfire_cpu::gemv` is a transcription of the *decode*, not of the GPU
//! kernel's accumulation order (the offload path's contract is coherence, see
//! `docs/perf-checkpoints/2026-09-26-llamacpp-offload-scaling-baseline.md`), so
//! this asserts a tolerance rather than bit-identity. What it does prove per
//! format: the byte layout, the group header/codebook, the code unpacking and
//! the activation-rotation convention all match the real launcher — a wrong
//! nibble order, a missing codebook offset, or a mirrored FWHT lands orders of
//! magnitude outside `TOL`, not a rounding difference.
//!
//! Two sources of weights:
//!
//! * real projection tensors out of the pulled fixtures ([`REAL`]), and
//! * synthetic buffers for the formats no fixture on disk carries ([`SYNTH`]).
//!   A decode check does not care whether the bytes came from a quantizer, so
//!   this keeps the matrix complete instead of "whatever the local model
//!   directory happens to contain" — but the real tensors are the stronger
//!   evidence and run first.
//!
//! `#[ignore]`d: needs an RDNA GPU with a working HIP toolchain; the real-tensor
//! rows additionally need `hipfire pull qwen3.5:2b`, `:2b-mq3`, `:2b-mq6`,
//! `:2b-hf6`. Run explicitly:
//!
//!   cargo test -p hipfire-arch-qwen35 --release --test gpu_gemv_parity -- --ignored --nocapture

use std::collections::BTreeMap;
use std::path::PathBuf;

use hipfire_cpu::gemv::gemv as cpu_gemv;
use hipfire_cpu::quant::{rotate_x, CpuQuant};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::{GemvFamily, WeightRef};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::{DType, Gpu};

/// Relative tolerance against `max|reference|`.
///
/// Measured worst case over the whole matrix on gfx1201 (2026-09-27): `6.65e-7`
/// (`Mq3G256Lloyd`, real 2B tensor); several formats came back *bit-exact*, and
/// the element formats with a single accumulation chain agree exactly. `1e-4` is
/// ~150x that noise and still 3+ orders below any layout/sign failure (a wrong
/// nibble order or a mirrored FWHT changes the result by O(1) relative).
const TOL: f32 = 1e-4;

/// Largest tensor (elements) put through the CPU side, so the test stays fast.
/// The 9B's `down_proj` is `[4096, 12288]` = 50M elements, which the CPU side
/// handles in well under a second.
const MAX_ELEMS: usize = 64 << 20;

/// Tensors per (fixture, quant type), largest first — big `k` is what exercises
/// many groups and a real row stride.
const PER_FORMAT: usize = 2;

/// (fixture file, quant_type, DType, CpuQuant) — measured quant types, not
/// assumed: the registry's `-mq3` tags ship the Lloyd-Max tier (qt 20).
const REAL: &[(&str, u8, DType, CpuQuant)] = &[
    ("qwen3.5-2b.mq4", 13, DType::MQ4G256, CpuQuant::Mq4G256),
    (
        "qwen3.5-2b.mq3",
        20,
        DType::MQ3G256Lloyd,
        CpuQuant::Mq3G256Lloyd,
    ),
    ("qwen3.5-2b.mq6", 15, DType::MQ6G256, CpuQuant::Mq6G256),
    ("qwen3.5-2b.hf6", 8, DType::HFQ6G256, CpuQuant::Hfq6G256),
    // Same format, larger model: the 9B's 4096-wide rows exercise 16 groups.
    ("qwen3.5-9b.mq4", 13, DType::MQ4G256, CpuQuant::Mq4G256),
];

/// Formats with no fixture on disk: (quant_type, DType, CpuQuant).
const SYNTH: &[(u8, DType, CpuQuant)] = &[
    (13, DType::MQ4G256, CpuQuant::Mq4G256),
    (44, DType::MQ4G256V2, CpuQuant::Mq4G256V2),
    (15, DType::MQ6G256, CpuQuant::Mq6G256),
    (17, DType::MQ3G256, CpuQuant::Mq3G256),
    (20, DType::MQ3G256Lloyd, CpuQuant::Mq3G256Lloyd),
    (8, DType::HFQ6G256, CpuQuant::Hfq6G256),
    (6, DType::HFQ4G256, CpuQuant::Hfq4G256),
    (3, DType::Q8_0, CpuQuant::Q8F16),
];

fn models_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("HIPFIRE_MODELS_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME").unwrap_or_else(|| PathBuf::from("/").into());
    PathBuf::from(home).join(".hipfire").join("models")
}

/// Deterministic activation, magnitudes around 1e-3 like a normalized hidden
/// state. Both paths get the identical bytes.
fn activation(k: usize, salt: usize) -> Vec<f32> {
    (0..k)
        .map(|i| {
            let v = ((i + salt) as u64 * 2654435761) % 8192;
            (v as f32 - 4096.0) * 0.000_244_140_625
        })
        .collect()
}

/// Synthetic `[m, k]` weights: a sane header/codebook per group plus a
/// byte-diverse payload, so no value is `Inf`/`NaN` and every code path in the
/// decode is hit.
fn synth_weights(q: CpuQuant, m: usize, k: usize) -> Vec<u8> {
    let (ge, gb) = (q.group_elems(), q.group_bytes());
    let mut out = Vec::with_capacity(m * (k / ge) * gb);
    for row in 0..m {
        for g in 0..k / ge {
            let salt = (row * (k / ge) + g) * 7;
            let mut bytes: Vec<u8> = (0..gb).map(|i| ((i + salt) * 37 + 11) as u8).collect();
            let f32x2 = |a: f32, b: f32| {
                let mut v = [0u8; 8];
                v[..4].copy_from_slice(&a.to_le_bytes());
                v[4..].copy_from_slice(&b.to_le_bytes());
                v
            };
            let f16s = |vals: &[u16]| {
                let mut v = Vec::with_capacity(vals.len() * 2);
                for x in vals {
                    v.extend_from_slice(&x.to_le_bytes());
                }
                v
            };
            match q {
                CpuQuant::Mq4G256 | CpuQuant::Hfq4G256 => {
                    bytes[..8].copy_from_slice(&f32x2(0.03125, -0.5))
                }
                CpuQuant::Mq6G256 | CpuQuant::Hfq6G256 => {
                    bytes[..8].copy_from_slice(&f32x2(0.0078125, -0.125))
                }
                CpuQuant::Mq3G256 => bytes[..8].copy_from_slice(&f32x2(0.015625, 0.25)),
                CpuQuant::Mq4G256V2 => bytes[..8].copy_from_slice(&f16s(&[
                    0x2c00, 0xb400, 0x3800, 0x3a00, // 0.0625, -0.25, 0.5, 0.75
                ])),
                CpuQuant::Mq3G256Lloyd => bytes[..16].copy_from_slice(&f16s(&[
                    0xbc00, 0xb800, 0xb400, 0xb000, 0x3000, 0x3400, 0x3800, 0x3c00,
                ])),
                CpuQuant::Q8F16 => bytes[..2].copy_from_slice(&0x3800u16.to_le_bytes()),
                // The element formats are covered bit-exactly by the fixture
                // tables in `hipfire-cpu` and by the cross-check over real norms.
                CpuQuant::F16 | CpuQuant::F32 | CpuQuant::Bf16 => unreachable!("not in SYNTH"),
            }
            out.extend_from_slice(&bytes);
        }
    }
    out
}

struct Parity {
    max_abs: f32,
    rel: f32,
}

/// Compare the production launcher against the CPU transcription on identical
/// bytes and activation.
fn compare(
    gpu: &mut Gpu,
    gemv: &GemvFamily,
    label: &str,
    q: CpuQuant,
    dtype: DType,
    bytes: &[u8],
    m: usize,
    k: usize,
    worst: &mut BTreeMap<&'static str, Parity>,
) {
    let w = gpu.upload_raw(bytes, &[bytes.len()]).expect("upload w");
    let x_host = activation(k, m);
    let x_dev = gpu.upload_f32(&x_host, &[k]).expect("upload x");
    let y_dev = gpu.alloc_tensor(&[m], DType::F32).expect("alloc y");
    let ctx = DispatchCtx::new(gpu);
    let wr = WeightRef {
        buf: &w,
        dtype,
        m,
        k,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
    };
    gemv.run_auto(&ctx, gpu, &wr, &x_dev, &y_dev)
        .unwrap_or_else(|e| panic!("{label}: launcher failed: {e:?}"));
    gpu.hip.device_synchronize().expect("sync");
    let mut gpu_y = vec![0.0f32; m];
    let gpu_y_bytes =
        unsafe { std::slice::from_raw_parts_mut(gpu_y.as_mut_ptr() as *mut u8, m * 4) };
    gpu.hip.memcpy_dtoh(gpu_y_bytes, &y_dev.buf).expect("dtoh");

    let mut x_cpu = x_host;
    if q.is_fwht_g256() {
        rotate_x(&mut x_cpu);
    }
    let mut cpu_y = vec![0.0f32; m];
    cpu_gemv(q, bytes, m, k, &x_cpu, &mut cpu_y);

    let max_abs = gpu_y
        .iter()
        .zip(&cpu_y)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let scale = cpu_y.iter().fold(0.0f32, |a, b| a.max(b.abs())).max(1e-6);
    let rel = max_abs / scale;
    eprintln!(
        "{label:58} m={m:<6} k={k:<6} max_abs={max_abs:.3e} rel={rel:.3e}",
    );
    assert!(
        rel <= TOL,
        "{label} (qt {q:?}): relative error {rel:.3e} exceeds {TOL:.0e} \
         (max_abs {max_abs:.3e}, scale {scale:.3e})"
    );
    let entry = worst.entry(q_format_name(q)).or_insert(Parity {
        max_abs: 0.0,
        rel: 0.0,
    });
    entry.max_abs = entry.max_abs.max(max_abs);
    entry.rel = entry.rel.max(rel);

    gpu.free_tensor(w).ok();
    gpu.free_tensor(x_dev).ok();
    gpu.free_tensor(y_dev).ok();
}

#[test]
#[ignore]
fn gpu_cpu_gemv_parity_per_format() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP — no GPU ({e:?}).");
            return;
        }
    };
    gpu.ensure_mq_signs().expect("mq signs");
    let dir = models_dir();
    let gemv = GemvFamily::new();
    let mut worst: BTreeMap<&'static str, Parity> = BTreeMap::new();
    let mut synthetic_formats = 0usize;

    for (file, qt, dtype, q) in REAL {
        let path = dir.join(file);
        if !path.exists() {
            eprintln!("skip: {} not present (hipfire pull the matching tag)", path.display());
            continue;
        }
        let hfq = HfqFile::open(&path).expect("open fixture");
        // Largest `[m, k]` tensors of this format, so `k` spans many groups and
        // `m` spans many rows.
        let mut candidates: Vec<(usize, usize, String)> = hfq
            .tensors()
            .iter()
            .filter(|i| {
                i.quant_type == *qt
                    && i.shape.len() == 2
                    && !i.name.contains("embed_tokens")
                    && {
                        let (m, k) = (i.shape[0] as usize, i.shape[1] as usize);
                        k % 256 == 0 && m * k <= MAX_ELEMS
                    }
            })
            .map(|i| (i.shape[0] as usize, i.shape[1] as usize, i.name.clone()))
            .collect();
        candidates.sort_by_key(|(m, k, _)| std::cmp::Reverse(m * k));
        let mut done = 0usize;
        for (m, k, name) in candidates {
            if done >= PER_FORMAT {
                break;
            }
            done += 1;
            let (_, bytes) = hfq
                .tensor_data_vec(&name)
                .unwrap_or_else(|| panic!("{file}: no bytes for {name}"));
            let label = format!("{file} qt={qt} {}", name.rsplit('.').nth(1).unwrap_or(&name));
            compare(&mut gpu, &gemv, &label, *q, *dtype, &bytes, m, k, &mut worst);
        }
        if done == 0 {
            eprintln!("note: {file} carries no 2-D qt {qt} tensor under {MAX_ELEMS} elements");
        }
    }

    for (qt, dtype, q) in SYNTH {
        let (m, k) = (64usize, 1024usize);
        let bytes = synth_weights(*q, m, k);
        let label = format!("synthetic qt={qt} {q:?}");
        compare(&mut gpu, &gemv, &label, *q, *dtype, &bytes, m, k, &mut worst);
        synthetic_formats += 1;
    }

    eprintln!("\nper-format worst case (max_abs, relative):");
    for (fmt, p) in &worst {
        eprintln!("  {fmt:<14} {:.3e}  {:.3e}", p.max_abs, p.rel);
    }
    assert_eq!(
        synthetic_formats,
        SYNTH.len(),
        "every synthetic format must run"
    );
}

fn q_format_name(q: CpuQuant) -> &'static str {
    match q {
        CpuQuant::Mq4G256 => "Mq4G256",
        CpuQuant::Mq4G256V2 => "Mq4G256V2",
        CpuQuant::Mq6G256 => "Mq6G256",
        CpuQuant::Mq3G256 => "Mq3G256",
        CpuQuant::Mq3G256Lloyd => "Mq3G256Lloyd",
        CpuQuant::Hfq6G256 => "Hfq6G256",
        CpuQuant::Hfq4G256 => "Hfq4G256",
        CpuQuant::F16 => "F16",
        CpuQuant::F32 => "F32",
        CpuQuant::Bf16 => "Bf16",
        CpuQuant::Q8F16 => "Q8F16",
    }
}
