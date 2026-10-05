// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Cross-implementation parity for the routed-expert FFN (stage 2 item 4): the
//! production qt44 MoE kernels (`gate_up` + `down`) versus the `hipfire_cpu`
//! reference, over one real expert's bytes, device-resident AND host-mapped.
//!
//! Every other parity harness compares device against host — a dev-vs-host diff
//! of 0 passes even when both sides are equally wrong. This closes that hole for
//! both halves of the routed expert FFN: one real qt44 expert's bytes driven
//! through the production kernels, compared elementwise against `hipfire_cpu`
//! over the same bytes (the decoder proven bit-exact against the canonical
//! arithmetic). Each half runs twice — device-resident, then host-mapped with
//! the pointer table built from `sub_offset(slot*stride)` views exactly as the
//! loader builds it: the one combination nothing else in the tree performs
//! (independent reference AND real host placement).
//!
//! The graded extension (`graded_moe_tag_parity`, same ignore gate) covers the
//! mixed-dtype host layout: one real expert per tag bucket of the graded
//! fixture (MQ6 hot / MQ4 mid / MQ3L cold), driven through the merged
//! tag-branched kernels against `hipfire_cpu` over the same bytes, on both
//! placements. That is the per-expert evidence the spill decode needs — the
//! full-model resident baseline cannot load on a 16 GB card (19.7 GB of
//! weights), so device-parity at model scale is not runnable here.
//!
//! Contract is llama.cpp-level, not bit-identity: tolerance against
//! `max|reference|`, recorded per half and placement.
//!
//! Run explicitly (needs `--features lab` for `Gpu::init`):
//!
//!   HIPFIRE_MOE_PARITY_FIXTURE=$HOME/.hipfire/models/ornith-1.5-35b-a3b.mq4 \
//!     cargo test --release -p hipfire-arch-qwen35 --features lab --locked \
//!         --test gpu_moe_cpu_parity -- --ignored --test-threads=1 --nocapture

use hipfire_arch_qwen35::qwen35::load::{load_weight_tensor, qwen35_tensor_name_candidates};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::weight_backend::MemoryTarget;
use rdna_compute::{DType, Gpu};
use std::path::Path;
use std::sync::Mutex;

/// Serializes against the other GPU oracles: a multi-GB load on device 0 would
/// OOM or perturb a parallel run.
static GPU_ORACLE_LOCK: Mutex<()> = Mutex::new(());

const FIXTURE_ENV: &str = "HIPFIRE_MOE_PARITY_FIXTURE";
const DEFAULT_FIXTURE: &str = "ornith-1.5-35b-a3b.mq4";
/// Relative tolerance against `max|reference|` (llama.cpp-level contract).
const REL_TOL: f32 = 5e-3;

#[test]
#[ignore = "needs a real MoE fixture and a GPU; run with --ignored --features lab"]
fn gpu_moe_cpu_parity() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = GPU_ORACLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let path = std::env::var(FIXTURE_ENV).unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/.hipfire/models/{DEFAULT_FIXTURE}")
    });
    if !Path::new(&path).is_file() {
        eprintln!("skip: MoE fixture absent ({path}); set {FIXTURE_ENV}");
        return Ok(());
    }

    let hfq = HfqFile::open(Path::new(&path))?;
    let mut gpu = Gpu::init()?;

    // One real qt44 expert pair: gate_up feeds the gate_up kernel, whose
    // silu+rotate output feeds the down kernel — the full routed half.
    let want = ".mlp.experts.0.";
    let find = |frag: &str| {
        hfq.tensor_infos()
            .iter()
            .find(|t| {
                t.quant_type == 44
                    && t.shape.len() == 2
                    && t.name.contains(want)
                    && t.name.contains(frag)
            })
            .or_else(|| {
                hfq.tensor_infos().iter().find(|t| {
                    t.quant_type == 44
                        && t.name.contains(".mlp.experts.0.")
                        && t.name.contains(frag)
                })
            })
            .unwrap_or_else(|| panic!("no qt44 {frag} tensor matching {want}"))
            .name
            .clone()
    };
    let gu_name = find("gate_up_proj.weight");
    let dn_name = find("down_proj.weight");
    let gu_info = hfq.find_tensor_info(&gu_name).expect("gate_up info");
    let dn_info = hfq.find_tensor_info(&dn_name).expect("down info");
    let (gu_m, gu_k) = (gu_info.shape[0] as usize, gu_info.shape[1] as usize);
    let (m, k) = (dn_info.shape[0] as usize, dn_info.shape[1] as usize);
    eprintln!("picked {gu_name} gate_up {gu_m}x{gu_k}; {dn_name} down {m}x{k}");
    assert_eq!(gu_m, 2 * k, "fused gate_up rows must be 2*mi");
    assert_eq!(gu_k, m, "gate_up k (dim) must equal down m (dim)");

    let load = |gpu: &mut Gpu, name: &str, m: usize, k: usize, target: MemoryTarget| {
        load_weight_tensor(&hfq, gpu, name, m, k, qwen35_tensor_name_candidates, target)
    };

    // Plain normalized activation (rotated once for gate_up, matching the
    // launcher's InputBasis stage).
    let mut rng: u32 = 0x1234_5678;
    let x_raw: Vec<f32> = (0..m)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            (rng as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    let mut x_rot = x_raw.clone();
    hipfire_cpu::quant::rotate_x(&mut x_rot);

    // CPU reference for the FULL expert: gate_up GEMV + silu + hidden-rotate +
    // down GEMV (mirrors `run_experts` for one expert, weight 1.0).
    let cpu_expert = |gu_bytes: &[u8], dn_bytes: &[u8]| -> Vec<f32> {
        let q = hipfire_cpu::quant::CpuQuant::Mq4G256V2;
        let mi = k;
        let mut gate_up_out = vec![0.0f32; 2 * mi];
        let mut hidden = vec![0.0f32; mi];
        let mut down_out = vec![0.0f32; m];
        hipfire_cpu::gemv::gemv(q, gu_bytes, 2 * mi, m, &x_rot, &mut gate_up_out);
        hipfire_cpu::epilogue::silu_mul(&gate_up_out, &mut hidden);
        hipfire_cpu::quant::rotate_x(&mut hidden);
        hipfire_cpu::gemv::gemv(q, dn_bytes, m, mi, &hidden, &mut down_out);
        down_out
    };

    for target in [MemoryTarget::Device, MemoryTarget::HostMapped] {
        let label = match target {
            MemoryTarget::Device => "device",
            MemoryTarget::HostMapped => "host",
        };
        let gu = load(&mut gpu, &gu_name, gu_m, gu_k, target)?;
        let dn = load(&mut gpu, &dn_name, m, k, target)?;
        assert_eq!(gu.gpu_dtype, DType::MQ4G256V2);
        assert_eq!(dn.gpu_dtype, DType::MQ4G256V2);
        assert_eq!(
            gpu.host_located(&gu.buf),
            matches!(target, MemoryTarget::HostMapped),
            "{label} gate_up locality wrong"
        );
        let (gu_bytes, dn_bytes) = match target {
            MemoryTarget::Device => (
                gpu.download_raw_bytes(&gu.buf)?,
                gpu.download_raw_bytes(&dn.buf)?,
            ),
            MemoryTarget::HostMapped => (
                gpu.host_bytes(&gu.buf)
                    .expect("host gate_up bytes")
                    .to_vec(),
                gpu.host_bytes(&dn.buf).expect("host down bytes").to_vec(),
            ),
        };
        // Pointer tables from `sub_offset(0, stride)` views exactly as the
        // loader builds them (blob + slot*stride with slot 0).
        let gu_view = gu.buf.sub_offset(0, gu.buf.byte_size());
        let dn_view = dn.buf.sub_offset(0, dn.buf.byte_size());
        let gu_ptrs = gpu.upload_raw(&(gu_view.buf.as_ptr() as u64).to_ne_bytes(), &[8])?;
        let dn_ptrs = gpu.upload_raw(&(dn_view.buf.as_ptr() as u64).to_ne_bytes(), &[8])?;
        // k8-specialized kernel: grid.y is always 8 ranks; topk must hold 8
        // entries and y_gate/y_up [8 x mi] each; rank 0 is the comparison.
        let topk = gpu.upload_raw(&[0u8; 32], &[32])?;
        let x = gpu.upload_f32(&x_rot, &[m])?;

        let mut cpu_gu = vec![0.0f32; gu_m];
        hipfire_cpu::gemv::gemv(
            hipfire_cpu::quant::CpuQuant::Mq4G256V2,
            &gu_bytes,
            gu_m,
            m,
            &x_rot,
            &mut cpu_gu,
        );
        let (cpu_g, cpu_u) = cpu_gu.split_at(k);

        let yg = gpu.zeros(&[8 * k], DType::F32)?;
        let yu = gpu.zeros(&[8 * k], DType::F32)?;
        gpu.gemv_mq4g256v2_moe_gate_up_k8_indexed(&gu_ptrs, &topk, &x, &yg, &yu, gu_m, m)?;
        let (g_full, u_full) = (gpu.download_f32(&yg)?, gpu.download_f32(&yu)?);
        let (g, u) = (g_full[..k].to_vec(), u_full[..k].to_vec());
        let gs = cpu_gu.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
        let wgg = g
            .iter()
            .zip(cpu_g.iter())
            .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
        let wgu = g
            .iter()
            .zip(cpu_u.iter())
            .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
        let wug = u
            .iter()
            .zip(cpu_g.iter())
            .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
        let wuu = u
            .iter()
            .zip(cpu_u.iter())
            .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
        // Pass requires the CORRECT pairing on both halves; cross terms are
        // diagnostics only (a swapped split must FAIL, not min-pass).
        let wg = wgg.max(wuu);
        eprintln!(
            "  [{label}] gate/up split: g-vs-g {wgg:.3e} g-vs-u {wgu:.3e} u-vs-g {wug:.3e} u-vs-u {wuu:.3e}"
        );

        // down half from a CPU-derived rot, so a gate_up mismatch cannot leak in.
        let mut cpu_hidden = vec![0.0f32; k];
        hipfire_cpu::epilogue::silu_mul(&cpu_gu, &mut cpu_hidden);
        hipfire_cpu::quant::rotate_x(&mut cpu_hidden);
        let rot = gpu.upload_f32(&cpu_hidden, &[k])?;
        let out = gpu.zeros(&[m], DType::F32)?;
        gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
            &dn_ptrs, &topk, &rot, &out, m, k, 1, 1,
        )?;
        let kernel = gpu.download_f32(&out)?;
        let expect = cpu_expert(&gu_bytes, &dn_bytes);
        let scale = expect.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
        let worst = kernel
            .iter()
            .zip(&expect)
            .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
        eprintln!(
            "qt44 MoE [{label}] gate_up max|diff|={wg:.6e} (rel {:.3e}) down max|diff|={worst:.6e} (rel {:.3e})",
            wg / gs,
            worst / scale
        );
        assert!(
            wg <= gs * REL_TOL,
            "qt44 MoE gate_up kernel [{label}] disagrees with CPU: {wg:.6e} rel {:.3e}",
            wg / gs
        );
        assert!(
            worst <= scale * REL_TOL,
            "qt44 MoE down kernel [{label}] disagrees with CPU: {worst:.6e} rel {:.3e}",
            worst / scale
        );
        eprintln!(
            "MQ4V2_MOE_KERNEL_PARITY PASS (qt44 [{label}], gate_up {gu_m}x{m} + down {m}x{k})"
        );
    }
    Ok(())
}

const GRADED_FIXTURE_ENV: &str = "HIPFIRE_MOE_GRADED_PARITY_FIXTURE";
const GRADED_DEFAULT_FIXTURE: &str = "qwen3.6-35b-a3b.mq4p";

/// Per-tag parity for the graded host layout: one real expert per tag bucket
/// (MQ6 hot / MQ4 mid / MQ3L cold on the default fixture), driven through the
/// merged tag-branched kernels against `hipfire_cpu` over the same bytes, on
/// both placements. Same ignore gate as the qt44 test above.
#[test]
#[ignore = "needs a real graded MoE fixture and a GPU; run with --ignored --features lab"]
fn graded_moe_tag_parity() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = GPU_ORACLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let path = std::env::var(GRADED_FIXTURE_ENV).unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/.hipfire/models/{GRADED_DEFAULT_FIXTURE}")
    });
    if !Path::new(&path).is_file() {
        eprintln!("skip: graded MoE fixture absent ({path}); set {GRADED_FIXTURE_ENV}");
        return Ok(());
    }
    let hfq = HfqFile::open(Path::new(&path))?;
    let mut gpu = Gpu::init()?;
    // One expert per tag bucket, by quant-type pair: (15,15)=MQ6 tag 0,
    // (13,13)=MQ4 tag 2, (20,20)=MQ3L tag 3. Expert 0's pair is whatever the
    // fixture holds there; the scan below finds the first expert id carrying
    // each pair so the test does not hard-code the tier map.
    let mut bucket_expert: std::collections::BTreeMap<(u8, u8), usize> =
        std::collections::BTreeMap::new();
    for t in hfq.tensor_infos() {
        if !t.name.contains(".mlp.experts.") || !t.name.contains("gate_up_proj.weight") {
            continue;
        }
        let Some(eid) = t
            .name
            .split(".mlp.experts.")
            .nth(1)
            .and_then(|s| s.split('.').next()?.parse::<usize>().ok())
        else {
            continue;
        };
        let down_bare = t.name.replace("gate_up_proj.weight", "down_proj.weight");
        let down_name = qwen35_tensor_name_candidates(&down_bare)
            .into_iter()
            .find(|n| hfq.find_tensor_info(n).is_some());
        let (Some(down_name), Some(gu_info)) = (down_name, hfq.find_tensor_info(&t.name)) else {
            continue;
        };
        let dn_info = hfq.find_tensor_info(&down_name).expect("down info");
        bucket_expert
            .entry((gu_info.quant_type, dn_info.quant_type))
            .or_insert(eid);
    }
    eprintln!("graded buckets (gate_qt, down_qt) -> expert: {bucket_expert:?}");
    assert!(
        !bucket_expert.is_empty(),
        "no graded expert pairs found in {path}"
    );
    let load = |gpu: &mut Gpu, name: &str, m: usize, k: usize, target: MemoryTarget| {
        load_weight_tensor(&hfq, gpu, name, m, k, qwen35_tensor_name_candidates, target)
    };
    let mut rng: u32 = 0x1234_5678;
    let next_f32 = |rng: &mut u32| {
        *rng ^= *rng << 13;
        *rng ^= *rng >> 17;
        *rng ^= *rng << 5;
        (*rng as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    for ((gate_qt, down_qt), eid) in &bucket_expert {
        let gu_bare = format!(".mlp.experts.{eid}.gate_up_proj.weight");
        let dn_bare = format!(".mlp.experts.{eid}.down_proj.weight");
        let gu_name = hfq
            .tensor_infos()
            .iter()
            .find(|t| t.name.contains(&gu_bare))
            .expect("gate_up tensor")
            .name
            .clone();
        let dn_name = hfq
            .tensor_infos()
            .iter()
            .find(|t| t.name.contains(&dn_bare))
            .expect("down tensor")
            .name
            .clone();
        let gu_info = hfq.find_tensor_info(&gu_name).expect("gate_up info");
        let dn_info = hfq.find_tensor_info(&dn_name).expect("down info");
        let (gu_m, gu_k) = (gu_info.shape[0] as usize, gu_info.shape[1] as usize);
        let (m, k) = (dn_info.shape[0] as usize, dn_info.shape[1] as usize);
        assert_eq!(gu_m, 2 * k, "fused gate_up rows must be 2*mi");
        assert_eq!(gu_k, m, "gate_up k (dim) must equal down m (dim)");
        let x_raw: Vec<f32> = (0..m).map(|_| next_f32(&mut rng)).collect();
        let mut x_rot = x_raw.clone();
        hipfire_cpu::quant::rotate_x(&mut x_rot);
        // CPU decoder per gate dtype; skip the bucket when hipfire-cpu has no
        // decoder rather than failing a coverage gap as a parity failure.
        let cpu_quant = |qt: u8| -> Option<hipfire_cpu::quant::CpuQuant> {
            match qt {
                13 => Some(hipfire_cpu::quant::CpuQuant::Mq4G256),
                15 => Some(hipfire_cpu::quant::CpuQuant::Mq6G256),
                20 => Some(hipfire_cpu::quant::CpuQuant::Mq3G256Lloyd),
                _ => None,
            }
        };
        let (Some(gu_q), Some(dn_q)) = (cpu_quant(*gate_qt), cpu_quant(*down_qt)) else {
            eprintln!("skip bucket ({gate_qt},{down_qt}) expert {eid}: no hipfire-cpu decoder");
            continue;
        };
        let cpu_expert = |gu_bytes: &[u8], dn_bytes: &[u8]| -> Vec<f32> {
            let mi = k;
            let mut gate_up_out = vec![0.0f32; 2 * mi];
            let mut hidden = vec![0.0f32; mi];
            let mut down_out = vec![0.0f32; m];
            hipfire_cpu::gemv::gemv(gu_q, gu_bytes, 2 * mi, m, &x_rot, &mut gate_up_out);
            hipfire_cpu::epilogue::silu_mul(&gate_up_out, &mut hidden);
            hipfire_cpu::quant::rotate_x(&mut hidden);
            hipfire_cpu::gemv::gemv(dn_q, dn_bytes, m, mi, &hidden, &mut down_out);
            down_out
        };
        for target in [MemoryTarget::Device, MemoryTarget::HostMapped] {
            let label = match target {
                MemoryTarget::Device => "device",
                MemoryTarget::HostMapped => "host",
            };
            let gu = load(&mut gpu, &gu_name, gu_m, gu_k, target)?;
            let dn = load(&mut gpu, &dn_name, m, k, target)?;
            assert_eq!(
                gpu.host_located(&gu.buf),
                matches!(target, MemoryTarget::HostMapped),
                "[{label}] gate_up locality wrong"
            );
            let (gu_bytes, dn_bytes) = match target {
                MemoryTarget::Device => (
                    gpu.download_raw_bytes(&gu.buf)?,
                    gpu.download_raw_bytes(&dn.buf)?,
                ),
                MemoryTarget::HostMapped => (
                    gpu.host_bytes(&gu.buf)
                        .expect("host gate_up bytes")
                        .to_vec(),
                    gpu.host_bytes(&dn.buf).expect("host down bytes").to_vec(),
                ),
            };
            // Drive the merged tag-branched kernels with a one-entry tag table.
            // topk=expert id 0 (single expert), grid.y is k8-specialized so the
            // table holds 8 ranks with rank 0 live.
            let tag = hipfire_arch_qwen35::qwen35::mixed_expert_tag(gu.gpu_dtype, dn.gpu_dtype)
                .map_err(|e| format!("bucket ({gate_qt},{down_qt}): {}", e.message))?;
            let tags = gpu.upload_raw(&[tag], &[1])?;
            let gu_view = gu.buf.sub_offset(0, gu.buf.byte_size());
            let dn_view = dn.buf.sub_offset(0, dn.buf.byte_size());
            let gu_ptrs = gpu.upload_raw(&(gu_view.buf.as_ptr() as u64).to_ne_bytes(), &[8])?;
            let dn_ptrs = gpu.upload_raw(&(dn_view.buf.as_ptr() as u64).to_ne_bytes(), &[8])?;
            let topk = gpu.upload_raw(&[0u8; 32], &[32])?;
            let x = gpu.upload_f32(&x_rot, &[m])?;
            let mut cpu_gu = vec![0.0f32; gu_m];
            hipfire_cpu::gemv::gemv(gu_q, &gu_bytes, gu_m, m, &x_rot, &mut cpu_gu);
            let yg = gpu.zeros(&[8 * k], DType::F32)?;
            let yu = gpu.zeros(&[8 * k], DType::F32)?;
            gpu.gemv_mixed_moe_gate_up_k8_indexed_batched(
                &gu_ptrs, &tags, &topk, &x, &yg, &yu, gu_m, m, 8, 1,
            )?;
            let (g_full, u_full) = (gpu.download_f32(&yg)?, gpu.download_f32(&yu)?);
            let (g, u) = (g_full[..k].to_vec(), u_full[..k].to_vec());
            let (cpu_g, cpu_u) = cpu_gu.split_at(k);
            let gs = cpu_gu.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
            let wgg = g
                .iter()
                .zip(cpu_g.iter())
                .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
            let wuu = u
                .iter()
                .zip(cpu_u.iter())
                .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
            let wg = wgg.max(wuu);
            let mut cpu_hidden = vec![0.0f32; k];
            hipfire_cpu::epilogue::silu_mul(&cpu_gu, &mut cpu_hidden);
            hipfire_cpu::quant::rotate_x(&mut cpu_hidden);
            let rot = gpu.upload_f32(&cpu_hidden, &[k])?;
            let out = gpu.zeros(&[m], DType::F32)?;
            gpu.gemv_mixed_moe_down_k8_indexed_batched_expanded(
                &dn_ptrs, &tags, &topk, &rot, &out, m, k, 1, 1,
            )?;
            let kernel = gpu.download_f32(&out)?;
            let expect = cpu_expert(&gu_bytes, &dn_bytes);
            let scale = expect.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
            let worst = kernel
                .iter()
                .zip(&expect)
                .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
            eprintln!(
                "graded ({gate_qt},{down_qt}) tag {tag} expert {eid} [{label}]: gate_up rel {:.3e}, down rel {:.3e}",
                wg / gs,
                worst / scale
            );
            assert!(
                wg <= gs * REL_TOL,
                "graded gate_up kernel ({gate_qt},{down_qt}) [{label}] disagrees: rel {:.3e}",
                wg / gs
            );
            assert!(
                worst <= scale * REL_TOL,
                "graded down kernel ({gate_qt},{down_qt}) [{label}] disagrees: rel {:.3e}",
                worst / scale
            );
            eprintln!(
                "GRADED_TAG_PARITY PASS (qt{gate_qt}/qt{down_qt} tag {tag} [{label}], gate_up {gu_m}x{m} + down {m}x{k})"
            );
        }
    }
    Ok(())
}

/// Chained multi-expert batch-4 parity for the graded Path-1 route: gate_up
/// (mixed indexed, k_top=8, batch=4) -> GPU fused silu+rotate -> mixed down,
/// over 8 real host-mapped experts with mixed tags, against a per-slot CPU
/// reference. The single-expert tests above cannot see index, layout, or
/// chaining faults (batch=1, expert 0, CPU-fed rot); this is the
/// pointer/staging discriminator. Missing fixture is a hard error, never a
/// silent skip.
#[test]
#[ignore = "needs a real graded MoE fixture and a GPU; run with --ignored --features lab"]
fn chained_graded_moe_batch4_parity() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = GPU_ORACLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let path = std::env::var(GRADED_FIXTURE_ENV).unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/.hipfire/models/{GRADED_DEFAULT_FIXTURE}")
    });
    if !Path::new(&path).is_file() {
        return Err(format!("graded fixture absent ({path}); refusing silent skip").into());
    }
    let hfq = HfqFile::open(Path::new(&path))?;
    let mut gpu = Gpu::init()?;
    const N: usize = 4;
    const KTOP: usize = 8;
    // Prefer layer 3 (the handoff's divergence layer), fall back to layer 0.
    // Need >=8 decodable experts spanning >=2 (gate_qt, down_qt) buckets.
    let mut picked_layer: Option<usize> = None;
    let mut buckets: std::collections::BTreeMap<(u8, u8), Vec<(usize, String, String)>> =
        std::collections::BTreeMap::new();
    for l in [3usize, 0] {
        let needle = format!(".layers.{l}.mlp.experts.");
        let mut b: std::collections::BTreeMap<(u8, u8), Vec<(usize, String, String)>> =
            std::collections::BTreeMap::new();
        for t in hfq.tensor_infos() {
            if !t.name.contains(&needle) || !t.name.contains("gate_up_proj.weight") {
                continue;
            }
            let Some(eid) = t
                .name
                .split(".mlp.experts.")
                .nth(1)
                .and_then(|s| s.split('.').next()?.parse::<usize>().ok())
            else {
                continue;
            };
            let down_bare = t.name.replace("gate_up_proj.weight", "down_proj.weight");
            let down_name = qwen35_tensor_name_candidates(&down_bare)
                .into_iter()
                .find(|n| hfq.find_tensor_info(n).is_some());
            let (Some(down_name), Some(gu_info)) =
                (down_name, hfq.find_tensor_info(&t.name))
            else {
                continue;
            };
            let dn_info = hfq.find_tensor_info(&down_name).expect("down info");
            if ![13u8, 15, 20].contains(&gu_info.quant_type)
                || ![13u8, 15, 20].contains(&dn_info.quant_type)
            {
                continue;
            }
            b.entry((gu_info.quant_type, dn_info.quant_type))
                .or_default()
                .push((eid, t.name.clone(), down_name));
        }
        let total: usize = b.values().map(|v| v.len()).sum();
        if total >= KTOP && b.len() >= 2 {
            picked_layer = Some(l);
            buckets = b;
            break;
        }
    }
    let l = picked_layer.ok_or("no layer with >=8 decodable experts across >=2 buckets")?;
    eprintln!("chained parity layer {l} buckets: {buckets:?}");
    // Round-robin across buckets so the 8 slots span distinct tags.
    let key_list: Vec<(u8, u8)> = buckets.keys().cloned().collect();
    let mut per_key_idx = vec![0usize; key_list.len()];
    let mut chosen: Vec<(usize, String, String)> = Vec::new();
    while chosen.len() < KTOP {
        let mut moved = false;
        for (ki, kk) in key_list.iter().enumerate() {
            let v = &buckets[kk];
            if per_key_idx[ki] < v.len() && chosen.len() < KTOP {
                chosen.push(v[per_key_idx[ki]].clone());
                per_key_idx[ki] += 1;
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    assert_eq!(chosen.len(), KTOP, "bucket round-robin underfilled");
    let cpu_quant = |qt: u8| -> Option<hipfire_cpu::quant::CpuQuant> {
        match qt {
            13 => Some(hipfire_cpu::quant::CpuQuant::Mq4G256),
            15 => Some(hipfire_cpu::quant::CpuQuant::Mq6G256),
            20 => Some(hipfire_cpu::quant::CpuQuant::Mq3G256Lloyd),
            _ => None,
        }
    };
    struct Loaded {
        gu_bytes: Vec<u8>,
        dn_bytes: Vec<u8>,
        gu_q: hipfire_cpu::quant::CpuQuant,
        dn_q: hipfire_cpu::quant::CpuQuant,
        tag: u8,
        gu_ptr: u64,
        dn_ptr: u64,
    }
    let mut loaded: Vec<Loaded> = Vec::new();
    let (mut dim, mut mi) = (0usize, 0usize);
    // Keep owners alive: pointer tables point into these buffers.
    let mut owners: Vec<(hipfire_runtime::llama::WeightTensor, hipfire_runtime::llama::WeightTensor)> =
        Vec::new();
    for (eid, gu_name, dn_name) in &chosen {
        let gu_info = hfq.find_tensor_info(gu_name).expect("gate_up info");
        let dn_info = hfq.find_tensor_info(dn_name).expect("down info");
        let (gu_m, gu_k) = (gu_info.shape[0] as usize, gu_info.shape[1] as usize);
        let (m, k) = (dn_info.shape[0] as usize, dn_info.shape[1] as usize);
        assert_eq!(gu_m, 2 * k, "fused gate_up rows must be 2*mi");
        assert_eq!(gu_k, m, "gate_up k (dim) must equal down m (dim)");
        if dim == 0 {
            dim = m;
            mi = k;
        }
        assert_eq!((m, k), (dim, mi), "expert {eid}: geometry must match");
        let gu = load_weight_tensor(
            &hfq,
            &mut gpu,
            gu_name,
            gu_m,
            gu_k,
            qwen35_tensor_name_candidates,
            MemoryTarget::HostMapped,
        )?;
        let dn = load_weight_tensor(
            &hfq,
            &mut gpu,
            dn_name,
            m,
            k,
            qwen35_tensor_name_candidates,
            MemoryTarget::HostMapped,
        )?;
        assert!(gpu.host_located(&gu.buf), "expert {eid} gate_up not host-mapped");
        let gu_bytes = gpu.host_bytes(&gu.buf).expect("host gate_up bytes").to_vec();
        let dn_bytes = gpu.host_bytes(&dn.buf).expect("host down bytes").to_vec();
        let tag = hipfire_arch_qwen35::qwen35::mixed_expert_tag(gu.gpu_dtype, dn.gpu_dtype)
            .map_err(|e| format!("expert {eid}: {}", e.message))?;
        let gu_view = gu.buf.sub_offset(0, gu.buf.byte_size());
        let dn_view = dn.buf.sub_offset(0, dn.buf.byte_size());
        loaded.push(Loaded {
            gu_bytes,
            dn_bytes,
            gu_q: cpu_quant(gu_info.quant_type).expect("gate decoder"),
            dn_q: cpu_quant(dn_info.quant_type).expect("down decoder"),
            tag,
            gu_ptr: gu_view.buf.as_ptr() as u64,
            dn_ptr: dn_view.buf.as_ptr() as u64,
        });
        owners.push((gu, dn));
    }
    let tags_seen: std::collections::BTreeSet<u8> = loaded.iter().map(|e| e.tag).collect();
    assert!(tags_seen.len() >= 2, "need >=2 distinct tags, got {tags_seen:?}");
    eprintln!("chained parity tags: {:?}", loaded.iter().map(|e| e.tag).collect::<Vec<_>>());
    let gu_ptr_bytes: Vec<u8> = loaded.iter().flat_map(|e| e.gu_ptr.to_ne_bytes()).collect();
    let dn_ptr_bytes: Vec<u8> = loaded.iter().flat_map(|e| e.dn_ptr.to_ne_bytes()).collect();
    let tag_bytes: Vec<u8> = loaded.iter().map(|e| e.tag).collect();
    let gu_ptrs = gpu.upload_raw(&gu_ptr_bytes, &[gu_ptr_bytes.len()])?;
    let dn_ptrs = gpu.upload_raw(&dn_ptr_bytes, &[dn_ptr_bytes.len()])?;
    let tags = gpu.upload_raw(&tag_bytes, &[tag_bytes.len()])?;
    // topk covers all 8 local experts (row 0 alone is a permutation: 5r mod 8).
    let mut topk_ids: Vec<i32> = Vec::with_capacity(N * KTOP);
    for b in 0..N {
        for r in 0..KTOP {
            topk_ids.push(((b * 3 + r * 5 + b * r) % loaded.len()) as i32);
        }
    }
    assert!(topk_ids.iter().all(|&id| id >= 0 && (id as usize) < loaded.len()));
    let topk_bytes: Vec<u8> = topk_ids.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let topk = gpu.upload_raw(&topk_bytes, &[topk_bytes.len()])?;
    let mut rng: u32 = 0x9e37_79b9;
    let mut next_f32 = || {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        (rng as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let x_host: Vec<f32> = (0..N * dim).map(|_| next_f32()).collect();
    let x = gpu.upload_f32(&x_host, &[N * dim])?;
    let gate = gpu.zeros(&[N * KTOP * mi], DType::F32)?;
    let up = gpu.zeros(&[N * KTOP * mi], DType::F32)?;
    gpu.gemv_mixed_moe_gate_up_k8_indexed_batched(
        &gu_ptrs, &tags, &topk, &x, &gate, &up, 2 * mi, dim, KTOP, N,
    )?;
    let rot = gpu.zeros(&[N * KTOP * mi], DType::F32)?;
    gpu.fused_silu_mul_rotate_mq_batched(&gate, &up, &rot, mi, N * KTOP)?;
    let expanded = gpu.zeros(&[N * KTOP * dim], DType::F32)?;
    gpu.gemv_mixed_moe_down_k8_indexed_batched_expanded(
        &dn_ptrs, &tags, &topk, &rot, &expanded, dim, mi, KTOP, N,
    )?;
    let (g_all, u_all, rot_all, exp_all) = (
        gpu.download_f32(&gate)?,
        gpu.download_f32(&up)?,
        gpu.download_f32(&rot)?,
        gpu.download_f32(&expanded)?,
    );
    let (mut w_gate, mut w_rot, mut w_down) = (0.0f32, 0.0f32, 0.0f32);
    let (mut s_gate, mut s_rot, mut s_down) = (1e-6f32, 1e-6f32, 1e-6f32);
    for b in 0..N {
        let x_row = &x_host[b * dim..(b + 1) * dim];
        for r in 0..KTOP {
            let e = topk_ids[b * KTOP + r] as usize;
            let le = &loaded[e];
            let mut gu_out = vec![0.0f32; 2 * mi];
            hipfire_cpu::gemv::gemv(le.gu_q, &le.gu_bytes, 2 * mi, dim, x_row, &mut gu_out);
            let (cpu_g, cpu_u) = gu_out.split_at(mi);
            let goff = (b * KTOP + r) * mi;
            for i in 0..mi {
                w_gate = w_gate.max((g_all[goff + i] - cpu_g[i]).abs());
                w_gate = w_gate.max((u_all[goff + i] - cpu_u[i]).abs());
                s_gate = s_gate.max(cpu_g[i].abs()).max(cpu_u[i].abs());
            }
            let mut hidden = vec![0.0f32; mi];
            hipfire_cpu::epilogue::silu_mul(&gu_out, &mut hidden);
            hipfire_cpu::quant::rotate_x(&mut hidden);
            for i in 0..mi {
                w_rot = w_rot.max((rot_all[goff + i] - hidden[i]).abs());
                s_rot = s_rot.max(hidden[i].abs());
            }
            let mut d = vec![0.0f32; dim];
            hipfire_cpu::gemv::gemv(le.dn_q, &le.dn_bytes, dim, mi, &hidden, &mut d);
            let eoff = (b * KTOP + r) * dim;
            for i in 0..dim {
                w_down = w_down.max((exp_all[eoff + i] - d[i]).abs());
                s_down = s_down.max(d[i].abs());
            }
        }
    }
    eprintln!(
        "chained batch{N} k{KTOP} layer {l}: gate rel {:.3e}, gpu-rot rel {:.3e}, down rel {:.3e}",
        w_gate / s_gate,
        w_rot / s_rot,
        w_down / s_down
    );
    assert!(w_gate <= s_gate * REL_TOL, "chained gate_up disagrees: rel {:.3e}", w_gate / s_gate);
    assert!(w_rot <= s_rot * REL_TOL, "gpu silu+rotate disagrees: rel {:.3e}", w_rot / s_rot);
    assert!(w_down <= s_down * REL_TOL, "chained down disagrees: rel {:.3e}", w_down / s_down);
    eprintln!("CHAINED_BATCH4_PARITY PASS (layer {l}, tags {tags_seen:?})");
    Ok(())
}
