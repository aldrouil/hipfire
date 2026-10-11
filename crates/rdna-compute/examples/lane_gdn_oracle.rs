// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte-exactness oracle of the lane-batched DeltaNet (GDN) verify preamble +
//! recurrence of the DFlash continuous-batching trunk (exact gfx1201) against
//! today's per-lane launch sequence.
//!
//! 27B shapes (n_v_heads 48, n_key_heads 16, head_dim 128 -> k_dim 2048,
//! v_dim 6144, qkv_dim 10240); lanes in {1, 3, 8} x n in {2, 9, 16} (every
//! lane the same n, as the route requires) x EF residual {present, absent}.
//! All lanes live in ONE shared row scratch (lane i owns rows
//! `[i * n, (i + 1) * n)`) with random pre-activation qkv/alpha/beta rows,
//! random conv rings, random Q8 state (i8 codes, small positive scales) and a
//! random f16 EF residual.
//!
//! - arm A (today): per lane, on that lane's row views:
//!   `fused_sigmoid_alpha_gate_f32_batched`, `conv1d_silu_split_f32_n`,
//!   `fused_qk_l2_norm_scale_interleave_f32_batched`,
//!   `gated_delta_net_q8_batch_seq`;
//! - arm B (lane-batched): one `fused_sigmoid_alpha_gate_f32_batched` and one
//!   `fused_qk_l2_norm_scale_interleave_f32_batched` over all rows,
//!   `Gpu::conv1d_silu_split_f32_lanes`, and one `gated_delta_net_q8_fast_ml`
//!   over a device `GdnLayerTable` (one row per lane).
//!
//! Both arms start from identical buffers and the same stochastic-rounding
//! frame base (`gdn_requant_frame_checkpoint` / `restore_...`). Every shared
//! row buffer (qkv, alpha, beta, q_raw, k_raw, v, q, k, attn_out; output
//! buffers pre-filled with a sentinel so a missed store shows) and every
//! per-lane state (conv ring, Q8 state, scales, EF residual) is compared as
//! bytes. Prints PASS/FAIL per case; exits nonzero on any mismatch.
//!
//! `cargo run --release -p rdna-compute --features lab,deltanet --example lane_gdn_oracle`

use rdna_compute::dflash_gdn_replay::{table_bytes, GdnLayerTable};
use rdna_compute::norm::{gdn_requant_frame_checkpoint, restore_gdn_requant_frame_checkpoint};
use rdna_compute::{DType, Gpu, GpuTensor};

const NV: usize = 48;
const NK: usize = 16;
const HD: usize = 128;
const K_DIM: usize = NK * HD;
const V_DIM: usize = NV * HD;
const QKV_DIM: usize = 2 * K_DIM + V_DIM;
const EPS: f32 = 1e-6;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
    /// Uniform in [-1, 1).
    fn sym(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
}

/// Round-to-nearest-even f16 bits of a finite value with |v| in the normal
/// f16 range (callers keep magnitudes in [2^-13, 1]).
fn f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    let mut m10 = mant >> 13;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && m10 & 1 == 1) {
        m10 += 1;
    }
    if m10 == 0x400 {
        m10 = 0;
        exp += 1;
    }
    assert!((1..=30).contains(&exp), "{v} is not a normal f16");
    sign | ((exp as u16) << 10) | m10 as u16
}

/// Host-side initial contents shared by both arms.
struct Host {
    qkv: Vec<f32>,
    alpha: Vec<f32>,
    beta: Vec<f32>,
    conv: Vec<Vec<f32>>,
    s_q8: Vec<Vec<u8>>,
    scales: Vec<Vec<f32>>,
    ef: Vec<Vec<u8>>,
}

fn f32s(rng: &mut Rng, n: usize, amp: f32) -> Vec<f32> {
    (0..n).map(|_| rng.sym() * amp).collect()
}

fn make_host(rng: &mut Rng, lanes: usize, total: usize) -> Host {
    let qkv = f32s(rng, total * QKV_DIM, 1.5);
    let alpha = f32s(rng, total * NV, 2.0);
    let beta = f32s(rng, total * NV, 2.0);
    let conv = (0..lanes).map(|_| f32s(rng, QKV_DIM * 3, 1.0)).collect();
    let s_q8 = (0..lanes)
        .map(|_| (0..NV * HD * HD).map(|_| ((rng.next() % 255) as i32 - 127) as i8 as u8).collect())
        .collect();
    let scales = (0..lanes)
        .map(|_| (0..NV * HD).map(|_| 0.001 + rng.unit() * 0.02).collect())
        .collect();
    let ef = (0..lanes)
        .map(|_| {
            let mut b = Vec::with_capacity(NV * HD * HD * 2);
            for _ in 0..NV * HD * HD {
                let mut v = rng.sym() * 0.01;
                if v.abs() < 2f32.powi(-13) {
                    v = 0.001;
                }
                b.extend_from_slice(&f16_bits(v).to_le_bytes());
            }
            b
        })
        .collect();
    Host { qkv, alpha, beta, conv, s_q8, scales, ef }
}

/// Device buffers of one arm.
struct Arm {
    qkv: GpuTensor,
    alpha: GpuTensor,
    beta: GpuTensor,
    q_raw: GpuTensor,
    k_raw: GpuTensor,
    v: GpuTensor,
    q: GpuTensor,
    k: GpuTensor,
    attn: GpuTensor,
    conv: Vec<GpuTensor>,
    s_q8: Vec<GpuTensor>,
    scales: Vec<GpuTensor>,
    ef: Vec<Option<GpuTensor>>,
}

fn make_arm(gpu: &mut Gpu, h: &Host, total: usize, with_ef: bool) -> Arm {
    let sentinel = |gpu: &mut Gpu, w: usize| {
        gpu.upload_f32(&vec![f32::from_bits(0x7fc0_dead); total * w], &[total * w]).expect("alloc sentinel")
    };
    Arm {
        qkv: gpu.upload_f32(&h.qkv, &[h.qkv.len()]).expect("qkv"),
        alpha: gpu.upload_f32(&h.alpha, &[h.alpha.len()]).expect("alpha"),
        beta: gpu.upload_f32(&h.beta, &[h.beta.len()]).expect("beta"),
        q_raw: sentinel(gpu, K_DIM),
        k_raw: sentinel(gpu, K_DIM),
        v: sentinel(gpu, V_DIM),
        q: sentinel(gpu, V_DIM),
        k: sentinel(gpu, V_DIM),
        attn: sentinel(gpu, V_DIM),
        conv: h.conv.iter().map(|c| gpu.upload_f32(c, &[c.len()]).expect("conv")).collect(),
        s_q8: h.s_q8.iter().map(|s| gpu.upload_raw(s, &[s.len()]).expect("s_q8")).collect(),
        scales: h.scales.iter().map(|s| gpu.upload_f32(s, &[s.len()]).expect("scales")).collect(),
        ef: h
            .ef
            .iter()
            .map(|e| with_ef.then(|| gpu.upload_raw(e, &[e.len()]).expect("ef")))
            .collect(),
    }
}

fn snapshot(gpu: &Gpu, a: &Arm) -> Vec<(String, Vec<u8>)> {
    gpu.hip.device_synchronize().expect("sync before download");
    let mut out = Vec::new();
    let mut put = |name: String, t: &GpuTensor| out.push((name, gpu.download_raw_bytes(t).expect("download")));
    for (name, t) in [
        ("qkv", &a.qkv),
        ("alpha", &a.alpha),
        ("beta", &a.beta),
        ("q_raw", &a.q_raw),
        ("k_raw", &a.k_raw),
        ("v", &a.v),
        ("q", &a.q),
        ("k", &a.k),
        ("attn_out", &a.attn),
    ] {
        put(name.to_string(), t);
    }
    for i in 0..a.conv.len() {
        put(format!("conv[{i}]"), &a.conv[i]);
        put(format!("s_q8[{i}]"), &a.s_q8[i]);
        put(format!("scales[{i}]"), &a.scales[i]);
        if let Some(e) = &a.ef[i] {
            put(format!("ef[{i}]"), e);
        }
    }
    out
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    if gpu.arch != "gfx1201" {
        eprintln!("lane_gdn_oracle: needs exact gfx1201, got {}", gpu.arch);
        std::process::exit(2);
    }
    if !gpu.lane_gdn_batch_admitted(NV, NK, HD, HD) {
        eprintln!("lane_gdn_oracle: lane_gdn_batch_admitted is false (HIPFIRE_DN_REQUANT_PER_TOKEN set?)");
        std::process::exit(2);
    }
    println!("arch={} lane_gdn_batch_admitted=true", gpu.arch);

    let mut rng = Rng(0x6D1E_47A3);
    let dt_bias: Vec<f32> = (0..NV).map(|_| rng.sym() * 0.5).collect();
    let a_log: Vec<f32> = (0..NV).map(|_| rng.sym()).collect();
    let conv_w: Vec<f32> = (0..QKV_DIM * 4).map(|_| rng.sym() * 0.5).collect();
    let d_dt_bias = gpu.upload_f32(&dt_bias, &[NV]).expect("dt_bias");
    let d_a_log = gpu.upload_f32(&a_log, &[NV]).expect("a_log");
    let d_conv_w = gpu.upload_f32(&conv_w, &[conv_w.len()]).expect("conv_w");
    let ratio = NV / NK;
    let q_scale = 1.0 / (HD as f32).sqrt();

    let mut failures = 0usize;
    let mut cases = 0usize;
    for &lanes in &[1usize, 3, 8] {
        for &n in &[2usize, 9, 16] {
            for &with_ef in &[true, false] {
                cases += 1;
                let total = lanes * n;
                let host = make_host(&mut rng, lanes, total);
                let a = make_arm(&mut gpu, &host, total, with_ef);
                let b = make_arm(&mut gpu, &host, total, with_ef);
                let table = gpu
                    .zeros(&[(lanes * std::mem::size_of::<GdnLayerTable>()).div_ceil(4)], DType::F32)
                    .expect("table");

                gpu.hip.device_synchronize().expect("sync");
                let frame0 = gdn_requant_frame_checkpoint();

                // Arm A: per-lane sequence on that lane's row views.
                for i in 0..lanes {
                    let r0 = i * n;
                    let vw = |t: &GpuTensor, w: usize| t.sub_offset(r0 * w, n * w);
                    gpu.fused_sigmoid_alpha_gate_f32_batched(
                        &vw(&a.beta, NV), &vw(&a.alpha, NV), &d_dt_bias, &d_a_log, NV, n,
                    )
                    .expect("A sigmoid");
                    gpu.conv1d_silu_split_f32_n(
                        &vw(&a.q_raw, K_DIM), &vw(&a.k_raw, K_DIM), &vw(&a.v, V_DIM), &vw(&a.qkv, QKV_DIM),
                        &d_conv_w, &a.conv[i], K_DIM, V_DIM, n,
                    )
                    .expect("A conv");
                    gpu.fused_qk_l2_norm_scale_interleave_f32_batched(
                        &vw(&a.q_raw, K_DIM), &vw(&a.k_raw, K_DIM), &vw(&a.q, V_DIM), &vw(&a.k, V_DIM),
                        NK, ratio, HD, q_scale, EPS, n,
                    )
                    .expect("A interleave");
                    gpu.gated_delta_net_q8_batch_seq(
                        &vw(&a.q, V_DIM), &vw(&a.k, V_DIM), &vw(&a.v, V_DIM), &vw(&a.alpha, NV),
                        &vw(&a.beta, NV), &a.s_q8[i], &a.scales[i], &vw(&a.attn, V_DIM), n, NV, HD,
                        a.ef[i].as_ref(),
                    )
                    .expect("A gdn");
                }
                let snap_a = snapshot(&gpu, &a);

                // Arm B: lane-batched, from the same frame base.
                restore_gdn_requant_frame_checkpoint(frame0);
                gpu.fused_sigmoid_alpha_gate_f32_batched(&b.beta, &b.alpha, &d_dt_bias, &d_a_log, NV, total)
                    .expect("B sigmoid");
                let lane_args: Vec<(u64, usize, usize)> =
                    (0..lanes).map(|i| (b.conv[i].buf.as_ptr() as u64, i * n, n)).collect();
                gpu.conv1d_silu_split_f32_lanes(
                    &b.q_raw, &b.k_raw, &b.v, &b.qkv, &d_conv_w, &lane_args, K_DIM, V_DIM,
                )
                .expect("B conv");
                gpu.fused_qk_l2_norm_scale_interleave_f32_batched(
                    &b.q_raw, &b.k_raw, &b.q, &b.k, NK, ratio, HD, q_scale, EPS, total,
                )
                .expect("B interleave");
                let ptr = |t: &GpuTensor| t.buf.as_ptr() as u64;
                let rows: Vec<GdnLayerTable> = (0..lanes)
                    .map(|i| {
                        let r0 = i * n;
                        let vw = |t: &GpuTensor, w: usize| t.sub_offset(r0 * w, n * w);
                        GdnLayerTable {
                            q: ptr(&vw(&b.q, V_DIM)),
                            k: ptr(&vw(&b.k, V_DIM)),
                            v: ptr(&vw(&b.v, V_DIM)),
                            gate: ptr(&vw(&b.alpha, NV)),
                            beta: ptr(&vw(&b.beta, NV)),
                            s_q8: ptr(&b.s_q8[i]),
                            s_scales: ptr(&b.scales[i]),
                            output: ptr(&vw(&b.attn, V_DIM)),
                            ef: b.ef[i].as_ref().map_or(0, ptr),
                        }
                    })
                    .collect();
                gpu.upload_gdn_lane_table(&table, table_bytes(&rows)).expect("B table upload");
                gpu.gated_delta_net_q8_fast_ml(table.buf.as_ptr(), lanes, n, NV, HD).expect("B gdn");
                let snap_b = snapshot(&gpu, &b);

                let frames_a_b = gdn_requant_frame_checkpoint();
                let mut bad = Vec::new();
                for ((name, x), (_, y)) in snap_a.iter().zip(&snap_b) {
                    let mism = x.iter().zip(y).filter(|(p, q)| p != q).count() + x.len().abs_diff(y.len());
                    if mism != 0 {
                        bad.push(format!("{name}:{mism}"));
                    }
                }
                // Both arms reserve lanes * n frames from frame0.
                let frames_ok = frames_a_b == frame0 + (lanes * n) as u32;
                let ok = bad.is_empty() && frames_ok;
                if !ok {
                    failures += 1;
                }
                println!(
                    "{} lanes={lanes} n={n:<2} ef={with_ef:<5} frames_ok={frames_ok} mismatching_buffers=[{}]",
                    if ok { "PASS" } else { "FAIL" },
                    bad.join(", "),
                );

                // Each case advances the frame base by one arm's worth; the
                // next case's frame0 is read fresh above.
                for t in [a.qkv, a.alpha, a.beta, a.q_raw, a.k_raw, a.v, a.q, a.k, a.attn, b.qkv, b.alpha,
                    b.beta, b.q_raw, b.k_raw, b.v, b.q, b.k, b.attn, table]
                {
                    gpu.free_tensor(t).expect("free");
                }
                for arm in [a.conv, a.s_q8, a.scales, b.conv, b.s_q8, b.scales] {
                    for t in arm {
                        gpu.free_tensor(t).expect("free");
                    }
                }
                for t in a.ef.into_iter().chain(b.ef).flatten() {
                    gpu.free_tensor(t).expect("free");
                }
            }
        }
    }

    println!("cases={cases} failures={failures}");
    if failures != 0 {
        std::process::exit(1);
    }
}
