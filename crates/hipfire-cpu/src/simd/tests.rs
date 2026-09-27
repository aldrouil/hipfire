// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! SIMD dispatch and vector-vs-scalar agreement.

use super::*;
use crate::gemv::{gemv_with_simd, row_bytes};
use crate::quant::CpuQuant;
use crate::testfix;

/// The dispatch predicate is a pure function of `(available, requested)`: a
/// forced request must never turn into an illegal instruction on hardware that
/// lacks the feature, and `None` must follow detection. This is the test that
/// runs (and covers the branch) on non-AVX2 CI runners.
#[test]
fn dispatch_predicate_is_forced_only_where_supported() {
    assert!(use_avx2(true, None), "detected -> on");
    assert!(!use_avx2(false, None), "undetected -> off");
    assert!(!use_avx2(true, Some(false)), "forced off wins over detection");
    assert!(use_avx2(true, Some(true)), "forced on, supported");
    assert!(
        !use_avx2(false, Some(true)),
        "forced on, unsupported: must fall back rather than fault"
    );
}

/// The per-format dispatch is a pure function of `(format, available,
/// requested)`: only the formats with a hand-written kernel are ever
/// vectorised, and a forced `Some(true)` must fall back to scalar — not fault —
/// on hardware without the feature, which is what lets this test cover the
/// decision on any runner.
#[test]
fn row_dot_enabled_is_per_format_and_never_forces_unsupported_hardware() {
    assert_eq!(row_dot_enabled(CpuQuant::Mq4G256, None), avx2_available());
    assert_eq!(
        row_dot_enabled(CpuQuant::Mq3G256V2, None),
        avx2_f16c_available()
    );
    assert_eq!(
        row_dot_enabled(CpuQuant::Mq4G256, Some(true)),
        avx2_available(),
        "forced on, unsupported: fall back rather than fault"
    );
    assert_eq!(
        row_dot_enabled(CpuQuant::Mq3G256V2, Some(true)),
        avx2_f16c_available(),
        "the qt 49 kernel needs F16C on top of AVX2"
    );
    // Forced off wins everywhere, and a format with no kernel stays scalar even
    // when the caller asks for the vector path: silently vectorising one would
    // execute the Mq4 decode on foreign bytes.
    for q in [
        CpuQuant::Mq4G256,
        CpuQuant::Mq3G256V2,
        CpuQuant::Mq2G256V2,
        CpuQuant::Mq4G256V2,
        CpuQuant::Mq3G256,
        CpuQuant::Mq3G256Lloyd,
    ] {
        assert!(!row_dot_enabled(q, Some(false)), "{q:?}: forced off");
    }
    for q in [
        CpuQuant::Mq2G256V2,
        CpuQuant::Mq4G256V2,
        CpuQuant::Mq5G256V2,
        CpuQuant::Mq6G256V2,
        CpuQuant::Mq3G256,
        CpuQuant::Mq3G256Lloyd,
        CpuQuant::Hfq3G256,
    ] {
        assert!(!row_dot_enabled(q, Some(true)), "{q:?}: no kernel to use");
    }
}

/// The shipped fixture headers are exact powers of two (`0.0625`, `-0.25`, …),
/// which makes every decoded value, product and partial sum exactly
/// representable — the two summation orders then agree *bit for bit* and the
/// relative bound below would never be exercised. Real quantized weights carry
/// arbitrary fp16/f32 headers, so this rewrites each group's header with
/// awkward (non-power-of-two) values before the two paths are compared.
fn awkward_headers(q: CpuQuant, packed: &mut [u8], m: usize, k: usize) {
    let (ge, gb) = (q.group_elems(), q.group_bytes());
    let groups = k / ge;
    for row in 0..m {
        for g in 0..groups {
            let at = (row * groups + g) * gb;
            match q {
                CpuQuant::Mq4G256 => {
                    packed[at..at + 4].copy_from_slice(&0.0313f32.to_le_bytes());
                    packed[at + 4..at + 8].copy_from_slice(&(-0.4921f32).to_le_bytes());
                }
                // fp16: 0.031311, -0.122986, 0.270996, -0.088684.
                CpuQuant::Mq3G256V2 => {
                    for (i, bits) in [0x2802u16, 0xafdf, 0x3456, 0xadad].iter().enumerate() {
                        packed[at + 2 * i..at + 2 * i + 2].copy_from_slice(&bits.to_le_bytes());
                    }
                }
                _ => unreachable!("no AVX2 kernel for {q:?}"),
            }
        }
    }
}

/// The AVX2 kernels and the scalar reference are different summations of the
/// same products, so they are compared on a *relative* tolerance, not for
/// equality: this bounds the vector paths' deviation and would catch a decode
/// error (nibble order, 3-bit cross-byte packing, group stride, per-half affine
/// header) by orders of magnitude.
///
/// The fixture gives each group four *distinct* header values, so mixing up the
/// two 128-element halves of a `Mq3G256V2` group — or the header's scale/zero
/// order — is O(1) relative, not a rounding difference.
#[test]
fn avx2_and_scalar_agree_within_tolerance() {
    for (q, available) in [
        (CpuQuant::Mq4G256, avx2_available()),
        (CpuQuant::Mq3G256V2, avx2_f16c_available()),
    ] {
        if !available {
            eprintln!("skip {q:?}: feature absent on this runner");
            continue;
        }
        for (m, k) in [(1usize, 256usize), (3, 512), (2, 4096), (1, 12288)] {
            let mut packed = testfix::weight_bytes(q, m, k);
            awkward_headers(q, &mut packed, m, k);
            let x: Vec<f32> = (0..k)
                .map(|i| ((i as u64 * 2654435761) % 4096) as f32 * 0.001 - 2.0)
                .collect();
            let mut simd = vec![0.0f32; m];
            let mut scalar = vec![0.0f32; m];
            gemv_with_simd(q, &packed, m, k, &x, &mut simd, Some(true));
            gemv_with_simd(q, &packed, m, k, &x, &mut scalar, Some(false));
            for row in 0..m {
                let scale = scalar[row].abs().max(1e-6);
                let rel = (simd[row] - scalar[row]).abs() / scale;
                assert!(
                    rel <= 1e-5,
                    "{q:?} {m}x{k} row {row}: simd {} vs scalar {} (rel {rel:.3e})",
                    simd[row],
                    scalar[row]
                );
            }
        }
    }
}

/// The forced-scalar path must be *exactly* `gemv`'s historical arithmetic, so
/// the S1 expectation tables and the GPU parity numbers keep meaning what they
/// were measured against.
#[test]
fn forced_scalar_path_is_the_s1_arithmetic() {
    let q = CpuQuant::Mq4G256;
    let (m, k) = (4usize, 512usize);
    let packed = testfix::weight_bytes(q, m, k);
    let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.25 - 32.0).collect();
    let mut forced = vec![0.0f32; m];
    gemv_with_simd(q, &packed, m, k, &x, &mut forced, Some(false));
    // Hand-rolled over the same decode, in the same order.
    let ge = q.group_elems();
    let gb = q.group_bytes();
    let rb = row_bytes(q, k);
    for (row, got) in forced.iter().enumerate() {
        let mut acc = 0.0f32;
        for g in 0..k / ge {
            let mut codes = vec![0.0f32; ge];
            crate::quant::decode_group_codes(q, &packed[row * rb + g * gb..], &mut codes);
            let mut partial = 0.0f32;
            for i in 0..ge {
                partial += codes[i] * x[g * ge + i];
            }
            acc += partial;
        }
        assert_eq!(*got, acc, "row {row}");
    }
}
