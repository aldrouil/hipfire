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

/// The AVX2 kernel and the scalar reference are two different summations of the
/// same products, so they are compared on a *relative* tolerance, not for
/// equality: this bounds the vector path's deviation and would catch a decode
/// error (nibble order, group stride, affine header) by orders of magnitude.
#[test]
fn avx2_and_scalar_agree_within_tolerance() {
    if !avx2_available() {
        eprintln!("skip: no AVX2 on this runner (dispatch predicate still covered)");
        return;
    }
    let q = CpuQuant::Mq4G256;
    for (m, k) in [(1usize, 256usize), (3, 512), (2, 4096)] {
        let packed = testfix::weight_bytes(q, m, k);
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
                "{m}x{k} row {row}: simd {} vs scalar {} (rel {rel:.3e})",
                simd[row],
                scalar[row]
            );
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
