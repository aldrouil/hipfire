// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! AVX2 + FMA `Mq4G256` row dot.
//!
//! Layout reminder (see `crate::quant`): 136 B per 256-element group —
//! `[f32 scale][f32 zero][128 B of paired nibbles]`, element `i` taking the low
//! nibble of byte `i/2` for even `i` and the high nibble for odd `i`. The
//! activation arrives already FWHT-rotated, so this kernel only decodes and
//! multiplies.
//!
//! Arithmetic: a group's `Σ_i (scale·code_i + zero) · x_i` is evaluated as
//! `scale·Σ_i(code_i·x_i) + zero·Σ_i x_i` — mathematically identical, and the
//! reason a group costs a handful of vector FMAs plus two horizontal sums
//! instead of 256 scalar multiply-adds. It is deliberately *not* bit-identical
//! to the scalar path: `simd::tests` bounds the difference relative, and the
//! offload feature's contract is a tolerance, not device parity.

use core::arch::x86_64::*;

/// One 256-element group: 128 packed nibble bytes = 32 codes per 16-byte block.
///
/// `_mm_unpacklo_epi8` of the low/high nibble planes yields `[c0, c1, c2, …]`
/// in element order (byte `j` contributes its low nibble as `c[2j]` and its high
/// nibble as `c[2j+1]`), so no shuffle of the activation is needed.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn group_dot(gptr: *const u8, xg: *const f32) -> f32 {
    let scale = f32::from_le_bytes([*gptr, *gptr.add(1), *gptr.add(2), *gptr.add(3)]);
    let zero = f32::from_le_bytes([*gptr.add(4), *gptr.add(5), *gptr.add(6), *gptr.add(7)]);
    let mask = _mm_set1_epi8(0x0f);
    let mut acc = _mm256_setzero_ps();
    let mut acc_x = _mm256_setzero_ps();
    for blk in 0..8usize {
        let bytes = _mm_loadu_si128(gptr.add(8 + blk * 16) as *const __m128i);
        let lo = _mm_and_si128(bytes, mask);
        let hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
        let c_lo = _mm_unpacklo_epi8(lo, hi);
        let c_hi = _mm_unpackhi_epi8(lo, hi);
        let xb = xg.add(blk * 32);
        let c0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(c_lo));
        let c1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(c_lo, 8)));
        let c2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(c_hi));
        let c3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(c_hi, 8)));
        let x0 = _mm256_loadu_ps(xb);
        let x1 = _mm256_loadu_ps(xb.add(8));
        let x2 = _mm256_loadu_ps(xb.add(16));
        let x3 = _mm256_loadu_ps(xb.add(24));
        acc = _mm256_fmadd_ps(c0, x0, acc);
        acc = _mm256_fmadd_ps(c1, x1, acc);
        acc = _mm256_fmadd_ps(c2, x2, acc);
        acc = _mm256_fmadd_ps(c3, x3, acc);
        acc_x = _mm256_add_ps(
            acc_x,
            _mm256_add_ps(_mm256_add_ps(x0, x1), _mm256_add_ps(x2, x3)),
        );
    }
    scale * hsum256(acc) + zero * hsum256(acc_x)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum256(v: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

/// `Σ_j W[row][j] · x[j]` for a `Mq4G256` row.
///
/// # Safety
///
/// Requires AVX2 + FMA on the running CPU, `k % 256 == 0`, at least
/// `(k / 256) * 136` readable bytes at `row`, and `k` readable `f32` at `x`.
/// Callers reach this through [`super::mq4g256_row_dot`], which checks the CPU
/// feature first.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn mq4g256_row_dot(row: *const u8, k: usize, x: *const f32) -> f32 {
    let groups = k / 256;
    let mut acc = 0.0f32;
    for g in 0..groups {
        acc += group_dot(row.add(g * 136), x.add(g * 256));
    }
    acc
}
