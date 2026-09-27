// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! AVX2 + FMA row dots for the formats whose bytes the CPU offload path spills.
//!
//! `Mq4G256` (nibbles, f32 group header) and `Mq3G256V2` (3-bit cross-byte
//! packs, per-128 fp16 scale/zero) — see each kernel's own doc for its layout.
//!
//! ## `Mq4G256`
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

/// Shifts that pull chunk `c`'s eight 3-bit codes out of one 32-bit load.
///
/// The load is four bytes at `3c + 7`, i.e. the chunk's three bytes prefixed by
/// the byte before them: with `V = (load >> 8)` being the chunk's 24 bits
/// little-endian, code `j` is `(V >> 3j) & 7`. Folding the `>> 8` into the shift
/// vector makes that one `vpsrlvd` instead of a scalar shift plus a broadcast,
/// and lets the compiler fold the load into the broadcast.
const CODE3_SHIFTS: [i32; 8] = [8, 11, 14, 17, 20, 23, 26, 29];

/// One 8-element chunk of a `Mq3G256V2` group: `dot += Σ_j code_j · x_j`,
/// `sum += Σ_j x_j`.
///
/// The codes come out of one broadcast + one variable shift + one mask rather
/// than a byte at a time: the three bytes are already a contiguous 24-bit
/// little-endian field (`crate::quant::code3`, which this reproduces), so all
/// eight codes are its 3-bit lanes.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn chunk(
    gptr: *const u8,
    xg: *const f32,
    idx: usize,
    shifts: __m256i,
    mask: __m256i,
    dot: __m256,
    sum: __m256,
) -> (__m256, __m256) {
    // Bytes `3*idx+7 .. 3*idx+11`: the chunk's three payload bytes plus one
    // preceding byte the shift discards. In bounds for every chunk — the last
    // is 100..104 of a 104-byte group.
    let w = (gptr.add(7 + 3 * idx) as *const u32).read_unaligned() as i32;
    let codes = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(w), shifts), mask);
    let xv = _mm256_loadu_ps(xg.add(idx * 8));
    (
        _mm256_fmadd_ps(_mm256_cvtepi32_ps(codes), xv, dot),
        _mm256_add_ps(sum, xv),
    )
}

/// One 256-element `Mq3G256V2` group: 8 B of `[s0 z0 s1 z1]` fp16 header (each
/// pair covering 128 elements) then 96 B of 3-bit codes, 8 per 3 bytes.
///
/// Written as `s·Σ(c·x) + z·Σx` per half — the same restructure the `Mq4G256`
/// group dot uses, and the reason a half's affine header costs two vector FMAs
/// at the end rather than an `fma` per element. Not bit-identical to the scalar
/// path; `simd::tests` bounds it relative.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn mq3_group_dot(gptr: *const u8, xg: *const f32) -> f32 {
    // The whole header widens with one F16C convert: lanes are `s0 z0 s1 z1`.
    let hdr = _mm_cvtph_ps(_mm_loadl_epi64(gptr as *const __m128i));
    let mut h = [0.0f32; 4];
    _mm_storeu_ps(h.as_mut_ptr(), hdr);
    let shifts = _mm256_loadu_si256(CODE3_SHIFTS.as_ptr() as *const __m256i);
    let mask = _mm256_set1_epi32(7);
    let zero_v = _mm256_setzero_ps();
    let (mut dot0, mut sum0) = (zero_v, zero_v);
    for idx in 0..16 {
        (dot0, sum0) = chunk(gptr, xg, idx, shifts, mask, dot0, sum0);
    }
    let (mut dot1, mut sum1) = (zero_v, zero_v);
    for idx in 16..32 {
        (dot1, sum1) = chunk(gptr, xg, idx, shifts, mask, dot1, sum1);
    }
    (h[0] * hsum256(dot0) + h[1] * hsum256(sum0)) + (h[2] * hsum256(dot1) + h[3] * hsum256(sum1))
}

/// `Σ_j W[row][j] · x[j]` for a `Mq3G256V2` row.
///
/// # Safety
///
/// Requires AVX2 + FMA + F16C on the running CPU, `k % 256 == 0`, at least
/// `(k / 256) * 104` readable bytes at `row`, and `k` readable `f32` at `x`.
/// Callers reach this through [`super::mq3g256v2_row_dot_avx2`], which checks
/// the CPU features first.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn mq3g256v2_row_dot(row: *const u8, k: usize, x: *const f32) -> f32 {
    let groups = k / 256;
    let mut acc = 0.0f32;
    for g in 0..groups {
        acc += mq3_group_dot(row.add(g * 104), x.add(g * 256));
    }
    acc
}
