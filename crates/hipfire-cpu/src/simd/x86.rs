// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! AVX2 + FMA row dots for the formats the CPU offload path spills.
//!
//! Every kernel here computes the same thing — one weight row dotted with a
//! pre-rotated activation — and differs only in how a group's metadata and
//! payload bytes are laid out. So this file is a table of layouts over one
//! decode core rather than a kernel per format:
//!
//! * [`codes8`] — eight `BITS`-wide codes (the chunk of payload one vector
//!   lane group consumes) as `i32` lanes. `BITS` bytes hold eight codes at every
//!   width the formats use, so nibbles, the 2/3/5/6-bit cross-byte packs and
//!   BQ1's sign bits are all the same broadcast-and-shift.
//! * [`codes_dot_sum`] — `(Σ c·x, Σ x)` over a run of chunks.
//! * a per-group header: one `(scale, zero)` from an `f32` pair (the flat HFQ/MQ
//!   families) or an `fp16` pair (qt 45), a per-128 `fp16` quad (the V2 family),
//!   or a `fp16` codebook (the Lloyd family).
//!
//! A group is then `scale·Σ(c·x) + zero·Σx` (affine) or `Σ(cb[c]·x)`
//! (codebook) — mathematically identical to the scalar decode and the reason a
//! group costs a handful of vector FMAs plus two horizontal sums instead of 256
//! scalar multiply-adds. It is deliberately *not* bit-identical to the scalar
//! path: [`super::tests`] bounds the difference relative, and the offload
//! feature's contract is llama.cpp-level coherence, not device parity.
//!
//! Rotation is the caller's business — the activation arrives already
//! FWHT-rotated for the rotated formats (see [`crate::quant`]) — so one kernel
//! serves a rotated format and its byte-identical unrotated sibling.
//!
//! `Mq4G256` (qt 13) keeps its own hand-unrolled nibble path ([`mq4_group_dot`]):
//! it predates this core, decodes 32 codes per 16-byte load instead of 8 per
//! 4-byte load, and its numbers are the ones the recorded qt 13 measurements
//! were taken with, so it is left alone rather than re-derived from the table.

use core::arch::x86_64::*;

/// Horizontal sum of eight `f32` lanes.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum256(v: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

/// The `BITS` bytes at `p` as the little-endian integer they encode: eight
/// `BITS`-wide codes are exactly `BITS` bytes, so a chunk is one packed field.
///
/// Read *exactly* those bytes. A wider load would be one instruction and would
/// run up to three bytes past the group's end on the group's last chunk, which
/// the caller's `row` slice licenses only for the rows that are not the
/// tensor's last.
#[inline]
unsafe fn load_le<const BITS: usize>(p: *const u8) -> u64 {
    match BITS {
        1 => *p as u64,
        2 => (p as *const u16).read_unaligned() as u64,
        // Three bytes from one 32-bit load of the byte *before* the chunk plus
        // the chunk, shifted down. In bounds for every chunk: a group's payload
        // always starts at least one header byte in.
        3 => ((p.sub(1) as *const u32).read_unaligned() >> 8) as u64,
        4 => (p as *const u32).read_unaligned() as u64,
        5 => (p as *const u32).read_unaligned() as u64 | ((*p.add(4) as u64) << 32),
        6 => {
            (p as *const u32).read_unaligned() as u64
                | (((p.add(4) as *const u16).read_unaligned() as u64) << 32)
        }
        _ => unreachable!("no format packs codes wider than six bits"),
    }
}

/// Shift counts `[OFF·BITS, …, (OFF+3)·BITS]` for the 64-bit-lane path.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn shifts64<const BITS: usize, const OFF: usize>() -> __m256i {
    let b = BITS as i64;
    let o = (OFF * BITS) as i64;
    _mm256_setr_epi64x(o, o + b, o + 2 * b, o + 3 * b)
}

/// Eight `BITS`-wide codes from the chunk at `p`, in element order, as `i32`
/// lanes (each already masked to its width).
///
/// One broadcast of the chunk plus one variable shift per code. An 8-code chunk
/// is `8·BITS` bits, so widths up to four fit a 32-bit lane and a single shift;
/// the 5- and 6-bit packs are 40 and 48 bits and need the field in 64-bit lanes,
/// where each lane carries one code and the four low dwords of each half are
/// compacted back into eight lanes.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn codes8<const BITS: usize>(p: *const u8) -> __m256i {
    if BITS <= 4 {
        let v = _mm256_set1_epi32(load_le::<BITS>(p) as i32);
        let b = BITS as i32;
        let shifts = _mm256_setr_epi32(0, b, 2 * b, 3 * b, 4 * b, 5 * b, 6 * b, 7 * b);
        let mask = _mm256_set1_epi32(((1u32 << BITS) - 1) as i32);
        _mm256_and_si256(_mm256_srlv_epi32(v, shifts), mask)
    } else {
        let v = _mm256_set1_epi64x(load_le::<BITS>(p) as i64);
        let mask = _mm256_set1_epi64x(((1u64 << BITS) - 1) as i64);
        let lo = _mm256_and_si256(_mm256_srlv_epi64(v, shifts64::<BITS, 0>()), mask);
        let hi = _mm256_and_si256(_mm256_srlv_epi64(v, shifts64::<BITS, 4>()), mask);
        // Each 64-bit lane now holds one code in its low dword; take dwords
        // 0, 2, 4 and 6 of each half and join them.
        let pick = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
        let lo = _mm256_permutevar8x32_epi32(lo, pick);
        let hi = _mm256_permutevar8x32_epi32(hi, pick);
        _mm256_set_m128i(_mm256_castsi256_si128(hi), _mm256_castsi256_si128(lo))
    }
}

/// `(Σ c·x, Σ x)` over the `first..last` chunks of eight `BITS`-wide codes that
/// start `PAYLOAD` bytes into the group.
///
/// Both sums are vector accumulators, so a group pays two horizontal sums
/// rather than one per chunk.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn codes_dot_sum<const BITS: usize, const PAYLOAD: usize>(
    gptr: *const u8,
    xg: *const f32,
    first: usize,
    last: usize,
) -> (f32, f32) {
    let mut dot = _mm256_setzero_ps();
    let mut sum = _mm256_setzero_ps();
    for idx in first..last {
        let xv = _mm256_loadu_ps(xg.add(idx * 8));
        let c = _mm256_cvtepi32_ps(codes8::<BITS>(gptr.add(PAYLOAD + idx * BITS)));
        dot = _mm256_fmadd_ps(c, xv, dot);
        sum = _mm256_add_ps(sum, xv);
    }
    (hsum256(dot), hsum256(sum))
}

/// One 256-element V2 group: `[s0 z0 s1 z1]` `fp16` (one pair per 128
/// elements) then `256/BITS` payload bytes.
///
/// Each half is scored separately and then affinely corrected, so a half's
/// header costs two vector FMAs rather than an `fma` per element.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn v2_group_dot<const BITS: usize>(gptr: *const u8, xg: *const f32) -> f32 {
    let h = f16_quad(gptr);
    let (d0, s0) = codes_dot_sum::<BITS, 8>(gptr, xg, 0, 16);
    let (d1, s1) = codes_dot_sum::<BITS, 8>(gptr, xg, 16, 32);
    (h[0] * d0 + h[1] * s0) + (h[2] * d1 + h[3] * s1)
}

/// The V2 family's per-128 header, `[s0 z0 s1 z1]`, widened with one
/// `vcvtph2ps`.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn f16_quad(p: *const u8) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    _mm_storeu_ps(
        out.as_mut_ptr(),
        _mm_cvtph_ps(_mm_loadl_epi64(p as *const __m128i)),
    );
    out
}

/// One `f32` from the little-endian bytes at `p`.
#[inline]
unsafe fn f32_at(p: *const u8) -> f32 {
    f32::from_le_bytes(*(p as *const [u8; 4]))
}

/// The row driver every format shares: `Σ_g group(row + g·GB, x + g·GE)`.
///
/// A macro rather than a generic function because the group body is a
/// `#[target_feature]` function and its features do not propagate into a
/// closure or through a function pointer, so the call has to be monomorphized
/// in place.
macro_rules! row_dot {
    ($(#[$meta:meta])* $name:ident, $group:expr, $ge:literal, $gb:literal, $feat:literal) => {
        $(#[$meta])*
        ///
        /// # Safety
        ///
        /// Requires the features in this function's own `target_feature` list,
        /// `k % $ge == 0`, at least `(k / $ge) * $gb` readable bytes at `row`
        /// and `k` readable `f32` at `x`. Callers reach it through
        /// [`super::row_dot_avx2`], which resolves the CPU features first.
        #[target_feature(enable = $feat)]
        pub unsafe fn $name(row: *const u8, k: usize, x: *const f32) -> f32 {
            let mut acc = 0.0f32;
            for g in 0..k / $ge {
                acc += ($group)(row.add(g * $gb), x.add(g * $ge));
            }
            acc
        }
    };
}

row_dot!(
    /// qt 49 — `Mq3G256V2`: 3-bit cross-byte packs, per-128 `fp16` header.
    mq3g256v2_row_dot,
    v2_group_dot::<3>,
    256,
    104,
    "avx2,fma,f16c"
);

row_dot!(
    /// qt 44 — `Mq4G256V2`: the `mq4v2` SKU's format. Nibble payload (qt 13's
    /// geometry) under qt 49's V2 `fp16` header.
    mq4g256v2_row_dot,
    v2_group_dot::<4>,
    256,
    136,
    "avx2,fma,f16c"
);

row_dot!(
    /// qt 47 — `Mq6G256V2`: 6-bit packs under the V2 header.
    mq6g256v2_row_dot,
    v2_group_dot::<6>,
    256,
    200,
    "avx2,fma,f16c"
);

row_dot!(
    /// qt 48 — `Mq5G256V2`: 5-bit packs under the V2 header.
    mq5g256v2_row_dot,
    v2_group_dot::<5>,
    256,
    168,
    "avx2,fma,f16c"
);

row_dot!(
    /// qt 50 — `Mq2G256V2`: 2-bit packs under the V2 header (quality-rejected
    /// for product, still decodes).
    mq2g256v2_row_dot,
    v2_group_dot::<2>,
    256,
    72,
    "avx2,fma,f16c"
);

/// One 256-element `Mq4G256` (qt 13) group: `[f32 scale][f32 zero]` then 128 B
/// of paired nibbles, element `i` taking the low nibble of byte `i/2` for even
/// `i` and the high nibble for odd `i`. The activation arrives already
/// FWHT-rotated, so this kernel only decodes and multiplies.
///
/// `_mm_unpacklo_epi8` of the low/high nibble planes yields `[c0, c1, c2, …]` in
/// element order (byte `j` contributes its low nibble as `c[2j]` and its high
/// nibble as `c[2j+1]`), so no shuffle of the activation is needed.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn mq4_group_dot(gptr: *const u8, xg: *const f32) -> f32 {
    let scale = f32_at(gptr);
    let zero = f32_at(gptr.add(4));
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

row_dot!(
    /// qt 13 — `Mq4G256`: nibbles under the original `f32` header. The
    /// hand-unrolled nibble group above, at 32 codes per 16-byte load — see the
    /// module docs for why it is not the `codes8` path.
    mq4g256_row_dot,
    mq4_group_dot,
    256,
    136,
    "avx2,fma"
);
