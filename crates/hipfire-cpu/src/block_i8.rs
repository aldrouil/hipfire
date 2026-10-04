// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! `block_i8_128` — the int8 activation block shared with the gfx12 A8 MMQ
//! route (`kernels/src/block_i8_128_quant.hip`).
//!
//! One block is 128 contiguous K elements of one activation row: an `f32`
//! scale `d`, the exact `int32` sum `s` of the quantized codes, and the 128
//! codes `qs`. The contract is transcribed from the kernel header so the CPU
//! and the GPU produce identical blocks:
//!
//! ```text
//! amax = max |x| over the 128 elements          (fmaxf, f32)
//! d    = RN(amax / 127.0f)                       (IEEE divide)
//! q    = clamp(rintf(RN(x / d)), -127, 127)      (IEEE divide, ties-to-even)
//! zero block (amax == 0): d = 0, q = 0
//! s    = sum of the 128 q                        (exact int32)
//! ```
//!
//! `qs` stores the codes in nibble-pair order — within every 8-element group
//! `g`, `qs[8g+0..4] = q[8g+0], q[8g+2], q[8g+4], q[8g+6]` and
//! `qs[8g+4..8] = q[8g+1], q[8g+3], q[8g+5], q[8g+7]` — which is the byte order
//! one packed MQ4V2 weight dword (8 nibbles, lo/hi planes) presents, so the
//! integer dot pairs each weight code with its own activation with no shuffle.
//! `d` and `s` are order-free.
//!
//! [`BlockI8_128::quantize`] is the one definition; the A8 quality emulation
//! and the GPU oracle test against it.

/// Activation block width: 128 contiguous K elements.
pub const BLOCK_I8_128_K: usize = 128;

/// Bytes per block: `f32 d` + `int32 s` + 128 `i8` codes.
pub const BLOCK_I8_128_BYTES: usize = 136;

/// `struct block_i8_128` — `#[repr(C)]` layout matching `block_i8_128_quant.hip`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockI8_128 {
    pub d: f32,
    pub s: i32,
    pub qs: [i8; BLOCK_I8_128_K],
}

impl Default for BlockI8_128 {
    fn default() -> Self {
        Self {
            d: 0.0,
            s: 0,
            qs: [0; BLOCK_I8_128_K],
        }
    }
}

impl BlockI8_128 {
    /// Quantize one 128-element activation block. Only the first
    /// [`BLOCK_I8_128_K`] elements of `x` are read.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than [`BLOCK_I8_128_K`] elements.
    pub fn quantize(x: &[f32]) -> Self {
        assert!(
            x.len() >= BLOCK_I8_128_K,
            "block_i8_128: activation has {} elements, need {BLOCK_I8_128_K}",
            x.len()
        );
        let x = &x[..BLOCK_I8_128_K];
        // `fmaxf` over |x|; `f32::max` returns the non-NaN operand like fmaxf.
        let amax = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let mut out = BlockI8_128::default();
        if amax == 0.0 {
            // d = 0, q = 0, s = 0.
            return out;
        }
        let d = amax / 127.0;
        out.d = d;
        // `rintf` is round-to-nearest, ties-to-even; `f32::round` is ties-away
        // and would disagree with the GPU on a tie.
        let mut q = [0i8; BLOCK_I8_128_K];
        let mut s = 0i32;
        for (slot, &v) in q.iter_mut().zip(x.iter()) {
            let r = (v / d).round_ties_even().clamp(-127.0, 127.0);
            let qi = r as i8;
            *slot = qi;
            s += qi as i32;
        }
        out.s = s;
        // nibble-pair order (see the module docs)
        for g in 0..BLOCK_I8_128_K / 8 {
            let b = g * 8;
            out.qs[b] = q[b];
            out.qs[b + 1] = q[b + 2];
            out.qs[b + 2] = q[b + 4];
            out.qs[b + 3] = q[b + 6];
            out.qs[b + 4] = q[b + 1];
            out.qs[b + 5] = q[b + 3];
            out.qs[b + 6] = q[b + 5];
            out.qs[b + 7] = q[b + 7];
        }
        out
    }

    /// The 128 codes in *element* order (undoing the nibble-pair permutation).
    /// For tests and for a kernel that would rather index by K position.
    pub fn codes_in_element_order(&self) -> [i8; BLOCK_I8_128_K] {
        let mut q = [0i8; BLOCK_I8_128_K];
        for g in 0..BLOCK_I8_128_K / 8 {
            let b = g * 8;
            q[b] = self.qs[b];
            q[b + 2] = self.qs[b + 1];
            q[b + 4] = self.qs[b + 2];
            q[b + 6] = self.qs[b + 3];
            q[b + 1] = self.qs[b + 4];
            q[b + 3] = self.qs[b + 5];
            q[b + 5] = self.qs[b + 6];
            q[b + 7] = self.qs[b + 7];
        }
        q
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn layout_is_136_bytes() {
        assert_eq!(std::mem::size_of::<BlockI8_128>(), BLOCK_I8_128_BYTES);
        assert_eq!(std::mem::align_of::<BlockI8_128>(), 4);
    }

    #[test]
    fn zero_block_is_all_zero() {
        let b = BlockI8_128::quantize(&[0.0f32; BLOCK_I8_128_K]);
        assert_eq!(b, BlockI8_128::default());
        assert_eq!(b.d, 0.0);
        assert_eq!(b.s, 0);
    }

    /// A scoped reference that never touches the nibble permutation, so the
    /// permutation can be tested independently of the arithmetic.
    fn reference_q(x: &[f32]) -> ([i8; 128], f32, i32) {
        let amax = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        if amax == 0.0 {
            return ([0; 128], 0.0, 0);
        }
        let d = amax / 127.0;
        let mut q = [0i8; 128];
        let mut s = 0i32;
        for (i, &v) in x.iter().enumerate() {
            q[i] = (v / d).round_ties_even().clamp(-127.0, 127.0) as i8;
            s += q[i] as i32;
        }
        (q, d, s)
    }

    #[test]
    fn codes_and_sum_match_the_reference() {
        let x: Vec<f32> = (0..BLOCK_I8_128_K)
            .map(|i| ((i as f32) * 0.37).sin() * 3.0 - 0.5)
            .collect();
        let b = BlockI8_128::quantize(&x);
        let (q, d, s) = reference_q(&x);
        assert_eq!(b.d, d);
        assert_eq!(b.s, s);
        assert_eq!(b.codes_in_element_order(), q);
    }

    #[test]
    fn extrema_map_to_the_clamp_bounds() {
        let mut x = vec![0.0f32; BLOCK_I8_128_K];
        x[7] = 2.0;
        x[100] = -2.0;
        let b = BlockI8_128::quantize(&x);
        assert_eq!(b.d, 2.0 / 127.0);
        let q = b.codes_in_element_order();
        assert_eq!(q[7], 127);
        assert_eq!(q[100], -127);
    }

    #[test]
    fn rounding_is_ties_to_even_not_ties_away() {
        // amax = 127 so d = 1.0 exactly and the .5 tie points are reachable.
        let mut t = vec![0.0f32; BLOCK_I8_128_K];
        t[0] = 127.0;
        t[1] = 0.5; // ties to even -> 0
        t[2] = 1.5; // ties to even -> 2
        t[3] = 2.5; // ties to even -> 2
        t[4] = 3.5; // ties to even -> 4
        t[5] = -0.5; // ties to even -> 0
        t[6] = -1.5; // ties to even -> -2
        let b = BlockI8_128::quantize(&t);
        assert_eq!(b.d, 1.0);
        let q = b.codes_in_element_order();
        assert_eq!(
            [q[0], q[1], q[2], q[3], q[4], q[5], q[6]],
            [127, 0, 2, 2, 4, 0, -2]
        );
    }

    #[test]
    fn nibble_pair_order_is_a_permutation_of_element_order() {
        let x: Vec<f32> = (0..BLOCK_I8_128_K)
            .map(|i| ((i as f32) * 0.11).cos() * 5.0)
            .collect();
        let b = BlockI8_128::quantize(&x);
        let q = b.codes_in_element_order();
        // The stored `qs` is the element-order codes permuted *within* each
        // 8-group: qs[8g+0..4] = even indices, qs[8g+4..8] = odd indices.
        for g in 0..BLOCK_I8_128_K / 8 {
            let base = g * 8;
            assert_eq!(b.qs[base], q[base]);
            assert_eq!(b.qs[base + 1], q[base + 2]);
            assert_eq!(b.qs[base + 2], q[base + 4]);
            assert_eq!(b.qs[base + 3], q[base + 6]);
            assert_eq!(b.qs[base + 4], q[base + 1]);
            assert_eq!(b.qs[base + 5], q[base + 3]);
            assert_eq!(b.qs[base + 6], q[base + 5]);
            assert_eq!(b.qs[base + 7], q[base + 7]);
        }
    }
}
