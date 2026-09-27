// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Quant decode for the CPU-executed offload path.
//!
//! Every arm is a transcription of
//! `hipfire_runtime::weight_backend::dequantize_to_f32` (the single canonical
//! CPU decoder — see its doc comment: "there is exactly one copy") and, for
//! qt 3, of `hipfire_runtime::llama::dequantize_q8_0`. The transcription exists
//! rather than a call because this crate must not link the GPU stack (see the
//! crate docs). `crates/hipfire-runtime/tests/cpu_quant_cross_check.rs` is what
//! keeps the two copies honest.
//!
//! Two decodes are exposed, and the difference between them is *the* thing to
//! get right in this file:
//!
//! * [`decode_group_codes`] — the affine/level decode only. These are the code
//!   values the MQ GEMV kernels dot against a **forward-rotated** activation
//!   (`rotate_x`): the weights are stored post-rotation, and
//!   `dot(rot W, rot x) == dot(W, x)` is what makes the product come out in the
//!   original basis. `crate::gemv` uses this one.
//! * [`dequant_group`] — the same affine decode followed by the *inverse* FWHT
//!   ([`fwht256_inplace`]) for the FWHT-rotated formats, i.e. the weights in the
//!   original basis, matching `dequantize_to_f32` value for value.
//!
//! Mixing them — un-rotating the weights *and* rotating the activation — is a
//! silent $\mathcal{R}^{-2}$ error, not a crash. `f32` arithmetic here is
//! ordinary `mul`/`add` in the canonical decoder's order, so
//! [`test::dequant_group_matches_canonical_decoder_bits`] holds exactly.

use std::sync::LazyLock;

/// Seeds of the MagnumQuant FWHT sign vectors, mirroring
/// `Gpu::ensure_mq_signs` (which uploads `KvCache::gen_fwht_signs(42|1042, 256)`).
const FWHT_SEED1: u32 = 42;
const FWHT_SEED2: u32 = 1042;
/// Every G256 format rotates in fixed 256-element groups.
const FWHT_N: usize = 256;

/// Quant formats the CPU offload path can decode.
///
/// This is exactly the set the qwen3.5 dense registry offers, as measured on
/// the pulled fixtures rather than assumed: `qwen3.5-2b.mq4` (qt 13),
/// `qwen3.5-2b.mq3` (**qt 20**, not qt 17 — the `-mq3` tags ship the Lloyd-Max
/// 3-bit tier), `qwen3.5-2b.mq6` (qt 15), `qwen3.5-2b.hf6` (qt 8), the `mq4v2`
/// bodies current `hipfire-quantize --format mq4` writes (qt 44), and plain
/// qt 17 for checkpoints that carry the uniform 3-bit tier — plus the blocking
/// formats those files use for norms, the embedding and structural tensors. Anything else has no CPU implementation, so a step over it stays on
/// the GPU over PCIe: correctness is unaffected, only the bandwidth win is
/// partial, and the load-time coverage line names the format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuQuant {
    /// qt 13 — FWHT-rotated HFQ4-G256, 136 B / 256.
    Mq4G256,
    /// qt 44 — FWHT-rotated HFQ4-G256 v2, per-128 fp16 scale/zero, 136 B / 256.
    Mq4G256V2,
    /// qt 15 — FWHT-rotated HFQ6-G256, 200 B / 256.
    Mq6G256,
    /// qt 17 — FWHT-rotated HFQ3-G256, 104 B / 256.
    Mq3G256,
    /// qt 20 — FWHT-rotated 3-bit with a per-group 8-entry fp16 Lloyd-Max
    /// codebook, 112 B / 256. This is what the registry's `qwen3.5:2b-mq3`
    /// (and the other `-mq3` tags) actually contain — measured, not assumed.
    Mq3G256Lloyd,
    /// qt 8 — HFQ6-G256, no rotation, 200 B / 256.
    Hfq6G256,
    /// qt 6 — HFQ4-G256, no rotation, 136 B / 256.
    Hfq4G256,
    /// qt 1 — f16, 2 B / element.
    F16,
    /// qt 2 — f32, 4 B / element.
    F32,
    /// qt 16 — bf16, 2 B / element.
    Bf16,
    /// qt 3 — `Q8F16` (`DType::Q8_0`): f16 scale + 32 i8, 34 B / 32.
    Q8F16,
}

impl CpuQuant {
    /// Map the serialized HFQ `quant_type` byte (see `hipfire_quantize`'s
    /// `QuantType`) to a CPU-decodable format, or `None` when this crate has no
    /// implementation for it.
    pub fn from_quant_type(quant_type: u8) -> Option<Self> {
        match quant_type {
            1 => Some(Self::F16),
            2 => Some(Self::F32),
            3 => Some(Self::Q8F16),
            6 => Some(Self::Hfq4G256),
            8 => Some(Self::Hfq6G256),
            13 => Some(Self::Mq4G256),
            15 => Some(Self::Mq6G256),
            17 => Some(Self::Mq3G256),
            20 => Some(Self::Mq3G256Lloyd),
            44 => Some(Self::Mq4G256V2),
            _ => None,
        }
    }

    /// Elements per group. Rows are `(k / group_elems()) * group_bytes()` long.
    pub fn group_elems(self) -> usize {
        match self {
            Self::Q8F16 => 32,
            _ => 256,
        }
    }

    /// Bytes per group, including any per-group header.
    pub fn group_bytes(self) -> usize {
        match self {
            Self::Mq4G256 | Self::Mq4G256V2 | Self::Hfq4G256 => 136,
            Self::Mq6G256 | Self::Hfq6G256 => 200,
            Self::Mq3G256 => 104,
            Self::Mq3G256Lloyd => 112,
            Self::Q8F16 => 34,
            Self::F16 | Self::Bf16 => 512,
            Self::F32 => 1024,
        }
    }

    /// Whether the format's weights are stored FWHT-256-rotated, and therefore
    /// whether the activation must be forward-rotated before the dot product.
    ///
    /// Invariant: this agrees with `hipfire_dispatch::types::dtype_rotation_plan`
    /// on `RotationPlan::FwhtG256` for every dtype that maps here (pinned by
    /// `crates/hipfire-dispatch`'s `cpu_quant_rotation_agrees_with_plan`).
    pub fn is_fwht_g256(self) -> bool {
        matches!(
            self,
            Self::Mq4G256 | Self::Mq4G256V2 | Self::Mq6G256 | Self::Mq3G256 | Self::Mq3G256Lloyd
        )
    }
}

/// LCG behind the MagnumQuant sign vectors; mirrors
/// `KvCache::gen_fwht_signs` exactly (each step is `s = (s * 1103515245 + 12345)
/// mod 2^31` once the wrapping `u32` arithmetic and the `0x7fffffff` mask are
/// taken together).
pub fn fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fff_ffff;
            if (state >> 16) & 1 == 1 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

fn sign_table(seed: u32) -> [f32; FWHT_N] {
    fwht_signs(seed, FWHT_N)
        .try_into()
        .expect("gapless 256")
}

static SIGNS1: LazyLock<[f32; FWHT_N]> = LazyLock::new(|| sign_table(FWHT_SEED1));
static SIGNS2: LazyLock<[f32; FWHT_N]> = LazyLock::new(|| sign_table(FWHT_SEED2));

fn signs1() -> &'static [f32; FWHT_N] {
    &SIGNS1
}

fn signs2() -> &'static [f32; FWHT_N] {
    &SIGNS2
}

/// In-place radix-2 Hadamard butterfly over 256 elements (unnormalized: it
/// scales by 256, which the `1/16` post-scale below cancels into `1/sqrt(256)`).
fn butterfly(group: &mut [f32; FWHT_N]) {
    let mut stride = 1;
    while stride < FWHT_N {
        let mut j = 0;
        while j < FWHT_N {
            for k in 0..stride {
                let a = group[j + k];
                let b = group[j + k + stride];
                group[j + k] = a + b;
                group[j + k + stride] = a - b;
            }
            j += stride * 2;
        }
        stride <<= 1;
    }
}

/// Inverse FWHT-256: the *un*-rotation `dequantize_to_f32` applies to every
/// FWHT-rotated format — pre-scale by `signs2`, butterfly, post-scale by
/// `0.0625 * signs1` (`0.0625 = 1/16 = 1/sqrt(256)`).
///
/// A sign or normalization error here is the "token soup" attractor failure
/// mode, which is why the tests below pin both directions against independent
/// oracles rather than against each other.
pub fn fwht256_inplace(group: &mut [f32; FWHT_N], signs1: &[f32], signs2: &[f32]) {
    for i in 0..FWHT_N {
        group[i] *= signs2[i];
    }
    butterfly(group);
    let scale_inv = 0.0625;
    for i in 0..FWHT_N {
        group[i] *= scale_inv * signs1[i];
    }
}

/// Forward FWHT-256 rotation of an activation, matching the `mq_rotate_x`
/// kernels: pre-scale by `signs1`, butterfly, post-scale by `0.0625 * signs2`.
/// The sign tables appear in mirror order relative to [`fwht256_inplace`], which
/// is what makes the two transpose (and therefore mutually inverse, given the
/// `1/sqrt(256)` normalization).
///
/// `x.len()` must be a multiple of 256.
pub fn rotate_x(x: &mut [f32]) {
    assert!(
        x.len() % FWHT_N == 0,
        "rotate_x: length {} is not a multiple of {FWHT_N}",
        x.len()
    );
    for block in x.chunks_exact_mut(FWHT_N) {
        let group: &mut [f32; FWHT_N] = block.try_into().expect("chunk is 256");
        for i in 0..FWHT_N {
            group[i] *= signs1()[i];
        }
        butterfly(group);
        let scale = 0.0625;
        for i in 0..FWHT_N {
            group[i] *= scale * signs2()[i];
        }
    }
}

/// Divide an activation by a per-input-channel AWQ scale:
/// `x[i] /= scale[i]`, indexed in the **unrotated** basis.
///
/// This is the op the `rotate_x_mq_awq` / `fused_rmsnorm_mq_rotate_awq` /
/// `fused_silu_mul_rotate_mq_awq` kernels fold in ahead of the FWHT: the
/// quantizer pre-scaled those weights by `s`, so `(W·s) · (x/s) = W·x` only
/// holds if the activation is divided *before* it is rotated. Omitting it does
/// not fail — it silently computes `(W·s)·x`, a per-channel scale error on
/// every projection that carries a sidecar (`DType::supports_awq_sidecar`),
/// which is the "token soup" failure mode with no error to point at.
///
/// `scale` is the loader's already-widened f32 sidecar (`load_awq_scale_for`
/// converts the on-disk f16 to f32 on the host), one value per input channel.
pub fn divide_by_awq_scale(x: &mut [f32], scale: &[f32]) {
    assert!(
        scale.len() >= x.len(),
        "awq scale has {} channels, activation has {}",
        scale.len(),
        x.len()
    );
    for (v, s) in x.iter_mut().zip(scale) {
        *v /= *s;
    }
}

#[inline]
fn f16_at(bytes: &[u8], off: usize) -> f32 {
    half_f16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]))
}

#[inline]
fn f32_at(bytes: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

/// `half::f16::from_bits(bits).to_f32()`, inlined so this crate keeps its
/// dependency list empty of the GPU stack *and* of formatting crates.
/// `f16_widening_is_ieee` holds it against `half` for every encoding class.
#[inline]
fn half_f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let man = (bits & 0x3FF) as u32;
    let out = match exp {
        0 => {
            if man == 0 {
                sign << 31
            } else {
                // Subnormal: `man * 2^-24`, which is exact in f32 (10 significand
                // bits at an exponent no smaller than `2^-24`).
                f32::to_bits(man as f32 * f32::from_bits(0x3380_0000)) | (sign << 31)
            }
        }
        0x1F => (sign << 31) | 0x7F80_0000 | (man << 13),
        _ => (sign << 31) | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(out)
}

/// Decode one group's *code values* into `out[0..q.group_elems()]`: the affine
/// or codebook decode, with no rotation. These are the values the MQ kernels dot
/// against a forward-rotated activation (see the module docs).
pub fn decode_group_codes(q: CpuQuant, packed: &[u8], out: &mut [f32]) {
    let ge = q.group_elems();
    assert!(
        out.len() >= ge,
        "decode_group_codes({q:?}): out has {} elements, need {ge}",
        out.len()
    );
    assert!(
        packed.len() >= q.group_bytes(),
        "decode_group_codes({q:?}): packed has {} bytes, need {}",
        packed.len(),
        q.group_bytes()
    );
    match q {
        CpuQuant::F16 => {
            for (i, o) in out[..ge].iter_mut().enumerate() {
                *o = f16_at(packed, 2 * i);
            }
        }
        CpuQuant::F32 => {
            for (i, o) in out[..ge].iter_mut().enumerate() {
                *o = f32_at(packed, 4 * i);
            }
        }
        CpuQuant::Bf16 => {
            for (i, o) in out[..ge].iter_mut().enumerate() {
                *o = f32::from_bits((f16_at_u16(packed, 2 * i) as u32) << 16);
            }
        }
        CpuQuant::Q8F16 => {
            let scale = f16_at(packed, 0);
            for (i, o) in out[..ge].iter_mut().enumerate() {
                *o = packed[2 + i] as i8 as f32 * scale;
            }
        }
        CpuQuant::Mq4G256 | CpuQuant::Hfq4G256 => {
            let scale = f32_at(packed, 0);
            let zero = f32_at(packed, 4);
            for (i, o) in out[..ge].iter_mut().enumerate() {
                let byte = packed[8 + i / 2];
                let nibble = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };
                *o = scale * nibble as f32 + zero;
            }
        }
        CpuQuant::Mq4G256V2 => {
            for (i, o) in out[..ge].iter_mut().enumerate() {
                let h = i / 128;
                let scale = f16_at(packed, 4 * h);
                let zero = f16_at(packed, 4 * h + 2);
                let byte = packed[8 + i / 2];
                let nibble = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };
                *o = scale * nibble as f32 + zero;
            }
        }
        CpuQuant::Mq6G256 | CpuQuant::Hfq6G256 => {
            let scale = f32_at(packed, 0);
            let zero = f32_at(packed, 4);
            let mut i = 0;
            while i < ge {
                let bo = 8 + (i / 4) * 3;
                let b0 = packed[bo] as u32;
                let b1 = packed[bo + 1] as u32;
                let b2 = packed[bo + 2] as u32;
                out[i] = scale * ((b0 & 0x3F) as f32) + zero;
                out[i + 1] = scale * ((((b0 >> 6) | (b1 << 2)) & 0x3F) as f32) + zero;
                out[i + 2] = scale * ((((b1 >> 4) | (b2 << 4)) & 0x3F) as f32) + zero;
                out[i + 3] = scale * (((b2 >> 2) & 0x3F) as f32) + zero;
                i += 4;
            }
        }
        CpuQuant::Mq3G256 => {
            let scale = f32_at(packed, 0);
            let zero = f32_at(packed, 4);
            for chunk in 0..32 {
                let bo = 8 + chunk * 3;
                let b0 = packed[bo] as u32;
                let b1 = packed[bo + 1] as u32;
                let b2 = packed[bo + 2] as u32;
                let base = chunk * 8;
                let codes = [
                    b0 & 7,
                    (b0 >> 3) & 7,
                    ((b0 >> 6) | (b1 << 2)) & 7,
                    (b1 >> 1) & 7,
                    (b1 >> 4) & 7,
                    ((b1 >> 7) | (b2 << 1)) & 7,
                    (b2 >> 2) & 7,
                    (b2 >> 5) & 7,
                ];
                for (k, code) in codes.iter().enumerate() {
                    out[base + k] = scale * *code as f32 + zero;
                }
            }
        }
        CpuQuant::Mq3G256Lloyd => {
            // 16 B codebook (8 × fp16, ascending at quant time) then 96 B of
            // 3-bit indices in the same cross-byte packing as Mq3G256.
            let mut cb = [0.0f32; 8];
            for (k, c) in cb.iter_mut().enumerate() {
                *c = f16_at(packed, 2 * k);
            }
            for chunk in 0..32 {
                let bo = 16 + chunk * 3;
                let b0 = packed[bo] as u32;
                let b1 = packed[bo + 1] as u32;
                let b2 = packed[bo + 2] as u32;
                let base = chunk * 8;
                let codes = [
                    b0 & 7,
                    (b0 >> 3) & 7,
                    ((b0 >> 6) | (b1 << 2)) & 7,
                    (b1 >> 1) & 7,
                    (b1 >> 4) & 7,
                    ((b1 >> 7) | (b2 << 1)) & 7,
                    (b2 >> 2) & 7,
                    (b2 >> 5) & 7,
                ];
                for (k, code) in codes.iter().enumerate() {
                    out[base + k] = cb[*code as usize];
                }
            }
        }
    }
}

#[inline]
fn f16_at_u16(bytes: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([bytes[off], bytes[off + 1]])
}

/// Decode one group of `q.group_elems()` weights into `out`, in the **original
/// basis**: [`decode_group_codes`] followed by the inverse FWHT for the rotated
/// formats. Value-for-value the arithmetic of `dequantize_to_f32`.
///
/// The CPU GEMV path does *not* use this (it keeps the codes and rotates the
/// activation — see the module docs); it exists for the cross-check against the
/// canonical decoder and for any future host-side weight materialization.
pub fn dequant_group(q: CpuQuant, packed: &[u8], out: &mut [f32]) {
    decode_group_codes(q, packed, out);
    if q.is_fwht_g256() {
        let group: &mut [f32; FWHT_N] = (&mut out[..FWHT_N]).try_into().expect("group is 256");
        fwht256_inplace(group, signs1(), signs2());
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// The single-group fixture the expectation tables below were generated
    /// over (`testfix::group_bytes(q, 0)`).
    fn fixture(q: CpuQuant) -> Vec<u8> {
        crate::testfix::group_bytes(q, 0)
    }

    /// The f16 bit patterns every implementation in this workspace must agree on,
    /// computed by hand as shortest-round-trip literals in
    /// `crates/hipfire-runtime/tests/cpu_quant_cross_check.rs`'s generator.
    const EXPECTED_F16_BITS: [u16; 10] = [
        0x0000, 0x3c00, 0xbc00, 0x4000, 0xc000, 0x4200, 0xc200, 0x4500, 0x2c00, 0x3a00,
    ];

    // ── expectation tables, generated from the canonical decoder ────────────
    // Recipe (reproduce with a throwaway example in `hipfire-runtime`):
    // for each format, run
    // `weight_backend::dequantize_weight_to_f32(qt, &testfix::group_bytes(q, 0), q.group_elems())`
    // and print `v.to_bits()` as `0x%08x`. Bit patterns rather than decimal
    // literals, so the comparison is exact and immune to float-literal
    // round-tripping; the generator is deleted, the fixture is not.
    /// qt 13, 136 B / 256 elems.
    const EXPECTED_MQ4G256: [u32; 256] = [
        0x3df00000, 0x3e200000, 0xbd800000, 0xbe280000, 0xbe840000, 0xbed00000, 0xbd400000, 0x3eb40000,
        0xbe080000, 0xbe800000, 0x3ee80000, 0x3ebc0000, 0xbe840000, 0x3d800000, 0xbdc00000, 0x3cc00000,
        0x3ee40000, 0xbe100000, 0x3e500000, 0xbe180000, 0x3e580000, 0x3ed80000, 0x3e000000, 0x3f160000,
        0xbc000000, 0xbe880000, 0xbf280000, 0x3db00000, 0x3e380000, 0xbda00000, 0x3e500000, 0x3f0a0000,
        0xbc800000, 0xbdd00000, 0x3f120000, 0x3e500000, 0x3e980000, 0xbe680000, 0x3e580000, 0xbe400000,
        0xbd400000, 0x3c000000, 0x3e680000, 0x80000000, 0x3d400000, 0x3e280000, 0xbd900000, 0x3de00000,
        0xbd400000, 0x3e840000, 0xbef40000, 0xbe700000, 0xbc800000, 0xbe840000, 0x3e780000, 0xbe600000,
        0x3c800000, 0xbeac0000, 0x3d900000, 0x3eb00000, 0x3d400000, 0xbf060000, 0x3c000000, 0x3e980000,
        0xbe600000, 0x3e680000, 0xbd600000, 0x3ea00000, 0x3e000000, 0x3cc00000, 0x3e480000, 0x3f140000,
        0xbe100000, 0x3df00000, 0xbed40000, 0xbe200000, 0x3e700000, 0xbdb00000, 0x3e780000, 0xbea80000,
        0x3ef00000, 0x3d600000, 0xbd900000, 0x3f080000, 0x3dc00000, 0xbe480000, 0x3f020000, 0xbea80000,
        0x3e880000, 0xbefc0000, 0x3e480000, 0xbe800000, 0x3f140000, 0x3e840000, 0xbdf00000, 0xbeb80000,
        0xbe940000, 0x3ee00000, 0x3d800000, 0x3e940000, 0x3c000000, 0x3e900000, 0xbed80000, 0x3e840000,
        0x3e580000, 0xbec80000, 0xbf900000, 0x3f360000, 0x3e480000, 0xbe500000, 0x3e980000, 0x3df00000,
        0xbefc0000, 0xbed80000, 0x3de00000, 0x3eac0000, 0xbdf00000, 0xbe500000, 0xbeb00000, 0x3df00000,
        0xbf4e0000, 0xbe200000, 0xbeb80000, 0xbe080000, 0x3f4a0000, 0x3eb00000, 0x3f100000, 0x3f5e0000,
        0xbf1a0000, 0x3ea80000, 0x3da00000, 0x3e840000, 0xbc000000, 0xbe900000, 0x3ec80000, 0x3e9c0000,
        0x3e380000, 0xbe100000, 0x3d800000, 0x3cc00000, 0xbe9c0000, 0x3dc00000, 0x3ef00000, 0xbedc0000,
        0xbe9c0000, 0x3e100000, 0xbe700000, 0x3e840000, 0xbec40000, 0x3e400000, 0xbde00000, 0x3e180000,
        0x3dd00000, 0xbe700000, 0xbe900000, 0x3ecc0000, 0xbdb00000, 0xbd800000, 0xbe200000, 0x3e9c0000,
        0x3ea00000, 0xbd900000, 0x3e8c0000, 0xbd000000, 0xbc800000, 0xbeac0000, 0x3e480000, 0x3eb00000,
        0xbd800000, 0x3e380000, 0x3d900000, 0x3d400000, 0xbee80000, 0x3e940000, 0x3f260000, 0xbe700000,
        0xbf0c0000, 0x3ebc0000, 0xbd900000, 0x3e980000, 0x3e000000, 0xbd600000, 0x3db00000, 0xbd400000,
        0x3e300000, 0xbe480000, 0x3c000000, 0x3e900000, 0xbe000000, 0xbe180000, 0x3df00000, 0x3dc00000,
        0x3e880000, 0xbeac0000, 0x3dd00000, 0x3eb80000, 0x3e200000, 0x3e480000, 0xbd200000, 0xbea80000,
        0xbd000000, 0x3e780000, 0x3e8c0000, 0x3e500000, 0xbe100000, 0xbee40000, 0x3cc00000, 0xbe300000,
        0x3d000000, 0xbecc0000, 0x3e180000, 0xbdc00000, 0x3ec80000, 0x3e8c0000, 0x3d200000, 0x3dc00000,
        0x3da00000, 0xbee40000, 0x3f120000, 0x3d800000, 0x3f100000, 0xbe480000, 0x3dd00000, 0x3e900000,
        0xbe840000, 0xbef00000, 0x3eb00000, 0xbedc0000, 0xbedc0000, 0x3e980000, 0x3e900000, 0xbd600000,
        0xbd600000, 0x3ec80000, 0xbe000000, 0x3ea40000, 0xbc000000, 0x3e400000, 0xbe600000, 0x3e8c0000,
        0x3dd00000, 0xbd800000, 0x3e400000, 0x3db00000, 0x3c000000, 0x3de00000, 0x3f200000, 0xbe380000,
        0xbea40000, 0x3ed80000, 0xbd000000, 0x3e080000, 0xbd900000, 0x3f100000, 0xbf280000, 0x3db00000,
    ];
    /// qt 44, 136 B / 256 elems.
    const EXPECTED_MQ4G256V2: [u32; 256] = [
        0x40640000, 0xc0c48000, 0x408b8000, 0x40000000, 0x3f9a0000, 0xc0870000, 0xc0320000, 0x3fea0000,
        0xbf280000, 0x40878000, 0xc0190000, 0xc0180000, 0x40350000, 0x3ea00000, 0x403e0000, 0xc01d0000,
        0x3f2c0000, 0x3f580000, 0x3ec00000, 0x3ffa0000, 0xc09a0000, 0xc0630000, 0xc0ca8000, 0xbfac0000,
        0xc01d0000, 0xc0bb0000, 0x40120000, 0x40390000, 0xc0b40000, 0x3f9a0000, 0xc0a38000, 0xbff00000,
        0xc0f98000, 0x40040000, 0x3f980000, 0xbf9e0000, 0xc0a80000, 0x40fa8000, 0xc0010000, 0x3ec00000,
        0xc05d0000, 0xbef00000, 0xbf900000, 0xbfce0000, 0x408b0000, 0x40858000, 0x410ac000, 0xc0140000,
        0x3f280000, 0xc0190000, 0x40d98000, 0x40cc0000, 0xc0130000, 0x40060000, 0x3fd80000, 0x3fea0000,
        0x40d30000, 0x3fa20000, 0xc0210000, 0xc03e0000, 0x40310000, 0xbdc00000, 0xc01a0000, 0xbfd20000,
        0x3fc60000, 0x3d000000, 0xbf600000, 0xc0090000, 0xc0ee0000, 0x3f5c0000, 0xbfe60000, 0xc0970000,
        0xc0250000, 0x40810000, 0xbe200000, 0x3fee0000, 0xc0300000, 0xbf4c0000, 0xc1004000, 0x40600000,
        0xbd000000, 0x3ed80000, 0xbf440000, 0xc0260000, 0xc06d0000, 0x00000000, 0xc0580000, 0xc0410000,
        0xc0000000, 0x408f8000, 0x40898000, 0xbf080000, 0xc02d0000, 0xbfb40000, 0xc0500000, 0x40e08000,
        0xc06d0000, 0xbfe00000, 0xbfa80000, 0xc0c18000, 0x408b0000, 0x40330000, 0x3fda0000, 0xc0890000,
        0x3f7c0000, 0xc06c0000, 0x41518000, 0xbfe20000, 0x402e0000, 0xbe700000, 0x3fbe0000, 0xbf000000,
        0x40ca0000, 0x4108c000, 0x40410000, 0x3f080000, 0xbe500000, 0x403e0000, 0xbfa00000, 0xbfbe0000,
        0xbf780000, 0x402f0000, 0x3c800000, 0xbfb40000, 0xc07f0000, 0x40080000, 0x401c0000, 0xc0848000,
        0x40800000, 0xc0b98000, 0xc08a8000, 0xc00c0000, 0xbfae0000, 0x408b0000, 0xc02a0000, 0x40070000,
        0xbf480000, 0x409a8000, 0x40130000, 0xc0140000, 0x405f0000, 0x00000000, 0xc0560000, 0xbfee0000,
        0xbec80000, 0xbf9c0000, 0xbd800000, 0xc01f0000, 0xc0900000, 0x40690000, 0x40a78000, 0xc00e0000,
        0xc0270000, 0x40b30000, 0xc02a0000, 0xc0430000, 0x40ae0000, 0xbf3c0000, 0xc0a28000, 0x3f700000,
        0xc0e48000, 0x3fe00000, 0xbfb00000, 0x3fca0000, 0x40a20000, 0x40e78000, 0xc01b0000, 0xbd800000,
        0x40370000, 0x3dc00000, 0xbf300000, 0x3faa0000, 0x408f0000, 0xc0988000, 0xc1044000, 0xbff00000,
        0x3f480000, 0xc03b0000, 0xc0cc8000, 0x40b20000, 0x40090000, 0xbf9c0000, 0x3fb00000, 0xbffe0000,
        0xc0c70000, 0x3f960000, 0x40430000, 0xc0160000, 0xc00b0000, 0x3e900000, 0x3fc40000, 0x3fd60000,
        0x3fb20000, 0x3ed00000, 0xbf100000, 0xc01b0000, 0xc0f80000, 0x3ee80000, 0xc0110000, 0x409b0000,
        0xc0170000, 0x407a0000, 0x3dc00000, 0x3f8a0000, 0x400c0000, 0xbf920000, 0x4102c000, 0x40740000,
        0xbef00000, 0x3f240000, 0x3f0c0000, 0x402e0000, 0xc0770000, 0xbf700000, 0xc04c0000, 0xc03b0000,
        0x3fc80000, 0x40928000, 0xc0a08000, 0xbe900000, 0xc0470000, 0x40120000, 0xc0640000, 0xc0d98000,
        0x40570000, 0xbf700000, 0x3f800000, 0x40d08000, 0x40870000, 0xc0610000, 0xc0170000, 0xc05a0000,
        0xbf2c0000, 0xc0380000, 0x414d8000, 0xbfb60000, 0xc02e0000, 0xbf0c0000, 0x3ffa0000, 0x3e400000,
        0x40e40000, 0x40f88000, 0x407b0000, 0xbeb00000, 0xbec80000, 0x40060000, 0xbf500000, 0xbfda0000,
        0x3e600000, 0xc0550000, 0x3e980000, 0x3f280000, 0xc0868000, 0x3fb80000, 0x40280000, 0x40430000,
    ];
    /// qt 15, 200 B / 256 elems.
    const EXPECTED_MQ6G256: [u32; 256] = [
        0xbd480000, 0xbdd80000, 0x3dcc0000, 0xbe100000, 0xbe100000, 0xbd780000, 0xbe0c0000, 0xbe020000,
        0x3e500000, 0x3e420000, 0xbee80000, 0x3d680000, 0x3e740000, 0x3d280000, 0xbd800000, 0xbcb00000,
        0xbd780000, 0x3df80000, 0x3e420000, 0xbe500000, 0xbe3c0000, 0xbead0000, 0xbe180000, 0xbdd40000,
        0x3d880000, 0xbd9c0000, 0x3d980000, 0x3cb00000, 0xbc000000, 0x3dcc0000, 0xbd700000, 0xbe1e0000,
        0xbe3e0000, 0xbdc00000, 0x3d8c0000, 0x3e820000, 0xbe580000, 0x3eab0000, 0xbd300000, 0x3e2a0000,
        0xbd600000, 0xbe6a0000, 0xbdf00000, 0x3dd40000, 0xbc800000, 0x3e520000, 0x3e9a0000, 0xbe3e0000,
        0x3c900000, 0x3d000000, 0x3e9f0000, 0x3e9e0000, 0xbdf80000, 0xbb000000, 0x3e780000, 0x3eb10000,
        0x3c800000, 0xbdbc0000, 0xbea80000, 0xbe830000, 0x3ed00000, 0xbdc40000, 0xbe7c0000, 0xbdcc0000,
        0xbc000000, 0xbe060000, 0xbd100000, 0xbe020000, 0xbe4a0000, 0x3e5c0000, 0x3bc00000, 0xbe380000,
        0xbd700000, 0x3e260000, 0xbd900000, 0x3e7a0000, 0xbe140000, 0xbe1a0000, 0x3ca00000, 0x3dfc0000,
        0xbea60000, 0xbee10000, 0x3e980000, 0xbe020000, 0x3dcc0000, 0x3e1c0000, 0x3c200000, 0xbdf00000,
        0xbd000000, 0x3e0a0000, 0xbeaa0000, 0x3d280000, 0xbc000000, 0xbde40000, 0x3ec80000, 0x3e5e0000,
        0xbe4c0000, 0x3d940000, 0xbeb40000, 0xbe830000, 0x3e360000, 0xbd300000, 0xbddc0000, 0xbea80000,
        0xbe240000, 0xbd280000, 0x3f080000, 0xbc900000, 0xbe800000, 0x3ecd0000, 0xbd900000, 0xbcd00000,
        0x3ed00000, 0x3ec90000, 0xbdd80000, 0x3e1a0000, 0xbe5a0000, 0x3ea20000, 0xbcf00000, 0x3ea80000,
        0x3e240000, 0x3d380000, 0x3e480000, 0xbd940000, 0xbe980000, 0xbb000000, 0xbe300000, 0xbf248000,
        0x3e7a0000, 0xbcc00000, 0xbe2a0000, 0xbe4c0000, 0xbdfc0000, 0x3e8c0000, 0xbeed0000, 0x3c000000,
        0xbe440000, 0x3e1e0000, 0xbe2c0000, 0xbd940000, 0xbd080000, 0x3c800000, 0xbecb0000, 0x3e800000,
        0xbdb40000, 0xbca00000, 0x3e0e0000, 0xbe300000, 0x3da40000, 0xbd980000, 0xbd940000, 0xbe640000,
        0x3d700000, 0x3e1e0000, 0x3eba0000, 0xbe970000, 0x3dac0000, 0xbde80000, 0x3d8c0000, 0xbde80000,
        0xbdac0000, 0x3e300000, 0xbe2e0000, 0xbdc80000, 0x3e320000, 0x3d980000, 0x3c900000, 0xbe740000,
        0x3e080000, 0xbed30000, 0xbdd00000, 0xbdf40000, 0x3eaf0000, 0xbe680000, 0xbe5a0000, 0xbd000000,
        0x3e8d0000, 0x00000000, 0xbdec0000, 0xbeb20000, 0xbd380000, 0x3d700000, 0x3d180000, 0x3e4c0000,
        0xbe840000, 0x3df40000, 0xbcc00000, 0xbc600000, 0x3e260000, 0x3d980000, 0x3cd00000, 0xbb800000,
        0x3d700000, 0x3e2a0000, 0xbe040000, 0xbe2e0000, 0xbe940000, 0xbd940000, 0xbb800000, 0x3e4a0000,
        0xbd880000, 0xbcb00000, 0x3da80000, 0xbe660000, 0x3d280000, 0x3e200000, 0x3e0a0000, 0x3e100000,
        0x3d500000, 0xbd780000, 0xbdd80000, 0xbdac0000, 0xbd400000, 0xbe1e0000, 0x3e540000, 0xbde40000,
        0x3de80000, 0x3c200000, 0xbe240000, 0xbb000000, 0xbd580000, 0x3e3c0000, 0xbc600000, 0xbe340000,
        0xbe400000, 0x3e910000, 0x3d000000, 0x3e2a0000, 0x3ecc0000, 0xbea70000, 0x3d500000, 0xbe1e0000,
        0xbec60000, 0xbe060000, 0x3e1c0000, 0xbe220000, 0x3d080000, 0x3d100000, 0x3e4e0000, 0x3e4c0000,
        0x3d880000, 0xbc600000, 0x3b800000, 0x3cf00000, 0xbdb80000, 0x3d8c0000, 0x3d000000, 0xbdac0000,
        0xbdd80000, 0xbec50000, 0xbe640000, 0xbe660000, 0xbdfc0000, 0xbdc00000, 0x3e3a0000, 0xbdf00000,
    ];
    /// qt 17, 104 B / 256 elems.
    const EXPECTED_MQ3G256: [u32; 256] = [
        0xbdf00000, 0xbe9a0000, 0x3e840000, 0x3d880000, 0x3df00000, 0x3e3c0000, 0xbd400000, 0xbe4c0000,
        0x3dfc0000, 0x3e420000, 0xbecf0000, 0xbec10000, 0x3d080000, 0x3d780000, 0x3d380000, 0xbe890000,
        0xbe5c0000, 0x3d600000, 0xbdd80000, 0x3d000000, 0xbed60000, 0xbef40000, 0xbede0000, 0xbe780000,
        0xbd680000, 0x3d180000, 0x3efd0000, 0xbd280000, 0xbea70000, 0x3e3e0000, 0xbebf0000, 0xbf278000,
        0xbe9e0000, 0x3e480000, 0xbede0000, 0xbdd00000, 0xbee60000, 0x3efc0000, 0xbdb80000, 0x3e800000,
        0xbe2a0000, 0xbe1e0000, 0xbeb30000, 0xbd780000, 0x3d180000, 0xbe120000, 0x3ea30000, 0xbe910000,
        0x80000000, 0xbdc80000, 0x3f320000, 0x3f110000, 0xbd800000, 0x3ede0000, 0xbd400000, 0x3e3c0000,
        0x3e8f0000, 0x3e830000, 0xbd840000, 0xbee30000, 0x3e760000, 0x3e890000, 0xbead0000, 0xbe810000,
        0x3e760000, 0xbea90000, 0xbd480000, 0xbe2e0000, 0xbe930000, 0x3db40000, 0xbdac0000, 0xbeff0000,
        0xbd000000, 0x3e500000, 0x3e900000, 0x3ea80000, 0xbeba0000, 0x3dc80000, 0xbebe0000, 0x3e2c0000,
        0xbec30000, 0xbe020000, 0x3d380000, 0xbee10000, 0xbd9c0000, 0x3e870000, 0xbef90000, 0x3df40000,
        0xbeba0000, 0x3ee60000, 0xbe540000, 0x3e1c0000, 0xbeb00000, 0xbc800000, 0x3d400000, 0x3f040000,
        0xbcb00000, 0xbf0a8000, 0xbe320000, 0xbead0000, 0x3e160000, 0xbe990000, 0x3e1a0000, 0xbf058000,
        0x3e040000, 0x3da80000, 0x3fa58000, 0xbf090000, 0x80000000, 0xbc800000, 0xbe380000, 0xbe780000,
        0x3eef0000, 0x3f2c8000, 0xbd8c0000, 0x3b000000, 0xbd940000, 0x3f008000, 0x3e2a0000, 0xbdb40000,
        0x3ec00000, 0x3d000000, 0x3e480000, 0xbd900000, 0xbf2d0000, 0x3d880000, 0xbed20000, 0xbf770000,
        0x3f2d0000, 0xbec40000, 0xbe640000, 0xbe8c0000, 0x3da00000, 0x3ece0000, 0xbeb00000, 0xbdc80000,
        0xbe120000, 0x3e910000, 0xbd480000, 0xbd280000, 0x3ec30000, 0xbddc0000, 0xbef90000, 0x3ea10000,
        0x3e600000, 0xbe5c0000, 0x3e800000, 0xbeae0000, 0x3e340000, 0xbd000000, 0x3da80000, 0xbe900000,
        0xbe4a0000, 0x3ec10000, 0x3dec0000, 0xbee30000, 0x3e3a0000, 0x3dac0000, 0xbd480000, 0xbedb0000,
        0xbea80000, 0x3d100000, 0xbea00000, 0x3e0c0000, 0x3de80000, 0x3eb80000, 0xbea20000, 0xbea00000,
        0x3cd00000, 0xbe320000, 0x3d080000, 0xbd080000, 0x3efd0000, 0xbeed0000, 0xbf2a8000, 0x3e420000,
        0x3ed60000, 0xbed00000, 0xbd880000, 0xbe200000, 0xbdd00000, 0x3e340000, 0xbdf00000, 0x3c400000,
        0xbe830000, 0x3dac0000, 0x3e520000, 0xbd8c0000, 0x3dbc0000, 0x3dac0000, 0xbe620000, 0xbda40000,
        0xbe5a0000, 0x3ebf0000, 0x3cb00000, 0xbe9f0000, 0xbed90000, 0xbe320000, 0xbdc40000, 0x3ec90000,
        0xbdb80000, 0xbca00000, 0xbe540000, 0xbe8e0000, 0x3d500000, 0x3e040000, 0x3e860000, 0x3e9a0000,
        0xbe320000, 0x3eb70000, 0xbd9c0000, 0x3d480000, 0xbee10000, 0xbecb0000, 0xbcf00000, 0xbe6a0000,
        0xbe100000, 0x3eb00000, 0xbf360000, 0xbd800000, 0xbee80000, 0x3e940000, 0xbe700000, 0xbe900000,
        0x3e9f0000, 0x3f038000, 0xbe760000, 0x3f168000, 0x3ee10000, 0xbf0a8000, 0xbeab0000, 0x3e0e0000,
        0xbe500000, 0xbe940000, 0x3eb40000, 0xbdd00000, 0xbdf00000, 0xbe840000, 0x3eb40000, 0xbde00000,
        0x3e9b0000, 0x3e020000, 0x3e220000, 0xbe560000, 0xbdc40000, 0xbe810000, 0xbefd0000, 0x3d280000,
        0x3e540000, 0xbf010000, 0x3c400000, 0xbe540000, 0xbd500000, 0xbed20000, 0x3f1d0000, 0xbe8e0000,
    ];
    /// qt 20, 112 B / 256 elems.
    const EXPECTED_MQ3G256LLOYD: [u32; 256] = [
        0x3ee80000, 0x3ec00000, 0x3fa00000, 0x3f4c0000, 0x3d000000, 0xbe100000, 0xbd400000, 0x3f080000,
        0xbef00000, 0x3f340000, 0xbd400000, 0xbef00000, 0xbef80000, 0xbe000000, 0x3ea00000, 0x3f0c0000,
        0x3f800000, 0x3c800000, 0xbee80000, 0xbee00000, 0xbe700000, 0xbec00000, 0x3f200000, 0x3f820000,
        0xbf240000, 0x3f280000, 0xbdc00000, 0x3f0c0000, 0xbed00000, 0x3e500000, 0x3d400000, 0xbd000000,
        0x3e700000, 0xbe600000, 0xbe200000, 0x3da00000, 0xbec00000, 0x3eb80000, 0x3e700000, 0x3f600000,
        0xbf080000, 0x3f7c0000, 0x3de00000, 0xbf780000, 0xbe980000, 0xbe000000, 0x3f300000, 0xc01f0000,
        0xbd800000, 0x3f7c0000, 0x3e980000, 0x3f200000, 0x3f3c0000, 0x3ea00000, 0xbd800000, 0xbf440000,
        0x3e300000, 0xbee00000, 0x3d800000, 0xbf0c0000, 0x3f400000, 0xbe100000, 0xbf2c0000, 0x3f700000,
        0xbf580000, 0x3da00000, 0x3e880000, 0x80000000, 0xbec80000, 0xbf200000, 0xbf680000, 0x3d400000,
        0xbf0c0000, 0xbf080000, 0xbf500000, 0x3f340000, 0x3f400000, 0x3e880000, 0xbf040000, 0xbef00000,
        0xbf440000, 0xbec00000, 0x3f380000, 0xbef80000, 0x3f600000, 0xbc800000, 0x3e980000, 0x3e200000,
        0xbd800000, 0xbf540000, 0xbe700000, 0x3fa40000, 0xbda00000, 0xbf100000, 0xbf180000, 0x3d400000,
        0xbf600000, 0xbf1c0000, 0xbeb80000, 0xbf280000, 0x3da00000, 0xbf780000, 0xbee00000, 0xbea80000,
        0xbf1c0000, 0x3f080000, 0xbe400000, 0x3f0c0000, 0xbec00000, 0xbde00000, 0x3ea80000, 0xbeb00000,
        0x3f2c0000, 0xbf880000, 0x3dc00000, 0x3c800000, 0xbf400000, 0x3f1c0000, 0xbfa20000, 0x3dc00000,
        0xbef00000, 0x3f540000, 0x3e300000, 0x3f500000, 0x3eb80000, 0x3ed00000, 0x3e400000, 0xbf0c0000,
        0x3ed80000, 0x3fc00000, 0xbe400000, 0xbe100000, 0xbf180000, 0xbf820000, 0x3eb80000, 0xbeb00000,
        0xbe000000, 0x3e880000, 0x3ed80000, 0x3f980000, 0x3da00000, 0xbf180000, 0xbe900000, 0xbe500000,
        0x3e800000, 0x3e880000, 0x3f8e0000, 0xbf600000, 0x3ef80000, 0x3f200000, 0xbd800000, 0x3ed80000,
        0xbe100000, 0x3e400000, 0xbf300000, 0x3c800000, 0xbea00000, 0xbf240000, 0x3d400000, 0xbf400000,
        0x3f6c0000, 0xbec00000, 0x80000000, 0x3f960000, 0x3e900000, 0x3f2c0000, 0x3f1c0000, 0x3f580000,
        0xbf840000, 0x3ed80000, 0x3f920000, 0xbf080000, 0x3eb80000, 0x3ec00000, 0x3ec00000, 0xbf440000,
        0xbf280000, 0x3f6c0000, 0x3e500000, 0xbf280000, 0xbf3c0000, 0xbf280000, 0x3ef00000, 0x3f140000,
        0x3ef80000, 0xbe400000, 0x3e000000, 0x3e300000, 0xbf200000, 0xbf6c0000, 0xbf860000, 0xbe000000,
        0xbe000000, 0xbef80000, 0xbf6c0000, 0xbed00000, 0x3ef80000, 0x3f840000, 0xbd800000, 0xbf0c0000,
        0x3f6c0000, 0x3f940000, 0xbe000000, 0x3f1c0000, 0x3ee00000, 0x3f0c0000, 0x3f3c0000, 0x3f380000,
        0xbf4c0000, 0x3d000000, 0x3f300000, 0x3d400000, 0x3e200000, 0x3d400000, 0xbf440000, 0x3e800000,
        0xbee00000, 0xbf2c0000, 0xbf6c0000, 0xbdc00000, 0xbf7c0000, 0x00000000, 0xbe200000, 0x3ed80000,
        0xbec00000, 0xbee80000, 0xbde00000, 0x3f280000, 0xbf740000, 0xbd000000, 0x3f300000, 0x3f440000,
        0x3f8a0000, 0x3e000000, 0x3e200000, 0x3eb80000, 0xbf580000, 0x3de00000, 0xbe980000, 0xbf200000,
        0xbef80000, 0x3f300000, 0x3dc00000, 0x3e700000, 0x3d800000, 0x3e880000, 0xbd400000, 0xbdc00000,
        0xbe000000, 0x3d400000, 0x3ed80000, 0xbe600000, 0xbec80000, 0xbe400000, 0xbe200000, 0x3e300000,
    ];

    /// qt 8, 200 B / 256 elems.
    const EXPECTED_HFQ6G256: [u32; 256] = [
        0x3e8c0000, 0x3e000000, 0x3d200000, 0x3df00000, 0x3e100000, 0x3de00000, 0xbd000000, 0x3eac0000,
        0x3c000000, 0x3d800000, 0x3e8c0000, 0x3d400000, 0xbe000000, 0x3d400000, 0x3e500000, 0x3e880000,
        0x3e780000, 0x3cc00000, 0x3c000000, 0xbc800000, 0x3de00000, 0xbcc00000, 0xbd800000, 0x3e500000,
        0xbcc00000, 0xbd200000, 0x3ebc0000, 0xbdb00000, 0x3eb00000, 0xbdc00000, 0x3e300000, 0x3e080000,
        0x3e580000, 0xbde00000, 0x3dd00000, 0x3eb40000, 0x3da00000, 0x3eb00000, 0xbdd00000, 0x3d900000,
        0xbd600000, 0x3ea80000, 0x3ea80000, 0x3e900000, 0x3ea00000, 0x3e9c0000, 0x3e080000, 0x00000000,
        0x3e380000, 0x3e840000, 0x3d800000, 0x3e600000, 0x3d400000, 0x3e780000, 0xbc000000, 0xbd800000,
        0xbdb00000, 0x3e480000, 0x3e980000, 0x3e180000, 0x3e900000, 0x3e300000, 0x3e680000, 0x3ebc0000,
        0x3e180000, 0x3e000000, 0x3d000000, 0x3db00000, 0x3c800000, 0x3de00000, 0xbd200000, 0x3e9c0000,
        0xbdf00000, 0x3d800000, 0x3e880000, 0x3c800000, 0x3e800000, 0x3d200000, 0x3e480000, 0x3e700000,
        0x3df00000, 0x3cc00000, 0x00000000, 0xbd400000, 0xbc800000, 0xbcc00000, 0xbd900000, 0x3e300000,
        0x3eb40000, 0xbd400000, 0x3eb80000, 0xbdf00000, 0x3e600000, 0xbdc00000, 0x3e280000, 0x3dd00000,
        0x3db00000, 0xbde00000, 0x3dc00000, 0x3ea40000, 0xbd400000, 0x3eb00000, 0xbde00000, 0x3d200000,
        0x3ea40000, 0x3ea40000, 0x3ea40000, 0x3e800000, 0x3e400000, 0x3e9c0000, 0x3e000000, 0xbd000000,
        0x3d600000, 0x3e840000, 0x3d600000, 0x3e400000, 0xbda00000, 0x3e780000, 0xbc800000, 0xbdc00000,
        0x3e940000, 0x3e400000, 0x3e940000, 0x3df00000, 0x3e200000, 0x3e300000, 0x3e600000, 0x3eac0000,
        0x3cc00000, 0x3e000000, 0x3cc00000, 0x3d600000, 0xbde00000, 0x3de00000, 0xbd400000, 0x3e8c0000,
        0x3e840000, 0x3db00000, 0x3e840000, 0xbc800000, 0x3e000000, 0x3d200000, 0x3e400000, 0x3e500000,
        0xbc000000, 0x3cc00000, 0x3df00000, 0xbda00000, 0x3eb80000, 0xbd000000, 0xbda00000, 0x3e100000,
        0x3e680000, 0xbd400000, 0x3eb40000, 0x3eb40000, 0x3dc00000, 0xbdc00000, 0x3e200000, 0x3d900000,
        0xbd200000, 0xbde00000, 0x3db00000, 0x3e940000, 0x3ea80000, 0x3ebc0000, 0xbdf00000, 0x3c000000,
        0x3e480000, 0x3ea40000, 0x3ea00000, 0x3e600000, 0x3d800000, 0x3e9c0000, 0x3e780000, 0xbd800000,
        0xbd900000, 0x3e840000, 0x3d400000, 0x3e200000, 0x3e980000, 0x3e700000, 0xbcc00000, 0xbe000000,
        0x3e280000, 0x3e400000, 0x3e900000, 0x3db00000, 0x3d000000, 0x3e300000, 0x3e580000, 0x3e9c0000,
        0xbdd00000, 0x3e000000, 0x3c800000, 0x3cc00000, 0x3e880000, 0x3dd00000, 0xbd600000, 0x3e780000,
        0x3e080000, 0x3db00000, 0x3e800000, 0xbd400000, 0x00000000, 0x3d200000, 0x3e380000, 0x3e300000,
        0x3ebc0000, 0x3c800000, 0x3de00000, 0xbde00000, 0x3e700000, 0xbd000000, 0xbdb00000, 0x3de00000,
        0x3dd00000, 0xbd400000, 0x3eb00000, 0x3ea40000, 0xbd000000, 0xbdc00000, 0x3e180000, 0x3d200000,
        0x3eac0000, 0xbdf00000, 0x3da00000, 0x3e840000, 0x3e500000, 0x3ebc0000, 0xbe000000, 0xbcc00000,
        0x3d900000, 0x3ea40000, 0x3e9c0000, 0x3e400000, 0xbd800000, 0x3e9c0000, 0x3e700000, 0xbdc00000,
        0x3e9c0000, 0x3e800000, 0x3d200000, 0x3e000000, 0x3e300000, 0x3e700000, 0xbd000000, 0x3eb00000,
        0x3d200000, 0x3e400000, 0x3e8c0000, 0x3d600000, 0xbdc00000, 0x3e300000, 0x3e500000, 0x3e8c0000,
    ];
    /// qt 6, 136 B / 256 elems.
    const EXPECTED_HFQ4G256: [u32; 256] = [
        0xbed00000, 0xbed00000, 0xbe800000, 0xbeb00000, 0xbdc00000, 0xbe900000, 0xbee00000, 0xbe400000,
        0xbe900000, 0xbe000000, 0xbe000000, 0xbd800000, 0xbef00000, 0xbef00000, 0xbea00000, 0xbed00000,
        0xbe200000, 0xbeb00000, 0xbf000000, 0xbe800000, 0xbeb00000, 0xbe400000, 0xbe400000, 0xbe000000,
        0xbd000000, 0xbd800000, 0xbec00000, 0xbef00000, 0xbe600000, 0xbed00000, 0xbd800000, 0xbeb00000,
        0xbed00000, 0xbe800000, 0xbe800000, 0xbe400000, 0xbdc00000, 0xbe000000, 0xbee00000, 0xbd000000,
        0xbe900000, 0xbef00000, 0xbe000000, 0xbed00000, 0xbef00000, 0xbea00000, 0xbea00000, 0xbe800000,
        0xbe200000, 0xbe400000, 0xbf000000, 0xbdc00000, 0xbeb00000, 0xbd000000, 0xbe400000, 0xbef00000,
        0xbd000000, 0xbed00000, 0xbec00000, 0xbea00000, 0xbe600000, 0xbe800000, 0xbd800000, 0xbe400000,
        0xbed00000, 0xbdc00000, 0xbe800000, 0xbd000000, 0xbdc00000, 0xbef00000, 0xbee00000, 0xbec00000,
        0xbe900000, 0xbea00000, 0xbe000000, 0xbe800000, 0xbef00000, 0xbe200000, 0xbea00000, 0xbdc00000,
        0xbe200000, 0xbd000000, 0xbf000000, 0xbee00000, 0xbeb00000, 0xbec00000, 0xbe400000, 0xbea00000,
        0xbd000000, 0xbe800000, 0xbec00000, 0xbe200000, 0xbe600000, 0xbdc00000, 0xbd800000, 0xbd000000,
        0xbed00000, 0xbee00000, 0xbe800000, 0xbec00000, 0xbdc00000, 0xbea00000, 0xbee00000, 0xbe600000,
        0xbe900000, 0xbe200000, 0xbe000000, 0xbdc00000, 0xbef00000, 0xbf000000, 0xbea00000, 0xbee00000,
        0xbe200000, 0xbec00000, 0xbf000000, 0xbe900000, 0xbeb00000, 0xbe600000, 0xbe400000, 0xbe200000,
        0xbd000000, 0xbdc00000, 0xbec00000, 0xbf000000, 0xbe600000, 0xbee00000, 0xbd800000, 0xbec00000,
        0xbed00000, 0xbe900000, 0xbe800000, 0xbe600000, 0xbdc00000, 0xbe200000, 0xbee00000, 0xbd800000,
        0xbe900000, 0xbf000000, 0xbe000000, 0xbee00000, 0xbef00000, 0xbeb00000, 0xbea00000, 0xbe900000,
        0xbe200000, 0xbe600000, 0xbf000000, 0xbe000000, 0xbeb00000, 0xbd800000, 0xbe400000, 0xbf000000,
        0xbd000000, 0xbee00000, 0xbec00000, 0xbeb00000, 0xbe600000, 0xbe900000, 0xbd800000, 0xbe600000,
        0xbed00000, 0xbe000000, 0xbe800000, 0xbd800000, 0xbdc00000, 0xbf000000, 0xbee00000, 0xbed00000,
        0xbe900000, 0xbeb00000, 0xbe000000, 0xbe900000, 0xbef00000, 0xbe400000, 0xbea00000, 0xbe000000,
        0xbe200000, 0xbd800000, 0xbf000000, 0xbef00000, 0xbeb00000, 0xbed00000, 0xbe400000, 0xbeb00000,
        0xbd000000, 0xbe900000, 0xbec00000, 0xbe400000, 0xbe600000, 0xbe000000, 0xbd800000, 0xbd800000,
        0xbed00000, 0xbef00000, 0xbe800000, 0xbed00000, 0xbdc00000, 0xbeb00000, 0xbee00000, 0xbe800000,
        0xbe900000, 0xbe400000, 0xbe000000, 0xbe000000, 0xbef00000, 0xbd000000, 0xbea00000, 0xbef00000,
        0xbe200000, 0xbed00000, 0xbf000000, 0xbea00000, 0xbeb00000, 0xbe800000, 0xbe400000, 0xbe400000,
        0xbd000000, 0xbe000000, 0xbec00000, 0xbd000000, 0xbe600000, 0xbef00000, 0xbd800000, 0xbed00000,
        0xbed00000, 0xbea00000, 0xbe800000, 0xbe800000, 0xbdc00000, 0xbe400000, 0xbee00000, 0xbdc00000,
        0xbe900000, 0xbd000000, 0xbe000000, 0xbef00000, 0xbef00000, 0xbec00000, 0xbea00000, 0xbea00000,
        0xbe200000, 0xbe800000, 0xbf000000, 0xbe200000, 0xbeb00000, 0xbdc00000, 0xbe400000, 0xbd000000,
        0xbd000000, 0xbef00000, 0xbec00000, 0xbec00000, 0xbe600000, 0xbea00000, 0xbd800000, 0xbe800000,
    ];
    /// qt 1, 512 B / 256 elems.
    const EXPECTED_F16: [u32; 256] = [
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
    ];
    /// qt 2, 1024 B / 256 elems.
    const EXPECTED_F32: [u32; 256] = [
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
    ];
    /// qt 16, 512 B / 256 elems.
    const EXPECTED_BF16: [u32; 256] = [
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
    ];
    /// qt 3, 34 B / 32 elems.
    const EXPECTED_Q8F16: [u32; 32] = [
        0x422a0000, 0x42740000, 0xc2420000, 0xc1f00000, 0xc1380000, 0x40e00000, 0x41cc0000, 0x42300000,
        0x427a0000, 0xc23c0000, 0xc1e40000, 0xc1200000, 0x41080000, 0x41d80000, 0x42360000, 0xc2800000,
        0xc2360000, 0xc1d80000, 0xc1080000, 0x41200000, 0x41e40000, 0x423c0000, 0xc27a0000, 0xc2300000,
        0xc1cc0000, 0xc0e00000, 0x41380000, 0x41f00000, 0x42420000, 0xc2740000, 0xc22a0000, 0xc1c00000,
    ];

    fn assert_bits(got: &[f32], expected: &[u32], what: &str) {
        assert_eq!(got.len(), expected.len(), "{what}: length");
        for (i, (g, e)) in got.iter().zip(expected).enumerate() {
            assert_eq!(
                g.to_bits(),
                *e,
                "{what}: element {i} = {g} (0x{:08x}) != {} (0x{e:08x})",
                g.to_bits(),
                f32::from_bits(*e)
            );
        }
    }

    #[test]
    fn dequant_group_matches_canonical_decoder_bits() {
        for (q, expected) in [
            (CpuQuant::Mq4G256, &EXPECTED_MQ4G256[..]),
            (CpuQuant::Mq4G256V2, &EXPECTED_MQ4G256V2[..]),
            (CpuQuant::Mq6G256, &EXPECTED_MQ6G256[..]),
            (CpuQuant::Mq3G256, &EXPECTED_MQ3G256[..]),
            (CpuQuant::Mq3G256Lloyd, &EXPECTED_MQ3G256LLOYD[..]),
            (CpuQuant::Hfq6G256, &EXPECTED_HFQ6G256[..]),
            (CpuQuant::Hfq4G256, &EXPECTED_HFQ4G256[..]),
            (CpuQuant::F16, &EXPECTED_F16[..]),
            (CpuQuant::F32, &EXPECTED_F32[..]),
            (CpuQuant::Bf16, &EXPECTED_BF16[..]),
            (CpuQuant::Q8F16, &EXPECTED_Q8F16[..]),
        ] {
            let packed = fixture(q);
            let mut out = vec![0.0f32; q.group_elems()];
            dequant_group(q, &packed, &mut out);
            assert_bits(&out, expected, &format!("dequant_group({q:?})"));
        }
    }

    #[test]
    fn decode_group_codes_is_the_affine_decode() {
        // The codes decode must be the un-rotated half of `dequant_group`: for a
        // rotated format, the affine decode has no 256-element structure of its
        // own, so its first 8 values are exactly `scale * code + zero` on the
        // nibble stream the fixture writes.
        let packed = fixture(CpuQuant::Mq4G256);
        let mut codes = [0.0f32; 256];
        decode_group_codes(CpuQuant::Mq4G256, &packed, &mut codes);
        let scale = f32::from_le_bytes([packed[0], packed[1], packed[2], packed[3]]);
        let zero = f32::from_le_bytes([packed[4], packed[5], packed[6], packed[7]]);
        for i in 0..256 {
            let byte = packed[8 + i / 2];
            let nibble = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };
            assert_eq!(codes[i], scale * nibble as f32 + zero, "element {i}");
        }

        // And for an unrotated format the two decodes are identical.
        let mut a = vec![0.0f32; 256];
        let mut b = vec![0.0f32; 256];
        decode_group_codes(CpuQuant::Hfq4G256, &fixture(CpuQuant::Hfq4G256), &mut a);
        dequant_group(CpuQuant::Hfq4G256, &fixture(CpuQuant::Hfq4G256), &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn quant_type_round_trip_and_group_sizes() {
        for (qt, q) in [
            (1u8, CpuQuant::F16),
            (2, CpuQuant::F32),
            (3, CpuQuant::Q8F16),
            (6, CpuQuant::Hfq4G256),
            (8, CpuQuant::Hfq6G256),
            (13, CpuQuant::Mq4G256),
            (15, CpuQuant::Mq6G256),
            (17, CpuQuant::Mq3G256),
            (20, CpuQuant::Mq3G256Lloyd),
            (44, CpuQuant::Mq4G256V2),
        ] {
            assert_eq!(CpuQuant::from_quant_type(qt), Some(q), "qt {qt}");
        }
        // No CPU implementation: these stay on the GPU over PCIe.
        for qt in [0u8, 4, 5, 7, 11, 12, 14, 18, 19, 30, 40, 41, 45, 47, 50, 255] {
            assert_eq!(CpuQuant::from_quant_type(qt), None, "qt {qt}");
        }
        // The layout table, pinned literally: `gemv`'s row stride is
        // `(k / group_elems) * group_bytes`, so a wrong entry here is a wrong
        // row pointer, not a rounding difference.
        for (q, ge, gb) in [
            (CpuQuant::Mq4G256, 256usize, 136usize),
            (CpuQuant::Mq4G256V2, 256, 136),
            (CpuQuant::Mq6G256, 256, 200),
            (CpuQuant::Mq3G256, 256, 104),
            (CpuQuant::Mq3G256Lloyd, 256, 112),
            (CpuQuant::Hfq6G256, 256, 200),
            (CpuQuant::Hfq4G256, 256, 136),
            (CpuQuant::F16, 256, 512),
            (CpuQuant::F32, 256, 1024),
            (CpuQuant::Bf16, 256, 512),
            (CpuQuant::Q8F16, 32, 34),
        ] {
            assert_eq!((q.group_elems(), q.group_bytes()), (ge, gb), "{q:?}");
            assert_eq!(
                q.is_fwht_g256(),
                matches!(
                    q,
                    CpuQuant::Mq4G256
                        | CpuQuant::Mq4G256V2
                        | CpuQuant::Mq6G256
                        | CpuQuant::Mq3G256
                        | CpuQuant::Mq3G256Lloyd
                ),
                "{q:?} rotation"
            );
        }
    }

    #[test]
    fn f16_widening_is_ieee() {
        // Every value the generator uses, plus the awkward encodings: subnormal,
        // zero, min/max normal, inf, NaN.
        for bits in EXPECTED_F16_BITS {
            let want = half::f16::from_bits(bits).to_f32();
            assert_eq!(half_f16_to_f32(bits), want, "bits 0x{bits:04x}");
        }
        for bits in [0x0001u16, 0x03ff, 0x0400, 0x7bff, 0x7c00, 0x7c01, 0xfc00, 0x8000, 0xffff] {
            let want = half::f16::from_bits(bits).to_f32();
            let got = half_f16_to_f32(bits);
            if want.is_nan() {
                assert!(got.is_nan(), "bits 0x{bits:04x}: {got} is not NaN");
            } else {
                assert_eq!(got, want, "bits 0x{bits:04x}");
            }
        }
    }

    #[test]
    fn divide_by_awq_scale_is_per_channel_and_pre_rotation() {
        // (W·s)·(x/s) = W·x: the divide must cancel the pre-scaled weight, so
        // dividing then rotating a probe activation must equal rotating the
        // unscaled activation of a (W/s)-weighted... — the property that matters
        // and is checkable in isolation is that the divide is per channel and
        // precedes the rotation, i.e. R(x/s) != R(x)/s in general and equals a
        // hand-built R applied to x/s.
        let scale = [2.0f32, 4.0, 0.5, 8.0];
        let mut x = [4.0f32, 8.0, 1.0, 16.0];
        divide_by_awq_scale(&mut x, &scale);
        assert_eq!(x, [2.0, 2.0, 2.0, 2.0]);
        // A longer scale than activation is accepted (the loader's buffer is
        // exactly k, but the contract is `>=`), a shorter one is a caller bug.
        divide_by_awq_scale(&mut x, &[1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(x, [2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "awq scale has 3 channels")]
    fn divide_by_awq_scale_rejects_a_short_scale() {
        let mut x = [0.0f32; 4];
        divide_by_awq_scale(&mut x, &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn fwht_signs_are_pinned_to_the_existing_generator() {
        // Literal transcription of `KvCache::gen_fwht_signs(42, 256)`'s first 16
        // and last 4 entries, taken from the generator that produced the tables
        // above; a change in the LCG must fail here rather than silently produce
        // different weights under the same name.
        let s1 = fwht_signs(FWHT_SEED1, FWHT_N);
        let s2 = fwht_signs(FWHT_SEED2, FWHT_N);
        let lit = |v: &[f32]| -> Vec<f32> { v.to_vec() };
        assert_eq!(s1.len(), 256);
        assert_eq!(s2.len(), 256);
        assert_eq!(
            lit(&s1[..16]),
            vec![
                1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0,
                -1.0
            ],
            "signs1 head"
        );
        assert_eq!(
            lit(&s2[..16]),
            vec![
                1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
                1.0
            ],
            "signs2 head"
        );
        assert_eq!(lit(&s1[252..]), vec![-1.0, -1.0, 1.0, -1.0], "signs1 tail");
        assert_eq!(lit(&s2[252..]), vec![1.0, 1.0, -1.0, -1.0], "signs2 tail");
        // Balanced and unit-magnitude on any seed/length.
        for seed in [0u32, 1, 42, 1042, u32::MAX] {
            let s = fwht_signs(seed, 4096);
            assert!(s.iter().all(|v| *v == 1.0 || *v == -1.0));
            let sum: i32 = s.iter().map(|v| *v as i32).sum();
            assert!(sum.abs() < 200, "seed {seed} is badly imbalanced: {sum}");
        }
    }

    /// Independent oracle for the forward rotation: `R = 0.0625 * D2 * H * D1`
    /// where `H[i][j] = (-1)^popcount(i & j)` is the unnormalized
    /// Walsh-Hadamard matrix built from the definition, not from the butterfly
    /// (which is what `rotate_x` implements). `R e_j` is column `j`.
    #[test]
    fn rotate_x_matches_the_hadamard_matrix() {
        let s1 = fwht_signs(FWHT_SEED1, FWHT_N);
        let s2 = fwht_signs(FWHT_SEED2, FWHT_N);
        let r = |i: usize, j: usize| -> f32 {
            let sign = if ((i & j).count_ones() % 2) == 0 {
                1.0
            } else {
                -1.0
            };
            0.0625 * s2[i] * sign * s1[j]
        };
        for j in [0usize, 1, 17, 128, 255] {
            let mut x = vec![0.0f32; FWHT_N];
            x[j] = 1.0;
            rotate_x(&mut x);
            for i in 0..FWHT_N {
                assert_eq!(x[i], r(i, j), "column {j}, row {i}");
            }
        }
    }

    #[test]
    fn fwht256_inplace_inverts_rotate_x() {
        // Round trip on a spread of deterministic inputs, including a second
        // 256-block so a wrong block stride cannot pass.
        for seed in 0..3u64 {
            let mut x: Vec<f32> = (0..512)
                .map(|i| {
                    let v = (i as u64 * 2654435761 + seed * 7919) % 1000;
                    (v as f32 - 500.0) * 0.03125
                })
                .collect();
            let original = x.clone();
            rotate_x(&mut x);
            assert_ne!(x, original, "rotation is a no-op");
            for block in x.chunks_exact_mut(FWHT_N) {
                let group: &mut [f32; FWHT_N] = block.try_into().unwrap();
                fwht256_inplace(group, signs1(), signs2());
            }
            for (a, b) in x.iter().zip(&original) {
                assert!(
                    (a - b).abs() <= 1e-6 * b.abs().max(1.0),
                    "round trip drifted: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn rotate_x_preserves_norm_and_rejects_unaligned() {
        let mut x: Vec<f32> = (0..256).map(|i| (i as f32) * 0.5 - 64.0).collect();
        let before: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        rotate_x(&mut x);
        let after: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (before - after).abs() <= 1e-3 * before,
            "rotation is not norm preserving: {before} -> {after}"
        );
    }

    #[test]
    #[should_panic(expected = "not a multiple of 256")]
    fn rotate_x_panics_on_unaligned_input() {
        rotate_x(&mut [0.0f32; 255]);
    }
}
