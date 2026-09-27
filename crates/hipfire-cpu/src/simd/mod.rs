// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! SIMD kernels for the CPU matmul core.
//!
//! `std::simd` is nightly-only (`portable_simd`) and this crate targets stable,
//! so the vector kernels are `core::arch` intrinsics behind `#[target_feature]`
//! with runtime detection, plus the plain scalar path as both the fallback and
//! the reference the vector path is tested against.
//!
//! The vector path is a *throughput* choice, not a numerical one: both compute
//! ordinary `f32` sequences and are compared within a relative tolerance (see
//! `simd::tests`), never bit-for-bit — the offload feature's contract is
//! llama.cpp-level coherence, not device parity, and the vector path is what
//! makes the CPU side competitive with reading the same bytes over PCIe.
//!
//! Parallelism stays in [`crate::gemv`] (rayon over output rows); the kernels
//! here compute a single row so the two compose without either owning the other.

#[cfg(target_arch = "x86_64")]
mod x86;

use crate::quant::CpuQuant;

/// Whether the CPU reports AVX2 + FMA.
#[cfg(target_arch = "x86_64")]
pub fn avx2_available() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn avx2_available() -> bool {
    false
}

/// Whether the CPU additionally reports F16C, which the `Mq3G256V2` row dot
/// needs to widen its per-half fp16 scale/zero header.
///
/// Every AVX2 part in practice has F16C, but "in practice" is not a hardware
/// guarantee and the conversion is the kernel's only fp16 step, so it is
/// detected rather than assumed.
#[cfg(target_arch = "x86_64")]
pub fn avx2_f16c_available() -> bool {
    avx2_available() && std::arch::is_x86_feature_detected!("f16c")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn avx2_f16c_available() -> bool {
    false
}

/// Whether the vector row dot can run for format `q` on this CPU, honouring a
/// forced `requested` (see [`use_avx2`]).
///
/// The formats with a hand-written kernel are exactly the ones the kernels in
/// [`x86`] decode: `Mq4G256` and `Mq3G256V2`. Every other format keeps the
/// scalar decode, which is a throughput choice, never a correctness one.
pub fn row_dot_enabled(q: CpuQuant, requested: Option<bool>) -> bool {
    match q {
        CpuQuant::Mq4G256 => use_avx2(avx2_available(), requested),
        CpuQuant::Mq3G256V2 => use_avx2(avx2_f16c_available(), requested),
        _ => false,
    }
}

/// Pure dispatch predicate. `requested` forces the decision (`Some(true)` /
/// `Some(false)`, for tests and for the SIMD-vs-scalar comparison), `None`
/// follows `available`.
///
/// A forced `Some(true)` on hardware without AVX2 would execute undefined
/// instructions, so the force is honoured only when `available` says the feature
/// is present: a forced request on unsupported hardware falls back to scalar
/// rather than faulting, which is also what lets the predicate be unit-tested on
/// any runner.
pub fn use_avx2(available: bool, requested: Option<bool>) -> bool {
    match requested {
        Some(true) => available,
        Some(false) => false,
        None => available,
    }
}

/// One `Mq4G256` weight row (`k / 256` groups of 136 B) dotted with a
/// pre-rotated activation, on the AVX2 path.
///
/// The caller has already decided to use it ([`row_dot_enabled`]), so this is
/// the only place the `unsafe` lives for the format: `row.len() >= (k / 256) * 136`,
/// `x.len() >= k`, and AVX2 + FMA are present.
pub(crate) fn mq4g256_row_dot_avx2(row: &[u8], k: usize, x: &[f32]) -> f32 {
    debug_assert!(avx2_available(), "AVX2 row dot requires the feature");
    debug_assert!(row.len() >= (k / 256) * 136 && x.len() >= k);
    unsafe { x86::mq4g256_row_dot(row.as_ptr(), k, x.as_ptr()) }
}

/// One `Mq3G256V2` weight row (`k / 256` groups of 104 B) dotted with a
/// pre-rotated activation, on the AVX2 path.
///
/// Same contract as [`mq4g256_row_dot_avx2`], plus F16C for the fp16 header:
/// `row.len() >= (k / 256) * 104`, `x.len() >= k`.
pub(crate) fn mq3g256v2_row_dot_avx2(row: &[u8], k: usize, x: &[f32]) -> f32 {
    debug_assert!(
        avx2_f16c_available(),
        "AVX2+F16C row dot requires the features"
    );
    debug_assert!(row.len() >= (k / 256) * 104 && x.len() >= k);
    unsafe { x86::mq3g256v2_row_dot(row.as_ptr(), k, x.as_ptr()) }
}

#[cfg(test)]
mod tests;
