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

/// Whether the CPU reports AVX2 + FMA.
#[cfg(target_arch = "x86_64")]
pub fn avx2_available() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn avx2_available() -> bool {
    false
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
/// The caller has already decided to use it ([`use_avx2`] with
/// [`avx2_available`]), so this is the only place the `unsafe` lives:
/// `row.len() >= (k / 256) * 136`, `x.len() >= k`, and AVX2 + FMA are present.
pub(crate) fn mq4g256_row_dot_avx2(row: &[u8], k: usize, x: &[f32]) -> f32 {
    debug_assert!(avx2_available(), "AVX2 row dot requires the feature");
    debug_assert!(row.len() >= (k / 256) * 136 && x.len() >= k);
    unsafe { x86::mq4g256_row_dot(row.as_ptr(), k, x.as_ptr()) }
}

#[cfg(test)]
mod tests;
