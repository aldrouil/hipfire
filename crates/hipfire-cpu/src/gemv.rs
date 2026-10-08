// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Host-side matmul core: GEMV / GEMM over quantized weight bytes that live in
//! system RAM.
//!
//! The activation contract is the GPU launcher's, not a new one: the MQ GEMV
//! kernels dot the *stored codes* against a **forward-rotated** activation
//! (`rotate_x`), because the weights were encoded post-rotation and
//! `dot(rot W, rot x) = dot(W, x)`. Callers therefore pass `x` already rotated
//! for the FWHT-rotated formats (`CpuQuant::is_fwht_g256`) and unrotated
//! otherwise — see [`crate::quant`] and [`crate::quant::decode_group_codes`].
//! Rotating it here instead would be wrong for the `Prerotated` inputs the
//! dispatch seam receives from the GPU, so the rotation belongs to whoever
//! knows which of the two it holds.
//!
//! Accumulation is plain `f32` in a fixed order (group-wise partial sums). It is
//! deliberately *not* the GPU kernel's 4-accumulator interleave: the contract for
//! the offload path is coherence, not bit-identity against a device kernel — see
//! the crate docs.

use rayon::prelude::*;

use crate::quant::{decode_group_codes, CpuQuant};
use crate::simd;

/// Largest group size over [`CpuQuant`] (256 for every G256 format). The decode
/// scratch is stack-local, so no allocation is on the inner loop.
const MAX_GROUP_ELEMS: usize = 256;

/// Bytes per weight row: `(k / group_elems) * group_bytes`.
///
/// G256 formats need `k % 256 == 0`, which every fixture shape satisfies and
/// which [`gemv`]/[`gemm`] assert.
pub fn row_bytes(q: CpuQuant, k: usize) -> usize {
    (k / q.group_elems()) * q.group_bytes()
}

/// `y[0..m] = W[m,k] · x[0..k]`.
///
/// `packed` is the whole weight tensor's bytes, `m * row_bytes(q, k)` of them.
/// `x` carries the format's rotation (see the module docs).
pub fn gemv(q: CpuQuant, packed: &[u8], m: usize, k: usize, x: &[f32], y: &mut [f32]) {
    gemv_with_simd(q, packed, m, k, x, y, None)
}

/// [`gemv`] with the SIMD decision forced (`Some(false)` = scalar, `None` =
/// runtime detection). The vector path is a throughput choice with a tolerance
/// contract, so the scalar path stays reachable and exact-testable, and
/// `simd::tests` compares the two.
pub fn gemv_with_simd(
    q: CpuQuant,
    packed: &[u8],
    m: usize,
    k: usize,
    x: &[f32],
    y: &mut [f32],
    requested: Option<bool>,
) {
    if m == 0 || k == 0 {
        return;
    }
    let rb = row_bytes(q, k);
    assert!(
        k % 256 == 0,
        "gemv({q:?}): k={k} is not a multiple of 256 (shape {m}x{k})"
    );
    assert!(
        x.len() >= k,
        "gemv({q:?}): x has {} elements, need k={k}",
        x.len()
    );
    assert!(
        y.len() >= m,
        "gemv({q:?}): y has {} elements, need m={m}",
        y.len()
    );
    assert!(
        packed.len() >= m * rb,
        "gemv({q:?}): packed has {} bytes, need m={m} rows of {rb}",
        packed.len()
    );
    let x = &x[..k];
    // Resolve the vector/scalar decision once for the whole call rather than
    // re-detecting the CPU feature per row (`m` is thousands on every real
    // shape).
    let use_simd = simd::row_dot_enabled(q, requested);
    y[..m].par_iter_mut().enumerate().for_each(|(row, out)| {
        *out = dot_row_simd(q, &packed[row * rb..], k, x, use_simd);
    });
}

/// Per-row GEMV over `n` activation rows:
/// `out[row*m .. row*m+m] = W · x[row*k .. row*k+k]`.
///
/// Parallel over every (row, output) pair. `n == 1` is [`gemv`]'s shape with the
/// same work split, so a single-token batch does not serialize.
pub fn gemm(q: CpuQuant, packed: &[u8], m: usize, k: usize, x: &[f32], n: usize, out: &mut [f32]) {
    gemm_with_simd(q, packed, m, k, x, n, out, None)
}

/// [`gemm`] with the SIMD decision forced; see [`gemv_with_simd`].
pub fn gemm_with_simd(
    q: CpuQuant,
    packed: &[u8],
    m: usize,
    k: usize,
    x: &[f32],
    n: usize,
    out: &mut [f32],
    requested: Option<bool>,
) {
    if n == 0 || m == 0 || k == 0 {
        return;
    }
    let rb = row_bytes(q, k);
    assert!(
        k % 256 == 0,
        "gemm({q:?}): k={k} is not a multiple of 256 (shape {n}x{m}x{k})"
    );
    assert!(
        x.len() >= n * k,
        "gemm({q:?}): x has {} elements, need n*k={}",
        x.len(),
        n * k
    );
    assert!(
        out.len() >= n * m,
        "gemm({q:?}): out has {} elements, need n*m={}",
        out.len(),
        n * m
    );
    assert!(
        packed.len() >= m * rb,
        "gemm({q:?}): packed has {} bytes, need m={m} rows of {rb}",
        packed.len()
    );
    let x = &x[..n * k];
    let use_simd = simd::row_dot_enabled(q, requested);
    out[..n * m]
        .par_iter_mut()
        .enumerate()
        .for_each(|(flat, o)| {
            let (token, row) = (flat / m, flat % m);
            *o = dot_row_simd(q, &packed[row * rb..], k, &x[token * k..], use_simd);
        });
}

/// Random-access view of a batch of shared-weight jobs sharing one `(m, k)`.
///
/// A layer's routed-expert CPU FFN issues exactly two such batches — every
/// selected expert's gate/up projection, then every expert's down projection —
/// and [`gemv_shared_sourced`] runs one batch's jobs in a single rayon region.
/// It is a trait rather than a materialized `&[Job]` slice because a job's
/// inputs borrow the caller's buffers (`packed` from the weight owner, the
/// activation rows from its scratch) and its output is a disjoint mutable
/// sub-range of one caller buffer: building that as a slice would force a
/// per-call heap allocation, and holding it in a thread-local is not expressible
/// with safe lifetimes. The caller keeps its *index* metadata (job boundaries,
/// slot→expert and slot→activation maps) in reusable storage and answers these
/// reads instead.
///
/// Each method must be pure and cheap: `n`/`quant`/`packed`/`xs_row` are called
/// once per job (plus once per activation row for the up-front validation).
pub trait SharedJobSource {
    /// Number of jobs in the batch.
    fn jobs(&self) -> usize;
    /// Activation rows for job `j` (`m * n(j)` outputs are written for it).
    fn n(&self, j: usize) -> usize;
    /// Weight format of job `j`. Jobs may differ — a graded layer's MQ6 / MQ4 /
    /// MQ3-Lloyd experts carry different tiers.
    fn quant(&self, j: usize) -> CpuQuant;
    /// Job `j`'s `m` weight rows of `k` in `quant(j)`'s packed layout.
    fn packed(&self, j: usize) -> &[u8];
    /// Job `j`'s activation row `t` (at least `k` long, already carrying the
    /// format's rotation — see the module docs).
    fn xs_row(&self, j: usize, t: usize) -> &[f32];
}

/// Upper bound on the jobs [`gemv_shared_sourced`] splits into a stack array of
/// output chunks before falling back to a heap `Vec`. Decode (1 row) and the
/// ≤4-row verify window keep a layer's job count (one per distinct selected
/// expert) far below this; a wider batch keeps working through the fallback.
const MAX_STACK_JOBS: usize = 256;

/// Batched shared-weight projection: job `j` is `m` weight rows of `k` applied
/// to each of its `n(j)` activation rows, writing `m * n(j)` outputs
/// **row-major** (`out[r * n + t]`), the jobs concatenated in order in `out`.
///
/// The row-major layout is the point of the job. A routed expert selected by
/// several tokens is one weight row read from memory and applied to `n`
/// activations, so the row's bytes stay hot across the job's tokens: the row
/// loop is outer, the token loop inner — the same work split this had when the
/// jobs were a materialized slice. (The *decode* is still per output element,
/// exactly as [`gemv`] does it — the win is a single memory read of the weight
/// row, not a shared decode; `use_simd` is resolved once per job, not per
/// element.) Each element is the same [`dot_row_simd`] call [`gemv`] makes, so a
/// job with one activation row is bit-identical to [`gemv`], and a job with
/// several is bit-identical to calling [`gemv`] once per activation row — the
/// per-element accumulation order does not depend on `n`.
///
/// Parallel over jobs, and within a job over its weight rows; the nesting is
/// deliberate, so a layer whose selected experts are all distinct still fills
/// the pool per job, while a single expert shared by every token parallelizes
/// over that expert's rows instead.
pub fn gemv_shared_sourced<S: SharedJobSource + Sync>(
    m: usize,
    k: usize,
    out: &mut [f32],
    src: &S,
    requested: Option<bool>,
) {
    let jobs = src.jobs();
    if m == 0 || k == 0 || jobs == 0 {
        return;
    }
    assert!(
        k % 256 == 0,
        "gemv_shared_sourced: k={k} is not a multiple of 256"
    );
    // Validate once per job, not per output element (the inner loop is m × n).
    let mut total = 0usize;
    for j in 0..jobs {
        let n = src.n(j);
        let q = src.quant(j);
        let rb = row_bytes(q, k);
        let packed = src.packed(j);
        assert!(
            packed.len() >= m * rb,
            "gemv_shared_sourced({q:?}): weight has {} bytes, need {m} rows of {rb}",
            packed.len()
        );
        for t in 0..n {
            let x = src.xs_row(j, t);
            assert!(
                x.len() >= k,
                "gemv_shared_sourced({q:?}): activation has {} elements, need k={k}",
                x.len()
            );
        }
        total += m * n;
    }
    assert!(
        out.len() >= total,
        "gemv_shared_sourced: out has {} elements, need {total}",
        out.len()
    );
    // Split `out` into one row-major chunk per job without a heap allocation:
    // the chunks live in a fixed-capacity stack array, with the `Vec` fallback
    // reached only by a batch wider than MAX_STACK_JOBS.
    let mut stack: [Option<&mut [f32]>; MAX_STACK_JOBS] = [const { None }; MAX_STACK_JOBS];
    let mut heap: Vec<Option<&mut [f32]>>;
    let chunks: &mut [Option<&mut [f32]>] = if jobs <= MAX_STACK_JOBS {
        let mut rest: &mut [f32] = out;
        for j in 0..jobs {
            let (chunk, tail) = rest.split_at_mut(m * src.n(j));
            stack[j] = Some(chunk);
            rest = tail;
        }
        &mut stack[..jobs]
    } else {
        heap = Vec::with_capacity(jobs);
        let mut rest: &mut [f32] = out;
        for j in 0..jobs {
            let (chunk, tail) = rest.split_at_mut(m * src.n(j));
            heap.push(Some(chunk));
            rest = tail;
        }
        &mut heap
    };
    chunks.par_iter_mut().enumerate().for_each(|(j, slot)| {
        let Some(o) = slot.as_deref_mut() else {
            return;
        };
        let n = src.n(j);
        if n == 0 {
            return;
        }
        let q = src.quant(j);
        let rb = row_bytes(q, k);
        let use_simd = simd::row_dot_enabled(q, requested);
        o[..m * n]
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(r, chunk)| {
                let row = &src.packed(j)[r * rb..];
                for (t, v) in chunk.iter_mut().enumerate() {
                    *v = dot_row_simd(q, row, k, &src.xs_row(j, t)[..k], use_simd);
                }
            });
    });
}

/// One output element: `Σ_j W[row][j] * x[j]`, accumulating one group at a time.
///
/// `use_simd` is resolved once per GEMV call by the caller
/// ([`simd::row_dot_enabled`]); every format has a vector kernel, so when it is
/// set the row is the kernel's and the scalar decode below is the ARM and
/// non-AVX2 fallback (and the reference `simd::tests` compares against).
fn dot_row_simd(q: CpuQuant, row: &[u8], k: usize, x: &[f32], use_simd: bool) -> f32 {
    if use_simd {
        return simd::row_dot_avx2(q, row, k, x);
    }
    dot_row_scalar(q, row, k, x)
}

/// The `Mq4G256` row dot on the scalar path — the SIMD fallback and the
/// reference `simd::tests` compares against.
pub(crate) fn mq4g256_row_dot_scalar(row: &[u8], k: usize, x: &[f32]) -> f32 {
    dot_row_scalar(CpuQuant::Mq4G256, row, k, x)
}

fn dot_row_scalar(q: CpuQuant, row: &[u8], k: usize, x: &[f32]) -> f32 {
    let ge = q.group_elems();
    let gb = q.group_bytes();
    let mut scratch = [0.0f32; MAX_GROUP_ELEMS];
    let mut acc = 0.0f32;
    for g in 0..k / ge {
        let codes = &mut scratch[..ge];
        decode_group_codes(q, &row[g * gb..], codes);
        let xg = &x[g * ge..g * ge + ge];
        let mut partial = 0.0f32;
        for i in 0..ge {
            partial += codes[i] * xg[i];
        }
        acc += partial;
    }
    acc
}

#[cfg(test)]
mod test {
    use super::*;

    /// `[m, k]` weight bytes with distinct payloads per (row, group), so a wrong
    /// row stride or a wrong group offset cannot pass by symmetry.
    fn weights(q: CpuQuant, m: usize, k: usize) -> Vec<u8> {
        crate::testfix::weight_bytes(q, m, k)
    }

    /// Independent oracle in `f64`: decode the group with the crate's own
    /// decoder and accumulate in double precision. Exactness is not the claim —
    /// `1e-4` relative bounds the *f32 accumulation* this crate does.
    fn oracle(q: CpuQuant, packed: &[u8], m: usize, k: usize, x: &[f32]) -> Vec<f64> {
        let ge = q.group_elems();
        let gb = q.group_bytes();
        let rb = row_bytes(q, k);
        (0..m)
            .map(|row| {
                let mut acc = 0.0f64;
                for g in 0..k / ge {
                    let mut codes = vec![0.0f32; ge];
                    decode_group_codes(q, &packed[row * rb + g * gb..], &mut codes);
                    for i in 0..ge {
                        acc += codes[i] as f64 * x[g * ge + i] as f64;
                    }
                }
                acc
            })
            .collect()
    }

    fn x_of(k: usize) -> Vec<f32> {
        (0..k)
            .map(|i| {
                let v = (i as u64 * 2654435761) % 4096;
                (v as f32 - 2048.0) * 0.001_953_125
            })
            .collect()
    }

    /// `gemv_shared_sourced` must be bit-identical to calling `gemv` once per
    /// activation row, for every job — including jobs of different formats in
    /// one call (a graded layer's MQ6 / MQ4 / MQ3-Lloyd experts) and several
    /// activation rows against one weight (an expert selected by several
    /// verification rows). It is the routed-expert CPU path's replacement for a
    /// loop of per-expert GEMVs, and a rounding change there would flip greedy
    /// tokens.
    #[test]
    fn gemv_shared_matches_per_row_gemv() {
        /// A batched source over plain slices — the same index/slice shape the
        /// CPU MoE splice builds from its reusable metadata.
        struct Batch<'a> {
            quants: &'a [CpuQuant],
            packs: &'a [&'a [u8]],
            xs: &'a [&'a [&'a [f32]]],
        }
        impl SharedJobSource for Batch<'_> {
            fn jobs(&self) -> usize {
                self.quants.len()
            }
            fn n(&self, j: usize) -> usize {
                self.xs[j].len()
            }
            fn quant(&self, j: usize) -> CpuQuant {
                self.quants[j]
            }
            fn packed(&self, j: usize) -> &[u8] {
                self.packs[j]
            }
            fn xs_row(&self, j: usize, t: usize) -> &[f32] {
                self.xs[j][t]
            }
        }

        let (m, k) = (8usize, 512usize);
        // A graded set: three tiers plus a repeated uniform format, with one,
        // two, three and four activation rows so both the mixed-format and the
        // shared-weight (expert across rows) shapes run in one call.
        let quants = [
            CpuQuant::Mq6G256,
            CpuQuant::Mq4G256,
            CpuQuant::Mq3G256Lloyd,
            CpuQuant::Mq4G256,
        ];
        let ns = [3usize, 1, 4, 2];
        let packed: Vec<Vec<u8>> = quants.iter().map(|&q| weights(q, m, k)).collect();
        let xs: Vec<Vec<Vec<f32>>> = ns
            .iter()
            .enumerate()
            .map(|(j, &n)| {
                (0..n)
                    .map(|t| x_of(k).iter().map(|v| v + (j * 10 + t) as f32).collect())
                    .collect()
            })
            .collect();
        let xs_refs: Vec<Vec<&[f32]>> = xs
            .iter()
            .map(|rows| rows.iter().map(|v| v.as_slice()).collect())
            .collect();
        let packs: Vec<&[u8]> = packed.iter().map(|p| p.as_slice()).collect();
        let xs_nested: Vec<&[&[f32]]> = xs_refs.iter().map(|r| r.as_slice()).collect();
        let total: usize = ns.iter().map(|n| n * m).sum();
        let src = Batch {
            quants: &quants,
            packs: &packs,
            xs: &xs_nested,
        };
        let mut out = vec![0.0f32; total];
        gemv_shared_sourced(m, k, &mut out, &src, None);
        let mut reference = vec![0.0f32; total];
        let mut off = 0usize;
        for (j, &q) in quants.iter().enumerate() {
            let n = ns[j];
            // `gemv` produces one activation row at a time (token-major); the
            // batched output is row-major (`out[r * n + t]`), so transpose.
            let mut token_major = vec![0.0f32; n * m];
            for t in 0..n {
                gemv(
                    q,
                    &packed[j],
                    m,
                    k,
                    &xs[j][t],
                    &mut token_major[t * m..(t + 1) * m],
                );
            }
            for r in 0..m {
                for t in 0..n {
                    reference[off + r * n + t] = token_major[t * m + r];
                }
            }
            off += n * m;
        }
        assert_eq!(out, reference, "gemv_shared_sourced != per-row gemv");
    }

    #[test]
    fn gemv_matches_the_f64_reference() {
        for (m, k) in [(3usize, 512usize), (1, 4096)] {
            for q in [
                CpuQuant::Mq4G256,
                CpuQuant::Mq4G256V2,
                CpuQuant::Mq4CG256,
                CpuQuant::Mq6G256,
                CpuQuant::Mq6G256V2,
                CpuQuant::Mq5G256,
                CpuQuant::Mq5G256V2,
                CpuQuant::Mq3G256,
                CpuQuant::Mq3G256V2,
                CpuQuant::Mq3G256Lloyd,
                CpuQuant::Mq2G256,
                CpuQuant::Mq2G256V2,
                CpuQuant::Mq2G256Lloyd,
                CpuQuant::Mq2G256LloydU,
                CpuQuant::Mq4G256Lloyd,
                CpuQuant::Hfq6G256,
                CpuQuant::Hfq4G256,
                CpuQuant::Hfq4G128,
                CpuQuant::Hfq3G256,
                CpuQuant::Hfq3G128,
                CpuQuant::Hfq2G256,
                CpuQuant::Hfq2G128,
                CpuQuant::Tq2G128,
                CpuQuant::Bq1G128,
                CpuQuant::F16,
                CpuQuant::F32,
                CpuQuant::Bf16,
                CpuQuant::Q8F16,
            ] {
                let packed = weights(q, m, k);
                let x = x_of(k);
                let mut y = vec![0.0f32; m];
                gemv(q, &packed, m, k, &x, &mut y);
                let want = oracle(q, &packed, m, k, &x);
                for (row, (got, want)) in y.iter().zip(&want).enumerate() {
                    let scale = want.abs().max(1e-6);
                    assert!(
                        ((*got as f64) - want).abs() / scale < 1e-4,
                        "{q:?} {m}x{k} row {row}: {got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn gemv_matches_a_direct_scalar_dot_exactly() {
        // The whole-tensor entry point must agree with a hand-rolled loop over
        // the same decode, bit for bit — this is what pins the row stride and the
        // group offsets (a swapped stride would still "match" the f64 oracle to
        // within 1e-4 for the wrong reason only if the data were degenerate).
        let q = CpuQuant::Mq4G256;
        let (m, k) = (5usize, 512usize);
        let packed = weights(q, m, k);
        let x = x_of(k);
        let mut y = vec![0.0f32; m];
        gemv(q, &packed, m, k, &x, &mut y);
        let ge = q.group_elems();
        let gb = q.group_bytes();
        for (row, got) in y.iter().enumerate() {
            let mut acc = 0.0f32;
            for g in 0..k / ge {
                let mut codes = vec![0.0f32; ge];
                decode_group_codes(q, &packed[row * row_bytes(q, k) + g * gb..], &mut codes);
                let mut partial = 0.0f32;
                for i in 0..ge {
                    partial += codes[i] * x[g * ge + i];
                }
                acc += partial;
            }
            assert_eq!(*got, acc, "row {row}");
        }
    }

    #[test]
    fn gemm_equals_independent_gemv_calls() {
        let q = CpuQuant::Mq6G256;
        let (m, k, n) = (4usize, 512usize, 3usize);
        let packed = weights(q, m, k);
        let x: Vec<f32> = (0..n)
            .flat_map(|t| x_of(k).into_iter().map(move |v| v + t as f32))
            .collect();
        let mut batch = vec![0.0f32; n * m];
        gemm(q, &packed, m, k, &x, n, &mut batch);
        for t in 0..n {
            let mut single = vec![0.0f32; m];
            gemv(q, &packed, m, k, &x[t * k..t * k + k], &mut single);
            assert_eq!(&batch[t * m..t * m + m], &single[..], "token {t}");
        }
    }

    #[test]
    fn degenerate_shapes_are_no_ops() {
        let mut y = [7.0f32; 4];
        let x = [1.0f32; 4];
        gemv(CpuQuant::Mq4G256, &[], 0, 4, &x, &mut y);
        gemv(CpuQuant::Mq4G256, &[], 4, 0, &[], &mut y);
        assert!(y.iter().all(|v| *v == 7.0));
        let mut out = [7.0f32; 4];
        gemm(CpuQuant::Mq4G256, &[], 0, 4, &x, 2, &mut out);
        gemm(CpuQuant::Mq4G256, &[], 2, 0, &[], 2, &mut out);
        gemm(CpuQuant::Mq4G256, &[], 2, 4, &x, 0, &mut out);
        assert!(out.iter().all(|v| *v == 7.0));
    }

    #[test]
    fn row_bytes_follows_the_group_layout() {
        // 9B qwen3.5 shape: k = 4096.
        assert_eq!(row_bytes(CpuQuant::Mq4G256, 4096), 16 * 136);
        assert_eq!(row_bytes(CpuQuant::Mq4G256V2, 4096), 16 * 136);
        assert_eq!(row_bytes(CpuQuant::Mq6G256, 4096), 16 * 200);
        assert_eq!(row_bytes(CpuQuant::Mq3G256, 4096), 16 * 104);
        assert_eq!(row_bytes(CpuQuant::F16, 4096), 4096 * 2);
        assert_eq!(row_bytes(CpuQuant::F32, 4096), 4096 * 4);
        assert_eq!(row_bytes(CpuQuant::Q8F16, 4096), 128 * 34);
    }

    #[test]
    #[should_panic(expected = "not a multiple of 256")]
    fn unaligned_k_panics() {
        let mut y = [0.0f32; 1];
        gemv(
            CpuQuant::Mq4G256,
            &[0u8; 200],
            1,
            128,
            &[0.0f32; 128],
            &mut y,
        );
    }

    #[test]
    #[should_panic(expected = "packed has")]
    fn short_weight_buffer_panics() {
        let mut y = [0.0f32; 1];
        gemv(
            CpuQuant::Mq4G256,
            &[0u8; 135],
            1,
            256,
            &[0.0f32; 256],
            &mut y,
        );
    }
}

/// Property tests: the batched / partitioned entry points are bit-identical to
/// the singleton ones. These live in-file because they build weights with
/// `crate::testfix` (`pub(crate)`).
///
/// Why exact equality is the right claim: every output element is produced by
/// the same `dot_row_simd(q, row, k, x_token, use_simd)` call regardless of how
/// the (token, row) pairs are batched or partitioned, so any difference is a
/// stride/offset bug, never rounding.
#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    /// Every `CpuQuant` variant (all are valid for `gemm`: `k % 256 == 0`
    /// satisfies each format's group size).
    const ALL: [CpuQuant; 28] = [
        CpuQuant::Mq4G256,
        CpuQuant::Mq4G256V2,
        CpuQuant::Mq6G256,
        CpuQuant::Mq3G256,
        CpuQuant::Mq3G256Lloyd,
        CpuQuant::Hfq6G256,
        CpuQuant::Hfq3G256,
        CpuQuant::Hfq2G256,
        CpuQuant::Hfq4G128,
        CpuQuant::Hfq3G128,
        CpuQuant::Hfq2G128,
        CpuQuant::Mq5G256,
        CpuQuant::Mq2G256,
        CpuQuant::Mq2G256Lloyd,
        CpuQuant::Mq2G256LloydU,
        CpuQuant::Mq4G256Lloyd,
        CpuQuant::Mq4CG256,
        CpuQuant::Mq6G256V2,
        CpuQuant::Mq5G256V2,
        CpuQuant::Mq3G256V2,
        CpuQuant::Mq2G256V2,
        CpuQuant::Tq2G128,
        CpuQuant::Bq1G128,
        CpuQuant::Hfq4G256,
        CpuQuant::F16,
        CpuQuant::F32,
        CpuQuant::Bf16,
        CpuQuant::Q8F16,
    ];

    /// Compile-time-ish guard: adding a `CpuQuant` variant without extending
    /// `ALL` fails to compile here (exhaustive match, no wildcard).
    #[allow(dead_code)]
    fn all_is_exhaustive(q: CpuQuant) {
        match q {
            CpuQuant::Mq4G256
            | CpuQuant::Mq4G256V2
            | CpuQuant::Mq6G256
            | CpuQuant::Mq3G256
            | CpuQuant::Mq3G256Lloyd
            | CpuQuant::Hfq6G256
            | CpuQuant::Hfq3G256
            | CpuQuant::Hfq2G256
            | CpuQuant::Hfq4G128
            | CpuQuant::Hfq3G128
            | CpuQuant::Hfq2G128
            | CpuQuant::Mq5G256
            | CpuQuant::Mq2G256
            | CpuQuant::Mq2G256Lloyd
            | CpuQuant::Mq2G256LloydU
            | CpuQuant::Mq4G256Lloyd
            | CpuQuant::Mq4CG256
            | CpuQuant::Mq6G256V2
            | CpuQuant::Mq5G256V2
            | CpuQuant::Mq3G256V2
            | CpuQuant::Mq2G256V2
            | CpuQuant::Tq2G128
            | CpuQuant::Bq1G128
            | CpuQuant::Hfq4G256
            | CpuQuant::F16
            | CpuQuant::F32
            | CpuQuant::Bf16
            | CpuQuant::Q8F16 => {}
        }
    }

    /// SplitMix64: deterministic byte/salt stream from the proptest-drawn seed.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// Bytes of the per-group scale/codebook header that must stay valid (finite
    /// decode): everything after it is free-form code payload.
    fn header_len(q: CpuQuant) -> usize {
        match q {
            CpuQuant::Mq4G256Lloyd => 32,
            CpuQuant::Mq3G256Lloyd => 16,
            _ => 8,
        }
    }

    /// `[m, k]` weight bytes: the crate fixture group (valid, finite header) with
    /// a random salt and random code payload. The unquantized formats keep the
    /// fixture's finite small-value tables (random bits there would be NaN/Inf).
    fn random_weights(q: CpuQuant, m: usize, k: usize, seed: u64) -> Vec<u8> {
        let mut rng = Rng(seed);
        let groups = k / q.group_elems();
        let quantized = !matches!(q, CpuQuant::F16 | CpuQuant::F32 | CpuQuant::Bf16);
        let mut out = Vec::with_capacity(m * groups * q.group_bytes());
        for _ in 0..m * groups {
            let mut g = crate::testfix::group_bytes(q, (rng.next() % 4096) as usize);
            if quantized {
                for b in &mut g[header_len(q)..] {
                    *b = rng.next() as u8;
                }
            }
            out.extend_from_slice(&g);
        }
        out
    }

    /// Finite activations in `[-512, 512)` on a 2^-6 grid.
    fn random_x(len: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng(seed ^ 0xA5A5_5A5A_DEAD_BEEF);
        (0..len)
            .map(|_| ((rng.next() % 65536) as i64 - 32768) as f32 / 64.0)
            .collect()
    }

    #[derive(Debug)]
    struct Case {
        q: CpuQuant,
        /// `Some(false)` = forced scalar, `None` = runtime SIMD detection.
        simd: Option<bool>,
        /// Output features (weight rows).
        m: usize,
        k: usize,
        /// Batch (tokens).
        n: usize,
        packed: Vec<u8>,
        x: Vec<f32>,
        /// Raw split-point entropy (reduced mod the relevant range by the property).
        cuts: Vec<usize>,
    }

    fn case() -> impl Strategy<Value = Case> {
        (
            0..ALL.len(),
            any::<bool>(),
            1usize..=24,
            1usize..=4,
            1usize..=9,
            any::<u64>(),
            prop::collection::vec(any::<usize>(), 0..5),
        )
            .prop_map(|(qi, scalar, m, groups, n, seed, cuts)| {
                let q = ALL[qi];
                // gemv/gemm assert `k % 256 == 0`, which is a multiple of every
                // format's group size (32/128/256).
                let k = 256 * groups;
                Case {
                    q,
                    simd: scalar.then_some(false),
                    m,
                    k,
                    n,
                    packed: random_weights(q, m, k, seed),
                    x: random_x(n * k, seed),
                    cuts,
                }
            })
    }

    /// Sorted, deduplicated interior split points in `1..len`, plus both ends.
    fn boundaries(raw: &[usize], len: usize) -> Vec<usize> {
        let mut b: Vec<usize> = raw
            .iter()
            .filter(|_| len > 1)
            .map(|c| 1 + c % (len - 1))
            .collect();
        b.push(0);
        b.push(len);
        b.sort_unstable();
        b.dedup();
        b
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|f| f.to_bits()).collect()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// gemm over `n` tokens == `n` independent gemv calls, bit for bit.
        #[test]
        fn prop_gemm_equals_n_gemv(c in case()) {
            let Case { q, simd, m, k, n, packed, x, .. } = c;
            let mut batch = vec![0.0f32; n * m];
            gemm_with_simd(q, &packed, m, k, &x, n, &mut batch, simd);
            for t in 0..n {
                let mut single = vec![0.0f32; m];
                gemv_with_simd(q, &packed, m, k, &x[t * k..(t + 1) * k], &mut single, simd);
                prop_assert_eq!(
                    bits(&batch[t * m..(t + 1) * m]),
                    bits(&single),
                    "{:?} m={} k={} n={} token {}", q, m, k, n, t
                );
            }
        }

        /// gemm over the whole batch == concatenation of gemm over contiguous
        /// chunks of the batch (random split points), bit for bit.
        #[test]
        fn prop_gemm_batch_partition_invariant(c in case()) {
            let Case { q, simd, m, k, n, packed, x, cuts } = c;
            let mut full = vec![0.0f32; n * m];
            gemm_with_simd(q, &packed, m, k, &x, n, &mut full, simd);
            let b = boundaries(&cuts, n);
            let mut stitched: Vec<f32> = Vec::with_capacity(n * m);
            for w in b.windows(2) {
                let (lo, hi) = (w[0], w[1]);
                let mut part = vec![0.0f32; (hi - lo) * m];
                gemm_with_simd(q, &packed, m, k, &x[lo * k..hi * k], hi - lo, &mut part, simd);
                stitched.extend_from_slice(&part);
            }
            prop_assert_eq!(
                bits(&full),
                bits(&stitched),
                "{:?} m={} k={} n={} partition {:?}", q, m, k, n, b
            );
        }

        /// Output-row split. Weights are row-major with `row_bytes(q, k)` per
        /// output row (the layout `gemv`/`gemm` index by `row * rb`), so a
        /// contiguous row range is a contiguous byte slice. gemm/gemv over that
        /// slice == the matching columns of the full result, bit for bit.
        #[test]
        fn prop_output_row_split_invariant(c in case()) {
            let Case { q, simd, m, k, n, packed, x, cuts } = c;
            let rb = row_bytes(q, k);
            let mut full = vec![0.0f32; n * m];
            gemm_with_simd(q, &packed, m, k, &x, n, &mut full, simd);
            let b = boundaries(&cuts, m);
            for w in b.windows(2) {
                let (lo, hi) = (w[0], w[1]);
                let sub = &packed[lo * rb..hi * rb];
                let mr = hi - lo;
                let mut part = vec![0.0f32; n * mr];
                gemm_with_simd(q, sub, mr, k, &x, n, &mut part, simd);
                for t in 0..n {
                    prop_assert_eq!(
                        bits(&part[t * mr..(t + 1) * mr]),
                        bits(&full[t * m + lo..t * m + hi]),
                        "{:?} m={} k={} n={} rows {}..{} token {} (gemm)", q, m, k, n, lo, hi, t
                    );
                    let mut single = vec![0.0f32; mr];
                    gemv_with_simd(q, sub, mr, k, &x[t * k..(t + 1) * k], &mut single, simd);
                    prop_assert_eq!(
                        bits(&single),
                        bits(&full[t * m + lo..t * m + hi]),
                        "{:?} m={} k={} n={} rows {}..{} token {} (gemv)", q, m, k, n, lo, hi, t
                    );
                }
            }
        }
    }
}
