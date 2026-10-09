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

#[cfg(target_arch = "x86_64")]
use std::cell::RefCell;
#[cfg(target_arch = "x86_64")]
use std::collections::hash_map::Entry;
#[cfg(target_arch = "x86_64")]
use std::collections::HashMap;

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
    // Prepare each unique eligible activation row's Σx once, before the rayon
    // region; the region then consumes immutable views of that scratch (never
    // the `RefCell`/map itself). On other arches the batched path is unchanged.
    #[cfg(target_arch = "x86_64")]
    {
        ACTIVATION_SUMS.with(|cell| {
            // Keep the borrow alive as an owned guard and reborrow it as a
            // plain `&mut ActivationSumCache`: field-splitting (`lookup` vs
            // `sums`) is then visible to the borrow checker, which a `RefMut`
            // target hides behind `DerefMut`.
            let mut cache_guard = cell.borrow_mut();
            let cache = &mut *cache_guard;
            cache.reset();
            for j in 0..jobs {
                let n = src.n(j);
                let q = src.quant(j);
                // Match the batched region, which returns before resolving any
                // feature decision for an empty job: nothing is prepared and no
                // token window exists. The placeholder pushes keep the per-job
                // vectors positionally aligned with the job index (they are
                // never read for a job with no rows).
                if n == 0 {
                    cache.job_simd.push(false);
                    cache.token_base.push(cache.offsets.len());
                    continue;
                }
                // Resolved once per job, shared by the Σ prep and the row loop.
                let use_simd = simd::row_dot_enabled(q, requested);
                cache.job_simd.push(use_simd);
                let base = cache.offsets.len();
                cache.token_base.push(base);
                let kind = if use_simd {
                    simd::activation_sum_kind(q)
                } else {
                    None
                };
                for t in 0..n {
                    let off = match kind {
                        Some(kind) => {
                            let x = src.xs_row(j, t);
                            // Identity is meaningful only within this call — the
                            // table is cleared above — so a recycled activation
                            // buffer can never hit a value prepared for an
                            // earlier call, layer or request. GU's experts alias
                            // the same activation rows and dedup here; DN's
                            // per-expert rows are distinct addresses and do not.
                            let key = (x.as_ptr() as usize, k, kind);
                            match cache.lookup.entry(key) {
                                Entry::Occupied(e) => *e.get(),
                                Entry::Vacant(e) => {
                                    let groups = k / 256;
                                    let start = cache.sums.len();
                                    cache.sums.resize(start + groups, 0.0);
                                    simd::prepare_activation_sums(
                                        kind,
                                        &x[..k],
                                        &mut cache.sums[start..],
                                    );
                                    e.insert(start);
                                    start
                                }
                            }
                        }
                        None => usize::MAX,
                    };
                    cache.offsets.push(off);
                }
            }
            // The outer TLS borrow is held for the whole region; the workers
            // only ever see `&[…]` slices, so nothing shared-mutable is touched.
            run_shared_jobs(
                m,
                k,
                chunks,
                src,
                requested,
                &cache.sums,
                &cache.offsets,
                &cache.token_base,
                &cache.job_simd,
            );
        });
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        run_shared_jobs(m, k, chunks, src, requested, &[], &[], &[], &[]);
    }
}

/// Thread-local, capacity-reusing scratch for the batched GEMV's prepared
/// activation sums (x86_64 only — the vector row kernels are the only
/// consumer).
#[cfg(target_arch = "x86_64")]
#[derive(Default)]
struct ActivationSumCache {
    /// `Σx` for every *unique* eligible activation row seen this call, `k / 256`
    /// f32 each, concatenated in first-seen order.
    sums: Vec<f32>,
    /// Start index into `sums` per (job, token), flattened in job order;
    /// `usize::MAX` marks a token whose job is not prepared (scalar path or a
    /// format without an activation-sum kernel). `token_base[j]` locates job
    /// `j`'s window.
    offsets: Vec<usize>,
    /// First index into `offsets` for each job.
    token_base: Vec<usize>,
    /// The `use_simd` decision for each job, resolved once here so both the Σ
    /// prep and the row loop agree.
    job_simd: Vec<bool>,
    /// `(activation row ptr, k, kind)` → `sums` start, for intra-call
    /// deduplication. Cleared every call: pointer identity is only meaningful
    /// while the caller's buffers are unchanged, i.e. within one call.
    lookup: HashMap<(usize, usize, simd::ActivationSumKind), usize>,
}

#[cfg(target_arch = "x86_64")]
thread_local! {
    static ACTIVATION_SUMS: RefCell<ActivationSumCache> =
        RefCell::new(ActivationSumCache::default());
}

#[cfg(target_arch = "x86_64")]
impl ActivationSumCache {
    /// Drop every logical entry while keeping the backing allocations, so a
    /// warm call adds no Σ data/map allocation.
    fn reset(&mut self) {
        self.sums.clear();
        self.offsets.clear();
        self.token_base.clear();
        self.job_simd.clear();
        self.lookup.clear();
    }
}

/// The batched GEMV's per-job rayon region. On x86_64 `sums`/`offsets`/
/// `token_base`/`job_simd` are the prepared views (see [`ActivationSumCache`]);
/// on other arches they are empty and each job resolves its own `use_simd`,
/// exactly as before.
#[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
fn run_shared_jobs<S: SharedJobSource + Sync>(
    m: usize,
    k: usize,
    chunks: &mut [Option<&mut [f32]>],
    src: &S,
    _requested: Option<bool>,
    sums: &[f32],
    offsets: &[usize],
    token_base: &[usize],
    job_simd: &[bool],
) {
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
        #[cfg(target_arch = "x86_64")]
        let use_simd = job_simd[j];
        #[cfg(not(target_arch = "x86_64"))]
        let use_simd = simd::row_dot_enabled(q, _requested);
        #[cfg(target_arch = "x86_64")]
        let prepared = use_simd && simd::activation_sum_kind(q).is_some();
        o[..m * n]
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(r, chunk)| {
                let row = &src.packed(j)[r * rb..];
                #[cfg(target_arch = "x86_64")]
                if prepared {
                    let base = token_base[j];
                    for (t, v) in chunk.iter_mut().enumerate() {
                        let off = offsets[base + t];
                        debug_assert_ne!(off, usize::MAX, "unprepared token in a prepared job");
                        *v = simd::row_dot_with_activation_sums(
                            q,
                            row,
                            k,
                            &src.xs_row(j, t)[..k],
                            &sums[off..off + k / 256],
                        );
                    }
                    return;
                }
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

    /// The batched path caches each unique activation row's Σx in a
    /// thread-local keyed by `(row pointer, k, reduction style)`, and that
    /// identity is only meaningful within one call. This drives two calls over
    /// one activation buffer whose contents are rewritten in place (same
    /// pointer, same capacity) and checks every element against an independent
    /// `gemv_with_simd` per (job, token).
    ///
    /// The fixture is built so each failure mode shows up: jobs of qt 13
    /// (`Mq4G256`, style `Mq4Quartets`) and qt 15 (`Mq6G256`, style `Mq6Pairs`)
    /// read the *same* activation row pointer (a key that ignored the style
    /// would alias them), a duplicate-format job revisits row 0 out of order
    /// (a wrong job/token offset), and one job has no rows (the empty-job
    /// placeholder must keep the per-job scratch aligned). Only the second call
    /// can catch a Σx that was not recomputed after the in-place value change.
    #[test]
    fn gemv_shared_recomputes_prepared_sums_for_a_reused_activation_buffer() {
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

        let (m, k, tokens) = (5usize, 512usize, 3usize);
        // qt 13 and qt 15 are the two formats with a prepared Σx and they use
        // different reduction styles; both carry the same rotation, so one
        // activation buffer serves every job.
        let quants = [
            CpuQuant::Mq4G256,
            CpuQuant::Mq6G256,
            CpuQuant::Mq4G256,
            CpuQuant::Mq4G256,
        ];
        let packed: Vec<Vec<u8>> = quants.iter().map(|&q| weights(q, m, k)).collect();
        let packs: Vec<&[u8]> = packed.iter().map(|p| p.as_slice()).collect();
        // Token rows of the one shared buffer each (job, token) reads: job 1
        // reuses job 0's row 0 pointer under the other style, job 2 is empty,
        // job 3 (same format as job 0) revisits row 0 after row 2.
        let rows: [&[usize]; 4] = [&[0usize, 1], &[0], &[], &[2, 0]];

        let mut act: Vec<f32> = (0..tokens)
            .flat_map(|t| x_of(k).into_iter().map(move |v| v + t as f32))
            .collect();
        let cap = act.capacity();
        let ptr = act.as_ptr() as usize;

        let total: usize = rows.iter().map(|r| r.len() * m).sum();
        let mut out = vec![0.0f32; total];

        // `requested = None` is resolved identically by both entry points (the
        // shared path once per job, `gemv_with_simd` once per call), so the
        // exact comparison holds with or without the SIMD kernels.
        for pass in 0..2 {
            {
                let xs_refs: Vec<Vec<&[f32]>> = rows
                    .iter()
                    .map(|r| r.iter().map(|&tok| &act[tok * k..tok * k + k]).collect())
                    .collect();
                let xs_nested: Vec<&[&[f32]]> = xs_refs.iter().map(|r| r.as_slice()).collect();
                let src = Batch {
                    quants: &quants,
                    packs: &packs,
                    xs: &xs_nested,
                };
                gemv_shared_sourced(m, k, &mut out, &src, None);
            }
            let mut reference = vec![0.0f32; total];
            let mut base = 0usize;
            for (j, &q) in quants.iter().enumerate() {
                let n = rows[j].len();
                let mut token_major = vec![0.0f32; n * m];
                for t in 0..n {
                    let tok = rows[j][t];
                    gemv_with_simd(
                        q,
                        &packed[j],
                        m,
                        k,
                        &act[tok * k..tok * k + k],
                        &mut token_major[t * m..(t + 1) * m],
                        None,
                    );
                }
                for r in 0..m {
                    for t in 0..n {
                        reference[base + r * n + t] = token_major[t * m + r];
                    }
                }
                base += n * m;
            }
            assert_eq!(
                out, reference,
                "pass {pass}: batched Σx disagrees with independent gemv_with_simd"
            );
            // Rewrite the buffer in place: pointer and capacity must not change,
            // so only a recomputation can make the next pass correct.
            for v in act.iter_mut() {
                *v = *v * 0.5 - 0.017_578_125;
            }
            assert_eq!(act.capacity(), cap, "pass {pass}: buffer reallocated");
            assert_eq!(act.as_ptr() as usize, ptr, "pass {pass}: buffer moved");
        }
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
