# Amendment 1 — pass-back cross-model spread (2B/9B/27B) — 2026-10-02

**Lifecycle:** `historical`

**Amends (does not modify):**
[`2026-10-02-offload-passback-model-spread-gfx1201.md`](2026-10-02-offload-passback-model-spread-gfx1201.md),
per [`README.md`](README.md).

## 1. The `SHARE_MAX` explanation is refuted by measurement

The original's Reading point 2 states the 2B's pass-back-vs-`pcie` loss is because
"the 2B's whole-model rates put the balance point at ≈0.59, above the
`SHARE_MAX = 0.50` cap, so the scheduler cannot collapse to the GPU-only route."
That was **inferred from whole-model rates and it is wrong.** The measured
scheduler state (`HIPFIRE_CPU_EXEC_TRACE=1`, 2B, 6 of 24 spilled) is:

```
share=0.268 … share=0.337 … share=0.348 … share=0.350 … share=0.438 … share=0.471
```

Every settled share is **well under 0.50**, so the cap is **not binding** and the
controller is not being pinned. (The same trace shows `reopens` of 1/2/5/6, so the
tuned re-open path is live.)

Corrected: on the 2B, `pcie` (101.0 tok/s) still beats pass-back (79.3) and `cpu`
(69.9), but **the cause is not established** by this sweep. Two open hypotheses,
neither measured: (a) the split's fixed per-step cost (blocking D2H + H2D +
launch) is a large fraction of a small model's short steps — consistent with the
Phase-0 record's `d2h+join ≈ 21 %` of a 9B step, proportionally worse here; (b)
the `[0.05, 0.50]` share range admits no all-GPU point, so pass-back cannot
degenerate to `pcie` even if the balance point were above 0.5. (a) predicts the
size ordering that was seen (longer steps amortise it); (b) is untested. The
inference and the 0.59 figure are withdrawn.

## 2. The 27B's usable spill ceiling on this host is RAM-bound

The original's 27B rows stop at 25 % spilled. Attempts to extend to 37.5 %
(24 of 64) and 50 % were **not runnable on this host**: the run stalled with a
`daemon` at 98 % CPU and **14.4 GB RSS** while the machine sat at **11 GB in
swap** with active page scanning (28 GB RAM, 15.6 GB model) — thrash, not a
pass-back result. So the 27B curve here is **three points**, 6.25 / 12.5 / 25 %
of 64 layers, and the ceiling is the host's RAM, not the mode.

Added point — **two durable blocks** (the original and a re-run; both committed):

| 27B spilled | `cpu` | `passback` | `pcie` | passback vs cpu |
|---:|---:|---:|---:|---:|
| 4 / 64 (6.25 %) | 20.6 `[20.3–21.4]` n=6 | **22.5** `[9.5–23.5]` n=6 | 17.8 `[17.5–18.1]` n=6 | **+9.0 %** |

The wide passback range is the one non-reproducing outlier, kept visible — see § 3.
Merged medians (`data-…/sp_27b_extra.jsonl` + `sp_27b_60.jsonl`).

This extends the 27B trend: the pass-back gain over `cpu` **rises with the spill
fraction** — **+9.0 %** (6.25 %) → +12.7 % (12.5 %) → +17.7 % (25 %).

## 3. The 9.5 outlier is kept visible, not silently dropped

The 27B budget-60 block in the original run produced a 9.5 against 21.9/22.4. A
**re-run block reproduces 22.6/22.6/23.5** (with 20.8/21.3/21.4 `cpu`), so the 9.5
did not reproduce and is read as a transient that coincided with **memory
pressure** (that session was paging: a daemon at 14.4 GB RSS while the host sat at
11 GB swap — § 2). It is **kept in the committed data** (`sp_27b_extra.jsonl`) and
its range is shown in § 2's table rather than hidden; the merged median (n=6,
22.5) absorbs it. Every other 27B row is tight. The 4-round 9B medians in the
original are unaffected.

Raw files: `data-2026-10-02-offload-passback-model-spread/sp_27b_extra.jsonl`
(the block containing the 9.5) and `sp_27b_60.jsonl` (the re-run).

## 4. Baseline-dependent framing (replaces "no size trend")

The original's point 3 said "a bigger model does not show a bigger pass-back gain".
That holds **only against the `cpu` baseline**: pass-back-over-`cpu` is ≈flat
(+13–22 % everywhere). Against the **best single-engine route** it grows sharply
with size, because that route changes identity — `pcie` on the 2B (+44 % over
`cpu`), `cpu` on the 9B and 27B (`pcie` −16 to −28 %). So the correct product
statement is that the *recommended mode is size-dependent* (`pcie` small,
`passback` mid/large), and the user's intuition ("bigger ⇒ bigger effect") is right
once measured against the right baseline.

## 5. Pass-back output is not bit-reproducible (`cpu` and `pcie` are)

The original labelled the 2B/27B splits "smoke-loaded" and made no identity claim;
the attempt to add one produced a **real, larger finding**. Greedy identity check
on the 2B (same prompt, `-t 0 -n 1024`, budget 18), with controls:

| run pair | result |
|---|---|
| `cpu` vs `cpu` | **byte-identical** (2046 B) — deterministic |
| `pcie` vs `pcie` | **byte-identical** (3554 B) — deterministic |
| `cpu` vs `passback` | differs (2046 vs 3464 B) |
| `passback` vs `passback` | **differs** (3670 vs 3078 B) |
| `HIPFIRE_OFFLOAD_PASSBACK_SHARE=0` vs `cpu` | **byte-identical** (2046 B) |

So the seam machinery is sound — `share=0` *is* the `cpu` path exactly — and the
difference is entirely the **scheduled share**: it is derived per process
(`probe_weight` + online arm timings), so two identical invocations divide the
same step differently, mix the two engines' numerics differently, and can flip
greedy tokens. **`cpu` and `pcie` are deterministic modes; `passback` is not.**

This reframes the 2026-10-01 record's § 6 byte-identity claim: it held on that
record's specific 9B prompts, where the argmax happened to be insensitive; it is
not a general property of the mode. Any gate that diffs pass-back output against a
reference must pin `HIPFIRE_OFFLOAD_PASSBACK_SHARE` (or use `0` to reach the `cpu`
twin). Recorded as a property of the mode, not a bug: any share > 0 is a documented
mix of two numerically different engines.

The 27B pair could not be formed (its `think` channel consumed the whole budget,
leaving `content` empty on both arms), so 27B identity stays unverified; mq3
identity is likewise unverified (throughput-only).

## Not changed

Method, fixture, the 2B/9B/27B medians and paired statistics, and the coverage/
share observations from the earlier records.
