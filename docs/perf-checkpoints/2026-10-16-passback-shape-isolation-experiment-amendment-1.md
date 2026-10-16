# Amendment 1 — passback shape-isolation experiment (candidate 2) power unit correction

**Amends:** [`2026-10-16-passback-shape-isolation-experiment.md`](2026-10-16-passback-shape-isolation-experiment.md) (the **unchanged** original; append-only ledger rule — see [`README.md`](README.md) §Preservation).
**Lifecycle:** `historical`. **Disposition:** unchanged — candidate 2 remains **`rejected`**.
**Date:** 2026-10-16.

## Correction

The original record's *Power / clock / memory telemetry* prose paragraph
states the post-run `power1_average` figures in **milliwatts** (mW). That unit
label is **wrong**:

> "The post-run `power1_average` of the candidate samples (70–75 mW) is **not**
> directly comparable to the baseline samples (117–131 mW) …"

The raw field is `power1_average_uW` — **microwatts**. The values captured are
`70,000,000 … 131,000,000 µW`, i.e. **70–131 W**, not 70–131 mW:

- **candidate 2 (B1–B3) post snapshot: `70,000,000 … 75,000,000 µW` = 70–75 W.**
- **baseline (A1–A3) post snapshot: `117,000,000 … 131,000,000 µW` = 117–131 W.**

The **raw values table is unaffected** — it already carries the raw
`power1_average_uW` field verbatim in µW (`74,000,000`, `117,000,000`, etc.),
which is correct. Only the prose unit label "mW" is corrected here; the original
file is left as-is per the append-only rule.

## What does NOT change

- **Throughput / disposition unchanged:** baseline overall process median 19.8
  vs candidate 18.7 tok/s = **−5.6%**, all candidate below all baseline,
  **`rejected`**.
- **No power or perf-causal claim is added.** The correction does not make the
  figures comparable: they remain **point snapshots** taken after each run, with
  the arms landing at different DPM/clock states (`pp_sclk_star` differs run to
  run), so **no power conclusion is drawn** — same as the original. The
  candidate's lower post-run watts (70–75 W vs 117–131 W) is consistent with the
  candidate finishing its (identical) 128-token rows in a shorter wall window at
  the sampled instant, but this is **not** evidence of a power or energy effect,
  and nothing here attributes the decode regression to power.
- `rocm-smi --showpower` remains **empty** (unavailable); sysfs
  `power1_average_uW` remains the only power field.
- No raw receipt, digest, run order, method, or identity entry changes.

## Raw evidence

No new raw evidence is produced by this amendment; it re-reads the existing
`data-2026-10-16-passback-shape-isolation-experiment/C2-*.meta.log` receipts
(unchanged, digest-stable).
