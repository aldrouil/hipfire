# Passback sampled GPU-event timing (candidate 5 / gfx1201) — 9B primary **+1.9 %** initial / **+1.4 %** reverse-order, river **+0.95 %** — 2026-10-16

**Lifecycle:** `historical`. **Disposition: fixture-bound candidate evidence — NOT a promotion, NOT a current baseline, NOT an admission.** This record measures one candidate worktree delta (`C5`, "sampled GPU-event timing") against the immediately preceding candidate (`C4`, "asynchronous split returns") on **one fixture family, one model (9B), one quant (MQ4), one GPU (gfx1201), greedy decoding**, fresh-process, unprofiled, `passback` route with `share auto`, budget 16. The numbers here do **not** transfer across host, model, quant, GPU, prompt, route, share, budget, or method without a new record. The effect is **small** (a low-single-digit-percent decode-rate edge over C4, itself +6.3 % over the original build; see [`2026-10-16-passback-asynchronous-split-returns-gfx1201.md`](2026-10-16-passback-asynchronous-split-returns-gfx1201.md)). **27B was not validated here** — the scope of every number below is **9B only**, and no promotion decision is made.

This is a **new** checkpoint; no existing checkpoint or amendment is modified (append-only: [`README.md`](README.md) §Preservation, root perf-ledger rule). It follows the C4 record's style. Related candidate ledgers: [`C1`](2026-10-16-passback-timing-interval-experiment.md), [`C2`](2026-10-16-passback-shape-isolation-experiment.md) + [`C2 amendment 1`](2026-10-16-passback-shape-isolation-experiment-amendment-1.md), [`C3`](2026-10-16-passback-pinned-copy-worker-experiments.md), [`C4`](2026-10-16-passback-asynchronous-split-returns-gfx1201.md). Raw receipts: [`data-2026-10-16-passback-sampled-gpu-event-timing/`](data-2026-10-16-passback-sampled-gpu-event-timing/) (**80 files**: 79 receipts + `SHA256SUMS`).

## Headline (9B primary, unprofiled, N=4 fresh processes per arm)

The comparison is **C5 vs C4** (not vs the original build). Each arm's sample is the median of its three warm rows (rows 2–4) inside one fresh serve process; the per-arm figure is the median of the four process medians.

| arm | process medians (tok/s) | **arm median** | vs C4 |
|---|---|---:|---:|
| C4 (candidate 4) | 20.90, 21.00, 20.70, 20.20 | **20.80** | — |
| C5 (candidate 5) | 21.40, 21.10, 21.30, 21.10 | **21.20** | **+1.9 % median** (+2.5 % mean) |

Mean ± sample sd / CV (N=4 process medians): C4 **20.70 ± 0.36, CV 1.7 %** (range 20.2–21.0); C5 **21.23 ± 0.15, CV 0.7 %** (range 21.1–21.4). **Separation:** all four C5 process medians (≥ 21.1) exceed all four C4 process medians (≤ 21.0) — no overlap at the process level in this block. Sample arm medians `+1.9 %` = 21.20 / 20.80 − 1.

**C4-session caution (must be read with the number).** The initial block's C4 arm **drifted down across its own run** (20.90 → 21.00 → 20.70 → **20.20**), its last process ~0.7 tok/s below its first, while C5 held flat (21.4/21.1/21.3/21.1). Part of the **+1.9 %** may therefore be C4 late-session sag rather than a pure C5 gain; the reverse-order replication below runs C5 first and is the more conservative primary estimate.

## Replication — 9B primary, reverse order `BAABBAAB`

Same frozen binaries, same fixture, same protocol; run order reversed so the arms swap time slots (receipt-verified by JSON completion mtimes, §Method). N=4/arm.

| arm | process medians (tok/s) | **arm median** | vs C4 |
|---|---|---:|---:|
| C4 | 20.60, 21.00, 20.90, 21.20 | **20.95** | — |
| C5 | 21.20, 21.20, 21.40, 21.30 | **21.25** | **+1.4 % median** (+1.7 % mean) |

C4 mean ± sd / CV: **20.93 ± 0.25, CV 1.2 %**; C5 **21.27 ± 0.10, CV 0.45 %**. Process-median ranges **touch at one point** (C4 max 21.2 = C5 min 21.2), so this block does **not** claim process-level separation — but 7 of 8 C5 processes are ≥ all but one C4 process, and the direction survives the order swap.

**Primary replication verdict:** C5's direction replicated; the two primary blocks give **+1.9 %** and **+1.4 %** median (**+2.5 %** / **+1.7 %** mean). **The conservative primary estimate is the smaller replicated one, +1.4 %** (reverse order, where the C5 arm did not enjoy a C4 sag).

## Secondary prompt (river), N=3 processes per arm

Fresh fixture `warm3x_prose_river_short.json` = 4 exact copies of `benchmarks/prompts/prose_river_short.txt` bytes (260 B, no normalization); wrapper `C5REP-ab_sample_river.sh` = the primary wrapper with the fixture swapped. Interleave A B B A A B, receipt-verified.

| arm | process medians (tok/s) | **arm median** | vs C4 |
|---|---|---:|---:|
| C4 | 20.80, 21.10, 21.20 | **21.10** | — |
| C5 | 21.40, 21.30, 21.20 | **21.30** | **+0.95 % median** (+1.3 % mean) |

C4 mean ± sd: **21.03 ± 0.21**; C5 **21.30 ± 0.10**. Same direction as the primary, **no genre degradation**; ranges touch at 21.2. Smaller than the primary, and N=3/arm — recorded as **direction only**.

## Pooled primary (N=8 processes per arm) — two independent balanced blocks, not eight free samples

Pooling the initial (`ABBAABBA`) and reverse (`BAABBAAB`) primary blocks: C4 median **20.90** (mean 20.81 ± 0.31), C5 median **21.25** (mean 21.25 ± 0.12) → **+1.7 % median, +2.1 % mean**. This is **two balanced ABBA blocks of 4**, not 8 independent samples: the pool's purpose is to show the direction is not one block's accident, and it must not be read as 8×4 = 24 independent rows or as a tighter confidence claim than the two block medians support. Pooled ranges touch (C4 max 21.2 overlaps C5 min 21.1).

## What was tested (C5 vs C4)

C5 is an **uncommitted worktree delta on `12daf4fe99017a88242469f60853b860609c20ab`** (`perf/passback-opt`), built **on top of C4**. Its `CHANGELOG`/source states the mechanism:

1. **The share controller measures both arms' own concurrent durations instead of inferring the GPU arm from the join.** A *sampled* split step hands `observe` the CPU arm's multiply time (`gemv_ns` — the interval that actually overlaps the GPU arm, since the arm's blocking D2H precedes the launch and its H2D join follows it) and the GPU arm's own elapsed time, bracketed by a reusable pair of stream-ordered markers (`Gpu::span_begin` / `Gpu::span_end`, read with `Gpu::span_elapsed_ns`) on the stream the arm and its H2D both run on. The elapsed time is read only after the sampled step's **completed H2D**, so the measurement adds no event synchronization and no fence.
2. **The old join-floor inference and its censoring are removed.** Previously the GPU rate was recovered from the completed H2D join relative to the shape's own no-wait floor and was observable only on a *waited* join: `min_join_ns` never rose, a shape whose joins sat at the floor carried a dead GPU anchor, and a `share += 0.02` upward ratchet had no downward counterpart. The balance point `r_gpu / (r_gpu + r_cpu)` now comes from both real durations — no floor, no wait gate, no ratchet.
3. **The split trace line re-forms.** `cpu=` is now the multiply interval alone; the measured GPU rate prints as `gpu=` (a `—` placeholder until a sample exists) with the sampled-step count; `gpu_samples`/`min_join_ns`/`last_waited` are gone. The pinned async return, the ~one-in-four aperiodic per-shape LCG sampling, the shape keys, eligibility gates, row partition, probe, EWMAs, share clamp, latch and reopen bands are otherwise C4's, unchanged.

C5 is identified by its frozen binary md5s and by [`source.patch`](data-2026-10-16-passback-sampled-gpu-event-timing/) (120,834 B, sha256 `5dbcf649…66ab2`) and by [`delta-from-c4.patch`](data-2026-10-16-passback-sampled-gpu-event-timing/) (58,379 B, sha256 `2e8b81fe…1fb2c`), which touch `CHANGELOG.md`, `crates/hipfire-dispatch/src/{cpu_exec.rs,offload_split.rs}` and `crates/rdna-compute/src/{dispatch.rs,lib.rs,span_timer.rs,map.md}`. The C4 delta (pinned async returns) is the **functional prerequisite** of C5.

## Method

- **9B / MQ4 / budget 16 / `HIPFIRE_OFFLOAD_EXEC=passback` / `HIPFIRE_OFFLOAD_PASSBACK_SHARE=auto`**, greedy (temp 0.0), thinking off, `max_tokens 128`, `max_seq 4096`, spec off (AR), `kv auto` → K=q8 V=q8 VMM, `HIPFIRE_DPM_WARMUP_SECS=10`. Port 11541, serial hardware.
- **Four requests per fresh serve process**: row 1 is the **throwaway**; rows 2–4 are measured, and **their median is ONE process sample**. The 3 warm rows inside a process are **not independent samples** and are never pooled; per-arm N is the number of **fresh processes**.
- **Primary N=4 processes per arm** in each block: initial `A1..A4` (C4), `B1..B4` (C5), order `ABBAABBA`; replication `RA1..RA4` (C4), `RB1..RB4` (C5), order `BAABBAAB`. Secondary river N=3/arm, `VA1..VA3` (C4) / `VB1..VB3` (C5), interleave `A B B A A B`.
- **Run order is receipt-verified** from JSON completion mtimes (UTC−04:00). Initial: A1 17:30:33 → B1 17:31:40 → B2 17:32:44 → A2 17:33:48 → A3 17:34:55 → B3 17:35:59 → B4 17:37:03 → A4 17:38:09. Trace 17:39:14. Replication: RB1 17:43:01 → RA1 17:44:06 → RA2 17:45:11 → RB2 17:46:15 → RB3 17:47:19 → RA3 17:48:23 → RA4 17:49:27 → RB4 17:50:31. River: VA1 17:51:36 → VB1 17:52:40 → VB2 17:53:44 → VA2 17:54:48 → VA3 17:55:52 → VB3 17:57:09.
- **Trace OFF for every headline row** — the wrappers `env -u` TRACE/PROFILE/PROFILE_DECODE/MEMSET_DUMP and `env -u RAYON_NUM_THREADS`; all 22 headline meta logs record `unset check (must be empty): []`, `RAYON check (must be unset): [unset]`, and `trace-line check (expect 0)` prints `1` = the load-time banner `cpu exec: 16/16 spilled layers fully covered by the CPU seam…`, never a per-step line (0 `split:` lines in every headline serve log). The separately archived `C5CMP-T1-c5trace` run is **profiled by design** (`HIPFIRE_CPU_EXEC_TRACE=1`) and is **not** a headline sample (§Attribution).
- **Cold row 1 discarded everywhere — no cold-start claim.** Initial cold rows: 9.1 tok/s decode, ~27.0–27.3 s wall, ~21.0–21.2 s TTFT.

## Results — 9B primary initial, `ABBAABBA` (per process; row 1 excluded)

| sample | arm | warm rows 2–4 (tok/s) | **process median** | TTFT med | wall med |
|---|---|---|---:|---:|---:|
| A1 | C4 | 20.9, 20.8, 21.0 | **20.90** | 0.633 | 6.704 |
| A2 | C4 | 21.1, 21.0, 20.9 | **21.00** | 0.629 | 6.686 |
| A3 | C4 | 20.7, 20.6, 20.7 | **20.70** | 0.635 | 6.792 |
| A4 | C4 | 20.5, 20.1, 20.2 | **20.20** | 0.635 | 6.940 |
| B1 | C5 | 21.2, 21.4, 21.5 | **21.40** | 0.628 | 6.573 |
| B2 | C5 | 21.1, 21.0, 21.2 | **21.10** | 0.628 | 6.657 |
| B3 | C5 | 21.3, 21.4, 21.2 | **21.30** | 0.629 | 6.609 |
| B4 | C5 | 21.1, 21.2, 21.1 | **21.10** | 0.631 | 6.643 |

C4 arm 20.80, C5 arm 21.20 → **+1.9 % median** (+2.5 % mean). Process ranges do not overlap (C4 20.2–21.0, C5 21.1–21.4). Cold rows: C4 9.1 / C5 9.1.

## Results — 9B primary replication, `BAABBAAB` (per process; row 1 excluded)

| sample | arm | warm rows 2–4 (tok/s) | **process median** | TTFT med | wall med |
|---|---|---|---:|---:|---:|
| RB1 | C5 | 21.3, 21.2, 21.2 | **21.20** | 0.624 | 6.620 |
| RA1 | C4 | 20.0, 20.6, 20.9 | **20.60** | 0.630 | 6.808 |
| RA2 | C4 | 21.0, 20.9, 21.1 | **21.00** | 0.631 | 6.678 |
| RB2 | C5 | 21.4, 21.2, 21.2 | **21.20** | 0.633 | 6.625 |
| RB3 | C5 | 21.1, 21.4, 21.4 | **21.40** | 0.633 | 6.579 |
| RA3 | C4 | 20.9, 20.9, 20.7 | **20.90** | 0.621 | 6.714 |
| RA4 | C4 | 21.3, 21.1, 21.2 | **21.20** | 0.631 | 6.635 |
| RB4 | C5 | 21.3, 21.4, 21.0 | **21.30** | 0.631 | 6.611 |

C4 arm 20.95, C5 arm 21.25 → **+1.4 % median** (+1.7 % mean). Ranges touch at 21.2 (C4 max = C5 min). Cold rows: C4 8.9 / C5 9.1.

## Results — secondary river, `A B B A A B` (per process; row 1 excluded)

| sample | arm | warm rows 2–4 (tok/s) | **process median** | TTFT med | wall med |
|---|---|---|---:|---:|---:|
| VA1 | C4 | 20.8, 20.8, 20.7 | **20.80** | 0.251 | 6.370 |
| VA2 | C4 | 20.7, 21.1, 21.1 | **21.10** | 0.253 | 6.283 |
| VA3 | C4 | 21.2, 21.3, 21.2 | **21.20** | 0.250 | 6.234 |
| VB1 | C5 | 21.5, 21.3, 21.4 | **21.40** | 0.245 | 6.190 |
| VB2 | C5 | 21.3, 21.4, 21.3 | **21.30** | 0.251 | 6.217 |
| VB3 | C5 | 21.2, 21.3, 21.2 | **21.20** | 0.253 | 6.244 |

C4 arm 21.10, C5 arm 21.30 → **+0.95 % median** (+1.3 % mean). Direction consistent, magnitude smaller; ranges touch at 21.2.

## Attribution (profiled, N=1 — mechanism, not headline)

`C5CMP-T1-c5trace` is a **separate, profiled** C5 serve process run after the initial A/B block (receipt 17:39:14), **not** a member of the N=4 arms; it carries its own meta log but the trace wrapper prints no `unset check` line. Its row 1 is cold (9.1 tok/s); warm rows 21.0/21.1/21.2, median 21.10 — quoted as traced-process context, not a sample. Its serve log holds **83 `split: step` lines** (the per-shape cumulative states) plus 30 `cpu exec:` window/idle lines.

**Final steady states, last `split:` per (m, k, rotated, residual) — 6 shapes, all `frozen`:**

| m | k | rot/res | share | `gpu=` (measured) | sampled | target | applied/reopens | d2h/gemv/return (ms) | cpu |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| 1024 | 4096 | f/f | 0.402 | 25.4 GB/s | 10524 | 0.402 | 244/13 | 0.05/0.08/0.01 | 16.0 |
| 8192 | 4096 | f/f | 0.402 | 25.4 GB/s | 10524 | 0.402 | 244/13 | 0.04/0.28/0.02 | 36.1 |
| 12288 | 4096 | f/f | 0.402 | 23.7 GB/s | 10526 | 0.402 | 244/13 | 0.07/0.40/0.03 | 38.6 |
| 4096 | 4096 | f/f | 0.411 | 22.8 GB/s | 7048 | 0.411 | 233/11 | 0.06/0.19/0.01 | 27.3 |
| 4096 | 4096 | t/t | 0.402 | 25.4 GB/s | 10524 | 0.402 | 244/13 | 0.07/0.20/0.01 | 25.8 |
| 4096 | 12288 | t/t | **0.375** | **23.8 GB/s** | 2051 | 0.375 | 64/0 | 0.11/0.43/0.02 | 39.1 |

The traced measured GPU rate is ~23–25 GB/s across shapes; `m=4096 k=12288` settles at share **0.375** with a measured `gpu=23.8 GB/s` over 2051 samples, frozen (above the old C4 join-floor-inferred regime). **The trace is one observational run: it does not prove 0.375 is optimal** (no counterfactual shares were measured) **and it does not isolate a single cause** for the end-to-end edge (see §Mechanism).

**Mechanism statement (as required).** C5 combines **a real GPU-arm duration (start/stop stream markers)**, **a CPU interval that is the multiply alone (D2H excluded)**, and **the removal of the join-floor censoring and the `+0.02` push-up ratchet**. The small end-to-end edge **cannot be attributed to any one of these alone**: all three move the same controller and the same sampling path, and this single observational trace cannot separate them. No claim is made that the observed shares are optimal, nor that the gain comes solely from "more correct GPU timing".

## Output health

All **32** primary rows (8 processes × 4) and all **24** replication rows (8 × 4): `finish="length"`, `runaway=true`, `attractor=false`, `empty=false`, `gen=128`, `ctx=59`, `cached=0`, spec fields (`tau`/`dflash`/`mtp`/`ngram_*`) **null**, `decode_estimated=false` — the 128-token cap truncating a valid Python answer. All 56 primary rows **byte-identical**: `sha256 7690b0d32d444edec66c90d6326ad69b9249ca5affdabda50ef31e6306bfcacc` (520 B), `request_md5 dd93dcc0528f194dbd44994d188b03d8`, `prompt_md5 5835c71e471849b4a72e1dc8e39695e7` — the same text as the C1–C4 rows.

Secondary river: all **18** rows identical `sha256 e9764e2f8fa77e4f22f3e6ff91df3fde6cbf87d37a9e102569b7cf070aa3e9ab` (591 B), same flags, `ctx=65`, `request_md5 20833ab1d0d9766a969f24398996741d`, `prompt_md5 07a7880965142971dbb3cc7493f8fb94`.

**Scope limit (fixture-specific, not a golden policy).** Greedy output was **byte-identical within each genre across arms** here, despite `passback`'s documented run-to-run nondeterminism (its row split is scheduled per process). That is an **observed property of these fixtures and this greedy prompt only** — the argmax happened to be insensitive to the engine mix. It is **not** a claim that `passback` output is reproducible, and `passback share auto` is **not** a golden reference; any gate that diffs pass-back output against a reference must pin the share. No token-id-level identity is asserted (the harness records decoded text + `gen` + `request_md5`, not raw greedy ids).

## Fixture identity (measured in this host, verified against the receipts)

| artifact | digest |
|---|---|
| branch / base source | `perf/passback-opt` off **HEAD `12daf4fe99017a88242469f60853b860609c20ab`** (hipfire 0.4.3), base `upstream/beta`; `cargo build --release` (**not** `-p`, feature unification differs) |
| C4 binaries (prerequisite) | `hipfire` md5 `fb32e2e2eab279381a36f59672f1acbc` (47,337,536 B); `daemon` md5 `d400ea368e1258d14c29733293c6c153` (88,582,384 B) — `candidate4-bin/` |
| C5 binaries | `hipfire` md5 `23181d5ba45c532ec6910b0801ea1492` (47,340,216 B); `daemon` md5 `98265d8b809d3b40245f05382b3d9d44` (88,583,272 B) — `candidate5-bin/` |
| C4 source delta | `source.patch` 75,671 B, sha256 `95c0e7c03d5b5c0d50c1228244cadde9e699ceee22ff9f8022e4bb13a38b4551` |
| C5 source delta | `source.patch` 120,834 B, **sha256 `5dbcf649ef225ec4ba5adf3aac72405f92bccd2cd90e23c4e29776fd77066ab2`**; `delta-from-c4.patch` 58,379 B, sha256 `2e8b81fea2afef3f1a270eeafee02d27b203f38aec8701ab42c0337903a1fb2c` — archived with the record |
| toolchain | `rustc 1.98.1 (48a229cea 2026-09-01)`, LLVM 22.1.8, host `x86_64-unknown-linux-gnu` ([`source-provenance.json`](data-2026-10-16-passback-sampled-gpu-event-timing/)) |
| model | `~/.hipfire/models/qwen3.5-9b.mq4`, md5 `296092bf1e6a45d78c1acf815eb93366`, 5,313,750,016 B, tag `qwen3.5:9b` |
| primary fixture | `fixtures/warm3x_gpu_offload_probe.json`, md5 `c4c78f65b100f645cd1ab3ff0fcbf413`; inner prompt `benchmarks/prompts/gpu_offload_probe.txt` md5 `5835c71e471849b4a72e1dc8e39695e7` (215 B), `request_md5 dd93dcc0528f194dbd44994d188b03d8` |
| secondary fixture | `fixtures/warm3x_prose_river_short.json` md5 `da9ca66228d1c686dd98e582f727c374`; inner prompt `benchmarks/prompts/prose_river_short.txt` md5 `07a7880965142971dbb3cc7493f8fb94` (260 B), `request_md5 20833ab1d0d9766a969f24398996741d` |
| wrappers | `C5CMP-ab_sample.sh` md5 `92079a11e07821497b5a1cdb445558d3`; `C5CMP-ab_sample_trace.sh` md5 `ff000460ad0519b3a6ac0b2439bcbfe7`; `C5REP-ab_sample.sh` md5 `e3e4c48acc71579a1233bd281e31a76a`; `C5REP-ab_sample_river.sh` md5 `d68542ceab33b66eb0e1628e091cb0e5`; `power_snap.sh` md5 `6726f0f70161abb83bc6a3bfdbfe5a03` |
| route envs | `HIPFIRE_DPM_WARMUP_SECS=10 HIPFIRE_OFFLOAD_EXEC=passback HIPFIRE_GPU_LAYER_BUDGET=16 HIPFIRE_OFFLOAD_PASSBACK_SHARE=auto` |

Host: 1× Radeon RX 9070 XT `gfx1201` (16,304 MiB), ROCm/HIP 7.2, Ryzen 7 7800X3D (8c/16t), 28 GB RAM. Per-run load banner: 16 of 32 layers host-placed, `passback: 16/16 spilled layers splittable; share auto; min step 2 MiB`, `cpu exec: 16/16 spilled layers fully covered by the CPU seam`, `hipGraph capture disabled (host-executed steps present)`; `KV: mode=auto → K=q8 V=q8`; cache warming 10.00 s.

**`RAYON_NUM_THREADS`:** the actual wrappers apply `env -u RAYON_NUM_THREADS` to the serve process and print `RAYON check (must be unset): [unset]`, which reads the real variable **`RAYON_NUM_THREADS`** — receipt-verified `[unset]` in **all 23** meta logs. (A follow-up prose note that wrote "RAYON_THREADS" is a typo and is **not** the authority; the script and its receipt are.)

## Power / clock / temperature (µW snapshots only — NOT steady-state)

sysfs (card1) `power1_average_uW` pre → post via `power_snap.sh`, every meta log. **`rocm-smi --showpower` returned empty in the meta logs** (`=== Power Consumption ===` has no lines). Every run idles at `perf_level=auto`, `power1_cap_uW = 374,000,000` (374 W), `throttle_log` = the "thermal throttling logging enabled" banner only, edge 44–53 °C, junction 45–56 °C.

| run | power pre → post (µW) | run | power pre → post (µW) |
|---|---|---|---|
| C5CMP-A1 | 46 → 130 | C5CMP-B1 | 47 → 130 |
| C5CMP-A2 | 47 → 122 | C5CMP-B2 | 47 → 136 |
| C5CMP-A3 | 47 → 124 | C5CMP-B3 | 47 → 121 |
| C5CMP-A4 | 47 → 105 | C5CMP-B4 | 47 → 141 |
| C5CMP-T1 (trace) | 47 → 74 | | |
| C5REP-RA1 | 47 → 129 | C5REP-RB1 | 47 → 99 |
| C5REP-RA2 | 47 → 108 | C5REP-RB2 | 47 → 96 |
| C5REP-RA3 | 47 → 128 | C5REP-RB3 | 48 → 124 |
| C5REP-RA4 | 47 → 131 | C5REP-RB4 | 49 → 141 |
| C5REP-VA1 | 48 → 125 | C5REP-VB1 | 48 → 131 |
| C5REP-VA2 | 46 → 133 | C5REP-VB2 | 47 → 106 |
| C5REP-VA3 | 48 → 66 | C5REP-VB3 | 46 → 126 |

These are **point snapshots at differing DPM states**, **not comparable across arms**; **no power or energy claim is drawn**. `sclk_Hz` pre/post swings bear no relation to arm.

## C4 / C3 link and other candidate ledgers

- **C4** ([`2026-10-16-passback-asynchronous-split-returns-gfx1201.md`](2026-10-16-passback-asynchronous-split-returns-gfx1201.md)) is the **baseline here**: pinned async split returns, +6.3 % over the original build (19.75 → 21.0), with a provisional river secondary and CPU/PCIe controls. C5's async-return mechanics are C4's; C5 changes only the controller's measurement and the old join-floor censoring.
- **C3** was `rejected` as a standalone optimization (−1.5 %); its pinned storage carries forward as C4's functional prerequisite.
- **C1** (interval) and **C2** (shape-key) were fully reverted.

## Status

The C5 worktree delta **(with its binaries and source patch) is frozen as of this record**; the C5 source patch and its `delta-from-c4.patch` are the frozen artifacts this evidence binds to. Correctness/validation (dispatch library suite, `passback_row_offset_equivalence`, `cpu_offload_return_stream_and_reuse_parity`, span-timer/`offload_split` lifetimes) is **not** claimed here — it was reported through the build provenance ([`source-provenance.json`](data-2026-10-16-passback-sampled-gpu-event-timing/): 410 dispatch tests passed, 1 ignored; both hardware oracles passed), not re-measured in this record. **27B: not validated** — no 27B sample exists here, and this record makes **no promotion or current-baseline claim**.

## What was NOT measured / provenance gaps (recorded honestly)

- **The effect is small.** +1.9 % (initial) and +1.4 % (replication) median over C4 are low-single-digit-percent and at the edge of the C4 arm's own spread (CV 1.2–1.7 %); only the C5 arm is materially tighter (CV 0.45–0.7 %). Read the **replicated** +1.4 % as the conservative primary estimate, not the initial +1.9 %.
- **Initial-block C4 sag** — C4's last initial process (20.2) is ~0.7 tok/s below its first; the initial +1.9 % may partly reflect it. The reverse-order block (C5 first) gives the smaller +1.4 %.
- **No strict paired four-arm round** — the primary, replication, river and trace windows are sequential, not interleaved against a single reference; drift is acknowledged.
- **River is N=3/arm** — direction only, not a resolved magnitude.
- **CPU/PCIE/share0 controls were NOT re-run for C5** — the C4 record carries them; C5 changes only the scheduler's measurement, which does not touch those routes, but no C5 control receipt exists here.
- **Raw greedy token IDs** — never available in the harness output; text identity is asserted at **decoded text + `gen` + `request_md5`** level only, and is fixture-specific.
- **GPU-side optimum share** — the trace measures ~23–25 GB/s and one frozen 0.375 shape, but no counterfactual shares; optimality is **not** shown.
- **Single-cause attribution** — the trace cannot separate the GPU-event-duration, CPU-interval (gemv-only), and removed-censoring changes; mechanism is stated as the combined policy.
- **`C5CMP-T1-c5trace` meta log** records the trace wrapper's env (no `unset check` line); its serve log is archived in full, untrimmed.
- **`rocm-smi --showpower` unavailable**; telemetry is sysfs point snapshots only.
- No share/budget sweep, no `min_step` sweep, no other model/quant/GPU, no `HIPFIRE_PROFILE` headline runs, no **27B**.

## Reproduction

```
bash C5CMP-ab_sample.sh <cli_bin> <daemon_bin> passback auto 16 ~/.hipfire/models/qwen3.5-9b.mq4 qwen3.5:9b 11541 /tmp/ab/home-<X> /tmp/ab/<X>-passback
# replication: C5REP-ab_sample.sh … ; river: C5REP-ab_sample_river.sh … ; trace: C5CMP-ab_sample_trace.sh …
```
A = C4 (`fb32e2e2`/`d400ea36`), B = C5 (`23181d5b`/`98265d8b`). Primary order `ABBAABBA` ×4, replication `BAABBAAB` ×4, river `A B B A A B` ×3. `OUT` already ends in `.json`, so the wrapper writes `${OUT}.json` (hence `*.json.json`) plus `.serve.log` / `.meta.log`.

## Raw evidence

**80 files** in [`data-2026-10-16-passback-sampled-gpu-event-timing/`](data-2026-10-16-passback-sampled-gpu-event-timing/) (79 receipts + `SHA256SUMS`), every receipt `cmp`-verified byte-identical to its source in `.codeinsight+research/passback-opt/`:

- `C5CMP-{A1..A4}-c4.json.{json,serve.log,meta.log}`, `C5CMP-{B1..B4}-c5.json.{json,serve.log,meta.log}` — primary initial `ABBAABBA` (24 receipts).
- `C5CMP-T1-c5trace.json.{json,serve.log,meta.log}` — profiled N=1 mechanism run (full serve log retained verbatim, **not** trimmed).
- `C5REP-{RA1..RA4}-c4.json.{json,serve.log,meta.log}`, `C5REP-{RB1..RB4}-c5.json.{json,serve.log,meta.log}` — primary replication `BAABBAAB` (24 receipts).
- `C5REP-{VA1..VA3}-c4river.json.{json,serve.log,meta.log}`, `C5REP-{VB1..VB3}-c5river.json.{json,serve.log,meta.log}` — river secondary (18 receipts).
- `C5CMP-ab_sample.sh`, `C5CMP-ab_sample_trace.sh`, `C5REP-ab_sample.sh`, `C5REP-ab_sample_river.sh` — exact wrappers (command proof; RAYON/TRACE receipts).
- `power_snap.sh`, `warm3x_gpu_offload_probe.json`, `warm3x_prose_river_short.json` — the power-snapshot helper and the fixtures used.
- `source.patch`, `delta-from-c4.patch`, `source-provenance.json` — the C5 frozen source delta, its C4-relative delta, and the build/toolchain provenance (text only; **no executables are committed**; the frozen binaries stay in `.codeinsight+research/passback-opt/{candidate4-bin,candidate5-bin}/`).
- `SHA256SUMS` — integrity digests of all 79 receipts.
