# Passback asynchronous split returns (candidate 4 / gfx1201) — 9B primary **+6.3%**, secondary provisional, controls + share0 identity — 2026-10-16

**Lifecycle:** `historical`. **Disposition: fixture-bound candidate evidence — NOT a promotion, NOT a current baseline, NOT an admission.** This record measures one candidate worktree delta (`C4`, "asynchronous split returns") on **one fixture family, one model (9B), one quant (MQ4), one GPU (gfx1201), greedy decoding**, fresh-process, unprofiled, `passback` route with `share auto`, budget 16. The numbers here do **not** transfer across host, model, quant, GPU, prompt, route, share, budget, or method without a new record. **C5 was not measured and is not part of this record.** **27B was not validated here** — the scope of every number below is **9B only**.

This is a **new** checkpoint; no existing checkpoint or amendment is modified (append-only: [`README.md`](README.md) §Preservation, root perf-ledger rule). It follows the C3 ledger style: [`2026-10-16-passback-pinned-copy-worker-experiments.md`](2026-10-16-passback-pinned-copy-worker-experiments.md). Related rejected/exploratory ledgers: [`C1`](2026-10-16-passback-timing-interval-experiment.md), [`C2`](2026-10-16-passback-shape-isolation-experiment.md) + [`C2 amendment 1`](2026-10-16-passback-shape-isolation-experiment-amendment-1.md), [`C3`](2026-10-16-passback-pinned-copy-worker-experiments.md). Raw receipts: [`data-2026-10-16-passback-asynchronous-split-returns/`](data-2026-10-16-passback-asynchronous-split-returns/) (**72 files**, 71 receipts + `SHA256SUMS`; no executables committed).

## Headline (9B primary, unprofiled, N=4 fresh processes per arm)

| arm | process medians (tok/s) | **arm median** | vs baseline | warm wall med | warm TTFT med | warm prefill med |
|---|---|---:|---:|---:|---:|---:|
| baseline (`12daf4fe`) | 19.7, 19.8, 19.8, 19.7 | **19.75** | — | 7.067 s (7.040–7.088) | 0.637 s (0.634–0.638) | 585.0 ms |
| C4 (candidate 4) | 21.1, 20.9, 21.1, 19.9 | **21.0** | **+6.3%** | 6.685 s (6.647–7.019) | 0.631 s (0.629–0.636) | 584.0 ms |

**Separation:** all four C4 process medians (≥ 19.9) exceed all four baseline process medians (≤ 19.8) — **no overlap at the process level**. At the level of *individual warm rows* the arms do overlap (baseline 19.6–20.3, C4 19.7–21.2), so the claim is the process-level one, which is the counting unit here. `+6.3%` = 21.0 / 19.75 − 1 (−40 ms wall/step, decode-rate only; prefill and TTFT are flat within their bands).

Run order is **ABBAABBA**, recovery from JSON completion mtimes (UTC−04:00): A1 16:24:29 → B1 16:26:42 → B2 16:27:50 → A2 16:29:56 → A3 16:31:59 → B3 16:34:05 → B4 16:36:13 → A4 16:38:58.

## What was tested (C4)

C4 is an **uncommitted worktree delta on `12daf4fe99017a88242469f60853b860609c20ab`** (`perf/passback-opt`). Its own `CHANGELOG` entry (verbatim in `source.patch`) states three deltas:

1. **Pinned staging of the split arm's four transfers** (activation D2H, AWQ-scale D2H on rotate, accumulator D2H for the residual form, CPU-rows H2D return) into the reusable pinned transfer buffer the host-executed MoE splice already owns, completed on the GPU arm's stream — the read still completes before the GPU arm launches, the H2D is completed as the join only **when sampled**.
2. **Asynchronous CPU-row return for unsampled split steps** — `cpu_arm_finish`/`upload_arm_result` take a private "wait for return" argument; every whole-CPU caller (`offload_exec=cpu`, `share = 0`, seeding probe, `run_host_mapped_gemv{,_residual}`) still completes it as before, while `offload_split` chooses per step. A **sampled** step (~one in four, top two bits of a per-shape LCG) waits; the rest return asynchronously.
3. **CPU-exec route ordering fix** — the whole-CPU read now synchronizes the active non-blocking stream before its blocking copy (a GPU consumer enqueued on a `hipStreamNonBlocking` stream could otherwise read stale bytes before the residual `add_inplace`).

No source commit exists for C4 (deliberately not invented): the measured delta is identified by the frozen binary md5s and by [`source.patch`](data-2026-10-16-passback-asynchronous-split-returns/) (75,671 B, sha256 `95c0e7c0…b4551`, md5 `2dfc7205…`), which touches `CHANGELOG.md`, `crates/hipfire-arch-qwen35/tests/gpu_gemv_parity.rs`, `crates/hipfire-dispatch/src/cpu_exec.rs`, `crates/hipfire-dispatch/src/offload_split.rs`. The C3 pinned storage is the **functional prerequisite** of C4 (§C3 link).

## Method

- **9B / MQ4 / budget 16 / `HIPFIRE_OFFLOAD_EXEC=passback` / `HIPFIRE_OFFLOAD_PASSBACK_SHARE=auto`**, greedy (temp 0.0), thinking off, `max_tokens 128`, `max_seq 4096`, spec off (AR), `kv auto` → K=q8 V=q8 VMM, `HIPFIRE_DPM_WARMUP_SECS=10`. Port 11541, serial hardware.
- **Four requests per fresh serve process**: row 1 is the **throwaway** (JIT/DPM/alloc settling), rows 2–4 are measured, and **their median is ONE process sample**. The 3 warm rows inside a process are **not independent samples** and are never pooled; per-arm N is the number of **fresh processes**.
- **Primary N=4 processes per arm** (not N=12): `A1..A4` baseline, `B1..B4` C4, order ABBAABBA. Secondary N=3/arm; controls N=3 each.
- **Trace OFF for every headline row** — `ab_sample.sh` / `ab_sample_river.sh` `env -u` TRACE/PROFILE/PROFILE_DECODE/MEMSET_DUMP; all 21 meta logs record `unset check (must be empty): []`, and the `trace-line check (expect 0)` prints `1` = the load-time banner `cpu exec: 16/16 spilled layers fully covered by the CPU seam…`, never a per-step line. The separately archived `C4-trace-attribution` run is **profiled by design** and is **not** a headline sample (§Attribution).
- **Cold row 1 discarded everywhere — no cold-start claim.** Primary cold rows: 8.8–9.1 tok/s decode, 27.16–27.70 s wall, 21.03–21.21 s TTFT, 13.02–13.15 s prefill; river cold rows (different prompt): 8.5–8.8 tok/s, 28.19–29.13 s wall.
- Counting unit is visibly **processes**; the brief's "N=4 not N=12" and "N=3" are process counts.

## Results — 9B primary (per process; row 1 excluded)

| sample | arm | warm rows 2–4 (tok/s) | **process median** | wall med (min–max) | TTFT med | prefill med |
|---|---|---|---:|---|---:|---:|
| A1 | baseline | 20.3, 19.7, 19.7 | **19.70** | 7.080 (6.899–7.091) | 0.638 | 587.9 |
| A2 | baseline | 19.8, 19.8, 19.6 | **19.80** | 7.055 (7.045–7.125) | 0.637 | 582.9 |
| A3 | baseline | 19.8, 19.8, 19.9 | **19.80** | 7.040 (7.032–7.061) | 0.634 | 584.3 |
| A4 | baseline | 19.7, 19.8, 19.7 | **19.70** | 7.088 (7.067–7.092) | 0.636 | 585.8 |
| B1 | C4 | 21.1, 21.0, 21.1 | **21.10** | 6.656 (6.652–6.687) | 0.631 | 584.4 |
| B2 | C4 | 21.0, 20.9, 20.5 | **20.90** | 6.714 (6.682–6.823) | 0.629 | 583.5 |
| B3 | C4 | 21.2, 21.1, 20.9 | **21.10** | 6.647 (6.636–6.694) | 0.636 | 589.4 |
| B4 | C4 | 19.7, 19.9, 20.2 | **19.90** | 7.019 (6.941–7.086) | 0.630 | 582.2 |

- Arm medians **19.75 → 21.0 tok/s = +6.3%**; process-median ranges do not overlap (baseline 19.7–19.8, C4 19.9–21.1).
- **The one slow C4 process (B4, 19.9) is itself above every baseline process**; the one low C4 warm row (B4 row 2 = 19.7) is below baseline's high row (A1 row 2 = 20.3), so the effect is not "every row beats every row".
- Warm wall arm median **7.067 → 6.685 s**; TTFT arm median **0.637 → 0.631 s**; prefill arm medians 585.0 → 584.0 ms. Latency bands overlap; only the decode rate separates.

## Results — secondary prompt (river), **PROVISIONAL**

Fresh fixture `warm3x_prose_river_short.json` = 4 exact copies of `benchmarks/prompts/prose_river_short.txt` bytes (260 B, no normalization), round-trip byte-verified; wrapper `ab_sample_river.sh` = `ab_sample.sh` with only the fixture swapped (plus a `RAYON` print). N=3 processes per arm, operator-stated interleave A B B A A B (see §Provenance gaps — **not** receipt-verifiable).

| process | arm | warm rows 2–4 | **process median** | wall med | TTFT med |
|---|---|---|---:|---:|---:|
| S1A1 | original | 19.7, 19.7, 18.8 | **19.70** | 6.706 | 0.255 |
| S1A2 | original | 19.5, 18.4, 17.8 | **18.40** | 7.188 | 0.262 |
| S1A3 | original | 19.3, 18.5, 18.4 | **18.50** | 7.120 | 0.259 |
| S1B1 | C4 | 19.4, 19.5, 18.9 | **19.40** | 6.809 | 0.255 |
| S1B2 | C4 | 19.5, 19.3, 18.5 | **19.30** | 6.885 | 0.278 |
| S1B3 | C4 | 19.2, 19.4, 19.2 | **19.20** | 6.872 | 0.257 |

Arm medians **18.5 → 19.3 tok/s = +4.3%**, directionally consistent with the primary. **This is NOT a clean replication.** The original arm's last two processes dipped late-session (S1A2 rows 18.4/17.8, S1A3 rows 18.5/18.4) while C4 stayed tight (all its process medians 19.2–19.4 and every C4 process median ≥ every original process median); row-level ranges still overlap (original 17.8–19.7, C4 18.5–19.5). The dip pattern is consistent with **late-session drift / contention or ordering**, which the receipts cannot separate from a real effect. Recorded as **provisional direction only** — no claim that C4 replicates on this prompt.

## Results — CPU / PCIe controls (C4 binaries)

N=3 each, operator-stated interleave CPU PCIe CPU PCIe CPU PCIe, same fixture and method; all six runs functional, decoded text valid code on every row.

| mode | warm rows 2–4 → process medians | **arm median** | warm wall med | warm TTFT med |
|---|---|---:|---:|---:|
| CPU | 16.50, 16.80, 16.90 | **16.8** (range 16.5–16.9) | 8.367 / 8.181 / 8.164 s | 0.654 / 0.641 / 0.640 |
| PCIe | 13.00, 13.00, 13.00 | **13.0** | 10.421 / 10.446 / 10.440 s | 0.667 / 0.662 / 0.652 |

Relative to the **primary C4 passback** arm median (21.0 tok/s): **CPU +25%** (21.0/16.8), **PCIe +62%** (21.0/13.0) in favor of C4 passback. **Provenance caveat:** these are **matched C4 binaries on the current primary controls**, but the modes are **not a strict paired four-arm round** — the CPU/PCIE window is temporally separate from the C4/baseline primary window (primary receipt completion 16:24–16:39; controls/secondary 17:0x), so **drift is acknowledged** and the ratios are cross-window, not paired. No bit-identity is asserted between mode pairs; `passback share auto` is non-deterministic by design (its own serve banner says so) and **a generic auto-passback run is not a golden reference**.

## share0 vs CPU identity smoke (fixture-scoped)

`HIPFIRE_OFFLOAD_PASSBACK_SHARE=0` on the C4 binary, primary fixture, greedy, 4 rows. Serve log records `passback: … share 0; min step 2 MiB` and `cpu exec: 16/16 spilled layers fully covered by the CPU seam`. All 4 rows emit the **identical 520-char code text** (`sha256 7690b0d3…`), `gen=128`, `finish=length`, warm median **16.9 tok/s ×3**.

**PASS, on this fixture only:** share0's bytes are identical to **all 12 CPU rows** and to all 32 primary baseline/C4 rows (`request_md5 dd93dcc0…`). **Scope limit:** the harness records **decoded text + generation count + `request_md5`**, not raw greedy **token IDs** (no such API in the receipt), so identity is asserted at the text/gen/request level — the documented "exact CPU twin" behavior is confirmed for this fixture, not re-established as a general invariant.

## Attribution (profiled, N=1 — mechanism, not headline)

`C4-trace-attribution` is a **separate, profiled** C4 serve process run after the unprofiled A/B (receipt 16:41:15), **not** a member of the N=4 arms; it has **no `.meta.log`** (no env/power receipt). Its own row 1 is cold (9.1 tok/s, 27.199 s); warm rows are 21.1/21.1/21.1 — quoted here only as traced-process context, not as a sample.

Trace observations (final cumulative lines of the 192-line serve log):

- `gemv m=4096 k=12288` (rotated + residual): **share `0.230` frozen** (64 applied, 0 reopens), mean per split step `d2h=0.08 ms`, `gemv=0.48 ms`, `return=0.01 ms`; final `gpu samples 1039, 2051 sampled`. **The k12288 split still freezes low and unbalanced** — C4's sampled joins did **not** move it toward balance.
- `gemv m=4096 k=4096`: share **`0.350` frozen** (108 applied, 2 reopens).
- Window idle lines report `cpu exec: idle 3.6%` / `4.0%` (17 µs CPU/step) — explicitly a **GPU-idle lower bound** over a wall window, not a per-step accounting.
- `split` transfer means stay sub-0.1 ms `d2h` / sub-0.03 ms `return`; the split wall is dominated by the GPU `gemv` arm.

**Mechanism statement (as required):** C4 combines **pinned async H2D + sampled joins**. The `+6.3%` **cannot be attributed solely to copy overlap**: sampling also changes the controller's **feedback density** (fewer per-step wait observations change the share trajectory), and the traced split shares are similar to before. The **GPU-side actual elapsed time was not measured**, so nothing here shows `0.35`–`0.45` is optimal, nor that the k12288 freeze is under-allocation. Mechanism is stated as the *combined* change; no single-cause claim.

## Output health

All **32** primary rows (8 processes × 4): `finish="length"`, `runaway=true`, `attractor=false`, `empty=false`, `saw_done=true`, `terminal_count=1`, `post_terminal_bytes=0`, `think_words=0`, `ans_words=70`, `gen=128`, `ctx=59`, `cached=0`, spec fields (`tau`/`dflash`/`mtp`/`ngram_*`) **null**, `decode_estimated=false` — the 128-token cap truncating a valid Python answer (length cap, not a degenerate loop). All 32 rows **byte-identical**: `md5 2ab86019e06ad00b7a003df166d72be5`, `sha256 7690b0d32d444edec66c90d6326ad69b9249ca5affdabda50ef31e6306bfcacc` (520 B), `request_md5 dd93dcc0528f194dbd44994d188b03d8`, `prompt_md5 5835c71e471849b4a72e1dc8e39695e7` — the same text as the C1/C2/C3 rows. The `+6.3%` is a **rate** effect on identical output.

Secondary river: all **24** rows identical `md5 43d9a0f5f5f2a0a38a8a44fc927b7ce2` (591 B), same flags, `ans_words=110`, `ctx=65`, `request_md5 20833ab1d0d9766a969f24398996741d`, `prompt_md5 07a7880965142971dbb3cc7493f8fb94`. CPU/PCIE/share0: all rows identical to the primary text (`sha256 7690b0d3…`).

## Fixture identity (measured in this host, verified against the receipts)

| artifact | digest |
|---|---|
| branch / base source | `perf/passback-opt` off **HEAD `12daf4fe99017a88242469f60853b860609c20ab`** (hipfire 0.4.3), base `upstream/beta`; `cargo build --release` (not `-p`, feature unification differs) |
| baseline binaries | `hipfire` md5 `c9feb2095ffe043c7e93477a7ec02a9a` (47,336,152 B); `daemon` md5 `c1da18f268129005c6e827ea85ceff49` (88,582,088 B) — frozen copies `baseline-bin/hipfire-12daf4fe`, `baseline-bin/daemon-12daf4fe` |
| C4 binaries | `hipfire` md5 `fb32e2e2eab279381a36f59672f1acbc` (47,337,536 B); `daemon` md5 `d400ea368e1258d14c29733293c6c153` (88,582,384 B) — `candidate4-bin/` |
| C4 source delta | `candidate4-bin/source.patch` 75,671 B, **sha256 `95c0e7c03d5b5c0d50c1228244cadde9e699ceee22ff9f8022e4bb13a38b4551`**, md5 `2dfc7205c0fca96a81e184d32fb79558`; archived as `source.patch` |
| model | `~/.hipfire/models/qwen3.5-9b.mq4`, md5 `296092bf1e6a45d78c1acf815eb93366`, 5,313,750,016 B, tag `qwen3.5:9b` |
| primary fixture | `fixtures/warm3x_gpu_offload_probe.json`, md5 `c4c78f65b100f645cd1ab3ff0fcbf413`; inner prompt `benchmarks/prompts/gpu_offload_probe.txt` md5 `5835c71e471849b4a72e1dc8e39695e7` (215 B), `request_md5 dd93dcc0528f194dbd44994d188b03d8` |
| secondary fixture | `fixtures/warm3x_prose_river_short.json` md5 `da9ca66228d1c686dd98e582f727c374` (= `C4SEC-warm3x_prose_river_short.json`); inner prompt `benchmarks/prompts/prose_river_short.txt` md5 `07a7880965142971dbb3cc7493f8fb94` (260 B), `request_md5 20833ab1d0d9766a969f24398996741d` |
| samplers | `fixtures/ab_sample.sh` md5 `616359700ad51339705d82cf7c82ccc0`; `ab_sample_river.sh` md5 `51bed78235a94310695ec5c88c63d544` (identical to the `fixtures/` copy); `power_snap.sh` md5 `6726f0f70161abb83bc6a3bfdbfe5a03` |
| route envs | `HIPFIRE_DPM_WARMUP_SECS=10 HIPFIRE_OFFLOAD_EXEC=passback HIPFIRE_GPU_LAYER_BUDGET=16 HIPFIRE_OFFLOAD_PASSBACK_SHARE=auto` (controls overrode `EXEC`/`SHARE`) |

Host: 1× Radeon RX 9070 XT `gfx1201` (16,304 MiB), ROCm/HIP 7.2, Ryzen 7 7800X3D (8c/16t), 28 GB RAM. Per-run load banner: 16 of 32 layers host-placed, device weights 3299 MiB of 14697 MiB, pinned host 1753 MiB, `passback: 16/16 spilled layers splittable; share auto; min step 2 MiB`, cache warming 10.00 s, `hipGraph capture disabled (host-executed steps present)`; `KV: mode=auto → K=q8 V=q8`; DeltaNet state Q8.

## Power / clock / temperature (µW snapshots only — NOT steady-state)

sysfs (card1) `power1_average_uW` pre → post via `power_snap.sh`, every meta log. **`rocm-smi --showpower` returned empty in all 21 meta logs** (`=== Power Consumption ===` has no lines). Every run idles at `perf_level=auto`, `power1_cap_uW = 374,000,000` (374 W), `throttle_log` = the "thermal throttling logging enabled" banner only, junction 42–53 °C.

| run | power pre → post (µW; = W) | edge °C | junction °C | mem °C | `sclk_Hz` pre → post |
|---|---|---|---|---|---|
| C4-A1 | 33 → 50 (33 → 50 W) | 45→50 | 46→52 | 68→72 | 75e6 → 23e6 |
| C4-A2 | 47 → 51 | 48→51 | 50→53 | 72→74 | 41e6 → 41e6 |
| C4-A3 | 46 → 124 | 47→51 | 49→53 | 70→74 | 921e6 → 43e6 |
| C4-A4 | 47 → 127 | 47→51 | 49→53 | 70→74 | 79e6 → 41e6 |
| C4-B1 | 46 → 51 | 46→51 | 48→53 | 70→74 | 237e6 → 41e6 |
| C4-B2 | 46 → 51 | 47→51 | 48→53 | 70→74 | 59e6 → 41e6 |
| C4-B3 | 47 → 124 | 47→51 | 49→53 | 70→74 | 30e6 → 41e6 |
| C4-B4 | 46 → 89 | 47→51 | 49→53 | 70→74 | 41e6 → 92e6 |
| CPU1/2/3 | 47→63 / 47→66 / 48→66 | 49–51→50–53 | — | — | 0/0/41e6 → 41e6/41e6/53e6 |
| PCIE1/2/3 | 30→140 / 47→131 / 47→147 | 48–49→49–51 | — | — | 0/9e6/41e6 → 54e6/41e6/219e6 |
| SHARE0 | 47 → 65 | 51→53 | 51→53 | 72→74 | 12e6 → 1610e6 |
| S1A1/A2/A3 | 29→116 / 46→90 / 48→76 | 40–49→48–51 | — | — | 1503e6/41e6/915e6 → 940e6/1601e6/58e6 |
| S1B1/B2/B3 | 47→69 / 47→105 / 47→73 | 45–49→48–51 | — | — | 0/0/41e6 → 58e6/1559e6/70e6 |

These are **point snapshots at differing DPM states** (pre-post `sclk` swings bear no relation to arm), **not comparable across arms**; **no power or energy claim is drawn**.

## C3 / C4 link and other rejected ledgers

- **C3 (pinned-copy split transfer with a completion wait before every return, i.e. the all-waits variant + CPU-worker screen)** was `rejected` as a standalone optimization (−1.5% overall, fully overlapping process ranges) — its own A/B was **null**, and its pinned storage carries forward only as the **functional prerequisite** of C4 here ([`C3`](2026-10-16-passback-pinned-copy-worker-experiments.md)). C4 is the first record in which that storage is exercised with **async split returns** (waits only on sampled steps).
- **C1** (interval) and **C2** (shape-key) were fully reverted; see [`C1`](2026-10-16-passback-timing-interval-experiment.md), [`C2`](2026-10-16-passback-shape-isolation-experiment.md) + [`amendment 1`](2026-10-16-passback-shape-isolation-experiment-amendment-1.md).

## Status

The C4 worktree delta **(with its binaries and source patch) is frozen as of this record**; the C4 source patch is the frozen artifact this evidence binds to, and C5 work proceeds against source separately. **C5 is not measured and not part of this record.** Correctness/validation (library suite, split hardware oracle, CPU-stream oracle, lifetimes review) is **not** claimed here — it was reported through the campaign ledger, not re-measured in this record. **27B: not validated** in this record (campaign policy: 9B iterative, 27B only at final validation); **no 27B sample exists here**.

## What was NOT measured / provenance gaps (recorded honestly)

- **C4 effect on cold start** — not measured (row 1 discarded); the cold rows are recorded but no claim is made.
- **Secondary (river) transfer** — **provisional**, N=3/arm, single prompt family, late-session dip/overlap unseparated from a real effect; not a clean replication.
- **CPU/PCIE controls** — not a strict paired four-arm round; separate time window, drift acknowledged; no bit-identity asserted between mode pairs.
- **Raw greedy token IDs** — never available in the harness output; share0/CPU/C4/baseline identity is asserted at **decoded text + `gen` + `request_md5`** level only.
- **`C4-trace-attribution` has no `.meta.log`** — the profiled run carries no env/power/clock receipt (its serve log is archived in full).
- **`RAYON_NUM_THREADS` unset** is **receipted only for the 6 river runs** (`RAYON check (must be unset): [unset]`, added by `ab_sample_river.sh`). The primary, CPU, PCIE and share0 meta logs (15 runs) record only the TRACE/PROFILE/MEMSET `unset check` — the primary wrapper `ab_sample.sh` has no `RAYON` line. RAYON-unset for those runs rests on the operator's clean-shell invocation (the wrapper does not clear `RAYON_*` or set affinity/`taskset`), **not** on a receipt line.
- **C4SEC run order** is **not receipt-verifiable** — all C4SEC JSON completion times are the same bulk timestamp (17:08:15, a copy artifact); the CPU/PCIE interleave and the river A B B A A B order are operator statements only. (Primary ABBAABBA **is** receipt-verified by spread mtimes.)
- **GPU-side actual elapsed / true optimum share** — not measured; the trace cannot show `.35`–`.45` is optimal nor that the k12288 freeze is under-allocation.
- **`rocm-smi --showpower` unavailable**; telemetry is sysfs point snapshots only.
- No share/budget sweep, no `min_step` sweep, no other model/quant/GPU, no `HIPFIRE_PROFILE` headline runs, no 27B, no C5.

## Reproduction

```
bash ab_sample.sh <cli_bin> <daemon_bin> passback auto 16 ~/.hipfire/models/qwen3.5-9b.mq4 qwen3.5:9b 11541 /tmp/ab/home-<X> /tmp/ab/<X>-passback
```
A = baseline (`c9feb209`/`c1da18f2`), B = C4 (`fb32e2e2`/`d400ea36`), order ABBAABBA ×4 processes/arm. Secondary: `bash ab_sample_river.sh …` on the river fixture, A B B A A B ×3/arm. Controls: `… cpu 16 …` and `… pcie 16 …` on the C4 binaries, interleaved ×3 each. share0: `… passback 0 …`.

## Raw evidence

**72 files** in [`data-2026-10-16-passback-asynchronous-split-returns/`](data-2026-10-16-passback-asynchronous-split-returns/) (71 receipts + `SHA256SUMS`), every receipt `cmp`-verified byte-identical to its source in `.codeinsight+research/passback-opt/`:

- `C4-{A1..A4}-baseline.{json,serve.log,meta.log}`, `C4-{B1..B4}-candidate4.{json,serve.log,meta.log}` — primary A/B: full harness JSON (all 4 rows, complete `content`/`assistant_content`), full serve logs, env/power/temperature meta logs.
- `C4-trace-attribution.{json,serve.log}` — profiled N=1 mechanism run (full 192-line trace retained verbatim, **not** trimmed); **no meta log exists**.
- `C4SEC-S1{A1,A2,A3,B1,B2,B3}-river.{json,serve.log,meta.log}` — secondary river arm.
- `C4SEC-CPU{1,2,3}.{json,serve.log,meta.log}`, `C4SEC-PCIE{1,2,3}.{json,serve.log,meta.log}`, `C4SEC-SHARE0.{json,serve.log,meta.log}` — controls + identity smoke.
- `C4SEC-ab_sample_river.sh`, `C4SEC-warm3x_prose_river_short.json`, `ab_sample.sh`, `power_snap.sh`, `warm3x_gpu_offload_probe.json` — exact wrappers, fixtures, power-snapshot helper.
- `source.patch` — the C4 frozen source delta (text only; **no executables are committed**; the frozen binaries stay in `.codeinsight+research/passback-opt/{baseline-bin,candidate4-bin}/`).
- `SHA256SUMS` — integrity digests of all 71 receipts.
