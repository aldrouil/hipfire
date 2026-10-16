# Passback pinned-copy + CPU worker experiments (candidate 3, rejected; CPU-worker screen, exploratory) — 2026-10-16

**Lifecycle:** `historical`. Two independent items, both **not** admissions:

- **Candidate 3 (C3) — `rejected`.** A pinned-copy split-transfer candidate on the 9B primary: **−1.5%** on the overall process median (baseline 19.85 vs candidate 19.55 tok/s) with **fully overlapping** process ranges. **Not retained as a standalone optimization.** Its owned storage landed as a **functional prerequisite** of a separate **C4 actual-async-return** experiment (not yet measured). C1 (interval) and C2 (shape-key) were **fully reverted**.
- **CPU-worker screen (`RAYON-*`) — `exploratory` / `null`.** Single-shot (N=1 per config) env screen on the **frozen baseline** binaries: `RAYON_NUM_THREADS=16` ≈ `8` (parity, ~19.9–20.0 tok/s); `=4` collapsed to **14.5** (slow). **No accepted performance claim.**

Neither is a [`docs/BENCHMARKS.md`](../BENCHMARKS.md) claim, and neither transfers across host, model, quant, GPU, prompt, route, or method. **New** checkpoint; no existing checkpoint or amendment is modified (append-only: [`README.md`](README.md) §Preservation). Related: [`C1`](2026-10-16-passback-timing-interval-experiment.md), [`C2`](2026-10-16-passback-shape-isolation-experiment.md) + [`C2 amendment 1`](2026-10-16-passback-shape-isolation-experiment-amendment-1.md). **Cause is NOT established:** the −1.5% is the observed whole effect on this fixture; the receipts hold no trace/counter/share data that attributes it. No gain claimed, no mechanism asserted.

## What was tested (C3)

C3 is an **uncommitted worktree delta** on the original `12daf4fe99017a88242469f60853b860609c20ab` (per the campaign ledger, not a result of this record): the D2H split batches into pinned storage **with a completion wait before return**, and the H2D arm likewise copies into pinned storage **with completion before every return**; the **original scheduler is retained**. A separate fix in the same worktree corrected a **CPU-legacy active-producer drain bug**; **default CPU behaviour is unchanged**. No source commit hash exists (deliberately not invented) — the measured change is identified by the frozen binary md5 below.

## Method

Single-prompt-family, single-share, fresh-process, **unprofiled** A/B. All 9B, budget 16 (16 of 32 layers host-placed), `HIPFIRE_OFFLOAD_EXEC=passback`, share `auto`.

- **Cold row 1 discarded** (fresh-process JIT/DPM settling); **warm rows 2–4 median** is the per-process statistic (N=3 rows → median).
- **Fresh process per sample, N=4 per arm** (`A1..A4` baseline, `B1..B4` candidate 3).
- **Trace OFF** — `ab_sample.sh` unsets `HIPFIRE_CPU_EXEC_TRACE`/`PROFILE`/`PROFILE_DECODE`/`MEMSET_DUMP`; every meta log shows `unset check (must be empty): []`. The meta `trace-line check (expect 0)` prints `1` — the load-time banner `cpu exec: 16/16 spilled layers fully covered by the CPU seam…`, not a per-step line; all runs are genuinely unprofiled.
- **Run order (observed ABBAABBA)** from JSON completion mtimes: A1 15:35:09 → B1 15:37:21 → B2 15:39:28 → A2 15:41:39 → A3 15:44:03 → B3 15:46:09 → B4 15:48:16 → A4 15:50:23 (UTC−04:00).

## Results (9B primary, C3)

Decode `tok/s` (row 1 shown then excluded):

| sample | arm | row 1 (disc.) | warm rows 2–4 | min | **median** | max |
|---|---|---:|---|---:|---:|---:|
| A1 | baseline | 9.0 | 19.8, 19.5, 19.6 | 19.5 | **19.6** | 19.8 |
| A2 | baseline | 8.9 | 20.1, 20.1, 20.1 | 20.1 | **20.1** | 20.1 |
| A3 | baseline | 8.9 | 20.0, 19.7, 19.5 | 19.5 | **19.7** | 20.0 |
| A4 | baseline | 8.9 | 19.9, 20.0, 20.0 | 19.9 | **20.0** | 20.0 |
| B1 | candidate 3 | 8.8 | 20.1, 20.0, 19.8 | 19.8 | **20.0** | 20.1 |
| B2 | candidate 3 | 8.7 | 19.2, 19.2, 19.2 | 19.2 | **19.2** | 19.2 |
| B3 | candidate 3 | 8.7 | 19.2, 19.4, 19.8 | 19.2 | **19.4** | 19.8 |
| B4 | candidate 3 | 8.9 | 20.1, 19.6, 19.7 | 19.6 | **19.7** | 20.1 |

| arm | process medians | min / median / max | mean | pop. var / sd | sample var / sd | range |
|---|---|---|---:|---|---|---|---:|
| baseline (A) | 19.6, 20.1, 19.7, 20.0 | 19.6 / **19.85** / 20.1 | 19.850 | 0.0425 / 0.2062 | 0.0567 / 0.2380 | 0.5 |
| candidate 3 (B) | 20.0, 19.2, 19.4, 19.7 | 19.2 / **19.55** / 20.0 | 19.575 | 0.0919 / 0.3031 | 0.1225 / 0.3500 | 0.8 |

**Overall process median: baseline 19.85 → candidate 19.55 tok/s = −0.30 tok/s (−1.5%).** The arms **fully overlap** (A 19.6–20.1, B 19.2–20.0), and the candidate's best process (B1 20.0) exceeds the baseline's worst (A1 19.6) — unlike C2 this sits inside the combined scatter and is consistent with shared session drift, so it is **not** a clean separation. Recorded as whole effect only.

**Counting unit: processes, not rows** — the headline N is **4 independent fresh processes per arm**, each contributing one median over 3 warm rows; the 12 warm rows/arm (16 with the discarded cold row) are **not** independent samples and are never pooled.

## Latency (warm rows 2–4, per-arm range)

| arm | prefill | prefill tok/s | TTFT | wall |
|---|---|---|---|---|
| baseline A1–A4 | 571.8–591.9 ms | 99.7–103.2 | 0.620–0.642 s | 6.938–7.152 s |
| candidate 3 B1–B4 | 575.4–590.8 ms | 99.9–102.5 | 0.624–0.641 s | 6.960–7.265 s |
| cold row 1 (disc.) | 12994.5–13082.5 ms | 4.5 | 20.98–21.16 s | 27.28–27.72 s |

Bands overlap; no latency win/loss.

## Output health

All **32** C3 rows (8 runs × 4): `finish="length"`, `runaway=true`, `attractor=false`, `empty=false`, `saw_done=true`, `terminal_count=1`, `post_terminal_bytes=0`, `think_words=0`, `ans_words=70` — the 128-token cap truncating a valid Python answer ("length cap", not a degenerate loop). All 32 rows **byte-identical**: single-row `sha256 = 7690b0d32d444edec66c90d6326ad69b9249ca5affdabda50ef31e6306bfcacc` (521 B); identical `request_md5 = dd93dcc0528f194dbd44994d188b03d8` and `prompt_md5 = 5835c71e471849b4a72e1dc8e39695e7`; the decoded text is **byte-identical to the C1/C2 rows** (`longest_substring` truncated mid-comment) — the −1.5% is a **rate** effect on identical output, not an output change.

## Worker screen (RAYON, exploratory, N=1/config, frozen baseline only)

**Not an A/B and not N=3.** One fresh process per config, on the **frozen baseline** binaries (md5 `c9feb209`/`c1da18f2`); `RAYON_NUM_THREADS` set as an external env prefix on the `ab_sample.sh` invocation (the wrapper does not clear `RAYON_*`; its `env -u` list is TRACE/PROFILE/MEMSET only). Same fixture/method (row 1 discarded, rows 2–4 median).

| config (`RAYON_NUM_THREADS`) | median tok/s (warm 2–4) | row 1 | warm prefill / TTFT / wall |
|---|---:|---:|---|
| R16a = 16 (explicit) | **20.0** | 8.8 | 582–586 ms / 0.631–0.634 s / 6.975–7.064 s |
| R8 = 8 | **19.9** | 8.9 | 574–580 ms / 0.621–0.631 s / 6.978–7.034 s |
| R4 = 4 | **14.5** | 7.6 | 577–593 ms / 0.643–0.659 s / 9.373–9.443 s |
| R16b = 16 (explicit repeat) | **19.9** | 8.9 | 574–583 ms / 0.624–0.632 s / 6.961–7.043 s |

**Read:** 16 ≈ 8 (parity, all 8 documented values in the ~19.8–20.1 band) → **null** at 16 vs 8; 4 is clearly **slower** (14.5, ~27% below the 16/8 level; row 1 also slowest 7.6, wall ~9.4 s). The R16a/R16b repeat pair agrees (20.0 vs 19.9).

**Verification gap (preserved):** the **daemon-side Rayon pool count was never directly observed** — no serve-log line exposes the rayon worker count (`grep RAYON|thread|pool` in the four `RAYON-*.serve.log` finds nothing). The `R4=14.5` collapse is an **observed negative effect of the env setting**, **not** a verified pool-count claim and **not proof** of anything about pool sizing. The daemon/serve env allowlist banner is unchanged across the four runs; thread topology (first siblings CPU0–7, seconds CPU8–15 on the 8c/16t Ryzen 7 7800X3D) is recorded from the campaign with **no** affinity/`taskset` run.

**Rayon env + wrapper passthrough (verbatim in `RAYON-pins.txt`):** external env prefix; wrapper does not clear `RAYON_*`; default (unset) behaviour for all A/B arms is a clean shell (no RAYON/taskset/affinity vars); future arms run with `RAYON_NUM_THREADS` unset.

**RAYON run order is NOT recoverable from the receipts.** All four JSONs carry the same completion time (15:58:32.883–.884, within ~1 ms) and all four meta logs 16:00:13.229 — impossible for four independent ~65 s runs, so mtimes are a bulk-write/copy artifact. `RAYON-pins.txt` lists the configs in the operator order R16a, R8, R4, R16b (R16b = explicit repeat), but that is an operator note, **not** a receipt-verified execution order. The C3 A/B mtimes above *are* spread ~2 min apart and match the stated ABBAABBA.

## Fixture identity (measured)

| artifact | digest |
|---|---|
| branch / base source | `perf/passback-opt` off `12daf4fe99017a88242469f60853b860609c20ab` (hipfire 0.4.3); `cargo build --release` |
| baseline binaries (A and all RAYON runs) | `hipfire` md5 `c9feb2095ffe043c7e93477a7ec02a9a`; `daemon` md5 `c1da18f268129005c6e827ea85ceff49` |
| candidate 3 binaries (B) | `hipfire` md5 `f79b5775d1cc3a546567e3d9590d728f` (47,337,600 B); `daemon` md5 `fe64ebecbdad776b15e848aec14aabc6` (88,582,488 B) |
| frozen binary copies | `.codeinsight+research/passback-opt/{baseline-bin/hipfire-12daf4fe,baseline-bin/daemon-12daf4fe,candidate3-bin/hipfire,candidate3-bin/daemon}` |
| model | `~/.hipfire/models/qwen3.5-9b.mq4`, md5 `296092bf1e6a45d78c1acf815eb93366`, 5,313,750,016 B, tag `qwen3.5:9b` |
| prompt fixture | `warm3x_gpu_offload_probe.json` (4 identical prose rows), md5 `c4c78f65b100f645cd1ab3ff0fcbf413`, sha256 `1e058b674082be29a6245eaa526c7100792abc6fa81f09e01ed032ffd9f8e60a`, 1,042 B |
| inner prompt text | md5 `5835c71e471849b4a72e1dc8e39695e7` (215 B; = `benchmarks/prompts/gpu_offload_probe.txt`) |
| sampling / KV | greedy (temp 0.0), thinking off, `max_tokens 128`, `max_seq 4096`, spec off (AR); `kv auto` → K=q8 V=q8 VMM (`kv_backend = vmm`), 8/32 layers carry KV; DeltaNet state Q8 |
| route envs | `HIPFIRE_DPM_WARMUP_SECS=10 HIPFIRE_OFFLOAD_EXEC=passback HIPFIRE_GPU_LAYER_BUDGET=16 HIPFIRE_OFFLOAD_PASSBACK_SHARE=auto` |
| harness / port | `ab_sample.sh` over `scripts/serve_harness.py --mode battery`; serve-log port 11541 (all 12 runs) |

Host: 1× Radeon RX 9070 XT `gfx1201` (16304 MiB), ROCm/HIP 7.2, Ryzen 7 7800X3D (8c/16t), 28 GB RAM. Load banner per run: 16 of 32 host-placed, device weights 3299 MiB of **14697 MiB** (all 12 runs), pinned host 1753 MiB, `passback: 16/16 spilled layers splittable; share auto; min step 2 MiB`; cache warming 10.00 s; `passback: hipGraph capture disabled`.

## Power / clock / memory telemetry (units: µW; µW / 1e6 = W)

sysfs (card1) via `power_snap.sh`, pre → post, every meta log. `rocm-smi --showpower` was **empty in all 12 logs** (unavailable); `=== Power Consumption ===` sections contain no lines.

| run | `power1_average_uW` pre → post (µW; = W) | edge mC | junction mC | mem mC | `sclk_Hz` pre → post |
|---|---|---|---|---|---|
| C3-A1 | 44,000,000 → 49,000,000 (44 → 49 W) | 49→51 | 50→53 | 70→72 | 222e6 → 73e6 |
| C3-A2 | 46,000,000 → 75,000,000 (46 → 75 W) | 47→51 | 48→53 | 70→74 | 299e6 → 1600e6 |
| C3-A3 | 47,000,000 → 123,000,000 (47 → 123 W) | 47→51 | 49→53 | 70→74 | 0 → 866e6 |
| C3-A4 | 46,000,000 → 129,000,000 (46 → 129 W) | 47→51 | 49→53 | 70→74 | 0 → 58e6 |
| C3-B1 | 46,000,000 → 105,000,000 (46 → 105 W) | 47→51 | 48→53 | 70→74 | 210e6 → 93e6 |
| C3-B2 | 46,000,000 → 98,000,000 (46 → 98 W) | 47→49 | 49→51 | 70→72 | 518e6 → 58e6 |
| C3-B3 | 47,000,000 → 105,000,000 (47 → 105 W) | 47→50 | 48→52 | 70→72 | 0 → 41e6 |
| C3-B4 | 46,000,000 → 125,000,000 (46 → 125 W) | 46→51 | 48→53 | 70→74 | 0 → 29e6 |
| R16a | 33,000,000 → 120,000,000 (33 → 120 W) | 43→49 | 44→51 | 64→72 | 858e6 → 43e6 |
| R8 | 46,000,000 → 114,000,000 (46 → 114 W) | 49→49 | 50→51 | 72→72 | 604e6 → 1599e6 |
| R4 | 47,000,000 → 73,000,000 (47 → 73 W) | 49→48 | 50→49 | 72→70 | 43e6 → 945e6 |
| R16b | 47,000,000 → 129,000,000 (47 → 129 W) | 48→51 | 49→53 | 70→74 | 165e6 → 92e6 |

All 12 idle at `perf_level=auto`, `power1_cap_uW = 374,000,000` (374 W), `throttle_log` = throttling-enabled banner only (no throttle entries); junction 49–53 °C, well below any limit. These are **point snapshots** at differing `pp_sclk_star` DPM states, **not** comparable across arms — **no power or energy claim is drawn**.

## Status / correctness context / 27B

C3 **not retained as a standalone optimization** (own A/B null-to-negative). Its **pinned-copy owned storage is now a functional prerequisite** of a separate **C4 actual-async-return** experiment currently being implemented by the sole code owner; **no C4 measurements exist yet**. C1 and C2 were **fully reverted**. Per the campaign ledger (reported here, **not** re-measured in this record): master library suite passed (412); split hardware oracle bit-exact (18 cases × 3 streams); original CPU-stream oracle passed after a minimal real ordering fix; lifetimes reviewer-approved — no independent correctness claim is made here. **27B: not validated here** (campaign policy: 9B iterative, 27B only at final validation); no 27B sample exists in this record.

## What was NOT measured / missing fields

- **C3 cause** — not measured; receipts hold no trace/counter/share data (trace OFF). −1.5% recorded as whole effect only.
- **CPU-worker screen at N>1** — **N=1 per config**, no repeats beyond the single R16a/R16b pair, **no N=3**, no affinity/`taskset`, **no config-vs-baseline A/B** (baseline binary only).
- **Rayon pool count** — never observed in any log (verification gap above); R4 effect is an env-setting observation only.
- **RAYON execution order** — not recoverable (bulk-clustered mtimes); operator note only.
- **Power matching** — `rocm-smi --showpower` empty; sysfs point samples at differing DPM states; no power claim.
- No separate per-run identity file (identity lives in each `*.meta.log` + shared `IDENTITY.md`); no `min_step`/share sweep, no other model/quant/GPU, no cross-context cells, no `HIPFIRE_PROFILE` runs, no 27B.

## Reproduction

`bash ab_sample.sh <cli_bin> <daemon_bin> passback auto 16 ~/.hipfire/models/qwen3.5-9b.mq4 qwen3.5:9b 11541 /tmp/ab/home-<X> /tmp/ab/<X>-passback` — C3 arms A = baseline (`c9feb209`/`c1da18f2`), B = candidate 3 (`f79b5775`/`fe64ebec`); RAYON screen = frozen baseline only, prefix `RAYON_NUM_THREADS={16,8,4,16}` (N=1 each).

## Raw evidence

Complete raw receipts copied verbatim into `data-2026-10-16-passback-pinned-copy-worker-experiments/` (all 37 receipt files byte-identical to `.codeinsight+research/passback-opt/raw/`, `cmp`-verified):

- `C3-{A1..A4}-baseline.{json,serve.log,meta.log}`, `C3-{B1..B4}-candidate3.{json,serve.log,meta.log}` — full harness JSON (all 4 rows, complete `content`/`assistant_content`), full serve logs, identity/env/power/temperature meta logs.
- `RAYON-{R16a,R8,R4,R16b}.{json,serve.log,meta.log}` + `RAYON-pins.txt` (N=1 worker screen + pin/env record); `ab_sample.sh`, `power_snap.sh`, `warm3x_gpu_offload_probe.json` (exact sampler, power-snapshot helper, fixture); `SHA256SUMS` (integrity digests of every file above).
