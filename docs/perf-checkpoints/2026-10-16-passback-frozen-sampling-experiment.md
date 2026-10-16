# Passback frozen-sampling experiment (candidate 6, **rejected**) — 9B primary **−1.5 %** vs C5 — 2026-10-16

**Lifecycle:** `historical`. **Disposition: `REJECTED`.** This is a rejection record, not an admission, not a current baseline, and not a promotion. It measures one candidate worktree delta (`C6`, "frozen sampling" — the split arm's sampling density drops from one-in-four to one-in-sixteen once a shape's share has latched) against the **kept** candidate (`C5`, "sampled GPU-event timing") on **one fixture family, one model (9B), one quant (MQ4), one GPU (gfx1201), greedy decoding**, fresh-process, unprofiled, `passback` route with `share auto`, budget 16. The numbers here do **not** transfer across host, model, quant, GPU, prompt, route, share, budget, or method without a new record.

**Verdict: C6 shows no reproducible end-to-end gain — a small loss with a wider spread — so it is rejected and C5 is retained.** The balanced `ABBAABBA` block puts C6 at **20.3** tok/s vs C5 at **20.6** (median of four fresh-process samples per arm): **−1.5 % median / −1.8 % mean**, with C6's *best* process (20.5) below C5's *median* and C6's *worst* (19.7) the session low. Wall time moved the same way (C5 6.809 s vs C6 6.897 s, +1.3 %); TTFT was parity (0.639 vs 0.637). All 32 measured rows are byte-identical to the C1–C5 rows, so this is a **rate** effect on identical output, not an output change.

**The C6 code delta has been reverted by the code owner; C5 is kept.** Per the campaign's predefined stop rule (start the river fixture and the load-recovery diagnostic only on a C6 win), **neither was run** after the null-to-loser headline. **No 27B sample exists in this record**, and this record makes **no new promotional or overall performance claim**.

This is a **new** checkpoint; no existing checkpoint or amendment is modified (append-only: [`README.md`](README.md) §Preservation). Related candidate ledgers: [`C1`](2026-10-16-passback-timing-interval-experiment.md), [`C2`](2026-10-16-passback-shape-isolation-experiment.md) + [`C2 amendment 1`](2026-10-16-passback-shape-isolation-experiment-amendment-1.md), [`C3`](2026-10-16-passback-pinned-copy-worker-experiments.md) (`rejected`), [`C4`](2026-10-16-passback-asynchronous-split-returns-gfx1201.md), [`C5`](2026-10-16-passback-sampled-gpu-event-timing-gfx1201.md) (kept). Raw receipts: [`data-2026-10-16-passback-frozen-sampling-experiment/`](data-2026-10-16-passback-frozen-sampling-experiment/) (**35 files**: 34 receipts + `SHA256SUMS`).

## Headline (9B primary, unprofiled, N=4 fresh processes per arm)

The comparison is **C6 vs C5** (not vs C4 or the original build). Each arm's sample is the median of its three warm rows (rows 2–4) inside one fresh serve process; the per-arm figure is the median of the four process medians. Run order **`ABBAABBA`** (A = C5, B = C6), one serve process at a time.

| arm | process medians (tok/s) | **arm median** | vs C5 |
|---|---|---:|---:|
| C5 (candidate 5, kept) | 20.7, 20.7, 20.5, 20.4 | **20.6** | — |
| C6 (candidate 6) | 19.7, 20.3, 20.5, 20.3 | **20.3** | **−1.5 % median** (−1.8 % mean) |

Mean ± sample sd / CV (N=4 process medians): C5 **20.575 ± 0.15, CV 0.7 %** (range 20.4–20.7); C6 **20.20 ± 0.35, CV 1.7 %** (range 19.7–20.5). **Separation:** C6's best process (20.5) sits **below** C5's median (20.6), and C6's worst (19.7) is the session low — but the arms' ranges **touch at 20.5** (C6 max = one C5 process), so this is a small loss read on a single balanced block, **not** a process-level separation. Sample arm medians `−1.5 %` = 20.3 / 20.6 − 1.

**Latency (warm rows 2–4, per-arm medians):** wall C5 **6.809 s** vs C6 **6.897 s** (C6 **+1.3 % slower**, the same direction as the decode delta); TTFT C5 **0.639 s** vs C6 **0.637 s** — parity. **No PERF claim is drawn from component timings**; none were inspected beyond these harness fields.

## What was tested (C6 vs C5)

C6 is an **uncommitted worktree delta on `12daf4fe99017a88242469f60853b860609c20ab`** (`perf/passback-opt`), built **on top of C5**. It is a **single production change** in `crates/hipfire-dispatch/src/offload_split.rs` plus its doc-comment/test material and a design-doc/CHANGELOG edit — **identified by its frozen binary md5s and by [`C6-source.patch`](data-2026-10-16-passback-frozen-sampling-experiment/) (143,498 B, sha256 `ec155f6b…e5919`)**. Relative to C5's frozen source it adds **+139 lines in `offload_split.rs`** (411/269 vs C5's 272/269) and touches `docs/plans/partial-gpu-offload-design.md` (69/55) and `CHANGELOG.md` (+1 entry). The mechanism, read from that patch:

1. **Two-density sampling from the convergence latch.** C5's `ShapeState::take_sample` advances a per-shape LCG and samples when its top two bits are zero (`sample_ctr >> 30 == 0`) — about one in four split steps. C6 keeps the same LCG and the same advance, but reads the shape's `frozen` latch to pick the cut-off: top two bits (1/4) while the shape's share is still being scheduled, top four bits (`>> 28`, 1/16) once the shape has latched. Because the density is a *read* of the latch and nothing is reset at a latch/unlatch, the next step after a shape unlatches is drawn at the quarter density again — a recovery invariant, not bookkeeping.
2. **Only a sampled step pays the join.** A split step's CPU-rows H2D is *completed* (the blocking join, the point the two arms are measured) only on a sampled step; on an unsampled step it is merely enqueued and drained by the next required D2H completion. C6 therefore cuts the fraction of split steps that wait on the join from 1/4 to 1/16 once a shape has latched.
3. **The stated cost is slower frozen tracking, not faster reaction.** The C6 doc-comment and its new gated unit test state the trade honestly: `APPLY_EVERY` × `REOPEN_CONFIRM` = 12 observations that a re-open needs span **~192 of this shape's split steps instead of ~48** — i.e. **"4× fewer frozen observations"**, ~4× slower tracking of a host-load move, in exchange for frozen join-waits falling 1/4 → 1/16. This is a **statistical step count, explicitly not a wall-clock or hard latency bound and explicitly not a fast-response claim**.
4. **New gated test material.** `gated_plant` (a ±1 µs jitter plant), `simulate_gated` (the real `sample_due`/`observe` gate across a latch) and the `a_gated_shape_latches_and_recovers_under_a_persistent_move` test were added; no numerics change.

**Recovery expectation is a design statement, not measured here.** The ~4× slower reaction is asserted by the source comment and covered by the new gated unit test; **no load-move/recovery e2e challenge was run** in this session (the stop rule fired on the null-to-loser headline), so this record makes **no fast-reaction claim and no recovery-latency claim**.

## Method

- **9B / MQ4 / budget 16 / `HIPFIRE_OFFLOAD_EXEC=passback` / `HIPFIRE_OFFLOAD_PASSBACK_SHARE=auto`**, greedy (temp 0.0), thinking off, `max_tokens 128`, `max_seq 4096`, spec off (AR), `kv auto` → K=q8 V=q8 VMM, `HIPFIRE_DPM_WARMUP_SECS=10`. Port 11551, serial hardware.
- **Four requests per fresh serve process**: row 1 is the **throwaway**; rows 2–4 are measured, and **their median is ONE process sample**. The 3 warm rows inside a process are **not independent samples** and are never pooled; per-arm N is the number of **fresh processes**.
- **N=4 processes per arm**, one balanced block, order **`ABBAABBA`**: `A1 A2 A3 A4` (C5), `B1 B2 B3 B4` (C6). **No replication block was run** (the stop rule fired before any reverse-order or second block).
- **Run order is receipt-verified** — both by the driver log's `START`/`END` timestamps and by JSON completion mtimes (UTC−04:00, matching the driver's `date -u` values): A1 22:37:56→22:39:02 → B1 22:39:02→22:40:07 → B2 →22:41:12 → A2 22:41:12→22:42:15 → A3 22:42:15→22:43:19 → B3 22:43:19→22:44:23 → B4 22:44:23→22:45:27 → A4 22:45:27→22:46:30 → `ALL DONE 2026-10-16T22:46:30Z`. Every process `rc=0`, serial, no stray serve.
- **Trace OFF for every row** — the reused wrapper `C5REP-ab_sample.sh` applies `env -u RAYON_NUM_THREADS -u HIPFIRE_CPU_EXEC_TRACE -u HIPFIRE_PROFILE -u HIPFIRE_PROFILE_DECODE -u HIPFIRE_MEMSET_DUMP` and prints `unset check (must be empty): []`, `RAYON check (must be unset): [unset]` in **all 8** meta logs. The meta `trace-line check (expect 0)` prints `1` in all 8 — the load-time banner `cpu exec: 16/16 spilled layers fully covered by the CPU seam…`, never a per-step line (**0 `split:` lines** in every one of the 8 serve logs). The check counts `^split:|^cpu exec:` and the single match is that seam banner, matching the C1–C5 records.
- **Predefined stop rule honored.** The campaign starts the secondary river fixture and a load-recovery diagnostic **only on a C6 win**; the headline read null-to-loser, so both were **not run**.

## Results — 9B primary `ABBAABBA` (per process; row 1 excluded)

| sample | arm | warm rows 2–4 (tok/s) | **process median** | TTFT med | wall med |
|---|---|---|---:|---:|---:|
| A1 | C5 | 20.7, 20.9, 20.7 | **20.7** | 0.639 | 6.762 |
| A2 | C5 | 19.3, 20.7, 20.9 | **20.7** | 0.639 | 6.773 |
| A3 | C5 | 21.2, 19.8, 20.5 | **20.5** | 0.633 | 6.846 |
| A4 | C5 | 20.9, 20.4, 19.2 | **20.4** | 0.649 | 6.858 |
| B1 | C6 | 19.7, 19.6, 19.9 | **19.7** | 0.664 | 7.105 |
| B2 | C6 | 20.2, 20.3, 20.7 | **20.3** | 0.639 | 6.895 |
| B3 | C6 | 20.5, 20.5, 20.2 | **20.5** | 0.634 | 6.831 |
| B4 | C6 | 20.1, 20.3, 21.0 | **20.3** | 0.634 | 6.899 |

C5 arm 20.6, C6 arm 20.3 → **−1.5 % median** (−1.8 % mean). Process ranges touch at 20.5 (C6 max = A3). C6's within-process scatter is visibly larger (B1's three warm rows 19.6–19.9 vs A1's 20.7–20.9) — the wider CV of 1.7 % vs 0.7 % reflects that.

**Cold row 1 (discarded everywhere):** C5 8.9–9.1 tok/s, wall 27.25–28.07 s, TTFT 21.22–21.84 s; C6 8.5–8.9 tok/s, wall 27.57–28.65 s, TTFT 21.47–22.24 s. Never used.

**Warm prefill band (rows 2–4):** C5 583.1–604.5 ms (~96–101 tok/s), C6 580.8–614.2 ms (~96–101 tok/s) — overlapping; no latency claim.

## Session-drift caveat (must be read with the number)

The **absolute** decode level of this session is **~3 % below the C5REP session** recorded in the [`C5`](2026-10-16-passback-sampled-gpu-event-timing-gfx1201.md) checkpoint: C5 here is **20.6** vs the C5REP C5-arm median **21.25**. Absolute tok/s **must not be compared across sessions**. The balanced `ABBAABBA` interleave (C5 and C6 alternating across the same wall-clock window) is what controls for that drift, so the **relative** null-to-loser reading is the only one drawn — and it rests on **one balanced block of four** per arm, not on any cross-session absolute. This is also why this record does **not** reuse C5's `+1.4 %` figure as a reference: C6 was compared against C5 *in this session*, not against C5's own result.

## Output health

All **32** measured rows (8 processes × 4): `finish="length"`, `runaway=true`, `attractor=false`, `empty=false`, `saw_done=true`, `terminal_count=1`, `post_terminal_bytes=0`, `gen=128`, `ctx=59`, `cached=0`, `think_words=0`, `decode_estimated=false`, spec fields (`tau`/`dflash`/`mtp`/`ngram_*`) **null**, `kv_backend="vmm"` — the 128-token cap truncating a valid Python answer. **Zero** `stream_error`, zero `tool_calls`, zero retrieval misses. **All 32 rows byte-identical**: single-row `sha256 = 7690b0d32d444edec66c90d6326ad69b9249ca5affdabda50ef31e6306bfcacc` (`code` 520 chars, `ans_words=70`), `request_md5 = dd93dcc0528f194dbd44994d188b03d8`, `prompt_md5 = 5835c71e471849b4a72e1dc8e39695e7` — the **same decoded text as the C1–C5 rows**. **No correctness delta**: C6's change is on the sampling-density/controller path only, and the greedy argmax was unaffected on this fixture.

**Scope limit (fixture-specific).** Greedy output was byte-identical across arms here despite `passback`'s documented run-to-run nondeterminism; that is an **observed property of this fixture and prompt only**, **not** a claim that `passback` output is reproducible, and `passback share auto` is **not** a golden reference. No token-id-level identity is asserted (the harness records decoded text + `gen` + `request_md5`, not raw greedy ids).

## Fixture identity (measured in this host, verified against the receipts)

| artifact | digest |
|---|---|
| branch / base source | `perf/passback-opt` off **HEAD `12daf4fe99017a88242469f60853b860609c20ab`** (hipfire 0.4.3); `cargo build --release` (**not** `-p`, feature unification differs) |
| C5 binaries (kept arm, A) | `hipfire` md5 `23181d5ba45c532ec6910b0801ea1492` (47,340,216 B); `daemon` md5 `98265d8b809d3b40245f05382b3d9d44` (88,583,272 B) — per-run md5 recorded in all 4 C5 meta logs |
| C6 binaries (B) | `hipfire` md5 `03ba4e18c59b53d4996972fe6a3a760b` (47,340,208 B); `daemon` md5 `32c19e1c1887d34a0ad03c8d33db51ea` (88,583,240 B) — per-run md5 recorded in all 4 C6 meta logs |
| C6 source delta | [`C6-source.patch`](data-2026-10-16-passback-frozen-sampling-experiment/) 143,498 B, **sha256 `ec155f6b1be63cd7294aafdc41003e3948635d48c6026993199feb37c82e5919`** (full worktree delta vs base; C6-relative-to-C5 = +139 lines in `offload_split.rs`, `partial-gpu-offload-design.md`, `CHANGELOG.md`) |
| C5 source delta (kept; frozen) | [`C5-source.patch`](data-2026-10-16-passback-frozen-sampling-experiment/) 120,834 B, **sha256 `5dbcf649ef225ec4ba5adf3aac72405f92bccd2cd90e23c4e29776fd77066ab2`**; [`C5-delta-from-c4.patch`](data-2026-10-16-passback-frozen-sampling-experiment/) 58,379 B, sha256 `2e8b81fea2afef3f1a270eeafee02d27b203f38aec8701ab42c0337903a1fb2c` |
| build provenance | [`C5-source-provenance.json`](data-2026-10-16-passback-frozen-sampling-experiment/) + [`C6-source-provenance.json`](data-2026-10-16-passback-frozen-sampling-experiment/) — `rustc 1.98.1 (48a229cea 2026-09-01)`, LLVM 22.1.8, host `x86_64-unknown-linux-gnu` |
| model | `~/.hipfire/models/qwen3.5-9b.mq4`, md5 `296092bf1e6a45d78c1acf815eb93366`, 5,313,750,016 B, tag `qwen3.5:9b` |
| fixture | `warm3x_gpu_offload_probe.json`, md5 `c4c78f65b100f645cd1ab3ff0fcbf413`, sha256 `1e058b674082be29a6245eaa526c7100792abc6fa81f09e01ed032ffd9f8e60a` (1,042 B); inner prompt `benchmarks/prompts/gpu_offload_probe.txt` md5 `5835c71e471849b4a72e1dc8e39695e7` (215 B), `request_md5 dd93dcc0528f194dbd44994d188b03d8` |
| wrapper | `C5REP-ab_sample.sh` md5 `e3e4c48acc71579a1233bd281e31a76a` (the C5 checkpoint's replication wrapper, **reused unchanged**); `power_snap.sh` md5 `6726f0f70161abb83bc6a3bfdbfe5a03` |
| route envs | `HIPFIRE_DPM_WARMUP_SECS=10 HIPFIRE_OFFLOAD_EXEC=passback HIPFIRE_GPU_LAYER_BUDGET=16 HIPFIRE_OFFLOAD_PASSBACK_SHARE=auto` |

Host: 1× Radeon RX 9070 XT `gfx1201` (16,304 MiB), ROCm/HIP 7.2, Ryzen 7 7800X3D (8c/16t), 28 GB RAM. Per-run load banner: 16 of 32 layers host-placed, `passback: 16/16 spilled layers splittable; share auto; min step 2 MiB`, `cpu exec: 16/16 spilled layers fully covered by the CPU seam`, `hipGraph capture disabled (host-executed steps present)`; `KV: mode=auto → K=q8 V=q8`; cache warming 10.00 s.

**`RAYON_NUM_THREADS`:** the wrapper applies `env -u RAYON_NUM_THREADS` to the serve process and prints `RAYON check (must be unset): [unset]`, reading the real variable — receipt-verified `[unset]` in **all 8** meta logs. **Profiler variables** (`HIPFIRE_CPU_EXEC_TRACE`, `HIPFIRE_PROFILE`, `HIPFIRE_PROFILE_DECODE`, `HIPFIRE_MEMSET_DUMP`) are `env -u`'d and `unset check` is `[]` in all 8.

## Power / clock / temperature (µW snapshots only — NOT steady-state)

sysfs (card1) `power1_average_uW` pre → post via `power_snap.sh`, every meta log. **`rocm-smi --showpower` returned empty in all meta logs** (`=== Power Consumption ===` has no lines). Every run idles at `perf_level=auto`, `power1_cap_uW = 374,000,000` (374 W), `throttle_log` = the "thermal throttling logging enabled" banner only.

| run | power pre → post (µW) | run | power pre → post (µW) |
|---|---|---|---|
| C6CMP-A1 | 48 → 53 | C6CMP-B1 | 49 → 134 |
| C6CMP-A2 | 49 → 138 | C6CMP-B2 | 49 → 49 |
| C6CMP-A3 | 49 → 82 | C6CMP-B3 | 49 → 123 |
| C6CMP-A4 | 49 → 127 | C6CMP-B4 | 49 → 137 |

These are **point snapshots at differing DPM states**, **not comparable across arms** (B2's post equals its pre — a snapshot caught between DPM steps); **no power or energy claim is drawn**.

## Link to the candidate ledgers

- **C5** ([`2026-10-16-passback-sampled-gpu-event-timing-gfx1201.md`](2026-10-16-passback-sampled-gpu-event-timing-gfx1201.md)) is the **baseline and the kept candidate here**: it hands the controller both arms' own concurrent durations (CPU multiply interval + stream-marker GPU arm elapsed) and removes the old join-floor censoring. **C6 changes only the sampling density** on top of C5 — it alters *how often* a step measures, not *what* a sampled step measures.
- **C4** ([`2026-10-16-passback-asynchronous-split-returns-gfx1201.md`](2026-10-16-passback-asynchronous-split-returns-gfx1201.md)) — pinned async split returns, C5's functional prerequisite; **C3** was `rejected` (−1.5 %) and its pinned storage carries forward; **C1** (interval) and **C2** (shape-key) were fully reverted.

## Status

**C6 is rejected.** Its uncommitted worktree delta **has been reverted by the code owner**, restoring the frozen C5 source; **C5 is retained**. The C6 and C5 frozen source patches, their `source-provenance.json`, and the C5 C4-relative delta are archived below as text (**[`C6-source.patch`](data-2026-10-16-passback-frozen-sampling-experiment/), [`C5-source.patch`](data-2026-10-16-passback-frozen-sampling-experiment/), [`C5-delta-from-c4.patch`](data-2026-10-16-passback-frozen-sampling-experiment/)**); **no executables are committed** — the frozen binaries stay in `.codeinsight+research/passback-opt/{candidate5-bin,candidate6-bin}/`. Correctness was **not re-measured here** (the campaign prohibits regression reruns); the build provenance records `dispatch_lib 411 passed, 1 ignored` and both GPU oracles passing for C6 ([`C6-source-provenance.json`](data-2026-10-16-passback-frozen-sampling-experiment/) points at `artifact://819`/`artifact://821`), and `410 passed, 1 ignored` for C5. **27B: not validated** — no 27B sample exists here, and this record makes **no promotion, current-baseline, or overall-performance claim**.

## What was NOT measured / provenance gaps (recorded honestly)

- **No replication block** — unlike the C5 record (which had a reverse-order `BAABBAAB` block), this is **one `ABBAABBA` block of N=4/arm**. The −1.5 % rests on a single balanced block; C6's own CV (1.7 %) is wider than C5's (0.7 %) and the arms' ranges touch, so the loss is **small and single-block**, not a resolved magnitude.
- **Cause is NOT established.** C6's only production change is the sampling-density cut-off (`>> 28` vs `>> 30` when `frozen`), which removes 3/4 of the frozen join-waits — yet the e2e result is a small *loss*. The receipts hold **no trace/counter/share data** (trace OFF) and no component timing, so they **cannot attribute** the −1.5 % to the reduced join-wait, to the wider sampling scatter, or to session noise. Recorded as **whole observed effect only**; no mechanism is asserted for the loss.
- **No recovery/load-move challenge.** The ~4×-slower frozen tracking is a **design statement from the source doc-comment and its gated unit test**, not measured end-to-end here; no load-recovery, step-change, or river fixture was run (stop rule).
- **No 27B, no other model/quant/GPU**, no share/budget/`min_step` sweep, no profiled headline runs, no CPU/PCIe/share0 controls.
- **Absolute levels drift ~3 % below the C5REP session** (see the caveat above); no cross-session absolute is quoted.
- **Power/clock:** `rocm-smi --showpower` unavailable; sysfs point snapshots at differing DPM states; no power or energy claim.
- **No token-id-level identity** — text identity at decoded-text + `gen` + `request_md5` level only, fixture-specific.

## Reproduction

```
bash C6CMP-run.sh
# driver over the reused C5REP-ab_sample.sh wrapper, one serve process at a time:
#   C5REP-ab_sample.sh <cli> <daemon> passback auto 16 ~/.hipfire/models/qwen3.5-9b.mq4 qwen3.5:9b 11551 /tmp/C6CMP-home-<X> .codeinsight+research/passback-opt/raw/C6CMP-<X>
```
A = C5 (`23181d5b`/`98265d8b`), B = C6 (`03ba4e18`/`32c19e1c`), order `A1 B1 B2 A2 A3 B3 B4 A4`. `OUT` has no extension, so the wrapper writes `C6CMP-<X>.json` / `.serve.log` / `.meta.log` (the C5REP wrapper only appends `.json` to its `--out`, hence the C6 receipts are `*.json` not `*.json.json`).

## Raw evidence

**35 files** in [`data-2026-10-16-passback-frozen-sampling-experiment/`](data-2026-10-16-passback-frozen-sampling-experiment/) (34 receipts + `SHA256SUMS`), every receipt `cmp`-verified byte-identical to its source under `.codeinsight+research/passback-opt/`:

- `C6CMP-{A1..A4}-c5.{json,serve.log,meta.log}`, `C6CMP-{B1..B4}-c6.{json,serve.log,meta.log}` — **24 receipts**: full harness JSON (all 4 rows, complete `content`/`assistant_content`), full serve logs (untrimmed), and identity/env/power/temperature meta logs.
- `C6CMP-driver.log`, `C6CMP-run.sh` — the exact driver (run order, `START`/`END` timestamps, `ALL DONE 22:46:30Z`) and its script.
- `C5REP-ab_sample.sh`, `power_snap.sh`, `warm3x_gpu_offload_probe.json` — the **reused** wrapper, the power-snapshot helper, and the fixture (the wrapper's `RAYON`/`unset` command proof and the measured fixture).
- `C6-source.patch`, `C6-source-provenance.json` — the rejected C6 frozen source delta and its build/toolchain provenance.
- `C5-source.patch`, `C5-delta-from-c4.patch`, `C5-source-provenance.json` — the **kept** C5 frozen source delta, its C4-relative delta, and its provenance (**text only; no executables**).
- `SHA256SUMS` — integrity digests of all 34 receipts.
