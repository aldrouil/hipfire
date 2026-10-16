# Passback timing-interval experiment (candidate 1, rejected) — 2026-10-16

**Lifecycle:** `historical`

**Disposition:** `rejected`. A targeted, fixture-bound A/B of one scheduler
candidate on the 9B primary. **No throughput win** (+0.5% on the overall
process median, inside the 7–8% process-to-process spread). The candidate code
change is being **reverted**; this record is the durable account of the attempt
and its raw receipts. It is **not** an admission, **not** a
[`docs/BENCHMARKS.md`](../BENCHMARKS.md) claim, and **not** transferable across
host, model, quant, GPU, prompt, route, or method.

This is a **new** checkpoint. No existing checkpoint is modified. The rejected
candidate retains its rejection (append-only; see
[`README.md`](README.md) §Preservation).

---

## What was tested

Candidate 1 altered one line of the pass-back split scheduler's observation
input. In `crates/hipfire-dispatch/src/offload_split.rs`, `run_with` feeds the
learned CPU-side interval into `observe(...)`:

| build | CPU interval fed to scheduler |
|---|---|
| baseline (A) | `arm.d2h_ns + gemv_ns` (blocking D2H **plus** the CPU multiply) |
| candidate 1 (B) | `gemv_ns` (the CPU multiply alone) |

Rationale (from the trace attribution in the research scratchpad, **not** a
finding of this record): the prepare's blocking D2H completes before the GPU
arm is launched, so the interval the GPU arm actually overlaps is the CPU
multiply, not D2H + multiply. The baseline therefore over-states the CPU
interval. Candidate 1 removes the `arm.d2h_ns` term. `observe` and every
scheduling policy downstream are unchanged; only the scalar passed in changes.

This experiment isolates **only** that one-line input change. It does **not**
implement the separately-hypothesised `(dtype, k)` key split, per-shape state,
or the censored-join estimator — those remain open and are out of scope here.

---

## Method

Single-prompt, single-share, fresh-process, **unprofiled** A/B. All 9B, budget
16 (16 of 32 layers host-placed), `HIPFIRE_OFFLOAD_EXEC=passback`, share `auto`.

- **Cold row 1 is discarded.** Each run is a fresh serve process exercised by a
  fixture of 4 byte-identical prose rows on the real production harness path.
  Row 1 carries the one-time costs of the fresh process (first-forward graph/JIT
  settling + DPM state re-pin), not steady state: row-1 prefill is ~13.0–13.4 s
  versus ~0.58–0.63 s on rows 2–4, and row-1 decode is 8.7–8.9 tok/s versus
  18.2–20.1 on rows 2–4.
- **Warm rows 2–4 median** is the per-process statistic (N=3 rows → median).
- **Fresh process per sample, N=3 samples per arm** (`A1..A3`, `B1..B3`). Each
  process is independent; its kernel cache is per-home and pre-populated by row 1.
- **Trace is OFF.** `ab_sample.sh` unsets `HIPFIRE_CPU_EXEC_TRACE` /
  `HIPFIRE_PROFILE` / `HIPFIRE_PROFILE_DECODE` / `HIPFIRE_MEMSET_DUMP` and
  records an empty unset-check (`unset check (must be empty): []`, present in the
  A2/A3/B1/B2/B3 meta logs). The meta log's `trace-line check (expect 0)` prints
  `1`; that single match is the load-time config banner line
  `cpu exec: 16/16 spilled layers fully covered by the CPU seam; …`, not a
  per-step trace line — the arms are genuinely unprofiled.

### Run order (ABBA intent; observed ABABBA)

Order was determined from filesystem completion mtimes of the six archived JSON
payloads:

| # | arm | label | json completed (UTC−04:00) |
|---:|---|---|---|
| 1 | A (baseline) | A1 | 14:40:32 |
| 2 | B (candidate 1) | B1 | 14:42:18 |
| 3 | A | A2 | 14:43:48 |
| 4 | B | B2 | 14:45:45 |
| 5 | B | B3 | 14:48:27 |
| 6 | A | A3 | 14:50:34 |

The arms alternate A/B to cancel monotonic drift, but the **observed** sequence
is `A B A B B A` (ABABBA), **not** a strict A/B/B/A ABBA block. Disclosed as a
method caveat: the last two samples are B3 then A3, so the final A sample sits
at the tail of the session where both arms are slowest (see the late-session dip
below).

---

## Results (9B primary)

Decode `tok/s`, straight from the archived harness JSON (row 1 shown for the
cold-start disclosure, then excluded from the median):

| sample | arm | row 1 (discarded) | warm rows 2–4 | min | **median** | max |
|---|---|---:|---|---:|---:|---:|
| A1 | baseline | 8.8 | 19.6, 19.6, 19.5 | 19.5 | **19.6** | 19.6 |
| A2 | baseline | 8.8 | 19.8, 19.8, 19.8 | 19.8 | **19.8** | 19.8 |
| A3 | baseline | 8.7 | 18.9, 18.5, 18.4 | 18.4 | **18.5** | 18.9 |
| B1 | candidate 1 | 8.9 | 20.1, 19.9, 19.7 | 19.7 | **19.9** | 20.1 |
| B2 | candidate 1 | 8.8 | 19.7, 19.7, 19.2 | 19.2 | **19.7** | 19.7 |
| B3 | candidate 1 | 8.7 | 18.2, 18.3, 18.3 | 18.2 | **18.3** | 18.3 |

Per-arm distribution over the 3 process medians:

| arm | process medians | min / median / max | mean | sd (population / sample) | range |
|---|---|---|---:|---|---:|
| baseline (A) | 19.6, 19.8, 18.5 | 18.5 / **19.6** / 19.8 | 19.30 | 0.571 / 0.700 | 1.30 |
| candidate 1 (B) | 19.9, 19.7, 18.3 | 18.3 / **19.7** / 19.9 | 19.30 | 0.712 / 0.872 | 1.60 |

**Overall process median: baseline 19.6 → candidate 19.7 tok/s = +0.10 tok/s
(+0.51%).** The two arms have **identical means (19.30)**. The between-process
range is 1.3–1.6 tok/s (**≈7–8% of the level**), so the +0.1 tok/s median shift
is a fraction of a single process's own scatter. **Verdict: no win.**

Both arms show the same monotonic late-session dip (A's slowest sample is the
last one, A3 = 18.5; B's slowest is the last one, B3 = 18.3), i.e. shared
session drift, not an arm effect.

### Latency / prefill (from the same receipts)

Warm rows (2–4), per sample, range:

| arm | prefill (rows 2–4) | prefill tok/s | TTFT (rows 2–4) | wall (rows 2–4) |
|---|---|---|---|---|
| A1 | 593.0 / 597.6 / 598.9 ms | 98.5–99.5 | 0.644–0.647 s | 7.109–7.161 s |
| A2 | 594.5 / 597.4 / 597.5 ms | 98.7–99.2 | 0.645–0.651 s | 7.059 s |
| A3 | 604.4 / 607.0 / 626.6 ms | 94.2–97.6 | 0.661–0.677 s | 7.415–7.562 s |
| B1 | 591.2 / 597.4 / 599.0 ms | 98.5–99.8 | 0.641–0.648 s | 6.980–7.096 s |
| B2 | 589.7 / 590.8 / 598.3 ms | 98.6–100.0 | 0.637–0.647 s | 7.089–7.265 s |
| B3 | 577.8 / 586.3 / 592.1 ms | 99.7–102.1 | 0.627–0.642 s | 7.593–7.623 s |

Cold row 1 (discarded): prefill 12989.0–13401.1 ms (4.4–4.5 tok/s), TTFT
20.976–21.550 s, wall 27.619–28.014 s.

No latency win/loss: the prefill/TTFT/wall bands overlap completely between
arms; the visible spread tracks the same session drift as the decode column.

### Output health

Every one of the 24 rows (6 samples × 4 rows) completed with:

- `finish = "length"`, `runaway = true` (harness definition: `runaway := (finish == "length")`),
- `attractor = false`, `empty = false`, `saw_done = true`,
- `terminal_count = 1`, `terminal_reasons = ["length"]`, `post_terminal_bytes = 0`,
- `think_words = 0`, `ans_words = 70`.

This is the 128-token `max-tokens` cap truncating a valid, coherent Python
answer mid-function — **"length cap", not a degenerate/attractor loop**. All 24
rows produced **byte-identical** assistant content
(`sha256 e572f7d34c43dc877c0875b0396cc7d1bc0950f96666cddcf7a35336aaac7abd`) and
identical `request_md5 dd93dcc0528f194dbd44994d188b03d8` — expected here, since
pass-back is byte-invariant on this fixture at 128 tokens; it is **not**
evidence of determinism in general (pass-back's row split is per-process
non-deterministic by design, per the serve log banner).

---

## Fixture identity (measured)

| artifact | digest |
|---|---|
| branch / base source | `perf/passback-opt` off `12daf4fe99017a88242469f60853b860609c20ab` (hipfire 0.4.3) |
| baseline source build | clean tree at `12daf4fe…`; `cargo build --release` |
| candidate 1 source build | same tree **plus the one-line `observe`-input change**; **uncommitted worktree delta — no source commit hash exists** (deliberately not invented). Behavior-neutral `rustfmt` normalisation is also present in the touched files and does not affect the semantic delta. |
| `hipfire` binary A (md5) | `c9feb2095ffe043c7e93477a7ec02a9a` (47,336,152 B) |
| `daemon` binary A (md5) | `c1da18f268129005c6e827ea85ceff49` (88,582,088 B) |
| `hipfire` binary B (md5) | `e3f6cc8d24f05a3f5fdcae0d263f4405` (47,336,120 B) |
| `daemon` binary B (md5) | `73de751f05c929e5115e6ca3eefd8dd7` (88,582,064 B) |
| frozen binary copies | `.codeinsight+research/passback-opt/{baseline-bin/hipfire-12daf4fe,baseline-bin/daemon-12daf4fe,candidate1-bin/hipfire,candidate1-bin/daemon}` |
| model | `~/.hipfire/models/qwen3.5-9b.mq4`, md5 `296092bf1e6a45d78c1acf815eb93366`, 5,313,750,016 B, tag `qwen3.5:9b` |
| prompt fixture | `warm3x_gpu_offload_probe.json` (4 identical prose rows), md5 `c4c78f65b100f645cd1ab3ff0fcbf413`, 1,042 B |
| inner prompt text | md5 `5835c71e471849b4a72e1dc8e39695e7` (215 B; = `benchmarks/prompts/gpu_offload_probe.txt`) |
| sampling | greedy (temp 0.0), thinking off (sentinel cap 1), `max_tokens 128`, `max_seq 4096`, spec off (AR) |
| KV | `auto` → effective K=q8 V=q8 VMM (`kv_backend = vmm`), 8/32 layers carry KV; DeltaNet state Q8 |
| route envs | `HIPFIRE_DPM_WARMUP_SECS=10 HIPFIRE_OFFLOAD_EXEC=passback HIPFIRE_GPU_LAYER_BUDGET=16 HIPFIRE_OFFLOAD_PASSBACK_SHARE=auto` |
| harness | `fixtures/ab_sample.sh` (one fresh-process sample; row 1 = throwaway, rows 2–4 = measured median) over `scripts/serve_harness.py --mode battery` |
| homes / port | `/tmp/ab/home-{A1,A2,A3,B1,B2,B3}`, port 11541, per-home `.hipfire_kernels/gfx1201` |

Host: 1× Radeon RX 9070 XT `gfx1201` (16304 MiB), ROCm/HIP 7.2, Ryzen 7 7800X3D
(8c/16t), 28 GB RAM. Baseline identical to
`.codeinsight+research/passback-opt/IDENTITY.md`. Load banner per run:
16 of 32 layers host-placed, device weights 3299 MiB of 14669–14697 MiB,
pinned host 1753 MiB, `passback: 16/16 spilled layers splittable; share auto;
min step 2 MiB`.

---

## Methodological correction: the earlier "kept" numbers are excluded

The pass-back-opt research scratchpad's first "kept" series (`k-passback16-*`,
`k-cpu16-*`, `k-pcie16-*`, `k-pin0*-*`, and the untracked `RESULTS.md` /
`SAMPLING.md` summaries that quote them, e.g. pass-back auto ≈ 8.8 tok/s, cpu
≈ 8.1, pcie ≈ 7.3) ran with **`HIPFIRE_CPU_EXEC_TRACE=1` exported** (see
`/tmp/pb-base/kept_sample.sh`), and used a **single-prompt** `--prompt-file`
fixture with a **separate warm throwaway process** rather than this record's
4-row in-process fixture. Active trace output is present in those receipts
(97 and 98 `split:`/`cpu exec:` lines in `k-passback16-run1` and
`k-cpu16-run1`).

Those runs are therefore **attribution-only and excluded from every headline
in this record.** The trace prints sit on the profiled step's critical path, so
their wall-clock is not a clean decode rate; and their fixture/method differs
from the A/B here. The ~2× level gap between that series (~8.8 tok/s) and this
clean A/B (~19.6 tok/s) is consistent with trace perturbation plus the fixture
delta, but **no measurement isolating the two causes was run** — treat the gap
as unexplained in magnitude. No ratio derived from the trace-enabled series is
quoted in this record.

(The trace-enabled series remains valid *as attribution*: it is the source of
the `d2h`/`gemv`/`join` interval estimates that motivated candidate 1. Those
estimates are inputs to the hypothesis, not a result of this experiment.)

---

## Correctness evidence (not a performance claim)

The candidate did not alter arithmetic, and correctness was established
separately from this A/B:

- **Master dispatch suite: all 412 tests pass.**
- **GPU/split hardware oracle: 17/17 cases bit-exact** on `gfx1201`
  (`crates/hipfire-arch-qwen35/tests/gpu_gemv_parity.rs`).
  One of those 17 originally failed the oracle because an MFP4 MoE tensor
  (~1.13 MiB) is *below* the split gate's `MIN_SPLIT_BYTES` (2 MiB), so the
  pass-back declined it as a whole-CPU step. The test was repaired by tiling
  **real fixture rows** (same stride, real bytes) up to the floor via
  `extend_from_within`, with grown cases explicitly labelled `tiled-m=<m>`;
  all existing assertions and coverage were retained. 17/17 pass after the
  repair. This test repair is a separate correctness fix, bundled in the same
  worktree, and is **not** part of the perf experiment or its verdict.

No generation-quality regression was observed: all rows healthy and
byte-identical (above).

---

## 27B status

**The 27B real target was not validated in this experiment.** No 27B sample
exists in this record. Nothing here supports — or is meant to support — any
27B improvement claim. Per campaign policy, 27B validation is deferred to the
end of the campaign, not run per pass.

---

## What was NOT measured

- **Power.** The meta logs' `=== Power Consumption ===` sections are **empty**:
  `rocm-smi --showpower` returned no power values on this host, pre or post.
  Power/temperature-matched arms are therefore **not available** and no power
  claim is made. Temperature (edge/junction/memory) *was* captured and is
  archived.
- **A1's identity block.** The A1 meta log is truncated (it begins at
  `=== power post ===`): its `env allowlist` / CLI+daemon md5 / unset-check /
  `power pre` block is **absent from the receipt**. A1's binary identity is
  recoverable only from the shared baseline identity (binaries frozen at 13:36,
  md5 in the table above). This is a genuine gap in the raw evidence, preserved
  as-is and not back-filled.
- **No separate per-run "identity file".** Per-run identity lives inside each
  `*.meta.log` (`env allowlist`, `CLI md5`, `DAEMON md5`, `unset check`) plus
  the shared `IDENTITY.md`; no standalone identity artifact was produced.
- No re-measurement, no `min_step`/share sweep, no key-split variant, no
  event-based GPU-arm timing, no 27B, no other model/quant/GPU, no
  cross-context cells, no `HIPFIRE_PROFILE` runs.

## Reproduction

```bash
# one fresh-process sample (row 1 = throwaway, rows 2-4 median); trace OFF inside
bash ab_sample.sh \
  <hipfire-bin> <daemon-bin> passback auto 16 \
  ~/.hipfire/models/qwen3.5-9b.mq4 qwen3.5:9b 11541 \
  /tmp/ab/home-<X> /tmp/ab/<X>-passback
# arms A = baseline binaries (md5 c9feb209 / c1da18f2)
# arms B = candidate-1 binaries (md5 e3f6cc8d / 73de751f)
```

## Raw evidence

Complete raw receipts are committed next to this record at
`data-2026-10-16-passback-timing-interval-experiment/`, preserving every
original row (no aggregation, no trimming):

- `AB-{A1,A2,A3}-baseline.json`, `AB-{B1,B2,B3}-candidate1.json` — full
  harness JSON (all 4 rows each, complete `content`/`assistant_content`).
- `AB-*-baseline.serve.log`, `AB-*-candidate1.serve.log` — full serve logs
  (load banner, KV, routing, per-step cache lines).
- `AB-*-baseline.meta.log`, `AB-*-candidate1.meta.log` — identity block
  (env allowlist, CLI/daemon md5, unset check), power/temperature, trace-line
  check.
- `ab_sample.sh`, `warm3x_gpu_offload_probe.json` — the exact sampler and
  fixture used.
- `SHA256SUMS` — integrity digests of every file above.

These are copies of `/tmp/ab/{A1..B3}-passback.{json,meta.log,serve.log}`
(md5-verified byte-identical for the JSON payloads before copy).
