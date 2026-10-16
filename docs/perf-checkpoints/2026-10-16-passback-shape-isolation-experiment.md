# Passback shape-key isolation experiment (candidate 2, rejected) — 2026-10-16

**Lifecycle:** `historical`

**Disposition:** `rejected`. A targeted, fixture-bound A/B of one scheduler
candidate on the 9B primary. The candidate showed a **consistent decode
regression — every baseline process above every candidate process, −5.6% on the
overall process median** (baseline 19.8 vs candidate 18.7 tok/s). The candidate
code change is being **reverted**. This record is the durable, append-only
account of the attempt and its raw receipts. It is **not** an admission, **not**
a [`docs/BENCHMARKS.md`](../BENCHMARKS.md) claim, and **not** transferable
across host, model, quant, GPU, prompt, route, or method.

This is a **new** checkpoint. No existing checkpoint is modified. The rejected
candidate retains its rejection (append-only; see
[`README.md`](README.md) §Preservation).

**Relationship to candidate 1.** This record is a **separate, independently
measured** A/B of a **different** candidate. Candidate 1 (the one-line
`observe`-input change, ~parity) and its receipts are recorded in
[`2026-10-16-passback-timing-interval-experiment.md`](2026-10-16-passback-timing-interval-experiment.md);
that numeric experiment is **not repeated here**. The two candidates were not
measured against each other in this session; each was measured against the same
frozen 12daf4fe baseline binaries. See the shared baseline identity in
`.codeinsight+research/passback-opt/IDENTITY.md`.

**Cause is NOT established.** This record documents the observed whole system
effect (`−5.6%` overall process median) and **does not attribute it**. Two
candidate mechanisms were floated in campaign discussion — per-shape
convergence (the exact `(dtype, m, k, rotate_input, residual)` key fragmenting
the learned state so each shape converges more slowly) versus added per-step
metadata/hash overhead from the enlarged key — and **neither is established by
this experiment**. No measurement, trace, or counter in these receipts
discriminates between them or against any third cause. The raw decode-level
effect is recorded; the mechanism is left open.

---

## What was tested

Candidate 2 changed the pass-back split scheduler's **keying and eligibility
order**, per the campaign ledger (the source change itself is **not** a result
of this record):

- the learned state is keyed on an **exact shape key**
  `(dtype, m, k, rotate_input, residual)` instead of conflating those
  dimensions under one key, and
- the **same-eligibility check runs before the probes** rather than after.

Candidate 2 does **not** carry candidate 1's one-line interval fix. No source
commit hash exists for candidate 2: it is an **uncommitted worktree delta** on
top of `12daf4fe99017a88242469f60853b860609c20ab`, so the identity of the
measured change is the frozen binary md5 below, not a commit (no commit hash is
invented). Behavior-neutral `rustfmt` normalisation may also be present in the
touched files; it does not change the semantic delta.

This experiment isolates **only** that candidate-2 worktree delta. It does not
test direct GPU-arm timing, the censored-join estimator, or any other hypothesis
from the campaign backlog — those remain open and out of scope here.

---

## Method

Single-prompt-family, single-share, fresh-process, **unprofiled** A/B. All 9B,
budget 16 (16 of 32 layers host-placed), `HIPFIRE_OFFLOAD_EXEC=passback`,
share `auto`.

- **Cold row 1 is discarded.** Each run is a fresh serve process exercised by a
  fixture of 4 byte-identical prose rows on the real production harness path.
  Row 1 carries the one-time costs of the fresh process (first-forward
  graph/JIT settling + DPM state re-pin), not steady state: row-1 prefill is
  ~13.1–13.4 s versus ~0.57–0.61 s on rows 2–4, and row-1 decode is 8.2–8.9
  tok/s versus 17.5–20.2 on rows 2–4.
- **Warm rows 2–4 median** is the per-process statistic (N=3 rows → median).
- **Fresh process per sample, N=3 samples per arm** (`A1..A3` baseline,
  `B1..B3` candidate 2). Each process is independent; its kernel cache is
  per-home and pre-populated by row 1.
- **Trace is OFF.** `ab_sample.sh` unsets `HIPFIRE_CPU_EXEC_TRACE` /
  `HIPFIRE_PROFILE` / `HIPFIRE_PROFILE_DECODE` / `HIPFIRE_MEMSET_DUMP` and
  records an empty unset-check (`unset check (must be empty): []`) in **all six**
  meta logs. The meta log's `trace-line check (expect 0)` prints `1`; that
  single match is the load-time config banner line
  `cpu exec: 16/16 spilled layers fully covered by the CPU seam; …`, not a
  per-step trace line — the arms are genuinely unprofiled.

### Run order (observed ABBAAB)

Order was determined from filesystem completion mtimes of the six archived JSON
payloads:

| # | arm | label | json completed (UTC−04:00) |
|---:|---|---|---|
| 1 | A (baseline) | A1 | 15:04:14 |
| 2 | B (candidate 2) | B1 | 15:06:41 |
| 3 | B | B2 | 15:08:50 |
| 4 | A | A2 | 15:10:55 |
| 5 | A | A3 | 15:13:11 |
| 6 | B | B3 | 15:15:18 |

The observed sequence is `A B B A A B` (**ABBAAB**): an ABBA block followed by
an AB block. It is **not** a strict A/B/A/B alternation and **not** a complete
ABBAABBA; the second half is one AB pair, so B samples (2 of 3) and A samples
(2 of 3) occupy different positions in the session. Disclosed as a method
caveat: A3 (the best baseline sample, 20.2) sits late in the session and B1 (the
worst candidate sample, 17.6) sits early, so the arms' best/worst samples are
not position-matched.

---

## Results (9B primary)

Decode `tok/s`, straight from the archived harness JSON (row 1 shown for the
cold-start disclosure, then excluded from the median):

| sample | arm | row 1 (discarded) | warm rows 2–4 | min | **median** | max |
|---|---|---:|---|---:|---:|---:|
| A1 | baseline | 8.8 | 19.6, 19.4, 19.3 | 19.3 | **19.4** | 19.6 |
| A2 | baseline | 8.8 | 19.9, 19.8, 19.7 | 19.7 | **19.8** | 19.9 |
| A3 | baseline | 8.9 | 20.2, 20.2, 20.0 | 20.0 | **20.2** | 20.2 |
| B1 | candidate 2 | 8.2 | 17.5, 18.0, 17.6 | 17.5 | **17.6** | 18.0 |
| B2 | candidate 2 | 8.6 | 18.8, 18.7, 18.7 | 18.7 | **18.7** | 18.8 |
| B3 | candidate 2 | 8.6 | 18.7, 18.7, 18.6 | 18.6 | **18.7** | 18.7 |

Per-arm distribution over the 3 process medians:

| arm | process medians | min / median / max | mean | pop. var / sd | sample var / sd | range |
|---|---|---|---:|---|---|---|---:|
| baseline (A) | 19.4, 19.8, 20.2 | 19.4 / **19.8** / 20.2 | 19.800 | 0.1067 / 0.3266 | 0.1600 / 0.4000 | 0.8 |
| candidate 2 (B) | 17.6, 18.7, 18.7 | 17.6 / **18.7** / 18.7 | 18.333 | 0.2689 / 0.5185 | 0.4033 / 0.6351 | 1.1 |

**Overall process median: baseline 19.8 → candidate 18.7 tok/s = −1.1 tok/s
(−5.6%).** The separation is clean at the process level: the candidate's
**highest** process median (18.7) is below the baseline's **lowest** (19.4) —
**all 3 candidate processes are below all 3 baseline processes**. Unlike
candidate 1 (identical means, shift a fraction of one process's own scatter),
this candidate's arm means differ by 1.47 tok/s (~3.7× the baseline sample sd).
The candidate arm is also more dispersed (sample sd 0.635 vs baseline 0.400),
driven almost entirely by the single low B1 sample (17.6); B2 and B3 are
tightly clustered (18.7, 18.7). No position-matched interpretation is possible
at N=3, so the −5.6% figure is reported as the observed whole-effect and not
further decomposed.

### Latency / prefill (from the same receipts)

Warm rows (2–4), per sample, range:

| arm | prefill (rows 2–4) | prefill tok/s | TTFT (rows 2–4) | wall (rows 2–4) |
|---|---|---|---|---|
| A1 | 591.6 / 590.0 / 593.7 ms | 99.4–100.0 | 0.640–0.643 s | 7.115–7.209 s |
| A2 | 582.9 / 586.7 / 569.7 ms | 100.6–103.6 | 0.616–0.634 s | 7.007–7.067 s |
| A3 | 585.5 / 579.3 / 573.0 ms | 100.8–103.0 | 0.622–0.632 s | 6.912–6.967 s |
| B1 | 609.2 / 588.1 / 586.5 ms | 96.8–100.6 | 0.638–0.673 s | 7.702–7.925 s |
| B2 | 579.9 / 588.1 / 588.0 ms | 100.3–101.7 | 0.630–0.645 s | 7.407–7.448 s |
| B3 | 574.8 / 584.5 / 586.9 ms | 100.5–102.6 | 0.626–0.641 s | 7.431–7.475 s |

Cold row 1 (discarded): prefill 13089.7–13425.8 ms (4.4–4.5 tok/s), TTFT
21.129–21.774 s, wall 27.592–28.961 s.

Prefill/TTFT bands overlap between arms. The **wall** band shows the same
direction as decode: candidate warm walls are 7.407–7.925 s versus baseline
6.912–7.209 s — the candidate takes visibly longer wall-clock for the same 128
generated tokens. Per-row decode and wall are consistent with the same effect.

### Output health

Every one of the 24 rows (6 samples × 4 rows) completed with:

- `finish = "length"`, `runaway = true` (harness definition: `runaway := (finish == "length")`),
- `attractor = false`, `empty = false`, `saw_done = true`,
- `terminal_count = 1`, `terminal_reasons = ["length"]`, `post_terminal_bytes = 0`,
- `think_words = 0`, `ans_words = 70`.

This is the 128-token `max-tokens` cap truncating a valid, coherent Python
answer mid-function — **"length cap", not a degenerate/attractor loop**. All 24
rows produced **byte-identical** assistant content:

- `sha256 = e572f7d34c43dc877c0875b0396cc7d1bc0950f96666cddcf7a35336aaac7abd`
- identical `request_md5 = dd93dcc0528f194dbd44994d188b03d8` on all 24 rows
- identical `prompt_md5 = 5835c71e471849b4a72e1dc8e39695e7`

Decoded excerpt (identical across all rows; truncated by the length cap):

```
```python
def longest_substring(s: str) -> int:
    """
    Returns the length of the longest substring without repeating characters.

    Args:
        s (str): The input string.

    Returns:
        int: Length of the longest substring with all unique characters.
    """
    char_index_map = {}
    max_length = 0
    start = 0

    for end, char in enumerate(s):
        if char in char_index_map and char_index_map[char] >= start:
            # Move start to the right of the previous occurrence of this character
```

Byte-invariance here is expected (pass-back is byte-invariant on this fixture at
128 tokens) and is **not** evidence of determinism in general — pass-back's row
split is per-process non-deterministic by design, per the serve-log banner
(`passback: memory.offload_exec=passback is non-deterministic by design …`).
The regression is therefore a **rate** effect on identical output, not an output
change.

---

## Fixture identity (measured)

| artifact | digest |
|---|---|
| branch / base source | `perf/passback-opt` off `12daf4fe99017a88242469f60853b860609c20ab` (hipfire 0.4.3) |
| baseline source build | clean tree at `12daf4fe…`; `cargo build --release` |
| candidate 2 source build | same tree **plus the uncommitted worktree delta** (exact shape key + early same-eligibility); **no source commit hash exists** (deliberately not invented) |
| `hipfire` binary A (md5) | `c9feb2095ffe043c7e93477a7ec02a9a` (47,336,152 B) |
| `daemon` binary A (md5) | `c1da18f268129005c6e827ea85ceff49` (88,582,088 B) |
| `hipfire` binary B (md5) | `5ad40778945d67217a26295b7f011ea3` (47,338,032 B) |
| `daemon` binary B (md5) | `5e179e1f39585e864b123ccf29f3fdb2` (88,584,656 B) |
| frozen binary copies | `.codeinsight+research/passback-opt/{baseline-bin/hipfire-12daf4fe,baseline-bin/daemon-12daf4fe,candidate2-bin/hipfire,candidate2-bin/daemon}` |
| model | `~/.hipfire/models/qwen3.5-9b.mq4`, md5 `296092bf1e6a45d78c1acf815eb93366`, 5,313,750,016 B, tag `qwen3.5:9b` |
| prompt fixture | `warm3x_gpu_offload_probe.json` (4 identical prose rows), md5 `c4c78f65b100f645cd1ab3ff0fcbf413`, sha256 `1e058b674082be29a6245eaa526c7100792abc6fa81f09e01ed032ffd9f8e60a`, 1,042 B |
| inner prompt text | md5 `5835c71e471849b4a72e1dc8e39695e7` (215 B; = `benchmarks/prompts/gpu_offload_probe.txt`) |
| sampling | greedy (temp 0.0), thinking off (sentinel cap 1), `max_tokens 128`, `max_seq 4096`, spec off (AR) |
| KV | `auto` → effective K=q8 V=q8 VMM (`kv_backend = vmm`), 8/32 layers carry KV; DeltaNet state Q8 |
| route envs | `HIPFIRE_DPM_WARMUP_SECS=10 HIPFIRE_OFFLOAD_EXEC=passback HIPFIRE_GPU_LAYER_BUDGET=16 HIPFIRE_OFFLOAD_PASSBACK_SHARE=auto` |
| harness | `ab_sample.sh` (one fresh-process sample; row 1 = throwaway, rows 2–4 = measured median) over `scripts/serve_harness.py --mode battery` |
| port | 11541 (serve-log banner, all six runs) |

Host: 1× Radeon RX 9070 XT `gfx1201` (16304 MiB), ROCm/HIP 7.2, Ryzen 7 7800X3D
(8c/16t), 28 GB RAM. Baseline identical to
`.codeinsight+research/passback-opt/IDENTITY.md`. Serve-log load banner per run:
16 of 32 layers host-placed, device weights 3299 MiB of **14669–14697 MiB**
available (B1 saw a lower free figure, 14669/16152, i.e. ~28 MiB less free at
load — a transient host/VRAM-allocation difference, not a binary difference),
pinned host 1753 MiB, `passback: 16/16 spilled layers splittable; share auto;
min step 2 MiB`. Kernel cache warming was 10.00 s in all six runs.

---

## Power / clock / memory telemetry (availability and values)

Power was captured **from sysfs** (dGPU = card1) by `power_snap.sh` pre and post
in every meta log, plus `rocm-smi --showpower --showtemp`. Availability:

- **sysfs (`power1_average_uW`): AVAILABLE.** Values in µW (the field is a
  driver-reported average; raw values are recorded, not converted to a claim).
- **`rocm-smi --showpower`: NOT AVAILABLE.** The `=== Power Consumption ===`
  sections are **empty** in all six meta logs — `rocm-smi --showpower` returned
  no power lines on this host, pre or post (same as candidate 1's record). No
  wattage-matched power claim is made.
- **Temperature (`rocm-smi --showtemp` + sysfs `temp*_input`): AVAILABLE** and
  archived for edge/junction/memory in every meta log.
- **Clocks (`sclk_Hz`/`mclk_Hz`), `perf_level`, active DPM star point,
  `gpu_busy_pct`/`mem_busy_pct`, `vram_used_B`/`vram_total_B`/`gtt_used_B`,
  `host_MemAvailable_kB`: AVAILABLE** in every meta log, pre and post.

Actual sysfs power/clock values (pre → post, µW / Hz), straight from the meta
logs, not assumed:

| sample | `power1_average_uW` pre → post | `temp_edge_mC` pre → post | `temp_junction_mC` pre → post | `temp_mem_mC` pre → post | `sclk_Hz` pre → post | `mclk_Hz` pre → post |
|---|---|---|---|---|---|---|
| A1 | 29,000,000 → 117,000,000 | 42000 → 49000 | 43000 → 50000 | 64000 → 72000 | 91,000,000 → 101,000,000 | 456,000,000 → 1,258,000,000 |
| A2 | 47,000,000 → 122,000,000 | 46000 → 51000 | 48000 → 53000 | 70000 → 74000 | 29,000,000 → 41,000,000 | 1,258,000,000 → 1,258,000,000 |
| A3 | 47,000,000 → 131,000,000 | 47000 → 51000 | 49000 → 53000 | 70000 → 74000 | 1,480,000,000 → 981,000,000 | 1,258,000,000 → 1,258,000,000 |
| B1 | 47,000,000 → 70,000,000 | 46000 → 48000 | 48000 → 49000 | 68000 → 72000 | 1,523,000,000 → 58,000,000 | 1,258,000,000 → 1,258,000,000 |
| B2 | 45,000,000 → 75,000,000 | 45000 → 47000 | 47000 → 50000 | 68000 → 72000 | 58,000,000 → 1,447,000,000 | 1,124,000,000 → 1,258,000,000 |
| B3 | 47,000,000 → 74,000,000 | 47000 → 49000 | 49000 → 51000 | 70000 → 72000 | 29,000,000 → 58,000,000 | 1,258,000,000 → 1,258,000,000 |

All six runs idle at `perf_level=auto`, `power1_cap_uW = 374,000,000`, and
`throttle_log = 0000:03:00.0: thermal throttling logging enabled, with interval
60 seconds` (logging-enabled banner, no throttle entries). Temperatures are
well below any throttle point (junction 49–53 °C). The post-run `power1_average`
of the candidate samples (70–75 mW) is **not** directly comparable to the
baseline samples (117–131 mW): the snapshot is a point sample taken after the
run, and the arms' samples landed at different DPM/clock states
(`pp_sclk_star` differs run to run), so **no power conclusion is drawn** — the
fields are recorded for completeness.

---

## Correctness context (reported by the campaign; not re-measured here)

Reported in the campaign ledger for candidate 2, and **not** re-run or
re-verified as part of this record: the master dispatch suite passed (413
tests), the GPU/split hardware oracle was 17/17 bit-exact on `gfx1201`, and the
LSP was clean. These numbers are reproduced here only as context; the A/B above
measures rate, not correctness, and this record makes no independent
correctness claim.

---

## 27B status

**The 27B real target was not validated in this experiment.** No 27B sample
exists in this record. Per campaign policy (9B iterative; 27B only at the final
validation), 27B validation is deferred to the end of the campaign, not run per
pass. Nothing here supports — or is meant to support — any 27B claim.

---

## What was NOT measured

- **Cause / mechanism.** Not measured. The per-shape-convergence vs
  metadata-overhead hypotheses are **not distinguished** by any receipt here.
- **Power-matched arms.** sysfs power was sampled (above) but `rocm-smi
  --showpower` returned nothing, and post-run point samples landed at different
  DPM states; no wattage-matched claim is made.
- **A trace / per-step profile of candidate 2.** Trace was OFF; no per-shape
  share, key-hit, or probe-count counters were captured. This is why the
  mechanism is left open.
- **A direct A-vs-B-vs-candidate-1 three-way measurement.** Candidate 1 and
  candidate 2 were each measured against the same frozen baseline, not against
  each other.
- **No separate per-run "identity file".** Per-run identity lives inside each
  `*.meta.log` (`env allowlist`, `CLI md5`, `DAEMON md5`, unset check) plus the
  shared `IDENTITY.md`; no standalone identity artifact was produced.
- No re-measurement, no `min_step`/share sweep, no key-split-vs-baseline at
  other budgets, no event-based GPU-arm timing, no 27B, no other model/quant/GPU,
  no cross-context cells, no `HIPFIRE_PROFILE` runs.

## Reproduction

```bash
# one fresh-process sample (row 1 = throwaway, rows 2-4 median); trace OFF inside
bash ab_sample.sh \
  <hipfire-bin> <daemon-bin> passback auto 16 \
  ~/.hipfire/models/qwen3.5-9b.mq4 qwen3.5:9b 11541 \
  /tmp/ab/home-<X> /tmp/ab/<X>-passback
# arms A = baseline binaries (md5 c9feb209 / c1da18f2)
# arms B = candidate-2 binaries (md5 5ad40778 / 5e179e1f)
```

## Raw evidence

Complete raw receipts are committed next to this record at
`data-2026-10-16-passback-shape-isolation-experiment/`, preserving every
original row (no aggregation, no trimming):

- `C2-{A1,A2,A3}-baseline.json`, `C2-{B1,B2,B3}-candidate2.json` — full
  harness JSON (all 4 rows each, complete `content`/`assistant_content`).
- `C2-*-baseline.serve.log`, `C2-*-candidate2.serve.log` — full serve logs
  (load banner, KV, routing, per-step cache lines).
- `C2-*-baseline.meta.log`, `C2-*-candidate2.meta.log` — identity block
  (env allowlist, CLI/daemon md5, unset check), power/temperature (sysfs +
  rocm-smi), trace-line check.
- `ab_sample.sh`, `power_snap.sh`, `warm3x_gpu_offload_probe.json` — the exact
  sampler, power snapshot helper, and fixture used.
- `SHA256SUMS` — integrity digests of every file above.

These are byte copies of
`.codeinsight+research/passback-opt/raw/C2-{A1..B3}-{baseline,candidate2}.{json,serve.log,meta.log}`
and the fixture files, copied without modification.
