# CPU-exec offload benchmarks — handoff runbook

**Audience:** the agent (or human) running the performance measurements for
`memory.offload_exec=cpu`. Written 2026-09-27 by the agent that implemented it.
Read [`perf-benchmarking.md`](perf-benchmarking.md) for the general protocol and
[`../plans/partial-gpu-offload-design.md`](../plans/partial-gpu-offload-design.md)
§ 6.2.1 for what the feature is. This file is only *how to measure it*.

**Non-goals, state them in whatever you report:** these numbers are not an
admission, not a `docs/BENCHMARKS.md` claim, and not comparable across host,
model, quant, GPU, prompt or method. Do not tune anything while measuring. Do not
change code to make a number look better; if a run looks wrong, re-measure.

---

## 0 · What is being measured, and what cannot be

`memory.offload_exec` decides **who multiplies** a spilled layer's weights —
`pcie` (default) runs the GPU kernels against host-mapped weights over the link,
`cpu` executes those GEMVs on the CPU. Placement (`memory.gpu_layer_budget`),
VRAM accounting and KV residency are unchanged either way.

Consequences for measurement design:

- **The sequential daemon path is the one this feature accelerates.** Use
  `hipfire bench --backend noslots`. The slots/serve path
  (`forward_batch_slots` → `dense_ffn_body_slots`) uses batched GEMM kernels that
  never enter `execute_steps`; a `serve`/`serve_harness` number therefore shows
  almost none of this feature (only its per-slot lm_head `Step::Gemv`).
- **Prefill is GPU-side** (batched kernels), so `prefill_tok_s` should be
  unaffected — treat a large prefill delta as a smell, not a result.
- **README.md**/`tests` numbers coming from `hipfire run` are a *coherence* tool,
  not a rate tool (see § 5).
- The feature's win is bounded by the host's memory bandwidth and the per-step
  D2H/H2D pairs. Expect a **modest** win, and only when the per-layer weight
  bytes are large (see the crossover in § 6).

## 1 · Fixture identity (verify before you start)

```bash
cd <repo>; git rev-parse HEAD          # 924d7fd8f6b3c674b95a60ed3b45df0e75f08d30 at handoff
md5sum target/release/daemon target/release/hipfire
md5sum ~/.hipfire/models/qwen3.5-2b.mq4 ~/.hipfire/models/qwen3.5-9b.mq4
md5sum /mnt/sx8200/qwen3.8-27b.mq3-xt
md5sum benchmarks/prompts/gpu_offload_probe.txt benchmarks/prompts/humaneval_3_below_zero.txt
```

| artifact | md5 | sha256 | notes |
|---|---|---|---|
| `target/release/daemon` | `904368995cddb8cda8e82a7f8d31ae96` | — | at handoff |
| `target/release/hipfire` | `030f080ce4c3d68ca038a614059d21b2` | — | at handoff |
| `~/.hipfire/models/qwen3.5-2b.mq4` | `9ed6628f2df83ef4b1c062afd4a85bfb` | `bb386f7bd24397db5ef6ba28aab5985053c9db319d1645a2ba4d732a77badc6a` | 24 layers, 30.4 MB/layer |
| `~/.hipfire/models/qwen3.5-9b.mq4` | `31a8d8dc7603226801b08d8319015602` | `829a84c708eed3db785febfe80b9a46dab2bb52172b9ac40ab17856f8f1260b3` | 32 layers, 114.9 MB/layer |
| `/mnt/sx8200/qwen3.8-27b.mq3-xt` | `80bb9198e6a565fc006b2ae1b7c89eca` | `3e04fc8db80bda557b965ec60ac876cf2500fced7f340624f3fcbeae134af5c5` | 11,777,616,896 B; 64 layers, 154.6 MB/layer; **every projection is qt 49** |
| `benchmarks/prompts/gpu_offload_probe.txt` | `5835c71e471849b4a72e1dc8e39695e7` | `b6eddc54931a1daa28c32fc8381a72d982eded30093ddf13a6a71049a695066f` | 215 B, 59 tokens — the 2026-09-26 sweep's prompt, use it for the sweep |
| `benchmarks/prompts/humaneval_3_below_zero.txt` | `37c5aad9f9efe93b5c47f27256bdf149` | — | used for the arm-D rate comparison |

Host: 1× RX 9070 XT `gfx1201` (16304 MB), ROCm/HIP 7.2, Ryzen 7 7800X3D
(8c/16t), 28 GB RAM, THP `[always]`. Copy this table into your report and fill in
*your* binary digests; a rebuild changes them and invalidates comparison with
anything above.

## 2 · The two knobs

- `HIPFIRE_GPU_LAYER_BUDGET=N` — `N` layers stay **on** the GPU, the prefix
  `[0 .. n_layers-N)` is spilled to host-mapped RAM. It is a **hard pin**, not a
  fit heuristic: `=12` on a 24-layer model spills 12 layers even though they
  would fit.
- `HIPFIRE_OFFLOAD_EXEC=pcie|cpu` — who multiplies (default `pcie`; unset, empty
  and unknown all fail closed to `pcie`).

**Assert the split actually happened.** `residency_report` prints
`partial offload: … i_gpu_start=…` **only when a budget was configured**, so a
missing line means the env never reached the daemon — not that nothing spilled.
Grep the daemon output for both lines:

```bash
grep -E "partial offload|cpu exec: [0-9]+/[0-9]+ spilled layers fully covered" "$ERR"
```

The second line appears only with `HIPFIRE_OFFLOAD_EXEC=cpu` and must read
`{spilled}/{spilled} … uncovered quants: none`. A non-empty uncovered list is not
a bug — it names formats that stay on PCIe (`Q8HFQ`, `MQ8G256`, the HFP4/MFP4
family, PARO) — but it changes what the arm measures, so report it.

## 3 · The measurement loop (and the five things that bit me)

```bash
PROMPT=benchmarks/prompts/gpu_offload_probe.txt
COMMON=(--spec off --runs 5 --warmups 2 --max-tokens 64 --backend noslots \
        --workload stateless --prompt-file "$PROMPT" --json)
run_arm() {  # <tag> <model> <budget> <pcie|cpu>
  pkill -f 'target/release/daemon'; rm -f ~/.hipfire/daemon.pid
  for _ in $(seq 20); do pgrep -f 'target/release/daemon' >/dev/null || break; sleep 0.5; done
  env HIPFIRE_GPU_LAYER_BUDGET=$3 HIPFIRE_OFFLOAD_EXEC=$4 \
    ./target/release/hipfire bench "$2" "${COMMON[@]}" > "/tmp/arm_$1.json" 2> "/tmp/arm_$1.err"
}
```

1. **Kill daemons BY PATH, then wait for exit.** A stale `~/.hipfire/daemon.pid`
   can name a *reused* pid; `kill -9 $(cat ~/.hipfire/daemon.pid)` then kills
   something unrelated (it killed my sweep driver twice). Conversely, removing
   the pid file while the daemon is still dying makes the next point fail with
   `FATAL: hipfire daemon already running` — every subsequent point then fails in
   ~1 s with no JSON. That failure mode cost me three sweep legs.
2. **The host must be otherwise idle.** A concurrent `cargo build --release` made
   the `cpu` arm read 3.1 tok/s instead of 24.1 — a 6× error that looks exactly
   like "the CPU path is slower". Check `uptime` (this box idles at ~0.5) before
   trusting any number.
3. **Interleave the arms and use a fresh process per point.** Same-arm runs batched
   together drift on this box; `pcie, cpu, pcie, cpu, …` with one process per arm
   per round, then take the median of the per-run medians.
4. **`--spec off --backend noslots --workload stateless` must be explicit.** The
   default backend is `both`, which measures two things at once.
5. **Prompt bytes are part of the fixture.** Use the committed prompt files and
   record their md5. One newline moves τ by up to 17% (AGENTS.md §0 rule 2). Do
   not use the inline prompt from the implementation session — it has no file and
   therefore no digest.

## 4 · Scope of the run (what to actually measure)

**A. Arm D — the rate comparison at a fixed spill (do this first).**
9B, budget 24 (8 of 32 spilled), three fresh processes per arm, interleaved:

```bash
run_arm 9b_pcie_1 qwen3.5:9b 24 pcie
run_arm 9b_cpu_1  qwen3.5:9b 24 cpu
run_arm 9b_pcie_2 qwen3.5:9b 24 pcie
run_arm 9b_cpu_2  qwen3.5:9b 24 cpu
run_arm 9b_pcie_3 qwen3.5:9b 24 pcie
run_arm 9b_cpu_3  qwen3.5:9b 24 cpu
```

**B. The spill sweep** — reproduce the 2026-09-26 curve *and* add the `cpu` arm
at every point, so the `pcie` arm is the control for the binary change:

| model | layers | spilled points (budget = layers − spilled) |
|---|---|---|
| `qwen3.5:2b-mq4` | 24 | 0, 3, 6, 9, 12 (budgets 24, 21, 18, 15, 12) |
| `qwen3.5:9b` | 32 | 0, 4, 8, 12, 16 (budgets 32, 28, 24, 20, 16) |
| `qwen3.8-27b.mq3-xt` (path) | 64 | 0, 8, 16 (budgets 64, 56, 48) |

Three fresh processes per point for the 2B and 9B (single-process 2B points swung
~4% between passes); one process per point is acceptable for the 27B if you
report the spread.

**C. Capacity parity (a pass condition, not a bonus).** `vram_free_mb` from the
bench JSON must be **the same** across arms at the same budget. This feature moves
no bytes out of VRAM; a difference means the host-mapped allocation path
regressed. Also record `vram_free_before_mb`.

**D. Per-step attribution** (optional but cheap, and it explains the delta):

```bash
env HIPFIRE_GPU_LAYER_BUDGET=24 HIPFIRE_OFFLOAD_EXEC=cpu HIPFIRE_CPU_EXEC_TRACE=1 \
  ./target/release/hipfire bench qwen3.5:9b "${COMMON[@]}" 2>&1 >/dev/null | grep "cpu exec: step"
```

Each line gives the running-mean D2H / GEMV / H2D split for a step shape, and the
second counter *must* be `0 host-mapped steps still on GPU`. A non-zero value
means a step shape never reached the CPU — report it, do not average it away.

## 5 · Reading the output, not just the numbers

- **Decode the text.** `hipfire bench` sets reasoning off (closed-think template),
  so its text is degraded by construction: treat bench as rate/metadata only. The
  coherence read belongs to `hipfire run` (sequential path) or
  `python3 scripts/serve_harness.py` (serve path).
- **Reasoning checkpoints fail closed.** On `qwen3.8-*`, `hipfire run` with a
  token budget smaller than the think block ends with
  `daemon error: … open think span at end of generation (validation)` and
  releases **no text**. 600 tokens was not enough for the 27B; either raise
  `-n` a lot, or set `reasoning.effort=none` (a config change — ask before
  making it) — `hipfire run` has no reasoning flag, and the corresponding config
  keys have no env spelling.
- **A suspiciously tight stddev on a spec-decode bench is a warning, not a win**
  (AGENTS.md §0 rule 3). Eyeball the text whenever τ or the spread looks unusual.
- **Do not trust a single `hipfire run` comparison for rate.** It samples at the
  configured temperature unless you pass `--temp 0`, and its prompt is inline.

## 6 · Results already in hand (sanity anchors, same fixture)

| measurement | value | notes |
|---|---|---|
| 9B budget 24, 3 fresh processes/arm, interleaved | `pcie` 21.6 / `cpu` **24.1** tok/s (+11.6%) | humaneval prompt, 128 gen, daemon md5 `6b2f8558…` (pre-hoist; qt 13 unaffected) |
| 9B budget 16 (16 spilled), 1 pass | `pcie` 11.0 / `cpu` **14.0** tok/s | |
| 2B, probe prompt, 64 gen | pcie/cpu: 0 → 252/259, 3 → 133/100, 6 → 91/65, 9 → 69/50, 12 → 57/40 tok/s | `cpu` **loses** on the 2B at every spill ≥3 |
| 2026-09-26 `pcie` curve (same fixture, previous binaries) | 2B 250/128/86/65/52; 9B 101/35/21/15/12 tok/s | your `pcie` arm should land within ~5% of these |
| 27B qt 49 step, m=12288 k=5120 | 5.70 ms (was 19.95 ms before the header hoist) | `HIPFIRE_CPU_EXEC_TRACE=1` |
| kernel ceiling, 195.8 MB host-mapped buffer, 16 threads | 44.5 GB/s (44.8 over a plain heap `Vec`) | the end-to-end ~23 GB/s is per-step serialization, not codegen |

**Crossover to expect.** The CPU arm wins when a spilled layer's weight bytes
dominate that layer's per-step copy+sync cost: 2B ~30 MB/layer loses, 9B
~115 MB/layer wins at ≥8 spilled, 27B ~155 MB/layer should win by more. Report
the point where the arms cross rather than a single "cpu is faster" claim.

**Retracted readings — do not reproduce or cite.** An earlier pass reported
`cpu` 3.1 tok/s vs `pcie` 19-21, and a second 5.6 tok/s. Both ran while a release
build and another sweep were on the same 16 threads. If you see ~3 tok/s,
suspect host contention or a lingering daemon before suspecting the feature.

## 7 · 27B specifics (read this before running it)

- **Footprint:** `hipHostMalloc` memory is pinned and unreclaimable. Budget 56
  (8 spilled ≈ 1.2 GB pinned) loads and covers `8/8`; budget 48 (16 spilled
  ≈ 2.5 GB) stalled twice at load layer 62/64 with 0 GB RAM free and host load
  ~6-10 from unrelated work. Start at 56, and only go to 48 on an idle box.
- **All 497 projections are qt 49 (`MQ3G256V2`)** — covered, with no canonical
  host decoder, so its oracle is the production launcher on real tensors:
  `HIPFIRE_PARITY_EXTRA_MODEL=/mnt/sx8200/qwen3.8-27b.mq3-xt
  HIPFIRE_PARITY_EXTRA_QT=49 cargo test -p hipfire-arch-qwen35 --release
  --test gpu_gemv_parity -- --ignored` → 2.391e-7 / 3.530e-7 at m=12288 k=5120.
  Run that before the sweep: it is 1 s and it separates "slow" from "wrong".
- qt 49 currently uses the **scalar** decode (only `Mq4G256` has an AVX2 kernel),
  so its step time is decode-bound (~11 GMAC/s) rather than bandwidth-bound. That
  is the expected next lever, not a defect to chase in the sweep.

## 8 · Reporting (this is the deliverable)

1. Identity table (§ 1) filled with *your* digests, including `git rev-parse HEAD`.
2. The bench JSON **whole**, not summarised — the `samples` array is the evidence
   that a delta cleared the noise; keep all three processes per arm.
3. `residency_report` + coverage line for every point (they prove the requested
   split and that nothing silently stayed on PCIe).
4. `vram_free_before_mb` / `vram_free_mb` per arm (capacity parity).
5. The decoded text for at least one `cpu` point and its `pcie` control, read by
   a human, with the first-divergence offset if they differ.
6. A dated file in [`docs/perf-checkpoints/`](../perf-checkpoints/README.md)
   (lifecycle `historical`; **never** edit an existing one — corrections are new
   dated amendments) plus the numbers in the commit message body. Follow that
   directory's README rules before adding.
7. Say explicitly what you did **not** measure (slots/serve path, prefill,
   formats that stayed on PCIe).

## 9 · Known open items you may be asked to close

- 2B/9B sweep: 2B leg measured (three processes per point), **9B leg not**.
- 27B sweep on `qwen3.8-27b.mq3-xt`: **not measured at all** (only per-step
  timings and coverage).
- 27B decoded-text read: **never done** — every attempt hit the reasoning-budget
  gate (§ 5).
