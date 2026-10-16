# Passback offload optimization (gfx1201) — controlled native original→FINAL summary, 9B **+6.3 %** / 27B **+6.0 %** — 2026-10-16

**Lifecycle:** `historical`. **Disposition: fixture-bound measured evidence — NOT a current baseline, NOT an admission, NOT a product claim.** This record closes the `perf/passback-opt` campaign: it measures the **frozen FINAL pair** (retained `C4` asynchronous owned returns + `C5` direct sampled GPU timing; `C6` fully reverted) against the **original build at HEAD `12daf4fe`** on **one host, one GPU (gfx1201), one quant (MQ4), two models (9B primary; 27B campaign-end), greedy decoding**, via the **official native `hipfire bench` instrument only**. The numbers here do **not** transfer across host, model, quant, GPU, prompt, route, share, budget, or method without a new record.

This is a **new** checkpoint; no existing checkpoint or amendment is modified (append-only: [`README.md`](README.md) §Preservation). It links — and does not re-run — the candidate-investigation ledgers (`C1`–`C6`) below. Raw receipts for every number and claim in this record are in [`data-2026-10-16-passback-optimization-summary/`](data-2026-10-16-passback-optimization-summary/) (**129 files**: 128 receipts + `SHA256SUMS`), each `cmp`-verified byte-identical to its source under `.codeinsight+research/passback-opt/`.

**Headline (official native instrument, decode tok/s, N=3 fresh CLI+daemon processes per arm; ONE process sample = median of the 3 internal measured runs).**

| model / fixture | ORIG passback | FINAL passback | **Δ** | FINAL cpu | FINAL pcie |
|---|---:|---:|---:|---:|---:|
| 9B, budget 16/32 resident (16 spill) | 19.1 | 20.3 | **+6.3 %** | 15.9 | 13.2 |
| 27B, budget 56/64 resident (8 spill) | 16.7 | 17.7 | **+6.0 %** | 15.1 | 11.6 |

The two figures are the **actual** native measurements; there is **no compound ≈8 % headline** — the earlier development-probe deltas (`C1`…`C5` over separate, sequential candidate worktrees) are superseded by these one-shot original→FINAL numbers.

## Scope and provenance corrections (read before citing)

- **Retained source.** FINAL is the worktree at base **HEAD `12daf4fe99017a88242469f60853b860609c20ab`** (`perf/passback-opt`, hipfire 0.4.3). The worktree **started clean** at that commit; **every** changed file is this campaign's. The FINAL binaries were **built after the retained source edits** and **frozen before timing** (the md5s below are the measured artifacts) — the peer intro's "built before those edits / pre-existing edits" is **wrong**.
- **Git tracking vs validation base.** The branch's configured tracking remote is `aldrouil` = `https://github.com/aldrouil/hipfire` (`branch.perf/passback-opt.remote=aldrouil`, `merge=refs/heads/perf/passback-opt`); `git ls-remote` over that HTTPS remote returns **no** `perf/passback-opt` branch — the destination is configured for eventual publication, **nothing is pushed**. The **validation/PR base is `upstream/beta`** (`branch.perf/passback-opt.vscode-merge-base=upstream/beta`) — **not** `origin/beta`. `origin` (`git@github.com:aldrouil/hipfire.git`, SSH) is untouched.
- **KV resolution differs by model** (uniform within a model, all four arms identical): 9B `--kv-mode auto` → **K=q8, V=q8**, backend `vmm` (`kv_mode='q8'`); 27B → **KV=fp8**, backend `vmm` (`kv_mode='fp8'`). Do **not** claim both models resolve the same format, and do **not** compare rates across models.
- **Native finish status is not exposed.** The native `bench --json` output carries no per-run `finish`/`gen` field; `decode_tok_s = generated / decode_s` uses `generated = 128` = the **requested** `--max-tokens` (the metric's numerator). This record therefore does **not** claim the native runs verified `finish=length` from JSON — that status is only observable in the separate serve-harness receipts (§5).
- **Power units.** sysfs `power1_average_uW` snapshots are in **µW**: `48e6 µW = 48 W` (the earlier peer prose "48 mW" is a unit error). See §4.
- **Statistics.** Per-arm `sd`/`cv` are the **population** standard deviation (`statistics.pstdev`) of the **N=3 process medians** (not of the 9 internal runs). Native `stdev` fields inside one JSON are likewise population. The non-overlap of the two arms' ranges is **descriptive only**, not a confidence or significance claim.

## 1. Original architecture and bottlenecks

Partial offload places GPU-prefix rows `[0, g)` and CPU-tail rows `[g, m)` of a projection (g 8-aligned, step ≥ 2 MiB), with host-mapped, unpadded, row-stride-decoded **quantized** weights and SIMD-eligibility plus a per-`(dtype, k)` probe/EWMA/slew/latch/reopen controller (the F32-extent checks apply to the arm's **outputs/residuals**, not to the quantized weight rows) (`crates/hipfire-dispatch/src/offload_split.rs`, `cpu_exec.rs`). In the **original** build the pass-back split path serialised each step at host boundaries:

1. **Every split-step transfer was a blocking pageable `hipMemcpy`** — activation D2H, the AWQ-scale D2H on rotation, the residual accumulator D2H, and the CPU rows' H2D return. Each forced the host to wait, so a step was a host sync point and the CPU and GPU arms could not overlap with the next step's work.
2. **The CPU return was blocking**, so the caller could not proceed until the CPU rows were back on the device.
3. **The share controller inferred the GPU arm's rate from the completed H2D join** relative to the shape's own no-wait floor, observable only on a *waited* join: `min_join_ns` never rose, an at-floor shape carried a dead GPU anchor, and a `share += 0.02` upward ratchet had no downward counterpart.
4. **The whole-CPU route read back with the legacy blocking `hipMemcpy` on the default stream**, which orders other **blocking** streams but **not** a `hipStreamNonBlocking` producer; a non-blocking `active_stream` consumer (e.g. the residual `add_inplace` between two CPU steps) could still be in flight, so the read could see stale bytes.

The GPU reads its own host-mapped weight rows over PCIe rather than staging weights per step; the remaining cost is the per-step boundary serialisation above plus host/PCIe contention.

## 2. Delegated investigations and findings

Read-only scouts (findings only, no code): **ScheduleScout** (scheduling/granularity), **OverlapScout** (concurrency/sync), **CriticalPathScout** (intervals, shape inference), **HipCopyScout** (exact blocking HIP calls and the existing pinned owner), **HistoricalPassbackScout** (local recorded data; **no** historical re-runs/bisect), **CpuScheduleScout** (Rayon path/CPU topology), **GpuTimingOwnerScout** (event APIs; ownership inferences corrected against the actual `Gpu` ownership/lifetime model; `HipRuntime` is `Send + Sync`), **NextScheduleScout** (key pooling vs frozen sampling; corrected to a CPU trace **per real shape**, only the GPU column pooled), **CaptureFrontierScout** (capture/replay guard + validation frontier).

Implementers: **OverlapTimingFix** (`C1`), **ShapeIsolationFix** (`C2`), **PinnedSplitTransfer** (`C3`) + `C2` revert, **RepairSplitOracle** (`MFP4` real-fixture tiling), **CpuStreamDependencyFix** (legacy CPU ordering), **AsyncSplitReturn** (`C4`), **DirectGpuArmTiming** + **DirectTimingPolish** (`C5`, fail-closed + 4-state errors corrected before bench), **FrozenSamplingPolicy** (`C6`), **RevertFrozenSampling** (exact rollback).

Reviewers: **PinnedCopyReview**, **AsyncReturnReview**, **DirectTimingReview** (resource/stream/lock/lifetime approval after parent-required fixes: generic failed-end/read-state fix, library-retention proof). Benchmarks: **BaselineBench**, **C4ControlsBench**, **C5CompareBench**, **C6CompareBench**, **FinalAcceptanceBench** — **all timings by the benchmark runner**, never by implementers. Evidence/docs: **RejectedTrialLedger**, **RejectedShapeLedger**, **C4EvidenceLedger**, **C5EvidenceAndDocs**, **C6RejectedLedger**. Early traced cold/instrumented `k-*` runs are **invalid as headline** and were replaced by the strict same-process warm protocol before the controlled `C1` results.

**Candidate findings** (each recorded immutably in its own ledger; these are development probes on the serve harness, **not** the headline instrument):

- `C1` (CPU-GEMV-only interval): 19.6 → 19.7 (+0.5 %) — **null, REVERTED**.
- `C2` (exact shape/op key): 19.8 → 18.7 (−5.6 %) — **REGRESSED, REVERTED**.
- `C3` (pinned copies at all host boundaries): 19.85 → 19.55 (−1.5 %) — **no standalone win**; its **pinned storage is retained only as the enabler** of the measured combined gain. Worker `RAYON_NUM_THREADS` screen 16/8/4/16: 20.0/19.9/14.5/19.9 — no gain, no affinity; no CPU-SIMD change.
- `C4` (asynchronous split returns): original 19.75 → 21.0 (+6.3 %) on the serve harness (N=4, `ABBAABBA`); CPU 16.8 / PCIe 13.0 controls. River secondary 18.5 → 19.3 (+4.3 %) is **provisional** (drift/overlap, not clean replication). The input/pinned and sampling changes were **combined**; no single cause isolated.
- `C5` (direct GPU timing + CPU interval + removed censoring): `C4` 20.8 → `C5` 21.2 (+1.9 %, N=4); reverse 20.95 → 21.25 (+1.4 %, N=4); river 21.1 → 21.3 (+0.95 %, N=3). Small, replicated edge; GPU trace `k=12288` share 0.375, rGPU 23.8 GB/s — **optimality not proved** (no fixed-share counterfactual).
- `C6` (frozen 1/16 vs ¼ sampling): 20.6 → 20.3 (−1.5 %, N=4) — **REJECTED, FULLY REVERTED**; no river/load challenge because the pre-defined stop fired.

## 3. Retained implementation

FINAL is exactly `C4 + C5`; the `C6` delta is absent. Source delta `source.patch` (135,827 B, sha256 `d0a5db1f…cf330`) touches `CHANGELOG.md`, `crates/hipfire-dispatch/src/{cpu_exec.rs,offload_split.rs}`, `crates/hipfire-arch-qwen35/tests/gpu_gemv_parity.rs`, `crates/rdna-compute/{map.md,src/dispatch.rs,src/lib.rs}`, and the new `crates/rdna-compute/src/span_timer.rs`; `docs/plans/partial-gpu-offload-design.md` is updated in step.

1. **Pinned, owned split transfers without changing the required read ordering.** The split step's four copies (activation D2H, AWQ-scale D2H on rotation, accumulator D2H for the residual form, CPU-rows H2D) enqueue into the reusable pinned allocation the host-executed MoE splice already owns, on the stream the GPU arm runs on. The **read still completes before the GPU arm launches** (the boundary the blocking D2H provided) and the host still copies staged bytes into the arm's own vectors before the CPU multiply; the H2D is completed as the blocking join **only on the sampled step**. Persistent ownership preserves the storage; the next same-stream D2H is ordered behind the prior H2D and its producers before any host read; resize/stream/device-change/`Drop` drains or retains.
2. **Asynchronous split returns, sampled observation.** `cpu_arm_finish`/`upload_arm_result` take a private "wait for return" flag: whole-CPU callers (`memory.offload_exec=cpu`, `share=0`, the seeding probe, `run_host_mapped_gemv{,_residual}`) still complete it; `offload_split` chooses **per step** — **~one in four** sampled (top two bits of a per-shape LCG, so the cadence cannot phase-lock to the layer/projection sequence that revisits a `(dtype,k)` key; each shape's first step is always sampled), the rest queued and not waited.
3. **Direct sampled GPU-arm timing + real CPU interval; join-floor censoring removed.** A sampled step hands `observe` the CPU arm's multiply interval (`gemv_ns` — the interval that actually overlaps the GPU arm, D2H excluded) and the GPU arm's own elapsed time, bracketed by a **reusable per-`:Gpu` event pair** (`Gpu::span_begin`/`span_end`/`span_elapsed_ns`, `crates/rdna-compute/src/span_timer.rs`) on the stream the arm and its H2D both run on. The elapsed time is read only after the sampled step's **completed H2D**, so the measurement adds no event synchronization and no fence; the old join-floor inference, its wait gate, and the `+0.02` ratchet are gone, and the balance `r_gpu/(r_gpu+r_cpu)` comes from both real durations.
4. **Whole-CPU reads ordered against the active non-blocking stream.** The whole-CPU route synchronizes the actual active stream before the blocking copy when one is set (no active stream ⇒ the blocking copy already has the ordering), fixing the non-blocking-stream residual short-by-one-add case (`cpu_offload_return_stream_and_reuse_parity`).

`span_timer` is a generic lazy per-`:Gpu` two-event owner with an explicit `Idle/Started/Recorded/Failed` state: an unresolved pair is drained **before** any re-record or destroy, destroy binds the owning device; failures propagate to the caller (a measurement that cannot be taken fails the step rather than silently returning a stale/missing one); `HipRuntime` outlives normal teardown and a leaked event has no host callback/pointer. `docs/plans/partial-gpu-offload-design.md` records this design.

## 4. Official controlled before/after (CPU/PCIe controls) — 9B and 27B

**Instrument (native `hipfire bench`, AGENTS §3.1 headline).** `bench MODEL --runs 3 --warmups 3 --spec off --backend noslots --workload stateless --max-tokens 128 --max-seq 4096 --kv-mode auto --kv-backend vmm --prompt-file benchmarks/prompts/gpu_offload_probe.txt --json`. **N=3 fresh CLI+daemon processes per arm**, balanced interleave with rotated round order; fresh isolated `HIPFIRE_HOME`+lockdir per process; `HIPFIRE_DAEMON_BIN` launcher logs `WRAP-INVOCATION daemon_md5=… cli_md5=…` **per sample** (`*.daemonproof.log`); `RAYON_NUM_THREADS` unset, traces/profilers unset, normal affinity, `HIPFIRE_DPM_WARMUP_SECS=10`, no registry fetch, `HIPFIRE_LOCAL=1`. ONE process sample = median of the 3 internal measured runs (after 3 warmups). Exact commands: [`FIN-one.sh`](data-2026-10-16-passback-optimization-summary/FIN-one.sh) + [`FIN-9B.sh`](data-2026-10-16-passback-optimization-summary/FIN-9B.sh) / [`FIN-27B.sh`](data-2026-10-16-passback-optimization-summary/FIN-27B.sh); analysis [`FIN-analyze.py`](data-2026-10-16-passback-optimization-summary/FIN-analyze.py).

### 9B — `qwen3.5-9b.mq4`, budget 16 resident / 16 spill (50 % of 32 layers)

| arm | process medians (tok/s) | **median** | mean | sd (pop) | CV | range |
|---|---|---:|---:|---:|---:|---:|
| ORIG passback | 19.1, 19.3, 18.3 | **19.1** | 18.90 | 0.43 | 2.3 % | 18.3–19.3 |
| FINAL passback | 20.3, 21.3, 19.8 | **20.3** | 20.47 | 0.62 | 3.0 % | 19.8–21.3 |
| FINAL cpu | 16.3, 15.8, 15.9 | **15.9** | 16.00 | 0.22 | 1.4 % | 15.8–16.3 |
| FINAL pcie | 13.2, 13.5, 12.8 | **13.2** | 13.17 | 0.29 | 2.2 % | 12.8–13.5 |

Per-sample internal decode runs: ORIG-pb R1 [19.0,19.1,19.2] / R2 [19.3,19.3,19.4] / R3 [18.1,18.3,18.4]; FIN-pb R1 [20.3,20.3,20.7] / R2 [21.3,21.5,21.1] / R3 [19.8,19.7,20.2]; FIN-cpu R1 [16.3,16.3,15.9] / R2 [15.6,15.8,15.8] / R3 [16.2,15.9,15.9]; FIN-pcie R1 [13.2,13.0,13.2] / R2 [12.9,13.5,13.5] / R3 [12.8,12.8,12.3].

- **ORIG→FINAL: 19.1 → 20.3 = +6.3 %** (median); ranges **non-overlapping** (orig 18.3–19.3, final 19.8–21.3) — descriptive only.
- FINAL passback vs FINAL cpu **+27.7 %** (15.9); vs FINAL pcie **+53.8 %** (13.2).
- TTFT medians (ms): orig-pb 596.4 / fin-pb 585.3 / cpu 595.3 / pcie 590.2 — flat. Prefill medians ≈ 97–101 tok/s = launch overhead at 59 prompt tokens (§Caveats).

### 27B — `qwen3.8-27b.mq4`, budget 56 resident / 8 spill (12.5 % of 64 layers, uniform all arms)

Fit: budget 56 loads (device weights 13379 MiB of 14530 MiB, pinned host 1543 MiB); the old budget-8 config thrashed (≈3.3 tok/s) and is **not** comparable. 56/8 was held for **all** arms.

| arm | process medians (tok/s) | **median** | mean | sd (pop) | CV | range |
|---|---|---:|---:|---:|---:|---:|
| ORIG passback | 16.1, 16.8, 16.7 | **16.7** | 16.53 | 0.31 | 1.9 % | 16.1–16.8 |
| FINAL passback | 17.6, 17.8, 17.7 | **17.7** | 17.70 | 0.08 | 0.5 % | 17.6–17.8 |
| FINAL cpu | 15.3, 15.1, 15.1 | **15.1** | 15.17 | 0.09 | 0.6 % | 15.1–15.3 |
| FINAL pcie | 11.6, 11.6, 11.6 | **11.6** | 11.60 | 0.00 | 0.0 % | 11.6–11.6 |

Per-sample internal decode runs: ORIG-pb R1 [16.1,16.0,16.4] / R2 [16.8,16.8,16.9] / R3 [16.7,16.7,16.8]; FIN-pb R1 [17.6,17.7,17.5] / R2 [17.8,17.8,17.6] / R3 [17.7,17.7,17.6]; FIN-cpu R1 [15.3,15.3,15.2] / R2 [15.1,15.1,15.1] / R3 [15.0,15.1,15.1]; FIN-pcie all three [11.6,11.6,11.6].

- **ORIG→FINAL: 16.7 → 17.7 = +6.0 %** (median); ranges **non-overlapping** (orig 16.1–16.8, final 17.6–17.8) — descriptive only.
- FINAL passback vs FINAL cpu **+17.2 %** (15.1); vs FINAL pcie **+52.6 %** (11.6).
- TTFT medians (ms): orig-pb 124.8 / fin-pb 125.1 / cpu 124.0 / pcie 124.7 — flat. Prefill medians ≈ 467–477 tok/s (same 59-token caveat).
- The FINAL-pcie `sd 0.00` is a **reported-precision artifact**: the archived per-run decode samples are **one-decimal** arrays and each process median is exactly `11.6`, so the spread computed from them is zero — **not** evidence of physically zero variance. The native JSON exposes no finer per-run precision, so this record cannot distinguish "no variance" from "variance below the reported precision."

**Placement proof (stderr, every sample).** 9B: `partial offload: 16 of 32 layers host-placed`, `cpu exec: 16/16 spilled layers fully covered`, passback `16/16 spilled layers splittable; share auto; min step 2 MiB`; pcie arms show **zero** `cpu exec` lines (pure PCIe). 27B: `partial offload: 8 of 64 layers host-placed`, `cpu exec: 8/8 spilled layers fully covered`, `passback: 8/8 spilled layers splittable`. Daemon md5s proven per sample (orig `c1da18f2` / final `98265d8b`).

**Power (sysfs `power1_average_uW` point snapshots pre → post; µW; 48e6 µW = 48 W — NOT steady-state, NOT comparable across arms, no power/energy claim drawn).**

| 9B run | pre → post (×10⁶ µW) | 9B run | pre → post | 27B run | pre → post | 27B run | pre → post |
|---|---|---|---|---|---|---|---|
| R1-orig-pb | 47 → 54 | R1-fin-pb | 48 → 56 | R1-orig-pb | 48 → 62 | R1-fin-pb | 49 → 67 |
| R2-orig-pb | 48 → 49 | R2-fin-pb | 48 → 54 | R2-orig-pb | 49 → 60 | R2-fin-pb | 48 → 61 |
| R3-orig-pb | 48 → 54 | R3-fin-pb | 50 → 56 | R3-orig-pb | 48 → 57 | R3-fin-pb | 48 → 60 |
| R1-fin-cpu | 48 → 54 | R1-fin-pcie | 48 → 53 | R1-fin-cpu | 49 → 52 | R1-fin-pcie | 48 → 61 |
| R2-fin-cpu | 48 → 50 | R2-fin-pcie | 48 → 55 | R2-fin-cpu | 49 → 52 | R2-fin-pcie | 49 → 57 |
| R3-fin-cpu | 47 → 51 | R3-fin-pcie | 48 → 64 | R3-fin-cpu | 49 → 49 | R3-fin-pcie | 48 → 60 |

## 5. Correctness

Existing `serve_harness.py` on the FINAL binaries, same offload settings, greedy, thinking off, `max-tokens 256`, `max-seq 4096`, `kv auto`, budget 16 (9B) / 56 (27B) ([`FIN-correct.sh`](data-2026-10-16-passback-optimization-summary/FIN-correct.sh)):

- **9B battery** (hw-gate 5-prompt fixture): 5/5 `finish=stop`, `attractor=false`, `empty=false` (gen 149/91/17/31/17); code row returns a working `longest_unique_substring`; sky/Paris/JSON/43 rows correct.
- **9B chain**: t1 `finish=length` (`runaway=true`, gen 256 — the 256-token cap truncating a valid answer, **not** an attractor), then 4 turns `finish=stop` with cached prefixes 300→866; coherent multi-turn state.
- **27B battery**: 5/5 `finish=stop` (gen 132/82/19/31/96), expect-substrings hold. **27B chain**: 5/5 `finish=stop`, cached 0→675, coherent.
- **Decoded text read in full** (above + archived JSONs); no attractor, no runaway-empty.
- **share0-vs-CPU identity smoke** (fixture-scoped, **not** a generic auto-golden): passback `share=0` vs `cpu`, 9B canonical prompt, budget 16 — **520/520 chars byte-identical**, `sha256 7690b0d3…`. `share=0` collapses to the CPU engine by design, so this proves the seam wiring on that fixture only; no token-ID-level or general float-equality claim.

**Dispatch route** (existing `redline_daemon_harness.py`, FINAL daemon; [`FIN-redline.sh`](data-2026-10-16-passback-optimization-summary/FIN-redline.sh)):

- **Resident 9B pcie/budget 32 — multi-position (N=3) compatibility only**: `resident-pcie32-multiposition.json` → `pass=True` with `--shadow-iterations 3` (427 dispatches, 1 packet); prefill stable (516 launches, 20 AQL contract kernels), decode capture stable (427 dispatches). **PM4/HIP/blob state parity over 3 consecutive positions** (the shadow result field is named `aql_shadow`, with `backend=pm4_ib` — a recorded-command-stream state equality, **not** a runtime AQL timing claim): `bit_exact=True`, `blob_bit_exact=True`, `logits_equal=True`, `kv_equal=True`, `recurrent_equal=True`, `gdn_frame_equal=True`, `blob_gdn_frame_equal=True`. This is a **compatibility/state-equality** check, **not** an admitted timed performance arm, so no decode-rate or PM4-vs-HIP timing figures are cited here. The earlier `resident-pcie32.json` (iterations=1) is retained but **superseded** for the parity criterion. Model identity: report path `/mnt/sx8200/hipfire/models/qwen3.5-9b.mq4` = same file as `~/.hipfire/models/qwen3.5-9b.mq4` (symlink-resolved).
- **Passback and CPU offload capture: FAILS CLOSED** (rc=1), with the **verbatim observed** error archived in [`refusal-proof.md`](data-2026-10-16-passback-optimization-summary/FINRED/refusal-proof.md) (recovered from the already-observed stdout — **not re-run, not reconstructed**): `load failed: qwen35: memory.offload_exec=passback conflicts with the retained-replay (Redline) backend: the replay tape records GPU launches and does not execute the CPU-executed steps, so a replayed route would compute from stale activations…` (and the identical `cpu` variant). The per-arm daemon-stderr `*-attempt.log` (17/15 loader lines) ends before the refusal; the conflict text arrived on harness stdout. No replacement harness and no universal-gate claim.
- **Scope note (CaptureFrontierScout).** The transfer-guard is verified in `cpu_exec.rs:2004-2007` (`with_cpu_transfer` refuses when `graphs.capture_mode || replay.is_recording()`); full retained-recording non-split path coverage is **reported, not claimed**. The resident test does **not** prove the CPU splice; the changed-path proof is the parent-verified numeric GPU oracles (dispatch-library suite 410 pass/1 ignored; `passback_row_offset_equivalence` and `cpu_offload_return_stream_and_reuse_parity` PASS) plus the serve evidence above.

## 6. Remaining bottlenecks and next measurable work (none implemented)

- **Per-step host/PCIe boundary remains.** The pre-launch D2H + CPU activation transform are still a serial phase; before any event-based launch reordering, a profile of GPU-producer-wait vs CPU-transform is required. Ordering caveat: the input stream's completion currently occurs **before** the GPU-prefix enqueue; moving that **full stream wait** *after* the prefix enqueue would wait for the arm and destroy the overlap. A future early GPU launch would therefore need the input-ready dependency expressed as an **event** completion, not a moved full-stream wait.
- **Per-projection GPU-elapsed aggregation before enlarging a `(dtype,k)` key** is unproven (the `C2` exact-shape key regressed on the old model); a counterfactual share sweep is needed to show the sampled shares (e.g. the traced `m=4096 k=12288` frozen 0.375) are near-optimal.
- **Single-cause attribution of the retained gain is not established.** `C4`'s combined input/pinned + sampling effect and `C5`'s three controller changes move the same path; the one observational trace cannot separate them.
- **Host RAM / PCIe contention and required activation barriers remain**; no CPU or quant-kernel changes were made (user constraint).
- **Not measured here:** cold-start, ≥256-token prefill throughput, other models/quants/GPUs, `HIPFIRE_PROFILE` headline runs, and any XTS/quality admission. (`C6`'s sparser frozen sampling **was** measured — N=4, 20.6 → 20.3 = −1.5 % — and rejected; its full receipts are in its own ledger, so ¼ sampling is kept on that evidence, not for lack of a measurement.)

## Candidate ledgers (linked; not re-run)

- `C1` — [`2026-10-16-passback-timing-interval-experiment.md`](2026-10-16-passback-timing-interval-experiment.md) (+[data](data-2026-10-16-passback-timing-interval-experiment/)).
- `C2` — [`2026-10-16-passback-shape-isolation-experiment.md`](2026-10-16-passback-shape-isolation-experiment.md) and [`…-amendment-1.md`](2026-10-16-passback-shape-isolation-experiment-amendment-1.md) (+[data](data-2026-10-16-passback-shape-isolation-experiment/)).
- `C3` — [`2026-10-16-passback-pinned-copy-worker-experiments.md`](2026-10-16-passback-pinned-copy-worker-experiments.md) (+[data](data-2026-10-16-passback-pinned-copy-worker-experiments/)).
- `C4` — [`2026-10-16-passback-asynchronous-split-returns-gfx1201.md`](2026-10-16-passback-asynchronous-split-returns-gfx1201.md) (+[data](data-2026-10-16-passback-asynchronous-split-returns/)).
- `C5` — [`2026-10-16-passback-sampled-gpu-event-timing-gfx1201.md`](2026-10-16-passback-sampled-gpu-event-timing-gfx1201.md) (+[data](data-2026-10-16-passback-sampled-gpu-event-timing/)).
- `C6` — [`2026-10-16-passback-frozen-sampling-experiment.md`](2026-10-16-passback-frozen-sampling-experiment.md) (+[data](data-2026-10-16-passback-frozen-sampling-experiment/)).

**Historical inquiry (data only — no historical re-runs/bisect):** [`2026-10-02-offload-passback-model-spread-gfx1201.md`](2026-10-02-offload-passback-model-spread-gfx1201.md) + [`…-amendment-1.md`](2026-10-02-offload-passback-model-spread-gfx1201-amendment-1.md) (+[data](data-2026-10-02-offload-passback-model-spread/)). Recorded there: original 9B 16-spill cpu 16.5 / passback 19.9 / pcie 13.2; 27B 8-spill 15.0 / 16.9 / 11.7, 16-spill 9.6 / 11.3 / 6.9. The recorded serve baseline 19.75 is close to the recorded 19.9 but uses a **different method** — no upstream-regression or absence-of-regression claim is drawn. (An initial bad 27B "56 spill 3.3 tok/s" reading was traced RAM thrash, not a baseline.)

## Fixture identity (measured in this host, verified against the receipts)

| artifact | digest |
|---|---|
| branch / base | `perf/passback-opt` at **HEAD `12daf4fe99017a88242469f60853b860609c20ab`** (hipfire 0.4.3); dirty source-only worktree (this campaign); validation base `upstream/beta`; tracking `aldrouil` HTTPS (remote branch absent, no push) |
| FINAL binaries | `final-bin/hipfire` md5 `23181d5ba45c532ec6910b0801ea1492` (47,340,216 B); `final-bin/daemon` md5 `98265d8b809d3b40245f05382b3d9d44` (88,583,272 B) |
| ORIGINAL binaries | `baseline-bin/hipfire-12daf4fe` md5 `c9feb2095ffe043c7e93477a7ec02a9a` (47,336,152 B); `baseline-bin/daemon-12daf4fe` md5 `c1da18f268129005c6e827ea85ceff49` (88,582,088 B) |
| FINAL source delta | `source.patch` 135,827 B, sha256 `d0a5db1f57c83130d9229fa8b3a8a4edba2fcf4abce1362412d4f87b2c47f330`; base `12daf4fe`; offload-split sha `6cb00fb79c456548e2e5e3bf7884227c26eb6989d9ce466282dc27e413267562` |
| toolchain | `rustc 1.98.1 (48a229cea 2026-09-01)`, LLVM 22.1.8, host `x86_64-unknown-linux-gnu`; `cargo build --release` (NOT `-p`) |
| model 9B | `~/.hipfire/models/qwen3.5-9b.mq4`, md5 `296092bf1e6a45d78c1acf815eb93366`, 5,313,750,016 B, 32 layers |
| model 27B | `~/.hipfire/models/qwen3.8-27b.mq4`, md5 `d1292b4d5bd6046693604201a6ca8074`, 15,662,615,552 B, 64 layers |
| prompt | `benchmarks/prompts/gpu_offload_probe.txt`, md5 `5835c71e471849b4a72e1dc8e39695e7`, 215 B (59 tokens) |
| route envs | `HIPFIRE_OFFLOAD_EXEC=<passback\|cpu\|pcie>`, `HIPFIRE_GPU_LAYER_BUDGET=16` (9B) / `56` (27B), `HIPFIRE_OFFLOAD_PASSBACK_SHARE=auto` (passback arms only), `HIPFIRE_DPM_WARMUP_SECS=10` |

Host: 1× Radeon RX 9070 XT `gfx1201` (16,304 MiB), ROCm/HIP 7.2, Ryzen 7 7800X3D (8c/16t), ~28 GB RAM.

## Caveats

- **Short prompt.** 59 prompt tokens < 256 ⇒ `prefill_tok_s` is launch overhead (JSON warning preserved); only `decode_tok_s`/`wall_tok_s`/`ttft_ms` are cited. No prompt normalization.
- **Warm, not cold.** `--warmups 3` precedes the 3 measured runs; no cold-start claim.
- **N=3 independent processes per arm.** The 3 internal measured runs inside a process are **not** independent samples and are never pooled; pooled/`sd` figures across the whole campaign's blocks must not be read as 9 independent samples.
- **No statistical-confidence claim** beyond the descriptive range comparison.
- **27B is a real target but not canonical XTS** — no quality pin / admission.
- **Serve and native instruments are never averaged**; serve harness rates are development probes only.
- **No Redline timed-arm route proof** — manual shadow/parity is discovery/correctness evidence only; the resident capture does not prove the CPU splice.

## Raw evidence

**129 files** in [`data-2026-10-16-passback-optimization-summary/`](data-2026-10-16-passback-optimization-summary/) (128 receipts + `SHA256SUMS`), each `cmp`-verified byte-identical to its source:

- `FIN9B/` — 12 native samples × (`*.json` incl. `samples` arrays, `*.meta.log`, `*.stderr.log`, `*.daemonproof.log`) = 48 receipts.
- `FIN27B/` — 12 native samples × 4 = 48 receipts.
- `FINCORR/` — 6 correctness JSONs + 6 serve logs (`9B/27B battery+chain`, `ident-share0`/`ident-cpu`).
- `FINRED/` — `resident-pcie32.json/log` (N1 kept, superseded), `resident-pcie32-multiposition.json/log` (N3 parity PASS), `passback-attempt.log`, `cpu-attempt.log`, and `refusal-proof.md` (verbatim stdout + exact commands).
- Runners: `FIN-one.sh`, `FIN-9B.sh`, `FIN-27B.sh`, `FIN-correct.sh`, `FIN-redline.sh`, `FIN-analyze.py`; `FIN-power-pre.txt` (loader banner); `power_snap.sh`; `gpu_offload_probe.txt`; `warm3x_gpu_offload_probe.json`.
- Provenance (text only, **no executables**): `01-final-source.patch`, `01-final-source-provenance.json`, `00-binaries-and-toolchain.txt`.
- `SHA256SUMS` — integrity digests of all 128 receipts.
