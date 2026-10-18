# Dense Qwen3.5 CPU offload: pinned asynchronous transfers (gfx1201)

**Lifecycle:** historical.  
**Date:** 2026-10-18.  
**Authority:** fixture-bound measured evidence, not a default, product floor, admission, or another model's baseline.  
**Disposition:** the combined pinned-staging / batched-read / deferred-return change improves this deliberately spilled 9B fixture. No arithmetic/transfer overlap gain is claimed.

## Implementation and ownership

Code commit: `b274a9852` (`perf(cpu): route dense CPU steps via pinned async transfer lease`). Only `crates/hipfire-dispatch/src/cpu_exec.rs` changes production behavior.

Dense `memory.offload_exec=cpu` steps reuse `hip_bridge::PinnedTransferBuffer`. One lease holds activation, optional AWQ scale, optional accumulator, and output. D2H copies use the actual producer/consumer stream, followed by the owner's reusable input-ready event. CPU transforms and GEMV wait that event, then operate directly in pinned slices: no intermediate activation/result vectors or pageable upload source. Residual addition retains the original order and destination.

The H2D return is only enqueued, not immediately host-waited. Same-stream GPU consumers are ordered after it; another event wait on that same stream would be redundant. The owner stays pending. The next D2H batch queues behind the upload and its input-ready event retires both before host reuse. Growth, stream/device changes, errors and teardown retain the existing drain-or-retain safety policy. No auxiliary stream is introduced.

`pcie`, `passback` (including its synchronous share-zero fallback), quant arithmetic, layer placement and multi-token GPU GEMM routes are unchanged. `Gpu::memcpy_htod_auto` is deliberately unchanged: globally making temporary pageable-slice callers asynchronous would not establish their lifetime safety. Dense CPU transfers bypass it through the owned pinned API instead.

## Identity

Both arms were fresh release builds, not the stale binaries initially present. A was built from an isolated original worktree; B was built before the code commit from the same source plus the CPU-exec patch. The commit records the implemented source, not a claim that these binary build-version strings name that later commit.

| Item | Identity |
|---|---|
| Original source A | `c3849a867e5a4ed04cd425d7770939901b9c6168` |
| Candidate source B | A plus the `cpu_exec.rs` change committed as `b274a9852`; recorded patch SHA-256 `6adddeac5dd99192a1f5bd85199454b52bda437d486dff1609a1775b574df448` |
| A CLI / daemon MD5 | `1bc5c7a40ce1f2278f8daea92f3a670f` / `2cd58df46817b243bdddb556e87ba917` |
| B CLI / daemon MD5 | `84b69854db03fef4abb3153c5c7af25a` / `db93549222893d5ac5cffa732639571f` |
| Model | `/home/avery/.hipfire/models/qwen3.5-9b.mq4`, 5,313,750,016 bytes; MD5 `296092bf1e6a45d78c1acf815eb93366` |
| Stateless prompt | `benchmarks/prompts/gpu_offload_probe.txt`, 215 bytes / 59 prompt tokens; MD5 `5835c71e471849b4a72e1dc8e39695e7` |
| Session fixture | `benchmarks/prompts/session_coding.json`; MD5 `942c1c627af7fde0e4b9b88b53bb4436` |
| GPU | RX 9070 XT, `gfx1201`, PCI `0000:03:00.0` |
| Runtime / driver | HIP 7.2; Linux `7.2.10-1-cachyos` |
| Placement | `HIPFIRE_GPU_LAYER_BUDGET=16`: 16 of 32 layers spilled; 16/16 spilled layers covered by CPU seam in both arms |
| Execution | `HIPFIRE_OFFLOAD_EXEC=cpu`, speculation off, greedy, thinking off, q8 VMM KV, Q8 DeltaNet, `max_seq=4096` |
| Process policy | isolated configuration/home per fresh process; DPM warmup 10 seconds; no simultaneous benchmark arms |

Full native JSON reports, including their `samples`, load acknowledgments and available power/memory telemetry, are preserved in [the data directory](data-2026-10-18-dense-cpu-async-gfx1201/). Several GPU clock/power/throttle fields report `unavailable`; do not infer clock equality from them. The comparison uses interleaving and observed sample spread, not an assumed universal noise band. This is an intentional manual-spill configuration, not default-pull performance.

## Unprofiled throughput

Stateless order: **A B B A B A A B**. Four fresh processes per arm; each process discards three warmups and reports three measured runs. Headline values are medians of the four process medians, not profiled timing.

```sh
HIPFIRE_LOCAL=1 HIPFIRE_NO_REGISTRY_FETCH=1 \
HIPFIRE_DPM_WARMUP_SECS=10 \
HIPFIRE_OFFLOAD_EXEC=cpu HIPFIRE_GPU_LAYER_BUDGET=16 \
HIPFIRE_DAEMON_BIN=<matching-archived-daemon> \
<matching-archived-hipfire> bench /home/avery/.hipfire/models/qwen3.5-9b.mq4 \
  --runs 3 --warmups 3 --spec off --backend noslots --workload stateless \
  --max-tokens 128 --max-seq 4096 --kv-mode auto --kv-backend vmm \
  --prompt-file benchmarks/prompts/gpu_offload_probe.txt --json
```

| Metric | A synchronous | B asynchronous | Change |
|---|---:|---:|---:|
| Decode tok/s, process medians | 16.5 / 16.2 / 16.7 / 16.6 | 18.4 / 18.3 / 18.3 / 17.8 | — |
| Decode tok/s, median | 16.55 | 18.30 | +10.6% |
| End-to-end wall tok/s, median | 15.35 | 16.90 | +10.1% |
| Eight-turn session wall, median | 78.4 s | 73.5 s | −6.2% |

Stateless decode ranges are 16.2–16.7 and 17.8–18.4 tok/s: the 1.75 tok/s median gap exceeds either arm's observed spread, and B wins all four interleaved pairs. Prefill medians on the short probe were 98.7 and 99.2 tok/s; **59 tokens is a launch-overhead probe, not a prefill-throughput claim**. TTFT ranges overlap (A 586–604 ms; B 587–600 ms).

Resident-session order: **A B B A A B A B**, four fresh processes per arm. Each process runs the same eight-turn fixture:

```sh
HIPFIRE_NO_REGISTRY_FETCH=1 HIPFIRE_DPM_WARMUP_SECS=10 \
HIPFIRE_OFFLOAD_EXEC=cpu HIPFIRE_GPU_LAYER_BUDGET=16 \
HIPFIRE_CLI_BIN=<matching-archived-hipfire> \
HIPFIRE_DAEMON_BIN=<matching-archived-daemon> \
python3 scripts/serve_harness.py \
  --model /home/avery/.hipfire/models/qwen3.5-9b.mq4 \
  --mode session --session benchmarks/prompts/session_coding.json \
  --sampling greedy --thinking off --max-tokens 128 --max-seq 4096 \
  --kv-backend vmm --home <fresh-isolated-home> \
  --serve-log <sample.log> --out <sample.json>
```

The cold first-turn decode median is 8.1 → 8.4 tok/s; turns 2–8 improve roughly 9–11%. B wins all 32 matched turn pairs. Session wall ranges are A 78.39–78.49 s and B 73.32–73.85 s. An earlier single candidate session had slower late turns; this was **not reproduced** by the interleaved four-per-arm campaign. Its cause was not established, so it is excluded rather than assigned a speculative thermal/configuration cause.

All eight reports produce identical per-turn text, finish reasons and generated counts; combined decoded-text MD5 is `3086356dd7ae8f8c04df7d48a6a512ed`. The outputs were read: coherent Rust deduplication discussion/code and the requested haiku, with no empty output or token attractor. Six turns hit the 128-token length budget and carry the harness's corresponding health flags. **The session harness returns 1 in both arms:** the final summary omits the expected literal words `rayon` and `chunk` (recall 1/3). This is identical baseline behavior, not a newly passing session acceptance gate.

## Transfer latency, synchronization and overlap

These are **attribution-only profiled runs**, not additional headline throughput samples. The successful route was `rocprofv3` launch mode with profiler environment inherited by the GPU-owning daemon:

```sh
rocprofv3 --hip-runtime-trace --memory-copy-trace --output-format csv \
  -d <arm-trace-dir> -o trace -- \
  python3 <native-protocol-driver> <matching-archived-daemon> <arm-tag>
```

The driver preserves `os.environ`, then applies the same CPU/16-layer/q8-VMM configuration, sends three 64-token warmups and one 64-token diagnostic generation. Its protocol requests use greedy sampling, closed-think assistant prefix, the exact probe prompt and `max_seq=4096`. Dropping profiler variables with a minimal child environment had previously produced empty traces. A handle-`dlsym` interposer passed a lookup probe but failed real VMM model load even as a passthrough; no interposer is used for the results below. Original and candidate native loads remain successful without it.

The benchmark runner's final 64-token decode-window attribution:

| Boundary | A synchronous | B asynchronous |
|---|---:|---:|
| Selected copy/sync host API wall per token, summed | 14.66 ms | 10.48 ms |
| Copy API wall, median | `hipMemcpy` 28.5 µs (p95 65.8 µs) | `hipMemcpyAsync` enqueue 2.6 µs |
| Copy API calls per token | 282.7 synchronous | 279.7 asynchronous + 3.0 remaining synchronous |
| Input-ready event record / wait per token | none at this seam | 123.9 / 123.9 |
| Event-record API wall, median | — | 2.1 µs |
| Input-ready event wait wall, median | — | 33.5 µs (p95 55.8 µs) |
| Profiler-visible device D2H span, median | 5.76 µs (5.3–10.5 µs range) | 4.44 µs (4.1–5.9 µs range) |
| Profiler-visible device H2D span, median | 5.44 µs (4.5–10.8 µs range) | 5.84 µs (4.8–12.4 µs range) |
| Device-wide synchronization during selected decode | 0 | 0 |

A's completed copy API wall includes blocking/producer wait. B's enqueue wall is not DMA latency; its input-ready wait includes producer queue delay and transfer completion, not just event-call overhead. The remaining three synchronous copies per token are outside the converted dense CPU seam. The new path batches input copies and eliminates the immediate H2D host wait; it does not eliminate the necessary D2H read boundary. Existing unprofiled host StepTiming corroborates return-wall reduction (roughly 30–70 µs completed return → 0–10 µs submission), but rounded `0.00 ms` never means zero DMA latency.

**Coverage limit:** the device memory-copy trace reports only 1,024 D2H and 3,069 H2D spans in each selected window, fewer than the HIP copy API calls. These are the profiler-visible copy implementation's spans, not proof that every transfer implementation is covered. Kernel tracing was disabled. Do not call the sum of visible DMA time total transfer time, or assign an exact percentage of the throughput gain to it. There was no ablation separating removed waits, batched D2H boundaries and eliminated temporary allocations/copies.

**Measured CPU/DMA overlap proxy:** intersecting the final 7,930 same-thread input-event-return → next-copy-submit host intervals with all profiler-visible candidate DMA spans yields **zero overlapping nanoseconds** across 2.707 seconds of those host regions. These regions cover CPU transform/GEMV/residual work plus host overhead, not an isolated GEMV timer. The [receipt](data-2026-10-18-dense-cpu-async-gfx1201/cpu-dma-overlap.json) states the exact selection and visibility limit. All reported copies use stream 0. **[INFERENCE]** The same-stream dependency chain cannot overlap a GEMV with its own input D2H, nor overlap its in-stream GPU consumers with an unfinished upload. Absolute GPU-kernel/CPU or kernel/DMA overlap was not measured without kernel dispatch tracing. Deferred H2D preserves host-dispatch opportunity; no positive arithmetic-overlap claim is made.

Compressed complete HIP API and memory-copy CSVs are retained in the data directory as `profile-{A,B}-{hip-api,memory-copy}.csv.gz`, alongside diagnostic daemon `done` logs. Profiler overhead is excluded from the unprofiled A/B conclusion.

## Numerical correctness and lifetime checks

- `cargo test -p hipfire-dispatch --lib`: **410 passed, 1 ignored**.
- Release GPU oracles `cpu_offload_return_stream_and_reuse_parity` and `pinned_input_ready_marker_semantics`: **both passed**. Coverage includes raw/prerotated AWQ handling, residual chaining, immediate GPU consumers, default and non-blocking streams, stream changes and staging growth/shrink reuse.
- Existing `resident_lowered_passback_hw` synchronous-twin case: **444 tensors byte-exact**, including all 12 logits rows and full valid KV/DeltaNet state. Both arms execute 1,488 CPU steps and zero host-mapped GPU fallback steps. CPU mode is compared to the unchanged `passback` share-zero synchronous CPU implementation, not to the numerically different PCIe engine.

```sh
HIPFIRE_RESIDENT_ORACLE_TWIN=1 HIPFIRE_RESIDENT_ORACLE_CASES=4 \
HIPFIRE_RESIDENT_ORACLE_MODEL=/home/avery/.hipfire/models/qwen3.5-9b.mq4 \
HIPFIRE_RESIDENT_ORACLE_STEPS=12 HIPFIRE_RESIDENT_ORACLE_KV_MODE=q8 \
HIPFIRE_RESIDENT_ORACLE_OUT=<new-output-dir> \
cargo test --release -p hipfire-arch-qwen35 --test resident_lowered_passback_hw \
  resident_lowered_passback_oracle -- --exact --ignored --nocapture --test-threads=1
```

- Multi-row proof used a throwaway copy of `mq4_prefill_state_oracle`, **not its unmodified fully resident load**: `Layout::single(...).with_placement(Placement::dense_prefix(n_layers, 16))`, the normal host-CPU-weight gate, and an assertion that continuation increases actual CPU-step count. The same model/prompt/Q8 VMM state is run under `cpu` versus `passback` share zero, with lowered execution off in both; `HIPFIRE_DN_STATE_EF=1`. After **2 rows + 4 decode steps**, all 230 manifest records / **87,101,440 bytes** match exactly. After **513 rows + 4 decode steps**, all 230 records / **104,892,416 bytes** match exactly. Comparisons include prefill and continued logits, state codes/scales/EF, convolution and valid KV, not only generated text or timing. The temporary source/driver were removed after proof.
- Oracle binary MD5s: state twin `9316d52f63b09b9ad8f7922f2f47d5a2`; placed multi-row oracle `7892fbe35ef6f9ed132ffc00ee72726e`. Exact commands and comparison receipts are in `numeric-results.json`; twin output is in `state-twin.log`.
- The Redline daemon harness was exercised with the actual CPU/16-layer configuration and **refused the load as designed**: CPU-executed steps cannot be retained/replayed as a GPU-only tape. `redline-refusal.log` preserves that result. This is fail-closed evidence, **not a PM4 parity pass or Redline certification**. No resident-only route was substituted for the changed CPU route.
- `bash scripts/pre-commit-ci.sh origin/master`: **all checks passed**, including ratchets, crate maps, scratch growth, changelog, cargo-deny bans/licenses/sources, rustfmt, workspace/all-target clippy, env docs and lifecycle. `origin/master` is this fork's available base; there is no `origin/beta` ref.

## Evidence boundary

The sibling data directory preserves all eight complete stateless JSON reports and all eight resident-session reports, profiler CSVs, overlap receipt, numerical commands/results and refusal/twin logs. Large raw numerical-state dumps and archived executables remain local under `.scratch/dense-cpu-async/`; those local paths are discovery pointers, not published Git objects. No 27B, other-GPU, default-placement, sampled/speculative or concurrent-serving performance claim is made.
