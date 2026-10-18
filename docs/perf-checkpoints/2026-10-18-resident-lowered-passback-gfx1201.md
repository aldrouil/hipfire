# Resident-only lowered Qwen3.5 dense passback — correctness and native A/B

**Lifecycle:** `historical`. **Disposition:** implemented and numerically validated on the stated 9B fixture; **no resolved end-to-end throughput gain** under the predeclared comparison rule. No performance-based promotion or resident-execution latency reduction is established. The implementation remains uncommitted for review. The 27B gate was not met, so no 27B campaign or additional attribution campaign was run. Stop after reporting.

Measurement: **2026-10-18 00:55:03–01:00:56 UTC**. The [evidence directory](data-2026-10-17-resident-lowered-gfx1201/) retains the authorization-date prefix, 2026-10-17. Earlier Phase 3 checkpoints are unchanged.

## Narrow implementation

`crates/hipfire-arch-qwen35/src/qwen35/forward.rs` retains the existing whole-model lowered admission and the hand layer loop. Within that loop, dense `passback` models with an auto/positive share may execute one existing lowered layer program when every projection weight of that layer is device-resident. Allocation-free residency screening occurs before program construction; spilled layers retain their existing hand/passback arms. `run_layer_program` keeps its independent host-mapped-weight refusal.

Absolute KV layer indices, compact DeltaNet indices, residual/scratch ownership, layer-local fusion flags, active-stream ordering, pending pinned CPU returns, per-layer epilogues, and final norm/head execution are retained. No new synchronization is added. CPU mode, share-zero CPU fallback, hidden-state extraction, mrope, MoE and graph/retained-replay restrictions are unchanged. `HIPFIRE_FORWARD_LOWERED=0` is the all-hand escape/reference. No GEMV kernel, scheduler, quantization, program caching or new production knob was changed.

Existing lowered programs can fuse convolution/QK normalization and use compact Q8 recurrence without materializing repeated Q/K. Other optimizations are architecture/shape-gated; this is not a new kernel campaign. The implementation does not imply that all resident-layer time can be removed or that 9B and 27B have different architectures.

## Correctness before timing

New ignored hardware test: `crates/hipfire-arch-qwen35/tests/resident_lowered_passback_hw.rs`. It reuses the existing HFQ/model placement, `forward_scratch`, VMM KV, DeltaNet state and transfer infrastructure. Arms run in fresh child processes because route/share configuration is process-cached. `HIPFIRE_FORWARD_ORACLE` was verified to be comment/documentation-only and was not used.

Each primary case feeds **12 identical teacher-forced tokens** through candidate mixed and candidate all-hand forwards with identical weights, placement and fixed row shares. Every logits row is compared. Full valid state is compared initially and after **3, 6 and 12 tokens**: KV keys/values including Q8 block scales; all DeltaNet recurrent codes/scales, convolution histories and f16 error-feedback planes. Initial state bytes are checked; unused KV capacity is not compared. Non-finite data fails, and **byte equality**, not an adjusted tolerance, determines the result. Dumps occur only after complete tokens, never between layers.

| Case | First resident layer | Fixed GPU share | Active stream | Result |
|---|---|---:|---|---|
| 16 spilled / 16 resident | DeltaNet | 0.375 | Default | 444 tensors byte-exact |
| 15 spilled / 17 resident | Full attention | 0.375 | Default | 444 tensors byte-exact |
| 16 spilled / 16 resident | DeltaNet | 0.5 | Default | 444 tensors byte-exact |
| 16 spilled / 16 resident | DeltaNet | 0.375 | Non-blocking | 444 tensors byte-exact |
| CPU versus passback share-zero, lowering enabled on passback | DeltaNet boundary | 0 | Default | 444 tensors byte-exact |

Actual projection ownership matched the spill prefix. Both sampled joins and unsampled queued CPU-tail returns were exercised. Diagnostic launch counters, kept as metadata rather than permanent numerical assertions, showed **24 fewer launches per token** in every mixed primary case: 528→504 at cut 16 and 524→500 at cut 15. This proves execution-path engagement, not a kernel-time or throughput speedup.

The release build and oracle completed successfully after fixing a test-only non-`Copy` stream-handle ownership compile error; that failed build log is retained. A separate native candidate-daemon serve smoke emitted a coherent, nonempty 331-token answer and stopped normally. That smoke had thinking enabled; it is **not** the timed thinking-off fixture or a universal auto-share output-equivalence claim. Timed arms below used thinking off.

Reproduction entry point, with the stated model/GPU prerequisites:

```sh
HIPFIRE_RESIDENT_ORACLE_TWIN=1 cargo test -p hipfire-arch-qwen35 --release \
  --test resident_lowered_passback_hw -- --ignored --nocapture
```

[Correctness receipts](data-2026-10-17-resident-lowered-gfx1201/CORRECTNESS-summary.txt), full oracle logs, per-arm manifests and compressed raw state dumps are preserved. Environment documentation validation (`python3 scripts/check-env-docs.py`) passed. No full workspace CI result is claimed.

## Exact control and bounded performance protocol

- GPU/host: RX 9070 XT, gfx1201; Ryzen 7 7800X3D; HIP `7.2.53211-3d9ef42`, Rust/Cargo 1.98.1.
- Model: `/home/avery/.hipfire/models/qwen3.5-9b.mq4`, MD5 `296092bf1e6a45d78c1acf815eb93366`, 5,313,750,016 bytes. The model fits in VRAM; the **16-layer spill is deliberate**.
- Prompt: committed `benchmarks/prompts/gpu_offload_probe.txt`, MD5 `5835c71e471849b4a72e1dc8e39695e7`, 215 bytes / 59 tokens. No short-prompt prefill performance claim.
- Control **C**: exact validated Phase 3 FINAL2, base `b236099f64779b0291f5b48ff4ec0aa68d872b40` + input-ready patch SHA-256 `f855d7b92c919e454ee4c364f903e3f762d7ede264b48c80c2f6bc45639d30e0`. CLI MD5 `72ab7a028cf4dfe4adb6638741b84a83`; daemon `fb71e7d781a26671c8d8cb5ae6885e50`. Source/binaries were not modified.
- Candidate **K**: isolated exact-FINAL2 copy + **only** the forward feature patch, SHA-256 `d46df385d1e324d3beef4ae18d1bbfbc5c5fa219a34aa67ced0134b591904b1d`. CLI MD5 `a11eb0a0488b8aada0832220c1a0d935`; daemon `16c8abdd03c589c436fd46fd072eaa0e`. The live checkout's formatted FINAL3 input-ready files were not substituted for the control or candidate base.
- Native `hipfire bench` generation protocol, `--backend noslots --workload stateless --spec off`, thinking off (`--reasoning-on` absent), q8 KV/VMM, max_seq 4096, max_tokens 128, **share=auto/unset**, Rayon unset, isolated configuration homes, registry fetch disabled and DPM warmup 10 seconds. Trace/profile switches off.
- **Eight fresh CLI/daemon processes**, four per arm, each with three warmups and three measured requests. Serialized order: **C1 K1 K2 C2 | K3 C3 C4 K4** (ABBA then BAAB). Analysis unit is each process's median, not pooled requests or best samples. Every process returned zero; hashes, prompt, KV and speculation settings were checked per process. Power/memory snapshots and full sample arrays are retained; throttle residency was unavailable.

## Process-level throughput

Values are tok/s. Process indices identify each arm's samples, not simultaneous pairs; temporal order is specified above.

| Process index | C decode | K decode | C wall | K wall |
|---|---:|---:|---:|---:|
| 1 | 20.5 | 20.5 | 18.7 | 18.6 |
| 2 | 20.4 | 20.6 | 18.5 | 18.7 |
| 3 | 20.4 | 20.8 | 18.6 | 18.9 |
| 4 | 21.0 | 21.0 | 19.0 | 19.1 |

| Metric | C median [min–max] | K median [min–max] | Descriptive median delta |
|---|---:|---:|---:|
| Decode | 20.45 [20.4–21.0] | 20.70 [20.5–21.0] | +1.22% |
| Native wall throughput | 18.65 [18.5–19.0] | 18.80 [18.6–19.1] | +0.80% |

**No resolved gain.** The decode gap is 0.25 tok/s against arm ranges of 0.60 and 0.50; the wall gap is 0.15 against ranges of 0.50 each. Both blocks favor the candidate in direction, but neither clears contemporaneous spread: the first block's decode gap equals its arm-pair ranges, while the second is smaller than its control range. Both arms also rose between blocks; they did **not** have identical shifts. The corrected [analysis](data-2026-10-17-resident-lowered-gfx1201/AB-ANALYSIS.txt) records those shifts separately.

This fails the [predeclared rule](data-2026-10-17-resident-lowered-gfx1201/PROTOCOL-predeclared.txt) requiring a reproducible improvement beyond spread in both blocks. It is not proof of equivalence, a regression, or statistical significance. Rates are reported at native output precision; tiny deltas should not be overinterpreted.

## Attribution, limits and stop point

Fewer launches and byte-exact state are demonstrated on the fixed-share correctness fixture. **Reduced resident-layer execution time is not measured**, and the small positive native medians cannot be causally assigned to resident kernel time, host dispatch, auto-controller observations or session drift. Per protocol, no extra profiling campaign was launched after the throughput gate failed.

The 27B validation condition was therefore **not met**. No 27B performance or state validation is claimed for this change. Same dense architecture does not make these 9B measurements transferable to the larger shape or other GPUs. Coverage is one model/placement family, one prompt, one host/GPU and a bounded four-process-per-arm comparison; oracle coverage is 12 teacher-forced steps at the stated fixed shares/streams, not arbitrary contexts or all configurations.

[Evidence bundle](data-2026-10-17-resident-lowered-gfx1201/) includes all process JSON/sample arrays, metadata, daemon proofs, power snapshots, commands, source/build provenance, the feature patch, oracle manifests/logs and SHA-256-protected `ORACLE-dumps-{run1,twin}.tar.zst` raw state archives. Early compile failures and the corrected analysis wording are retained transparently.

**Stop:** implementation and evaluation delivered; no further fusion, caching, kernel, scheduling, or unrelated optimization pursued. No commit or deployment performed.
