# Passback Phase 3, Step 2 — trace-off throughput

**Lifecycle:** `historical`. **Disposition:** recommend retaining the exact Step 1 input-ready boundary for its modest, fixture-bound net throughput benefit. No scheduler change, optimal-share claim, product admission, or broader performance claim. Stop after this comparison.

The boundary improves both models at fixed share 0.45 in both ordering blocks: decode +4.0% on 9B and +1.8% on 27B; wall throughput +3.8%/+1.8%. These gaps exceed their observed process-median spreads. Other cells are positive but not uniformly decisive. In particular, 9B auto is confounded by a substantial common block shift. The best observed setting remains 0.375 for both implementations and models on this limited grid; there is no measured shift toward a larger GPU share.

## Method and identity

Follow-up to [Step 1](2026-10-17-passback-phase3-input-ready-gfx1201.md). [Raw evidence](data-2026-10-17-passback-phase3-throughput-gfx1201/) includes every valid process JSON, stderr, timestamp/config/hash metadata, daemon-wrapper proof, orchestration scripts, build provenance, and process analysis. Read `P3T-analysis-amend1.txt` together with `P3T-analysis.txt`: the amendment retracts the original 15/16 block-sign count and its overbroad application of 9B-auto variation to other cells.

- RX 9070 XT, gfx1201, PCI `0000:03:00.0`; HIP `7.2.53211-3d9ef42`, rustc `1.98.1`.
- A: retained C4+C5 source `a5a87e6cd`; production `crates/` diff against Step 1 base `b236099f64779b0291f5b48ff4ec0aa68d872b40` is empty. CLI MD5 `3413c7bdfe10a1b79abbe63d4f3db452`, daemon `f6cb5fec835e6fede7bbddf28ff5b828`.
- B: isolated worktree `/home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-final2`, base `b236099f64779b0291f5b48ff4ec0aa68d872b40` plus archived `FINAL2-production.patch`, applied without rejects. Applied four-file diff SHA-256 exactly `f855d7b92c919e454ee4c364f903e3f762d7ede264b48c80c2f6bc45639d30e0`. Built with `cargo build --release`. CLI MD5 `72ab7a028cf4dfe4adb6638741b84a83`; daemon `fb71e7d781a26671c8d8cb5ae6885e50`, byte-identical to Step 1. Fresh CLI differs from Step 1's CLI; cause not established. Each arm used its own verified CLI/daemon pair consistently.
- The live checkout/binaries matched FINAL3, not FINAL2. User selected rebuilding exact FINAL2 in isolation. Earlier invalid FINAL3 partials are preserved under `QUARANTINE-FINAL3ARM/` and excluded completely. Live production source was not modified.
- 9B: `/home/avery/.hipfire/models/qwen3.5-9b.mq4`, MD5 `296092bf1e6a45d78c1acf815eb93366`; 16 resident/16 spilled layers; q8 KV, VMM.
- 27B: `/home/avery/.hipfire/models/qwen3.8-27b.mq4`, MD5 `d1292b4d5bd6046693604201a6ca8074`; 56 resident/8 spilled layers; fp8 KV, VMM. This is the Phase 2 custom fixture, not the canonical MQ4XTS quality pin.
- Prompt: `benchmarks/prompts/gpu_offload_probe.txt`, MD5 `5835c71e471849b4a72e1dc8e39695e7`, 215 bytes/59 tokens. Short-prompt prefill numbers are not used for a prefill claim.
- 64 valid fresh CLI/daemon processes: two models × four share settings × two implementations × four processes. Each cell uses ABBA followed by BAAB, serialized GPU execution. Three warmups and three measured requests per process; analysis unit is each process's median, not the twelve pooled requests. An additional candidate native benchmark smoke completed successfully. All valid processes returned zero.
- Greedy; thinking off through native bench's default (no `--reasoning-on`); all speculation explicitly off. `HIPFIRE_LOCAL=1`, `HIPFIRE_NO_REGISTRY_FETCH=1`, `HIPFIRE_DPM_WARMUP_SECS=10`, `HIPFIRE_OFFLOAD_EXEC=passback`, `HIPFIRE_GPU_LAYER_BUDGET=16|56`. `HIPFIRE_OFFLOAD_PASSBACK_SHARE=0.5|0.375|0.45`, or unset for auto. Isolated per-process config homes, Rayon unset, diagnostic/profile/trace controls off, graph/replay overrides unset. Scripts and per-process metadata retain exact invocations and environment policy.

Native command, using each arm's verified CLI and daemon override:

```sh
hipfire bench MODEL --runs 3 --warmups 3 --spec off \
  --backend noslots --workload stateless --max-tokens 128 --max-seq 4096 \
  --kv-mode auto --kv-backend vmm \
  --prompt-file benchmarks/prompts/gpu_offload_probe.txt --json
```

Predeclared comparison: median of four process medians; inspect process min/max and both ordering blocks. A delta smaller than the cell's observed spread is inconclusive. Retention requires a reproducible benefit in both model buckets, not a best-of-one point. This is a descriptive screen, not a confidence interval or statistical significance test.

## Results

Decode columns are **median [minimum–maximum]** of four process medians, tok/s. Wall delta is the change in median native benchmark `wall_tok_s` (UX-level throughput, not an isolated kernel metric).

| Model | GPU share | C4+C5 decode | Input-ready decode | Decode delta | Wall delta |
|---|---|---:|---:|---:|---:|
| 9B | 0.5 | 17.15 [16.9–17.4] | 17.95 [17.7–19.2] | +4.7% | +4.4% |
| 9B | auto | 20.25 [18.9–21.4] | 20.95 [19.3–22.0] | +3.5% | +3.2% |
| 9B | 0.375 | 21.25 [21.0–21.3] | 21.55 [21.5–21.7] | +1.4% | +1.3% |
| 9B | 0.45 | 20.10 [20.1–20.1] | 20.90 [20.9–20.9] | +4.0% | +3.8% |
| 27B | 0.5 | 15.85 [15.8–16.2] | 16.25 [16.1–16.4] | +2.5% | +2.6% |
| 27B | auto | 17.30 [17.3–17.4] | 17.60 [17.5–17.6] | +1.7% | +1.2% |
| 27B | 0.375 | 17.65 [17.6–17.7] | 17.90 [17.8–17.9] | +1.4% | +0.9% |
| 27B | 0.45 | 16.80 [16.8–16.8] | 17.10 [17.1–17.1] | +1.8% | +1.8% |

All eight cell medians and all sixteen within-cell block-mean comparisons favor B. Zero displayed range means identical one-decimal process medians, not zero physical noise; individual request samples still vary. Raw values and ordering are preserved rather than selecting the best process.

- **Same-share benefit:** strongest cross-model evidence at 0.45, consistent across both blocks. 9B 0.5 has a candidate 19.2 outlier; its 0.8 median gap is smaller than the candidate's 1.5 range. The 27B 0.5 gap equals A's 0.4 range; not decisive alone. At 0.375, the 9B gap equals A's 0.3 range, while 27B's 0.25 gap exceeds both 0.1 ranges.
- **Share preference:** 0.375 has the highest observed median for every model/implementation on the tested grid. This does not establish a global optimum: share settings were measured in separate time blocks. Nothing here supports increasing GPU share because execution starts earlier.
- **Automatic controller:** 27B auto shows a small repeatable net gain; 9B auto's 0.7 arm gap is smaller than a common roughly 2.4 tok/s block shift. Its apparent improvement is inconclusive. Selected auto shares are not exposed in trace-off benchmark JSON, so controller decisions/adaptation were not observed. Auto remains below the observed 0.375 medians in both arms; this is not controller-optimality evidence.
- **Event overhead:** net gains at the stable fixed-share cells show the complete boundary's extra event work does not erase its benefit there. Event record/fence/wait cost and overlap gain were not separately measured. No specific causal share of the change is assigned to event overhead, overlap, contention, or host scheduling.

## Limits and recommendation

Power/clock/memory snapshots and native power objects are retained. `throttle_pct` is unavailable; absence of stderr errors does not prove no throttling. There is no universal noise bound. One prompt, two custom partial-offload fixtures, one GPU/session and four processes per arm/cell limit generality.

Native benchmark JSON carries no decoded text. This campaign did not inspect new decoded output; it reuses Step 1's decoded serving and numerical evidence for the exact source, including Step 1's documented baseline-recurring 27B empty chain turn. Extensive numerical/profiler validation was not rerun; no production patch or instrumentation was changed. This is throughput evidence, not a new correctness certification.

**Recommend retain**, scoped to the validated boundary: both models show reproducible modest decode and wall gains at 0.45, without a measured negative cell. Do not market the noisier +4.7%/+3.5% headline estimates as established gains. Do not change scheduler policy or infer a larger optimal GPU share. No additional optimization campaign or scheduling modification was begun.
