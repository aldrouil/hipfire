# Ordered MQ4V2 AVX2 experiments — 2026-10-13

**Lifecycle:** historical. Fixture-bound measured evidence, not admission or a product floor. Supplements, without modifying, [the earlier dense CPU checkpoint](2026-10-13-dense-cpu-simd-zen4.md).

## Scope and method

User requested three experiments in order: exact per-half activation-sum reuse, specialized AVX2 nibble unpacking, interleaved half-group computation. Each was independently implemented, validated, benchmarked, and reverted before the next. No AVX512, activation requantization, weight layout, GPU synchronization, or row scheduling changes.

Incumbent source: `bab82f0458915dd2da4e3442cb39a95e960b7b77`, retained exact MQ4/MQ6 sums and64-row grain from the earlier campaign. Incumbent archived binary MD5s: CLI `c47e36394c107adaa14d2778ee85ed4a`, daemon `3d80e29f578ddc7937babbb927caa4f0`.

Primary fixture: local Qwen3.8-27B MQ4, qt44 MQ4V2, MD5 `d1292b4d5bd6046693604201a6ca8074` (not canonical XTS). Ryzen7800X3D8c/16t, RX9070XTgfx1201, HIP7.2. `HIPFIRE_GPU_LAYER_BUDGET=48` (16 of64 layers onCPU), `HIPFIRE_OFFLOAD_EXEC=cpu`, Rayon16, device0, graph/speculation off, HIP replay, local native daemon. Native `hipfire bench --backend noslots --workload stateless --spec off --runs 5 --warmups 2 --max-tokens 64 --kv-mode q8 --max-seq 512 --json --prompt-file PATH`.

Each candidate: three fresh interleaved pairs per prompt, two committed prompts; all performance delegated exclusively to benchmark-runner agents. No concurrent timing or builds/tests. Separate traced diagnostics. Prompts: `gpu_offload_probe.txt` MD5 `5835c71e471849b4a72e1dc8e39695e7` (59tokens), `humaneval_3_below_zero.txt` MD5 `37c5aad9f9efe93b5c47f27256bdf149` (129tokens). Short prompts; claims limited to these fixtures.

All runs proved16/16CPU coverage, no uncovered quants, qt44 traces with zero host-mapped GPU fallback. Pinned host3086MiB/device11836MiB; freeVRAMbefore16180MiB. Fresh initial incumbent medians10.3probe/10.4hum tok/s; decisions use each candidate's matched contemporaneous incumbent, not those initial rates.

## 1. Exact per-half activation-sum reuse — REVERT

Prepared two sums per256group once perGEMV, each using its original128-half even/odd AVX2 reduction tree. Preserved dot FMAs, header conversion, half correction/combine, and row accumulation. Per-kind scratch counts prevented confusion between one-sum and two-sum styles.

CPU41tests passed, including exact cached/uncached qt44 and reused-buffer tests. Both decoded completions byte-identical to incumbent. Exploratory GEMV shape timings improved3–8%, but end-to-end did not resolve a win:

- Probe decode incumbent10.4/10.4/10.4 → candidate10.5/10.4/10.5; median+1.0%. Wallmedian10.1→10.2.
- HumanEval incumbent10.4/10.4/10.3 → candidate10.5/10.4/10.5; median+1.0%. Wallmedian10.0→10.1.
- Some matched pairs tied; pooledSD≤0.05tok/s, differences inside the declared1–3% session noise criterion. Reverted as inconclusive, not incorrect.

## 2. Specialized nibble unpacking — REVERT

16-byte loads decoded32codes using nibble masks/interleave/widen, replacing repeated8-code broadcast/variable-shift decoding. Preserved per-half even/odd FMA and activation-sum trees. CPU42tests passed, including full-GEMV bitwise generic-reference comparisons and reused inputs. Both greedy outputs identical.

- Probe decode−4.8%, HumanEval−3.8% versus matched incumbent medians.
- All six matched pairs slower; wall throughput also regressed (roughly0.3–0.4tok/s per pair).
- Exploratory large-shape GEMVs4–8%slower. ISA confirmed intended unpack with no `vpsrlvd`, but more stack/frame references (9 versus2), consistent with increased register pressure.

Resolved regression: reverted.

## 3. Interleaved half-group computation — REVERT

Used incumbent `codes8<4>` decode while processing both128-halves concurrently. Eight independent vector accumulators preserved each half's original dot/sum associations and final combine. Generic reference retained only during tests. CPU43tests passed, including cancellation-prone inputs, longK and unaligned last-row comparisons. Both decoded completions byte-identical.

- Probe incumbent10.4/10.4/10.4 → candidate10.4/10.4/10.3; median0.0%.
- HumanEval incumbent10.4/10.4/10.4 → candidate10.4/10.3/10.3; median−1.0%. Wall10.1→10.0.
- No positive pair among six. PooledSD≤0.049tok/s. Mixed-sign hot-shape changes, no systematic kernel win. Test-binary ISA showed incumbent broadcast/shift decode and no added hot-path stack references.

No resolved gain: reverted.

## Disposition and receipts

All three arithmetic-preserving candidates removed. Incumbent production code unchanged; earlier retained exact sums/row grain remain. No separate threading of halves was introduced. These are local Zen4/27B/budget48 results, not universal rejection of these techniques.

Full hashes, candidate diffs, archived binary pairs, raw JSON samples, stderr route proof, numerical logs, ISA excerpts and per-process reports: `.codeinsight+research/mq4v2-avx2-ordered/{baseline,half-sums,nibble,interleave}/`. Local paths are discovery pointers, not externally durable archives. Experiment log: `.codeinsight+research/mq4v2-avx2-ordered.md`. No performance gain from this ordered campaign is claimed.
