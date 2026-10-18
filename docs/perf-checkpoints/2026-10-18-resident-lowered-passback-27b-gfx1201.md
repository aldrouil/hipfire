# Resident-only lowered passback — explicitly requested 27B native A/B

**Lifecycle:** `historical`. **Disposition:** exact-fixture fixed-share state correctness passed; **no resolved end-to-end throughput gain** in this bounded auto-share native A/B. No performance-based promotion, kernel-time reduction, or additional optimization is established.

**Measurement:** 2026-10-18, approximately **02:28–02:38 UTC**; full process timestamps are in the [evidence bundle](data-2026-10-18-resident-lowered-passback-27b-gfx1201/). The user explicitly requested this larger-model sweep after the [9B comparison](2026-10-18-resident-lowered-passback-gfx1201.md) was inconclusive. That new instruction superseded its conditional no-27B stop; it did not turn the failed 9B gate into a pass. Earlier records and their dispositions are unchanged.

## Fixture and frozen comparison

- RX 9070 XT (`gfx1201`), Ryzen 7 7800X3D, HIP `7.2.53211-3d9ef42`.
- Established custom Phase 3 model: `/home/avery/.hipfire/models/qwen3.8-27b.mq4`, **15,662,615,552 bytes**, MD5 `d1292b4d5bd6046693604201a6ca8074`. Despite the filename, the loader uses the same Qwen3.5 dense architecture/implementation as the 9B fixture: 64 layers here. This is **not** the canonical MQ4XTS quality fixture.
- Placement: **8 spilled / 56 resident** layers, `HIPFIRE_GPU_LAYER_BUDGET=56`; passback, share **auto/unset**, KV auto resolved **fp8/VMM** in every timed process. GDN recurrent state remains Q8.
- Byte-identical prompt: `benchmarks/prompts/gpu_offload_probe.txt`, MD5 `5835c71e471849b4a72e1dc8e39695e7`, 215 bytes / 59 tokens. No short-prompt prefill-throughput claim.
- **C:** exact validated FINAL2, base `b236099f64779b0291f5b48ff4ec0aa68d872b40` plus input-ready patch SHA-256 `f855d7b92c919e454ee4c364f903e3f762d7ede264b48c80c2f6bc45639d30e0`. CLI MD5 `72ab7a028cf4dfe4adb6638741b84a83`; daemon `fb71e7d781a26671c8d8cb5ae6885e50`.
- **K:** the same isolated FINAL2 candidate used in the 9B sweep, plus only resident-lowering feature patch SHA-256 `d46df385d1e324d3beef4ae18d1bbfbc5c5fa219a34aa67ced0134b591904b1d`. CLI MD5 `a11eb0a0488b8aada0832220c1a0d935`; daemon `16c8abdd03c589c436fd46fd072eaa0e`.

The feature is committed as `e92a1c3e98c0bd89147a2b86c9e5bbdd89bf279b`; its required post-measurement source formatting is recorded in the [committed-source amendment](2026-10-18-resident-lowered-passback-gfx1201-amendment-1.md). Neither that formatting nor a live-checkout binary was substituted for the frozen measured binaries. The FP8 oracle extension below changed only the test executable; C/K CLI and daemon binaries were not rebuilt and their identities were rechecked.

## Correctness gate before timing

The existing oracle initially refused cut 8 because it pinned only the 9B boundaries, and it constructed/dumped Q8 KV only. These were test-coverage limitations, not a reproduced production-forward failure. An unmodified 27B cut-16/Q8 comparison passed, but was **not** accepted as coverage of the timed cut-8/FP8 fixture. Failed-cut receipts are retained.

Only `crates/hipfire-arch-qwen35/tests/resident_lowered_passback_hw.rs` was extended: explicit `HIPFIRE_RESIDENT_ORACLE_FIXTURE=27b`, opt-in `HIPFIRE_RESIDENT_ORACLE_KV_MODE=fp8`, existing filtered FP8 VMM construction, and complete valid-prefix raw-byte dumps. FP8 rows include every E4M3FN code and inline f16 head scale; finite checks decode those formats separately. Model filenames do not select the fixture. Q8 defaults, GDN Q8/EF, fresh-process arms, active-stream ordering and byte-exact comparison remain unchanged; no production code, kernels, scheduling, quantization or caching changed for this sweep.

Approved oracle source SHA-256: `9c57732d278082fa82829452ab0518d1eafc97669fc2ed64f29de3c540d59046`; release test executable MD5 `71a5940c9e69ac25ae575ca5f3c7f95b`. Each pair used identical weights, teacher-forced inputs, fixed shares and zero initial recurrent state. Every one of **12 logits rows**, initial recurrent state, and full valid KV/GDN state after tokens **3, 6 and 12** were compared.

| Spill cut / first resident layer | Fixed GPU share | Stream | Result |
|---|---:|---|---|
| 8 / DeltaNet | 0.375 | Default | 876 tensors byte-exact |
| 7 / full attention | 0.375 | Default | 876 tensors byte-exact |
| 8 / DeltaNet | 0.5 | Default | 876 tensors byte-exact |
| 8 / DeltaNet | 0.375 | Non-blocking | 876 tensors byte-exact |

Each arm contains 96 FP8 KV **tensor dumps**, each holding its checkpoint's valid token prefix; these are not 96 individual token rows. GDN codes, scales, convolution histories and EF planes are included. Both sampled and queued unsampled passback returns were exercised. A legacy 9B/Q8 case also passed with 444 tensors byte-exact after the test extension. Auto-share greedy output equality is not asserted: auto scheduling is process-dependent.

Existing launch-counter metadata confirms engagement at cut 8: **973→805 launches/token**, 168 fewer. This is not a resident-layer timing measurement. Separate native candidate-daemon decoded smoke, thinking off and max_seq 4096, produced five inspected coherent responses: three stop / two length-capped, no empty or attractor output. Capped rows are not a complete-function or quality certification. Two earlier smoke configuration failures are retained: a thinking-cap check, then automatic context sizing that exceeded the VMM VRAM-fit check; explicit max_seq 4096 matched the timed fixture and passed.

Reproduction entry point on the stated fixture/GPU:

```sh
HIPFIRE_RESIDENT_ORACLE_FIXTURE=27b \
HIPFIRE_RESIDENT_ORACLE_MODEL="$HOME/.hipfire/models/qwen3.8-27b.mq4" \
HIPFIRE_RESIDENT_ORACLE_KV_MODE=fp8 \
cargo test -p hipfire-arch-qwen35 --release \
  --test resident_lowered_passback_hw -- --ignored --nocapture
```

## Controlled native throughput

Native `hipfire bench` generation protocol **`native-generate-v1`**, not a matrix/synthetic probe: `--backend noslots --workload stateless --spec off --runs 3 --warmups 3 --max-tokens 128 --max-seq 4096 --kv-mode auto --kv-backend vmm --prompt-file benchmarks/prompts/gpu_offload_probe.txt --json`. Thinking off; Rayon unset; isolated configuration homes; local spawn, registry fetch disabled, DPM warmup 10 seconds; trace/profile/temporary diagnostics off and graph/replay overrides unset.

Eight serialized fresh CLI/daemon processes, four per arm, in balanced order **C1 K1 K2 C2 | K3 C3 C4 K4**. Each process contributed its three-request median after three warmups. Every process returned zero; complete request arrays, daemon proofs and power/memory snapshots are retained. No failed or slow measured process was discarded.

Values below are process medians in tok/s. Indices identify each arm's samples, not simultaneous pairs.

| Process index | C decode | K decode | C wall | K wall |
|---|---:|---:|---:|---:|
| 1 | 17.0 | 18.1 | 16.7 | 17.8 |
| 2 | 18.1 | 18.0 | 17.8 | 17.7 |
| 3 | 17.7 | 17.7 | 17.4 | 17.4 |
| 4 | 17.5 | 17.7 | 17.2 | 17.4 |

| Metric | C median [min–max] | K median [min–max] | Descriptive median delta |
|---|---:|---:|---:|
| Decode | 17.60 [17.0–18.1] | 17.85 [17.7–18.1] | +1.4% |
| Native wall throughput | 17.30 [16.7–17.8] | 17.55 [17.4–17.8] | +1.4% |

**No resolved gain under the predeclared rule.** Both ordering blocks favor K in direction, but neither clears contemporaneous spread on decode or wall. Block 1's midpoint gap is +0.50 tok/s on both metrics against its control-pair range of 1.1; block 2's gap is +0.10 against 0.2. C1 was slower than the other control processes and remains included; its cause was not established. See the runner's [full analysis](data-2026-10-18-resident-lowered-passback-27b-gfx1201/AB-ANALYSIS.txt).

The same conservative rule as 9B requires decode and wall improvement beyond spread in both blocks. These data do not pass it, and are not proof of equivalence, regression or statistical significance. Native one-decimal process rates limit interpretation of small differences. Requests within a process are not independent process samples.

## Attribution, preservation and stop

Byte-exact fixed-share state and fewer launches are demonstrated. **Reduced resident-layer elapsed time is not measured.** The positive native medians cannot be assigned to resident kernel time versus host dispatch, auto-controller observations, or session drift without further evidence. No extra profiler capture or optimization campaign was initiated. Small historical differences in CPU/GPU overlap between the two model sizes are not evidence of different architectures or distinct bottlenecks.

Coverage is one custom model, placement family, short benchmark prompt, host/GPU and four processes per arm; oracle coverage is the stated 12-step fixed-share matrix, not arbitrary contexts, all hardware or all configurations. Decoded smoke establishes bounded output health, not full quality. Throttle residency is unavailable; pre/post telemetry does not establish a throttle-free run. Absolute 9B and 27B rates are not an arm-causal comparison.

The [evidence bundle](data-2026-10-18-resident-lowered-passback-27b-gfx1201/) preserves identity/protocol/source-approval records, test-extension patch and hashes, oracle receipts, decoded-smoke results and failures, all process JSON/sample arrays, commands, stderr, telemetry and daemon proofs. Large raw tensor dumps remain local-only, not Git objects or remotely published artifacts; their source roots include `/tmp/oracle-27b-fp8` and the supplemental Q8/legacy roots recorded in the receipts. A fresh Git checkout contains compact evidence, not raw tensor bytes.

**Stop:** requested 27B evaluation complete; no demonstrated throughput win on either model under the bounded comparison rule. No further fusion, caching, kernel, scheduler, or unrelated optimization pursued.
