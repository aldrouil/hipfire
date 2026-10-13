# Dense CPU SIMD experiments on Zen 4 — 2026-10-13

**Lifecycle:** historical. **Authority:** fixture-bound measured evidence, not admission or a product floor.

## Fixture and method

AMD Ryzen 7 7800X3D, 8 cores/16 threads, 96 MiB L3; RX 9070 XT gfx1201 device 0, HIP 7.2. Original source `b02ec9669ae5401898bfb61e98c2edfca27c8255`; retained candidate adds ordinary MQ4/MQ6 activation-sum reuse and a 64-row minimum Rayon split. Arithmetic/layout/features unchanged.

Native `hipfire bench`, `--spec off --runs 5 --warmups 2 --max-tokens 64 --backend noslots --workload stateless --kv-mode q8 --max-seq 512 --prompt-file PATH --json`. Env: `HIPFIRE_OFFLOAD_EXEC=cpu RAYON_NUM_THREADS=16 HIP_VISIBLE_DEVICES=0 HIPFIRE_REPLAY_BACKEND=hip HIPFIRE_GRAPH=0 HIPFIRE_LOCAL=1`; explicit GPU layer budget 56 on 27B, 24 on 9B, 16 on 2B (eight spilled layers each). Three fresh interleaved processes per arm; no concurrent timing/builds. Traced diagnostics separate from timing. All performance work executed by benchmark-runner agents.

Model MD5s: 27B local `qwen3.8-27b.mq4` `d1292b4d5bd6046693604201a6ca8074` (qt44; NOT canonical XTS); 9B MQ4 `296092bf1e6a45d78c1acf815eb93366`; MQ6 `8ab062e92deb07a3d56c17fd5a0411c1`; MQ3 `a25c2361b41343ac17348b1a1c3631cd`; 2B MQ4 `9ed6628f2df83ef4b1c062afd4a85bfb`.

Prompt MD5s: `benchmarks/prompts/gpu_offload_probe.txt` `5835c71e471849b4a72e1dc8e39695e7` (59 tokens, short-prompt warning); `benchmarks/prompts/humaneval_3_below_zero.txt` `37c5aad9f9efe93b5c47f27256bdf149`.

Original binary MD5s: CLI `c8a2926a638cdf8a2d29b638ecccba97`, daemon `29f760632e55270969c1ba10e9e8b2da`. Retained E3 timing pair: CLI `c47e36394c107adaa14d2778ee85ed4a`, daemon `3d80e29f578ddc7937babbb927caa4f0`. Final rebuilt daemon `c6d1bb260b65fc08ecf8aacca27718a5` was smoke-tested, not used for these timing claims.

## Primary 27B: original versus retained changes

Per-process decode medians (five measurements each):

- Probe: original 15.6/15.7/15.9; retained 16.0/15.9/16.2 tok/s. Median 15.7→16.0 (+1.9%). Wall 15.3→15.6 (+2.0%).
- HumanEval: original 15.8/15.9/15.9; retained 16.0/16.1/16.0. Median 15.9→16.0 (+0.6%). Wall 15.3→15.5 (+1.3%).
- Within-process SD≤0.05 tok/s; six of six decode pairs faster, but small differences are near cross-process variation. Small consistent edge, not a large or broadly transferable gain.
- All eight spilled layers CPU-covered; 71 trace shape lines, all qt44, zero host-mapped GPU fallback. Pinned host1543MiB/device13379MiB; identical free VRAM before load.
- Greedy decoded output byte-identical on both prompts (1535/1208 characters), inspected and coherent. Qt44 does not use the retained prepared-sum path: this measures row grain only.

## Smaller fixtures and isolated levers

- E1 prepared sums, 9B MQ4: matched medians decode30.8→31.7 (+2.9%), wall26.9→27.6 (+2.6%); per-process SD0.05 both arms. MQ6 tied23.6→23.6; MQ3 unchanged control.
- E1 exploratory per-shape GEMV timings: gate/up0.57→0.54ms, down0.58→0.54ms, QKV0.43→0.40ms. Not standalone microbenchmark claims.
- E3 grain versus E1, 9B: MQ4decode31.2→33.0 (+5.8%), wall27.2→28.5; MQ3decode32.6→35.0 (+7.4%), wall29.5→31.2; MQ6decode23.2→24.1 (+3.9%). Every interleaved pair faster; within-process SD≤0.34. MQ6 reference includes22.9 low outlier. Do not combine absolute rates from different sessions as matched cumulative evidence.
- Final 2B MQ4 original versus retained: decode58.7/58.7/57.8→65.3/65.0/64.9, median+10.7%; wall56.3→62.0 (+10.1%).
- Q5 inspected/tested numerically through existing coverage, not model-benchmarked: no dense Q5 fixture identified.

## Rejected local experiments

- Four AVX2 MQ4 accumulators: +1.29%decode/+1.11%wall versus E1. Real projection max normalized difference versus old CPU6.3e-5/RMS7.2e-6; versus f64 reference oldmax1.9e-5/new5.2e-5. GPU parity suite3/3passed, but one of two greedy prompts changed at character614. Reverted: small gain insufficient for precision cost.
- Genuine AVX512 qt13/20/44, same binary opt-in on/off: 9Bqt13decode33.1→33.8 (+2.1%); qt20 34.5→33.7 (−2.3%, every pair slower); primary27Bqt44 tie on both prompts at budget56 and exploratorybudget48. zmm disassembly and real-row breakpoint hits proved execution; no illegal-ISA or hot-path spills observed.
- AVX512 qt20/44 synthetic results bit-exact versus AVX2; qt13 synthetic maxrelative≈2e-7uncached/5.6e-7cached. Twelve real-row incremental comparisons and four decoded completions equal. Absolute f64 AVX512 probe VOID: omitted AWQ transform, not accuracy evidence.
- AVX512 removed from shipped code, including experimental env flag. This is a Zen4 disposition, not proof it loses on Zen5 or Intel.

## Architecture evidence and remaining work

Desktop Zen5 has materially wider backend resources than Zen4's split-512 execution, while mobile Zen5 differs. Alexander Yee's [Zen5 teardown](https://www.numberworld.org/blogs/2024_8_7_zen5_avx512_teardown/) reports near2×512-bit IPC in compute-bound microbenchmarks but only1–3%in memory-touching cases. [Zen5 frequency analysis](https://chipsandcheese.com/p/zen-5s-avx-512-frequency-behavior) describes workload-dependent transition effects. [Intel Ice Lake measurements](https://travisdowns.github.io/blog/2020/08/19/icl-avx512-freq.html) show small/absent all-core AVX512 clock penalties. [Phoronix server comparison](https://www.phoronix.com/review/intel-sapphirerapids-avx512/8) reports broader mixed-workload gains; those are not Hipfire quant-GEMV predictions. No matched public F32-activation low-bit GEMV comparison was found.

Ranked next work: native-width desktop Zen5/Intel hardware qualification of archived512experiments; exact per128-half activation-sum reuse for qt44; SIMD decode/uop profiling of MQ3/Q5/Q6 on representative fixtures. Activation int8/VNNI, repacking and broader scheduling changes require separate accuracy/memory justification and are not implemented.

## Verification and receipts

Final CPU lib41/41passed; final CLI/daemon release build succeeded. Freshly rebuilt native27BCPU smoke generated coherent sliding-window code (768-token length finish), 8/8CPU-covered. Existing GPU-reference suite3/3passed during numerical candidate evaluation. Retained arithmetic is bit-identical to incumbent CPU, not a claim of CPU/GPU equality.

Experiment log: `.codeinsight+research/dense-cpu-simd-2026-10-13.md`. Raw JSON, stderr, hashes, binary archives and complete per-process samples: `.codeinsight+research/dense-cpu-simd-2026-10-13/{e1,e2,e3,e4,final}/`, `BASELINE.md`, `final/REPORT.md`. Local paths are discovery pointers, not durable external archives. No automatic promotion/admission implied.
