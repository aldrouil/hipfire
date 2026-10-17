#!/bin/bash
# P3T s0375/s045 conditional sweep: 9B(b16)+27B(b56), A=retained C4C5 / B=FINAL2-exact,
# 4 fresh processes per cell, ABBA then BAAB, serialized GPU.
set -u
cd /home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-opt
A_CLI=.codeinsight+research/passback-phase2/baseline-bin/hipfire-a5a87e6cd
A_DAE=.codeinsight+research/passback-phase2/baseline-bin/daemon-a5a87e6cd
B_CLI=/home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-final2/target/release/hipfire
B_DAE=/home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-final2/target/release/daemon
M9="$HOME/.hipfire/models/qwen3.5-9b.mq4"
M27="$HOME/.hipfire/models/qwen3.8-27b.mq4"
D=docs/perf-checkpoints/data-2026-10-17-passback-phase3-throughput-gfx1201
W=$D/SCRIPTS/P3T-one.py
run_one() {
  echo "=== START $6 $(date -u +%FT%TZ) ==="
  python3 "$W" "$1" "$2" "$3" "$4" "$5" "$D/$6" "$7"
  echo "=== END $6 rc=$? $(date -u +%FT%TZ) ==="
}
# 9B s0375 ABBA then BAAB
run_one "$A_CLI" "$A_DAE" 0.375 16 "$M9" 9B-S0375/A-r1 p3t-9m-A1
run_one "$B_CLI" "$B_DAE" 0.375 16 "$M9" 9B-S0375/B-r1 p3t-9m-B1
run_one "$B_CLI" "$B_DAE" 0.375 16 "$M9" 9B-S0375/B-r2 p3t-9m-B2
run_one "$A_CLI" "$A_DAE" 0.375 16 "$M9" 9B-S0375/A-r2 p3t-9m-A2
run_one "$B_CLI" "$B_DAE" 0.375 16 "$M9" 9B-S0375/B-r3 p3t-9m-B3
run_one "$A_CLI" "$A_DAE" 0.375 16 "$M9" 9B-S0375/A-r3 p3t-9m-A3
run_one "$A_CLI" "$A_DAE" 0.375 16 "$M9" 9B-S0375/A-r4 p3t-9m-A4
run_one "$B_CLI" "$B_DAE" 0.375 16 "$M9" 9B-S0375/B-r4 p3t-9m-B4
# 9B s045 ABBA then BAAB
run_one "$A_CLI" "$A_DAE" 0.45 16 "$M9" 9B-S045/A-r1 p3t-9p-A1
run_one "$B_CLI" "$B_DAE" 0.45 16 "$M9" 9B-S045/B-r1 p3t-9p-B1
run_one "$B_CLI" "$B_DAE" 0.45 16 "$M9" 9B-S045/B-r2 p3t-9p-B2
run_one "$A_CLI" "$A_DAE" 0.45 16 "$M9" 9B-S045/A-r2 p3t-9p-A2
run_one "$B_CLI" "$B_DAE" 0.45 16 "$M9" 9B-S045/B-r3 p3t-9p-B3
run_one "$A_CLI" "$A_DAE" 0.45 16 "$M9" 9B-S045/A-r3 p3t-9p-A3
run_one "$A_CLI" "$A_DAE" 0.45 16 "$M9" 9B-S045/A-r4 p3t-9p-A4
run_one "$B_CLI" "$B_DAE" 0.45 16 "$M9" 9B-S045/B-r4 p3t-9p-B4
# 27B s0375 ABBA then BAAB
run_one "$A_CLI" "$A_DAE" 0.375 56 "$M27" M27B-S0375/A-r1 p3t-27m-A1
run_one "$B_CLI" "$B_DAE" 0.375 56 "$M27" M27B-S0375/B-r1 p3t-27m-B1
run_one "$B_CLI" "$B_DAE" 0.375 56 "$M27" M27B-S0375/B-r2 p3t-27m-B2
run_one "$A_CLI" "$A_DAE" 0.375 56 "$M27" M27B-S0375/A-r2 p3t-27m-A2
run_one "$B_CLI" "$B_DAE" 0.375 56 "$M27" M27B-S0375/B-r3 p3t-27m-B3
run_one "$A_CLI" "$A_DAE" 0.375 56 "$M27" M27B-S0375/A-r3 p3t-27m-A3
run_one "$A_CLI" "$A_DAE" 0.375 56 "$M27" M27B-S0375/A-r4 p3t-27m-A4
run_one "$B_CLI" "$B_DAE" 0.375 56 "$M27" M27B-S0375/B-r4 p3t-27m-B4
# 27B s045 ABBA then BAAB
run_one "$A_CLI" "$A_DAE" 0.45 56 "$M27" M27B-S045/A-r1 p3t-27p-A1
run_one "$B_CLI" "$B_DAE" 0.45 56 "$M27" M27B-S045/B-r1 p3t-27p-B1
run_one "$B_CLI" "$B_DAE" 0.45 56 "$M27" M27B-S045/B-r2 p3t-27p-B2
run_one "$A_CLI" "$A_DAE" 0.45 56 "$M27" M27B-S045/A-r2 p3t-27p-A2
run_one "$B_CLI" "$B_DAE" 0.45 56 "$M27" M27B-S045/B-r3 p3t-27p-B3
run_one "$A_CLI" "$A_DAE" 0.45 56 "$M27" M27B-S045/A-r3 p3t-27p-A3
run_one "$A_CLI" "$A_DAE" 0.45 56 "$M27" M27B-S045/A-r4 p3t-27p-A4
run_one "$B_CLI" "$B_DAE" 0.45 56 "$M27" M27B-S045/B-r4 p3t-27p-B4
echo "=== ALL DONE P3T-sweep $(date -u +%FT%TZ) ==="
