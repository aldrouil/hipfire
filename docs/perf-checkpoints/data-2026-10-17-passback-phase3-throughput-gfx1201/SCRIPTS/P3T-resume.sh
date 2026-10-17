#!/bin/bash
# P3T-resume.sh — resume FINAL2 campaign from M27B-S05/A-r2 (interrupted sample
# left only an empty daemonproof; A-r1/B-r1/B-r2 complete). Serialized GPU.
set -u
cd /home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-opt
A_CLI=.codeinsight+research/passback-phase2/baseline-bin/hipfire-a5a87e6cd
A_DAE=.codeinsight+research/passback-phase2/baseline-bin/daemon-a5a87e6cd
B_CLI=/home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-final2/target/release/hipfire
B_DAE=/home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-final2/target/release/daemon
M27="$HOME/.hipfire/models/qwen3.8-27b.mq4"
D=docs/perf-checkpoints/data-2026-10-17-passback-phase3-throughput-gfx1201
W=$D/SCRIPTS/P3T-one.py
run_one() {
  echo "=== START $6 $(date -u +%FT%TZ) ==="
  python3 "$W" "$1" "$2" "$3" "$4" "$5" "$D/$6" "$7"
  echo "=== END $6 rc=$? $(date -u +%FT%TZ) ==="
}
run_one "$A_CLI" "$A_DAE" 0.5 56 "$M27" M27B-S05/A-r2 p3t-27s-A2
run_one "$B_CLI" "$B_DAE" "" 56 "$M27" M27B-AUTO/B-r1 p3t-27a-B1
run_one "$A_CLI" "$A_DAE" "" 56 "$M27" M27B-AUTO/A-r1 p3t-27a-A1
run_one "$A_CLI" "$A_DAE" "" 56 "$M27" M27B-AUTO/A-r2 p3t-27a-A2
run_one "$B_CLI" "$B_DAE" "" 56 "$M27" M27B-AUTO/B-r2 p3t-27a-B2
run_one "$B_CLI" "$B_DAE" 0.5 56 "$M27" M27B-S05/B-r3 p3t-27s-B3
run_one "$A_CLI" "$A_DAE" 0.5 56 "$M27" M27B-S05/A-r3 p3t-27s-A3
run_one "$A_CLI" "$A_DAE" 0.5 56 "$M27" M27B-S05/A-r4 p3t-27s-A4
run_one "$B_CLI" "$B_DAE" 0.5 56 "$M27" M27B-S05/B-r4 p3t-27s-B4
run_one "$A_CLI" "$A_DAE" "" 56 "$M27" M27B-AUTO/A-r3 p3t-27a-A3
run_one "$B_CLI" "$B_DAE" "" 56 "$M27" M27B-AUTO/B-r3 p3t-27a-B3
run_one "$B_CLI" "$B_DAE" "" 56 "$M27" M27B-AUTO/B-r4 p3t-27a-B4
run_one "$A_CLI" "$A_DAE" "" 56 "$M27" M27B-AUTO/A-r4 p3t-27a-A4
echo "=== ALL DONE P3T-resume $(date -u +%FT%TZ) ==="
