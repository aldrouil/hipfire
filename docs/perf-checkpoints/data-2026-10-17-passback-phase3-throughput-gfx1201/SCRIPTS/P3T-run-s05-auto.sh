# P3T s05/auto campaign: 9B(b16) + 27B(b56), arms A=retained C4C5 / B=FINAL2-exact (isolated worktree),
# 4 fresh processes per cell, balanced ABBA/BAAB per block pair, serialized GPU.
# Model dirs: 9B-S05/{A,B}-r{1..4}, 9B-AUTO/..., M27B-S05/..., M27B-AUTO/...
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
run_one() { # $1=cli $2=dae $3=share $4=budget $5=model $6=out $7=home
  echo "=== START $6 $(date -u +%FT%TZ) ==="
  python3 "$W" "$1" "$2" "$3" "$4" "$5" "$D/$6" "$7"
  echo "=== END $6 rc=$? $(date -u +%FT%TZ) ==="
}
# Block 1: 9B s05 ABBA (A B B A), 9B auto BAAB (B A A B)
run_one "$A_CLI" "$A_DAE" 0.5 16 "$M9" 9B-S05/A-r1 p3t-9s-A1
run_one "$B_CLI" "$B_DAE" 0.5 16 "$M9" 9B-S05/B-r1 p3t-9s-B1
run_one "$B_CLI" "$B_DAE" 0.5 16 "$M9" 9B-S05/B-r2 p3t-9s-B2
run_one "$A_CLI" "$A_DAE" 0.5 16 "$M9" 9B-S05/A-r2 p3t-9s-A2
run_one "$B_CLI" "$B_DAE" "" 16 "$M9" 9B-AUTO/B-r1 p3t-9a-B1
run_one "$A_CLI" "$A_DAE" "" 16 "$M9" 9B-AUTO/A-r1 p3t-9a-A1
run_one "$A_CLI" "$A_DAE" "" 16 "$M9" 9B-AUTO/A-r2 p3t-9a-A2
run_one "$B_CLI" "$B_DAE" "" 16 "$M9" 9B-AUTO/B-r2 p3t-9a-B2
# Block 2 (reversed): 9B s05 BAAB, 9B auto ABBA
run_one "$B_CLI" "$B_DAE" 0.5 16 "$M9" 9B-S05/B-r3 p3t-9s-B3
run_one "$A_CLI" "$A_DAE" 0.5 16 "$M9" 9B-S05/A-r3 p3t-9s-A3
run_one "$A_CLI" "$A_DAE" 0.5 16 "$M9" 9B-S05/A-r4 p3t-9s-A4
run_one "$B_CLI" "$B_DAE" 0.5 16 "$M9" 9B-S05/B-r4 p3t-9s-B4
run_one "$A_CLI" "$A_DAE" "" 16 "$M9" 9B-AUTO/A-r3 p3t-9a-A3
run_one "$B_CLI" "$B_DAE" "" 16 "$M9" 9B-AUTO/B-r3 p3t-9a-B3
run_one "$B_CLI" "$B_DAE" "" 16 "$M9" 9B-AUTO/B-r4 p3t-9a-B4
run_one "$A_CLI" "$A_DAE" "" 16 "$M9" 9B-AUTO/A-r4 p3t-9a-A4
# Block 3: 27B s05 ABBA, 27B auto BAAB
run_one "$A_CLI" "$A_DAE" 0.5 56 "$M27" M27B-S05/A-r1 p3t-27s-A1
run_one "$B_CLI" "$B_DAE" 0.5 56 "$M27" M27B-S05/B-r1 p3t-27s-B1
run_one "$B_CLI" "$B_DAE" 0.5 56 "$M27" M27B-S05/B-r2 p3t-27s-B2
run_one "$A_CLI" "$A_DAE" 0.5 56 "$M27" M27B-S05/A-r2 p3t-27s-A2
run_one "$B_CLI" "$B_DAE" "" 56 "$M27" M27B-AUTO/B-r1 p3t-27a-B1
run_one "$A_CLI" "$A_DAE" "" 56 "$M27" M27B-AUTO/A-r1 p3t-27a-A1
run_one "$A_CLI" "$A_DAE" "" 56 "$M27" M27B-AUTO/A-r2 p3t-27a-A2
run_one "$B_CLI" "$B_DAE" "" 56 "$M27" M27B-AUTO/B-r2 p3t-27a-B2
# Block 4 (reversed): 27B s05 BAAB, 27B auto ABBA
run_one "$B_CLI" "$B_DAE" 0.5 56 "$M27" M27B-S05/B-r3 p3t-27s-B3
run_one "$A_CLI" "$A_DAE" 0.5 56 "$M27" M27B-S05/A-r3 p3t-27s-A3
run_one "$A_CLI" "$A_DAE" 0.5 56 "$M27" M27B-S05/A-r4 p3t-27s-A4
run_one "$B_CLI" "$B_DAE" 0.5 56 "$M27" M27B-S05/B-r4 p3t-27s-B4
run_one "$A_CLI" "$A_DAE" "" 56 "$M27" M27B-AUTO/A-r3 p3t-27a-A3
run_one "$B_CLI" "$B_DAE" "" 56 "$M27" M27B-AUTO/B-r3 p3t-27a-B3
run_one "$B_CLI" "$B_DAE" "" 56 "$M27" M27B-AUTO/B-r4 p3t-27a-B4
run_one "$A_CLI" "$A_DAE" "" 56 "$M27" M27B-AUTO/A-r4 p3t-27a-A4
echo "=== ALL DONE P3T-s05-auto $(date -u +%FT%TZ) ==="
