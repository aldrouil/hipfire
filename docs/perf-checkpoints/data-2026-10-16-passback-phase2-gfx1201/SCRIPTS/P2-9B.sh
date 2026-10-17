#!/bin/bash
# P2-9B.sh — phase2 9B: auto + 4 fixed shares x N3 fresh processes, rotated interleave.
# Shares: auto, 0.25, 0.375, 0.45, 0.5. Budget 16. Frozen a5a87e6cd binaries.
set -u
cd /home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-opt
CLI=.codeinsight+research/passback-phase2/baseline-bin/hipfire-a5a87e6cd
DAE=.codeinsight+research/passback-phase2/baseline-bin/daemon-a5a87e6cd
MODEL="$HOME/.hipfire/models/qwen3.5-9b.mq4"
D=.codeinsight+research/passback-phase2/raw/P2-9B
W=.codeinsight+research/passback-phase2/P2-one.sh
run_one() { # $1=share $2=out $3=home
  echo "=== START $2 $(date -u +%FT%TZ) ==="
  bash "$W" "$CLI" "$DAE" passback "$1" 16 "$MODEL" "$D/$2" "$3"
  echo "=== END $2 rc=$? $(date -u +%FT%TZ) ==="
}
run_one auto    R1-auto   r1-auto
run_one 0.25    R1-s025   r1-s025
run_one 0.375   R1-s0375  r1-s0375
run_one 0.45    R1-s045   r1-s045
run_one 0.5     R1-s05    r1-s05
run_one 0.5     R2-s05    r2-s05
run_one 0.45    R2-s045   r2-s045
run_one 0.375   R2-s0375  r2-s0375
run_one 0.25    R2-s025   r2-s025
run_one auto    R2-auto   r2-auto
run_one 0.25    R3-s025   r3-s025
run_one auto    R3-auto   r3-auto
run_one 0.5     R3-s05    r3-s05
run_one 0.375   R3-s0375  r3-s0375
run_one 0.45    R3-s045   r3-s045
echo "=== ALL DONE P2-9B $(date -u +%FT%TZ) ==="
