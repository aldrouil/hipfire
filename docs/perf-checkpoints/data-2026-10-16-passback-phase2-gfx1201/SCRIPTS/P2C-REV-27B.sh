#!/bin/bash
# P2C-REV-27B.sh — reverse BAABBAAB, N=4/arm, frozen binaries, budget 56.
set -u
cd /home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-opt
BASE_CLI=.codeinsight+research/passback-phase2/baseline-bin/hipfire-a5a87e6cd
BASE_DAE=.codeinsight+research/passback-phase2/baseline-bin/daemon-a5a87e6cd
CAND_CLI=.codeinsight+research/passback-phase2/candidate-bin/hipfire-rotovl
CAND_DAE=.codeinsight+research/passback-phase2/candidate-bin/daemon-rotovl
MODEL="$HOME/.hipfire/models/qwen3.8-27b.mq4"
D=.codeinsight+research/passback-phase2/raw/P2C-REV-27B
W=.codeinsight+research/passback-phase2/P2-rev-one.py
run_one() {
  echo "=== START $3 $(date -u +%FT%TZ) ==="
  python3 "$W" "$1" "$2" passback auto 56 "$MODEL" "$D/$3" "$4"
  echo "=== END $3 rc=$? $(date -u +%FT%TZ) ==="
}
run_one "$BASE_CLI" "$BASE_DAE" R1-base rev27-r1b
run_one "$CAND_CLI" "$CAND_DAE" R1-cand rev27-r1c
run_one "$CAND_CLI" "$CAND_DAE" R2-cand rev27-r2c
run_one "$BASE_CLI" "$BASE_DAE" R2-base rev27-r2b
run_one "$BASE_CLI" "$BASE_DAE" R3-base rev27-r3b
run_one "$CAND_CLI" "$CAND_DAE" R3-cand rev27-r3c
run_one "$CAND_CLI" "$CAND_DAE" R4-cand rev27-r4c
run_one "$BASE_CLI" "$BASE_DAE" R4-base rev27-r4b
echo "=== ALL DONE P2C-REV-27B $(date -u +%FT%TZ) ==="
