#!/bin/bash
# P2C-ABBA-9B.sh — candidate vs frozen C4+C5 baseline, N=4/arm, ABBA-interleaved rounds.
# Arms: base=passback auto (3413c7bd/f6cb5fec) vs cand=passback auto (bbd6e077/f82f4fbac). Budget 16.
set -u
cd /home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-opt
BASE_CLI=.codeinsight+research/passback-phase2/baseline-bin/hipfire-a5a87e6cd
BASE_DAE=.codeinsight+research/passback-phase2/baseline-bin/daemon-a5a87e6cd
CAND_CLI=.codeinsight+research/passback-phase2/candidate-bin/hipfire-rotovl
CAND_DAE=.codeinsight+research/passback-phase2/candidate-bin/daemon-rotovl
MODEL="$HOME/.hipfire/models/qwen3.5-9b.mq4"
D=.codeinsight+research/passback-phase2/raw/P2C-ABBA-9B
W=.codeinsight+research/passback-phase2/P2-one.sh
run_one() { # $1=cli $2=daemon $3=out $4=home
  echo "=== START $3 $(date -u +%FT%TZ) ==="
  bash "$W" "$1" "$2" passback auto 16 "$MODEL" "$D/$3" "$4"
  echo "=== END $3 rc=$? $(date -u +%FT%TZ) ==="
}
run_one "$CAND_CLI" "$CAND_DAE" A1-cand abba9-a1c
run_one "$BASE_CLI" "$BASE_DAE" A1-base abba9-a1b
run_one "$BASE_CLI" "$BASE_DAE" B1-base abba9-b1b
run_one "$CAND_CLI" "$CAND_DAE" B1-cand abba9-b1c
run_one "$CAND_CLI" "$CAND_DAE" B2-cand abba9-b2c
run_one "$BASE_CLI" "$BASE_DAE" B2-base abba9-b2b
run_one "$BASE_CLI" "$BASE_DAE" A2-base abba9-a2b
run_one "$CAND_CLI" "$CAND_DAE" A2-cand abba9-a2c
echo "=== ALL DONE P2C-ABBA-9B $(date -u +%FT%TZ) ==="
