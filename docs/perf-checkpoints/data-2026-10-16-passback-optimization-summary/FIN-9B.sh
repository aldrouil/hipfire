#!/bin/bash
# FIN-9B.sh — FINAL 9B official native-bench campaign: 4 arms x N3 fresh
# processes, balanced interleave (one per arm per round, order rotated).
# Arms: ORIG-passback(baseline bin) vs FINAL-passback vs FINAL-cpu vs FINAL-pcie.
# Config: budget 16, share auto ONLY on passback arms, kv auto->q8 VMM,
# --runs3 --warmups3 --spec off --backend noslots --workload stateless
# --max-tokens128 --max-seq4096 canonical prompt. Serial, hardware exclusive.
set -u
cd /home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-opt
BASE=.codeinsight+research/passback-opt/baseline-bin/hipfire-12daf4fe
BASED=.codeinsight+research/passback-opt/baseline-bin/daemon-12daf4fe
FIN=.codeinsight+research/passback-opt/final-bin/hipfire
FIND=.codeinsight+research/passback-opt/final-bin/daemon
MODEL="$HOME/.hipfire/models/qwen3.5-9b.mq4"
D=.codeinsight+research/passback-opt/raw/FIN9B
W=.codeinsight+research/passback-opt/raw/FIN-one.sh
run_one() { # $1=cli $2=daemon $3=mode $4=share $5=out $6=home
  echo "=== START $5 $(date -u +%FT%TZ) ==="
  bash "$W" "$1" "$2" "$3" "$4" 16 "$MODEL" "$D/$5" "$6"
  echo "=== END $5 rc=$? $(date -u +%FT%TZ) ==="
}
# Round 1: ORIG-pb, FIN-pb, FIN-cpu, FIN-pcie
run_one "$BASE" "$BASED" passback auto R1-orig-pb  r1-oripb
run_one "$FIN" "$FIND" passback auto R1-fin-pb   r1-finpb
run_one "$FIN" "$FIND" cpu ""          R1-fin-cpu  r1-fincpu
run_one "$FIN" "$FIND" pcie ""         R1-fin-pcie r1-finpcie
# Round 2: rotated (reverse)
run_one "$FIN" "$FIND" pcie ""         R2-fin-pcie r2-finpcie
run_one "$FIN" "$FIND" cpu ""          R2-fin-cpu  r2-fincpu
run_one "$FIN" "$FIND" passback auto R2-fin-pb   r2-finpb
run_one "$BASE" "$BASED" passback auto R2-orig-pb  r2-oripb
# Round 3: rotated again
run_one "$FIN" "$FIND" cpu ""          R3-fin-cpu  r3-fincpu
run_one "$BASE" "$BASED" passback auto R3-orig-pb  r3-oripb
run_one "$FIN" "$FIND" pcie ""         R3-fin-pcie r3-finpcie
run_one "$FIN" "$FIND" passback auto R3-fin-pb   r3-finpb
echo "=== ALL DONE 9B $(date -u +%FT%TZ) ==="
