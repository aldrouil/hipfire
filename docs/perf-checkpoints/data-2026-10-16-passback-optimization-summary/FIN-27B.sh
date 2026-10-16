#!/bin/bash
# FIN-27B.sh — Campaign END 27B official native-bench: 4 arms x N3 fresh
# processes, balanced interleave (rotated). Budget 56 resident / 8 spill
# uniform ALL arms (fit-proven 2026-10-16: loads, device 4119 MiB). Same
# pins otherwise: --runs3 --warmups3 --spec off --backend noslots
# --workload stateless --max-tokens128 --max-seq4096 kv auto vmm canonical prompt.
set -u
cd /home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-opt
BASE=.codeinsight+research/passback-opt/baseline-bin/hipfire-12daf4fe
BASED=.codeinsight+research/passback-opt/baseline-bin/daemon-12daf4fe
FIN=.codeinsight+research/passback-opt/final-bin/hipfire
FIND=.codeinsight+research/passback-opt/final-bin/daemon
MODEL="$HOME/.hipfire/models/qwen3.8-27b.mq4"
D=.codeinsight+research/passback-opt/raw/FIN27B
W=.codeinsight+research/passback-opt/raw/FIN-one.sh
run_one() { # $1=cli $2=daemon $3=mode $4=share $5=out $6=home
  echo "=== START $5 $(date -u +%FT%TZ) ==="
  bash "$W" "$1" "$2" "$3" "$4" 56 "$MODEL" "$D/$5" "$6"
  echo "=== END $5 rc=$? $(date -u +%FT%TZ) ==="
}
run_one "$BASE" "$BASED" passback auto R1-orig-pb  r1-oripb
run_one "$FIN" "$FIND" passback auto R1-fin-pb   r1-finpb
run_one "$FIN" "$FIND" cpu ""          R1-fin-cpu  r1-fincpu
run_one "$FIN" "$FIND" pcie ""         R1-fin-pcie r1-finpcie
run_one "$FIN" "$FIND" pcie ""         R2-fin-pcie r2-finpcie
run_one "$FIN" "$FIND" cpu ""          R2-fin-cpu  r2-fincpu
run_one "$FIN" "$FIND" passback auto R2-fin-pb   r2-finpb
run_one "$BASE" "$BASED" passback auto R2-orig-pb  r2-oripb
run_one "$FIN" "$FIND" cpu ""          R3-fin-cpu  r3-fincpu
run_one "$BASE" "$BASED" passback auto R3-orig-pb  r3-oripb
run_one "$FIN" "$FIND" pcie ""         R3-fin-pcie r3-finpcie
run_one "$FIN" "$FIND" passback auto R3-fin-pb   r3-finpb
echo "=== ALL DONE 27B $(date -u +%FT%TZ) ==="
