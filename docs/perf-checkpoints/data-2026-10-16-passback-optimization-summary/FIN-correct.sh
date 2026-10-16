#!/bin/bash
# FIN-correct.sh — final correctness on FINAL binaries (existing serve_harness only):
# 1. 9B: varied battery + related chain, budget 16 passback auto.
# 2. 27B: varied battery + related chain, budget 56 passback auto.
# 3. share0-vs-CPU identity smoke on the FIXTURE (prose battery row1): passback share=0 vs cpu, exact same config; documents observed text identity fixture-scoped.
# Serial. Hardware exclusive with benchmarks (run after bench blocks).
set -u
cd /home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-opt
FINC=.codeinsight+research/passback-opt/final-bin/hipfire
FIND=.codeinsight+research/passback-opt/final-bin/daemon
D=.codeinsight+research/passback-opt/raw/FINCORR
B9="$HOME/.hipfire/models/qwen3.5-9b.mq4"
B27="$HOME/.hipfire/models/qwen3.8-27b.mq4"
HBAT=benchmarks/prompts/hw-gate/serve-battery.json
run() { # $1=out $2..=env-prefix... then harness args
  local OUT=$1; shift
  echo "=== START $OUT $(date -u +%FT%TZ) ==="
  "$@" 2>&1 | tail -8
  echo "=== END $OUT rc=$? $(date -u +%FT%TZ) ==="
}
mkhome() { rm -rf "/tmp/FINC-home-$1"; mkdir -p "/tmp/FINC-home-$1/.hipfire"; printf 'schema_version = 1\n' > "/tmp/FINC-home-$1/.hipfire/config.toml"; }
# 1. 9B battery + chain
mkhome 9bbat
run 9B-battery env -u RAYON_NUM_THREADS HIPFIRE_HOME=/tmp/FINC-home-9bbat/.hipfire HIPFIRE_DAEMON_BIN=$FIND HIPFIRE_CLI_BIN=$FINC HIPFIRE_LOCAL=1 HIPFIRE_NO_REGISTRY_FETCH=1 HIPFIRE_DPM_WARMUP_SECS=10 HIPFIRE_OFFLOAD_EXEC=passback HIPFIRE_GPU_LAYER_BUDGET=16 HIPFIRE_OFFLOAD_PASSBACK_SHARE=auto python3 scripts/serve_harness.py --model $B9 --tag qwen3.5:9b --sampling greedy --thinking off --mode battery --prompts-file $HBAT --max-tokens 256 --max-seq 4096 --kv auto --home /tmp/FINC-home-9bbat --serve-log $D/9B-battery.serve.log --port 11651 --out $D/9B-battery.json
mkhome 9bchain
run 9B-chain env -u RAYON_NUM_THREADS HIPFIRE_HOME=/tmp/FINC-home-9bchain/.hipfire HIPFIRE_DAEMON_BIN=$FIND HIPFIRE_CLI_BIN=$FINC HIPFIRE_LOCAL=1 HIPFIRE_NO_REGISTRY_FETCH=1 HIPFIRE_DPM_WARMUP_SECS=10 HIPFIRE_OFFLOAD_EXEC=passback HIPFIRE_GPU_LAYER_BUDGET=16 HIPFIRE_OFFLOAD_PASSBACK_SHARE=auto python3 scripts/serve_harness.py --model $B9 --tag qwen3.5:9b --sampling greedy --thinking off --mode chain --max-tokens 256 --max-seq 4096 --kv auto --home /tmp/FINC-home-9bchain --serve-log $D/9B-chain.serve.log --port 11652 --out $D/9B-chain.json
# 2. 27B battery + chain
mkhome 27bbat
 run 27B-battery env -u RAYON_NUM_THREADS HIPFIRE_HOME=/tmp/FINC-home-27bbat/.hipfire HIPFIRE_DAEMON_BIN=$FIND HIPFIRE_CLI_BIN=$FINC HIPFIRE_LOCAL=1 HIPFIRE_NO_REGISTRY_FETCH=1 HIPFIRE_DPM_WARMUP_SECS=10 HIPFIRE_OFFLOAD_EXEC=passback HIPFIRE_GPU_LAYER_BUDGET=56 HIPFIRE_OFFLOAD_PASSBACK_SHARE=auto python3 scripts/serve_harness.py --model $B27 --tag qwen3.8:27b --sampling greedy --thinking off --mode battery --prompts-file $HBAT --max-tokens 256 --max-seq 4096 --kv auto --home /tmp/FINC-home-27bbat --serve-log $D/27B-battery.serve.log --port 11653 --out $D/27B-battery.json
mkhome 27bchain
run 27B-chain env -u RAYON_NUM_THREADS HIPFIRE_HOME=/tmp/FINC-home-27bchain/.hipfire HIPFIRE_DAEMON_BIN=$FIND HIPFIRE_CLI_BIN=$FINC HIPFIRE_LOCAL=1 HIPFIRE_NO_REGISTRY_FETCH=1 HIPFIRE_DPM_WARMUP_SECS=10 HIPFIRE_OFFLOAD_EXEC=passback HIPFIRE_GPU_LAYER_BUDGET=56 HIPFIRE_OFFLOAD_PASSBACK_SHARE=auto python3 scripts/serve_harness.py --model $B27 --tag qwen3.8:27b --sampling greedy --thinking off --mode chain --max-tokens 256 --max-seq 4096 --kv auto --home /tmp/FINC-home-27bchain --serve-log $D/27B-chain.serve.log --port 11654 --out $D/27B-chain.json
# 3. share0 vs cpu identity smoke, 9B fixture row (single fixed prompt, exact CPU twin)
mkhome id0
run share0 env -u RAYON_NUM_THREADS HIPFIRE_HOME=/tmp/FINC-home-id0/.hipfire HIPFIRE_DAEMON_BIN=$FIND HIPFIRE_CLI_BIN=$FINC HIPFIRE_LOCAL=1 HIPFIRE_NO_REGISTRY_FETCH=1 HIPFIRE_DPM_WARMUP_SECS=10 HIPFIRE_OFFLOAD_EXEC=passback HIPFIRE_GPU_LAYER_BUDGET=16 HIPFIRE_OFFLOAD_PASSBACK_SHARE=0 python3 scripts/serve_harness.py --model $B9 --sampling greedy --thinking off --mode battery --prompt-file benchmarks/prompts/gpu_offload_probe.txt --max-tokens 128 --max-seq 4096 --kv auto --home /tmp/FINC-home-id0 --serve-log $D/ident-share0.serve.log --port 11655 --out $D/ident-share0.json
mkhome idc
run cpu-twin env -u RAYON_NUM_THREADS HIPFIRE_HOME=/tmp/FINC-home-idc/.hipfire HIPFIRE_DAEMON_BIN=$FIND HIPFIRE_CLI_BIN=$FINC HIPFIRE_LOCAL=1 HIPFIRE_NO_REGISTRY_FETCH=1 HIPFIRE_DPM_WARMUP_SECS=10 HIPFIRE_OFFLOAD_EXEC=cpu HIPFIRE_GPU_LAYER_BUDGET=16 python3 scripts/serve_harness.py --model $B9 --sampling greedy --thinking off --mode battery --prompt-file benchmarks/prompts/gpu_offload_probe.txt --max-tokens 128 --max-seq 4096 --kv auto --home /tmp/FINC-home-idc --serve-log $D/ident-cpu.serve.log --port 11656 --out $D/ident-cpu.json
echo "=== ALL DONE CORRECTNESS $(date -u +%FT%TZ) ==="
