#!/bin/bash
# FIN-redline.sh — dispatch route on FINAL daemon (existing redline_daemon_harness.py only):
# A. resident 9B pcie budget32 stable capture + AQL + multi-position HIP/PM4 parity/compatibility
# B. passback offload capture attempt (expect fail-closed refusal) — report actual observed error
# C. cpu offload capture attempt (expect fail-closed refusal) — report actual observed error
set -u
cd /home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-opt
FIND=$(pwd)/.codeinsight+research/passback-opt/final-bin/daemon
MODEL="$HOME/.hipfire/models/qwen3.5-9b.mq4"
D=.codeinsight+research/passback-opt/raw/FINRED
mkdir -p $D
echo "=== A: resident 9B pcie/budget32 capture+AQL+parity ==="
env -u RAYON_NUM_THREADS -u HIPFIRE_CPU_EXEC_TRACE -u HIPFIRE_PROFILE -u HIPFIRE_PROFILE_DECODE -u HIPFIRE_MEMSET_DUMP \
  HIPFIRE_DPM_WARMUP_SECS=10 HIPFIRE_OFFLOAD_EXEC=pcie HIPFIRE_GPU_LAYER_BUDGET=32 HIPFIRE_NO_REGISTRY_FETCH=1 \
  python3 scripts/redline_daemon_harness.py --model "$MODEL" --daemon "$FIND" \
  --kv-mode q8 --kv-backend vmm --max-seq 4096 --decode-context 128 \
  --pm4 --out $D/resident-pcie32.json --log $D/resident-pcie32.log; echo "A rc=$?"
echo "=== B: passback offload capture attempt (expect refusal) ==="
env -u RAYON_NUM_THREADS -u HIPFIRE_CPU_EXEC_TRACE -u HIPFIRE_PROFILE -u HIPFIRE_PROFILE_DECODE -u HIPFIRE_MEMSET_DUMP \
  HIPFIRE_DPM_WARMUP_SECS=10 HIPFIRE_OFFLOAD_EXEC=passback HIPFIRE_GPU_LAYER_BUDGET=16 HIPFIRE_OFFLOAD_PASSBACK_SHARE=auto HIPFIRE_NO_REGISTRY_FETCH=1 \
  python3 scripts/redline_daemon_harness.py --model "$MODEL" --daemon "$FIND" \
  --kv-mode q8 --kv-backend vmm --max-seq 4096 --decode-context 128 --skip-prefill \
  --pm4 --out $D/passback-attempt.json --log $D/passback-attempt.log; echo "B rc=$?"
echo "=== C: cpu offload capture attempt (expect refusal) ==="
env -u RAYON_NUM_THREADS -u HIPFIRE_CPU_EXEC_TRACE -u HIPFIRE_PROFILE -u HIPFIRE_PROFILE_DECODE -u HIPFIRE_MEMSET_DUMP \
  HIPFIRE_DPM_WARMUP_SECS=10 HIPFIRE_OFFLOAD_EXEC=cpu HIPFIRE_GPU_LAYER_BUDGET=16 HIPFIRE_NO_REGISTRY_FETCH=1 \
  python3 scripts/redline_daemon_harness.py --model "$MODEL" --daemon "$FIND" \
  --kv-mode q8 --kv-backend vmm --max-seq 4096 --decode-context 128 --skip-prefill \
  --pm4 --out $D/cpu-attempt.json --log $D/cpu-attempt.log; echo "C rc=$?"
echo "=== ALL DONE REDLINE $(date -u +%FT%TZ) ==="
