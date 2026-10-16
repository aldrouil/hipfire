#!/bin/bash
# C5CMP-ab_sample_trace.sh — ONE attribution-only C5 trace (primary prompt x4).
# Same as C5CMP-ab_sample.sh but with HIPFIRE_CPU_EXEC_TRACE=1 for the serve env
# so per-shape split: lines display actual r_gpu/share/frozen/sampled counts.
# Trace is NOT a headline sample. Usage mirrors C5CMP-ab_sample.sh.
set -u
CLI=$1; DAEMON=$2; MODE=$3; SHARE=$4; BUDGET=$5; MODEL=$6; TAG=$7; PORT=$8; HOME1=$9; OUT=${10}
mkdir -p "$(dirname "$OUT")" "$HOME1"
{
echo "=== env allowlist ==="
echo "HIPFIRE_DPM_WARMUP_SECS=10 HIPFIRE_OFFLOAD_EXEC=$MODE HIPFIRE_GPU_LAYER_BUDGET=$BUDGET HIPFIRE_OFFLOAD_PASSBACK_SHARE=$SHARE HIPFIRE_CPU_EXEC_TRACE=1(trace-only)"
echo "CLI md5: $(md5sum "$CLI" | cut -d' ' -f1)  DAEMON md5: $(md5sum "$DAEMON" | cut -d' ' -f1)"
echo "RAYON check (must be unset): [${RAYON_NUM_THREADS-unset}]"
echo "RAYON unset-command: env -u RAYON_NUM_THREADS (applied to serve env below)"
echo "=== power pre (sysfs card1) ==="
bash .codeinsight+research/passback-opt/fixtures/power_snap.sh
} > "${OUT}.meta.log" 2>&1
env -u RAYON_NUM_THREADS -u HIPFIRE_PROFILE -u HIPFIRE_PROFILE_DECODE -u HIPFIRE_MEMSET_DUMP \
  HIPFIRE_DPM_WARMUP_SECS=10 HIPFIRE_CLI_BIN="$CLI" HIPFIRE_DAEMON_BIN="$DAEMON" \
  HIPFIRE_OFFLOAD_EXEC="$MODE" HIPFIRE_GPU_LAYER_BUDGET="$BUDGET" HIPFIRE_OFFLOAD_PASSBACK_SHARE="$SHARE" \
  HIPFIRE_CPU_EXEC_TRACE=1 \
  python3 scripts/serve_harness.py --model "$MODEL" --tag "$TAG" --sampling greedy --thinking off \
  --mode battery --prompts-file .codeinsight+research/passback-opt/fixtures/warm3x_gpu_offload_probe.json \
  --max-tokens 128 --max-seq 4096 --kv auto --home "$HOME1" --serve-log "${OUT}.serve.log" \
  --port "$PORT" --out "${OUT}.json" 2>&1 | grep -E "DONE|prose.?(t1|t2|t3|t4)" | tail -6
{
echo "=== power post (sysfs card1) ==="
bash .codeinsight+research/passback-opt/fixtures/power_snap.sh
rocm-smi --showpower --showtemp 2>&1 | grep -E "Power|Temperature" | head -6
echo "=== trace-line count ==="
grep -c '^split:\|^cpu exec:' "${OUT}.serve.log" || true
} >> "${OUT}.meta.log" 2>&1
