#!/bin/bash
# P3-one.sh — ONE narrow pre-change ordering sample (fresh CLI+daemon), share 0.5 fixed.
# Usage: P3-one.sh <out-prefix> <homename>
set -u
CLI=$PWD/target/release/hipfire; DAE=$PWD/target/release/daemon
MODEL="$HOME/.hipfire/models/qwen3.5-9b.mq4"
OUT=$1; HN=$2
HOME1="/tmp/P3-home-$HN"; WRAPDIR="/tmp/P3-wrap-$HN"
mkdir -p "$(dirname "$OUT")" "$HOME1/.hipfire" "$WRAPDIR"
cat > "$WRAPDIR/daemon-wrap.sh" <<WEOF
#!/bin/bash
echo "WRAP-INVOCATION daemon_md5=\$(md5sum "$DAE" | cut -d' ' -f1) cli_md5=\$(md5sum "$CLI" | cut -d' ' -f1)" >> "$OUT.daemonproof.log"
exec "$DAE" "\$@"
WEOF
chmod +x "$WRAPDIR/daemon-wrap.sh"
: > "$OUT.daemonproof.log"
{
echo "=== P3 sample: OUT=$OUT HN=$HN HOME=$HOME1 ==="
echo "date_utc=$(date -u +%FT%TZ)"
echo "cli_md5=$(md5sum "$CLI" | cut -d' ' -f1) daemon_md5=$(md5sum "$DAE" | cut -d' ' -f1)"
echo "model_md5=$(md5sum "$MODEL" | cut -d' ' -f1) model_bytes=$(stat -c%s "$MODEL")"
echo "prompt_md5=$(md5sum benchmarks/prompts/gpu_offload_probe.txt | cut -d' ' -f1)"
echo "commit=$(git rev-parse HEAD) dirty=[$(git status --porcelain | head -5 | tr '\n' ';')]"
echo "HIPFIRE_OFFLOAD_EXEC=passback HIPFIRE_GPU_LAYER_BUDGET=16 HIPFIRE_OFFLOAD_PASSBACK_SHARE=0.5"
echo "unset-check: [$(env | grep -iE 'HIPFIRE_SPLIT_DIAG|HIPFIRE_PROFILE|HIPFIRE_CPU_EXEC_TRACE' || true)]"
} > "$OUT.meta.log" 2>&1
MMODELS="$HOME/.hipfire/models"
printf 'schema_version = 1\n' > "$HOME1/.hipfire/config.toml"
[ -e "$HOME1/.hipfire/models" ] || ln -s "$MMODELS" "$HOME1/.hipfire/models"
env -u RAYON_NUM_THREADS -u HIPFIRE_SPLIT_DIAG -u HIPFIRE_CPU_EXEC_TRACE -u HIPFIRE_PROFILE -u HIPFIRE_PROFILE_DECODE \
  env HIPFIRE_HOME="$HOME1/.hipfire" HIPFIRE_MODELS_DIR="$HOME1/.hipfire/models" \
  HIPFIRE_DAEMON_BIN="$WRAPDIR/daemon-wrap.sh" HIPFIRE_LOCAL=1 HIPFIRE_NO_REGISTRY_FETCH=1 \
  HIPFIRE_DPM_WARMUP_SECS=10 \
  HIPFIRE_OFFLOAD_EXEC=passback HIPFIRE_GPU_LAYER_BUDGET=16 HIPFIRE_OFFLOAD_PASSBACK_SHARE=0.5 \
  "$CLI" bench "$MODEL" \
  --runs 3 --warmups 3 --spec off --backend noslots --workload stateless \
  --max-tokens 128 --max-seq 4096 --kv-mode auto --kv-backend vmm \
  --prompt-file benchmarks/prompts/gpu_offload_probe.txt --json \
  > "$OUT.json" 2> "$OUT.stderr.log"; RC=$?
echo "rc=$RC date_utc=$(date -u +%FT%TZ)" >> "$OUT.meta.log"
echo "=== $OUT rc=$RC ==="
