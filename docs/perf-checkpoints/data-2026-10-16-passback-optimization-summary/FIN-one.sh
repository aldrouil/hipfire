#!/bin/bash
# FIN-one.sh — ONE official native-bench process sample (fresh CLI+daemon).
# Usage: FIN-one.sh <cli_bin> <daemon_bin> <mode:passback|cpu|pcie> <share:auto|EMPTY-for-unset> <budget> <model> <out> <homename>
# Contract: N3 fresh processes PER ARM; median of the 3 internal measured
# runs (--runs 3) = ONE independent process sample. Official instrument only:
# native `hipfire bench`, AGENTS 3.1 headline. No serve_harness here.
set -u
CLI=$1; DAEMON=$2; MODE=$3; SHARE=$4; BUDGET=$5; MODEL=$6; OUT=$7; HN=$8
HOME1="/tmp/FIN-home-$HN"
WRAPDIR="/tmp/FIN-wrap-$HN"
mkdir -p "$(dirname "$OUT")" "$HOME1/.hipfire" "$WRAPDIR"
# Executable proof: launcher records invocation + md5s, then execs frozen daemon.
cat > "$WRAPDIR/daemon-wrap.sh" <<EOF
#!/bin/bash
echo "WRAP-INVOCATION daemon_md5=\$(md5sum "$DAEMON" | cut -d' ' -f1) cli_md5=\$(md5sum "$CLI" | cut -d' ' -f1)" >> "$OUT.daemonproof.log"
exec "$DAEMON" "\$@"
EOF
chmod +x "$WRAPDIR/daemon-wrap.sh"
: > "$OUT.daemonproof.log"
{
echo "=== FIN sample: OUT=$OUT HN=$HN HOME=$HOME1 ==="
echo "date_utc=$(date -u +%FT%TZ)"
echo "cli_md5=$(md5sum "$CLI" | cut -d' ' -f1) daemon_md5=$(md5sum "$DAEMON" | cut -d' ' -f1)"
echo "model_md5=$(md5sum "$MODEL" | cut -d' ' -f1) model_bytes=$(stat -c%s "$MODEL")"
echo "prompt_md5=$(md5sum benchmarks/prompts/gpu_offload_probe.txt | cut -d' ' -f1) prompt_bytes=$(stat -c%s benchmarks/prompts/gpu_offload_probe.txt)"
echo "commit=$(git rev-parse HEAD) dirty=[$(git status --porcelain | head -5 | tr '\n' ';')]"
echo "RAYON=[${RAYON_NUM_THREADS-unset}]"
echo "HIPFIRE_OFFLOAD_EXEC=$MODE HIPFIRE_GPU_LAYER_BUDGET=$BUDGET HIPFIRE_OFFLOAD_PASSBACK_SHARE=[$SHARE]"
echo "unset-check: [$(env | grep -iE 'CPU_EXEC_TRACE|HIPFIRE_PROFILE|MEMSET_DUMP|HIPFIRE_GRAPH|HIPFIRE_REPLAY|HIPFIRE_QWEN_CACHE_TRACE' || true)]"
echo "=== power pre ==="
bash .codeinsight+research/passback-opt/fixtures/power_snap.sh
echo "host_MemAvailable_pre_kB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')"
} > "$OUT.meta.log" 2>&1
MMODELS="$HOME/.hipfire/models"
printf 'schema_version = 1\n' > "$HOME1/.hipfire/config.toml"
[ -e "$HOME1/.hipfire/models" ] || ln -s "$MMODELS" "$HOME1/.hipfire/models"
if [ -n "$SHARE" ]; then SHARE_EXPORT="HIPFIRE_OFFLOAD_PASSBACK_SHARE=$SHARE"; else SHARE_EXPORT=""; fi
# shellcheck disable=SC2086
env -u RAYON_NUM_THREADS -u HIPFIRE_CPU_EXEC_TRACE -u HIPFIRE_PROFILE -u HIPFIRE_PROFILE_DECODE -u HIPFIRE_MEMSET_DUMP \
  env HIPFIRE_HOME="$HOME1/.hipfire" HIPFIRE_MODELS_DIR="$HOME1/.hipfire/models" \
  HIPFIRE_DAEMON_BIN="$WRAPDIR/daemon-wrap.sh" HIPFIRE_LOCAL=1 HIPFIRE_NO_REGISTRY_FETCH=1 \
  HIPFIRE_DPM_WARMUP_SECS=10 \
  HIPFIRE_OFFLOAD_EXEC="$MODE" HIPFIRE_GPU_LAYER_BUDGET="$BUDGET" $SHARE_EXPORT \
  "$CLI" bench "$MODEL" \
  --runs 3 --warmups 3 --spec off --backend noslots --workload stateless \
  --max-tokens 128 --max-seq 4096 --kv-mode auto --kv-backend vmm \
  --prompt-file benchmarks/prompts/gpu_offload_probe.txt --json \
  > "$OUT.json" 2> "$OUT.stderr.log"; RC=$?
echo "rc=$RC date_utc=$(date -u +%FT%TZ)" >> "$OUT.meta.log"
{
echo "=== power post ==="
bash .codeinsight+research/passback-opt/fixtures/power_snap.sh
echo "host_MemAvailable_post_kB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')"
echo "=== daemon proof ==="
cat "$OUT.daemonproof.log" 2>/dev/null || echo "MISSING-daemonproof"
echo "=== loaded/kv/samples from JSON ==="
python3 - "$OUT.json" <<'PY' 2>&1
import json,sys
p=sys.argv[1]
try:
  r=json.load(open(p))
except Exception as e:
  print(f"JSON-UNREADABLE: {e}"); sys.exit(0)
l=r.get("loaded",{})
for k in ["arch","layers","kv_mode","kv_backend","kv_backend_request","kv_backend_reason","kv_backend_legacy","max_seq","max_seq_bound","max_seq_reason","model_ctx","card_cap"]:
  print(f"loaded.{k}={l.get(k)!r}")
print("samples.decode=",r.get("samples",{}).get("decode"))
print("samples.prefill=",r.get("samples",{}).get("prefill"))
print("samples.ttft_ms=",r.get("samples",{}).get("ttft_ms"))
print("prompt_tokens=",r.get("prompt_tokens"),"prompt_md5=",r.get("prompt_md5"),"warnings=",r.get("warnings"))
print("decode_stats=",r.get("decode_tok_s"),"prefill_stats=",r.get("prefill_tok_s"),"wall=",r.get("wall_tok_s"),"ttft=",r.get("ttft_ms"))
PY
} >> "$OUT.meta.log" 2>&1
echo "=== $OUT rc=$RC ==="
