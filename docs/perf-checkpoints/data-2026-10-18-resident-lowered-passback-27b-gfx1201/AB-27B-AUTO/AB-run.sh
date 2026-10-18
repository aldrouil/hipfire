#!/bin/bash
# 27B resident-lowered A/B: C1 K1 K2 C2 | K3 C3 C4 K4, 3warm3measured, auto share, budget 56, fp8/vmm.
# Order is execution order; process indices pair per arm, not temporal pairs.
set -u
cd /home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-opt
CCLI=/home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-final2/target/release/hipfire
CDAE=/home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-final2/target/release/daemon
KCLI=/tmp/rl-candidate/target/release/hipfire
KDAE=/tmp/rl-candidate/target/release/daemon
D=docs/perf-checkpoints/data-2026-10-18-resident-lowered-passback-27b-gfx1201/AB-27B-AUTO
W=/tmp/rl-candidate/RL27-one.py
run_one() { # $1=cli $2=daemon $3=out $4=home $5=expect_cli $6=expect_dae
  echo "=== START $3 $(date -u +%FT%TZ) ==="
  python3 "$W" "$1" "$2" "$D/$3" "$4"
  echo "=== END $3 rc=$? $(date -u +%FT%TZ) ==="
  python3 - "$D/$3" "$5" "$6" <<'PYEOF' || { echo "RECEIPT-FAIL $3"; exit 1; }
import json,sys
base, exp_cli, exp_dae = sys.argv[1], sys.argv[2], sys.argv[3]
r = json.load(open(base+'.json'))
assert r.get('protocol')=='native-generate-v1', 'protocol'
assert r.get('prompt_md5')=='5835c71e471849b4a72e1dc8e39695e7', 'prompt md5'
assert r.get('prompt_tokens')==59, 'prompt tokens'
assert len(r['samples']['decode'])==3 and len(r['samples']['wall'])==3, 'sample count'
assert r.get('loaded',{}).get('kv_backend')=='vmm', 'kv backend'
assert r.get('loaded',{}).get('kv_mode')=='fp8', 'kv fp8'
dp = open(base+'.daemonproof.log').read()
assert 'WRAP' in dp and exp_cli in dp and exp_dae in dp, 'binary md5'
meta = open(base+'.meta.log').read()
assert 'rc=0' in meta, 'nonzero rc'
print('RECEIPT-OK', base, 'decode=', r['samples']['decode'], 'wall=', r['samples']['wall'])
PYEOF
}
C_CLI=72ab7a028cf4dfe4adb6638741b84a83; C_DAE=fb71e7d781a26671c8d8cb5ae6885e50
K_CLI=a11eb0a0488b8aada0832220c1a0d935; K_DAE=16c8abdd03c589c436fd46fd072eaa0e
case "${1:?usage: AB-run.sh block1|block2}" in
  block1)
    run_one "$CCLI" "$CDAE" C-r1 rl27-C1 "$C_CLI" "$C_DAE"
    run_one "$KCLI" "$KDAE" K-r1 rl27-K1 "$K_CLI" "$K_DAE"
    run_one "$KCLI" "$KDAE" K-r2 rl27-K2 "$K_CLI" "$K_DAE"
    run_one "$CCLI" "$CDAE" C-r2 rl27-C2 "$C_CLI" "$C_DAE" ;;
  block2)
    run_one "$KCLI" "$KDAE" K-r3 rl27-K3 "$K_CLI" "$K_DAE"
    run_one "$CCLI" "$CDAE" C-r3 rl27-C3 "$C_CLI" "$C_DAE"
    run_one "$CCLI" "$CDAE" C-r4 rl27-C4 "$C_CLI" "$C_DAE"
    run_one "$KCLI" "$KDAE" K-r4 rl27-K4 "$K_CLI" "$K_DAE" ;;
esac
echo "=== 27B $1 DONE $(date -u +%FT%TZ) ==="
