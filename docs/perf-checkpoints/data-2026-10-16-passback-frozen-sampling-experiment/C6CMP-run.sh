#!/bin/bash
# C6CMP-run.sh — C6 vs C5 HEADLINE primary N4/arm, balanced ABBAABBA (A=C5, B=C6).
# ONE process sample = median of rows 2-4 (row1 throwaway), via existing
# raw/C5REP-ab_sample.sh wrapper ONLY (no replacement harness).
# Env: explicit RAYON unset (in wrapper), no traces/profilers, normal affinity.
# Serial: one serve process at a time. Ports/homes recorded per sample.
set -u
cd /home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-opt
C5C=.codeinsight+research/passback-opt/candidate5-bin/hipfire
C5D=.codeinsight+research/passback-opt/candidate5-bin/daemon
C6C=.codeinsight+research/passback-opt/candidate6-bin/hipfire
C6D=.codeinsight+research/passback-opt/candidate6-bin/daemon
MODEL="$HOME/.hipfire/models/qwen3.5-9b.mq4"
W=.codeinsight+research/passback-opt/raw/C5REP-ab_sample.sh
run_one() { # $1=cli $2=daemon $3=outbase $4=home
  echo "=== START $3 $(date -u +%FT%TZ) ==="
  bash "$W" "$1" "$2" passback auto 16 "$MODEL" "qwen3.5:9b" 11551 "$4" "$3"
  echo "=== END $3 rc=$? $(date -u +%FT%TZ) ==="
}
# ABBAABBA: A1=C5 B1=C6 B2=C6 A2=C5 A3=C5 B3=C6 B4=C6 A4=C5
run_one "$C5C" "$C5D" .codeinsight+research/passback-opt/raw/C6CMP-A1-c5 /tmp/C6CMP-home-A1
run_one "$C6C" "$C6D" .codeinsight+research/passback-opt/raw/C6CMP-B1-c6 /tmp/C6CMP-home-B1
run_one "$C6C" "$C6D" .codeinsight+research/passback-opt/raw/C6CMP-B2-c6 /tmp/C6CMP-home-B2
run_one "$C5C" "$C5D" .codeinsight+research/passback-opt/raw/C6CMP-A2-c5 /tmp/C6CMP-home-A2
run_one "$C5C" "$C5D" .codeinsight+research/passback-opt/raw/C6CMP-A3-c5 /tmp/C6CMP-home-A3
run_one "$C6C" "$C6D" .codeinsight+research/passback-opt/raw/C6CMP-B3-c6 /tmp/C6CMP-home-B3
run_one "$C6C" "$C6D" .codeinsight+research/passback-opt/raw/C6CMP-B4-c6 /tmp/C6CMP-home-B4
run_one "$C5C" "$C5D" .codeinsight+research/passback-opt/raw/C6CMP-A4-c5 /tmp/C6CMP-home-A4
echo "=== ALL DONE $(date -u +%FT%TZ) ==="
