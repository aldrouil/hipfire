#!/bin/bash
# Direct daemon under rocprofv3: trace lands in-evidence-dir via absolute -d.
exec rocprofv3 --kernel-trace --memory-copy-trace --hip-trace --output-format csv -d /home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-opt/docs/perf-checkpoints/data-2026-10-17-passback-phase3-input-ready-gfx1201/rocp-daemon-passback -o daem -- /home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-opt/target/release/daemon "$@"
