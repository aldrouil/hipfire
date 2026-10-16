# Refusal proof — passback/CPU offload vs retained-replay backend (FINAL daemon)
Source: ALREADY-OBSERVED FIN-redline.sh arms B/C stdout/tool-result of 2026-10-16T23:38Z
(the `echo B/C rc=1` lines in the delivered job output). The per-arm `--log`
files hold only daemon stderr (17/15 loader lines, ending before the load
refusal); the conflict error text arrived on harness stdout as the raised
RuntimeError (exact original transcript below, verbatim). Archived here so the
full-report quoted error has a raw artifact, not a reconstruction. NO re-run
performed (expected failures not re-executed).

## Command (both arms; exec differs only)
env -u RAYON_NUM_THREADS -u HIPFIRE_CPU_EXEC_TRACE -u HIPFIRE_PROFILE \
  -u HIPFIRE_PROFILE_DECODE -u HIPFIRE_MEMSET_DUMP HIPFIRE_DPM_WARMUP_SECS=10 \
  HIPFIRE_OFFLOAD_EXEC=<passback|cpu> HIPFIRE_GPU_LAYER_BUDGET=16 \
  [passback only: HIPFIRE_OFFLOAD_PASSBACK_SHARE=auto] HIPFIRE_NO_REGISTRY_FETCH=1 \
  python3 scripts/redline_daemon_harness.py --model $HOME/.hipfire/models/qwen3.5-9b.mq4 \
  --daemon .codeinsight+research/passback-opt/final-bin/daemon \
  --kv-mode q8 --kv-backend vmm --max-seq 4096 --decode-context 128 --skip-prefill \
  --pm4 --out raw/FINRED/<passback|cpu>-attempt.json --log raw/FINRED/<passback|cpu>-attempt.log
Provenance: FINAL daemon md5 98265d8b809d3b40245f05382b3d9d44
(`final-bin/daemon`; enlisted via absolute path, spawned directly by the
harness `Daemon` class, not via HIPFIRE_DAEMON_BIN). Model
`qwen3.5-9b.mq4` md5 296092bf1e6a45d78c1acf815eb93366. Outcome rc=1 both arms.

## B/passback — original stdout (verbatim)
Traceback (most recent call last):
  File "/home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-opt/scripts/redline_daemon_harness.py", line 1146, in <module>
    main()
    ~~~~^^
  File "/home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-opt/scripts/redline_daemon_harness.py", line 858, in main
    loaded = daemon.request(
        {
    ...<3 lines>...
        }
    )
  File "/home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-opt/scripts/redline_daemon_harness.py", line 81, in request
    raise RuntimeError(response.get("message", "daemon error"))
RuntimeError: load failed: qwen35: memory.offload_exec=passback conflicts with the retained-replay (Redline) backend: the replay tape records GPU launches and does not execute the CPU-executed steps, so a replayed route would compute from stale activations. Set memory.offload_exec=pcie (HIPFIRE_OFFLOAD_EXEC=pcie) or replay.backend=hip. GPU: gfx1201 (16180 MB free / 16304 MB total); no model loaded (the prior model was unloaded before this load)
B rc=1

## C/cpu — original stdout (verbatim)
Traceback (most recent call last):
  File "/home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-opt/scripts/redline_daemon_harness.py", line 1146, in <module>
    main()
    ~~~~^^
  File "/home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-opt/scripts/redline_daemon_harness.py", line 858, in main
    loaded = daemon.request(
        {
    ...<3 lines>...
        }
    )
  File "/home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-opt/scripts/redline_daemon_harness.py", line 81, in request
    raise RuntimeError(response.get("message", "daemon error"))
RuntimeError: load failed: qwen35: memory.offload_exec=cpu conflicts with the retained-replay (Redline) backend: the replay tape records GPU launches and does not execute the CPU-executed steps, so a replayed route would compute from stale activations. Set memory.offload_exec=pcie (HIPFIRE_OFFLOAD_EXEC=pcie) or replay.backend=hip. GPU: gfx1201 (16152 MB free / 16304 MB total); no model loaded (the prior model was unloaded before this load)
C rc=1
