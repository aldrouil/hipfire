#!/usr/bin/env python3
# P3T-one.py - ONE Phase3-Step2 throughput process sample (fresh CLI+daemon).
# Usage: P3T-one.py <cli_bin> <daemon_bin> <share> <budget> <model> <out> <homename>
# Contract mirrors Phase2 P2-one.sh/P2-rev-one.py: fresh isolated HOME/config,
# daemon-wrap md5 proof, HIPFIRE_LOCAL=1, HIPFIRE_NO_REGISTRY_FETCH=1,
# HIPFIRE_DPM_WARMUP_SECS=10, RAYON unset, trace/diag/profile vars cleared,
# passback mode, budget, share pin (empty share => auto/unset).
# Bench: --runs 3 --warmups 3 --spec off --backend noslots --workload stateless
#        --max-tokens 128 --max-seq 4096 --kv-mode auto --kv-backend vmm
#        --prompt-file benchmarks/prompts/gpu_offload_probe.txt --json
# Thinking: bench default off (no --reasoning-on). Greedy default sampling.
import hashlib, json, os, subprocess, sys, time
cli, daemon, share, budget, model, out, hn = sys.argv[1:9]
home1 = f"/tmp/P3T-home-{hn}"
wrapdir = f"/tmp/P3T-wrap-{hn}"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
os.makedirs(f"{home1}/.hipfire", exist_ok=True)
os.makedirs(wrapdir, exist_ok=True)
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
wrap_cli, wrap_dae = md5(cli), md5(daemon)
with open(f"{wrapdir}/daemon-wrap.sh", "w") as f:
    f.write('#!/bin/bash\n')
    f.write(f'echo "WRAP-INVOCATION daemon_md5={wrap_dae} cli_md5={wrap_cli}" >> "{out}.daemonproof.log"\n')
    f.write(f'exec "{daemon}" "$@"\n')
import stat as _stat
os.chmod(f"{wrapdir}/daemon-wrap.sh", 0o755 | _stat.S_IXUSR | _stat.S_IXGRP | _stat.S_IXOTH)
open(f"{out}.daemonproof.log", "w").close()
def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)
prompt = "benchmarks/prompts/gpu_offload_probe.txt"
meta = []
meta.append(f"=== P3T sample: OUT={out} HN={hn} HOME={home1} ===")
meta.append(f"date_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
meta.append(f"cli_md5={md5(cli)} daemon_md5={md5(daemon)}")
meta.append(f"model_md5={md5(model)} model_bytes={os.path.getsize(model)}")
meta.append(f"prompt_md5={md5(prompt)} prompt_bytes={os.path.getsize(prompt)}")
meta.append(f"commit={sh('git rev-parse HEAD').stdout.strip()} dirty=[{sh('git status --porcelain | head -5').stdout.strip()}]")
meta.append(f"RAYON=[{os.environ.get('RAYON_NUM_THREADS', 'unset')}]")
meta.append(f"HIPFIRE_OFFLOAD_EXEC=passback HIPFIRE_GPU_LAYER_BUDGET={budget} HIPFIRE_OFFLOAD_PASSBACK_SHARE=[{share}]")
for v in ["HIPFIRE_SPLIT_DIAG","HIPFIRE_CPU_EXEC_TRACE","HIPFIRE_PROFILE","HIPFIRE_PROFILE_DECODE","HIPFIRE_MEMSET_DUMP","HIPFIRE_GRAPH","HIPFIRE_REPLAY","HIPFIRE_QWEN_CACHE_TRACE","HIPFIRE_QWEN_FFW_TRACE"]:
    meta.append(f"{v}_ambient=[{os.environ.get(v,'unset')}]")
meta.append("=== power pre ===")
meta.append(sh("bash .codeinsight+research/passback-opt/fixtures/power_snap.sh").stdout.strip())
open(f"{out}.meta.log", "w").write("\n".join(meta) + "\n")
mmodels = os.path.expanduser("~/.hipfire/models")
open(f"{home1}/.hipfire/config.toml", "w").write("schema_version = 1\n")
link = f"{home1}/.hipfire/models"
if not os.path.exists(link):
    os.symlink(mmodels, link)
child = dict(os.environ)
for v in ["RAYON_NUM_THREADS","HIPFIRE_CPU_EXEC_TRACE","HIPFIRE_SPLIT_DIAG","HIPFIRE_PROFILE","HIPFIRE_PROFILE_DECODE","HIPFIRE_MEMSET_DUMP","HIPFIRE_GRAPH","HIPFIRE_REPLAY"]:
    child.pop(v, None)
child.update({
    "HIPFIRE_HOME": f"{home1}/.hipfire",
    "HIPFIRE_MODELS_DIR": f"{home1}/.hipfire/models",
    "HIPFIRE_DAEMON_BIN": f"{wrapdir}/daemon-wrap.sh",
    "HIPFIRE_LOCAL": "1",
    "HIPFIRE_NO_REGISTRY_FETCH": "1",
    "HIPFIRE_DPM_WARMUP_SECS": "10",
    "HIPFIRE_OFFLOAD_EXEC": "passback",
    "HIPFIRE_GPU_LAYER_BUDGET": budget,
})
if share:
    child["HIPFIRE_OFFLOAD_PASSBACK_SHARE"] = share
elif "HIPFIRE_OFFLOAD_PASSBACK_SHARE" in child:
    del child["HIPFIRE_OFFLOAD_PASSBACK_SHARE"]
cmd = [cli, "bench", model, "--runs", "3", "--warmups", "3", "--spec", "off",
       "--backend", "noslots", "--workload", "stateless",
       "--max-tokens", "128", "--max-seq", "4096", "--kv-mode", "auto",
       "--kv-backend", "vmm", "--prompt-file", prompt, "--json"]
p = subprocess.run(cmd, env=child, capture_output=True, text=True)
open(f"{out}.json", "w").write(p.stdout)
open(f"{out}.stderr.log", "w").write(p.stderr)
rc = p.returncode
post = [f"rc={rc} date_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"]
post.append("=== power post ===")
post.append(sh("bash .codeinsight+research/passback-opt/fixtures/power_snap.sh").stdout.strip())
post.append("=== daemon proof ===")
try:
    post.append(open(f"{out}.daemonproof.log").read().strip() or "EMPTY-daemonproof")
except FileNotFoundError:
    post.append("MISSING-daemonproof")
post.append("=== loaded/kv/samples from JSON ===")
try:
    r = json.loads(p.stdout)
    l = r.get("loaded", {})
    for k in ["arch","layers","kv_mode","kv_backend","kv_backend_request","kv_backend_reason","kv_backend_legacy","max_seq","max_seq_bound","max_seq_reason","model_ctx","card_cap"]:
        post.append(f"loaded.{k}={l.get(k)!r}")
    post.append(f"samples.decode={r.get('samples',{}).get('decode')}")
    post.append(f"samples.prefill={r.get('samples',{}).get('prefill')}")
    post.append(f"samples.wall={r.get('samples',{}).get('wall')}")
    post.append(f"prompt_tokens={r.get('prompt_tokens')} prompt_md5={r.get('prompt_md5')} warnings={r.get('warnings')}")
    post.append(f"decode_stats={r.get('decode_tok_s')} prefill_stats={r.get('prefill_tok_s')} wall={r.get('wall_tok_s')} ttft={r.get('ttft_ms')}")
    post.append(f"throttle_pct={(r.get('power') or [{}])[0].get('throttle_pct') if r.get('power') else None}")
    post.append(f"load_speculation={r.get('load_speculation')} spec_route={r.get('spec_route')} spec_requested={r.get('spec_requested')}")
except Exception as e:
    post.append(f"JSON-UNREADABLE: {e}")
with open(f"{out}.meta.log", "a") as f:
    f.write("\n".join(post) + "\n")
print(f"=== {out} rc={rc} ===")
