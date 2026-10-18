# RL27 A/B one process sample: fresh CLI+daemon native bench, trace off, auto share, 27B budget 56.
import hashlib, json, os, subprocess, sys, time
cli, daemon, out, hn = sys.argv[1:5]
home1 = f"/tmp/RL27-home-{hn}"
wrapdir = f"/tmp/RL27-wrap-{hn}"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
os.makedirs(f"{home1}/.hipfire", exist_ok=True)
os.makedirs(wrapdir, exist_ok=True)
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
model = os.path.expanduser("~/.hipfire/models/qwen3.8-27b.mq4")
wrap_cli, wrap_dae = md5(cli), md5(daemon)
with open(f"{wrapdir}/daemon-wrap.sh", "w") as f:
    f.write('#!/bin/bash\n')
    f.write(f'echo "WRAP daemon_md5={wrap_dae} cli_md5={wrap_cli}" >> "{out}.daemonproof.log"\n')
    f.write(f'exec "{daemon}" "$@"\n')
import stat as _stat
os.chmod(f"{wrapdir}/daemon-wrap.sh", 0o755 | _stat.S_IXUSR | _stat.S_IXGRP | _stat.S_IXOTH)
open(f"{out}.daemonproof.log", "w").write("")
def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)
prompt = "benchmarks/prompts/gpu_offload_probe.txt"
meta = [f"=== RL27 A/B: OUT={out} HN={hn} HOME={home1} ===",
        f"date_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"cli_md5={md5(cli)} daemon_md5={md5(daemon)}",
        f"model_md5={md5(model)} model_bytes={os.path.getsize(model)}",
        f"prompt_md5={md5(prompt)} prompt_bytes={os.path.getsize(prompt)}",
        "RAYON=[unset]",
        "HIPFIRE_OFFLOAD_EXEC=passback HIPFIRE_GPU_LAYER_BUDGET=56 HIPFIRE_OFFLOAD_PASSBACK_SHARE=[auto/unset]"]
for v in ["HIPFIRE_SPLIT_DIAG","HIPFIRE_CPU_EXEC_TRACE","HIPFIRE_PROFILE","HIPFIRE_PROFILE_DECODE","HIPFIRE_MEMSET_DUMP","HIPFIRE_GRAPH","HIPFIRE_REPLAY","HIPFIRE_QWEN_CACHE_TRACE","HIPFIRE_QWEN_FFW_TRACE","HIPFIRE_FORWARD_LOWERED","HIPFIRE_RESIDENT_ORACLE_FIXTURE","HIPFIRE_RESIDENT_ORACLE_KV_MODE"]:
    meta.append(f"{v}_ambient=[{os.environ.get(v,'unset')}]")
meta.append("=== power pre ===")
meta.append(sh("bash .codeinsight+research/passback-opt/fixtures/power_snap.sh").stdout.strip())
open(f"{out}.meta.log", "w").write("\n".join(meta) + "\n")
open(f"{home1}/.hipfire/config.toml", "w").write("schema_version = 1\n")
link = f"{home1}/.hipfire/models"
if not os.path.exists(link):
    os.symlink(os.path.expanduser("~/.hipfire/models"), link)
child = dict(os.environ)
for v in ["RAYON_NUM_THREADS","HIPFIRE_CPU_EXEC_TRACE","HIPFIRE_SPLIT_DIAG","HIPFIRE_PROFILE","HIPFIRE_PROFILE_DECODE","HIPFIRE_MEMSET_DUMP","HIPFIRE_GRAPH","HIPFIRE_REPLAY","HIPFIRE_FORWARD_LOWERED","HIPFIRE_RESIDENT_ORACLE_FIXTURE","HIPFIRE_RESIDENT_ORACLE_KV_MODE","HIPFIRE_QWEN_CACHE_TRACE","HIPFIRE_QWEN_FFW_TRACE"]:
    child.pop(v, None)
child.update({"HIPFIRE_HOME": f"{home1}/.hipfire", "HIPFIRE_MODELS_DIR": f"{home1}/.hipfire/models",
    "HIPFIRE_DAEMON_BIN": f"{wrapdir}/daemon-wrap.sh", "HIPFIRE_LOCAL": "1",
    "HIPFIRE_NO_REGISTRY_FETCH": "1", "HIPFIRE_DPM_WARMUP_SECS": "10",
    "HIPFIRE_OFFLOAD_EXEC": "passback", "HIPFIRE_GPU_LAYER_BUDGET": "56"})
child.pop("HIPFIRE_OFFLOAD_PASSBACK_SHARE", None)  # auto
cmd = [cli, "bench", model,
       "--runs", "3", "--warmups", "3", "--spec", "off",
       "--backend", "noslots", "--workload", "stateless",
       "--max-tokens", "128", "--max-seq", "4096", "--kv-mode", "auto",
       "--kv-backend", "vmm", "--prompt-file", prompt, "--json"]
p = subprocess.run(cmd, env=child, capture_output=True, text=True, cwd="/home/avery/Documents/vibecoding/hipfire-special/hipfire-passback-opt")
open(f"{out}.json", "w").write(p.stdout)
open(f"{out}.stderr.log", "w").write(p.stderr)
post = [f"rc={p.returncode} date_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", "=== power post ===",
        sh("bash .codeinsight+research/passback-opt/fixtures/power_snap.sh").stdout.strip(), "=== daemon proof ==="]
try: post.append(open(f"{out}.daemonproof.log").read().strip() or "EMPTY")
except FileNotFoundError: post.append("MISSING")
try:
    r = json.loads(p.stdout)
    post.append(f"decode={r.get('decode_tok_s')} wall={r.get('wall_tok_s')} ttft={r.get('ttft_ms')}")
    post.append(f"samples.decode={r.get('samples',{}).get('decode')}")
    post.append(f"samples.wall={r.get('samples',{}).get('wall')}")
    kk = r.get('loaded',{})
    post.append(f"kv={kk.get('kv_mode')}/{kk.get('kv_backend')} spec={r.get('spec_route')} warnings={r.get('warnings')}")
    post.append(f"protocol={r.get('protocol')} prompt_tokens={r.get('prompt_tokens')} prompt_md5={r.get('prompt_md5')}")
except Exception as e: post.append(f"JSON-UNREADABLE: {e}")
with open(f"{out}.meta.log", "a") as f: f.write("\n".join(post) + "\n")
print(f"=== {out} rc={p.returncode} ===")
