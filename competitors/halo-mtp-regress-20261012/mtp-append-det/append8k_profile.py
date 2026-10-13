#!/usr/bin/env python3
"""Gate C1: Append-per-8K-chunk device profile, DET0 vs DET1, SAME binary, in ONE bounded fabric hold.

Method = MTP-APPEND-PROFILE.md: `hipfire bench MODEL --ttft --json --warmups 1 --runs 1 --max-tokens 1 --max-seq 262144
--prompt-file ciru-8k.txt` (8186 tokens, md5 4203b112...) with the daemon wrapped in rocprofv3 --kernel-trace --hip-trace,
fresh process + private HOME/lockdir per arm, default MTP K=3 (no tuning flags). The only arm difference is
HIPFIRE_QWEN4_MTP_APPEND_DET=1 (d1) vs =0 (d0, set explicitly: unset = default ON on gfx1151). Order alternates per rep (d0,d1 | d1,d0 ...). Profiled bench TTFT is NOT a speed claim.
The prompt is ciru/ciru-8k.txt from the clean Git worktree (fixtures.json-verified); launch.py prints the fabric argv:

  fabric-run.sh --class timing --card halo --hostmem-lock --mem-gb 110 --max-secs 700 --wait-secs 14400 \
     --priority P? --owner ID -- python3 -I <worktree>/competitors/halo-mtp-regress-20261012/mtp-append-det/append8k_profile.py \
     --out ROOT/runs/<stamp>-append8k --reps 2
"""
import argparse, glob, json, os, shlex, shutil, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gatelib as G
import analyze_append as AA

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--bin", default=os.path.join(G.ROOT, "bin"))
ap.add_argument("--fixtures", default=G.FIXTURES)
ap.add_argument("--kcache", default=os.path.join(G.ROOT, "kcache"))
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--rep-start", type=int, default=1, help="first rep number (alternation parity follows it)")
ap.add_argument("--max-secs", type=int, default=700)
ap.add_argument("--bench-timeout", type=int, default=240)
ap.add_argument("--allow-jit", action="store_true")
a = ap.parse_args()
PROMPT = os.path.join(a.fixtures, "ciru", "ciru-8k.txt")
PROMPT_MD5, PROMPT_TOKENS = "4203b112c9d01eea27bce2dc53d5d1b5", 8186
h = G.Hold("append8k-profile", a.out, a.max_secs)
R = h.receipt
R.update(reps=a.reps, prompt=PROMPT, prompt_md5_expected=PROMPT_MD5, baseline_append_ms=189.419264, target_ms=25.0)
ok = G.preflight_common(h, a.bin, [PROMPT, os.path.join(a.fixtures, "fixtures.json")])
if G.md5_file(PROMPT) != PROMPT_MD5:
    R["fixture_mismatch"] = [PROMPT]
    ok = False
if shutil.which("rocprofv3") is None:
    R["rocprofv3"] = "missing"
    ok = False
if not ok:
    h.finish("preflight_failed")
    print("PREFLIGHT FAILED", json.dumps({k: R.get(k) for k in ("model_identity", "device_probe", "git", "fixtures_bad", "fixture_mismatch", "rocprofv3")}, indent=1)[:2500])
    sys.exit(2)
events_py = os.path.join(G.SCRIPTS_DIR, "events.py")
plan = []
for rep in range(a.rep_start, a.rep_start + a.reps):
    plan += [(arm, rep) for arm in (("d0", "d1") if rep % 2 == 1 else ("d1", "d0"))]
R["plan"] = plan
rc_all = 0
for arm, rep in plan:
    name = f"{arm}-r{rep}"
    if not G.hold_budget_ok(h, 150):
        R["cells"].append({"name": name, "skipped": "insufficient hold budget or terminated", "remaining_s": h.remaining()})
        rc_all = rc_all or 5
        continue
    A = os.path.join(h.out, name)
    home = os.path.join(A, "home")
    G.write_config(home)
    os.makedirs(os.path.join(A, "lockdir"), exist_ok=True)
    wrap = os.path.join(A, "rocwrap.sh")
    open(wrap, "w").write("#!/usr/bin/env bash\ntee %s | rocprofv3 --kernel-trace --hip-trace --output-format csv -d %s -o kt -- %s \"$@\" | python3 %s %s\n" % (
        shlex.quote(os.path.join(A, "daemon.stdin.jsonl")), shlex.quote(os.path.join(A, "kt")), shlex.quote(os.path.join(a.bin, "daemon")),
        shlex.quote(events_py), shlex.quote(os.path.join(A, "events.jsonl"))))
    os.chmod(wrap, 0o755)
    extra = {"HIPFIRE_LOCK_DIR": os.path.join(A, "lockdir"), "HIPFIRE_DAEMON_BIN": wrap, "HIPFIRE_CLI_BIN": os.path.join(a.bin, "hipfire")}
    extra[G.DET_ENV] = "1" if arm == "d1" else "0"
    env = G.clean_env(home, a.kcache, extra, allow_jit=a.allow_jit)
    cmd = [os.path.join(a.bin, "hipfire"), "bench", G.MODEL, "--ttft", "--json", "--warmups", "1", "--runs", "1", "--max-tokens", "1",
           "--max-seq", "262144", "--prompt-file", PROMPT]
    rec = {"name": name, "arm": arm, "rep": rep, "argv": cmd, "env": G.selected_env(env), "start_utc": G.utc(), "out": A}
    R["cells"].append(rec)
    h.save()
    timeout = int(min(a.bench_timeout, h.remaining() - 60))
    with open(os.path.join(A, "bench.json"), "wb") as so, open(os.path.join(A, "stderr.txt"), "wb") as se:
        p = h.spawn(cmd, env, so, se)
        rec["pid"] = p.pid
        h.save()
        rc, info = h.wait(p, timeout)
    rec.update(rc=rc, wait=info, end_utc=G.utc())
    # profiler flush (profile.sh precedent: wait up to 60 s for the kernel trace)
    for _ in range(60):
        if glob.glob(os.path.join(A, "kt", "**", "*kernel_trace.csv"), recursive=True):
            break
        time.sleep(1)
    err = open(os.path.join(A, "stderr.txt"), errors="replace").read()
    rec["device_lines"] = G.device_lines(err)
    rec["device_ok"] = bool(rec["device_lines"])
    try:
        b = json.load(open(os.path.join(A, "bench.json")))
        rec["bench"] = {"prompt_tokens": b.get("prompt_tokens"), "prompt_md5": b.get("prompt_md5"), "ttft_samples_ms_profiled": (b.get("ttft") or {}).get("samples"),
                        "power_pci": (b.get("power_device") or {}).get("pci")}
        rec["bench_ok"] = b.get("prompt_tokens") == PROMPT_TOKENS and b.get("prompt_md5") == PROMPT_MD5
    except Exception as e:
        rec["bench_error"] = repr(e)
        rec["bench_ok"] = False
    try:
        r = AA.analyze(os.path.join(A, "kt"), arm, PROMPT_TOKENS)
        json.dump(r, open(os.path.join(A, "append-analysis.json"), "w"), indent=1, sort_keys=True)
        m = r.get("measured") or {}
        rec["append"] = {"busy_ms": m.get("append_busy_ms"), "span_ms": m.get("append_span_ms"), "launches": m.get("launches"), "rows": m.get("rows"),
                         "route_ok": r.get("route_ok"), "rows_match_prompt": r.get("rows_match_prompt_tokens"),
                         "incumbent_present": r.get("incumbent_present_in_measured"), "det_dense_launches": r.get("det_dense_launches_in_measured"),
                         "q8_shadow": r.get("q8_shadow_kernel")}
    except BaseException as e:
        rec["append_error"] = repr(e)
    if rc != 0:
        rc_all = rc_all or rc
    h.save()
    if h.terminated:
        break
cells = [c for c in R["cells"] if "append" in c]
d0 = [c["append"]["busy_ms"] for c in cells if c["arm"] == "d0" and c["append"].get("busy_ms") is not None]
d1 = [c["append"]["busy_ms"] for c in cells if c["arm"] == "d1" and c["append"].get("busy_ms") is not None]
R["summary"] = {"d0_busy_ms": d0, "d1_busy_ms": d1, "d0_median_ms": statistics.median(d0) if d0 else None,
                "d1_median_ms": statistics.median(d1) if d1 else None,
                "d1_max_ms": max(d1) if d1 else None, "target_ms": 25.0,
                "d1_all_le_target": bool(d1) and max(d1) <= 25.0,
                "all_cells_valid": bool(cells) and all(c.get("rc") == 0 and c.get("device_ok") and c.get("bench_ok") and c["append"].get("route_ok") and c["append"].get("rows_match_prompt") for c in cells)}
h.finish("complete" if rc_all == 0 and len(cells) == len(plan) else f"incomplete_rc{rc_all}")
print("APPEND8K_DONE", json.dumps(R["summary"]), h.out)
sys.exit(0 if R["status"] == "complete" and R["summary"]["all_cells_valid"] else 1)
