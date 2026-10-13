#!/usr/bin/env python3
"""Gate A (greedy identity + tau/route) and Gate C2 (9.3K/16K TTFT): ONE fabric hold, N fresh `hipfire serve`
processes, DET0 (HIPFIRE_QWEN4_MTP_APPEND_DET=0, explicit: unset = default ON on gfx1151) vs DET1 (=1), SAME binary, interleaved, via the existing
HaloMtpRegress harness (mixed-serve.py; driver.py for SSE TTFT) taken from the clean Git worktree.
Run inside a parent-admitted hold (launch.py prints the exact fabric argv):

  fabric-run.sh --class timing --card halo --hostmem-lock --mem-gb 110 --max-secs 900 --wait-secs 14400 \
     --priority P? --owner ID -- python3 -I <worktree>/competitors/halo-mtp-regress-20261012/mtp-append-det/serve_cells.py \
     --mode greedy --out ROOT/runs/<stamp>-greedy

--mode greedy : 5 prompts x {prose, Rust, Go, Java, 9K}, max_tokens 256, temperature 0, thinking off, rotated manifests
                r1/r2/r3 (the MTP-FLOOR-LAND set). Both arms carry the SAME explicit floor pin
                HIPFIRE_MTP_FLOOR_MIN_WINDOWS=16 (== product default) and no other route/floor override;
                MTP auto K3 / n-gram 5-3-3 / ctx 262144 / KV bf16 come from the shared private config.toml.
                HIPFIRE_MTP_TRACE=1 (both arms; --trace 0 disables) gives exact drafted/accepted/route per window.
--mode ttft   : one prompt per fresh process (chunk16k protocol): streaming SSE, temperature 0, thinking off,
                max_tokens 8, no preliminary request (cold first request); TTFT = POST->first non-empty content delta.
Manifests are resolved at run time (gatelib.resolve_manifest) so every prompt path is the verified worktree fixture.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gatelib as G

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=("greedy", "ttft"), required=True)
ap.add_argument("--prompt", choices=("p9k3", "p16k"), help="ttft mode: which prompt")
ap.add_argument("--out", required=True)
ap.add_argument("--bin", default=os.path.join(G.ROOT, "bin"), help="dir with daemon, hipfire, hipfire-weight-host (ONE set, both arms)")
ap.add_argument("--fixtures", default=G.FIXTURES)
ap.add_argument("--kcache", default=os.path.join(G.ROOT, "kcache"))
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--max-secs", type=int, default=0, help="MUST equal the hold's --max-secs (budget guard); default greedy 900, ttft 600")
ap.add_argument("--cell-timeout", type=int, default=240)
ap.add_argument("--health-timeout", type=int, default=300)
ap.add_argument("--port", type=int, default=0)
ap.add_argument("--trace", type=int, choices=(0, 1), default=1, help="greedy mode: HIPFIRE_MTP_TRACE in both arms")
ap.add_argument("--floor-min-windows", default=G.FLOOR_DEFAULT)
ap.add_argument("--allow-jit", action="store_true", help="do not set HIPFIRE_NO_DEVICE_COMPILER=1")
ap.add_argument("--det-log-regex", default=None, help="optional regex that must appear in DET1 serve.log and not in DET0")
ap.add_argument("--first-arm", choices=("d0", "d1"), default="d0")
ap.add_argument("--tag", default="")
a = ap.parse_args()
if a.mode == "ttft" and not a.prompt:
    sys.exit("--prompt required for ttft mode")
a.max_secs = a.max_secs or (900 if a.mode == "greedy" else 600)
port = a.port or (18771 if a.mode == "greedy" else 18772)
h = G.Hold("serve-" + a.mode + ("-" + a.prompt if a.prompt else ""), a.out, a.max_secs)
R = h.receipt
R.update(mode=a.mode, prompt=a.prompt, reps=a.reps, port=port, bin=a.bin, trace=a.trace, floor=a.floor_min_windows, tag=a.tag,
         det_arms={"d0": "HIPFIRE_QWEN4_MTP_APPEND_DET=0", "d1": "HIPFIRE_QWEN4_MTP_APPEND_DET=1"})
script = "mixed-serve.py" if a.mode == "greedy" else "driver.py"
harness = os.path.join(a.fixtures, script)
ok = G.preflight_common(h, a.bin, [os.path.join(a.fixtures, n) for n in ("mixed-serve.py", "driver.py", "fixtures.json")])
if not ok:
    h.finish("preflight_failed")
    print("PREFLIGHT FAILED", json.dumps({k: R.get(k) for k in ("model_identity", "device_probe", "git", "fixtures_bad")}, indent=1)[:2500])
    sys.exit(2)

# arm order: interleaved, alternating who goes first each rep (cancels drift)
first = ["d0", "d1"] if a.first_arm == "d0" else ["d1", "d0"]
plan = []
for rep in range(1, a.reps + 1):
    order = first if rep % 2 == 1 else first[::-1]
    for arm in order:
        plan.append((arm, rep))
R["plan"] = plan
h.save()
rc_all = 0
for arm, rep in plan:
    name = f"{arm}-r{rep}"
    if not G.hold_budget_ok(h, 150):
        R["cells"].append({"name": name, "skipped": "insufficient hold budget or terminated", "remaining_s": h.remaining()})
        rc_all = rc_all or 5
        continue
    cell = os.path.join(h.out, name)
    home = os.path.join(h.out, "home-" + name)
    G.write_config(home)
    env = G.clean_env(home, a.kcache, allow_jit=a.allow_jit)
    extra = [f"{G.FLOOR_ENV}={a.floor_min_windows}"]
    if a.mode == "greedy":
        if a.trace:
            extra.append("HIPFIRE_MTP_TRACE=1")
        src_manifest = os.path.join(a.fixtures, f"mixed-manifest-r{(rep - 1) % 3 + 1}.json")
    else:
        src_manifest = os.path.join(a.fixtures, "ttft-" + a.prompt + ".json")
    manifest = os.path.join(h.out, "manifests", f"{name}.json")
    G.resolve_manifest(src_manifest, manifest, a.fixtures)
    extra.append(f"{G.DET_ENV}={1 if arm == 'd1' else 0}")
    timeout = int(min(a.cell_timeout, h.remaining() - 30))
    cmd = [sys.executable, harness, "--out", cell, "--manifest", manifest, "--hipfire", os.path.join(a.bin, "hipfire"),
           "--daemon", os.path.join(a.bin, "daemon"), "--model", G.MODEL, "--port", str(port), "--timeout", str(timeout),
           "--health-timeout", str(a.health_timeout), "--tag", f"{h.gate}-{name}"]
    if a.mode == "ttft":
        cmd += ["--harness", os.path.join(a.fixtures, "mixed-serve.py"), "--no-warmup"]
    for e in extra:
        cmd += ["--env", e]
    os.makedirs(cell, exist_ok=True)
    rec = {"name": name, "arm": arm, "rep": rep, "argv": cmd, "extra_env": extra, "env": G.selected_env(env), "start_utc": G.utc(),
           "out": cell, "manifest_src": src_manifest, "manifest_resolved": manifest, "manifest_resolved_md5": G.md5_file(manifest),
           "det_value": "1" if arm == "d1" else None}
    R["cells"].append(rec)
    h.save()
    with open(os.path.join(cell, "driver.stdout"), "wb") as so, open(os.path.join(cell, "driver.stderr"), "wb") as se:
        p = h.spawn(cmd, env, so, se)
        rec["driver_pid"] = p.pid
        h.save()
        rc, info = h.wait(p, timeout + 330)   # harness cleanup may wait up to 180 s + 30 s after its own deadline
    rec.update(rc=rc, wait=info, end_utc=G.utc())
    if os.path.exists(os.path.join(cell, "cell.json")):
        c = json.load(open(os.path.join(cell, "cell.json")))
        rec["cell_status"] = c.get("status")
        rec["device_assertion"] = c.get("assertions", {}).get("device")
        rec["serve_pid"] = c.get("serve_pid")
        rec["binary_md5"] = {k: (c.get(k) or {}).get("md5") for k in ("hipfire", "daemon")}
    if a.det_log_regex and os.path.exists(os.path.join(cell, "serve.log")):
        import re
        hit = bool(re.search(a.det_log_regex, open(os.path.join(cell, "serve.log"), errors="replace").read()))
        rec["det_log_regex_hit"] = hit
        rec["det_log_regex_expected"] = (arm == "d1")
    if rc != 0:
        rc_all = rc_all or rc
    h.save()
    if h.terminated:
        break
status = "complete" if rc_all == 0 and len([c for c in R["cells"] if "skipped" not in c]) == len(plan) else f"incomplete_rc{rc_all}"
h.finish(status)
try:
    import analyze_serve
    analyze_serve.main([h.out, "--mode", a.mode])
except Exception as e:  # analysis is CPU-only and re-runnable; never fail the hold on it
    print("analysis deferred:", repr(e))
print("SERVE_CELLS_DONE", status, h.out)
sys.exit(0 if status == "complete" else 1)
