#!/usr/bin/env python3
"""Gate B hold driver: run ONE OR MORE arms of the existing ignored sampled_mtp_distribution_hw oracle as fresh processes of
the standalone test binary (the test's own documented split: "arms can run first as separate time-boxed invocations of
sampled_mtp_distribution_arm"). The exact parent tests are then re-run CPU-only by sampled_final.py; they reuse the arms'
`<arm>.ids.json` / `<arm>.identity.json` (+ config sidecars) and apply the unchanged chi-square / ALPHA 1e-3 / replay /
>=2000-trial criteria. This driver NEVER lowers trials (HIPFIRE_SAMPLED_MTP_TRIALS stays unset = 2000; a value <2000 is refused).

Arm table (kind:arm -> test entry, HIPFIRE_MTP_INCREMENTAL, extra env), copied from the test's parent functions:
  dist:ar|batched|interleaved        sampled_mtp_distribution_arm     unset|0|1      (HIPFIRE_MTP_SAMPLED=1)
  identity:ar|batched|interleaved|adaptive  sampled_mtp_identity_arm  unset|0|1|unset (+HIPFIRE_MTP_SAMPLED_MODE=naive)
  ngram:ar (== dist:ar, shared ar.ids.json)
  ngram:copy_ar|ngram_batched|ngram_interleaved|copy_batched|copy_interleaved   sampled_mtp_ngram_distribution_arm
        unset|0|1|0|1
DET placement: no arm relies on the default (unset = ON on gfx1151). Reference arms (ar, copy_ar) run with HIPFIRE_QWEN4_MTP_APPEND_DET=0
(AR never calls append_rows; it is the unchanged-law reference); every MTP arm runs with =1 under --det 1 (the gate) or =0 under
--det 0 (optional control).

  fabric-run.sh --class timing --card halo --hostmem-lock --mem-gb 110 --max-secs 1500 --wait-secs 14400 \
     --priority P? --owner ID -- python3 -I <worktree>/competitors/halo-mtp-regress-20261012/mtp-append-det/sampled_arm.py \
     --out ROOT/runs/<stamp>-sampled-dist-batched --testbin ROOT/testbin/sampled_mtp_distribution_hw --det 1 --arms dist:batched
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gatelib as G

TABLE = {
    "dist:ar": ("sampled_mtp_distribution_arm", None, {}, "ar.ids.json", True),
    "dist:batched": ("sampled_mtp_distribution_arm", "0", {}, "batched.ids.json", False),
    "dist:interleaved": ("sampled_mtp_distribution_arm", "1", {}, "interleaved.ids.json", False),
    "identity:ar": ("sampled_mtp_identity_arm", None, {"HIPFIRE_MTP_SAMPLED_MODE": "naive"}, "ar.identity.json", True),
    "identity:batched": ("sampled_mtp_identity_arm", "0", {"HIPFIRE_MTP_SAMPLED_MODE": "naive"}, "batched.identity.json", False),
    "identity:interleaved": ("sampled_mtp_identity_arm", "1", {"HIPFIRE_MTP_SAMPLED_MODE": "naive"}, "interleaved.identity.json", False),
    "identity:adaptive": ("sampled_mtp_identity_arm", None, {"HIPFIRE_MTP_SAMPLED_MODE": "naive"}, "adaptive.identity.json", False),
    "ngram:copy_ar": ("sampled_mtp_ngram_distribution_arm", None, {}, "copy_ar.ids.json", True),
    "ngram:ngram_batched": ("sampled_mtp_ngram_distribution_arm", "0", {}, "ngram_batched.ids.json", False),
    "ngram:ngram_interleaved": ("sampled_mtp_ngram_distribution_arm", "1", {}, "ngram_interleaved.ids.json", False),
    "ngram:copy_batched": ("sampled_mtp_ngram_distribution_arm", "0", {}, "copy_batched.ids.json", False),
    "ngram:copy_interleaved": ("sampled_mtp_ngram_distribution_arm", "1", {}, "copy_interleaved.ids.json", False),
}
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--testbin", required=True)
ap.add_argument("--arms", required=True, help="comma list of kind:arm (see table)")
ap.add_argument("--det", type=int, choices=(0, 1), default=1)
ap.add_argument("--sampled-root", default=os.path.join(G.ROOT, "sampled"), help="persistent OUT dirs: <root>/dist-det<N>, <root>/identity-det<N>")
ap.add_argument("--kcache", default=os.path.join(G.ROOT, "kcache"))
ap.add_argument("--max-secs", type=int, default=1500)
ap.add_argument("--allow-jit", action="store_true")
a = ap.parse_args()
if os.environ.get("HIPFIRE_SAMPLED_MTP_TRIALS"):
    if int(os.environ["HIPFIRE_SAMPLED_MTP_TRIALS"]) < 2000:
        sys.exit("refusing HIPFIRE_SAMPLED_MTP_TRIALS < 2000")
arms = a.arms.split(",")
for k in arms:
    if k not in TABLE:
        sys.exit(f"unknown arm {k}; known: {sorted(TABLE)}")
h = G.Hold("sampled-" + "+".join(arms).replace(":", "-"), a.out, a.max_secs)
R = h.receipt
R.update(arms=arms, det=a.det, testbin=a.testbin)
ok = G.preflight_common(h, None, [a.testbin])
if not ok:
    h.finish("preflight_failed")
    print("PREFLIGHT FAILED", json.dumps({k: R.get(k) for k in ("model_identity", "device_probe", "git", "fixtures_bad")}, indent=1)[:2500])
    sys.exit(2)
rc_all = 0
for key in arms:
    test, incr, extra_arm, idsfile, is_ref = TABLE[key]
    kind, arm = key.split(":")
    outdir = os.path.join(a.sampled_root, ("identity" if kind == "identity" else "dist") + f"-det{a.det}")
    os.makedirs(outdir, exist_ok=True)
    rec = {"key": key, "test": test, "out_dir": outdir, "ids_file": idsfile, "reference_arm": is_ref}
    R["cells"].append(rec)
    if os.path.exists(os.path.join(outdir, idsfile)):
        rec["skipped"] = "ids file already present (reused by the parent test; delete it to rerun)"
        continue
    if not G.hold_budget_ok(h, 300):
        rec["skipped"] = "insufficient hold budget or terminated"
        rc_all = rc_all or 5
        continue
    home = os.path.join(h.out, "home-" + key.replace(":", "-"))
    G.write_config(home)
    arm_env = {"HIPFIRE_MTP_IDENTITY_MODEL": G.MODEL, "HIPFIRE_SAMPLED_MTP_OUT": outdir, "HIPFIRE_SAMPLED_MTP_ARM": arm, "HIPFIRE_MTP_SAMPLED": "1"}
    arm_env.update(extra_arm)
    if incr is not None:
        arm_env["HIPFIRE_MTP_INCREMENTAL"] = incr
    arm_env[G.DET_ENV] = "1" if a.det == 1 and not is_ref else "0"
    env = G.clean_env(home, a.kcache, arm_env, allow_jit=a.allow_jit)
    cmd = [a.testbin, "--exact", test, "--ignored", "--nocapture", "--test-threads=1"]
    rec.update(argv=cmd, env=G.selected_env(env), det_applied=env.get(G.DET_ENV) == "1", start_utc=G.utc())
    h.save()
    t0 = time.monotonic()
    timeout = int(h.remaining() - 45)
    with open(os.path.join(outdir, arm + ".out"), "wb") as so, open(os.path.join(outdir, arm + ".err"), "wb") as se:
        p = h.spawn(cmd, env, so, se)
        rec["pid"] = p.pid
        h.save()
        rc, info = h.wait(p, timeout)
    rec.update(rc=rc, wait=info, wall_s=time.monotonic() - t0, end_utc=G.utc())
    out = open(os.path.join(outdir, arm + ".out"), errors="replace").read()
    err = open(os.path.join(outdir, arm + ".err"), errors="replace").read()
    rec["arch_lines"] = [l[:200] for l in (out + err).splitlines() if "arch=" in l or l.startswith("GPU dev")][:6]
    rec["arch_ok"] = any("gfx1151" in l for l in rec["arch_lines"])
    rec["trials_line"] = next((l for l in out.splitlines() if " trials in " in l or " cases in " in l), None)
    ids = os.path.join(outdir, idsfile)
    if os.path.exists(ids):
        rec["ids_md5"] = G.md5_file(ids)
        try:
            j = json.load(open(ids))
            rec["ids_len"] = len(j)
        except Exception as e:
            rec["ids_error"] = repr(e)
        cfg = ids.replace(".json", ".config.json")
        rec["config_sidecar"] = os.path.exists(cfg)
        if kind != "identity" and rec.get("ids_len", 0) < 2000:
            rec["trials_below_2000"] = True
            rc_all = rc_all or 6
    if rc != 0 or not os.path.exists(ids):
        rc_all = rc_all or (rc or 7)
    h.save()
    if h.terminated:
        break
h.finish("complete" if rc_all == 0 else f"incomplete_rc{rc_all}")
print("SAMPLED_ARMS_DONE", R["status"], json.dumps([{k: c.get(k) for k in ("key", "rc", "wall_s", "ids_len", "skipped", "det_applied")} for c in R["cells"]]))
sys.exit(0 if rc_all == 0 else 1)
