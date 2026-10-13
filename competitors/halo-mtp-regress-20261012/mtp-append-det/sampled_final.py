#!/usr/bin/env python3
"""CPU-only (GPUs hidden, no hold): re-run the EXACT ignored parent tests of sampled_mtp_distribution_hw. With every arm's
ids/identity file already present they only compare (chi-square homogeneity per position and (t1,t2) joint, ALPHA 1e-3, replay
and window-counter checks, naive seeded-ID identity, n-gram hit requirements). Provenance of each arm (DET applied or not) is
printed from the per-arm hold receipts. Do not run before all arms exist (the parent would otherwise launch GPU arms)."""
import argparse, glob, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gatelib as G
T = {"dist": "sampled_mtp_matches_ar_distribution_on_flash_next", "identity": "naive_sampled_mtp_emits_seeded_ar_ids",
     "ngram": "sampled_mtp_ngram_matches_ar_distribution_on_flash_next"}
NEED = {"dist": ["ar.ids.json", "batched.ids.json", "interleaved.ids.json"],
        "identity": ["ar.identity.json", "batched.identity.json", "interleaved.identity.json", "adaptive.identity.json"],
        "ngram": ["ar.ids.json", "copy_ar.ids.json", "ngram_batched.ids.json", "ngram_interleaved.ids.json", "copy_batched.ids.json",
                  "copy_interleaved.ids.json"]}
ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=sorted(T), required=True)
ap.add_argument("--det", type=int, choices=(0, 1), default=1)
ap.add_argument("--testbin", required=True)
ap.add_argument("--sampled-root", default=os.path.join(G.ROOT, "sampled"))
ap.add_argument("--receipts", default=os.path.join(G.ROOT, "runs", "*sampled*", "hold-receipt.json"))
ap.add_argument("--out", required=True)
a = ap.parse_args()
d = os.path.join(a.sampled_root, ("identity" if a.kind == "identity" else "dist") + f"-det{a.det}")
missing = [f for f in NEED[a.kind] if not os.path.exists(os.path.join(d, f))]
if missing:
    sys.exit(f"missing arm files in {d}: {missing}; run those arms first (this step must not start GPU arms)")
os.makedirs(a.out, exist_ok=True)
env = dict(os.environ)
for k in list(env):
    if k.startswith(("HIPFIRE_", "HSA_", "ROCR_", "HIP_VISIBLE")):
        env.pop(k)
env.update(ROCR_VISIBLE_DEVICES="-1", HIP_VISIBLE_DEVICES="-1", HIPFIRE_MTP_IDENTITY_MODEL=G.MODEL, HIPFIRE_SAMPLED_MTP_OUT=d)
prov = []
for rp in sorted(glob.glob(a.receipts)):
    try:
        r = json.load(open(rp))
    except Exception:
        continue
    for c in r.get("cells", []):
        if c.get("out_dir") == d and "rc" in c:
            prov.append({"receipt": rp, "key": c["key"], "rc": c["rc"], "det_applied": c.get("det_applied"), "reference_arm": c.get("reference_arm"),
                         "start_utc": c.get("start_utc"), "ids_md5": c.get("ids_md5"), "ids_len": c.get("ids_len"), "arch_ok": c.get("arch_ok"),
                         "git_head": (r.get("git") or {}).get("head"), "git_clean": (r.get("git") or {}).get("clean")})
cmd = [a.testbin, "--exact", T[a.kind], "--ignored", "--nocapture", "--test-threads=1"]
p = subprocess.run(cmd, env=env, capture_output=True, text=True)
rec = {"utc": G.utc(), "argv": cmd, "rc": p.returncode, "kind": a.kind, "det": a.det, "dir": d, "provenance": prov,
       "git": G.git_state(), "testbin_md5": G.md5_file(a.testbin), "stdout_tail": p.stdout[-4000:], "stderr_tail": p.stderr[-6000:]}
for rp in ("report.txt", "identity-report.txt", "ngram-report.txt"):
    f = os.path.join(d, rp)
    if os.path.exists(f):
        rec[rp] = open(f).read()
G.wr(os.path.join(a.out, f"final-{a.kind}-det{a.det}.json"), rec)
print(rec.get("report.txt") or rec.get("identity-report.txt") or rec.get("ngram-report.txt") or p.stderr[-3000:])
print("SAMPLED_FINAL", a.kind, "PASS" if p.returncode == 0 else f"FAIL rc={p.returncode}")
sys.exit(p.returncode)
