#!/usr/bin/env python3
"""No-GPU preflight (run on hipx outside any hold, from the clean Git worktree): clean-worktree check (HEAD recorded),
fixtures.json MD5 verification of every committed prompt/harness file, external binary/test-binary MD5s, model stat
(size/inode), rocminfo arch+BDF. Optional --hash-model computes the 125 GB md5 (I/O only; do NOT do this inside a hold).
Optional --expect-commit SHA fails when the worktree HEAD differs. Exit 0 only when everything required is OK."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gatelib as G
ap = argparse.ArgumentParser()
ap.add_argument("--hash-model", action="store_true")
ap.add_argument("--expect-commit")
ap.add_argument("--require-artifacts", action="store_true", help="also fail when bin/ testbin/ files are missing")
a = ap.parse_args()
git = G.git_state()
out = {"git": git, "fixtures_bad": G.fixtures_check(), "model": G.model_identity(), "bin": {}, "testbin": {}, "scripts_dir": G.SCRIPTS_DIR,
       "fixtures_dir": G.FIXTURES, "external_root": G.ROOT}
for n in ("daemon", "hipfire", "hipfire-weight-host"):
    p = os.path.join(G.ROOT, "bin", n)
    out["bin"][n] = G.md5_file(p) if os.path.exists(p) else None
for n in ("sampled_mtp_distribution_hw", "mtp_takeover_fill_hw", "qwen4_mtp_fill"):
    p = os.path.join(G.ROOT, "testbin", n)
    out["testbin"][n] = G.md5_file(p) if os.path.exists(p) else None
kc = os.path.join(G.ROOT, "kcache", "gfx1151")
out["kcache_entries"] = len(os.listdir(kc)) if os.path.isdir(kc) else 0
out["device_probe"] = G.probe_device(G.clean_env("/tmp", os.path.join(G.ROOT, "kcache")))
if a.hash_model:
    t = time.monotonic()
    out["model_md5"] = G.md5_file(G.MODEL)
    out["model_md5_ok"] = out["model_md5"] == G.MODEL_MD5
    out["model_md5_s"] = time.monotonic() - t
bad = (not git["clean"]) or out["fixtures_bad"] or not out["model"]["ok"] or not out["device_probe"]["ok"]
if a.expect_commit and git["head"] != a.expect_commit:
    out["commit_mismatch"] = {"expected": a.expect_commit, "head": git["head"]}
    bad = True
if a.require_artifacts and (any(v is None for v in out["bin"].values()) or any(v is None for v in out["testbin"].values())):
    bad = True
print(json.dumps(out, indent=1))
sys.exit(1 if bad else 0)
