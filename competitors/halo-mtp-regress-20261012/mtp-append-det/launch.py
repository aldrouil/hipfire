#!/usr/bin/env python3
"""Detached launcher for ONE hold (or CPU-only final step) from gates.json. Run from the DET worktree (hiptrx). Dry by default:
prints the exact fabric argv. It queues nothing unless BOTH --execute and --go "<verbatim parent GO text>" are given.

Source contract: the hold drivers run from a clean Git worktree ON THE REMOTE HOST (hipx), never from rsync/scp copies.
  --repo   absolute path of that remote worktree (required); the driver path is
           <repo>/<this script's directory relative to its own worktree root>/<driver>  (derived from __file__, not hard-coded)
  --commit full SHA the remote worktree must be at (required to execute; the remote `git status --porcelain
           --untracked-files=all` must also be empty). Every driver re-checks cleanliness and records HEAD in its receipt.
External (untracked) areas stay under gates.json "root": bin/, testbin/, kcache/, runs/, sampled/. Launch receipts go to
--receipts-dir (default: external qcal model-gates/launches), never into the repo.
  launch.py --list
  launch.py --hold A-greedy-tau --repo /path/to/remote/worktree --priority P1 --owner HaloShepherd.MtpAppendDet   # dry
  launch.py --hold A-greedy-tau --repo ... --commit SHA --priority P1 --owner NAME --go "HaloShepherd GO ..." --execute
  launch.py --final B-final-dist --repo ...      # prints the CPU-only (GPUs hidden) final-step command; never queues
"""
import argparse, json, os, shlex, subprocess, sys, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
FABRIC = "/home/kaden/qcal/locks/fabric/fabric-run.sh"
RECEIPTS = "/home/kaden/qcal/release-0.4.2/halo3000/mtp-append-det/model-gates/launches"


def worktree_rel(path):
    """Directory of this script relative to the root of the Git worktree containing it (found by walking up to `.git`)."""
    p = path
    while p != os.path.dirname(p):
        if os.path.exists(os.path.join(p, ".git")):
            return os.path.relpath(path, p)
        p = os.path.dirname(p)
    sys.exit(f"{path} is not inside a Git worktree (no .git found)")


ap = argparse.ArgumentParser()
ap.add_argument("--list", action="store_true")
ap.add_argument("--hold")
ap.add_argument("--final")
ap.add_argument("--repo", default="/home/kaden/pm-wave/mtp-append-det/source", help="absolute path of the clean detached Git worktree on the remote host")
ap.add_argument("--commit")
ap.add_argument("--go")
ap.add_argument("--execute", action="store_true")
ap.add_argument("--priority", help="P0..P3 (fabric scheduler is live: pass it with --owner)")
ap.add_argument("--owner")
ap.add_argument("--host", default="hipx")
ap.add_argument("--receipts-dir", default=RECEIPTS)
ap.add_argument("--extra-arg", action="append", default=[], help="extra driver arg appended verbatim (repeatable), e.g. --extra-arg=--allow-jit")
a = ap.parse_args()
g = json.load(open(os.path.join(HERE, "gates.json")))
if a.list or not (a.hold or a.final):
    for h in g["holds"]:
        print(f"{h['id']:34s} max_secs={h['max_secs']:5d} est_s={h['est_s']} mem_gb={h['mem_gb']}")
    for s in g["final_cpu_steps"]:
        print(f"{s['id']:34s} CPU-only final step (--final)")
    sys.exit(0)
if not a.repo:
    sys.exit("--repo (remote clean Git worktree path) is required")
rel = worktree_rel(HERE)
driver_dir = os.path.join(a.repo, rel)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
if a.final:
    s = next((x for x in g["final_cpu_steps"] if x["id"] == a.final), None)
    if s is None:
        sys.exit(f"unknown final step {a.final}")
    cmd = ["env", "ROCR_VISIBLE_DEVICES=-1", "HIP_VISIBLE_DEVICES=-1", "nice", "-n", "19", "python3", "-I", os.path.join(driver_dir, s["driver"])] + s["args"]
    print("FINAL (CPU-only, run on", a.host + "):", " ".join(map(shlex.quote, cmd)))
    sys.exit(0)
h = next((x for x in g["holds"] if x["id"] == a.hold), None)
if h is None:
    sys.exit(f"unknown hold {a.hold}")
out = f"{g['root']}/runs/{stamp}-{h['id']}"
argv = [FABRIC, "--class", "timing", "--card", "halo"]
if h.get("hostmem_lock"):
    argv.append("--hostmem-lock")
argv += ["--mem-gb", "0", "--max-secs", str(h["max_secs"]), "--wait-secs", str(h["wait_secs"])]
if h.get("hostmem_lock"):
    argv += ["--hostmem-gb", str(min(h["mem_gb"], 100))]
if a.priority:
    argv += ["--priority", a.priority]
if a.owner:
    argv += ["--owner", a.owner]
argv += ["--", "python3", "-I", os.path.join(driver_dir, h["driver"]), "--out", out] + h["args"] + a.extra_arg
log = out + ".nohup.log"
remote = f"mkdir -p {shlex.quote(g['root'] + '/runs')} && cd {shlex.quote(g['root'])} && {{ nohup setsid {' '.join(map(shlex.quote, argv))} > {shlex.quote(log)} 2>&1 < /dev/null & echo $!; }}"
print("HOLD", h["id"], "est_s", h["est_s"], "basis:", h["est_basis"])
print("DRIVER_DIR (remote Git worktree)", driver_dir)
print("ARGV", " ".join(map(shlex.quote, argv)))
if not (a.execute and a.go):
    print("DRY RUN (no queue call). Needs --execute, --go '<GO text>' and --commit SHA.")
    sys.exit(0)
if not a.commit:
    sys.exit("--commit SHA is required to execute")
chk = subprocess.run(["ssh", a.host, f"git -C {shlex.quote(a.repo)} rev-parse HEAD && git -C {shlex.quote(a.repo)} status --porcelain --untracked-files=all"],
                     capture_output=True, text=True, timeout=120)
lines = chk.stdout.splitlines()
if chk.returncode != 0 or not lines or lines[0] != a.commit or len(lines) > 1:
    sys.exit(f"remote worktree check failed (rc={chk.returncode}, head={lines[:1]}, dirty={lines[1:6]}, stderr={chk.stderr[:300]}); nothing queued")
r = subprocess.run(["ssh", a.host, remote], capture_output=True, text=True, timeout=60)
rec = {"stamp": stamp, "hold": h["id"], "pid": r.stdout.strip(), "rc": r.returncode, "log": log, "out": out, "argv": argv, "GO": a.go,
       "repo": a.repo, "commit": a.commit, "est_s": h["est_s"], "max_secs": h["max_secs"]}
os.makedirs(a.receipts_dir, exist_ok=True)
json.dump(rec, open(os.path.join(a.receipts_dir, f"{stamp}-{h['id']}.json"), "w"), indent=2)
print(json.dumps(rec, indent=2))
sys.exit(r.returncode)
