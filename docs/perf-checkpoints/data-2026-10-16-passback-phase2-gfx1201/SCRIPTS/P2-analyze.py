#!/bin/bash
# P2-analyze.py — summarize P2 fixed-share sweep JSONs: per-arm process medians,
# min/med/max, population sd, and skip failed (rc!=0 / empty JSON) samples.
# Usage: P2-analyze.py <rawdir> (e.g. raw/P2-9B)
import json, glob, os, statistics, sys
d = sys.argv[1]
arms = {}
for p in sorted(glob.glob(os.path.join(d, "R*-*.json"))):
    base = os.path.basename(p)[:-5]
    arm = base.split("-", 1)[1]
    try:
        r = json.load(open(p))
        runs = (r.get("samples") or {}).get("decode")
        if not runs:
            print(f"{base}: EMPTY-JSON (load refused or failed run)")
            continue
        med = statistics.median(runs)
        arms.setdefault(arm, []).append((base, med, runs))
    except Exception as e:
        print(f"{base}: UNREADABLE {e}")
for arm in sorted(arms):
    meds = [m for _, m, _ in arms[arm]]
    print(f"arm={arm} n={len(meds)} medians={sorted(meds)} min/med/max={min(meds)}/{statistics.median(meds)}/{max(meds)} pstdev={statistics.pstdev(meds):.3f}")
    for base, med, runs in arms[arm]:
        print(f"   {base}: runs={runs} med={med}")
