#!/usr/bin/env python3
"""FIN-analyze.py — summarize official native-bench FINAL campaign samples.
Usage: FIN-analyze.py <dir> <prefix>  (e.g. FIN9B R / FIN27B R)
Reads <dir>/<prefix>*-{orig-pb,fin-pb,fin-cpu,fin-pcie}.json, prints per-arm
process-sample medians + median/mean/sd/CV/range, and original->FINAL deltas.
ONE process sample = median of its internal --runs samples (already in JSON stats)."""
import json, glob, os, statistics, sys

def load(p):
    r = json.load(open(p))
    s = r.get("samples", {})
    l = r.get("loaded", {})
    return {
        "decode": s.get("decode", []),
        "prefill": s.get("prefill", []),
        "ttft": s.get("ttft_ms", []),
        "kv": (l.get("kv_mode"), l.get("kv_backend"), l.get("max_seq")),
        "ptok": r.get("prompt_tokens"),
        "warn": r.get("warnings", []),
    }

def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return None
    m = statistics.median(xs)
    return {"n": len(xs), "med": m, "mean": statistics.mean(xs),
            "sd": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
            "cv": (statistics.pstdev(xs)/statistics.mean(xs) if len(xs) > 1 and statistics.mean(xs) else 0.0),
            "min": min(xs), "max": max(xs)}

def main(d, prefix):
    arms = {}
    for arm in ["orig-pb", "fin-pb", "fin-cpu", "fin-pcie"]:
        files = sorted(glob.glob(os.path.join(d, f"{prefix}*-{arm}.json")))
        rows = []
        for f in files:
            try: rows.append((f, load(f)))
            except Exception as e: print(f"UNREADABLE {f}: {e}")
        # one process sample = median of internal runs
        pmeds = [statistics.median(r["decode"]) for _, r in rows if r["decode"]]
        ppref = [statistics.median(r["prefill"]) for _, r in rows if r["prefill"]]
        pttt = [statistics.median(r["ttft"]) for _, r in rows if r["ttft"]]
        arms[arm] = {"files": files, "rows": rows, "dec": stats(pmeds),
                     "pref": stats(ppref), "ttft": stats(pttt)}
        print(f"--- {arm} ({len(files)} process samples) ---")
        for f, r in rows:
            dm = statistics.median(r["decode"]) if r["decode"] else None
            print(f"  {os.path.basename(f)} internal_decode={r['decode']} med={dm} kv={r['kv']} ptok={r['ptok']}")
        print(f"  DECODE process-medians: {pmeds} -> {arms[arm]['dec']}")
        print(f"  PREFILL process-medians: {ppref} -> {arms[arm]['pref']}")
        print(f"  TTFT(ms) process-medians: {pttt} -> {arms[arm]['ttft']}")
    o, f, c, p = (arms[a]["dec"] for a in ["orig-pb", "fin-pb", "fin-cpu", "fin-pcie"])
    if o and f:
        print(f"ORIG->FINAL: {o['med']:.2f} -> {f['med']:.2f} = {(f['med']/o['med']-1)*100:+.1f}% "
              f"(orig spread {o['min']:.2f}-{o['max']:.2f}, final spread {f['min']:.2f}-{f['max']:.2f})")
    if f and c:
        print(f"FINALpb vs CPU: {f['med']:.2f} vs {c['med']:.2f} = {(f['med']/c['med']-1)*100:+.1f}%")
    if f and p:
        print(f"FINALpb vs PCIe: {f['med']:.2f} vs {p['med']:.2f} = {(f['med']/p['med']-1)*100:+.1f}%")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
