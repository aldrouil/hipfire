#!/usr/bin/env python3
"""Compare qwen4_mtp_fill JSONs: ignoring ONLY the top-level `batched_fill` route label (state_parity.rs run_mtp_fill_digest),
DET1 bf0 vs bf1 must be identical for every length (the gate); DET0 bf0 vs bf1 is the incumbent control; DET1 vs DET0 is
informational (DET numerics are allowed to differ from the incumbent)."""
import json, os, sys

def load(d, n):
    p = os.path.join(d, n, "fill.json")
    return json.load(open(p)) if os.path.exists(p) else None

def strip(j):
    j = dict(j)
    j.pop("batched_fill", None)
    return j

def diff_rows(a, b):
    out = {}
    ra = {r["length"]: r for r in a["rows"]}
    rb = {r["length"]: r for r in b["rows"]}
    for L in sorted(set(ra) | set(rb)):
        if L not in ra or L not in rb:
            out[L] = ["missing"]
            continue
        keys = [k for k in sorted(set(ra[L]) | set(rb[L])) if ra[L].get(k) != rb[L].get(k)]
        out[L] = keys
    other = [k for k in set(a) | set(b) if k not in ("rows", "batched_fill") and a.get(k) != b.get(k)]
    if other:
        out["top_level"] = other
    return out

def pair(d, x, y):
    a, b = load(d, x), load(d, y)
    if a is None or b is None:
        return {"error": f"missing {x if a is None else y}"}
    df = diff_rows(a, b)
    return {"equal": strip(a) == strip(b), "differing_fields_by_length": {str(k): v for k, v in df.items() if v},
            "lengths": [r["length"] for r in a["rows"]], "batched_fill_labels": [a.get("batched_fill"), b.get("batched_fill")]}

def compare_dir(d):
    r = {"det1_bf0_vs_bf1": pair(d, "det1-bf0", "det1-bf1"), "det0_bf0_vs_bf1_control": pair(d, "det0-bf0", "det0-bf1"),
         "det1_vs_det0_bf1_informational": pair(d, "det1-bf1", "det0-bf1")}
    if os.path.exists(os.path.join(d, "det1-bf0-spread")):
        r["det1_spread_bf0_vs_bf1"] = pair(d, "det1-bf0-spread", "det1-bf1-spread")
    gate = [r["det1_bf0_vs_bf1"]] + ([r["det1_spread_bf0_vs_bf1"]] if "det1_spread_bf0_vs_bf1" in r else [])
    want = [1, 15, 16, 17, 255, 256, 257]
    r["lengths_ok"] = all(g.get("lengths") == want for g in gate)
    r["verdict"] = "PASS" if all(g.get("equal") for g in gate) and r["lengths_ok"] else "NOT_PASS"
    return r

if __name__ == "__main__":
    res = compare_dir(sys.argv[1])
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["verdict"] == "PASS" else 1)
