#!/usr/bin/env python3
"""Append-per-chunk device attribution from a rocprofv3 --kernel-trace CSV (MTP-APPEND-PROFILE method).

An Append subchunk is the kernel run [embedding_q8_batched, hyper_norm_f32, ..., indexed_attention_pool_rope_f32]
whose first hyper_norm_f32 has Grid_Size_Y > 1 (batched prompt-fill rows; the single-row decode append has Grid_Y 1) and
that contains hyper_read_projected_f32. Contiguous subchunks form one prompt-fill group (one request's chunk). The LAST
group of a bench --warmups 1 --runs 1 trace is the measured request. Append ms = sum(End-Start) of every kernel launch
inside those windows (busy ms, identical to the 189.419264 ms historical baseline method); span ms is first start to last end.
Re-validate on the historical trace:  analyze_append.py <on/kt/kt_kernel_trace.csv> --expect-ms 189.419264
"""
import argparse, csv, glob, json, os, sys

INCUMBENT = ["gemm_bf16_xf32_multirow_r16w4t_gfx1151", "gemm_bf16_xf32_multirow_r16w4_gfx1151", "gemv_bf16_xf32_k4_rows_tiled_r8",
             "gemm_bf16_xf32_multirow_pto2", "gemv_q8_0_k2560_staged_rows_tiled"]
DET_ENTRY = "mtp_dense_det_pm_gfx1151_w8"
DET_Q8_SHADOW = "mtp_q8_0_to_f16_pm_gfx1151"
DET_ACT = ["convert_f32_to_f16", "hc_activation_fused_f32"]


def load(path):
    rows = list(csv.DictReader(open(path)))
    rows.sort(key=lambda r: (int(r["Start_Timestamp"]), int(r["Dispatch_Id"])))
    return rows


def find_trace(p):
    if os.path.isdir(p):
        c = glob.glob(os.path.join(p, "**", "*kernel_trace.csv"), recursive=True)
        if len(c) != 1:
            raise SystemExit(f"expected exactly one kernel_trace.csv under {p}, found {c}")
        return c[0]
    return p


def windows(rows):
    ws = []
    n = len(rows)
    for s, r in enumerate(rows):
        if r["Kernel_Name"] != "embedding_q8_batched" or s + 1 >= n:
            continue
        nx = rows[s + 1]
        if nx["Kernel_Name"] != "hyper_norm_f32" or int(nx["Grid_Size_Y"]) <= 1:
            continue
        e = next((j for j in range(s, n) if rows[j]["Kernel_Name"] == "indexed_attention_pool_rope_f32"), None)
        if e is None:
            continue
        w = rows[s:e + 1]
        if not any(x["Kernel_Name"] == "hyper_read_projected_f32" for x in w):
            continue
        ws.append((s, e, int(nx["Grid_Size_Y"])))
    return ws


def groups(ws):
    gs = []
    for w in ws:
        if gs and gs[-1][-1][1] + 1 == w[0]:
            gs[-1].append(w)
        else:
            gs.append([w])
    return gs


def summarize(rows, g):
    ker = {}
    busy = 0
    launches = 0
    for s, e, _ in g:
        for x in rows[s:e + 1]:
            d = int(x["End_Timestamp"]) - int(x["Start_Timestamp"])
            k = ker.setdefault(x["Kernel_Name"], {"launches": 0, "ms": 0.0, "grids": set()})
            k["launches"] += 1
            k["ms"] += d / 1e6
            k["grids"].add(f"{x['Grid_Size_X']}x{x['Grid_Size_Y']}x{x['Grid_Size_Z']}")
            busy += d
            launches += 1
    t0 = int(rows[g[0][0]]["Start_Timestamp"])
    t1 = int(rows[g[-1][1]]["End_Timestamp"])
    return {"subchunks": len(g), "rows": [w[2] for w in g], "rows_total": sum(w[2] for w in g), "launches": launches,
            "append_busy_ms": busy / 1e6, "append_span_ms": (t1 - t0) / 1e6,
            "kernels": {k: {"launches": v["launches"], "ms": v["ms"], "ms_per_launch": v["ms"] / v["launches"], "grids": sorted(v["grids"])}
                        for k, v in sorted(ker.items(), key=lambda kv: -kv[1]["ms"])}}


def analyze(path, arm=None, prompt_rows=None):
    rows = load(find_trace(path))
    gs = groups(windows(rows))
    res = {"trace": find_trace(path), "kernel_rows": len(rows), "groups": [summarize(rows, g) for g in gs]}
    if not gs:
        res["error"] = "no batched Append windows found"
        return res
    m = res["groups"][-1]
    res["measured"] = m
    names = set(m["kernels"])
    res["incumbent_present_in_measured"] = sorted(n for n in INCUMBENT if n in names)
    res["det_dense_launches_in_measured"] = m["kernels"].get(DET_ENTRY, {}).get("launches", 0)
    res["det_act_kernels_in_measured"] = {n: m["kernels"].get(n, {}).get("launches", 0) for n in DET_ACT}
    # one-time shadow work anywhere in the trace (outside or inside windows)
    sh = [r for r in rows if r["Kernel_Name"] == DET_Q8_SHADOW]
    res["q8_shadow_kernel"] = {"launches": len(sh), "ms": sum(int(r["End_Timestamp"]) - int(r["Start_Timestamp"]) for r in sh) / 1e6}
    if prompt_rows is not None:
        res["rows_match_prompt_tokens"] = m["rows_total"] == prompt_rows
    if arm == "d1":
        res["route_ok"] = (not res["incumbent_present_in_measured"]) and res["det_dense_launches_in_measured"] > 0
    elif arm == "d0":
        res["route_ok"] = len(res["incumbent_present_in_measured"]) == len(INCUMBENT) and res["det_dense_launches_in_measured"] == 0
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--arm", choices=("d0", "d1"))
    ap.add_argument("--prompt-rows", type=int)
    ap.add_argument("--expect-ms", type=float, help="assert measured busy ms equals this (6 dp), e.g. historical 189.419264")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    r = analyze(a.trace, a.arm, a.prompt_rows)
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1, sort_keys=True)
    m = r.get("measured")
    if m:
        print(f"groups={len(r['groups'])} measured: subchunks={m['subchunks']} rows={m['rows']} launches={m['launches']} "
              f"append_busy_ms={m['append_busy_ms']:.6f} span_ms={m['append_span_ms']:.6f} route_ok={r.get('route_ok')}")
    if a.expect_ms is not None:
        ok = m is not None and abs(m["append_busy_ms"] - a.expect_ms) < 5e-7
        print("EXPECT_MS", "OK" if ok else f"MISMATCH got {m and m['append_busy_ms']}")
        return 0 if ok else 1
    return 0 if m else 1


if __name__ == "__main__":
    sys.exit(main())
