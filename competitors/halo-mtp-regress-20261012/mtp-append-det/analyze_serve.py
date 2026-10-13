#!/usr/bin/env python3
"""CPU-only analysis of a serve_cells.py output dir (cells d0-rN / d1-rN, each a mixed-serve cell).

greedy: per matched prompt, DET0 vs DET1 (same binary):
  * complete UTF-8 content + reasoning_content, finish_reason, full usage dict: byte-identical to DET0 rep1 and across reps
  * tau (server timings.tau = accepted drafts / cycle) and route counts (mtp_windows, ar_windows, mtp_retired, ngram_mod_*,
    cycles) per rep; exact drafted/accepted/route per window from QWEN4_MTP_TRACE when present (attributed by position
    continuity; reported as unattributed, never guessed, when segment count != request count)
  * verdict per prompt (never aggregated):
      PASS | FAIL_TEXT | FAIL_TAU_LOSS | INCONCLUSIVE_ROUTE | FAIL_CELL
    FAIL_TAU_LOSS: any DET1 rep tau < median(DET0 taus) on a prompt whose route class (retired?, any AR window?) is
    identical in all reps of both arms. A prompt whose route class differs between arms is INCONCLUSIVE_ROUTE (no tau claim).
ttft: per-arm streaming TTFT samples (POST -> first non-empty content delta), paired deltas, medians. No threshold is applied.
Exit 0 only if every prompt PASSes (greedy) / all cells valid (ttft).
"""
import argparse, glob, hashlib, json, os, re, statistics, sys

TRACE = re.compile(r"QWEN4_MTP_TRACE (\{.*\})\s*$")


def load_cell(d):
    c = {"dir": d}
    p = os.path.join(d, "cell.json")
    c["cell"] = json.load(open(p)) if os.path.exists(p) else None
    reqs = []
    rp = os.path.join(d, "requests.jsonl")
    if os.path.exists(rp):
        reqs = [json.loads(l) for l in open(rp) if l.strip()]
    c["reqs"] = reqs
    c["trace"] = []
    sp = os.path.join(d, "serve.log")
    if os.path.exists(sp):
        for line in open(sp, errors="replace"):
            m = TRACE.search(line)
            if m:
                try:
                    c["trace"].append(json.loads(m.group(1)))
                except Exception:
                    pass
    return c


def segment_trace(trace, nreq_candidates):
    """Split by position continuity: window n+1 continues window n iff position == pos + len(committed)."""
    segs = []
    for i, w in enumerate(trace):
        if i and trace[i - 1]["position"] + len(trace[i - 1]["committed"]) == w["position"]:
            segs[-1].append(w)
        else:
            segs.append([w])
    return segs


def trace_summary(seg):
    by = {}
    for w in seg:
        k = f"{w.get('source')}/{w.get('verify_route')}"
        b = by.setdefault(k, {"windows": 0, "drafted": 0, "accepted": 0})
        b["windows"] += 1
        b["drafted"] += int(w.get("k", 0))
        b["accepted"] += int(w.get("accepted", 0))
    tot = {"windows": len(seg), "drafted": sum(b["drafted"] for b in by.values()), "accepted": sum(b["accepted"] for b in by.values())}
    return {"by_source_route": by, "total": tot}


def measured(c):
    return [r for r in c["reqs"] if r.get("kind") == "measured"]


def route_sig(t):
    t = t or {}
    return {k: t.get(k) for k in ("cycles", "mtp_windows", "ar_windows", "mtp_retired", "ngram_mod_windows", "ngram_mod_drafts",
                                  "ngram_mod_accepted")}


def route_class(t):
    t = t or {}
    return (bool(t.get("mtp_retired")), (t.get("ar_windows") or 0) > 0)


def utf8_ok(s):
    if s is None:
        return True
    try:
        s.encode("utf-8")
    except UnicodeError:
        return False
    return "\ufffd" not in s


def greedy(out, hist=None):
    cells = {}
    for d in sorted(glob.glob(os.path.join(out, "d[01]-r*"))):
        if os.path.isdir(d):
            cells[os.path.basename(d)] = load_cell(d)
    rep = {"out": out, "cells": {}, "prompts": {}, "verdict": None}
    for n, c in cells.items():
        cj = c["cell"] or {}
        env_det = ((cj.get("env_hipfire_keys") or {}).get("HIPFIRE_QWEN4_MTP_APPEND_DET"))
        want_det = "1" if n.startswith("d1") else "0"
        rep["cells"][n] = {"status": cj.get("status"), "device_ok": (cj.get("assertions", {}).get("device") or {}).get("ok"),
                           "det_env": env_det, "det_env_ok": env_det == want_det, "n_measured": len(measured(c)),
                           "errors": cj.get("errors"), "binary_md5": {k: (cj.get(k) or {}).get("md5") for k in ("hipfire", "daemon")},
                           "model_inode": (cj.get("model_stat") or {}).get("inode"), "start_iso": cj.get("start_iso")}
        # trace attribution
        tr = c["trace"]
        c["attr"] = None
        if tr:
            segs = segment_trace(tr, None)
            reqs = c["reqs"]
            if len(segs) == len(reqs):
                c["attr"] = {id(r): s for r, s in zip(reqs, segs)}
                rep["cells"][n]["trace_attribution"] = "position-continuity segments == requests (incl. warmup)"
            elif len(segs) == len(measured(c)):
                c["attr"] = {id(r): s for r, s in zip(measured(c), segs)}
                rep["cells"][n]["trace_attribution"] = "position-continuity segments == measured requests (warmup untraced)"
            else:
                rep["cells"][n]["trace_attribution"] = f"UNATTRIBUTED segments={len(segs)} requests={len(reqs)}"
        rep["cells"][n]["trace_windows"] = len(tr)
    bad_cells = [n for n, v in rep["cells"].items() if v["status"] != "ok" or not v["device_ok"] or not v["det_env_ok"] or v["n_measured"] != 5]
    names = []
    for c in cells.values():
        for r in measured(c):
            if r["name"] not in names:
                names.append(r["name"])
    all_pass = bool(cells) and not bad_cells
    for p in names:
        row = {"d0": [], "d1": []}
        for n, c in sorted(cells.items()):
            for r in measured(c):
                if r["name"] != p:
                    continue
                t = r.get("timings") or {}
                rec = {"cell": n, "content_md5": r.get("content_md5"), "finish": r.get("finish"), "usage": r.get("usage"),
                       "reasoning_md5": hashlib.md5((r.get("reasoning_content") or "").encode()).hexdigest() if r.get("reasoning_content") is not None else None,
                       "reasoning_content": r.get("reasoning_content"), "content": r.get("content"), "tau": t.get("tau"),
                       "route": route_sig(t), "route_class": route_class(t), "decode_tok_s": t.get("decode_tok_s"),
                       "http_status": r.get("http_status"), "error": r.get("error"),
                       "utf8_ok": utf8_ok(r.get("content")) and utf8_ok(r.get("reasoning_content")),
                       "trace": trace_summary(c["attr"][id(r)]) if c.get("attr") and id(r) in c["attr"] else None}
                row[n[:2]].append(rec)
        problems = []
        if len(row["d0"]) < 3 or len(row["d1"]) < 3:
            problems.append(f"need >=3 reps per arm, have d0={len(row['d0'])} d1={len(row['d1'])}")
        ref = row["d0"][0] if row["d0"] else None
        identical = True
        if ref:
            for rec in row["d0"] + row["d1"]:
                for k in ("content", "reasoning_content", "finish", "usage"):
                    if rec[k] != ref[k]:
                        identical = False
                        problems.append(f"{rec['cell']} {k} differs from {ref['cell']}")
                if not rec["utf8_ok"] or rec["error"] or rec["http_status"] != 200:
                    identical = False
                    problems.append(f"{rec['cell']} invalid response (utf8_ok={rec['utf8_ok']} error={rec['error']} http={rec['http_status']})")
        classes = {rec["route_class"] for rec in row["d0"] + row["d1"]}
        t0 = [x["tau"] for x in row["d0"] if x["tau"] is not None]
        t1 = [x["tau"] for x in row["d1"] if x["tau"] is not None]
        med0 = statistics.median(t0) if t0 else None
        losses = [x["cell"] + f" tau {x['tau']} < D0 median {med0}" for x in row["d1"] if x["tau"] is not None and med0 is not None and x["tau"] < med0 - 1e-9]
        if not identical:
            verdict = "FAIL_TEXT"
        elif len(classes) > 1:
            verdict = "INCONCLUSIVE_ROUTE"
            problems.append(f"route classes (retired, any_ar_window) differ across reps/arms: {sorted(map(str, classes))}")
        elif losses:
            verdict = "FAIL_TAU_LOSS"
            problems += losses
        elif problems or len(t0) < 3 or len(t1) < 3:
            verdict = "FAIL_CELL"
        else:
            verdict = "PASS"
        da = lambda arm: sorted({(x["trace"] or {}).get("total", {}).get("drafted") for x in row[arm]} - {None})
        aa = lambda arm: sorted({(x["trace"] or {}).get("total", {}).get("accepted") for x in row[arm]} - {None})
        rep["prompts"][p] = {"verdict": verdict, "problems": problems, "tau_d0": t0, "tau_d1": t1, "tau_d0_median": med0,
                             "tau_d1_median": statistics.median(t1) if t1 else None,
                             "routes_d0": [x["route"] for x in row["d0"]], "routes_d1": [x["route"] for x in row["d1"]],
                             "trace_drafted_d0": da("d0"), "trace_drafted_d1": da("d1"), "trace_accepted_d0": aa("d0"),
                             "trace_accepted_d1": aa("d1"), "content_md5_d0": sorted({x["content_md5"] for x in row["d0"]}),
                             "content_md5_d1": sorted({x["content_md5"] for x in row["d1"]}),
                             "completion_tokens": sorted({(x["usage"] or {}).get("completion_tokens") for x in row["d0"] + row["d1"]}),
                             "finish": sorted({x["finish"] for x in row["d0"] + row["d1"]}),
                             "decode_tok_s_d0": [x["decode_tok_s"] for x in row["d0"]], "decode_tok_s_d1": [x["decode_tok_s"] for x in row["d1"]],
                             "trace_by_source_route_d0": [(x["trace"] or {}).get("by_source_route") for x in row["d0"]],
                             "trace_by_source_route_d1": [(x["trace"] or {}).get("by_source_route") for x in row["d1"]]}
        if verdict != "PASS":
            all_pass = False
    rep["bad_cells"] = bad_cells
    rep["verdict"] = "PASS" if all_pass and names else "NOT_PASS"
    return rep


def ttft(out):
    cells = {os.path.basename(d): load_cell(d) for d in sorted(glob.glob(os.path.join(out, "d[01]-r*"))) if os.path.isdir(d)}
    rep = {"out": out, "cells": {}, "arms": {"d0": [], "d1": []}}
    ok = bool(cells)
    for n, c in cells.items():
        cj = c["cell"] or {}
        m = measured(c)
        r = m[0] if m else {}
        sc = (r.get("response") or {}).get("stream_client") or {}
        u = r.get("usage") or {}
        t = r.get("timings") or {}
        rec = {"status": cj.get("status"), "device_ok": (cj.get("assertions", {}).get("device") or {}).get("ok"),
               "ttft_s": sc.get("first_content_s"), "prompt_tokens": u.get("prompt_tokens"),
               "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "server_ttft_ms": t.get("ttft_ms"),
               "prefill_tok_s": t.get("prefill_tok_s"), "finish": r.get("finish"), "wall_s": r.get("wall_s"), "start_iso": cj.get("start_iso"),
               "det_env": (cj.get("env_hipfire_keys") or {}).get("HIPFIRE_QWEN4_MTP_APPEND_DET")}
        rep["cells"][n] = rec
        good = rec["status"] == "ok" and rec["device_ok"] and rec["ttft_s"] is not None and rec["cached_tokens"] == 0 and \
            rec["det_env"] == ("1" if n.startswith("d1") else "0")
        ok = ok and good
        rec["valid"] = good
        if good:
            rep["arms"][n[:2]].append((n, rec["ttft_s"] * 1000.0))
    s = {a: [v for _, v in sorted(rep["arms"][a])] for a in rep["arms"]}
    rep["samples_ms"] = s
    for a in s:
        if s[a]:
            rep[a + "_median_ms"] = statistics.median(s[a])
            rep[a + "_mean_ms"] = statistics.mean(s[a])
    rr = lambda a: {n.split("-r")[1]: v for n, v in rep["arms"][a]}
    paired = {k: rr("d1")[k] - rr("d0")[k] for k in rr("d0") if k in rr("d1")}
    rep["paired_delta_ms_d1_minus_d0"] = paired
    if s["d0"] and s["d1"]:
        rep["median_delta_ms"] = rep["d1_median_ms"] - rep["d0_median_ms"]
        rep["median_delta_pct"] = 100.0 * rep["median_delta_ms"] / rep["d0_median_ms"]
    rep["valid_cells"] = ok and len(s["d0"]) >= 3 and len(s["d1"]) >= 3
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--mode", choices=("greedy", "ttft"), required=True)
    a = ap.parse_args(argv)
    rep = greedy(a.out) if a.mode == "greedy" else ttft(a.out)
    p = os.path.join(a.out, "analysis.json")
    json.dump(rep, open(p, "w"), indent=1, sort_keys=True, default=str)
    if a.mode == "greedy":
        for pn, v in rep["prompts"].items():
            print(f"{pn}: {v['verdict']} tau d0={v['tau_d0']} d1={v['tau_d1']} routes d0={[(r['mtp_windows'], r['ar_windows'], r['mtp_retired']) for r in v['routes_d0']]} "
                  f"d1={[(r['mtp_windows'], r['ar_windows'], r['mtp_retired']) for r in v['routes_d1']]} {'; '.join(v['problems'])[:300]}")
        print("GREEDY_VERDICT", rep["verdict"], "bad_cells", rep["bad_cells"])
        return 0 if rep["verdict"] == "PASS" else 1
    print(json.dumps({k: rep.get(k) for k in ("samples_ms", "d0_median_ms", "d1_median_ms", "median_delta_ms", "median_delta_pct",
                                                "paired_delta_ms_d1_minus_d0", "valid_cells")}, indent=1))
    return 0 if rep["valid_cells"] else 1


if __name__ == "__main__":
    sys.exit(main())
