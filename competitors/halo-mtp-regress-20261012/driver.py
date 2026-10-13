#!/usr/bin/env python3
"""Decode-sweep receipt driver: mixed-serve.py plus SSE streaming and MTP phase receipts.

Standalone (not in the product repo).  Imports the existing one-cell harness
(`mixed-serve.py`, via importlib), monkeypatches its module-level `http_json`, and
calls its `main()` unchanged, so every harness argument, the process/device
assertions, the cell deadline and the cleanup stay exactly the harness's.

  driver.py --out DIR --manifest manifest.json [any mixed-serve.py arguments...] \
      [--harness PATH] [--verify-labels L1,L2,...] [--phase-slack-s 0.05]

Driver-only arguments (stripped from argv BEFORE the harness's argparse runs):
  --harness PATH        harness file to import (default: mixed-serve.py next to this file)
  --verify-labels CSV   phase labels counted as "verify", first present in a window wins
                        (default: target_verify,verify,target_row; see mtp_spec.rs notes)
  --phase-slack-s S     attribution slack around [t_send_wall, t_send_wall + wall_s] (default 0.05)
Everything else (including --out, --manifest, --env, --tag, ...) is passed through untouched.

Streaming: when a manifest item's request_extra has "stream": true the request body carries
stream=true (and stream_options.include_usage=true unless the manifest set stream_options),
the POST is consumed as SSE with urllib, and the deltas are folded back into the ordinary
non-stream response dict {choices[0].message.{content,reasoning_content,tool_calls},
finish_reason, usage, timings, hipfire, ...} (+ "stream_client" client-side stream stats), so
the harness's normal recording / result dict / text files work unchanged.
A server `error` chunk or a stream that ends without a finish_reason is returned as HTTP 502
(the harness records it as a request error); the partially assembled response is kept.

Phase receipts (post-main, also after harness failure): serve.log lines
`QWEN4_MTP_PHASE {json}` and `QWEN4_MTP_TRACE {json}` are attributed to requests by the
phase line's `t_end` (unix microseconds) falling inside [t_send_wall, t_send_wall + wall_s]
(plus slack) of each requests.jsonl record.  The serve must run with
HIPFIRE_MTP_PHASE_TIMING=1 (phase lines) and HIPFIRE_MTP_TRACE=1 (trace lines), e.g. via
--env HIPFIRE_MTP_PHASE_TIMING=1 --env HIPFIRE_MTP_TRACE=1.  A trace line has no timestamp: it
is paired with the next phase line of the same (source, verify_route, position), so trace
without phase timing cannot be attributed to a request (kept under unattributed_trace).
Outputs, all under --out:
  cell.json            harness cell.json, each measured result extended with phase metrics,
                       plus top-level "driver" and "phase_summary" keys
  phase-summary.json   raw per-request window records (raw log line, parsed JSON, paired
                       trace) for every request incl. warmup, plus every unattributed /
                       malformed MTP line; no raw line is dropped
"""
import importlib.util, json, os, re, sys, time, traceback, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_HARNESS = os.path.join(HERE, "mixed-serve.py")
DEF_VERIFY_LABELS = ("target_verify", "verify", "target_row")
LINE_RE = re.compile(r"QWEN4_MTP_(PHASE|TRACE) (\{.*\})\s*$")
TAG_RE = re.compile(r"QWEN4_MTP_(PHASE|TRACE)")


# --------------------------------------------------------------------------- argv handling
def split_args(argv):
    """Remove driver-only options; return (own, rest, out_dir)."""
    own = {"harness": DEF_HARNESS, "verify_labels": DEF_VERIFY_LABELS, "slack": 0.05}
    rest, out = [], None
    i = 0
    while i < len(argv):
        a = argv[i]

        def val(name):
            nonlocal i
            if a == name:
                if i + 1 >= len(argv):
                    raise SystemExit(f"driver: {name} needs a value")
                i += 1
                return argv[i]
            return a.split("=", 1)[1]

        if a in ("--harness", "--verify-labels", "--phase-slack-s") or a.startswith(("--harness=", "--verify-labels=", "--phase-slack-s=")):
            name = a.split("=", 1)[0]
            v = val(name)
            if name == "--harness":
                own["harness"] = os.path.abspath(v)
            elif name == "--verify-labels":
                own["verify_labels"] = tuple(x for x in v.split(",") if x)
            else:
                own["slack"] = float(v)
        else:
            rest.append(a)
            if a == "--out" and i + 1 < len(argv):
                out = argv[i + 1]
            elif a.startswith("--out="):
                out = a.split("=", 1)[1]
        i += 1
    return own, rest, out


# --------------------------------------------------------------------------- SSE streaming
def _plain_http(url, data, headers, timeout):
    req = urllib.request.Request(url, data, headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw.decode("utf-8", "replace")}


def _merge_tool_call(acc, tc):
    idx = tc.get("index", 0)
    cur = acc.setdefault(idx, {"index": idx, "id": None, "type": "function", "function": {"name": "", "arguments": ""}})
    if tc.get("id"):
        cur["id"] = tc["id"]
    if tc.get("type"):
        cur["type"] = tc["type"]
    fn = tc.get("function") or {}
    if fn.get("name"):
        cur["function"]["name"] += fn["name"]
    if fn.get("arguments"):
        cur["function"]["arguments"] += fn["arguments"]


def stream_chat(url, body, timeout):
    """POST `body` (stream=true) and fold the SSE stream into a non-stream response dict."""
    injected = False
    if "stream_options" not in body:
        body["stream_options"] = {"include_usage": True}
        injected = True
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data, {"Content-Type": "application/json", "Accept": "text/event-stream"})
    t0 = time.monotonic()
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw.decode("utf-8", "replace")}

    content, reasoning, tools = [], [], {}
    top = {"id": None, "object": "chat.completion", "created": None, "model": None}
    role, finish = "assistant", None
    carried = {"usage": None, "timings": None, "hipfire": None}
    st = {"chunks": 0, "data_events": 0, "content_deltas": 0, "reasoning_deltas": 0, "keepalive_lines": 0,
          "non_json_events": 0, "extra_choice_chunks": 0, "done_seen": False, "first_delta_s": None,
          "first_content_s": None, "first_reasoning_s": None, "last_event_s": None, "bytes": 0,
          "stream_options_injected": injected, "http_status": r.status, "first_byte_s": None,
          "headers_s": time.monotonic() - t0}
    error_chunk = None
    data_lines = []

    def dispatch():
        nonlocal role, finish, error_chunk
        if not data_lines:
            return
        payload = "\n".join(data_lines)
        data_lines.clear()
        st["data_events"] += 1
        if payload.strip() == "[DONE]":
            st["done_seen"] = True
            return
        try:
            ch = json.loads(payload)
        except Exception:
            st["non_json_events"] += 1
            st.setdefault("non_json_samples", []).append(payload[:300])
            return
        now = time.monotonic() - t0
        st["chunks"] += 1
        st["last_event_s"] = now
        if not isinstance(ch, dict):
            return
        if "error" in ch and error_chunk is None:
            error_chunk = ch["error"]
        for k in ("id", "created", "model"):
            if ch.get(k) is not None and top[k] is None:
                top[k] = ch[k]
        for k in carried:
            if ch.get(k) is not None:
                carried[k] = ch[k]
        for c in ch.get("choices") or []:
            if (c.get("index") or 0) != 0:
                st["extra_choice_chunks"] += 1
                continue
            d = c.get("delta") or {}
            if d.get("role"):
                role = d["role"]
            if d.get("content"):
                content.append(d["content"])
                st["content_deltas"] += 1
                if st["first_content_s"] is None:
                    st["first_content_s"] = now
            if d.get("reasoning_content"):
                reasoning.append(d["reasoning_content"])
                st["reasoning_deltas"] += 1
                if st["first_reasoning_s"] is None:
                    st["first_reasoning_s"] = now
            if (d.get("content") or d.get("reasoning_content")) and st["first_delta_s"] is None:
                st["first_delta_s"] = now
            for tc in d.get("tool_calls") or []:
                _merge_tool_call(tools, tc)
            if c.get("finish_reason"):
                finish = c["finish_reason"]

    try:
        with r:
            for raw in r:
                if st["first_byte_s"] is None:
                    st["first_byte_s"] = time.monotonic() - t0
                st["bytes"] += len(raw)
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if line == "":
                    dispatch()
                elif line.startswith(":"):
                    st["keepalive_lines"] += 1
                elif line.startswith("data:"):
                    v = line[5:]
                    data_lines.append(v[1:] if v.startswith(" ") else v)
                # event:/id:/retry: fields carry nothing we need
            dispatch()
    finally:
        st["stream_s"] = time.monotonic() - t0

    msg = {"role": role, "content": "".join(content), "reasoning_content": "".join(reasoning) or None}
    if tools:
        msg["tool_calls"] = [tools[i] for i in sorted(tools)]
    j = {"id": top["id"], "object": "chat.completion", "created": top["created"], "model": top["model"],
         "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
         "usage": carried["usage"], "timings": carried["timings"], "hipfire": carried["hipfire"],
         "stream_client": st}
    if error_chunk is not None:
        j["error"] = error_chunk
        return 502, j
    if finish is None:
        j["error"] = {"message": "stream ended without finish_reason", "done_seen": st["done_seen"]}
        return 502, j
    return r.status, j


def make_http_json(orig):
    def http_json(url, body=None, timeout=60):
        if isinstance(body, dict) and body.get("stream") is True and url.endswith("/v1/chat/completions"):
            return stream_chat(url, body, timeout)
        return orig(url, body, timeout)
    return http_json


# --------------------------------------------------------------------------- phase parsing
def mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def parse_log(path, verify_labels):
    """Return (phase_windows, unpaired_trace, malformed, nlines).  Nothing is dropped."""
    try:
        text = open(path, "rb").read().decode("utf-8", "replace")
    except OSError:
        return [], [], [], 0
    lines = text.split("\n")
    windows, malformed, pending = [], [], []
    unpaired_trace = []
    for n, line in enumerate(lines, 1):
        line = line.rstrip("\r")
        if not TAG_RE.search(line):
            continue
        m = LINE_RE.search(line)
        obj = None
        if m:
            try:
                obj = json.loads(m.group(2))
            except Exception:
                obj = None
        if not m or obj is None or not isinstance(obj, dict):
            malformed.append({"log_line": n, "raw": line})
            continue
        if m.group(1) == "TRACE":
            pending.append({"log_line": n, "raw": line, "parsed": obj})
            continue
        phases = obj.get("phases_us") or {}
        vlabel = next((l for l in verify_labels if isinstance(phases.get(l), dict) and num(phases[l].get("us")) is not None), None)
        w = {"log_line": n, "raw": line, "parsed": obj,
             "t_end_s": (obj["t_end"] / 1e6) if num(obj.get("t_end")) is not None else None,
             "verify_label": vlabel, "verify_us": phases[vlabel]["us"] if vlabel else None, "trace": None}
        key = (obj.get("source"), obj.get("verify_route"), obj.get("position"))
        for idx in range(len(pending) - 1, -1, -1):
            tp = pending[idx]["parsed"]
            if (tp.get("source"), tp.get("verify_route"), tp.get("position")) == key:
                w["trace"] = pending[idx]
                unpaired_trace.extend(pending[:idx])
                del pending[: idx + 1]
                break
        windows.append(w)
    unpaired_trace.extend(pending)
    return windows, unpaired_trace, malformed, len(lines)


def attribute(windows, reqs, slack):
    """Assign each window to a request index (or None) by t_end; strict first, then slack."""
    spans = []
    for i, r in enumerate(reqs):
        t0 = r.get("t_send_wall")
        wall = r.get("wall_s")
        if t0 is None:
            spans.append(None)
            continue
        if wall is not None:
            t1 = t0 + wall
        else:
            nxt = next((q.get("t_send_wall") for q in reqs[i + 1:] if q.get("t_send_wall") is not None), None)
            t1 = nxt if nxt is not None else float("inf")
        spans.append((t0, t1))
    owner = []
    for w in windows:
        te = w["t_end_s"]
        hit = None
        if te is not None:
            for i, s in enumerate(spans):
                if s and s[0] <= te <= s[1]:
                    hit = i
                    break
            if hit is None:
                best = None
                for i, s in enumerate(spans):
                    if not s:
                        continue
                    d = max(s[0] - te, te - s[1], 0.0)
                    if d <= slack and (best is None or d < best[0]):
                        best = (d, i)
                hit = best[1] if best else None
        owner.append(hit)
    return owner, spans


def counter(xs):
    out = {}
    for x in xs:
        out[str(x)] = out.get(str(x), 0) + 1
    return out


SPEC_SOURCES = ("mtp", "ngram")


def window_metrics(wins, timings, rec=None):
    """Means/verify stats cover mtp+ngram windows only (AR/calibration excluded);
    *_all fields and source_counts/route_counts cover every phase line of the request."""
    spec = [w for w in wins if w["parsed"].get("source") in SPEC_SOURCES]
    ps = [w["parsed"] for w in spec]
    allp = [w["parsed"] for w in wins]
    vus = [w["verify_us"] for w in spec if w["verify_us"] is not None]
    native_k = [p.get("k") for p in ps if p.get("source") == "mtp" and p.get("verify_route") == "batched" and num(p.get("k")) is not None]
    mtp_k = [p.get("k") for p in ps if p.get("source") == "mtp" and num(p.get("k")) is not None]
    em = [p.get("emitted") for p in ps if num(p.get("emitted")) is not None]
    ac = [p.get("accepted") for p in ps if num(p.get("accepted")) is not None]
    wu = [p.get("window_us") for p in ps if num(p.get("window_us")) is not None]
    wu_all = [p.get("window_us") for p in allp if num(p.get("window_us")) is not None]
    em_all = [p.get("emitted") for p in allp if num(p.get("emitted")) is not None]
    t = timings if isinstance(timings, dict) else {}
    kinds = {}
    for r_ in t.get("mtp_window_timings") or []:
        if not isinstance(r_, dict):
            continue
        k = kinds.setdefault(str(r_.get("kind")), {"n": 0})
        k["n"] += 1
        for key, v in r_.items():
            if key != "kind" and num(v) is not None:
                k[key] = k.get(key, 0) + v
    rec = rec or {}
    resp = rec.get("response") if isinstance(rec.get("response"), dict) else {}
    sc = resp.get("stream_client") if isinstance(resp.get("stream_client"), dict) else {}
    usage = rec.get("usage") if isinstance(rec.get("usage"), dict) else (resp.get("usage") if isinstance(resp.get("usage"), dict) else {})
    ctok, wall = num(usage.get("completion_tokens")), num(rec.get("wall_s"))
    return {
        "phase_windows": len(spec),
        "phase_windows_all": len(wins),
        "emitted_per_window": mean(em),
        "emitted_total": sum(em) if em else None,
        "emitted_total_all": sum(em_all) if em_all else None,
        "accepted_per_window": mean(ac),
        "accepted_total": sum(ac) if ac else None,
        "verify_ms_mean": (mean(vus) / 1000.0) if vus else None,
        "verify_ms_total": (sum(vus) / 1000.0) if vus else None,
        "verify_windows": len(vus),
        "verify_label_counts": counter(w["verify_label"] for w in spec if w["verify_label"]),
        "window_ms_mean": (mean(wu) / 1000.0) if wu else None,
        "window_ms_total": (sum(wu) / 1000.0) if wu else None,
        "window_ms_mean_all": (mean(wu_all) / 1000.0) if wu_all else None,
        "window_ms_total_all": (sum(wu_all) / 1000.0) if wu_all else None,
        "source_counts": counter(p.get("source") for p in allp),
        "route_counts": counter(p.get("verify_route") for p in allp),
        "route_counts_spec": counter(p.get("verify_route") for p in ps),
        "native_mean_k": mean(native_k),
        "native_windows": len(native_k),
        "mtp_mean_k": mean(mtp_k),
        "trace_windows_paired": sum(1 for w in wins if w["trace"]),
        "retirement": {k: t.get(k) for k in ("mtp_retired", "mtp_windows", "ar_windows")},
        "ngram_timing": {**{k: t.get(k) for k in ("mtp_ngram", "ngram_mod_windows", "ngram_mod_drafts",
                                                  "ngram_mod_accepted", "ngram_mod_accept_rate")},
                         "mtp_window_timings_n": len(t.get("mtp_window_timings") or []),
                         "mtp_window_timings_by_kind": kinds},
        # client-side definitions: wall_s = POST send -> last byte (monotonic, whole request,
        # incl. prefill); wall_tok_s = completion_tokens / wall_s; stream_* only for SSE requests.
        "client_wall_s": wall,
        "client_wall_tok_s": (ctok / wall) if ctok is not None and wall else None,
        "stream_first_byte_s": sc.get("first_byte_s"),
        "stream_first_delta_s": sc.get("first_delta_s"),
        "stream_first_content_s": sc.get("first_content_s"),
        "stream_total_s": sc.get("stream_s"),
        "stream_decode_tok_s": (ctok / (sc["stream_s"] - sc["first_delta_s"]))
                               if ctok and sc.get("stream_s") and sc.get("first_delta_s") is not None
                               and sc["stream_s"] > sc["first_delta_s"] else None,
    }


def postprocess(out, own, main_rc, argv_rest):
    cell_path = os.path.join(out, "cell.json")
    if not os.path.isfile(cell_path):
        print(f"driver: no cell.json under {out}; phase summary skipped", file=sys.stderr)
        return
    cell = json.load(open(cell_path))
    reqs, bad_req_lines = [], []
    rp = os.path.join(out, "requests.jsonl")
    if os.path.isfile(rp):
        for n, line in enumerate(open(rp, encoding="utf-8", errors="replace"), 1):
            if not line.strip():
                continue
            try:
                reqs.append(json.loads(line))
            except Exception:
                bad_req_lines.append({"line": n, "raw": line.rstrip("\n")})
    logp = os.path.join(out, "serve.log")
    windows, unpaired_trace, malformed, nlines = parse_log(logp, own["verify_labels"])
    owner, spans = attribute(windows, reqs, own["slack"])

    per_req, by_seq = [], {}
    for i, r in enumerate(reqs):
        wins = [w for w, o in zip(windows, owner) if o == i]
        resp = r.get("response")
        timings = r.get("timings") if isinstance(r.get("timings"), dict) else (resp.get("timings") if isinstance(resp, dict) else None)
        met = window_metrics(wins, timings, r)
        s = spans[i]
        ent = {"seq": r.get("seq"), "name": r.get("name"), "kind": r.get("kind"), "t_send_wall": r.get("t_send_wall"),
               "wall_s": r.get("wall_s"), "attribution_span_s": list(s) if s else None, "error": r.get("error"),
               "stream": bool((r.get("request") or {}).get("stream")), "metrics": met, "windows": wins}
        per_req.append(ent)
        by_seq[r.get("seq")] = met
    un_phase = [{**w, "reason": "no t_end" if w["t_end_s"] is None else "t_end outside every request span"}
                for w, o in zip(windows, owner) if o is None]

    n_attr = sum(1 for o in owner if o is not None)
    summary = {
        "generated_wall": time.time(), "out": out, "serve_log": logp, "serve_log_lines": nlines,
        "harness": own["harness"], "verify_labels": list(own["verify_labels"]), "slack_s": own["slack"],
        "main_rc": main_rc, "cell_status": cell.get("status"),
        "totals": {"phase_lines": len(windows), "attributed": n_attr, "unattributed_phase": len(un_phase),
                   "trace_unpaired": len(unpaired_trace), "malformed": len(malformed),
                   "requests": len(reqs), "bad_request_lines": len(bad_req_lines)},
        "warnings": ([] if windows else ["no QWEN4_MTP_PHASE lines in serve.log: run with --env HIPFIRE_MTP_PHASE_TIMING=1"])
                    + ([] if unpaired_trace or any(w["trace"] for w in windows) else
                        ["no QWEN4_MTP_TRACE lines: run with --env HIPFIRE_MTP_TRACE=1 if traces are wanted"]),
        "requests": per_req,
        "unattributed_phase": un_phase,
        "unattributed_trace": unpaired_trace,
        "malformed_lines": malformed,
        "bad_request_lines": bad_req_lines,
    }
    # imported harness's wr() would work too, but keep the driver self-contained
    def wr(path, obj):
        with open(path, "w") as f:
            json.dump(obj, f, indent=1, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

    wr(os.path.join(out, "phase-summary.json"), summary)

    for res in cell.get("results") or []:
        met = by_seq.get(res.get("seq"))
        if met is not None:
            res.update(met)
    cell["phase_summary"] = {"path": "phase-summary.json", "totals": summary["totals"], "warnings": summary["warnings"]}
    cell["driver"] = {"path": os.path.abspath(__file__), "harness": own["harness"], "verify_labels": list(own["verify_labels"]),
                      "slack_s": own["slack"], "harness_argv": argv_rest, "main_rc": main_rc}
    wr(cell_path, cell)
    print(f"phase summary: {summary['totals']} -> {os.path.join(out, 'phase-summary.json')}", flush=True)


# --------------------------------------------------------------------------- entry
def main():
    own, rest, out = split_args(sys.argv[1:])
    spec = importlib.util.spec_from_file_location("mixed_serve_harness", own["harness"])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mixed_serve_harness"] = mod
    spec.loader.exec_module(mod)
    mod.http_json = make_http_json(mod.http_json)
    sys.argv = [own["harness"]] + rest
    rc = None
    try:
        rc = mod.main()
        return rc
    except SystemExit as e:  # harness argparse error / explicit exit
        rc = e.code
        raise
    finally:
        if out is not None:
            try:
                postprocess(os.path.abspath(out), own, rc, rest)
            except Exception:
                traceback.print_exc()
                try:
                    with open(os.path.join(os.path.abspath(out), "phase-summary-error.txt"), "w") as f:
                        f.write(traceback.format_exc())
                except Exception:
                    pass


if __name__ == "__main__":
    sys.exit(main())
