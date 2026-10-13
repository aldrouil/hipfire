#!/usr/bin/env python3
"""One fresh `hipfire serve` per invocation (one cell); runs a manifest of prompts non-streaming.

Standalone receipt driver (not in the repo). Pattern follows
halo-decode-ab-20261012/drive.py, plus full artifacts and assertions.

  mixed-serve.py --out DIR --manifest manifest.json \
      [--hipfire /home/kaden/pm-wave/halo-mtp-regress/bin-current/hipfire] \
      [--daemon  /home/kaden/pm-wave/halo-mtp-regress/bin-current/daemon] \
      [--model   /home/kaden/pm-wave/fngptq/models/qwen3.8-flash-next-gptq3.mq4] \
      [--port 18761] [--hash-model] [--timeout 900]

Manifest: JSON list of objects
  {"name": str, "prompt_path": str,
   "max_tokens": int            (default 256),
   "thinking_enabled": bool     (default false -> chat_template_kwargs.enable_thinking=false),
   "temperature": float         (default 0),
   "request_extra": {..}        (optional; shallow-merged into the request body last)}

Request body (default): {"model": MODEL, "messages": [{"role":"user","content": PROMPT_TEXT}],
  "max_tokens": 256, "temperature": 0, "chat_template_kwargs": {"enable_thinking": false}}
Warmup (unrecorded as a measurement, but recorded in the artifacts):
  {"model": MODEL, "messages": [{"role":"user","content":"Say hi."}], "max_tokens": 8, "temperature": 0}
No max_seq is passed: the serve/daemon default applies; the load log + /health n_ctx are recorded.

Prompt file bytes are decoded as UTF-8 with no newline translation and no strip; md5/sha256
are of the raw file bytes. Prompt token count = response usage.prompt_tokens.

Order within one invocation: [hash model] -> start serve -> wait for health 200 + model resident
-> assert load log has gfx1151 and 0000:c0:00.0 -> warmup -> manifest prompts in order
-> cleanup (SIGTERM process group, wait 180s, SIGKILL fallback).

Artifacts under --out: cell.json (summary), requests.jsonl (request + full response per call),
text/NN-name.txt (full content), serve.log (serve stdout+stderr merged), env.json, manifest.json,
health-before.json / health-after.json.
Exit: 0 ok; 2 assertion/start failure; 3 request error(s); 4 timeout/signal.
The --timeout (default 900s) is a whole-cell wall deadline starting AFTER the optional model hash
(the hash has its own --hash-timeout).
"""
import argparse, hashlib, json, os, re, signal, subprocess, sys, time, urllib.request, urllib.error

DEF_HIPFIRE = "/home/kaden/pm-wave/halo-mtp-regress/bin-current/hipfire"
DEF_DAEMON = "/home/kaden/pm-wave/halo-mtp-regress/bin-current/daemon"
DEF_MODEL = "/home/kaden/pm-wave/fngptq/models/qwen3.8-flash-next-gptq3.mq4"
EXPECT_SHA = "8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972"


class CellTimeout(BaseException):
    pass


class CellSignal(BaseException):
    pass


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wr(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def http_json(url, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw.decode("utf-8", "replace")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--hipfire", default=DEF_HIPFIRE)
    ap.add_argument("--daemon", default=DEF_DAEMON)
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--port", type=int, default=18761)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--timeout", type=int, default=900, help="whole-cell wall seconds (after hash)")
    ap.add_argument("--health-timeout", type=int, default=600)
    ap.add_argument("--hash-model", action="store_true", help="sha256 the model once before serve start")
    ap.add_argument("--expect-sha256", default=EXPECT_SHA)
    ap.add_argument("--hash-timeout", type=int, default=3600)
    ap.add_argument("--expect-arch", default="gfx1151")
    ap.add_argument("--expect-pci", default="0000:c0:00.0")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--env", action="append", default=[], metavar="K=V",
                    help="extra env for the serve child (recorded); repeatable")
    ap.add_argument("--tag", default="", help="free-form label stored in cell.json")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    os.makedirs(os.path.join(out, "text"), exist_ok=True)
    manifest = json.load(open(a.manifest))
    assert isinstance(manifest, list) and manifest, "manifest must be a non-empty JSON list"
    wr(os.path.join(out, "manifest.json"), manifest)

    cell = {"tag": a.tag, "argv": sys.argv, "start_wall": time.time(), "start_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "host": os.uname().nodename, "model": a.model, "port": a.port, "timeout_s": a.timeout,
            "status": "started", "assertions": {}, "results": [], "errors": []}

    def save():
        wr(os.path.join(out, "cell.json"), cell)

    # --- binaries / model identity -------------------------------------------------------
    for key, p in (("hipfire", a.hipfire), ("daemon", a.daemon)):
        cell[key] = {"path": p, "md5": md5_file(p), "size": os.stat(p).st_size}
    st = os.stat(a.model)
    cell["model_stat"] = {"size": st.st_size, "mtime": st.st_mtime, "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(st.st_mtime)),
                          "inode": st.st_ino, "realpath": os.path.realpath(a.model)}
    save()

    if a.hash_model:
        t0 = time.monotonic()
        h = hashlib.sha256()
        with open(a.model, "rb") as f:
            for b in iter(lambda: f.read(8 << 20), b""):
                h.update(b)
                if time.monotonic() - t0 > a.hash_timeout:
                    cell["status"] = "hash_timeout"
                    save()
                    print("FAIL: model hash exceeded --hash-timeout", file=sys.stderr)
                    return 2
        sha = h.hexdigest()
        cell["model_sha256"] = sha
        cell["model_sha256_s"] = time.monotonic() - t0
        ok = sha == a.expect_sha256
        cell["assertions"]["model_sha256"] = {"ok": ok, "got": sha, "expected": a.expect_sha256}
        save()
        print(f"model sha256 {sha} ({'OK' if ok else 'MISMATCH'}) in {cell['model_sha256_s']:.0f}s", flush=True)
        if not ok:
            cell["status"] = "model_sha_mismatch"
            save()
            return 2

    # --- prompts -----------------------------------------------------------------------
    prompts = []
    for i, item in enumerate(manifest):
        raw = open(item["prompt_path"], "rb").read()
        prompts.append({"i": i, "name": item["name"], "path": item["prompt_path"], "bytes": len(raw),
                        "md5": hashlib.md5(raw).hexdigest(), "sha256": hashlib.sha256(raw).hexdigest(),
                        "text": raw.decode("utf-8"), "item": item})
    cell["prompts"] = [{k: v for k, v in p.items() if k not in ("text", "item")} for p in prompts]

    # --- env ------------------------------------------------------------------------------
    env = dict(os.environ)
    env["HIPFIRE_DAEMON_BIN"] = a.daemon
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    wr(os.path.join(out, "env.json"), env)
    cell["extra_env"] = a.env
    cell["env_hipfire_keys"] = {k: v for k, v in sorted(env.items()) if k.startswith(("HIPFIRE", "HSA", "ROC", "HIP", "FCLK", "AMD"))}

    base = f"http://{a.host}:{a.port}"
    if http_probe(base):
        cell["status"] = "port_in_use"
        save()
        print(f"FAIL: something already answers /health on {base}", file=sys.stderr)
        return 2

    # --- start serve under the cell deadline ---------------------------------------------
    proc = None
    rc = 0
    logf = open(os.path.join(out, "serve.log"), "wb")

    def on_term(signum, _frm):
        raise CellSignal(f"signal {signum}")

    def on_alarm(_s, _f):
        raise CellTimeout(f"cell exceeded {a.timeout}s")

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    signal.signal(signal.SIGALRM, on_alarm)
    t_cell = time.monotonic()
    signal.alarm(a.timeout)
    try:
        cmd = [a.hipfire, "serve", f"{a.host}:{a.port}", a.model]
        cell["serve_cmd"] = cmd
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env=env, start_new_session=True)
        cell["serve_pid"] = proc.pid
        save()
        t0 = time.monotonic()
        health = None
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"serve died rc={proc.returncode} before ready")
            if time.monotonic() - t0 > a.health_timeout:
                raise RuntimeError("serve not ready within --health-timeout")
            try:
                code, h = http_json(base + "/health", timeout=5)
                # model prewarm at startup: need status up AND model resident (n_ctx set, not loading)
                if code == 200 and h.get("model") and not h.get("loading_model"):
                    health = h
                    break
            except Exception:
                pass
            time.sleep(2)
        cell["ready_s"] = time.monotonic() - t0
        wr(os.path.join(out, "health-before.json"), health)
        cell["health_before"] = health

        # --- load-log assertion (after health, before any request) ---
        logf.flush()
        logtxt = open(os.path.join(out, "serve.log"), "rb").read().decode("utf-8", "replace")
        lines = logtxt.splitlines()
        dev_lines = [l for l in lines if "[devices]" in l]
        pci = a.expect_pci.lower()
        match = [l for l in lines if a.expect_arch in l and pci in l.lower()]
        bad = [l for l in dev_lines if "logical" in l and not (a.expect_arch in l and pci in l.lower())]
        cell["load_log"] = {
            "device_lines": dev_lines,
            "arch_pci_match_lines": match,
            "offending_device_lines": bad,
            "ctx_lines": [l for l in lines if re.search(r"max_seq|n_ctx|kv_mode|kv_backend|model_ctx|card_cap|mtp|dflash", l, re.I)][:60],
        }
        ok = bool(match) and not bad
        cell["assertions"]["device"] = {"ok": ok, "expect_arch": a.expect_arch, "expect_pci": a.expect_pci}
        save()
        if not ok:
            cell["status"] = "device_assertion_failed"
            save()
            print(f"FAIL: serve log lacks a line with {a.expect_arch} and {a.expect_pci}, or has foreign device lines: {bad}", file=sys.stderr)
            rc = 2
            return rc
        print(f"device OK: {match[0]}", flush=True)
        print(f"ready in {cell['ready_s']:.0f}s n_ctx={health.get('n_ctx')}", flush=True)

        # --- requests -----------------------------------------------------------------------
        reqf = open(os.path.join(out, "requests.jsonl"), "a")
        seq = [0]

        def do_request(name, body, prompt_meta, kind):
            n = seq[0]
            seq[0] += 1
            rec = {"seq": n, "name": name, "kind": kind, "request": body, "prompt": prompt_meta,
                   "t_send_wall": time.time()}
            t1 = time.monotonic()
            try:
                remaining = max(30, a.timeout - (time.monotonic() - t_cell))
                code, j = http_json(base + "/v1/chat/completions", body, timeout=remaining)
                rec["wall_s"] = time.monotonic() - t1
                rec["http_status"] = code
                rec["response"] = j
                ch = (j.get("choices") or [{}])[0] if isinstance(j, dict) else {}
                msg = ch.get("message") or {}
                content = msg.get("content") or ""
                rec["content"] = content
                rec["reasoning_content"] = msg.get("reasoning_content")
                rec["finish"] = ch.get("finish_reason")
                rec["usage"] = j.get("usage") if isinstance(j, dict) else None
                rec["timings"] = j.get("timings") if isinstance(j, dict) else None
                rec["hipfire"] = j.get("hipfire") if isinstance(j, dict) else None
                rec["content_md5"] = hashlib.md5(content.encode()).hexdigest()
                with open(os.path.join(out, "text", f"{n:02d}-{name}.txt"), "w") as tf:
                    tf.write(content)
                    if msg.get("reasoning_content"):
                        tf.write("\n\n===== reasoning_content =====\n" + msg["reasoning_content"])
                if code != 200:
                    rec["error"] = f"http {code}"
            except (CellTimeout, CellSignal):
                raise
            except Exception as e:
                rec["wall_s"] = time.monotonic() - t1
                rec["error"] = repr(e)[:500]
            reqf.write(json.dumps(rec) + "\n")
            reqf.flush()
            os.fsync(reqf.fileno())
            return rec

        if not a.no_warmup:
            wbody = {"model": a.model, "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8, "temperature": 0}
            wrec = do_request("warmup_say_hi", wbody, None, "warmup")
            print(f"warmup: err={wrec.get('error')} timings={json.dumps(wrec.get('timings'))[:200]}", flush=True)
            cell["warmup"] = {k: wrec.get(k) for k in ("wall_s", "error", "usage", "timings", "finish", "content")}
            save()

        for p in prompts:
            it = p["item"]
            body = {"model": a.model, "messages": [{"role": "user", "content": p["text"]}],
                    "max_tokens": int(it.get("max_tokens", 256)),
                    "temperature": it.get("temperature", 0),
                    "chat_template_kwargs": {"enable_thinking": bool(it.get("thinking_enabled", False))}}
            if it.get("request_extra"):
                body.update(it["request_extra"])
            pm = {k: p[k] for k in ("name", "path", "bytes", "md5", "sha256")}
            rec = do_request(p["name"], body, pm, "measured")
            t = rec.get("timings") or {}
            u = rec.get("usage") or {}
            r = {"name": p["name"], "seq": rec["seq"], "prompt_md5": p["md5"], "prompt_bytes": p["bytes"],
                 "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                 "decode_tok_s": t.get("decode_tok_s"), "prefill_tok_s": t.get("prefill_tok_s"),
                 "tau": t.get("tau"), "cycles": t.get("cycles"), "mtp": t.get("mtp"), "ttft_ms": t.get("ttft_ms"),
                 "finish": rec.get("finish"), "wall_s": rec.get("wall_s"), "content_md5": rec.get("content_md5"),
                 "error": rec.get("error")}
            cell["results"].append(r)
            if rec.get("error"):
                cell["errors"].append({"name": p["name"], "error": rec["error"]})
            save()
            print(json.dumps(r), flush=True)
            fmt = lambda v: "n/a" if v is None else (f"{v:.2f}" if isinstance(v, (int, float)) else str(v))
            print(f"  {p['name']}: decode={fmt(r['decode_tok_s'])} tok/s prefill={fmt(r['prefill_tok_s'])} tau={fmt(r['tau'])} "
                  f"ptok={r['prompt_tokens']} ctok={r['completion_tokens']} finish={r['finish']}", flush=True)

        try:
            code, h = http_json(base + "/health", timeout=10)
            wr(os.path.join(out, "health-after.json"), h)
            cell["health_after"] = h
        except Exception as e:
            cell["health_after"] = {"error": repr(e)}
        cell["status"] = "ok" if not cell["errors"] else "request_errors"
        rc = 0 if not cell["errors"] else 3
    except CellTimeout as e:
        cell["status"] = "timeout"
        cell["errors"].append(str(e))
        rc = 4
    except CellSignal as e:
        cell["status"] = "signal"
        cell["errors"].append(str(e))
        rc = 4
    except Exception as e:
        if cell["status"] in ("started",):
            cell["status"] = "failed"
        cell["errors"].append(repr(e)[:800])
        rc = rc or 2
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        cell["cleanup"] = cleanup(proc)
        logf.close()
        cell["cell_wall_s"] = time.monotonic() - t_cell
        cell["end_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save()
    print(f"cell status={cell['status']} rc={rc} out={out}", flush=True)
    return rc


def http_probe(base):
    try:
        code, _ = http_json(base + "/health", timeout=3)
        return True
    except Exception:
        return False


def cleanup(proc):
    info = {"had_proc": proc is not None}
    if proc is None:
        return info
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        info["sigterm"] = True
    except ProcessLookupError:
        info["sigterm"] = False
    try:
        proc.wait(180)
        info["exit_rc"] = proc.returncode
    except subprocess.TimeoutExpired:
        info["sigterm_wait_expired"] = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            info["sigkill"] = True
        except ProcessLookupError:
            pass
        try:
            proc.wait(30)
            info["exit_rc"] = proc.returncode
        except subprocess.TimeoutExpired:
            info["still_alive"] = True
    # group stragglers (daemon child that outlived serve)
    try:
        os.killpg(proc.pid, 0)
        info["group_alive_after_wait"] = True
        time.sleep(5)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            info["group_sigkill"] = True
        except ProcessLookupError:
            pass
    except ProcessLookupError:
        info["group_alive_after_wait"] = False
    return info


if __name__ == "__main__":
    sys.exit(main())
