#!/usr/bin/env python3
"""Gate D: whole-state invariance of the DET prompt fill, via the existing qwen4_mtp_fill digest example and the existing
ignored mtp_takeover_fill_hw test. Bounded holds, no GPU work outside them. Prompt = halofn/p8k.txt from the clean worktree.

--phase fill first runs ROOT/testbin/mtp_q8_det_byte_oracle (--seed 184; est <=30 s, own-child guard 45 s) with an independent
receipt (q8-oracle/{stdout.txt,stderr.txt,receipt.json}); nonzero rc, timeout, missing PASS line or non-gfx1151/0000:c0:00.0
start event sets state_pass=false (model cells still run). Same selectors, indexed kcache, HIPFIRE_NO_DEVICE_COMPILER=1.

--phase fill      4 fresh processes of `qwen4_mtp_fill MODEL OUT.json 4096 1,15,16,17,255,256,257 0 p8k.txt` (cold, real text,
                  lone prompt tail n=1 and the 16-row boundaries 15/16/17, 255/256/257):
                    DET1 x HIPFIRE_QWEN4_MTP_BATCHED_FILL={0,1}   (gate: JSON byte-equal except the explicit batched_fill label)
                    DET0 x HIPFIRE_QWEN4_MTP_BATCHED_FILL={0,1}   (incumbent control, informational)
                  Optional --spread adds the deterministic vocabulary-spread token fill (no PROMPT_FILE) for DET1.
--phase takeover  the existing ignored `mtp_takeover_fill_hw` (3 fresh child arms: batched-fill, serial-fill, interleaved) with
                  DET=1 (--det 1) or the DET=0 control (--det 0; sets HIPFIRE_QWEN4_MTP_APPEND_DET=0 explicitly: unset = default ON on gfx1151): DecodeFill (n-gram takeover fill) must stay incumbent, so the
                  test's own assertions (rejected_tail.same_state, pairing, batched==serial windows, one-row AR reference
                  digests) must still pass with DET on. NOTE (route read once per process) DET is inherited by the arm children.
  fabric-run.sh --class timing --card halo --hostmem-lock --mem-gb 110 --max-secs 600 --wait-secs 14400 \
     --priority P? --owner ID -- python3 -I <worktree>/competitors/halo-mtp-regress-20261012/mtp-append-det/fill_state.py \
     --phase fill --out ROOT/runs/<stamp>-fill --bin-fill ROOT/testbin/qwen4_mtp_fill
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gatelib as G

LENGTHS = "1,15,16,17,255,256,257"
ap = argparse.ArgumentParser()
ap.add_argument("--phase", choices=("fill", "takeover"), required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--bin-fill", default=os.path.join(G.ROOT, "testbin", "qwen4_mtp_fill"))
ap.add_argument("--testbin-takeover", default=os.path.join(G.ROOT, "testbin", "mtp_takeover_fill_hw"))
ap.add_argument("--q8-oracle", default=os.path.join(G.ROOT, "testbin", "mtp_q8_det_byte_oracle"), help="fill phase: Q8 DET byte oracle run before any model process")
ap.add_argument("--q8-oracle-timeout", type=int, default=45, help="own-child guard seconds (estimate <=30 s)")
ap.add_argument("--fixtures", default=G.FIXTURES)
ap.add_argument("--kcache", default=os.path.join(G.ROOT, "kcache"))
ap.add_argument("--det", type=int, choices=(0, 1), default=1, help="takeover phase DET value")
ap.add_argument("--spread", action="store_true")
ap.add_argument("--max-secs", type=int, default=600)
ap.add_argument("--allow-jit", action="store_true")
a = ap.parse_args()
h = G.Hold("fill-" + a.phase, a.out, a.max_secs)
R = h.receipt
prompt = os.path.join(a.fixtures, "halofn", "p8k.txt")
exe = a.bin_fill if a.phase == "fill" else a.testbin_takeover
need_files = [exe] + ([prompt, a.q8_oracle] if a.phase == "fill" else [])
missing = [p for p in need_files if not os.path.exists(p)]
if missing:
    R["missing_artifacts"] = missing
ok = not missing and G.preflight_common(h, None, need_files)
if a.phase == "fill" and os.path.exists(prompt) and G.md5_file(prompt) != "c3ccdc43015ed551bcd339eebb02c893":
    R["fixture_mismatch"] = [prompt]
    ok = False
if not ok:
    h.finish("preflight_failed")
    print("PREFLIGHT FAILED", json.dumps({k: R.get(k) for k in ("missing_artifacts", "model_identity", "device_probe", "git", "fixtures_bad", "fixture_mismatch")}, indent=1)[:2500])
    sys.exit(2)
runs = []
if a.phase == "fill":
    runs = [("det1-bf0", "1", "0", True), ("det1-bf1", "1", "1", True), ("det0-bf0", "0", "0", True), ("det0-bf1", "0", "1", True)]
    if a.spread:
        runs += [("det1-bf0-spread", "1", "0", False), ("det1-bf1-spread", "1", "1", False)]
else:
    runs = [(f"takeover-det{a.det}", "1" if a.det else "0", None, False)]
rc_all = 0
q8 = None
if a.phase == "fill":
    # Independent Q8 DET byte oracle (gfx1151 / 0000:c0:00.0 checked by the oracle itself; no timing, no model). Own receipt;
    # any nonzero rc, timeout, missing PASS line or wrong arch/BDF in its start event prevents the state PASS. Same fabric
    # selectors (inherited), indexed kcache, HIPFIRE_NO_DEVICE_COMPILER=1 always (the oracle requires it), own PID group only.
    qd = os.path.join(h.out, "q8-oracle")
    os.makedirs(qd, exist_ok=True)
    qhome = os.path.join(qd, "home")
    G.write_config(qhome)
    qenv = G.clean_env(qhome, a.kcache)
    qcmd = [a.q8_oracle, "--seed", "184"]
    q8 = {"name": "q8-oracle", "argv": qcmd, "env": G.selected_env(qenv), "start_utc": G.utc(), "out": qd, "guard_s": a.q8_oracle_timeout,
          "binary_md5": G.md5_file(a.q8_oracle)}
    R["q8_oracle"] = q8
    h.save()
    with open(os.path.join(qd, "stdout.txt"), "wb") as so, open(os.path.join(qd, "stderr.txt"), "wb") as se:
        p = h.spawn(qcmd, qenv, so, se)
        q8["pid"] = p.pid
        h.save()
        qrc, qinfo = h.wait(p, a.q8_oracle_timeout)
    q8.update(rc=qrc, wait=qinfo, end_utc=G.utc())
    qout = open(os.path.join(qd, "stdout.txt"), errors="replace").read()
    events = []
    for line in qout.splitlines():
        if line.startswith("{"):
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
    start = next((e for e in events if e.get("event") == "start"), {})
    q8["start_event"] = start
    q8["verdict_event"] = next((e for e in events if e.get("event") == "verdict"), None)
    q8["pass_line"] = "Q8 DET BYTE ORACLE PASS" in qout.splitlines()
    q8["device_ok"] = start.get("arch") == G.EXPECT_ARCH and str(start.get("pci_bus_id", "")).lower() == G.EXPECT_BDF
    q8["ok"] = qrc == 0 and q8["pass_line"] and q8["device_ok"] and not qinfo.get("timeout_hit")
    G.wr(os.path.join(qd, "receipt.json"), q8)
    h.save()
    print("Q8_ORACLE", "PASS" if q8["ok"] else f"FAIL rc={qrc}", qd)
    if not q8["ok"]:
        rc_all = 8
for name, det, bf, real in runs:
    if not G.hold_budget_ok(h, 150):
        R["cells"].append({"name": name, "skipped": "insufficient hold budget or terminated", "remaining_s": h.remaining()})
        rc_all = rc_all or 5
        continue
    cell = os.path.join(h.out, name)
    home = os.path.join(cell, "home")
    G.write_config(home)
    extra = {}
    if det:
        extra[G.DET_ENV] = det
    if a.phase == "fill":
        extra["HIPFIRE_QWEN4_MTP_BATCHED_FILL"] = bf
        cmd = [exe, G.MODEL, os.path.join(cell, "fill.json"), "4096", LENGTHS, "0"] + ([prompt] if real else [])
    else:
        extra["HIPFIRE_MTP_IDENTITY_MODEL"] = G.MODEL
        cmd = [exe, "--ignored", "--exact", "mtp_takeover_fill_hw", "--nocapture", "--test-threads=1"]
    env = G.clean_env(home, a.kcache, extra, allow_jit=a.allow_jit)
    rec = {"name": name, "argv": cmd, "env": G.selected_env(env), "det": det, "batched_fill": bf, "start_utc": G.utc(), "out": cell}
    R["cells"].append(rec)
    h.save()
    with open(os.path.join(cell, "stdout.txt"), "wb") as so, open(os.path.join(cell, "stderr.txt"), "wb") as se:
        p = h.spawn(cmd, env, so, se)
        rec["pid"] = p.pid
        h.save()
        rc, info = h.wait(p, int(min(480, h.remaining() - 30)))
    rec.update(rc=rc, wait=info, end_utc=G.utc())
    out = open(os.path.join(cell, "stdout.txt"), errors="replace").read()
    err = open(os.path.join(cell, "stderr.txt"), errors="replace").read()
    rec["arch_lines"] = [l[:200] for l in (out + err).splitlines() if "gfx1151" in l][:4]
    if a.phase == "fill":
        rec["fill_json_md5"] = G.md5_file(os.path.join(cell, "fill.json")) if os.path.exists(os.path.join(cell, "fill.json")) else None
    else:
        rec["summary_lines"] = [l[:300] for l in out.splitlines() if "MTP_TAKEOVER_FILL_SUMMARY" in l]
        rec["test_result"] = next((l for l in out.splitlines() if l.startswith("test result")), None)
    if rc != 0:
        rc_all = rc_all or rc
    h.save()
    if h.terminated:
        break
if a.phase == "fill":
    import compare_fill
    R["comparison"] = compare_fill.compare_dir(h.out)
R["state_pass"] = bool(rc_all == 0 and (a.phase != "fill" or (q8 and q8["ok"] and R["comparison"].get("verdict") == "PASS")))
h.finish("complete" if rc_all == 0 else f"incomplete_rc{rc_all}")
print("FILL_STATE_DONE", a.phase, R["status"], "q8_oracle=" + (str(q8["ok"]) if q8 else "n/a"), json.dumps(R.get("comparison", {}).get("verdict")), "state_pass=" + str(R["state_pass"]))
sys.exit(0 if R["state_pass"] else 1)
