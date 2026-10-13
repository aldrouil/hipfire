#!/usr/bin/env python3
"""Shared helpers for the MTP-append DET model-gate hold drivers (hipx / Halo gfx1151).

Source contract (paths are derived from this file's own location):
  SCRIPTS_DIR = directory of this file            = <git worktree>/competitors/halo-mtp-regress-20261012/mtp-append-det
  FIXTURES    = parent of SCRIPTS_DIR              = <git worktree>/competitors/halo-mtp-regress-20261012  (prompts, manifests, fixtures.json)
  ROOT        = external, untracked measurement area (binaries, test binaries, kernel cache, run outputs, sampled arm dirs)
Scripts and fixtures are ALWAYS used from a clean Git worktree (checked in every hold: `git status --porcelain
--untracked-files=all` empty, HEAD recorded); nothing is rsynced/scp'd. Prompt bytes are verified against fixtures.json.

Run ONLY inside a parent-admitted fabric timing hold (see launch.py). Nothing here queues, builds, compiles or hashes the
125 GB model. Every driver writes fsync'ed JSON receipts with UTC timestamp, argv, selected env, git HEAD/cleanliness,
binary/test-binary/script MD5s, model stat, and an arch+BDF probe (rocminfo agent enumeration, no kernel launched).
"""
import hashlib, json, os, signal, subprocess, sys, time, datetime

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.dirname(SCRIPTS_DIR)
ROOT = "/home/kaden/pm-wave/mtp-append-det/model-gates"   # external, untracked: bin/ testbin/ kcache/ runs/ sampled/
MODEL = "/home/kaden/pm-wave/fngptq/models/qwen3.8-flash-next-gptq3.mq4"
MODEL_SIZE = 125288544792
MODEL_INODE = 8292918            # prior provenance (MTP-FLOOR-LAND.md); stat only, never hashed in a hold
MODEL_MD5 = "be007fc3219e9f6cdb1d4dfa8380625f"
EXPECT_ARCH = "gfx1151"
EXPECT_BDF = "0000:c0:00.0"      # rocminfo BDFID 49152 == 0xc000 == bus c0 dev 0 fn 0, domain 0
EXPECT_BDFID = 49152
DET_ENV = "HIPFIRE_QWEN4_MTP_APPEND_DET"
FLOOR_ENV = "HIPFIRE_MTP_FLOOR_MIN_WINDOWS"
FLOOR_DEFAULT = "16"
CONFIG_TOML = (
    "schema_version = 1\n[diagnostic]\ngemm_dump = false\n[experimental.graph]\nforward = true\n"
    "[kernel]\ngfx11_iu4_gridspec = true\ngfx11_iu4_shape = false\n"
    "[speculation]\ndflash = \"off\"\nmode = \"auto\"\nmtp = \"auto\"\n"
)  # identical to MTP-FLOOR-LAND run-hold.sh (same config both arms)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wr(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


class Hold:
    """Per-hold state: deadline accounting, own-child tracking, SIGTERM forwarding, receipts."""

    def __init__(self, gate, out, max_secs, argv=None):
        self.gate = gate
        self.out = os.path.abspath(out)
        os.makedirs(self.out, exist_ok=True)
        self.t0 = time.monotonic()
        self.max_secs = max_secs
        self.children = {}     # pid -> Popen (own children only)
        self.terminated = False
        self.receipt = {"gate": gate, "start_utc": utc(), "start_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "argv": list(argv if argv is not None else sys.argv), "cwd": os.getcwd(), "pid": os.getpid(),
                        "host": os.uname().nodename, "max_secs": max_secs, "cells": [], "status": "started"}
        for k in ("HIPFIRE_FABRIC_ADMISSION", "HIPFIRE_LOCK_DIR", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "HIPFIRE_DEVICES"):
            if k in os.environ:
                self.receipt.setdefault("fabric_env", {})[k] = os.environ[k]
        signal.signal(signal.SIGTERM, self._term)
        signal.signal(signal.SIGINT, self._term)
        self.save()

    def _term(self, signum, _frm):
        self.terminated = True
        self.receipt["status"] = f"signal_{signum}"
        for p in list(self.children.values()):       # SIGTERM to OWN children only
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                try:
                    p.send_signal(signal.SIGTERM)
                except Exception:
                    pass

    def remaining(self):
        return self.max_secs - (time.monotonic() - self.t0)

    def save(self):
        wr(os.path.join(self.out, "hold-receipt.json"), self.receipt)

    def spawn(self, cmd, env, stdout, stderr, stdin=subprocess.DEVNULL, **kw):
        p = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr, stdin=stdin, start_new_session=True, **kw)
        self.children[p.pid] = p
        return p

    def wait(self, p, timeout):
        """Wait for own child; on timeout SIGTERM its group, give 120 s, then SIGKILL its group (own pid only)."""
        info = {"pid": p.pid, "timeout_s": timeout}
        try:
            rc = p.wait(timeout)
        except subprocess.TimeoutExpired:
            info["timeout_hit"] = True
            for sig, grace in ((signal.SIGTERM, 120), (signal.SIGKILL, 30)):
                try:
                    os.killpg(p.pid, sig)
                    info["sent_" + sig.name] = True
                except ProcessLookupError:
                    pass
                try:
                    p.wait(grace)
                    break
                except subprocess.TimeoutExpired:
                    continue
            rc = p.returncode
        if self.terminated and p.returncode is None:
            try:
                p.wait(120)
            except subprocess.TimeoutExpired:
                pass
            rc = p.returncode
        info["rc"] = rc
        self.children.pop(p.pid, None)
        return rc, info

    def finish(self, status):
        self.receipt["status"] = status if not self.terminated else self.receipt["status"]
        self.receipt["end_utc"] = utc()
        self.receipt["wall_s"] = time.monotonic() - self.t0
        self.save()


def clean_env(home, kcache, extra=None, rocr="1", allow_jit=False):
    """Environment shared by every process of every arm: all HIPFIRE_*/HSA/ROCR/HIP selectors reset, then the device
    selectors the fabric delivered (ROCR_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES, HIPFIRE_DEVICES) are RETAINED verbatim.
    Only when none was delivered (outside a fabric hold: preflight/probe) is the documented fallback pinned:
    HIPFIRE_DEVICES=gfx1151, ROCR_VISIBLE_DEVICES=<rocr> (default 1). Every hold still asserts gfx1151 + BDF
    0000:c0:00.0 (rocminfo probe, per-cell device assertions). Arms differ only in `extra`."""
    env = dict(os.environ)
    keep = {k: env[k] for k in ("HIPFIRE_LOCK_DIR", "HIPFIRE_FABRIC_ADMISSION", "HIPFIRE_DEVICES", "ROCR_VISIBLE_DEVICES",
                                "HIP_VISIBLE_DEVICES") if k in env}
    for k in list(env):
        if k.startswith(("HIPFIRE_", "HSA_", "ROCR_", "HIP_VISIBLE", "CUDA_VISIBLE")):
            env.pop(k)
    env.update(HOME=home, HIPFIRE_HOME=os.path.join(home, ".hipfire"), HIPFIRE_KERNEL_CACHE=kcache,
               HIPFIRE_NO_REGISTRY_FETCH="1")
    if not any(k in keep for k in ("HIPFIRE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")):
        env.update(HIPFIRE_DEVICES=EXPECT_ARCH, ROCR_VISIBLE_DEVICES=str(rocr))
    env.update(keep)
    if not allow_jit:
        env["HIPFIRE_NO_DEVICE_COMPILER"] = "1"   # indexed pack miss fails closed; no HIP compile inside a hold
    if extra:
        env.update(extra)
    return env


def write_config(home):
    d = os.path.join(home, ".hipfire")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "config.toml"), "w").write(CONFIG_TOML)


def selected_env(env):
    return {k: v for k, v in sorted(env.items()) if k.startswith(("HIPFIRE", "HSA", "ROC", "HIP", "FCLK", "AMD"))}


def probe_device(env):
    """rocminfo enumeration under the arm env: exactly one GPU agent, gfx1151, BDFID 49152 (0000:c0:00.0)."""
    r = subprocess.run(["rocminfo"], env=env, capture_output=True, text=True, timeout=60)
    agents, cur = [], None
    for line in r.stdout.splitlines():
        s = line.strip()
        if s.startswith("Agent "):
            cur = {}
            agents.append(cur)
        elif cur is not None and ":" in s:
            k, v = s.split(":", 1)
            cur.setdefault(k.strip(), v.strip())
    gpus = [a for a in agents if a.get("Device Type") == "GPU"]
    ok = len(gpus) == 1 and gpus[0].get("Name") == EXPECT_ARCH and gpus[0].get("BDFID") == str(EXPECT_BDFID)
    return {"ok": ok, "rc": r.returncode, "expect_arch": EXPECT_ARCH, "expect_bdf": EXPECT_BDF, "expect_bdfid": EXPECT_BDFID,
            "gpu_agents": [{k: a.get(k) for k in ("Name", "Marketing Name", "BDFID", "Uuid")} for a in gpus],
            "selectors": {k: env.get(k) for k in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "HIPFIRE_DEVICES")}}


def model_identity():
    st = os.stat(MODEL)
    ok = st.st_size == MODEL_SIZE and st.st_ino == MODEL_INODE
    return {"path": MODEL, "size": st.st_size, "inode": st.st_ino, "mtime": st.st_mtime, "realpath": os.path.realpath(MODEL),
            "expect_size": MODEL_SIZE, "expect_inode": MODEL_INODE, "ok": ok, "md5_expected_not_rehashed": MODEL_MD5}


def file_md5s(paths):
    return {p: {"md5": md5_file(p), "size": os.stat(p).st_size} for p in paths}


def device_lines(text):
    """Lines of daemon/test logs that identify device: contain gfx1151 and the BDF."""
    return [l for l in text.splitlines() if EXPECT_ARCH in l and EXPECT_BDF in l.lower()]


def bin_set(bindir):
    return [os.path.join(bindir, n) for n in ("daemon", "hipfire", "hipfire-weight-host") if os.path.exists(os.path.join(bindir, n))]


def git_state(path=None):
    """Git identity of the worktree holding these scripts/fixtures. `clean` is True only when HEAD resolves and
    `git status --porcelain --untracked-files=all` is empty (no modified, staged or untracked file anywhere)."""
    path = path or SCRIPTS_DIR

    def git(*a):
        return subprocess.run(["git", "-C", path] + list(a), capture_output=True, text=True, timeout=120)
    top, head, st = git("rev-parse", "--show-toplevel"), git("rev-parse", "HEAD"), git("status", "--porcelain", "--untracked-files=all")
    dirty = st.stdout.splitlines()
    return {"toplevel": top.stdout.strip(), "head": head.stdout.strip(), "rc": [top.returncode, head.returncode, st.returncode],
            "dirty_count": len(dirty), "dirty_head": dirty[:30], "clean": top.returncode == 0 and head.returncode == 0
            and st.returncode == 0 and not dirty, "scripts_dir": SCRIPTS_DIR, "fixtures_dir": FIXTURES}


def fixtures_index(fixtures=None):
    fixtures = fixtures or FIXTURES
    d = json.load(open(os.path.join(fixtures, "fixtures.json")))
    return d, {f["path"]: f["md5"] for f in d["files"]}


def fixtures_check(fixtures=None):
    """Every file listed in fixtures.json exists in the worktree with the recorded MD5. Returns the list of problems."""
    fixtures = fixtures or FIXTURES
    _, want = fixtures_index(fixtures)
    bad = []
    for rel, md5 in sorted(want.items()):
        p = os.path.join(fixtures, rel)
        if not os.path.exists(p) or md5_file(p) != md5:
            bad.append(rel)
    return bad


def resolve_manifest(src, out, fixtures=None):
    """Write a copy of manifest `src` to `out` whose prompt_path values point at the verified worktree fixture files
    (matched by basename: short.txt, p8k.txt, p16k.txt, high-*.txt). Relative prompt_path values resolve against the
    fixtures dir. The committed manifests are never modified, so their recorded MD5s stay authoritative."""
    fixtures = fixtures or FIXTURES
    _, want = fixtures_index(fixtures)
    byname = {os.path.basename(k): k for k in want}
    items = json.load(open(src))
    for it in items:
        rel = byname.get(os.path.basename(it["prompt_path"]))
        if rel is None:
            raise SystemExit(f"manifest {src}: prompt {it['prompt_path']} not in fixtures.json")
        it["prompt_path"] = os.path.join(fixtures, rel)
    wr(out, items)
    return items


def preflight_common(h, bindir=None, extra_files=(), scripts=True):
    """Fail-fast, no-GPU-work identity block shared by all drivers. Returns False (and records) on mismatch:
    model stat, rocminfo arch/BDF, clean Git worktree, fixtures.json MD5 verification."""
    r = h.receipt
    with open("/proc/cmdline") as f:
        r["proc_cmdline"] = f.read().strip()
    r["model_identity"] = model_identity()
    r["git"] = git_state()
    r["fixtures_bad"] = fixtures_check()
    files = list(extra_files)
    if bindir:
        files += bin_set(bindir)
    if scripts:
        files += [os.path.join(SCRIPTS_DIR, n) for n in sorted(os.listdir(SCRIPTS_DIR)) if n.endswith(".py")]
    r["file_md5"] = file_md5s(files)
    r["det_env_in_parent_process"] = os.environ.get(DET_ENV)
    r["device_probe"] = probe_device(clean_env(h.out, os.path.join(ROOT, "kcache")))
    h.save()
    return r["model_identity"]["ok"] and r["device_probe"]["ok"] and r["git"]["clean"] and not r["fixtures_bad"]


def hold_budget_ok(h, need_s):
    return h.remaining() >= need_s and not h.terminated
