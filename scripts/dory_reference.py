#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 hipfire contributors — see LICENSE and NOTICE in the project root.
"""Independent CPU (NumPy, float32) reference oracle for Dory-4B, ONE recurrent loop.

Math source (adapted equations, Apache-2.0):
  https://raw.githubusercontent.com/vllm-project/vllm/55076950b6c155f384d6766d63563c3bc20a0e3f/vllm/model_executor/models/dory.py
  (DoryForCausalLM / DoryBackbone / DoryAttention / DoryMLP / justnorm / slerp_norm.)

Purpose: ground-truth logits + last-hidden states for the native hipfire
`hipfire-arch-dory` bring-up, which runs exactly ONE recurrent loop regardless
of the config default (n_recurrent_loops=2). Native baseline: BF16 weights kept
at exact BF16 values, F32 accumulation, NO global BF16 activation rounding in
the primary oracle. A `--bf16twin` mode rounds every matmul input/output to
BF16 (round-to-nearest-even, F32 accumulate) for precision diagnostics only.
Native parity is checked against the F32 primary oracle, not that twin's
larger rounding envelope.

Weight streaming: the 7.95 GB BF16 safetensors file is mmapped and each tensor
is converted BF16->F32 on demand (exact conversion); full-model F32 residency
(~16 GB) is never required. Peak RSS is ~2.5 GB (lm_head F32 + one layer).
The self-contained Q8 .hfq path streams the same way: each Q8F16 tensor (34 B
per 32 weights: f16 LE scale + 32 x int8) or each F32 tensor (qt=2)
decodes to F32 per tensor / per embedding row, never holding a full F32
model copy.

--model accepts a BF16 safetensors dir (default) or a self-contained Dory
.hfq file (config + tokenizer + weights; qt=3 Q8F16 group-32 matrices and
qt=2 F32 vectors; other quant_types are rejected). Forward math is the
same NumPy F32 path for both sources; only the weight decode differs.

Layout facts (the reference Dory config.json and safetensors header):
  vocab=131072 hidden=2560 layers=72 inter=7168 heads=16 kv_heads=4 head_dim=256
  n_input=16 n_recurrent=40 n_output=16 | even layer idx = attention, odd = mlp
  sliding window=1024 (sliding layers), full layers theta=1e7, sliding theta=1e4
  ngpt_alpha_init=0.05 initializer_range=0.0177
  attention_output_gate_logit_scale_init=null -> g0 = sqrt(hidden_size)

Deps: numpy, safetensors (header parse only), tokenizers (prompt only).
  No torch required. Tested with: numpy 2.5.4, safetensors 0.8.0, tokenizers 0.23.1.

Usage:
  python3 scripts/dory_reference.py --selftest
      # tiny deterministic fixture: F32 fast path vs float64 reference,
      # one attention + one MLP layer incl. rope/qk-norm/gate/slerp,
      # plus Q8F16 signed-scale and F32 bit-exactness decoder checks.
  python3 scripts/dory_reference.py --run --model ./dory-bf16 --out DIR
      # full model: prefill S[:prefill], KV-cached decode of S[prefill:],
      # prefill-vs-decode self-consistency check, emits artifact DIR/.
  python3 scripts/dory_reference.py --run --model dory-4b.q8.hfq --out DIR
      # same 16-token teacher-forced run from the self-contained Q8 file
      # (config/tokenizer from embedded HFQ metadata, no sidecar dir).

Native JSON schema for --compare:
  {"tokens": [...], "logits": [[...131072 floats...], ...],
   "hidden": [[...2560...], ...] (optional, for cosine),
   "positions": "prefill+decode", "loops": 1}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
from typing import Dict, List, Tuple

import numpy as np

EPS = 1e-6
F32 = np.float32
SRC_ATTRIBUTION = (
    "vllm vllm/model_executor/models/dory.py @55076950 "
    "(Apache-2.0, Copyright contributors to the vLLM project)"
)

# ----------------------------------------------------------------------------
# BF16 helpers (exact bit conversions; numpy has no BF16 without ml_dtypes)
# ----------------------------------------------------------------------------

def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    """Exact BF16 -> F32 widening. Input uint16 array."""
    u = np.ascontiguousarray(bits, dtype=np.uint16)
    return ((u.astype(np.uint32) << np.uint32(16)).view(np.float32)).astype(np.float32)


def f16_to_f32(bits: np.ndarray) -> np.ndarray:
    """Exact F16 -> F32 widening. Input uint16 array."""
    u = np.ascontiguousarray(bits, dtype=np.uint16).astype(np.int32)
    sign = (u >> 15) & 1
    exp = (u >> 10) & 0x1F
    mant = u & 0x3FF
    fsign = np.where(sign == 1, np.float32(-1.0), np.float32(1.0))
    out = np.empty(u.shape, dtype=np.float32)
    norm = (exp != 0) & (exp != 31)
    out[norm] = (fsign[norm] * (2.0 ** (exp[norm] - 15))
                 * (1.0 + mant[norm] / np.float32(1024.0)))
    sub = exp == 0
    out[sub] = fsign[sub] * np.float32(2.0 ** -14) * (mant[sub] / np.float32(1024.0))
    infnan = exp == 31
    inf = fsign * np.float32(np.inf)
    out[infnan] = np.where(mant[infnan] == 0, inf[infnan], np.float32(np.nan))
    return out.astype(np.float32, copy=False)


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """F32 -> BF16 bits, round-to-nearest-even."""
    a = np.ascontiguousarray(x, dtype=np.float32)
    u = a.view(np.uint32)
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    bias = np.uint32(0x7FFF) + lsb
    return ((u + bias) >> np.uint32(16)).astype(np.uint16)


def bf16_round(x: np.ndarray) -> np.ndarray:
    """Round an F32 array to BF16-representable values (twin mode)."""
    return bf16_to_f32(f32_to_bf16_bits(x))


# ----------------------------------------------------------------------------
# Streaming BF16 safetensors reader
# ----------------------------------------------------------------------------

class STReader:
    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "rb")
        (n,) = struct.unpack("<Q", self.f.read(8))
        self.header_len = n
        self.raw_header = self.f.read(n)
        self.header = json.loads(self.raw_header)
        self.base = 8 + n
        self.file_size = os.path.getsize(path)
        self._f32cache: Dict[str, np.ndarray] = {}

    def meta(self, name: str):
        return self.header[name]

    def bf16_f32(self, name: str) -> np.ndarray:
        """Load tensor, exact BF16->F32. Small tensors are cached."""
        e = self.header[name]
        ne = int(np.prod(e["shape"]))
        off = self.base + e["data_offsets"][0]
        raw = np.memmap(self.path, dtype=np.uint16, mode="r", offset=off, shape=(ne,))
        out = bf16_to_f32(np.asarray(raw)).reshape(tuple(e["shape"]))
        del raw
        # cache 1-D param vectors (scales/alphas) — tiny, reused every step
        if out.ndim <= 1 and name not in self._f32cache:
            self._f32cache[name] = out
            return out
        if name in self._f32cache:
            return self._f32cache[name]
        return out

    def row_gather_f32(self, name: str, ids: np.ndarray) -> np.ndarray:
        """Gather embedding rows without materializing the full matrix."""
        e = self.header[name]
        assert len(e["shape"]) == 2
        nrow, ncol = e["shape"]
        row_bytes = ncol * 2
        ids = np.asarray(ids, dtype=np.int64)
        out = np.empty((len(ids), ncol), dtype=np.float32)
        for j, r in enumerate(ids):
            off = self.base + e["data_offsets"][0] + int(r) * row_bytes
            raw = np.memmap(self.path, dtype=np.uint16, mode="r", offset=off,
                            shape=(ncol,))
            out[j] = bf16_to_f32(np.asarray(raw))
            del raw
        return out

    def header_fingerprint(self) -> str:
        # Identity of the weight FILE (first 64 KiB incl. 8-B length prefix
        # + JSON header start) — NOT a hash of the parsed header dict.
        # Labeled file_prefix_sha256_16 in the manifest; the full header
        # sha256 (of the n header bytes) is recorded separately by the runner.
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            h.update(f.read(1 << 16))
        return h.hexdigest()[:32]

# ----------------------------------------------------------------------------
# Streaming HFQM (.hfq) reader — Q8 (qt=3) + F32 (qt=2)
# ----------------------------------------------------------------------------
#
# Container layout (mirrors hipfire-runtime/src/hfq.rs `open_at_offset`):
#   header (32 B): "HFQM" | u32 version | u32 arch_id | u32 n_tensors
#                  | u64 metadata_offset | u64 data_offset
#   metadata JSON at metadata_offset (brace-terminated), then tensor index:
#   u32 count, per tensor: u16 name_len | name | u8 quant_type | u8 n_dims
#                  | n_dims x u32 shape | u32 group_size | u64 data_size.
#   Tensor payloads pack contiguously from data_offset in index order.
#   When the version word carries the overlay bit (0x8000_0000), the legacy
#   metadata is superseded by the overlay at EOF: metadata bytes +
#   u64 len | u32 schema | 32 B sha256 | "HFQMDOV1" (52-B footer).
# Decoders (mirrors hipfire-quantize/src/quant_q4.rs `quantize_q8f16` and
# crates/hipfire-runtime/src/weight_backend.rs `dequantize_to_f32`):
#   qt=3 Q8F16: 34 B per 32 weights = f16 LE scale + 32 x int8,
#               w = f32(scale) * q. qt=2 F32: 4 B LE per weight.
#   Any other quant_type is rejected explicitly.

HFQ_OVERLAY_FLAG = 0x80000000
HFQ_OVERLAY_MAGIC = b"HFQMDOV1"
HFQ_OVERLAY_FOOTER_LEN = 8 + 4 + 32 + 8


class HfqReader:
    """Streamed Q8/F32 tensors from a self-contained Dory .hfq file."""

    def __init__(self, path: str):
        self.path = path
        self.file_size = os.path.getsize(path)
        with open(path, "rb") as f:
            hdr = f.read(32)
        if len(hdr) < 32 or hdr[:4] != b"HFQM":
            raise SystemExit(f"not an HFQM container: {path}")
        raw_ver = struct.unpack("<I", hdr[4:8])[0]
        if raw_ver & ~HFQ_OVERLAY_FLAG != 1:
            raise SystemExit(f"{path}: unsupported HFQM version {raw_ver & ~HFQ_OVERLAY_FLAG}")
        self.has_overlay = bool(raw_ver & HFQ_OVERLAY_FLAG)
        self.arch_id = struct.unpack("<I", hdr[8:12])[0]
        n_tensors = struct.unpack("<I", hdr[12:16])[0]
        metadata_off = struct.unpack("<Q", hdr[16:24])[0]
        data_off = struct.unpack("<Q", hdr[24:32])[0]
        if not (metadata_off == 32 and metadata_off <= data_off <= self.file_size):
            raise SystemExit(f"{path}: invalid HFQ metadata/data offsets "
                             f"{metadata_off}/{data_off} for {self.file_size}-byte file")
        mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(self.file_size,))
        try:
            meta_bytes = bytes(mm[metadata_off:data_off])
            # brace-match the JSON end (metadata is followed by the index)
            depth = 0
            in_str = False
            esc = False
            json_end = 0
            for i, b in enumerate(meta_bytes):
                b = int(b)
                if esc:
                    esc = False
                    continue
                if b == 0x5C and in_str:  # backslash
                    esc = True
                    continue
                if b == 0x22:  # quote
                    in_str = not in_str
                    continue
                if not in_str:
                    if b == 0x7B:
                        depth += 1
                    elif b == 0x7D:
                        depth -= 1
                        if depth == 0:
                            json_end = i + 1
                            break
            if json_end == 0:
                raise SystemExit(f"{path}: HFQ metadata JSON is not brace-terminated")
            legacy_meta = meta_bytes[:json_end].decode("utf-8")
            pos = metadata_off + json_end
            (idx_n,) = struct.unpack("<I", bytes(mm[pos:pos + 4]))
            if idx_n != n_tensors:
                raise SystemExit(f"{path}: index count {idx_n} != header {n_tensors}")
            pos += 4
            self.tensors: Dict[str, dict] = {}
            cum = data_off
            for _ in range(n_tensors):
                (nl,) = struct.unpack("<H", bytes(mm[pos:pos + 2]))
                pos += 2
                name = bytes(mm[pos:pos + nl]).decode("utf-8")
                pos += nl
                qt = int(mm[pos])
                nd = int(mm[pos + 1])
                pos += 2
                shape = tuple(struct.unpack("<%dI" % nd, bytes(mm[pos:pos + 4 * nd]))) if nd else ()
                pos += 4 * nd
                (gs,) = struct.unpack("<I", bytes(mm[pos:pos + 4]))
                (ds,) = struct.unpack("<Q", bytes(mm[pos + 4:pos + 12]))
                pos += 12
                end = cum + ds
                if end > self.file_size:
                    raise SystemExit(f"{path}: tensor '{name}' payload "
                                     f"[{cum}, {end}) past {self.file_size}-byte EOF")
                self.tensors[name] = {"quant_type": qt, "shape": shape,
                                      "group_size": gs, "offset": cum, "size": ds}
                cum = end
            payload_end = cum
            if self.has_overlay:
                fl = self.file_size
                if fl < HFQ_OVERLAY_FOOTER_LEN:
                    raise SystemExit(f"{path}: metadata overlay footer truncated")
                fs = fl - HFQ_OVERLAY_FOOTER_LEN
                foot = bytes(mm[fs:fl])
                (mlen,) = struct.unpack("<Q", foot[:8])
                (schema,) = struct.unpack("<I", foot[8:12])
                digest = foot[12:44]
                magic = foot[44:52]
                if schema != 1 or magic != HFQ_OVERLAY_MAGIC:
                    raise SystemExit(f"{path}: bad metadata overlay footer "
                                     f"(schema={schema})")
                mstart = fs - mlen
                if mstart < payload_end:
                    raise SystemExit(f"{path}: overlay starts before payload end")
                raw = bytes(mm[mstart:fs])
                if hashlib.sha256(raw).digest() != digest:
                    raise SystemExit(f"{path}: metadata overlay checksum mismatch")
                self.metadata_json = raw.decode("utf-8")
            else:
                self.metadata_json = legacy_meta
        finally:
            del mm
        self.meta = json.loads(self.metadata_json)
        self.header_len = len(self.metadata_json.encode("utf-8"))
        self.raw_header = self.metadata_json.encode("utf-8")
        self._f32cache: Dict[str, np.ndarray] = {}

    def _check_supported(self, name: str) -> dict:
        e = self.tensors.get(name)
        if e is None:
            raise SystemExit(f"tensor not found in HFQ: {name}")
        if e["quant_type"] not in (2, 3):
            raise SystemExit(f"tensor '{name}': quant_type={e['quant_type']} unsupported "
                             f"in the CPU oracle (only qt=2 F32, qt=3 Q8F16)")
        if e["quant_type"] == 3 and e["group_size"] not in (0, 32):
            raise SystemExit(f"tensor '{name}': Q8F16 group_size={e['group_size']} "
                             f"(expected 32)")
        return e


    def _read_bytes(self, e: dict, start: int, n: int) -> np.ndarray:
        mm = np.memmap(self.path, dtype=np.uint8, mode="r",
                       offset=e["offset"] + start, shape=(n,))
        out = np.asarray(mm).copy()
        del mm
        return out

    @staticmethod
    def _decode_q8f16(raw: np.ndarray, ne: int) -> np.ndarray:
        """Decode whole-buffer Q8F16 (34 B blocks) to F32, row-major flat."""
        nb = (ne + 31) // 32
        if len(raw) < nb * 34:
            raise SystemExit(f"Q8F16 payload short: {len(raw)} < {nb * 34}")
        blk = np.ascontiguousarray(raw[:nb * 34]).reshape(nb, 34)
        sbits = (blk[:, 0].astype(np.uint16)
                 | (blk[:, 1].astype(np.uint16) << np.uint16(8)))
        scales = f16_to_f32(sbits).astype(np.float32)
        qs = blk[:, 2:].astype(np.int8).astype(np.float32)
        flat = (qs * scales[:, None]).reshape(-1)[:ne]
        return flat.astype(np.float32, copy=False)


    def bf16_f32(self, name: str) -> np.ndarray:
        """Load tensor dequantized to F32 (Q8F16 or F32 source)."""
        e = self._check_supported(name)
        shape = tuple(e["shape"])
        ne = int(np.prod(shape)) if shape else 1
        if name in self._f32cache:
            return self._f32cache[name]
        if e["quant_type"] == 2:
            if e["size"] < ne * 4:
                raise SystemExit(f"tensor '{name}': F32 payload short")
            mm = np.memmap(self.path, dtype="<f4", mode="r",
                           offset=e["offset"], shape=(ne,))
            out = np.asarray(mm).astype(np.float32).reshape(shape)
            del mm
        else:
            nb = (ne + 31) // 32
            raw = self._read_bytes(e, 0, nb * 34)
            out = self._decode_q8f16(raw, ne).reshape(shape)
        if out.ndim <= 1:
            self._f32cache[name] = out
        return out

    def row_gather_f32(self, name: str, ids: np.ndarray) -> np.ndarray:
        """Gather rows without materializing the full matrix."""
        e = self._check_supported(name)
        assert len(e["shape"]) == 2
        nrow, ncol = (int(e["shape"][0]), int(e["shape"][1]))
        ids = np.asarray(ids, dtype=np.int64)
        out = np.empty((len(ids), ncol), dtype=np.float32)
        if e["quant_type"] == 2:
            for j, r in enumerate(ids):
                mm = np.memmap(self.path, dtype="<f4", mode="r",
                               offset=e["offset"] + int(r) * ncol * 4, shape=(ncol,))
                out[j] = np.asarray(mm)
                del mm
        else:
            nb = (ncol + 31) // 32
            row_bytes = nb * 34
            for j, r in enumerate(ids):
                raw = self._read_bytes(e, int(r) * row_bytes, row_bytes)
                out[j] = self._decode_q8f16(raw, ncol)
        return out

    def header_fingerprint(self) -> str:
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            h.update(f.read(1 << 16))
        return h.hexdigest()[:32]


# ----------------------------------------------------------------------------
# Core math (mirrors upstream; all F32 unless twin mode passes bf16=True)
# ----------------------------------------------------------------------------

def justnorm(x: np.ndarray) -> np.ndarray:
    xf = x.astype(np.float32, copy=False)
    y = xf * (1.0 / np.sqrt(np.sum(xf * xf, axis=-1, keepdims=True) + EPS))
    return y.astype(np.float32, copy=False)


def slerp_alpha(a: np.ndarray, b: np.ndarray, alpha: np.ndarray,
                alpha_scale: float) -> np.ndarray:
    # upstream: slerp_norm(a, b, |alpha * scale|) = justnorm(ya + lr*(yb - ya));
    # `a` is the residual stream (owned weights), `b` the proposed sublayer update.
    ya = justnorm(a).astype(np.float32)
    yb = justnorm(b).astype(np.float32)
    lr = np.abs(alpha.astype(np.float32) * np.float32(alpha_scale))
    return justnorm(ya + lr * (yb - ya))


def silu(x: np.ndarray) -> np.ndarray:
    xf = x.astype(np.float32, copy=False)
    return (xf / (1.0 + np.exp(-xf))).astype(np.float32, copy=False)


def rope_apply(x: np.ndarray, pos: np.ndarray, inv_freq: np.ndarray) -> np.ndarray:
    """NeoX-style rope. x: (T,H,D), pos: (T,), inv_freq: (D/2,)."""
    ang = pos[:, None].astype(np.float32) * inv_freq[None, :].astype(np.float32)
    cos = np.cos(ang)[:, None, :].astype(np.float32)
    sin = np.sin(ang)[:, None, :].astype(np.float32)
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


class DoryRef:
    """One-loop Dory forward. Set bf16=True for the activation-rounded twin."""

    def __init__(self, cfg: dict, reader: STReader, n_loops: int = 1,
                 bf16: bool = False):
        self.c = cfg
        self.r = reader
        self.n_loops = n_loops
        self.bf16 = bf16
        H = cfg["hidden_size"]
        self.alpha_scale = cfg["ngpt_alpha_init"] / cfg["initializer_range"]
        self.sqk_factor = (cfg["head_dim"] ** 0.5) / cfg["initializer_range"]
        g0 = cfg.get("attention_output_gate_logit_scale_init") or (H ** 0.5)
        self.sgate_factor = (g0 ** 0.5) / cfg["initializer_range"]
        self.suv_factor = H ** 0.5
        d = cfg["head_dim"]
        self.inv_sliding = (1.0 / (10000.0 ** (np.arange(0, d, 2) / d))).astype(np.float32)
        self.inv_full = (1.0 / (10000000.0 ** (np.arange(0, d, 2) / d))).astype(np.float32)
        self.is_attn = [(t != "mlp") for t in cfg["layer_types"]]
        assert all((t != "mlp") == (i % 2 == 0) for i, t in enumerate(cfg["layer_types"])), \
            "expected even=attn / odd=mlp alternation"
        self.nH = cfg["num_attention_heads"]
        self.nKV = cfg["num_key_value_heads"]
        self.rep = self.nH // self.nKV
        assert self.rep * self.nKV == self.nH

    # -- linear helper -----------------------------------------------------
    def lin(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        if self.bf16:
            x = bf16_round(x)
            w = bf16_round(w)
        y = x @ w.T
        return bf16_round(y) if self.bf16 else y

    # -- sublayers ----------------------------------------------------------
    def attn(self, h: np.ndarray, pos: np.ndarray, li: int,
             kv: dict | None) -> Tuple[np.ndarray, dict | None]:
        c = self.c
        T = h.shape[0]
        q = self.lin(h, self.r.bf16_f32(f"backbone.layers.{li}.mixer.q_proj.weight"))
        k = self.lin(h, self.r.bf16_f32(f"backbone.layers.{li}.mixer.k_proj.weight"))
        v = self.lin(h, self.r.bf16_f32(f"backbone.layers.{li}.mixer.v_proj.weight"))
        g = self.lin(h, self.r.bf16_f32(f"backbone.layers.{li}.mixer.g_proj.weight"))
        q = q.reshape(T, self.nH, c["head_dim"])
        k = k.reshape(T, self.nKV, c["head_dim"])
        v = v.reshape(T, self.nKV, c["head_dim"])
        inv = self.inv_full if self.c["layer_types"][li] == "full_attention" \
            else self.inv_sliding
        q, k = rope_apply(q, pos, inv), rope_apply(k, pos, inv)
        # nGPT qk-norm AFTER rope
        sqk = (self.r.bf16_f32(f"backbone.layers.{li}.mixer.sqk").astype(np.float32)
               * np.float32(self.sqk_factor)).reshape(1, self.nH, c["head_dim"])
        q = (sqk * sqk * justnorm(q)).astype(np.float32)
        k = justnorm(k)
        was_decode = kv is not None
        if was_decode:
            # decode: append pre-expanded KV (rope already applied at abs pos)
            k = np.concatenate([kv["k"], k], axis=0)
            v = np.concatenate([kv["v"], v], axis=0)
        # ALWAYS (re)populate the cache: prefill seeds it, decode extends it.
        # (vLLM prefill fills the KV cache for the whole prompt; decode reuses it.
        # Branch on was_decode (the INPUT cache state), not on kv below.)
        kv = {"k": k, "v": v}
        kh = np.repeat(k, self.rep, axis=1)  # (S,nH,hd)
        vh = np.repeat(v, self.rep, axis=1)
        sc = 1.0 / np.sqrt(float(c["head_dim"]))
        S = kh.shape[0]
        scores = np.einsum("thd,shd->hts", q.astype(np.float32),
                           kh.astype(np.float32)) * np.float32(sc)
        if self.c["layer_types"][li] == "sliding_attention" and was_decode and T == 1:
            # decode (single-query step): cache holds every past key at
            # absolute positions 0..S-1; mask keys outside the window back
            # from the query position.
            win = self.c["sliding_window"]
            kpos = np.arange(S)[None, None, :]
            scores += np.where(pos[0] - kpos >= win, -1e30, 0.0)
        if not was_decode:
            if self.c["layer_types"][li] == "sliding_attention":
                # prefill: keys are exactly the query positions
                win = self.c["sliding_window"]
                tq = pos[:, None]
                tk = pos[None, :]
                scores += np.where((tk > tq) | (tq - tk >= win), -1e30, 0.0)
            else:
                # positions are contiguous from pos[0]; local causal index:
                qq = np.arange(T)[:, None]
                kk = np.arange(S)[None, :]
                scores += np.where(kk > qq + (S - T), -1e30, 0.0)
        scores = scores.astype(np.float32)
        scores -= scores.max(axis=-1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(axis=-1, keepdims=True)
        o = np.einsum("hts,shd->thd", p.astype(np.float32),
                      vh.astype(np.float32)).astype(np.float32)  # (T,nH,hd)
        sg = (self.r.bf16_f32(f"backbone.layers.{li}.mixer.s_gate").astype(np.float32)
              * np.float32(self.sgate_factor))
        sg = (sg * sg).reshape(1, self.nH, c["head_dim"])
        z = (g.reshape(T, self.nH, c["head_dim"]).astype(np.float32) * sg)
        gate = np.empty_like(z)
        neg = z < 0
        gate[neg] = np.exp(z[neg]) / (1.0 + np.exp(z[neg]))  # no overflow
        gate[~neg] = 1.0 / (1.0 + np.exp(-z[~neg]))
        o = (o * gate).astype(np.float32)
        o = justnorm(o.reshape(T, -1))
        o = self.lin(o, self.r.bf16_f32(f"backbone.layers.{li}.mixer.o_proj.weight"))
        return o, kv

    def mlp(self, h: np.ndarray, li: int) -> np.ndarray:
        gate = self.lin(h, self.r.bf16_f32(f"backbone.layers.{li}.mixer.gate_proj.weight"))
        up = self.lin(h, self.r.bf16_f32(f"backbone.layers.{li}.mixer.up_proj.weight"))
        sg = (self.r.bf16_f32(f"backbone.layers.{li}.mixer.suv_gate").astype(np.float32)
              * np.float32(self.suv_factor))
        su = (self.r.bf16_f32(f"backbone.layers.{li}.mixer.suv_up").astype(np.float32)
              * np.float32(self.suv_factor))
        m = (silu(gate * sg) * (up * su)).astype(np.float32)
        m = justnorm(m)
        return self.lin(m, self.r.bf16_f32(f"backbone.layers.{li}.mixer.down_proj.weight"))

    def layer(self, h: np.ndarray, pos: np.ndarray, li: int,
              kv: dict | None):
        if self.is_attn[li]:
            out, kv = self.attn(h, pos, li, kv)
            a = self.r.bf16_f32(f"backbone.layers.{li}.attn_alpha")
            return slerp_alpha(h, out, a, self.alpha_scale), kv
        out = self.mlp(h, li)
        a = self.r.bf16_f32(f"backbone.layers.{li}.mlp_alpha")
        return slerp_alpha(h, out, a, self.alpha_scale), kv

    # -- backbone ------------------------------------------------------------
    def backbone(self, emb: np.ndarray, pos: np.ndarray,
                 caches: List[dict | None] | None = None):
        c = self.c
        n_in, n_rec = c["n_input_layers"], c["n_recurrent_layers"]
        h = emb.astype(np.float32, copy=False)
        if caches is None:
            caches = [None] * c["num_hidden_layers"]
        for li in range(n_in):
            h, caches[li] = self.layer(h, pos, li, caches[li])
        input_h = h
        state = input_h
        for _ in range(self.n_loops):
            h = justnorm(state + input_h)
            for li in range(n_in, n_in + n_rec):
                h, caches[li] = self.layer(h, pos, li, caches[li])
            state = justnorm(h)
        h = state
        for li in range(n_in + n_rec, c["num_hidden_layers"]):
            h, caches[li] = self.layer(h, pos, li, caches[li])
        return h, caches

    def logits(self, h: np.ndarray) -> np.ndarray:
        w = self.r.bf16_f32("lm_head.weight")
        ls = self.r.bf16_f32("lm_head.logit_scale").astype(np.float32)
        if self.bf16:
            h = bf16_round(h)
            w = bf16_round(w)
        CH = 8192
        outs = []
        for s in range(0, h.shape[0], CH):
            outs.append(h[s:s + CH] @ w.T)
        y = np.concatenate(outs, axis=0)
        y = y * (ls / np.float32(self.c["initializer_range"]))[None, :]
        return bf16_round(y) if self.bf16 else y

# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def topk(y: np.ndarray, k: int = 5) -> List[int]:
    return list(np.argsort(y)[::-1][:k].astype(np.int64))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.astype(np.float64), b.astype(np.float64))
                 / (np.linalg.norm(a.astype(np.float64))
                    * np.linalg.norm(b.astype(np.float64)) + 1e-30))

# ----------------------------------------------------------------------------
# Tiny deterministic fixture (float64 reference vs F32 fast path)
# ----------------------------------------------------------------------------

def run_selftest_main() -> int:

    rng = np.random.default_rng(12345)
    H, nH, nKV, hd, F, T = 32, 4, 2, 8, 48, 5
    rep = nH // nKV
    std, a_init, g0 = 0.0177, 0.05, H ** 0.5
    asc, sqkf, sgtf, suvf = a_init / std, hd ** 0.5 / std, (g0 ** 0.5) / std, H ** 0.5
    W = lambda r, c_: (rng.standard_normal((r, c_)) * std).astype(np.float32)
    P = lambda n_: (rng.standard_normal(n_) * std).astype(np.float32)
    qW, kW, vW, gW, oW = W(nH * hd, H), W(nKV * hd, H), W(nKV * hd, H), W(nH * hd, H), W(H, nH * hd)
    sqk, sgate = P(nH * hd), P(nH * hd)
    gaW, upW, dnW = W(F, H), W(F, H), W(H, F)
    suv_g = (rng.standard_normal(F) + 2.0).astype(np.float32)
    suv_u = (rng.standard_normal(F) + 2.0).astype(np.float32)
    alpha_a, alpha_m = P(H), P(H)
    inv = (1.0 / (10000.0 ** (np.arange(0, hd, 2) / hd))).astype(np.float32)
    h = rng.standard_normal((T, H)).astype(np.float32)
    pos = np.arange(T)

    def jn64(x):
        xf = x.astype(np.float64)
        return xf * (1.0 / np.sqrt((xf * xf).sum(-1, keepdims=True) + 1e-6))

    h64 = h.astype(np.float64)
    # --- attention: float64 reference ---
    q = (h64 @ qW.astype(np.float64).T).reshape(T, nH, hd)
    k = (h64 @ kW.astype(np.float64).T).reshape(T, nKV, hd)
    v = (h64 @ vW.astype(np.float64).T).reshape(T, nKV, hd)
    g = h64 @ gW.astype(np.float64).T
    ang = pos[:, None] * inv.astype(np.float64)[None, :]
    co, si = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    d2 = hd // 2
    q = np.concatenate([q[..., :d2] * co - q[..., d2:] * si,
                        q[..., :d2] * si + q[..., d2:] * co], -1)
    k = np.concatenate([k[..., :d2] * co - k[..., d2:] * si,
                        k[..., :d2] * si + k[..., d2:] * co], -1)
    s = (sqk.astype(np.float64) * sqkf).reshape(1, nH, hd)
    q = (s * s) * jn64(q)
    k = jn64(k)
    kh = np.repeat(k, rep, axis=1)
    vh = np.repeat(v, rep, axis=1)
    sc = np.einsum("thd,shd->hts", q, kh) / np.sqrt(float(hd))
    qq = np.arange(T)[:, None]
    kk = np.arange(T)[None, :]
    sc = np.where(kk > qq, -1e30, sc)
    sc -= sc.max(-1, keepdims=True)
    p = np.exp(sc)
    p /= p.sum(-1, keepdims=True)
    o = np.einsum("hts,shd->thd", p, vh)
    sgs = (sgate.astype(np.float64) * sgtf)
    sgs = (sgs * sgs).reshape(1, nH, hd)
    o = o * (1.0 / (1.0 + np.exp(-(g.reshape(T, nH, hd) * sgs))))
    ref_attn = (jn64(o.reshape(T, -1)) @ oW.astype(np.float64).T)
    # --- attention: F32 fast path (same code shape as DoryRef) ---
    qf = (h @ qW.T).reshape(T, nH, hd)
    kf = (h @ kW.T).reshape(T, nKV, hd)
    vf = (h @ vW.T).reshape(T, nKV, hd)
    gf = h @ gW.T
    qf, kf = rope_apply(qf, pos, inv), rope_apply(kf, pos, inv)
    sf = (sqk * np.float32(sqkf)).reshape(1, nH, hd)
    qf = ((sf * sf) * justnorm(qf)).astype(np.float32)
    kf = justnorm(kf)
    khf = np.repeat(kf, rep, axis=1)
    vhf = np.repeat(vf, rep, axis=1)
    scf = np.einsum("thd,shd->hts", qf, khf) / np.float32(np.sqrt(float(hd)))
    scf = np.where(kk > qq, np.float32(-1e30), scf).astype(np.float32)
    scf -= scf.max(-1, keepdims=True)
    pf = np.exp(scf)
    pf /= pf.sum(-1, keepdims=True)
    of = np.einsum("hts,shd->thd", pf.astype(np.float32), vhf.astype(np.float32))
    sgf = (sgate * np.float32(sgtf))
    sgf = (sgf * sgf).reshape(1, nH, hd)
    off = (of * (1.0 / (1.0 + np.exp(-(gf.reshape(T, nH, hd) * sgf))))).astype(np.float32)
    fast_attn = justnorm(off.reshape(T, -1)) @ oW.T
    # --- MLP both paths ---
    g64 = h64 @ gaW.astype(np.float64).T
    u64 = h64 @ upW.astype(np.float64).T
    gg = g64 * suv_g.astype(np.float64) * suvf
    m64 = (gg / (1.0 + np.exp(-gg))) * (u64 * suv_u.astype(np.float64) * suvf)
    ref_mlp = jn64(m64) @ dnW.astype(np.float64).T
    gf2 = h @ gaW.T
    uf2 = h @ upW.T
    mfast = (silu(gf2 * suv_g * np.float32(suvf)) * (uf2 * suv_u * np.float32(suvf)))
    fast_mlp = justnorm(mfast) @ dnW.T
    # --- slerp both paths ---
    def slerp64(a, b, al):
        ya, yb = jn64(a), jn64(b)
        z = ya + np.abs(al * asc) * (yb - ya)
        return z * (1.0 / np.sqrt((z * z).sum(-1, keepdims=True) + 1e-6))
    ref_sa = slerp64(h64, ref_attn, alpha_a.astype(np.float64))
    fast_sa = slerp_alpha(h, fast_attn, alpha_a, asc)
    ref_sm = slerp64(h64, ref_mlp, alpha_m.astype(np.float64))
    fast_sm = slerp_alpha(h, fast_mlp, alpha_m, asc)
    # --- decode-vs-prefill consistency on the fixture ---
    # single-query decode at last position reusing KV must match prefill row
    last = T - 1
    qd = (h[last:last + 1] @ qW.T).reshape(1, nH, hd)
    kd = (h[last:last + 1] @ kW.T).reshape(1, nKV, hd)
    vd = (h[last:last + 1] @ vW.T).reshape(1, nKV, hd)
    gd = h[last:last + 1] @ gW.T
    qd, kd = rope_apply(qd, np.array([last]), inv), rope_apply(kd, np.array([last]), inv)
    kd = np.concatenate([kf[:last], justnorm(kd)], axis=0)
    vd = np.concatenate([vf[:last].reshape(last, nKV, hd), vd.reshape(1, nKV, hd)], axis=0)
    sfd = (sqk * np.float32(sqkf)).reshape(1, nH, hd)
    qd = ((sfd * sfd) * justnorm(qd)).astype(np.float32)
    khd = np.repeat(kd, rep, axis=1)
    vhd = np.repeat(vd, rep, axis=1)
    scd = np.einsum("thd,shd->hts", qd, khd) / np.float32(np.sqrt(float(hd)))
    scd -= scd.max(-1, keepdims=True)
    pd = np.exp(scd)
    pd /= pd.sum(-1, keepdims=True)
    od = np.einsum("hts,shd->thd", pd.astype(np.float32), vhd.astype(np.float32))
    odd = (od * (1.0 / (1.0 + np.exp(-(gd.reshape(1, nH, hd) * sgf))))).astype(np.float32)
    dec_attn = justnorm(odd.reshape(1, -1)) @ oW.T
    da = float(np.abs(fast_attn.astype(np.float64) - ref_attn).max())
    dm = float(np.abs(fast_mlp.astype(np.float64) - ref_mlp).max())
    ds = max(float(np.abs(fast_sa.astype(np.float64) - ref_sa).max()),
             float(np.abs(fast_sm.astype(np.float64) - ref_sm).max()))
    dd = float(np.abs(dec_attn.astype(np.float64) - fast_attn[last:last + 1].astype(np.float64)).max())
    print(f"selftest attn   F32-vs-F64 maxabs = {da:.3e} (tol 5e-5)")
    print(f"selftest mlp    F32-vs-F64 maxabs = {dm:.3e} (tol 5e-5)")
    print(f"selftest slerp  F32-vs-F64 maxabs = {ds:.3e} (tol 5e-5)")
    print(f"selftest decode-vs-prefill maxabs = {dd:.3e} (tol 1e-5)")
    ok = da < 5e-5 and dm < 5e-5 and ds < 5e-5 and dd < 1e-5
    # --- Q8F16 block decoder fixture (signed int8 + f16 scale) ---
    # one 32-group: scale = 0.5 exactly, q = [-128..-101] + zeros
    import struct as _st
    qvals = np.array([-128 + i if i < 28 else 0 for i in range(32)], dtype=np.int8)
    qraw = np.empty(34, dtype=np.uint8)
    qraw[0:2] = np.frombuffer(_st.pack("<e", np.float16(0.5)), dtype=np.uint8)
    qraw[2:] = qvals.view(np.uint8)
    got = HfqReader._decode_q8f16(qraw, 32)
    want = qvals.astype(np.float32) * np.float32(0.5)
    dq = float(np.abs(got.astype(np.float64) - want.astype(np.float64)).max())
    print(f"selftest q8f16   signed-scale maxabs = {dq:.3e} (tol 0)")
    ok = ok and dq == 0.0
    # --- F32 bit-exactness fixture ---
    fvals = np.array([1.0, -2.5, 0.0, 3.140625], dtype=np.float32)
    back = np.frombuffer(fvals.tobytes(), dtype="<f4").astype(np.float32)
    df = float(np.abs(back.astype(np.float64) - fvals.astype(np.float64)).max())
    print(f"selftest f32     bits maxabs = {df:.3e} (tol 0)")
    ok = ok and df == 0.0
    return 0 if ok else 1

# ----------------------------------------------------------------------------
# Full-model run
# ----------------------------------------------------------------------------

def encode_prompt(tokenizer_json: str, prompt: str) -> List[int]:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_json)
    return tok.encode(prompt).ids


def encode_prompt_hfq(metadata: dict, prompt: str) -> List[int]:
    """Encode with the tokenizer embedded in HFQ metadata (no sidecar files)."""
    from tokenizers import Tokenizer
    blob = metadata.get("tokenizer")
    if not isinstance(blob, str) or not blob:
        raise SystemExit("HFQ metadata has no embedded `tokenizer` string")
    fd, tmppath = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(blob)
        tok = Tokenizer.from_file(tmppath)
        return tok.encode(prompt).ids
    finally:
        try:
            os.unlink(tmppath)
        except OSError:
            pass


def _open_source(model: str):
    """Open a model path: .hfq file (self-contained) or BF16 safetensors dir.

    Returns (cfg, reader, model_label, tokenizer_ids_fn, source_kind).
    """
    if os.path.isfile(model) and model.endswith(".hfq"):
        reader = HfqReader(model)
        meta = reader.meta
        cfg = meta.get("config")
        if not isinstance(cfg, dict):
            raise SystemExit(f"{model}: HFQ metadata has no `config` object")
        return cfg, reader, model, lambda prompt: encode_prompt_hfq(meta, prompt), "hfq"
    cfg = json.load(open(os.path.join(model, "config.json")))
    reader = STReader(os.path.join(model, "model.safetensors"))
    tok_json = os.path.join(model, "tokenizer.json")
    return cfg, reader, model, lambda prompt: encode_prompt(tok_json, prompt), "dir"


def run_full(args) -> int:
    model_arg = args.model
    cfg, reader, model_label, encode_fn, kind = _open_source(model_arg)
    assert cfg["n_recurrent_loops"] == 2, "config changed; oracle pins 1 loop explicitly"
    if kind == "dir":
        nparams = sum(int(np.prod(v["shape"])) for k, v in reader.header.items()
                      if k != "__metadata__")
        ntensors = len(reader.header) - ("__metadata__" in reader.header)
    else:
        nparams = sum(int(np.prod(v["shape"])) for v in reader.tensors.values())
        ntensors = len(reader.tensors)
    print(f"tensors={ntensors} "
          f"params={nparams / 1e9:.3f}B fp={reader.header_fingerprint()} "
          f"size={reader.file_size}", flush=True)
    toks = encode_fn(args.prompt)
    S = toks[:args.seq]
    if len(S) < args.seq:
        raise SystemExit(f"prompt encodes to {len(toks)} ids < --seq {args.seq}")
    P, D = args.prefill, S[args.prefill:args.seq]
    print(f"tokens[{len(S)}] = {S} (prefill {P}, decode {len(D)})", flush=True)
    ref = DoryRef(cfg, reader, n_loops=1, bf16=False)

    # prefill
    import time
    t0 = time.time()
    emb = reader.row_gather_f32("backbone.embeddings.weight", np.array(S[:P]))
    h_pre, caches = ref.backbone(emb, np.arange(P))
    log_pre = ref.logits(h_pre)
    t1 = time.time()
    print(f"prefill {P} tok: {t1 - t0:.1f}s", flush=True)

    # KV-cached decode of the tail, one token at a time
    h_dec_rows, log_dec_rows = [], []
    t0 = time.time()
    for j, tid in enumerate(D):
        pos = np.array([P + j])
        e = reader.row_gather_f32("backbone.embeddings.weight", np.array([tid]))
        h, caches = ref.backbone(e, pos, caches)
        h_dec_rows.append(h[0])
        log_dec_rows.append(ref.logits(h)[0])
    t1 = time.time()
    print(f"decode {len(D)} tok: {t1 - t0:.1f}s", flush=True)
    h_dec = np.stack(h_dec_rows).astype(np.float32)
    log_dec = np.stack(log_dec_rows).astype(np.float32)

    # self-consistency: decode rows must reproduce a full-prefill reference.
    # (full prefill over S reuses no cache; compares at tail positions)
    emb_full = reader.row_gather_f32("backbone.embeddings.weight", np.array(S))
    h_full, _ = ref.backbone(emb_full, np.arange(len(S)))
    log_full = ref.logits(h_full)
    dh = np.abs(h_full[P:].astype(np.float64) - h_dec.astype(np.float64)).max()
    dl = np.abs(log_full[P:].astype(np.float64) - log_dec.astype(np.float64)).max()
    cosmin = min(cosine(h_full[P + j], h_dec[j]) for j in range(len(D)))
    print(f"self-consistency decode-vs-prefill: max|dhid|={dh:.3e} "
          f"max|dlogit|={dl:.3e} mincos={cosmin:.8f}")

    # greedy next token from the last position (informational only)
    last_top = topk(log_full[-1], 8)
    print("last-position top8:", last_top,
          [round(float(log_full[-1][i]), 3) for i in last_top])

    out = args.out
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "logits_f32.npy"), log_full.astype(np.float32))
    np.save(os.path.join(out, "hidden_f32.npy"), h_full.astype(np.float32))
    manifest = {
        "source_math": SRC_ATTRIBUTION,
        "model_dir": model_label,
        "model_kind": kind,
        "config_snapshot": {k: cfg[k] for k in (
            "vocab_size", "hidden_size", "num_hidden_layers", "intermediate_size",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "n_input_layers", "n_recurrent_layers", "n_output_layers",
            "n_recurrent_loops", "sliding_window", "ngpt_alpha_init",
            "initializer_range", "attention_output_gate_logit_scale_init",
            "tie_word_embeddings", "layer_types")},
        "loops_executed": 1,
        "precision": ("F32 accumulate, Q8F16/F32 dequant weights, "
                      "no activation rounding"
                      if kind == "hfq" else
                      "F32 accumulate, exact BF16->F32 weights, no activation rounding"),
        "weight_file_prefix_sha256_16": reader.header_fingerprint(),
        "weight_header_sha256": hashlib.sha256(reader.raw_header).hexdigest(),
        "weight_header_bytes": reader.header_len,
        "prompt": args.prompt,
        "tokens": S,
        "n_prefill": P,
        "self_consistency": {
            "max_abs_hidden_diff": float(dh),
            "max_abs_logit_diff": float(dl),
            "min_hidden_cosine": float(cosmin),
        },
        "per_position": [
            {"pos": i, "top1": int(np.argmax(log_full[i])),
             "top5": [int(x) for x in topk(log_full[i])],
             "top1_logit": float(log_full[i].max()),
             "hidden_norm": float(np.linalg.norm(h_full[i].astype(np.float64)))}
            for i in range(len(S))
        ],
        "last_position_top8": [int(x) for x in last_top],
        "deps": {"numpy": np.__version__, "safetensors": _pkg_ver("safetensors"),
                 "tokenizers": _pkg_ver("tokenizers"), "torch": None},
        "files": {"logits": "logits_f32.npy", "hidden": "hidden_f32.npy"},
    }
    with open(os.path.join(out, "dory_one_loop_f32.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {out}/dory_one_loop_f32.json + logits_f32.npy + hidden_f32.npy")
    ok = dh < 1e-4 and dl < 5e-2 and cosmin > 0.99999
    print("FULL-RUN SELF-CONSISTENCY " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def run_bf16twin(args) -> int:
    """Measure the BF16-activation envelope over all artifact positions.

    Runs prefill over the artifact's tokens in twin mode and compares
    all-position logits/hidden against the F32 artifact. The measured
    deltas ground the native-parity tolerances.
    """
    art = args.artifact
    man = json.load(open(os.path.join(art, "dory_one_loop_f32.json")))
    model_src = man["model_dir"]
    cfg, reader, _, _, _ = _open_source(model_src)
    S = man["tokens"]
    ref = DoryRef(cfg, reader, n_loops=1, bf16=True)
    import time
    t0 = time.time()
    emb = reader.row_gather_f32("backbone.embeddings.weight", np.array(S))
    h, _ = ref.backbone(emb, np.arange(len(S)))
    log = ref.logits(h)
    print(f"twin prefill {len(S)} tok: {time.time() - t0:.1f}s", flush=True)
    log_f32 = np.load(os.path.join(art, "logits_f32.npy"))
    hid_f32 = np.load(os.path.join(art, "hidden_f32.npy"))
    dl = np.abs(log.astype(np.float64) - log_f32.astype(np.float64))
    dh = np.abs(h.astype(np.float64) - hid_f32.astype(np.float64))
    top1_same = sum(int(np.argmax(log[i])) == int(np.argmax(log_f32[i]))
                    for i in range(len(S)))
    env = {
        "mode": "BF16 activation rounding (RNE) on every matmul in/out, F32 accumulate",
        "max_abs_logit_diff": float(dl.max()),
        "mean_abs_logit_diff": float(dl.mean()),
        "p99_abs_logit_diff": float(np.quantile(dl, 0.99)),
        "max_abs_hidden_diff": float(dh.max()),
        "min_hidden_cosine": float(min(cosine(h[i], hid_f32[i]) for i in range(len(S)))),
        "top1_match": f"{top1_same}/{len(S)}",
    }
    print(json.dumps(env, indent=2))
    with open(os.path.join(art, "bf16_envelope.json"), "w") as f:
        json.dump(env, f, indent=2)
    return 0


# ----------------------------------------------------------------------------
# Native comparison
# ----------------------------------------------------------------------------

# Parity tolerances (tight, per Main direction): native keeps BF16 weights
# with F32 activations, i.e. the SAME arithmetic the oracle models (exact
# BF16->F32 weights, F32 accumulate, no activation rounding). The BF16 twin
# envelope (bf16_envelope.json) is reported for context ONLY — it must NOT
# set the pass bar. Expected reduction-order differences are ~1e-3..1e-2 on
# logits; the bar is set there. First measured discrepancy rules: if native
# lands outside, the port (not the tolerance) is suspect.
TOL = {
    "min_hidden_cosine": 0.99999,
    "max_abs_logit_diff": 2e-2,   # hard FAIL above; WARN above 1e-3
    "warn_abs_logit_diff": 1e-3,
    "required_top1_match": 16,    # all positions must agree
}


def run_compare(args) -> int:
    art = args.artifact
    man = json.load(open(os.path.join(art, "dory_one_loop_f32.json")))
    nat = json.load(open(args.compare))
    assert nat.get("loops", 1) == 1, "native dump must be one-loop"
    assert list(nat["tokens"]) == list(man["tokens"]), \
        f"token mismatch: native {nat['tokens']} vs ref {man['tokens']}"
    log_r = np.load(os.path.join(art, "logits_f32.npy")).astype(np.float64)
    hid_r = np.load(os.path.join(art, "hidden_f32.npy")).astype(np.float64)
    log_n = np.asarray(nat["logits"], dtype=np.float64)
    assert log_n.shape == log_r.shape, f"{log_n.shape} vs {log_r.shape}"
    npos = log_r.shape[0]
    dl = np.abs(log_n - log_r)
    top1 = sum(int(np.argmax(log_n[i])) == int(np.argmax(log_r[i])) for i in range(npos))
    print(f"positions={npos} max|dlogit|={dl.max():.5f} "
          f"(FAIL>{TOL['max_abs_logit_diff']:.1e} WARN>{TOL['warn_abs_logit_diff']:.1e}) "
          f"mean={dl.mean():.5f} top1={top1}/{npos}")
    env_path = os.path.join(art, "bf16_envelope.json")
    if os.path.exists(env_path):
        env = json.load(open(env_path))
        print(f"(context only, NOT the bar: BF16-twin max|dlogit|={env['max_abs_logit_diff']:.4f} "
              f"mincos={env['min_hidden_cosine']:.6f})")
    ok = True
    if dl.max() > TOL["max_abs_logit_diff"]:
        print("FAIL max|dlogit| above tolerance"); ok = False
    elif dl.max() > TOL["warn_abs_logit_diff"]:
        print("WARN max|dlogit| above 1e-3 aim (triage reduction order before widening)");
    if top1 < npos:
        print("FAIL top-1 mismatch"); ok = False
    if "hidden" in nat and nat["hidden"] is not None:
        hid_n = np.asarray(nat["hidden"], dtype=np.float64)
        cosmin = min(cosine(hid_n[i], hid_r[i]) for i in range(npos))
        print(f"min hidden cosine={cosmin:.8f} (tol {TOL['min_hidden_cosine']})")
        if cosmin < TOL["min_hidden_cosine"]:
            print("FAIL hidden cosine below tolerance"); ok = False
    else:
        print("(no native hidden vectors; cosine check skipped)")
    # per-position detail catches state errors (e.g. stale KV after pos k)
    for i in range(npos):
        mark = "" if int(np.argmax(log_n[i])) == int(np.argmax(log_r[i])) else "  <-- TOP1 DIFF"
        print(f"  pos {i:2d} tok={man['tokens'][i]:6d} "
              f"ref_top1={int(np.argmax(log_r[i])):6d} "
              f"nat_top1={int(np.argmax(log_n[i])):6d} "
              f"max|dlogit|={dl[i].max():.5f}{mark}")
        if mark:
            ok = False
    print("NATIVE PARITY " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def _pkg_ver(name: str):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Dory one-loop CPU reference oracle")
    ap.add_argument("--model", default="./dory-bf16",
                    help="BF16 safetensors dir or self-contained Dory .hfq "
                         "(qt=3 Q8F16 + qt=2 F32)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--bf16twin", action="store_true")
    ap.add_argument("--compare", metavar="NATIVE_JSON", default=None)
    ap.add_argument("--artifact", default="benchmarks/references/dory_one_loop")
    ap.add_argument("--prompt", default="The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is",
                    help="prompt; first --seq ids form the fixed sequence")
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--prefill", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.selftest:
        return run_selftest_main()
    if args.run:
        args.out = args.out or args.artifact
        return run_full(args)
    if args.bf16twin:
        return run_bf16twin(args)
    if args.compare:
        return run_compare(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
