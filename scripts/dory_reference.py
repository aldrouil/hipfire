#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 hipfire contributors — see LICENSE and NOTICE in the project root.
"""Independent CPU (NumPy, float32) reference oracle for Dory-4B, configurable
multi-loop inference (loops 1..=4) with per_loop / last_loop KV semantics.

Math source (adapted equations, Apache-2.0):
  https://raw.githubusercontent.com/vllm-project/vllm/55076950b6c155f384d6766d63563c3bc20a0e3f/vllm/model_executor/models/dory.py
  (DoryForCausalLM / DoryBackbone / DoryAttention / DoryMLP / justnorm / slerp_norm.)

Purpose: ground-truth logits + last-hidden states for the native hipfire
`hipfire-arch-dory` bring-up (I->R^N->O: input block once, shared recurrent
block N=1..4 passes with justnorm(state+input_h) reinjection each pass, output
block once; same absolute position every loop). Native baseline: BF16 weights kept
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
  python3 scripts/dory_reference.py --run --model dory-4b.q8.hfq --out DIR
      # canonical same-weight pack (config/tokenizer from embedded HFQ
      # metadata, no sidecar dir): F32-reference schedule (per_loop bulk
      # prefill + tokenwise decode; last_loop fully tokenwise) with
      # same-schedule self-consistency; emits artifact DIR/.
      # Never mix a BF16-dir CPU run against a Q8 native dump.
Native JSON schema for --compare (EFFECTIVE kv_mode; F32 canonical, packed honest):
  {"tokens": [...], "logits": [[...131072 floats...], ...],
   "hidden": [[...2560...], ...] (optional, for cosine),
   "loops": N, "recurrent_kv_cache_mode": "per_loop"|"last_loop",
   "kv_mode": EFFECTIVE f32|q8|fwht2|fwht3|fwht4 (never raw auto).
   Packed whole-model keeps the strict 0.02 max|dlogit| bar and may honestly
   FAIL; --kv-trace replay then isolates codec vs state causes.}
   "positions": "prefill+decode"}
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
# KV-cache quantization oracle (production codec mirrors).
# ----------------------------------------------------------------------------
#
# Source: kernels/src/kv_cache_write_q8_0.hip,
#   kv_cache_write_asym_k_fwht{2,3,4}.hip, turbo_common.h.
# Dory geometry: n_kv_heads=4, head_dim=256. V is always Q8_0 (v_mode_bits=8);
# only the K writer rotates. Shared-cache constructors set
# boundary_layers=0 / layer_is_boundary=[] (see saddle-core/src/kv.rs
# new_gpu_{q8,fwht2,fwht3,fwht4}_filtered) — i.e. NO Q8 boundary assumption;
# the oracle mirrors the ACTUAL constructor state, not a first/last-Q8 rule.
#
# Canonical math (bit-parity targets unless noted):
#   Q8 K/V per 32-dim block b: amax_b = max|x_i|; s_b = f16(amax_b/127)
#     (f16 via f16 round-trip of the f32 quotient, matching (_Float16)scale on
#     device); q_i = clamp(rint(x_i / s_f32... see note), -127, 127) with
#     round-half-to-even (__float2int_rn); decode x~ = f32(s_b) * q_i.
#     NOTE: x_i / s uses the STORED f16 scale converted back to f32, not the
#     f32 amax/127 (divisor mismatch shows up ~1ulp on adversarial inputs;
#     the oracle uses stored-scale division exactly like the kernel's
#     reciprocal-multiply path only up to that 1ulp — comparison tolerances
#     below absorb it).
#   FWHT signs: gen_fwht_signs(seed, n): state=seed; per element
#     state = (state*1103515245 + 12345) & 0x7fffffff; sign = +1 iff bit16 set.
#     K rotation uses seeds (42, 1042): 256-wide for fwht3, 128-wide shared
#     across halves for fwht2/4 (halves [0..128)/[128..256) independent).
#   FWHT-256 forward: x*=s1; 8 butterfly passes (strides 1..128);
#     x *= (1/16) * s2. FWHT-128 forward: x*=s1; 7 passes (strides 1..64);
#     x *= (1/sqrt(128)) * s2.
#   Norm: cnorm input x is the post-RoPE, post-justnorm K row (whole head L2).
#     fwht3: inv = 1/orig_norm (no sqrt2); fwht2/4 with head_dim=256:
#     inv = sqrt(2)/orig_norm (128-wide coords of a unit 256-head have
#     variance 1/256 vs the 128-family 1/128 the centroids assume).
#   Centroids/thresholds: TURBO_C2/C4 (128-family, fwht2/4), TURBO_C2_256 /
#     TURBO_C3_256 / TURBO_C4_256 (fwht3 3-bit uses the 256-family LUT but the
#     _256 thresholds). Thresholds are exact `>` comparisons producing
#     index = #{thresholds exceeded} (equivalent to the branchless sums).
#   Layouts @D256 (per head): fwht2 [4B cnorm][2x32B halves, 4idx/B];
#     fwht3 [4B cnorm][32x3B, 8idx little-endian per 24b];
#     fwht4 [4B cnorm][2x64B halves, 2idx/B low-first]; Q8 [8x34B blocks:
#     f16 scale + 32 int8].
#   Attention-equivalence check: Hadamard rotation is unitary (up to the
#     absorbed scale), so Q.K == Qr.Kr in exact arithmetic; the oracle
#     verifies production by comparing ROTATED-space dots against dequantized
#     rotated K, which catches wrong-signs/wrong-norm/packed-offset bugs
#     while staying independent of Q-readout numerics.
#
# Quantizer families (MUST match turbo_common.h exactly):
TURBO_C2 = [-0.133466, -0.040022, 0.040022, 0.133466]
TURBO_C3 = [-0.190685, -0.117832, -0.065717, -0.021460, 0.021460, 0.065717,
            0.117832, 0.190685]
TURBO_C4 = [-0.241565, -0.182875, -0.143012, -0.111016, -0.083262, -0.057983,
            -0.034295, -0.011225, 0.011225, 0.034295, 0.057983, 0.083262,
            0.111016, 0.143012, 0.182875, 0.241565]
TURBO_C2_256 = [-0.094376, -0.028300, 0.028300, 0.094376]
TURBO_C3_256 = [-0.134860, -0.083320, -0.046469, -0.015176, 0.015176,
                0.046469, 0.083320, 0.134860]
TURBO_C4_256 = [-0.170807, -0.129321, -0.101134, -0.078505, -0.058869,
                -0.041003, -0.024249, -0.007938, 0.007938, 0.024249, 0.041003,
                0.058869, 0.078505, 0.101134, 0.129321, 0.170807]
THR_C2 = [-0.086744, 0.0, 0.086744]
THR_C3 = [-0.154258, -0.091775, -0.043589, 0.0, 0.043589, 0.091775, 0.154258]
THR_C4 = [-0.212220, -0.162944, -0.127014, -0.097139, -0.070622, -0.046139,
          -0.022760, 0.0, 0.022760, 0.046139, 0.070622, 0.097139, 0.127014,
          0.162944, 0.212220]
THR_C3_256 = [-0.109068, -0.064903, -0.030827, 0.0, 0.030827, 0.064903,
              0.109068]
# fwht2 shares the 128-family thresholds; fwht4 shares them too.
THR_C2_FWHT = THR_C2
THR_C4_FWHT = THR_C4

KV_MODES = ("f32", "q8", "fwht2", "fwht3", "fwht4")


def gen_fwht_signs(seed: int, n: int) -> np.ndarray:
    """Deterministic +/-1 signs (matches KvCache::gen_fwht_signs LCG)."""
    state = int(seed) & 0xFFFFFFFF
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        out[i] = np.float32(1.0 if ((state >> 16) & 1) else -1.0)
    return out


def _fwht_butterfly(x: np.ndarray) -> np.ndarray:
    """In-place radix-2 Hadamard stage loop (exact kernel pass order)."""
    x = x.copy()
    n = x.shape[0]
    stride = 1
    while stride < n:
        for j in range(0, n, stride * 2):
            for k in range(stride):
                a = float(x[j + k])
                b = float(x[j + k + stride])
                x[j + k] = np.float32(a + b)
                x[j + k + stride] = np.float32(a - b)
        stride <<= 1
    return x.astype(np.float32, copy=False)


def fwht_forward_256(x: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Signed-FWHT-256 forward (fwht_shfl_forward_256 scalar equivalent)."""
    y = (x.astype(np.float32) * s1[:256]).astype(np.float32)
    y = _fwht_butterfly(y)
    return (y * (np.float32(0.0625) * s2[:256])).astype(np.float32)


def fwht_forward_128(x: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Signed-FWHT-128 forward (fwht_shfl_forward scalar equivalent)."""
    y = (x.astype(np.float32) * s1[:128]).astype(np.float32)
    y = _fwht_butterfly(y)
    return (y * (np.float32(0.0883883476) * s2[:128])).astype(np.float32)


def _quant_idx(x: float, thr: List[float]) -> int:
    """Branchless-sum equivalent: #{t in thr : x > t}."""
    c = 0
    for t in thr:
        if x > t:
            c += 1
    return c


def _f32_to_f16bytes(v: float) -> bytes:
    import struct as _st
    return _st.pack("<e", float(np.float32(v)))


def _f16bytes_to_f32(b: bytes) -> np.float32:
    import struct as _st
    return np.float32(_st.unpack("<e", b)[0])


def q8_encode_row(x: np.ndarray) -> Tuple[bytes, np.ndarray]:
    """Q8_0 encode one head row (D=256); returns (blob, dequant F32).

    Mirrors kv_cache_write_q8_0.hip: per-32 block amax, f16 scale
    (_Float16)(amax/127) via RNE f32->f16, q = rint(x * (127/amax)) clamped
    to [-127, 127] (__float2int_rn = round-half-to-even; np.rint matches on
    the f32-magnitude inputs here, exact .5 ties vanishingly rare).
    """
    xd = x.astype(np.float32)
    hd = xd.shape[0]
    assert hd % 32 == 0
    blob = bytearray()
    dec = np.empty(hd, dtype=np.float32)
    for b in range(hd // 32):
        blk = xd[b * 32:(b + 1) * 32]
        amax = float(np.abs(blk).max()) if blk.size else 0.0
        f32s = np.float32(amax / 127.0) if amax > 0 else np.float32(0.0)
        sb = _f32_to_f16bytes(f32s)
        blob += sb
        inv = np.float32(127.0 / amax) if amax > 0 else np.float32(0.0)
        for v in blk:
            q = int(np.rint(float(np.float32(v) * inv))) if amax > 0 else 0
            q = max(-127, min(127, q))
            blob.append(q & 0xFF)
        sf = _f16bytes_to_f32(sb)
        dec[b * 32:(b + 1) * 32] = (np.frombuffer(
            bytes(blob[-32:]), dtype=np.int8).astype(np.float32) * sf)
    return bytes(blob), dec


def q8_decode_blob(blob: bytes, hd: int) -> np.ndarray:
    """Decode a Q8_0 head blob to F32 (independent of the encoder above)."""
    import struct as _st
    out = np.empty(hd, dtype=np.float32)
    off = 0
    for b in range(hd // 32):
        sdec = _st.unpack("<e", blob[off:off + 2])[0]
        qs = np.frombuffer(blob[off + 2:off + 34], dtype=np.int8)
        out[b * 32:(b + 1) * 32] = qs.astype(np.float32) * np.float32(sdec)
        off += 34
    return out


def fwht_encode_k(x: np.ndarray, mode: str, s1: np.ndarray,
                  s2: np.ndarray) -> Tuple[bytes, np.ndarray]:
    """FWHT K-row encode (one head, D=256). Returns (blob, rotated dequant).

    Dequant is in ROTATED space (cnorm * LUT[idx]) — the exact values the
    attention tile dots against the rotated Q. Callers compare production
    rotated dots against this, and full-attention outputs against a softmax
    over these values (see kv_attention_ref).
    """
    import struct as _st
    xd = x.astype(np.float32)
    assert xd.shape[0] == 256, "dory oracle pins D=256"
    orig = float(np.sqrt(float(np.sum(xd.astype(np.float64) ** 2))))
    if mode == "fwht3":
        assert orig > 0
        xr = fwht_forward_256(xd / np.float32(orig), s1, s2)
        idx = np.array([_quant_idx(float(v), THR_C3_256) for v in xr],
                       dtype=np.uint8)
        lut = TURBO_C3_256
        recon = float(np.sqrt(float(np.sum(
            np.array([lut[i] for i in idx], dtype=np.float64) ** 2))))
        cnorm = np.float32(orig / recon) if recon > 0 else np.float32(orig)
        blob = bytearray(_st.pack("<f", float(cnorm)))
        for t in range(32):
            pack = 0
            for i in range(8):
                pack |= (int(idx[t * 8 + i]) & 7) << (i * 3)
            blob += bytes([pack & 0xFF, (pack >> 8) & 0xFF,
                           (pack >> 16) & 0xFF])
        dec = (np.array([lut[i] for i in idx],
                        dtype=np.float32) * cnorm).astype(np.float32)
        return bytes(blob), dec
    elif mode in ("fwht2", "fwht4"):
        q_scale = np.float32(1.41421356)  # sqrt(2): 128-wide of a unit 256-head
        inv = np.float32(q_scale / orig) if orig > 0 else np.float32(0.0)
        xn = (xd * inv).astype(np.float32)
        nbits, lut, bph = ((2, TURBO_C2, 68) if mode == "fwht2"
                           else (4, TURBO_C4, 132))
        idx = np.empty(256, dtype=np.uint8)
        thr = THR_C2_FWHT if mode == "fwht2" else THR_C4_FWHT
        for half in range(2):
            seg = xn[half * 128:(half + 1) * 128]
            xr = fwht_forward_128(seg, s1, s2)
            for i, v in enumerate(xr):
                idx[half * 128 + i] = _quant_idx(float(v), thr)
        recon = float(np.sqrt(float(np.sum(
            np.array([lut[i] for i in idx], dtype=np.float64) ** 2))))
        cnorm = np.float32(orig / recon) if recon > 0 else np.float32(orig)
        blob = bytearray(_st.pack("<f", float(cnorm)))
        if mode == "fwht2":
            for half in range(2):
                for t in range(32):
                    b = 0
                    for i in range(4):
                        b |= (int(idx[half * 128 + t * 4 + i]) & 3) << (i * 2)
                    blob.append(b)
        else:
            for half in range(2):
                for t in range(32):
                    i0 = idx[half * 128 + t * 4 + 0]
                    i1 = idx[half * 128 + t * 4 + 1]
                    i2 = idx[half * 128 + t * 4 + 2]
                    i3 = idx[half * 128 + t * 4 + 3]
                    blob.append((int(i0) & 0xF) | ((int(i1) & 0xF) << 4))
                    blob.append((int(i2) & 0xF) | ((int(i3) & 0xF) << 4))
        dec = (np.array([lut[i] for i in idx],
                        dtype=np.float32) * cnorm).astype(np.float32)
        return bytes(blob), dec
    else:
        raise SystemExit(f"unknown KV mode {mode}")


def fwht_decode_k_rotated(blob: bytes, mode: str) -> np.ndarray:
    """Decode an FWHT K blob to ROTATED-space F32 (independent path)."""
    import struct as _st
    cnorm = _st.unpack("<f", blob[:4])[0]
    if mode == "fwht3":
        out = np.empty(256, dtype=np.float32)
        for t in range(32):
            pack = blob[4 + t * 3] | (blob[4 + t * 3 + 1] << 8) | (
                blob[4 + t * 3 + 2] << 16)
            for i in range(8):
                out[t * 8 + i] = np.float32(
                    cnorm * TURBO_C3_256[(pack >> (i * 3)) & 7])
        return out
    if mode == "fwht2":
        out = np.empty(256, dtype=np.float32)
        for half in range(2):
            for t in range(32):
                b = blob[4 + half * 32 + t]
                for i in range(4):
                    out[half * 128 + t * 4 + i] = np.float32(
                        cnorm * TURBO_C2[(b >> (i * 2)) & 3])
        return out
    if mode == "fwht4":
        out = np.empty(256, dtype=np.float32)
        for half in range(2):
            for t in range(32):
                b0 = blob[4 + half * 64 + t * 2]
                b1 = blob[4 + half * 64 + t * 2 + 1]
                out[half * 128 + t * 4 + 0] = np.float32(cnorm * TURBO_C4[b0 & 0xF])
                out[half * 128 + t * 4 + 1] = np.float32(
                    cnorm * TURBO_C4[(b0 >> 4) & 0xF])
                out[half * 128 + t * 4 + 2] = np.float32(cnorm * TURBO_C4[b1 & 0xF])
                out[half * 128 + t * 4 + 3] = np.float32(
                    cnorm * TURBO_C4[(b1 >> 4) & 0xF])
        return out
    raise SystemExit(f"unknown KV mode {mode}")


def kv_encode_pos(krow: np.ndarray, vrow: np.ndarray, mode: str, s1: np.ndarray,
                  s2: np.ndarray) -> Tuple[bytes, bytes, np.ndarray, np.ndarray]:
    """Encode one (K,V) position; returns (kblob, vblob, kdec, vdec).

    kdec is ROTATED-space for fwht* (what the tile dots), natural-space for
    q8/f32. vdec is always natural-space Q8 dequant.
    """
    hd = krow.shape[0]
    assert hd == 256 and vrow.shape[0] == 256
    if mode == "f32":
        return (krow.astype(np.float32).tobytes(), vrow.astype(np.float32).tobytes(),
                krow.astype(np.float32), vrow.astype(np.float32))
    vblob, vdec = q8_encode_row(vrow)
    if mode == "q8":
        kblob, kdec = q8_encode_row(krow)
        return kblob, vblob, kdec, vdec
    kblob, kdec = fwht_encode_k(krow, mode, s1, s2)
    return kblob, vblob, kdec, vdec


def kv_attention_ref(q: np.ndarray, kdec_rows: List[np.ndarray],
                     vdec_rows: List[np.ndarray], mode: str, s1: np.ndarray,
                     s2: np.ndarray, positions: List[int],
                     window: int = 0) -> np.ndarray:
    """Single-query attention over decoded cache rows (F32, exact softmax).

    For fwht* modes K rows are rotated-space dequants, so Q is rotated with
    the SAME forward transform (Hadamard unitary => identical dots); V rows
    are natural-space Q8 dequants. `positions` selects+orders the attended
    rows (SWA trailing window or full prefix); causal mask holds by
    construction (single query attends all listed rows). Scale 1/sqrt(256).
    """
    hd = q.shape[-1]
    assert hd == 256
    sc = np.float32(1.0 / np.sqrt(256.0))
    if mode in ("fwht2", "fwht4"):
        qr = np.empty(256, dtype=np.float32)
        for half in range(2):
            qr[half * 128:(half + 1) * 128] = fwht_forward_128(
                q[half * 128:(half + 1) * 128].astype(np.float32), s1, s2)
    elif mode == "fwht3":
        qr = fwht_forward_256(q.astype(np.float32), s1, s2)
    else:
        qr = q.astype(np.float32)
    rows = positions if window <= 0 else positions[-window:]
    K = np.stack([kdec_rows[p] for p in rows]).astype(np.float32)
    V = np.stack([vdec_rows[p] for p in rows]).astype(np.float32)
    scores = (K @ qr.astype(np.float32)) * sc
    scores = scores.astype(np.float32)
    scores -= scores.max()
    p = np.exp(scores).astype(np.float32)
    p /= p.sum()
    return (p.astype(np.float32) @ V.astype(np.float32)).astype(np.float32)

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


KV_ATOL = {
    # STRICT same-codec equivalence (Main contract): CPU and GPU quantize
    # with the SAME codec, so quantization damage vs F32 is COMMON to both
    # sides and does NOT justify GPU-vs-CPU slack. What remains is pure
    # F32 reduction/accumulation-order noise between the CPU einsum/softmax
    # path and the GPU tile+2-pass-reduce path (~1e-4 scale-bound on
    # attention outputs; the 2e-2 full-model logit bar below already covers
    # whole-network accumulation across 36 layers — widen ONLY on a measured
    # specific threshold-boundary flip, never blanket per-mode).
    # Optional diagnostics MAY report loss-vs-F32 separately, but the PASS
    # bar is same-codec agreement. Centroid-half-step reasoning is the WRONG
    # contract for GPU-vs-CPU (both sides already carry the same centroids).
    "q8": {"attn": 2e-3, "logits": 2e-2, "cos": 0.99999},
    "fwht4": {"attn": 2e-3, "logits": 2e-2, "cos": 0.99999},
    "fwht3": {"attn": 2e-3, "logits": 2e-2, "cos": 0.99999},
    "fwht2": {"attn": 2e-3, "logits": 2e-2, "cos": 0.99999},
}


class DoryRef:
    """Multi-loop Dory forward (I -> R^N -> O, shared recurrent block).

    Architecture (upstream vllm dory.py): 8 input / 20 recurrent / 8 output
    transformer BLOCKS; the checkpoint encodes each block as 2 physical
    attention/MLP sublayers: split 16/40/16 = 72 physical sublayers.
    Per token: input block once -> snapshot input_h; state=input_h; each loop
    h=justnorm(state+input_h), shared physical recurrent block, state=justnorm(h);
    output block once; lmhead once. Same absolute token position every loop;
    next_pos advances ONCE after all loops. No weight duplication.

    KV semantics (mirror DoryConfig::kv_layer_index):
    - per_loop: dense virtual cache; input physical i; recurrent i+loop_idx*n_rec;
      output i+(N-1)*n_rec. Layers 72+40*(N-1). Batched/incremental invariant:
      bulk prefill == chunked prefill == tokenwise decode.
    - last_loop: cache index = physical i; each loop overwrites the CURRENT
      absolute position row; previous tokens retain final-loop KV. 72 slots.
      Mode-aware schedule: parity walks T==1 tokenwise steps (native prefill
      IS tokenwise); a bulk call is NOT parity-equivalent (bulk loop-0 token1
      attends token0 loop-0, native tokenwise attends token0 final-loop) and
      stays research-only.
    Set bf16=True for the activation-rounded twin.
    """

    def __init__(self, cfg: dict, reader: STReader, n_loops: int = 1,
                 recurrent_mode: str = "per_loop",
                 bf16: bool = False, kv_mode: str = "f32"):
        if recurrent_mode not in ("per_loop", "last_loop"):
            raise ValueError(
                f"recurrent_mode {recurrent_mode!r} not in ('per_loop','last_loop')")
        if not isinstance(n_loops, int) or not (1 <= n_loops <= 4):
            raise ValueError(f"n_loops={n_loops!r} must be an integer 1..4")
        self.c = cfg
        self.r = reader
        self.n_loops = n_loops
        self.recurrent_mode = recurrent_mode
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
        assert kv_mode in KV_MODES, f"kv_mode {kv_mode} not in {KV_MODES}"
        assert not (bf16 and kv_mode != "f32"), \
            "bf16 twin + lossy KV composes two error sources; run separately"
        self.kv_mode = kv_mode
        # FWHT sign tables mirror the ACTUAL shared-cache constructors:
        # 256-wide (42/1042) for fwht3, 128-wide shared halves for fwht2/4.
        self.kv_s1 = gen_fwht_signs(42, 256)
        self.kv_s2 = gen_fwht_signs(1042, 256)
        # Optional lab KV trace (see --kv-trace): when a list, attn() appends
        # one row per (position, attention layer) with precache q/k/v floats
        # + raw pre-gate attention out. None = disabled. Never affects math.
        self.kv_trace = None

    # -- linear + recurrence helpers -----------------------------------------
    def n_virtual_layers(self) -> int:
        c = self.c
        if self.recurrent_mode == "per_loop":
            return (c["num_hidden_layers"]
                    + c["n_recurrent_layers"] * (self.n_loops - 1))
        return c["num_hidden_layers"]

    def kv_layer_index(self, physical: int, loop_index: int) -> int:
        c = self.c
        n_in = c["n_input_layers"]
        n_rec = c["n_recurrent_layers"]
        if self.recurrent_mode == "last_loop":
            return physical
        if physical < n_in:
            return physical
        if physical < n_in + n_rec:
            return physical + loop_index * n_rec
        return physical + (self.n_loops - 1) * n_rec

    def kv_layer_mask(self) -> List[bool]:
        n_virt = self.n_virtual_layers()
        phys_attn = [(t != "mlp") for t in self.c["layer_types"]]
        n_in = self.c["n_input_layers"]
        n_rec = self.c["n_recurrent_layers"]
        mask = [False] * n_virt
        for vi in range(n_virt):
            if self.recurrent_mode == "last_loop":
                mask[vi] = phys_attn[vi]
                continue
            if vi < n_in:
                mask[vi] = phys_attn[vi]
            elif vi < n_in + self.n_loops * n_rec:
                mask[vi] = phys_attn[n_in + (vi - n_in) % n_rec]
            else:
                mask[vi] = phys_attn[vi - (self.n_loops - 1) * n_rec]
        return mask

    def _new_caches(self) -> List[dict | None]:
        return [None] * self.n_virtual_layers()

    def lin(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        if self.bf16:
            x = bf16_round(x)
            w = bf16_round(w)
        y = x @ w.T
        return bf16_round(y) if self.bf16 else y

    # -- sublayers ----------------------------------------------------------
    def attn(self, h: np.ndarray, pos: np.ndarray, li: int,
             kv: dict | None, loop_index: int = 0, cache_layer: int | None = None) -> Tuple[np.ndarray, dict | None]:
        if cache_layer is None:
            cache_layer = self.kv_layer_index(li, loop_index)
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
        # Precached current-token rows for the lab trace (pre-concat views;
        # decode rebinds k/v below, these keep the current token only).
        k_cur, v_cur = k, v
        was_decode = kv is not None
        if self.kv_mode != "f32":
            return self.attn_kv_quant(h, q, k, v, g, pos, li, kv, was_decode, loop_index, cache_layer)
        # NOTE (research-only bulk path): under last_loop a bulk call with T>1
        # is NOT native-equivalent — bulk loop-0 token1 attends token0 loop-0
        # while native tokenwise attends token0 final-loop. run_full and the
        # parity drivers always walk T==1 steps; this path stays only for
        # offline research, never for parity verdicts.
        if was_decode:
            if self.recurrent_mode == "last_loop":
                # Overwrite (never append): last-loop owns the CURRENT absolute
                # position row; earlier tokens keep final-loop KV. Decode cache
                # holds exactly the walked prefix; a fresh pos == len extends it.
                p0 = int(pos[0])
                assert T == 1 and p0 <= len(kv["k"]), \
                    f"layer {li}: last_loop decode pos {p0} outside cache {len(kv['k'])}"
                if p0 == len(kv["k"]):
                    k = np.concatenate([kv["k"], k], axis=0)
                    v = np.concatenate([kv["v"], v], axis=0)
                else:
                    k = np.concatenate([kv["k"][:p0], k, kv["k"][p0 + 1:]], axis=0)
                    v = np.concatenate([kv["v"][:p0], v, kv["v"][p0 + 1:]], axis=0)
            else:
                # decode: append pre-expanded KV (rope already applied at abs pos)
                k = np.concatenate([kv["k"], k], axis=0)
                v = np.concatenate([kv["v"], v], axis=0)
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
        if self.kv_trace is not None:
            # F32 trace: ONLY current-token rows (post-RoPE, post-qk-norm
            # q/k/v) + raw attention out BEFORE the output gate.
            for t in range(T):
                self.kv_trace.append({
                    "position": int(pos[t]),
                    "layer": int(li),
                    "loop_index": int(loop_index),
                    "cache_layer": int(cache_layer),
                    "q": q[t].reshape(-1).astype(np.float32).tolist(),
                    "k": k_cur[t].reshape(-1).astype(np.float32).tolist(),
                    "v": v_cur[t].reshape(-1).astype(np.float32).tolist(),
                    "attention": o[t].reshape(-1).astype(np.float32).tolist(),
                })
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

    def attn_kv_quant(self, h: np.ndarray, q: np.ndarray, k: np.ndarray,
                      v: np.ndarray, g: np.ndarray, pos: np.ndarray, li: int,
                      kv: dict | None, was_decode: bool,
                      loop_index: int = 0, cache_layer: int | None = None) -> Tuple[np.ndarray, dict | None]:
        if cache_layer is None:
            cache_layer = self.kv_layer_index(li, loop_index)
        # Cache rows hold PACKED blobs (bit-parity with kv_encode_pos): a wrong
        # packed offset/layer index/rotation/norm/Q8 fallback shows as attention
        # error. Mirrors the fused write (K+V at each position, V always Q8).
        c = self.c
        T = q.shape[0]
        mode = self.kv_mode
        hd = c["head_dim"]
        s1, s2 = self.kv_s1, self.kv_s2
        last_loop = self.recurrent_mode == "last_loop"
        if kv is None or "kblob" not in kv:
            kv = {"kblob": [], "vblob": [], "kdec": [], "vdec": []}
        if not was_decode:
            if last_loop:
                # Scheduling caveat (research-only bulk path): bulk loop-0
                # token1 attends token0 loop-0 while native tokenwise attends
                # token0 final-loop, so bulk is NOT parity-equivalent. run_full
                # walks T==1 steps; the exact-position write below also serves
                # contiguous bulk research calls without appending duplicates.
                assert T == int(pos[T - 1]) - int(pos[0]) + 1, \
                    "last_loop prefill requires contiguous tokenwise positions"
                p0 = int(pos[0])
                need = p0 + T
                while len(kv["kdec"]) // max(1, self.nKV) < need:
                    kv["kblob"].extend([b""] * self.nKV)
                    kv["vblob"].extend([b""] * self.nKV)
                    kv["kdec"].extend([np.zeros(hd, dtype=np.float32)] * self.nKV)
                    kv["vdec"].extend([np.zeros(hd, dtype=np.float32)] * self.nKV)
                for t in range(T):
                    for hh in range(self.nKV):
                        kb, vb, kd, vd = kv_encode_pos(
                            k[t, hh].astype(np.float32), v[t, hh].astype(np.float32),
                            mode, s1, s2)
                        idx = (p0 + t) * self.nKV + hh
                        kv["kblob"][idx] = kb
                        kv["vblob"][idx] = vb
                        kv["kdec"][idx] = kd
                        kv["vdec"][idx] = vd
                    assert q8_decode_blob(
                        kv["vblob"][(p0 + t) * self.nKV], hd).max() is not None
                S = len(kv["kdec"]) // self.nKV
                kv["kk"] = [kv["kdec"][s * self.nKV:(s + 1) * self.nKV]
                            for s in range(S)]
                kv["vv"] = [kv["vdec"][s * self.nKV:(s + 1) * self.nKV]
                            for s in range(S)]
            else:
                kv = {"kblob": [], "vblob": [], "kdec": [], "vdec": []}
                for t in range(T):
                    for hh in range(self.nKV):
                        kb, vb, kd, vd = kv_encode_pos(
                            k[t, hh].astype(np.float32), v[t, hh].astype(np.float32),
                            mode, s1, s2)
                        assert q8_decode_blob(vb, hd).max() is not None  # shape guard
                        kv["kblob"].append(kb)
                        kv["vblob"].append(vb)
                        kv["kdec"].append(kd)
                        kv["vdec"].append(vd)
                kv["kk"] = [kv["kdec"][t * self.nKV:(t + 1) * self.nKV] for t in range(T)]
                kv["vv"] = [kv["vdec"][t * self.nKV:(t + 1) * self.nKV] for t in range(T)]
        else:
            assert T == 1, "decode path is single-query"
            if last_loop:
                assert kv and "kdec" in kv and len(kv["kdec"]) > 0, \
                    f"layer {li}: last_loop decode with empty cache"
                p0 = int(pos[0])
                S = len(kv["kdec"]) // self.nKV
                assert 0 <= p0 <= S, \
                    f"layer {li}: last_loop decode pos {p0} outside cache {S}"
                if p0 == S:
                    for hh in range(self.nKV):
                        kb, vb, kd, vd = kv_encode_pos(
                            k[0, hh].astype(np.float32), v[0, hh].astype(np.float32),
                            mode, s1, s2)
                        kv["kblob"].append(kb)
                        kv["vblob"].append(vb)
                        kv["kdec"].append(kd)
                        kv["vdec"].append(vd)
                    S += 1
                else:
                    for hh in range(self.nKV):
                        kb, vb, kd, vd = kv_encode_pos(
                            k[0, hh].astype(np.float32), v[0, hh].astype(np.float32),
                            mode, s1, s2)
                        idx = p0 * self.nKV + hh
                        kv["kblob"][idx] = kb
                        kv["vblob"][idx] = vb
                        kv["kdec"][idx] = kd
                        kv["vdec"][idx] = vd
                kv["kk"] = [kv["kdec"][s * self.nKV:(s + 1) * self.nKV]
                            for s in range(S)]
                kv["vv"] = [kv["vdec"][s * self.nKV:(s + 1) * self.nKV]
                            for s in range(S)]
            else:
                for hh in range(self.nKV):
                    kb, vb, kd, vd = kv_encode_pos(
                        k[0, hh].astype(np.float32), v[0, hh].astype(np.float32),
                        mode, s1, s2)
                    kv["kblob"].append(kb)
                    kv["vblob"].append(vb)
                    kv["kdec"].append(kd)
                    kv["vdec"].append(vd)
                S = len(kv["kdec"]) // self.nKV
                kv["kk"] = [kv["kdec"][t * self.nKV:(t + 1) * self.nKV]
                            for t in range(S)]
                kv["vv"] = [kv["vdec"][t * self.nKV:(t + 1) * self.nKV]
                            for t in range(S)]
        o_heads = np.empty((T, self.nH, hd), dtype=np.float32)
        S = len(kv["kk"])
        # ATTEND-PATH SELF-CHECK (independent decode): re-decode the stored
        # packed blob of the LAST position/head-0 against the encoder's
        # dequant. FWHT's scalar and vectorized cnorm products can differ
        # by an F32 ulp; this local bound does not relax native parity.
        last = len(kv["kblob"]) - self.nKV
        if mode == "q8":
            chk = q8_decode_blob(kv["kblob"][last], hd)
        elif mode != "f32":
            chk = fwht_decode_k_rotated(kv["kblob"][last], mode)
        else:
            chk = None
        if chk is not None:
            dd = float(np.abs(chk.astype(np.float64)
                              - kv["kdec"][last].astype(np.float64)).max())
            redecode_tol = 0.0 if mode == "q8" else 3e-7
            assert dd <= redecode_tol, f"layer {li}: packed blob re-decode mismatch {dd}"
        is_sliding = self.c["layer_types"][li] == "sliding_attention"
        win = self.c["sliding_window"] if is_sliding else 0
        for t in range(T):
            absp = int(pos[t])
            if was_decode:
                if last_loop:
                    # Overwritten in place: cache covers the walked prefix, the
                    # query row is the current absolute position.
                    assert 0 <= absp < S, f"layer {li}: abs pos {absp} outside cache {S}"
                    rows = list(range(0 if win <= 0 else max(0, absp - win + 1), absp + 1))
                else:
                    assert absp == S - 1, f"layer {li}: abs pos {absp} != cache end {S - 1}"
                    rows = list(range(0 if win <= 0 else max(0, S - win), S))
            else:
                # prefill: keys are exactly the query positions; replicate the
                # F32 path's window mask (tk > tq | tq - tk >= win -> -inf).
                rows = None
            for hh in range(self.nH):
                kh = hh // self.rep
                if rows is None:
                    # prefill full-window softmax over this query's allowed keys
                    krows = [kv["kk"][s][kh] for s in range(T)]
                    vrows = [kv["vv"][s][kh] for s in range(T)]
                    allowed = []
                    for s in range(T):
                        ok = True
                        if is_sliding:
                            tq, tk = int(pos[t]), int(pos[s])
                            ok = not ((tk > tq) or (tq - tk >= win))
                        else:
                            ok = s <= t
                        allowed.append(ok)
                    krows = [r for r, a in zip(krows, allowed) if a]
                    vrows = [r for r, a in zip(vrows, allowed) if a]
                    o_heads[t, hh] = kv_attention_ref(
                        q[t, hh].astype(np.float32), krows, vrows, mode,
                        s1, s2, list(range(len(krows))), 0)
                else:
                    krows = [kv["kk"][s][kh] for s in rows]
                    vrows = [kv["vv"][s][kh] for s in rows]
                    o_heads[t, hh] = kv_attention_ref(
                        q[t, hh].astype(np.float32), krows, vrows, mode,
                        s1, s2, list(range(len(krows))), 0)
        o = o_heads
        if self.kv_trace is not None:
            # Quant trace: method inputs are already the per-token precache
            # q/k/v rows; o_heads is raw BEFORE gate gain/justnorm/o_proj.
            for t in range(T):
                self.kv_trace.append({
                    "position": int(pos[t]),
                    "layer": int(li),
                    "loop_index": int(loop_index),
                    "cache_layer": int(cache_layer),
                    "q": q[t].reshape(-1).astype(np.float32).tolist(),
                    "k": k[t].reshape(-1).astype(np.float32).tolist(),
                    "v": v[t].reshape(-1).astype(np.float32).tolist(),
                    "attention": o_heads[t].reshape(-1).astype(np.float32).tolist(),
                })
        sg = (self.r.bf16_f32(f"backbone.layers.{li}.mixer.s_gate").astype(np.float32)
              * np.float32(self.sgate_factor))
        sg = (sg * sg).reshape(1, self.nH, hd)
        z = (g.reshape(T, self.nH, c["head_dim"]).astype(np.float32) * sg)
        gate = np.empty_like(z)
        neg = z < 0
        gate[neg] = np.exp(z[neg]) / (1.0 + np.exp(z[neg]))
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
              kv: dict | None, loop_index: int = 0,
              cache_layer: int | None = None):
        if cache_layer is None:
            cache_layer = self.kv_layer_index(li, loop_index)
        if self.is_attn[li]:
            out, kv = self.attn(h, pos, li, kv, loop_index, cache_layer)
            a = self.r.bf16_f32(f"backbone.layers.{li}.attn_alpha")
            return slerp_alpha(h, out, a, self.alpha_scale), kv
        out = self.mlp(h, li)
        a = self.r.bf16_f32(f"backbone.layers.{li}.mlp_alpha")
        return slerp_alpha(h, out, a, self.alpha_scale), kv

    # -- backbone: I -> R^N -> O (shared recurrent block, reinjected input) --
    def backbone(self, emb: np.ndarray, pos: np.ndarray,
                 caches: List[dict | None] | None = None):
        c = self.c
        n_in, n_rec = c["n_input_layers"], c["n_recurrent_layers"]
        h = emb.astype(np.float32, copy=False)
        if caches is None:
            caches = self._new_caches()
        assert len(caches) == self.n_virtual_layers(), \
            f"caches {len(caches)} != virtual layers {self.n_virtual_layers()}"
        # Input block once; snapshot input_h; state starts AT input_h.
        for li in range(n_in):
            vi = self.kv_layer_index(li, 0)
            h, caches[vi] = self.layer(h, pos, li, caches[vi], 0, vi)
        input_h = h
        state = input_h
        for loop in range(self.n_loops):
            h = justnorm(state + input_h)
            for li in range(n_in, n_in + n_rec):
                vi = self.kv_layer_index(li, loop)
                h, caches[vi] = self.layer(h, pos, li, caches[vi], loop, vi)
            state = justnorm(h)
        # Output block once from the post-recurrent state.
        h = state
        for li in range(n_in + n_rec, c["num_hidden_layers"]):
            vi = self.kv_layer_index(li, self.n_loops - 1)
            h, caches[vi] = self.layer(h, pos, li, caches[vi], 0, vi)
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
# Lab KV trace (diagnostic only; never affects the oracle math above)
# ----------------------------------------------------------------------------
#
# Schema dory_kv_trace_v2 (agreed with the native driver; both sides join on
# position+layer+loop_index; KV histories key by cache_layer):
#   {"schema":"dory_kv_trace_v2","kv_mode":EFFECTIVE,"loops":N,
#    "recurrent_kv_cache_mode":MODE,"n_heads":..,
#    "n_kv_heads":..,"head_dim":..,"sliding_window":..,
#    "is_full_attention":[bool x num_layers],"input_source":{...},"rows":[
#    {"position":..,"layer":..,"loop_index":..,"cache_layer":..,
#     "q":[H*D],"k":[Hkv*D],"v":[Hkv*D],"attention":[H*D]}]}
# Input/output rows use loop_index 0; recurrent rows use their loop pass.
# q/k/v are PRE-cache floats (post-RoPE/norm, pre-quantization); attention is
# RAW pre-gate output. Native pre-KV inputs are channel STIMULI: the CPU-side
# expected attention (kv_encode_pos + kv_attention_ref on THOSE exact rows)
# stays independent of this oracle's own forward — a separate whole-model CPU
# run (--run without --kv-trace) remains the model-level oracle.

TRACE_SCHEMA = "dory_kv_trace_v2"


def dump_kv_trace(path: str, cfg: dict, kv_mode: str, tokens: List[int],
                  model_label: str, weight_fp: str,
                  rows: List[dict], loops: int = 1,
                  recurrent_mode: str = "per_loop") -> None:
    """Write the lab KV trace doc, sorted by (position, layer, loop_index)."""
    rows_sorted = sorted(rows, key=lambda r: (r["position"], r["layer"],
                                              r.get("loop_index", 0)))
    doc = {
        "schema": TRACE_SCHEMA,
        "kv_mode": kv_mode,
        "loops": loops,
        "recurrent_kv_cache_mode": recurrent_mode,
        "n_heads": cfg["num_attention_heads"],
        "n_kv_heads": cfg["num_key_value_heads"],
        "head_dim": cfg["head_dim"],
        "sliding_window": cfg["sliding_window"],
        "is_full_attention": [t == "full_attention"
                              for t in cfg["layer_types"]],
        "input_source": {
            "oracle": "cpu",
            "model": model_label,
            "weight_fingerprint": weight_fp,
            "kv_mode": kv_mode,
            "loops": loops,
            "recurrent_kv_cache_mode": recurrent_mode,
            "tokens": list(tokens),
        },
        "rows": rows_sorted,
    }
    with open(path, "w") as f:
        json.dump(doc, f)
    print(f"kv-trace: wrote {path} "
          f"(schema={TRACE_SCHEMA} kv_mode={kv_mode} loops={loops} "
          f"mode={recurrent_mode} rows={len(rows_sorted)})",
          flush=True)

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
    # --- KV-quant fixtures (independent CPU paths, D=256 pins) ---
    # Deterministic K/V rows; sign tables mirror production seeds. Checks:
    # (a) packed blob re-decode is exact (encode-path byte layout);
    # (b) Q-rotation identity: (Qr.Kr) == (Q.K) in F32 (unitary check);
    # (c) single-query attention over decoded rows matches the F32
    #     direct-softmax reference within the per-mode KV_ATOL budget;
    # (d) sliding-window trailing selection matches the masked F32 path;
    # (e) reset/overwrite: re-encoding position p with different values
    #     fully replaces the row (no stale-tail mixing).
    krng = np.random.default_rng(777)
    HD = 256
    KQ = (krng.standard_normal((9, HD)).astype(np.float32)
          / np.float32(np.sqrt(float(HD))))
    VQ = (krng.standard_normal((9, HD)).astype(np.float32)
          / np.float32(np.sqrt(float(HD))))
    QQ = (krng.standard_normal((HD,)).astype(np.float32)
          / np.float32(np.sqrt(float(HD))))
    S1 = gen_fwht_signs(42, 256)
    S2 = gen_fwht_signs(1042, 256)
    # (a)+(c): per-mode encode -> rotated-dot attention vs F32 direct
    for m in ("q8", "fwht2", "fwht3", "fwht4"):
        kdec, vdec = [], []
        blobs = []
        for p in range(9):
            kb, vb, kd, vd = kv_encode_pos(KQ[p], VQ[p], m, S1, S2)
            blobs.append((kb, vb))
            kdec.append(kd)
            vdec.append(vd)
        # (a) independent re-decode path must reproduce the encoder dequant:
        # f32/q8 exact; fwht paths multiply by cnorm (F32 L2 ratio) so the
        # bar is one F32 ulp (~3e-7), NOT zero.
        worst = 0.0
        rdtol = 0.0 if m == "q8" else 3e-7
        for p in range(9):
            kb, vb = blobs[p]
            rk = (q8_decode_blob(kb, HD) if m == "q8"
                  else fwht_decode_k_rotated(kb, m))
            rv = q8_decode_blob(vb, HD)
            worst = max(worst, float(np.abs(
                rk.astype(np.float64) - kdec[p].astype(np.float64)).max()),
                float(np.abs(
                    rv.astype(np.float64) - vdec[p].astype(np.float64)).max()))
        print(f"selftest kv-{m} re-decode maxabs = {worst:.3e} (tol {rdtol:.1e})")
        ok = ok and worst <= rdtol
        # (c) lossy attention vs F32 direct softmax over the SAME rows
        got = kv_attention_ref(QQ, kdec, vdec, m, S1, S2, list(range(9)), 0)
        sc = np.float32(1.0 / np.sqrt(256.0))
        ref_sc = (KQ @ QQ.astype(np.float32)) * sc
        ref_sc -= ref_sc.max()
        ref_p = np.exp(ref_sc).astype(np.float32)
        ref_p /= ref_p.sum()
        want = (ref_p.astype(np.float32) @ VQ.astype(np.float32)).astype(np.float32)
        da_kv = float(np.abs(got.astype(np.float64) - want.astype(np.float64)).max())
        # DIAGNOSTIC ONLY: quant loss vs F32 must NOT gate the verdict (both
        # CPU and GPU carry the same codec; the PASS bar is re-decode exact
        # + rot-identity above and same-codec GPU agreement in the probe).
        print(f"selftest kv-{m} attn-vs-f32 maxabs = {da_kv:.3e} (diagnostic, no tol)")
        # (b) rotation identity on exact F32 (no quant): rotated dots == plain
        if m != "q8":
            if m == "fwht3":
                qr = fwht_forward_256(QQ, S1, S2)
                krot = [fwht_forward_256(r, S1, S2) for r in KQ]
            else:
                qr = np.concatenate([
                    fwht_forward_128(QQ[h:h + 128], S1, S2) for h in (0, 128)])
                krot = [np.concatenate([
                    fwht_forward_128(r[h:h + 128], S1, S2) for h in (0, 128)])
                    for r in KQ]
            d1 = float(QQ.astype(np.float32)
                       @ KQ[3].astype(np.float32))
            d2 = float(qr.astype(np.float32)
                       @ krot[3].astype(np.float32))
            di = abs(d1 - d2)
            print(f"selftest kv-{m} rot-identity |Qr.Kr - Q.K| = {di:.3e} (tol 1e-4)")
            ok = ok and di < 1e-4
    # (d) sliding-window trailing selection == masked F32 reference
    for m in ("q8", "fwht2", "fwht3", "fwht4"):
        kdec = [kv_encode_pos(KQ[p], VQ[p], m, S1, S2)[2] for p in range(9)]
        vdec = [kv_encode_pos(KQ[p], VQ[p], m, S1, S2)[3] for p in range(9)]
        win = 4
        got = kv_attention_ref(QQ, kdec, vdec, m, S1, S2, list(range(9)), win)
        want = kv_attention_ref(QQ, kdec[-win:], vdec[-win:], m, S1, S2,
                                list(range(win)), 0)
        dw = float(np.abs(got.astype(np.float64) - want.astype(np.float64)).max())
        print(f"selftest kv-{m} swa-trailing maxabs = {dw:.3e} (tol 0)")
        ok = ok and dw == 0.0
    # (e) reset/overwrite: position re-encoded with new values replaces fully
    for m in ("q8", "fwht2", "fwht3", "fwht4"):
        kdec = [kv_encode_pos(KQ[p], VQ[p], m, S1, S2)[2] for p in range(5)]
        vdec = [kv_encode_pos(KQ[p], VQ[p], m, S1, S2)[3] for p in range(5)]
        knew = (krng.standard_normal((HD,)).astype(np.float32) * np.float32(3.0))
        vnew = (krng.standard_normal((HD,)).astype(np.float32) * np.float32(3.0))
        _, _, kd2, vd2 = kv_encode_pos(knew, vnew, m, S1, S2)
        kdec[2], vdec[2] = kd2, vd2
        got = kv_attention_ref(QQ, kdec, vdec, m, S1, S2, list(range(5)), 0)
        # same query against a cache built directly with the new row
        kdec2 = [kv_encode_pos(KQ[p], VQ[p], m, S1, S2)[2] for p in range(5)]
        vdec2 = [kv_encode_pos(KQ[p], VQ[p], m, S1, S2)[3] for p in range(5)]
        kdec2[2], vdec2[2] = kd2, vd2
        want = kv_attention_ref(QQ, kdec2, vdec2, m, S1, S2, list(range(5)), 0)
        dr = float(np.abs(got.astype(np.float64) - want.astype(np.float64)).max())
        print(f"selftest kv-{m} overwrite maxabs = {dr:.3e} (tol 0)")
        ok = ok and dr == 0.0
    # --- recurrence fixtures (deterministic miniature I/R/O stack, F32) ---
    # Tiny stack: n_in=2 (attn,mlp), n_rec=2 (attn,mlp), n_out=2 (attn,mlp);
    # 6 sublayers, 3 attention layers, H=32. Encodes block counts (mimics the
    # real 8/20/8 blocks -> 16/40/16 sublayers) with real config plumbing.
    class _FakeReader:
        def __init__(self, H: int, F: int, nH: int, nKV: int, hd: int):
            self.H, self.F = H, F
            self.nH, self.nKV, self.hd = nH, nKV, hd
            rr = np.random.default_rng(999)
            def w(r: int, c: int) -> np.ndarray:
                return (rr.standard_normal((r, c)) * np.float32(std)).astype(np.float32)
            def v(n: int) -> np.ndarray:
                return (rr.standard_normal(n) * np.float32(std)).astype(np.float32)
            self.t = {}
            for li in range(6):
                if li % 2 == 0:
                    self.t[f"backbone.layers.{li}.mixer.q_proj.weight"] = w(nH * hd, H)
                    self.t[f"backbone.layers.{li}.mixer.k_proj.weight"] = w(nKV * hd, H)
                    self.t[f"backbone.layers.{li}.mixer.v_proj.weight"] = w(nKV * hd, H)
                    self.t[f"backbone.layers.{li}.mixer.g_proj.weight"] = w(nH * hd, H)
                    self.t[f"backbone.layers.{li}.mixer.o_proj.weight"] = w(H, nH * hd)
                    self.t[f"backbone.layers.{li}.mixer.sqk"] = v(nH * hd)
                    self.t[f"backbone.layers.{li}.mixer.s_gate"] = v(nH * hd)
                    self.t[f"backbone.layers.{li}.attn_alpha"] = v(H)
                else:
                    self.t[f"backbone.layers.{li}.mixer.gate_proj.weight"] = w(F, H)
                    self.t[f"backbone.layers.{li}.mixer.up_proj.weight"] = w(F, H)
                    self.t[f"backbone.layers.{li}.mixer.down_proj.weight"] = w(H, F)
                    self.t[f"backbone.layers.{li}.mixer.suv_gate"] = (
                        rr.standard_normal(F) + 2.0).astype(np.float32)
                    self.t[f"backbone.layers.{li}.mixer.suv_up"] = (
                        rr.standard_normal(F) + 2.0).astype(np.float32)
                    self.t[f"backbone.layers.{li}.mlp_alpha"] = v(H)
            self.t["backbone.embeddings.weight"] = w(64, H)
            self.t["lm_head.weight"] = w(64, H)
            self.t["lm_head.logit_scale"] = np.ones(64, dtype=np.float32)

        def bf16_f32(self, name: str) -> np.ndarray:
            return self.t[name].astype(np.float32)

        def row_gather_f32(self, name: str, ids: np.ndarray) -> np.ndarray:
            return self.t[name][np.asarray(ids)].astype(np.float32)

    tcfg = {
        "hidden_size": H, "num_hidden_layers": 6, "vocab_size": 64,
        "intermediate_size": F, "num_attention_heads": nH,
        "num_key_value_heads": nKV, "head_dim": hd, "sliding_window": 1024,
        "ngpt_alpha_init": a_init, "initializer_range": std,
        "attention_output_gate_logit_scale_init": None,
        "layer_types": ["sliding_attention", "mlp"] * 3,
        "n_input_layers": 2, "n_recurrent_layers": 2, "n_output_layers": 2,
        "n_recurrent_loops": 2,
    }
    fr = _FakeReader(H, F, nH, nKV, hd)
    rng_t = np.random.default_rng(20261019)
    emb_t = rng_t.standard_normal((3, H)).astype(np.float32)
    pos_t = np.arange(3)
    # (i) invalid counts/modes fail closed before any math.
    for bad in (0, 5, -1, "2", 2.0):
        try:
            DoryRef(tcfg, fr, n_loops=bad, recurrent_mode="per_loop")
            print(f"selftest recur invalid-loops {bad!r}: NO ERROR (FAIL)")
            ok = False
        except ValueError:
            pass
    try:
        DoryRef(tcfg, fr, n_loops=2, recurrent_mode="bogus")
        print("selftest recur invalid-mode: NO ERROR (FAIL)")
        ok = False
    except ValueError:
        pass
    print("selftest recur invalid counts/modes fail closed (tol exact)")
    # (ii) indexing: per_loop dense virtual layers 6+2*(N-1), last_loop flat.
    # Loops 1..4 including 3 (explicit user requirement).
    for N in (1, 2, 3, 4):
        rp = DoryRef(tcfg, fr, n_loops=N, recurrent_mode="per_loop")
        assert rp.n_virtual_layers() == 6 + 2 * (N - 1), N
        assert sum(rp.kv_layer_mask()) == 3 + 1 * (N - 1), N
        # input layer identity, recurrent stride, output shift.
        assert rp.kv_layer_index(0, 0) == 0, N
        assert rp.kv_layer_index(2, N - 1) == 2 + (N - 1) * 2, N
        assert rp.kv_layer_index(5, 0) == 5 + (N - 1) * 2, N
        rl = DoryRef(tcfg, fr, n_loops=N, recurrent_mode="last_loop")
        assert rl.n_virtual_layers() == 6, N
        assert all(v == p for p, v in
                   enumerate(rl.kv_layer_index(p, 0) for p in range(6))), N
    print("selftest recur indexing per_loop/last_loop exact (tol exact)")
    # (iii) per_loop: chunked prefill == full prefill AND == tokenwise decode
    # (batched/incremental invariance); N=1..4 including 3.
    for N in (1, 2, 3, 4):
        rp = DoryRef(tcfg, fr, n_loops=N, recurrent_mode="per_loop")
        full, _ = rp.backbone(emb_t, pos_t)
        pre, c1 = rp.backbone(emb_t[:2], pos_t[:2])
        tail, _ = rp.backbone(emb_t[2:], pos_t[2:], c1)
        d_chunk = float(np.abs(full[2:].astype(np.float64)
                               - tail.astype(np.float64)).max())
        print(f"selftest recur-N{N} per_loop prefill-chunk maxabs = {d_chunk:.3e} (tol 0)")
        ok = ok and d_chunk == 0.0
        _, cd = rp.backbone(emb_t[:1], pos_t[:1])
        rows = []
        for j in range(1, 3):
            h, cd = rp.backbone(emb_t[j:j + 1], pos_t[j:j + 1], cd)
            rows.append(h[0])
        d_tok = float(np.abs(full[1:].astype(np.float64)
                             - np.stack(rows).astype(np.float64)).max())
        print(f"selftest recur-N{N} per_loop tokenwise maxabs = {d_tok:.3e} (tol 0)")
        ok = ok and d_tok == 0.0
    # (iv) last_loop: tokenwise prefill == tokenwise decode; last-pass history
    # retained and CURRENT-row overwrite on the walked prefix. N=1..4 incl 3.
    for N in (1, 2, 3, 4):
        rl = DoryRef(tcfg, fr, n_loops=N, recurrent_mode="last_loop")
        pre_cache = None
        rows = []
        for j in range(3):
            h, pre_cache = rl.backbone(emb_t[j:j + 1], pos_t[j:j + 1], pre_cache)
            rows.append(h[0])
        tokwise = np.stack(rows)
        _, dec_cache = rl.backbone(emb_t[:1], pos_t[:1])
        drows = []
        for j in range(1, 3):
            h, dec_cache = rl.backbone(emb_t[j:j + 1], pos_t[j:j + 1], dec_cache)
            drows.append(h[0])
        dwalk = float(np.abs(tokwise[1:].astype(np.float64)
                             - np.stack(drows).astype(np.float64)).max())
        print(f"selftest recur-N{N} last_loop tokenwise maxabs = {dwalk:.3e} (tol 1e-5)")
        ok = ok and dwalk < 1e-5
        # Overwrite proof: re-walk position 1 with a DIFFERENT embedding; the
        # walked prefix must change at 1 (current row overwritten) and keep
        # history at 0. Compare full-row outputs for determinism.
        alt = (emb_t[1:2] * np.float32(3.0) + np.float32(1.0))
        hA, _ = rl.backbone(alt, pos_t[1:2], pre_cache)
        hB, _ = rl.backbone(emb_t[1:2], pos_t[1:2], pre_cache)
        d_ow = float(np.abs(hA.astype(np.float64) - hB.astype(np.float64)).max())
        print(f"selftest recur-N{N} last_loop current-row-overwrite diff = {d_ow:.3e} (expect >0)")
        ok = ok and d_ow > 0.0
        # History independence: earlier positions keep FINAL-pass KV while the
        # current row is overwritten per pass. Decode position 2 from the
        # walked prefix must equal a fresh tokenwise walk of 0..2 (same N,
        # same schedule) — stale earlier-pass rows would diverge.
        fresh = None
        frows = []
        for j in range(3):
            h, fresh = rl.backbone(emb_t[j:j + 1], pos_t[j:j + 1], fresh)
            frows.append(h[0])
        d_hist = float(np.abs(np.stack(frows).astype(np.float64)
                              - tokwise.astype(np.float64)).max())
        print(f"selftest recur-N{N} last_loop history-independence maxabs = {d_hist:.3e} (tol 1e-5)")
        ok = ok and d_hist < 1e-5
        # Reset: a fresh cache (None) over the same inputs reproduces the
        # walked rows exactly — no positional state leaks across resets.
        rst = None
        rrows = []
        for j in range(3):
            h, rst = rl.backbone(emb_t[j:j + 1], pos_t[j:j + 1], rst)
            rrows.append(h[0])
        d_rst = float(np.abs(np.stack(rrows).astype(np.float64)
                             - tokwise.astype(np.float64)).max())
        print(f"selftest recur-N{N} last_loop reset maxabs = {d_rst:.3e} (tol 0)")
        ok = ok and d_rst == 0.0
    # (v) per_loop history independence + reset: per-loop slots never alias,
    # so a fresh walk reproduces chunked rows exactly for N=1..4 incl 3.
    for N in (1, 2, 3, 4):
        rp = DoryRef(tcfg, fr, n_loops=N, recurrent_mode="per_loop")
        w1 = None
        a = []
        for j in range(3):
            h, w1 = rp.backbone(emb_t[j:j + 1], pos_t[j:j + 1], w1)
            a.append(h[0])
        A = np.stack(a)
        w2 = None
        b = []
        for j in range(3):
            h, w2 = rp.backbone(emb_t[j:j + 1], pos_t[j:j + 1], w2)
            b.append(h[0])
        d_rst = float(np.abs(A.astype(np.float64) - np.stack(b).astype(np.float64)).max())
        print(f"selftest recur-N{N} per_loop reset maxabs = {d_rst:.3e} (tol 0)")
        ok = ok and d_rst == 0.0
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
    loops = getattr(args, "loops", None)
    if loops is None:
        loops = int(cfg.get("n_recurrent_loops", 2))
    if not isinstance(loops, int) or not (1 <= loops <= 4):
        raise SystemExit(f"--loops {loops!r} must be an integer 1..4")
    rmode = getattr(args, "recurrent_kv_cache_mode", None) or cfg.get(
        "recurrent_kv_cache_mode", "per_loop")
    if rmode not in ("per_loop", "last_loop"):
        raise SystemExit(
            f"--recurrent-kv-cache-mode {rmode!r} must be per_loop|last_loop; "
            "checkpoint default is per_loop")
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
    print(f"loops={loops} recurrent_kv_cache_mode={rmode} "
          f"(checkpoint n_recurrent_loops={cfg.get('n_recurrent_loops')})", flush=True)
    ref = DoryRef(cfg, reader, n_loops=loops, recurrent_mode=rmode,
                  bf16=False, kv_mode=args.kv_mode)
    if args.kv_trace:
        # Lab diagnostic only: capture precache q/k/v + raw attention rows.
        # No math change (attn methods only append when this is not None).
        ref.kv_trace = []

    # Schedule by policy (native prefill is tokenwise):
    # - per_loop is batched/incremental invariant: bulk prefill S[:P] +
    #   tokenwise decode of the tail; reference is a bulk prefill over S.
    # - last_loop is tokenwise-only (NOT bulk-prefill-equivalent): a bulk
    #   loop-0 token1 would attend token0's loop-0 row, while native tokenwise
    #   scheduling attends token0's final-loop row. Every recurrent pass
    #   overwrites the CURRENT absolute-position row of the shared physical
    #   slot, so the CPU reference walks tokenwise (prefix + tail + a fresh
    #   full walk) to match native scheduling.
    import time
    t0 = time.time()
    if rmode == "per_loop":
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
        if args.kv_trace:
            # Lab diagnostic: detach the measured prefill+decode rows BEFORE
            # the self-consistency re-prefill below (must not pollute trace).
            traced = ref.kv_trace or []
            ref.kv_trace = None
            dump_kv_trace(args.kv_trace, cfg, args.kv_mode, S, model_label,
                          reader.header_fingerprint(), traced, loops, rmode)
        # self-consistency: decode rows must reproduce a full-prefill
        # reference (full prefill over S reuses no cache; tail positions).
        emb_full = reader.row_gather_f32("backbone.embeddings.weight", np.array(S))
        h_full, _ = ref.backbone(emb_full, np.arange(len(S)))
        log_full = ref.logits(h_full)
    else:
        # last_loop tokenwise prefill of S[:P] (rows flow into caches;
        # the fresh full walk below is the emitted reference)
        caches = None
        for i in range(P):
            pos = np.array([i])
            e = reader.row_gather_f32("backbone.embeddings.weight",
                                      np.array([S[i]]))
            h, caches = ref.backbone(e, pos, caches)
        t1 = time.time()
        print(f"prefill {P} tok (tokenwise): {t1 - t0:.1f}s", flush=True)
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
        if args.kv_trace:
            # Lab diagnostic: detach the measured tokenwise rows BEFORE the
            # fresh reference walk below (must not pollute the trace).
            traced = ref.kv_trace or []
            ref.kv_trace = None
            dump_kv_trace(args.kv_trace, cfg, args.kv_mode, S, model_label,
                          reader.header_fingerprint(), traced, loops, rmode)
        # self-consistency: a FRESH full tokenwise walk over S must reproduce
        # the walked prefix+tail rows (same scheduling both sides).
        h_full_rows, log_full_rows = [], []
        caches_full = None
        for i in range(len(S)):
            pos = np.array([i])
            e = reader.row_gather_f32("backbone.embeddings.weight",
                                      np.array([S[i]]))
            h, caches_full = ref.backbone(e, pos, caches_full)
            h_full_rows.append(h[0])
            log_full_rows.append(ref.logits(h)[0])
        h_full = np.stack(h_full_rows).astype(np.float32)
        log_full = np.stack(log_full_rows).astype(np.float32)
    # Verdict over the measured tail (decode rows) vs same-schedule reference.
    dh = np.abs(h_full[P:].astype(np.float64) - h_dec.astype(np.float64)).max()
    dl = np.abs(log_full[P:].astype(np.float64) - log_dec.astype(np.float64)).max()
    cosmin = min(cosine(h_full[P + j], h_dec[j]) for j in range(len(D)))
    print(f"self-consistency decode-vs-prefill ({rmode} schedule): "
          f"max|dhid|={dh:.3e} max|dlogit|={dl:.3e} mincos={cosmin:.8f}")

    last_top = topk(log_full[-1], 8)
    print("last-position top8:", last_top,
          [round(float(log_full[-1][i]), 3) for i in last_top])

    out = args.out
    os.makedirs(out, exist_ok=True)
    km = args.kv_mode
    tag = f"loops{loops}_{rmode}_{km}"
    np.save(os.path.join(out, f"logits_{tag}.npy"), log_full.astype(np.float32))
    np.save(os.path.join(out, f"hidden_{tag}.npy"), h_full.astype(np.float32))
    man_name = f"dory_loops{loops}_{rmode}_{km}.json"
    manifest = {
        "source_math": SRC_ATTRIBUTION,
        "model_dir": model_label,
        "model_kind": kind,
        "kv_mode": km,
        "loops": loops,
        "recurrent_kv_cache_mode": rmode,
        "kv_signs": {"seeds": [42, 1042],
                     "fwht3_width": 256, "fwht2_fwht4_width": 128,
                     "boundary_layers": 0, "layer_is_boundary": [],
                     "v_mode": "q8"},
        "config_snapshot": {k: cfg[k] for k in (
            "vocab_size", "hidden_size", "num_hidden_layers", "intermediate_size",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "n_input_layers", "n_recurrent_layers", "n_output_layers",
            "n_recurrent_loops", "sliding_window", "ngpt_alpha_init",
            "initializer_range", "attention_output_gate_logit_scale_init",
            "tie_word_embeddings", "layer_types")},
        "loops_executed": loops,
        "precision": (("F32 accumulate, Q8F16/F32 dequant weights, "
                       "no activation rounding"
                       if kind == "hfq" else
                       "F32 accumulate, exact BF16->F32 weights, no activation rounding")
                      + (f", KV {km} (K packed, V Q8)" if km != "f32" else ", KV F32")),
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
        "files": {"logits": f"logits_{tag}.npy", "hidden": f"hidden_{tag}.npy"},
    }
    with open(os.path.join(out, man_name), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {out}/{man_name} + logits_{tag}.npy + hidden_{tag}.npy")
    if km == "f32":
        ok = dh < 1e-4 and dl < 5e-2 and cosmin > 0.99999
    else:
        # Same-codec full model: existing 2e-2 logit bar until a measured
        # specific threshold-boundary flip justifies an exception (no blanket
        # widen; loss-vs-F32 is diagnostic-only, reported alongside).
        tol = KV_ATOL[km]
        ok = dh < 5e-2 and dl < 2e-2 and cosmin > tol["cos"]
        print(f"KV-aware self-consistency ({km}): max|dhid|={dh:.3e} "
              f"max|dlogit|={dl:.3e} mincos={cosmin:.8f} "
              f"(tol dhid<5e-2 dlogit<2e-2 cos>{tol['cos']})")
    print("FULL-RUN SELF-CONSISTENCY " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def run_bf16twin(args) -> int:
    """Measure the BF16-activation envelope over all artifact positions.

    Resolves the SAME loop count / cache policy as the F32 artifact being
    twinned (flags win, else the artifact's own manifest) and runs the same
    schedule the twin's F32 artifact used: bulk prefill over the full token
    sequence for per_loop, a fresh full tokenwise walk for last_loop. Native
    F32 parity reference tags the EFFECTIVE resolved mode (never "auto").
    """
    art = args.artifact
    loops_req = getattr(args, "loops", None)
    rmode_req = getattr(args, "recurrent_kv_cache_mode", None)
    man = None
    if loops_req is not None or rmode_req is not None:
        if not isinstance(loops_req if loops_req is not None else 2, int):
            raise SystemExit(f"--loops {loops_req!r} must be an integer 1..4")
        loops_probe = loops_req if loops_req is not None else None
        rmode_probe = rmode_req if rmode_req is not None else None
        # Resolve the artifact identity from explicit flags (exact match, no
        # guessing); flags omitted below fall back to manifest discovery.
        for f in sorted(os.listdir(art)):
            if not (f.startswith("dory_loops") and f.endswith("_f32.json")):
                continue
            try:
                m = json.load(open(os.path.join(art, f)))
            except OSError:
                continue
            if loops_probe is not None and int(m.get("loops", -1)) != loops_probe:
                continue
            if rmode_probe is not None and m.get("recurrent_kv_cache_mode") != rmode_probe:
                continue
            if man is None or f < man[0]:
                man = (f, m)
        if man is None:
            raise SystemExit("--bf16twin: no dory_loops*_f32.json manifest in "
                             f"{art} matches loops={loops_req} mode={rmode_req}")
        man_name = man[0]
        man = man[1]
    else:
        cands = sorted(f for f in os.listdir(art)
                       if f.startswith("dory_loops") and f.endswith("_f32.json"))
        if not cands:
            raise SystemExit(f"--bf16twin: no dory_loops<N>_<mode>_f32.json in {art}; "
                             "run --run --kv-mode f32 with --loops/--recurrent-kv-cache-mode first")
        if len(cands) > 1:
            raise SystemExit(f"--bf16twin: ambiguous {len(cands)} F32 manifests "
                             f"{cands}; pass --loops/--recurrent-kv-cache-mode")
        man_name = cands[0]
        man = json.load(open(os.path.join(art, man_name)))
    loops = int(man["loops"])
    rmode = man["recurrent_kv_cache_mode"]
    if loops_req is not None and loops_req != loops:
        raise SystemExit(f"--bf16twin: --loops {loops_req} != artifact {loops} ({man_name})")
    if rmode_req is not None and rmode_req != rmode:
        raise SystemExit(f"--bf16twin: --recurrent-kv-cache-mode {rmode_req} != "
                         f"artifact {rmode} ({man_name})")
    if rmode not in ("per_loop", "last_loop"):
        raise SystemExit(f"--bf16twin: artifact mode {rmode!r} invalid")
    model_src = man["model_dir"]
    cfg, reader, _, _, _ = _open_source(model_src)
    S = man["tokens"]
    tag = f"loops{loops}_{rmode}_f32"
    ref = DoryRef(cfg, reader, n_loops=loops, recurrent_mode=rmode, bf16=True)
    import time
    t0 = time.time()
    if rmode == "per_loop":
        emb = reader.row_gather_f32("backbone.embeddings.weight", np.array(S))
        h, _ = ref.backbone(emb, np.arange(len(S)))
        log = ref.logits(h)
    else:
        h_rows, log_rows = [], []
        caches = None
        for i in range(len(S)):
            e = reader.row_gather_f32("backbone.embeddings.weight", np.array([S[i]]))
            h_i, caches = ref.backbone(e, np.array([i]), caches)
            h_rows.append(h_i[0])
            log_rows.append(ref.logits(h_i)[0])
        h = np.stack(h_rows).astype(np.float32)
        log = np.stack(log_rows).astype(np.float32)
    print(f"twin {'prefill' if rmode == 'per_loop' else 'tokenwise walk'} "
          f"{len(S)} tok (loops={loops} mode={rmode}): {time.time() - t0:.1f}s",
          flush=True)
    log_f32 = np.load(os.path.join(art, f"logits_{tag}.npy"))
    hid_f32 = np.load(os.path.join(art, f"hidden_{tag}.npy"))
    dl = np.abs(log.astype(np.float64) - log_f32.astype(np.float64))
    dh = np.abs(h.astype(np.float64) - hid_f32.astype(np.float64))
    top1_same = sum(int(np.argmax(log[i])) == int(np.argmax(log_f32[i]))
                    for i in range(len(S)))
    env = {
        "mode": "BF16 activation rounding (RNE) on every matmul in/out, F32 accumulate",
        "loops": loops,
        "recurrent_kv_cache_mode": rmode,
        "artifact": man_name,
        "max_abs_logit_diff": float(dl.max()),
        "mean_abs_logit_diff": float(dl.mean()),
        "p99_abs_logit_diff": float(np.quantile(dl, 0.99)),
        "max_abs_hidden_diff": float(dh.max()),
        "min_hidden_cosine": float(min(cosine(h[i], hid_f32[i]) for i in range(len(S)))),
        "top1_match": f"{top1_same}/{len(S)}",
    }
    print(json.dumps(env, indent=2))
    with open(os.path.join(art, f"bf16_envelope_loops{loops}_{rmode}.json"), "w") as f:
        json.dump(env, f, indent=2)
    return 0


# ----------------------------------------------------------------------------
# Focused numeric KV probe (no weights; exercises the production codec path).
# ----------------------------------------------------------------------------
#
# Deterministic DISTINCT Q/K/V (H=16, Hkv=4, D=256): K rows from a seeded
# Gaussian scaled by 1/sqrt(256); V rows carry LARGE head sentinels
# (V[0]=+64, V[1]=-32, rest small Gaussian) so SWA inclusion/exclusion and
# GQA-mapping errors are LARGE vs the 1e-5 same-codec bar. Q is 16 DISTINCT
# post-sqk-scale rows (Qnorm ~ head_dim = 256, representative of Dory's
# post-(sqk*factor)^2 queries) — NOT unit norm: scores must depend
# meaningfully on keys, or a weak K-rotation test hides faults behind a
# nearly-uniform softmax. Positions 0..1025 (full + SWA at 1023/1024/1025
# with window=1024) plus: (a) overwrite of 1024/1025 with EXACT exported
# vectors (the GPU probe ingests the same inputs and compares both outputs);
# (b) reset + fresh-prefix expectations (rows 0..few re-encoded with
# distinct data while the old tail remains — stale-tail reads show as
# large sentinel-driven errors). Compares: packed-blob re-decode exactness
# and full/SWA attention outputs (ALL 16 heads, full + overwrite + reset).
def run_kv_probe(args) -> int:
    rng = np.random.default_rng(20261018)
    HD, NH, NKV, NP = 256, 16, 4, 1026
    assert NH % NKV == 0
    rep = NH // NKV
    K = (rng.standard_normal((NP, NKV, HD)).astype(np.float32)
         / np.float32(np.sqrt(float(HD))))
    Vsmall = (rng.standard_normal((NP, NKV, HD)).astype(np.float32)
              / np.float32(np.sqrt(float(HD))))
    # Sentinel V rows: position p carries +/- amplitude on ALL heads so any
    # SWA-boundary or GQA-mapping slip moves the output by O(10), not 1e-3.
    V = Vsmall.copy()
    V[0] = np.float32(64.0)
    V[1] = np.float32(-32.0)
    # DISTINCT Q rows at Dory post-sqk scale: unit Gaussian * head_dim.
    Q = (rng.standard_normal((NH, HD)).astype(np.float32)
         * np.float32(float(HD)))
    S1 = gen_fwht_signs(42, 256)
    S2 = gen_fwht_signs(1042, 256)
    WIN = 1024
    out: dict = {"schema": "dory_kv_probe_v2", "hd": HD, "nh": NH, "nkv": NKV,
                 "n_pos": NP, "window": WIN,
                 "signs": {"seeds": [42, 1042], "fwht3_width": 256,
                           "fwht2_fwht4_width": 128},
                 "q": Q.tolist(),
                 "k": K.tolist(), "v": V.tolist(),
                 "modes": {}}
    ok = True
    # Exact overwrite/reset vectors exported for the GPU probe (SAME inputs).
    KOW = (rng.standard_normal((2, NKV, HD)).astype(np.float32)
           * np.float32(3.0))
    VOW = (rng.standard_normal((2, NKV, HD)).astype(np.float32)
           * np.float32(3.0))
    KRS = (rng.standard_normal((8, NKV, HD)).astype(np.float32)
           * np.float32(2.0))
    VRS = (rng.standard_normal((8, NKV, HD)).astype(np.float32)
           * np.float32(2.0))
    # F32 branch BEFORE any Q8 decode of raw F32 bytes (no wrong-codec work).
    for m in ("f32", "q8", "fwht2", "fwht3", "fwht4"):
        # Per-head dequants over ALL kv heads (GQA mapping checked head-wise).
        kdec = [[kv_encode_pos(K[p, h], V[p, h], m, S1, S2)[2]
                 for h in range(NKV)] for p in range(NP)]
        vdec = [[kv_encode_pos(K[p, h], V[p, h], m, S1, S2)[3]
                 for h in range(NKV)] for p in range(NP)]
        worst = 0.0
        rdtol = 0.0 if m in ("f32", "q8") else 3e-7
        for p in (0, 511, 1023, 1024, 1025):
            for h in range(NKV):
                kb, _, kd, _ = kv_encode_pos(K[p, h], V[p, h], m, S1, S2)
                if m == "f32":
                    rk = K[p, h]
                elif m == "q8":
                    rk = q8_decode_blob(kb, HD)
                else:
                    rk = fwht_decode_k_rotated(kb, m)
                worst = max(worst, float(np.abs(
                    rk.astype(np.float64) - kd.astype(np.float64)).max()))
        # full + SWA boundary attention, ALL 16 heads.
        att = {}
        for tag, rows, w in (("full1023", list(range(1024)), 0),
                             ("full1025", list(range(1026)), 0),
                             ("swa1023", list(range(1024)), WIN),
                             ("swa1024", list(range(1025)), WIN),
                             ("swa1025", list(range(1026)), WIN)):
            outs = []
            for hh in range(NH):
                kh = hh // rep
                o = kv_attention_ref(Q[hh], [kdec[p][kh] for p in rows],
                                     [vdec[p][kh] for p in rows], m, S1, S2,
                                     list(range(len(rows))), w)
                outs.append([float(x) for x in o])
            att[tag] = outs
        # (a) overwrite 1024/1025 with the EXPORTED vectors; GPU compares both.
        K2, V2 = K.copy(), V.copy()
        K2[1024], V2[1024] = KOW[0], VOW[0]
        K2[1025], V2[1025] = KOW[1], VOW[1]
        kdec2 = [[kv_encode_pos(K2[p, h], V2[p, h], m, S1, S2)[2]
                  for h in range(NKV)] for p in range(NP)]
        vdec2 = [[kv_encode_pos(K2[p, h], V2[p, h], m, S1, S2)[3]
                  for h in range(NKV)] for p in range(NP)]
        for tag, w in (("swa1025_overwrite", WIN), ("full1025_overwrite", 0)):
            outs = []
            for hh in range(NH):
                kh = hh // rep
                o = kv_attention_ref(Q[hh],
                                     [kdec2[p][kh] for p in range(NP)],
                                     [vdec2[p][kh] for p in range(NP)],
                                     m, S1, S2, list(range(NP)), w)
                outs.append([float(x) for x in o])
            att[tag] = outs
        # (b) reset + fresh 8-row prefix with EXPORTED distinct data; old tail
        # remains (stale-tail reads leak sentinels into the prefix outputs).
        kdec3 = [[kv_encode_pos(KRS[p, h], VRS[p, h], m, S1, S2)[2]
                  for h in range(NKV)] for p in range(8)]
        vdec3 = [[kv_encode_pos(KRS[p, h], VRS[p, h], m, S1, S2)[3]
                  for h in range(NKV)] for p in range(8)]
        outs = []
        for hh in range(NH):
            kh = hh // rep
            o = kv_attention_ref(Q[hh],
                                 [kdec3[p][kh] for p in range(8)],
                                 [vdec3[p][kh] for p in range(8)],
                                 m, S1, S2, list(range(8)), 0)
            outs.append([float(x) for x in o])
        att["reset_prefix8"] = outs
        print(f"kv-probe {m}: re-decode worst={worst:.3e} (tol {rdtol:.1e}) "
              f"att(full1025,h0)[:4]={[round(float(x), 6) for x in att['full1025'][0][:4]]}")
        ok = ok and worst <= rdtol
        out["modes"][m] = {"re_decode_worst": worst, "attn": att,
                           "k_row1025_head0": [float(x) for x in K[1025, 0, :8]],
                           "tolerances": {"re_decode": rdtol,
                                          "attn_vs_gpu": 1e-5}}
    out["overwrite"] = {"k1024": KOW[0].tolist(), "v1024": VOW[0].tolist(),
                        "k1025": KOW[1].tolist(), "v1025": VOW[1].tolist()}
    out["reset_prefix"] = {"k": KRS.tolist(), "v": VRS.tolist()}
    path = args.kv_probe_out or os.path.join(args.artifact, "dory_kv_probe.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"wrote {path} (schema dory_kv_probe_v2; "
          f"GPU probe compares its write+attend outputs against attn/*)")
    print("KV-PROBE " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


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
    "required_top1_match": 16,    # all positions must agree (16-token oracle)
}


def run_compare(args) -> int:
    # Whole-model parity: F32 canonical plus packed honesty. The native dump
    # MUST tag the EFFECTIVE resolved KV mode (raw "auto" is rejected — it
    # silently skips files); old one-loop dumps without loop identity are
    # rejected rather than aliased. Packed whole-model comparison KEEPS the
    # strict 0.02 max|dlogit| bar and may honestly FAIL (known packed-model
    # discrepancy vs F32 at the same weights); exact-input --kv-trace replay
    # then isolates codec/state causes. Trace replay stays the independent
    # packed diagnostic; it does not replace the whole-model verdict.
    art = args.artifact
    km = getattr(args, "kv_mode", "f32") or "f32"
    assert km in KV_MODES, f"--kv-mode {km} not in {KV_MODES}"
    loops = getattr(args, "loops", None)
    rmode = getattr(args, "recurrent_kv_cache_mode", None)
    nat = json.load(open(args.compare))
    for key in ("loops", "recurrent_kv_cache_mode", "kv_mode"):
        if key not in nat:
            raise SystemExit(f"native dump missing required identity field {key!r}")
    if loops is None:
        loops = nat["loops"]
    if rmode is None:
        rmode = nat["recurrent_kv_cache_mode"]
    if not isinstance(loops, int) or not (1 <= loops <= 4):
        raise SystemExit(f"--loops {loops!r} must be an integer 1..4")
    if rmode not in ("per_loop", "last_loop"):
        raise SystemExit(
            f"--recurrent-kv-cache-mode {rmode!r} must be per_loop|last_loop")
    tag = f"loops{loops}_{rmode}_{km}"
    man = json.load(open(os.path.join(art, f"dory_loops{loops}_{rmode}_{km}.json")))
    assert nat["loops"] == loops, \
        f"native loops {nat.get('loops')} != {loops}"
    assert nat["recurrent_kv_cache_mode"] == rmode, \
        f"native mode {nat.get('recurrent_kv_cache_mode')} != {rmode}"
    assert nat.get("kv_mode") == km, \
        f"native kv_mode {nat.get('kv_mode')!r} != effective {km!r} (dumps tag EFFECTIVE; raw auto rejected)"
    assert man.get("kv_mode", km) == km, \
        f"artifact kv_mode {man.get('kv_mode')} != --kv-mode {km}"
    assert list(nat["tokens"]) == list(man["tokens"]), \
        f"token mismatch: native {nat['tokens']} vs ref {man['tokens']}"
    log_r = np.load(os.path.join(art, f"logits_{tag}.npy")).astype(np.float64)
    hid_r = np.load(os.path.join(art, f"hidden_{tag}.npy")).astype(np.float64)
    log_n = np.asarray(nat["logits"], dtype=np.float64)
    assert log_n.shape == log_r.shape, f"{log_n.shape} vs {log_r.shape}"
    npos = log_r.shape[0]
    dl = np.abs(log_n - log_r)
    top1 = sum(int(np.argmax(log_n[i])) == int(np.argmax(log_r[i])) for i in range(npos))
    if km == "f32":
        print(f"positions={npos} loops={loops} mode={rmode} kv=f32 "
              f"max|dlogit|={dl.max():.5f} "
              f"(FAIL>{TOL['max_abs_logit_diff']:.1e} WARN>{TOL['warn_abs_logit_diff']:.1e}) "
              f"mean={dl.mean():.5f} top1={top1}/{npos}")
    else:
        print(f"positions={npos} loops={loops} mode={rmode} kv={km} "
              f"max|dlogit|={dl.max():.5f} (strict FAIL>2e-2; may honestly FAIL) "
              f"mean={dl.mean():.5f} top1={top1}/{npos}")
    env_path = os.path.join(art, f"bf16_envelope_loops{loops}_{rmode}.json")
    if os.path.exists(env_path):
        env = json.load(open(env_path))
        print(f"(context only, NOT the bar: BF16-twin max|dlogit|={env['max_abs_logit_diff']:.4f} "
              f"mincos={env['min_hidden_cosine']:.6f})")
    ok = True
    if km == "f32":
        if dl.max() > TOL["max_abs_logit_diff"]:
            print("FAIL max|dlogit| above tolerance"); ok = False
        elif dl.max() > TOL["warn_abs_logit_diff"]:
            print("WARN max|dlogit| above 1e-3 aim (triage reduction order before widening)");
        if top1 < npos:
            print("FAIL top-1 mismatch"); ok = False
    else:
        # Packed whole model: strict 0.02 bar, unchanged; a FAIL here is an
        # honest known-discrepancy verdict, not a tolerance to widen.
        if dl.max() > 2e-2:
            print(f"FAIL kv={km} max|dlogit| above strict 2e-2 (honest packed verdict; "
                  "use --kv-trace replay to isolate codec vs state)"); ok = False
    if "hidden" in nat and nat["hidden"] is not None:
        hid_n = np.asarray(nat["hidden"], dtype=np.float64)
        cosmin = min(cosine(hid_n[i], hid_r[i]) for i in range(npos))
        cos_tol = TOL["min_hidden_cosine"] if km == "f32" else KV_ATOL[km]["cos"]
        print(f"min hidden cosine={cosmin:.8f} (tol {cos_tol})")
        if cosmin < cos_tol:
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
            # Top-1 agreement is REQUIRED incl. packed (a packed-offset or
            # rotation fault flips decisions); no lossy exemption.
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
    ap = argparse.ArgumentParser(description="Dory multi-loop CPU reference oracle "
                                "(loops 1..4, per_loop/last_loop; F32 canonical + packed honest)")
    ap.add_argument("--model", default="./dory-4b.q8.hfq",
                    help="canonical same-weight pack (local dory-4b.q8.hfq) or a "
                         "BF16 safetensors dir; both decode through the same F32 "
                         "forward (Q8F16/F32 dequant vs exact BF16->F32)")
    ap.add_argument("--selftest", action="store_true",
                    help="miniature I/R/O fixtures: math fast-path, codec "
                         "re-decode/overwrite, loop indexing, per_loop "
                         "chunk/tokenwise invariance, last_loop tokenwise + "
                         "overwrite/reset (no weights needed)")
    ap.add_argument("--run", action="store_true",
                    help="teacher-forced run: per_loop bulk prefill + tokenwise "
                         "decode, last_loop fully tokenwise; emits "
                         "logits_/hidden_<loops>_<mode>_<kv>.npy + "
                         "dory_loops<loops>_<mode>_<kv>.json")
    ap.add_argument("--bf16twin", action="store_true",
                    help="BF16-activation envelope for one F32 artifact "
                         "(same loops/mode; flags select, else unambiguous "
                         "manifest discovery)")
    ap.add_argument("--compare", metavar="NATIVE_JSON", default=None,
                    help="native dump vs the matching CPU artifact (same "
                         "loops/mode/EFFECTIVE kv_mode; F32 canonical, packed "
                         "honest at the strict 0.02 bar)")
    ap.add_argument("--artifact", default="benchmarks/references/dory_multiloop",
                    help="artifact dir for --run manifests/npy + --bf16twin "
                         "envelopes (no old one-loop names)")
    ap.add_argument("--prompt", default="The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is",
                    help="prompt; first --seq ids form the fixed sequence")
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--prefill", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--kv-mode", default="f32", choices=list(KV_MODES),
                    help="KV encoding for --run/--compare/--kv-probe: f32 "
                         "(internal F32 reference, default, canonical parity), "
                         "q8, fwht2/3/4 (K packed per turbo_common.h, V always "
                         "Q8_0; signs 42/1042, fwht3 256-wide, fwht2/4 128 "
                         "halves). Packed whole-model keeps the strict 0.02 "
                         "bar and may honestly FAIL; --kv-trace replay "
                         "isolates codec vs state causes.")
    ap.add_argument("--loops", type=int, default=None,
                    help="recurrent passes 1..4 for --run/--bf16twin/--compare "
                         "(default: checkpoint n_recurrent_loops; --compare "
                         "falls back to the native dump; --bf16twin selects "
                         "the matching F32 manifest)")
    ap.add_argument("--recurrent-kv-cache-mode", default=None,
                    choices=["per_loop", "last_loop"],
                    help="KV semantics for --run/--bf16twin/--compare "
                         "(default: checkpoint recurrent_kv_cache_mode, "
                         "else per_loop; --compare falls back to the native "
                         "dump's mode)")
    ap.add_argument("--kv-probe", action="store_true",
                    help="focused numeric KV probe: deterministic Q/K/V "
                         "(Hkv=4,D=256), all 4 packed modes + F32, full + "
                         "sliding-window (1023/1024/1025 inc. overwrite with "
                         "different values) attention + JSON dump for the GPU "
                         "probe comparison. No weights needed.")
    ap.add_argument("--kv-probe-out", default=None,
                    help="probe JSON output path (default: "
                         "<artifact>/dory_kv_probe.json)")
    ap.add_argument("--kv-trace", metavar="PATH", default=None,
                    help="lab diagnostic for --run: dump per-(position,layer,"
                         "loop_index) precache q/k/v + raw pre-gate attention "
                         "to PATH (schema dory_kv_trace_v2; joins with the "
                         "native parity --kv-trace dump on "
                         "position+layer+loop_index, histories by cache_layer; "
                         "never affects the oracle math or verdicts)")
    args = ap.parse_args()
    if args.selftest:
        return run_selftest_main()
    if args.kv_probe:
        return run_kv_probe(args)
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
