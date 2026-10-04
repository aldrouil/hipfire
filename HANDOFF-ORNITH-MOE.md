# HANDOFF — get `ornith-1.5:35b-a3b` decoding coherently

**Point-in-time document.** Updated 2026-10-03 at commit `3b3d2ab45` on branch
`feature/qwen35-moe-offload`. The uncommitted tree the prior revision described
is committed; `git status --short` is clean, and `stash@{0}` remains archaeology
(see Stash warning). Re-verify anything line-numbered before acting on it. Read
`AGENTS.md`, `CLAUDE.md`, and `docs/QUANTIZATION.md` first.

**RESOLVED 2026-10-03 — ornith decodes coherently.** The constant `!` was the
V1 `MQ6G256` shared-expert path fogging `fused_gate_up_key_for`, whose catch-all
aliased it to the qt13 `FusedGateUpHfq4G256` kernel (136 B/group) over 200
B/group 6-bit bytes — a silent misread, NaN from finite inputs. Fixed in
`3b3d2ab45` (one arm: `DType::MQ6G256 => KernelKey::FusedGateUpHfq6G256`,
`crates/hipfire-dispatch/src/families/fused_qkv.rs`); verified end-to-end below.
The "Already exonerated" table and the retired leads are the measurement record
of how it was narrowed, not open work.

## Verification (2026-10-03)

Repro as written, fresh build (23.6 s), `HIPFIRE_GRAPH=0`, `--kv-mode fwht3
--spec off`, greedy, `memory.max_seq = 2048`:

```
HIPFIRE_MOE_V2_HOST_ALLOW=1 HIPFIRE_MOE_EXPERT_BUDGET=20 \
  ./target/release/hipfire run ornith-1.5:35b-a3b -t 0 -n 32 --no-stream \
    --kv-mode fwht3 --spec off 'The capital of France is located in'
# → The capital of France is Paris, which is located in the northern central
#   part of the country, along the Seine River.      (~23 s, was `!!!!`)
```

Guard rail — the previously-passing qt13 control under the same spill, no allow:

```
HIPFIRE_MOE_EXPERT_BUDGET=20 \
  ./target/release/hipfire run qwen3.5:35b-a3b -t 0 -n 32 --no-stream \
    --kv-mode fwht3 --spec off 'The capital of France is located in'
# → The capital of France is **Paris**. Paris is located in the **north-central
#   part of France**, situated along the banks of the **Seine River**.
```

Stage-1 plan §Verification arm A on `qwen3.5:35b-a3b`, prompt
`benchmarks/prompts/moe_offload_probe.txt` (md5 `1b1a4e91d31f6eefb7514352e0063a0f`):

- resident (no knobs) → refused, `hipMalloc: out of memory` at layer 34;
- `HIPFIRE_GPU_LAYER_BUDGET=26` → loads, `14 of 40 layers host-placed`, coherent;
- `HIPFIRE_MOE_EXPERT_BUDGET=auto` → loads, `routed experts host on 14 layers`,
  coherent, numbers consistent with the per-layer figure (408 MiB/layer).

## Goal and constraint

`ornith-1.5:35b-a3b` must produce coherent text on this box (16 GiB gfx1201,
ROCm 7.2). It cannot run resident (19.0 GB; resident attempt OOMs at layer 34
with 14 MB free — that cell is unavailable here, stop retrying it), so it must
run under the repo's expert-spill work. The real goal is working offload; a
working ornith is the accepted milestone.

## Repro (works as written)

```bash
cd <repo>
cargo build --release -p hipfire-daemon -p hipfire-cli   # MUST include the daemon — see traps
export HIPFIRE_HOME="$PWD/.redline-work/verify-home" \
       HIPFIRE_MODELS_DIR="$HOME/.hipfire/models" HIPFIRE_LOCAL=1 HIPFIRE_GRAPH=0

HIPFIRE_MOE_V2_HOST_ALLOW=1 HIPFIRE_MOE_EXPERT_BUDGET=20 \
  ./target/release/hipfire run ornith-1.5:35b-a3b -t 0 -n 4 --no-stream \
    --kv-mode fwht3 --spec off 'The capital of France is located in'
# → !!!!   (~5 min; warns once that the run is diagnostic-only)
```

(`HIPFIRE_GRAPH=0` is required once any CPU splice is active — a CPU step is a
host sync point and refuses capture. The splice refuses capture/replay loudly;
do not "fix" that refusal.)

Without `HIPFIRE_MOE_V2_HOST_ALLOW=1` the load **refuses by name**
(`offload::unverified_host_refusal`). That refusal is correct and must stay.
It also fires for dense qt44 spill (`qwen3.8-27b.mq4` + `GPU_LAYER_BUDGET=3`);
that is the guard working, not a regression — add the allow flag for diagnosis.

- **Symptom**: `!!!!` on every run. The NaN enters in **layer 1's DeltaNet
  attention block, pre-MoE**: layer-0 MoE exit finite (`|delta| 4.1`,
  `|res-out| 16`), layer-1 MoE entry NaN (`res-in` NaN). Layer-1 `dn_qkv`
  (QKVZA projection output) is finite (8192/8192) but `q_raw/k_raw/v` are all
  NaN, and the conv ring `conv_states[1]` is NaN (0/24576) while
  `conv_states[0]` is clean — before its conv reads it. The router `-inf`
  logits are downstream damage (softmax over NaN input), not a cause
  (no-fuse run fails identically; layer-1 `x_rot_local` already NaN).
- **Control**: `qwen3.5-35b-a3b.mq4` (qt 13 / `MQ4G256` experts) decodes
  `The capital of France` under the identical spill, AND under the full
  CPU-both splice (`The capital of France is **Paris**...`). ornith is qt 44 /
  `MQ4G256V2`, uniform across experts.
- **No small qwen35 MoE fixture exists.** The registry's qwen3.5 line is
  0.8b/2b/4b/9b/27b (all dense) plus `35b-a3b`, the only MoE. `lfm2.5-8b-a1b` is a
  different arch (`lfm2_moe`) and proves nothing about this path.
- **Dense qt44 fixture confusion (resolved)**: `qwen3.5-2b.mq4` and
  `qwen3.5-9b.mq4` are **qt13** (`v3-awq-f1` legacy; harness prints
  `qt=13 / MQ4G256`). Passes on them prove the qt13 stack only. The genuine
  dense qt44 on disk is **`qwen3.8-27b.mq4`** (harness prints `qt=44 /
  MQ4G256V2`, 15 GB). The pinned `qwen3.8-27b.mq4-xts` (sha `3e38ccba…`) is
  NOT on disk. Resident dense qt44 (`HIPFIRE_MAX_SEQ=512`, q8 KV):
  `**Paris**...` coherent. Spilled PCIe + allow: coherent. Spilled CPU-exec:
  stalls at `The` (27B/64-layer CPU GEMV wall, not evidence either way).

## Already exonerated — do not re-chase

Each was killed by running something, not by argument.

| suspect | why it is dead |
|---|---|
| MQ4G256V2 kernels | `mq4v2_moe_down_parity` example (new, uncommitted): production `gemv_mq4g256v2_moe_down` vs proven CPU on real ornith expert-0 bytes → rel 1.7e-7; production `gemv_mq4g256v2_moe_gate_up` vs CPU → rel 4.6e-7 with the CORRECT pairing asserted (`g≈cpu_g ∧ u≈cpu_u`, cross terms ~1.18). Device AND host-mapped. Old `mq4v2_moe_parity` down-only result stands. |
| CPU qt44 group decode | `cpu_quant_cross_check` extended with `qwen3.8-27b.mq4` + ornith: **24,085 tensors, qt {1,3,8,13,15,20,44}, bit-exact** vs canonical decoder (uncommitted test change). The V2 dual-half header path is correct on real asymmetric bytes. |
| host read of a real qt44 routed expert, the indexed MoE arm over a host-built pointer table, and packed views | `host_offload_smoke -- <ornith> ".mlp.experts."` → all PASS. Also `.mlp.shared_expert.` (which is **MQ6G256**, not qt44 — a qt44 layer host-places a *mixed* dtype set) |
| packer write loop (the "honest open doubt" — now closed) | `HIPFIRE_MOE_PACK_AUDIT=1`: every layer × all 256 slots × both roles, packed staging bytes vs file bytes — all match (320 printed lines = layers × roles × first-4-slots; full 256 checked silently). Strides confirmed: gate_up 1114112 B, down 557056 B. |
| expert pointer table | `[ptr-audit]`: all 256 gate_up + 256 down entries per layer equal owner-dev-base + slot×stride on every layer. `sub_offset` on a host-mapped owner yields device-alias + offset (Borrowed view); the table is correct. |
| dispatch dtype selection | `pipeline/mod.rs:1046-1065` is a plain `match dtype` with **separate** `MQ4G256` / `MQ4G256V2` arms plus `reject_mq4g128v2` |
| host-pointer fallback in the allocator | its warning fires **0 times** |
| MTP | both runs used `--spec off`, no draft attached (`draft 0 MiB`) |
| "the V2 down kernel never builds" | artifact of a `SIGKILL`ed compile; it has a full `.hash`/`.hip`/`.hsaco`/`.radiowave.json` set once actually used |
| dense qt44 codec + dense host read | `qwen3.8-27b.mq4` resident AND spilled-PCIe coherent (`Paris`). |
| `run_experts` missing hidden rotation (found live this session) | `hipfire-cpu/src/moe.rs`: no `rotate_x` between silu and down GEMV. Fixed + unit test (FWHT formats rotate, `Mq2G256LloydU` does not). 32 lib tests green. This was a real stage-2 executor defect. |

## Current lead (2026-10-04): layer-1 static inputs (ring reading corrected)

CORRECTION to the prefill-ring claim: `[moe-dn-prefill-ring] ring=0.000e0` is
read *before* that layer's batch conv writes the ring — "not yet written",
not "clean". The only genuine ring readings are decode-time:
`conv_states[0]` = 24576/24576 finite, `conv_states[1]` = NaN 0/24576 before
its conv. Do not cite the prefill zeros as exoneration of the ring.
Related asymmetry on record: decode layer 0 is finite end-to-end
(`res-out` 16.0) while prefill layer 1's `dn_qkv_batch` is already NaN with
a zero (unwritten) ring — the two paths corrupt at different points, so one
explanation may not cover both.

Static weights measured clean (`[moe-dn-static]`: layer 0 AND layer 1
`conv_w=32768/32768 dt_bias=32/32 a_log=32/32`, branch-independent probe):
conv_weight/dt_bias/a_log are exonerated on both layers. Ring stands alone:
`conv_states[0]` 24576/24576 finite, `conv_states[1]` NaN 0/24576 before its
conv. Next: when is dl1's ring written — prefill batch conv, decode conv on
a prior token, or never (init gap)? Bracket with the same ring read right
after load/reset (before prefill) and right after the prefill chunk.

## RESOLVED: prefill shared-expert gate/up (fixed in `3b3d2ab45`)

`[moe-prefill-dtypes]` (measured, layer 0, uncommitted probe): router=Q8_0,
selector=Q8_0, shared gate/up/down=MQ6G256, routed gate_up/down=MQ4G256V2.
`[moe-prefill-shared]` layer 0: xnorm finite, sgate/sup/srot all-NaN,
scalar finite. So the NaN is created in the **shared-expert gate/up GEMM**
(`prefill_shared_gate_up_stage`, pipeline/mod.rs:2476): MQ6G256 takes the
fused branch (mod.rs:2518) via `fused_gate_up_key_for(MQ6G256)` → `_ =>
FusedGateUpHfq4G256` (fused_qkv.rs:11-28) — the **qt13 kernel on qt15 bytes**
(136 B stride vs 200 B, 4-bit vs 6-bit packing). Silent misread → NaN, from
finite inputs, on the first body. The routed chain (all qt44, all proven)
never runs before the poison. Related real gap (not today's bug):
`prefill_shared_gemm_key` returns `None` for `MQ6G256` (mod.rs:2469), so the
non-fused GEMM path would refuse it outright — only the fused path
mistranslates silently. Fix: key V1 MQ6 to a real MQ6 fused kernel if one
exists, else refuse loudly like the GEMM path does.

## Prior lead: prefill batched MoE body layer 0 (confirmed, narrowed above)

`[moe-prefill-moe] n=13 x_batch before=26624/26624 after=0/26624` — the FIRST
batched MoE body (layer 0) poisons `x_batch` from finite inputs. All later
NaN (prefill rings, decode layer-1 entry, router -inf) is inherited. Decode
layer 0 stays finite because it runs a different body (indexed decode); the
bug is in the **grouped-prefill path** (Path 2 on gfx1201:
`dispatch_grouped_gemm`, scatter/unscatter, `y_down_grouped`) — a path none
of the decode-side exonerations cover. Next: bracket prefill down vs combine
stages and `rot_batch`/`gate_batch` finite counts entering layer 0's body.

## Prior lead: prefill conv ring (as recorded, with correction above)

`[moe-dn-prefill-ring]` (uncommitted, prefill.rs batch-conv arm): with a
1-token prompt, prefill layer 0 is `qkv_batch` finite + ring zero-clean, but
**every later prefill layer is `qkv_batch` NaN with a still-zero ring**.
The ring is innocent (zeros at alloc, never written before the NaN appears);
the NaN is already in the projection output entering the batch conv. That
moves the site to **prefill's QKVZA projection or earlier** (norms, `wo` of
the prior layer, or the prefill MoE combine), NOT the conv kernels and NOT
decode. Decode-side `conv_states[1]` NaN is inherited from this prefill
poison. Next: print `dn_qkv_batch` + `x_batch` finite-counts per prefill
layer to find the first layer whose projection output goes NaN, then step one
block earlier (that layer's MoE combine / `wo` / norms).

## What is NOT exonerated (in order)

1. **Prefill QKVZA / pre-MoE chain** — current lead above. Unprobed.
2. **Shared expert (MQ6G256)** — never silenced; still open but demoted: the
   NaN predates the MoE entirely (prefill ring evidence).
3. **Post-upload device view** — pack audit compares pre-upload staging;
   `host_bytes(&owners)` sliced at slot×stride vs file bytes is still
   unwritten. Demoted: packer + table + kernels all pass; do only if (1) misses.
4. **Q8 `s_matrices` mistag (`DType::F32` on int8 storage, weights.rs:2638-2650;
   Q4 same at :2655)** — real latent defect (every generic consumer mis-sizes
   the tensor; caused the "NaN state" false lead via `download_f32` on codes).
   File separately; NOT the ornith cause (FP32-state run fails identically,
   `quant=FP32` confirmed in-probe).

## Traps that cost hours

- **Rebuild `target/release/daemon`.** `hipfire run` executes the daemon, not
  `hipfire-cli`; a stale one makes every instrument look broken.
- **`down_expanded` is all zeros on every layer of *both* models** — it is never
  written on this arm (ninepath-D4 self-combines into `x_residual`; the expanded
  path doesn't write it here either — confirmed with `HIPFIRE_MOE_NINEPATH=0`).
  Never use it as a GPU reference. `|gpu-down| 0` carries no information.
- **NaN defeats `if rel > worst`** — NaN-vs-NaN scores 0 and reads as "bit-exact
  agreement". Compare with an explicit `is_nan()` check.
- **`HIPFIRE_HOME` does not relocate the kernel cache** — `~/.hipfire_kernels` is
  the live view.
- **k8-specialized gate_up kernel**: grid.y is always 8 ranks; topk must hold 8
  entries and y_gate/y_up `[8×mi]` each. A 1-entry topk + `[mi]` outputs
  overflows ranks 1..7 into pooled neighbors (the rel-1.65 artifact).
- **Decode CPU-both input**: on the prerotated arm `x_norm` aliases the RAW
  residual (`resolve_decode_normalization`); the only normalized+rotated
  activation is `x_rot_local`. Mirror the launcher (`moe_program.rs:777-781`):
  `x_rot_local` when `needs_x_rot_local`, else `x_norm` + one CPU rotation.
  Feeding `x_norm` + rotate on the prerotated arm is an R² error by another name.
- **Prefill CPU-both input is the opposite**: `x_norm_batch` is plain
  (Normalize writes it plain; InputBasis rotates into `x_rot_batch`), so the CPU
  rotates once. Do NOT "fix" it to `x_rot_batch` — that double-rotates.
- **`run_experts` consumes post-SiLU-rotated hidden for down**: fixed this
  session; CPU-down never caught it (it consumed the GPU's already-rotated
  `rot_batch`).
- **Harness `min`-of-four split metric passes swaps**: assert `wgg` and `wuu`
  (correct pairings), keep cross terms diagnostic-only.
- **Q8 `s_matrices` lie about their dtype**: int8 codes tagged `DType::F32`
  (`weights.rs:2638-2650`, Q4 same). `download_f32` on them fabricates NaN
  (~1.9% scattered, `first_nan=52` stable, layers 1+ read denormal-tiny).
  Never probe Q8 state as f32; read `s_scales` (real F32) or force FP32 state.
- **`HIPFIRE_MOE_CPU_DOWN` gate is `!= Some("0")`** (same trap as MoE-AWQ):
  unset/empty enables; explicit `0` disables. Same for the BOTH_DECODE/PREFILL
  sub-gates (`is_ok()` — any value enables).
- **The line editor replaces 1:1** — every narrow PUT deletes a neighbor. Use
  python heredoc string-replace for multi-line surgery (as done for the prefill
  both-branch and the parity loop rewrite).

## Instruments already wired

- `HIPFIRE_MOE_CPU_ORACLE=1` — per layer: `|x|`, `|gpu-down|` (=0 always here,
  see traps), `|cpu-down|`, zero counts, `down_awq`.
- `HIPFIRE_MOE_V2_HOST_ALLOW=1` — the only way to reach the unverified path;
  warns once, diagnostic-only.
- `HIPFIRE_LOAD_TRACE=1` — load timing / zero-copy lines.
- `HIPFIRE_MOE_CPU_DOWN=1` (committed `4d2cde5`) — zeroed down sink + CPU down
  recompute (decode + prefill), fail-closed on AWQ/tags/capture.
- `HIPFIRE_MOE_CPU_BOTH_DECODE=1` / `HIPFIRE_MOE_CPU_BOTH_PREFILL=1`
  (uncommitted) — gate_up sink + full expert FFN on CPU. Bisect halves
  independently. Control: decode-only was broken (fixed via x_rot_local),
  prefill-only always coherent.
- `HIPFIRE_MOE_PACK_AUDIT=1` (uncommitted) — pack bytes vs file bytes (all
  slots) + pointer-table audit per layer.
- `HIPFIRE_MOE_RESIDUAL_DELTA=1` (uncommitted) — residual before/after the
  sealed MoE step per layer; the only valid GPU routed reference on this arm.
- `HIPFIRE_MOE_NINEPATH=0` — force off the ninepath-D4 arm.
- `HIPFIRE_MOE_DN_STAGE=1` (uncommitted) — `[moe-dn-in]` (GDN inputs +
  `dn_qkv`/`dn_normed` span split, quant, scales), `[moe-dn-stage]` (post-GDN
  output + state), `[moe-dn-qkv]` (projection vs conv/split), `[moe-dn-ring]`
  (pre-conv ring, decode `run_attend` + prefill batch-conv arm),
  `[moe-dn-prefill-ring]` (prefill `dn_qkv_batch` vs ring per layer).
- `HIPFIRE_MOE_NO_V2_FUSE=1` (uncommitted, dispatch `gate_fusable_mq4v2`
  kill-switch) — forces the generic four-GEMV gate-side branch. Ornith fails
  identically with and without: the V2 fused gate kernel is exonerated.
- `HIPFIRE_MOE_ATTN_EXIT=1` (uncommitted) — `|x|` after attention `wo` per
  layer, on both the legacy and lowered (default) decode paths.
- `HIPFIRE_CONV_QKNORM=0` — force the plain `conv1d_silu_split_f32` arm.
  No change: both conv variants consume the same poisoned ring.
- `HIPFIRE_STATE_QUANT=fp32` (uncommitted TEMP-DIAG: loader `carriers.rs`
  + CLI `main.rs` thread-through) — FP32 DeltaNet state. Layer-0 state fully
  finite under FP32 yet layer 1 still enters NaN: Q8 state exonerated.
- `mq4v2_moe_down_parity` example (uncommitted, registered in Cargo.toml) —
  kernel-vs-CPU for gate_up + down, device + host. Replaces/augments the older
  down-only `mq4v2_moe_parity` claim in the table above.
- The two `--features lab` examples from the prior revision.

## Recommended attempt: prefill-first-NaN hunt (MoE fully exonerated)

The MoE chain (packer, table, gate_up, down, router-fuse, CPU transcription,
dense qt44) is proven; the NaN predates it. Demote everything below:

1. Print `|x_residual|` immediately after `moe_ffn_decode_impl` returns in
   layer 0 alongside the already-computed CPU routed contribution. Routed
   finite + residual NaN ⇒ shared expert (or its add).
2. Confirm from the load trace whether `moe offload: 40 expert tensors`
   covers routed experts only (shared follows `moe_target`, i.e. layer
   residency, not `expert_target`).
3. If resident: extend the sink to the shared gate/up/down the same way
   (CPU `silu_mul` + down GEMV). If resident and clean: the fault is
   routing/combine/renorm — instrument `topk_weights` (`tw0=-inf` seen on
   NaN layers is already a smoke signal worth chasing: where does -inf come
   from?).

**Why this is the shortest path**: everything else in the MoE layer is now
measured-good. The escalation ladder (down → both → shared) has one rung left.

## State at handoff

- Committed `4d2cde5` (on top of the prior `bc85744da` stack): CPU-down splice
  (loader sink + ownership, dispatch `moe_cpu_down_residual` + batched twin,
  forward + prefill hooks, lifecycle regen). Result: ornith still `!!!!`
  (L0 cpu finite 1.2e2, L1 MoE input NaN); qt13 control coherent under the
  identical splice.
- Uncommitted (9 files, ~500 insertions): CPU-both (gate_up sink field +
  ownership, `moe_cpu_both_residual` + batched twin, decode hook with
  launcher-mirrored input selection, prefill hook), `run_experts` hidden-
  rotation fix + test, cross-check fixture extension (qt44), pack + ptr
  audits, residual-delta probe, parity example + Cargo registration,
  `docs/env-vars.md` regen. Verify with `git status --short`.
- Green: `cargo test -p hipfire-runtime` (973), `-p hipfire-cpu` (32),
  `-p hipfire-arch-qwen35` (242), `-p hipfire-dispatch` (305). All 0 failures.
  Cross-check: 24,085 tensors incl. qt44, bit-exact.
- **Retracted claims (do not cite):** "dense qt44 spilled is coherent" for the
  2b/9b runs (those are qt13); "GPU qt44 down reads zero" from `|gpu-down|`
  (unwritten buffer); "qt44 gate_up kernel wrong at rel 1.65" (harness shape
  artifact — correct metric passes at 4.6e-7).
- **Two claims in older commit messages on this branch are wrong and retracted:**
  1. "every host-mapped read is bit-exact" — a NaN-comparison artifact. The guard
     was reverted on this premise and has since been restored.
  2. "MoE-AWQ is off by default" — the gate is `!= Some("0")`, i.e. **on unless
     disabled**.

## 3.6-35b-a3b probe (2026-10-04, errors only, no fixes)

Fixture `~/.hipfire/models/qwen3.6-35b-a3b.mq4p` (19 GB) + `.mtp` sidecar on disk.
Goal was AR baseline, then DFlash and MTP arms. None reached speculation:

1. Expert spill (`HIPFIRE_MOE_EXPERT_BUDGET=20`, also tried 0/10/15): load
   refuses at layer 0 — `expert packing does not apply: every routed expert
   must share one dtype and stride, and only MQ4/MQ4V2/MQ4C are packable`.
   Fail-closed by design (per-expert host fallback would cost ~512 MiB tail
   pad/layer). Harness spot-check: `experts.1.down_proj` is qt13/MQ4G256, so
   layer 0 is NOT uniform-qt44 like ornith; full per-expert qt census
   (`experts.{0..N}.down_proj`) NOT yet run — do that before claiming
   graded/mixed as the cause.
2. Resident: OOM at layer 30/40 (86 MB free). 19 GB > 16 GB, expected.
3. Max dense spill (`HIPFIRE_GPU_LAYER_BUDGET=40`, experts resident): refuses
   with numbers — needs 18,827 MiB device vs 13,420 available (5,406 over).
4. Budget semantics (measured): `Layers(N)` pins N resident expert layers,
   spills the rest; `0` = spill all (earlier reading inverted). Every
   midpoint (10/15/20) still touches layer 0 first and refuses there, so no
   intermediate budget avoids the pack check.
5. One anomalous run (`BUDGET=0` + dense 40) packed all 40 layers
   (`routed experts host on 40 layers, 16410 MiB`) then died on HOST admission
   (16.0 GiB pinned + 4 GiB headroom vs 16.5 GiB MemAvailable, 28 GB desktop).
   Same file packed fine there but refuses elsewhere — path-dependent, not
   understood; host RAM was NOT freed to test further.

Net: DFlash/MTP unreachable on this box (no placement loads). Per AGENTS.md
§6, 3.6-A3B DFlash is a documented ~50% loss vs AR anyway (draft trained on
3.5 traces), so AR is the recommendation there regardless.

## Stash warning (do not pop)

`stash@{0}` ("TEMP-DIAG ornith probes") is archaeology, not pending work. It
contains the default-on `HIPFIRE_MOE_CPU_DOWN` splice gate (`!= Some("0")`,
unset enables) that silently regressed the qt13 control from `Paris` to
`open think span`, plus the `Box::leak` TEMP-DIAG thread-through in
`carriers.rs`. A bare `git stash pop` reintroduces both onto a working
branch. Recover single files with `git checkout stash@{0} -- <path>` only,
never pop.
