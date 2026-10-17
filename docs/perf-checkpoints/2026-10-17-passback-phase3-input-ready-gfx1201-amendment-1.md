# Passback Phase 3, Step 1 — amendment 1: committed source identity

**Lifecycle:** `historical`. **Disposition:** provenance correction for the delivered commit after required formatting. No new performance claim or admission.

This amends the delivered-source identity in the unchanged [Phase 3 input-ready report](2026-10-17-passback-phase3-input-ready-gfx1201.md). Its numerical and causal execution evidence remains bound to the source and binaries it names. The original report and archived receipts are preserved.

## Delivered implementation

Code commit: **`8e618657f` — `perf(passback): add owner-coupled input-ready event boundary`**.

The commit contains the four implementation/test files, the two new public-method entries in `crates/hip-bridge/map.md`, and the associated `CHANGELOG.md` update. To pass the repository's mandatory changed-file formatting gate, rustfmt was applied with edition 2021 and `skip_children=true` to those four Rust files only. This includes existing formatting debt in the touched bridge/test files; no other Rust source was formatted. No controller, quantization, runtime policy or event-ordering change was added during this commit preparation.

| Artifact | Identity |
|---|---|
| Previously measured four-file patch (`FINAL2`) | SHA-256 `f855d7b92c919e454ee4c364f903e3f762d7ede264b48c80c2f6bc45639d30e0` |
| Committed, formatted four-file patch relative to `b236099f` (`FINAL3`) | SHA-256 `7b1f55a470c4a20a32a026cce9f963cd7b2feca08b815625fcd768096bcfea09` |
| Formatted CLI | MD5 `f6a608cd59b577fe201fb661bf4260d0` |
| Formatted daemon | MD5 `4c5285dfbd96de9e30468c6b29c4f7d2` |

`FINAL3` supersedes `FINAL2` **only as the delivered source/binary identity**. It does not retroactively change the identity of the earlier oracle, serving or profiler runs. The two temporary staging calls used for the causal trace are absent from the committed source.

## Verification after formatting

- Workspace release build completed and the native 9B fixed-share smoke returned successfully on the formatted binaries, with q8/VMM and share 0.5. Its rates are smoke observations, not a throughput comparison.
- `scripts/pre-commit-ci.sh upstream/beta` completed with **all checks passed**, including crate maps, rustfmt, clippy, ratchets, cargo-deny, environment documentation and lifecycle checks. The earlier `origin/beta` comparison had no merge base in this checkout; the successful run used the available `upstream/beta` reference. The local raw success log is `.codeinsight+research/passback-phase3/precommit-FINAL3-pass.log` and is not part of the evidence commit.
- The earlier final decoded 9B battery and numerical/stream oracles remain evidence for their explicitly recorded pre-format implementation. No new profiler campaign or throughput A/B was performed after formatting.

Receipts: [evidence bundle](data-2026-10-17-passback-phase3-input-ready-gfx1201/), specifically `FINAL3-production.patch`, `FINAL3-production.patch.sha256`, `FINAL3-identity.txt`, `FINAL3-binaries.md5` and `FINAL3-9B-s05-R1.json`. The four-file patch digest deliberately excludes the crate-map and changelog documentation changes included in the code commit.

**Conclusion unchanged:** same-step earlier GPU execution was demonstrated under instrumentation; a throughput improvement has not been established. The implementation is ready for a controlled trace-off comparison within the report's stated coverage limits. Step 1 does not include that comparison or scheduler optimization.
