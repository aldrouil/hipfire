# Resident-only lowered passback — amendment 1: committed source and evidence packaging

**Lifecycle:** `historical`. **Disposition:** unchanged: correctness passed on the stated deliberately spilled 9B fixture; no resolved end-to-end throughput gain, no performance-based promotion, and no 27B or additional attribution campaign.

**Date:** 2026-10-18. Amends the [unchanged original measurement record](2026-10-18-resident-lowered-passback-gfx1201.md). This amendment records the subsequent requested commit and evidence packaging, not another measurement. The original's statements that the implementation was uncommitted and no commit had been performed describe its reporting-time state.

## Committed source

Implementation and ignored hardware oracle are committed as `e92a1c3e98c0bd89147a2b86c9e5bbdd89bf279b` (`feat(qwen35): lower eligible resident dense layers under passback`). Exactly two code paths are included:

- `crates/hipfire-arch-qwen35/src/qwen35/forward.rs`
- `crates/hipfire-arch-qwen35/tests/resident_lowered_passback_hw.rs`

The repository's whole-changed-file Rust formatting gate required formatting existing debt in `forward.rs`. This post-measurement change is mechanical Rust formatting only; it does not change the resident-lowering implementation, passback scheduling, kernels, quantization, program caching, or numerical/state contracts. The oracle source remains byte-identical to the tested source.

Source SHA-256 identities:

| Source | SHA-256 |
|---|---|
| `forward.rs`, before required commit-time formatting | `811b2ae5a0ddb582c90d935fa0e2554d6debe59ca059edaffaf74d84fa4c64c0` |
| `forward.rs`, committed formatting | `e0341dd31421fdf2ff5d84717985a955ea0ed8fe1d717f35ac0dd28079558a31` |
| Hardware oracle, tested and committed | `a5fcb7a6fb955e7a144b482bf0f03505e1b4ff7fe48d35903cd76b2e55672d46` |

The measured candidate remains exact FINAL2 plus the archived feature **patch** SHA-256 `d46df385d1e324d3beef4ae18d1bbfbc5c5fa219a34aa67ced0134b591904b1d`, with the original record's binary hashes. Patch and whole-file hashes are different identities. No newly built binary or post-formatting performance result is substituted for the measured candidate. No GPU oracle or benchmark was rerun solely for formatting and committing.

The required code-commit gate, `scripts/pre-commit-ci.sh upstream/beta`, passed after regenerating the lifecycle inventory for the new hardware-test environment tokens. The generated additions are classified as harness controls, not new production settings. The complete [code gate receipt](data-2026-10-17-resident-lowered-gfx1201/PRECOMMIT-code.log) is included. Its Clippy warnings are retained; a passing gate does not mean a warning-free workspace.

## Versioned versus local-only evidence

The associated documentation commit versions the original report, this amendment, configuration/environment documentation, changelog, and the compact [evidence bundle](data-2026-10-17-resident-lowered-gfx1201/): process JSON/sample arrays, metadata, daemon proofs, oracle logs and manifests, decoded-smoke/build receipts, protocol, source provenance and patch, and archive checksums.

The approximately 1.8 GB of `ORACLE-dumps-run1.tar.zst` and `ORACLE-dumps-twin.tar.zst` raw tensor archives remain **local, untracked files**, preserved beside those receipts. They are not included in Git and were not uploaded to remote artifact storage. Original `/tmp` dump roots are also retained. A fresh checkout therefore cannot retrieve raw tensor bytes from Git: reproduce the documented oracle on the fixture or obtain the preserved archives out of band and verify `ORACLE-dumps.sha256`.

The share-zero case names its arms `cpu` and `passback0`, not `hand` and `mixed`. Their compact manifests and child logs were additionally copied unchanged from the original twin run into `ORACLE-receipts/run572723/cut16_share0_nb0_{cpu,passback0}/`; this is evidence packaging, not a new run. Earlier provenance receipts remain unchanged, including the superseded oracle hash in `FEATURE-provenance.txt`; `CANDIDATE-build-provenance.txt` carries the final tested oracle identity above. See [packaging details](data-2026-10-17-resident-lowered-gfx1201/EVIDENCE-PACKAGING.txt).

Earlier unrelated Phase 3 untracked research and scratch files are outside these commits. No deployment, push, further optimization, or new benchmark campaign is performed.
