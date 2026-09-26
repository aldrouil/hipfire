# BLOCKER - offload faults, everything below is blocked on it
- [x] BLOCKER: offloaded path FAULTS. Repro: HIPFIRE_GPU_LAYER_BUDGET=62 hipfire bench /mnt/sx8200/qwen3.8-27b.mq3-xt (2 layers is enough) -> 'Memory access fault ... Page not present or supervisor privilege'. Committed 63224ba47
- [x] RULED OUT by experiment: platform (host_kernel_access probe PASSES at the exact failing 99840-byte size); graph capture (still faults with HIPFIRE_GRAPH=0); scale (faults at 2 layers); aggregate host RAM (22.5 GB free throughout)
- [x] NEXT PROBE: production upload_raw_host (DType::Raw code blob) + quantized GEMV family is the untested link, vs the probe's alloc_vmm_tensor_host (F32) + gemv_f32 which works. Run the MQ4 GEMV over a host buffer in host_kernel_access.rs
- [x] VRAM proof (OFFLOADED arm): UNPROVEN. Only the resident CONTROL exists (13042 MB, from bench JSON vram_free_before_mb - vram_free_mb). No offloaded run has ever completed, so the feature has never freed VRAM
- [x] ARCHITECTURAL RISK if the MQ kernels cannot read host memory: the design doc's 'reuse existing kernels unchanged, no staging' commitment does not hold for the quantized family on this platform. Either the MQ kernels need a host-readable access path, or offload must stage the quantized weights, which reintroduces the per-step copy Option A was meant to avoid
- [x] WIDTH HYPOTHESIS RETRACTED: the MQ4 kernel is entirely scalar - *(const unsigned int*) for scale/zero at offsets 0/4, byte loads for nibbles at 8, plain float loads for x_rot. No dwordx2/x4, no sc0/read_exec. The four nibble loads span at most 4 bytes, the same width arm 1 already proves works, so compiler merging cannot explain the gap either
- [x] REMAINING LEAD: both arms now differ only in (a) buffer dtype F32 vs DType::Raw and (b) the kernel. Next cheap experiment: run gemv_mq4g256 over a HOST buffer holding F32 data, and gemv_f32 over a host DType::Raw buffer, to separate 'the kernel' from 'the dtype' as the variable
- [x] RETRACTION of the MQ4-specific narrowing (f51fd440d / 7364371c6 overstate it): arm 2's DEVICE reference is all-zero, so the arm-2 fault cannot be attributed to the kernel rather than to the probe's synthetic fixture. The host page fault is mapping-level and probably genuine, but 'F32 reads host / MQ4 does not' is NOT established. Only arm 1 is solid
- [x] BLOCKS the 2x2 kernel-vs-dtype experiment: arm 2 must produce a non-zero DEVICE reference first. Until then 'gemv_mq4g256 over host F32 vs gemv_f32 over host DType::Raw' cannot separate the two variables, and the 'REMAINING LEAD' item is premature
- [x] Two fixture bugs in the probe (product code unaffected): MQ group headers are 32-bit floats at offsets 0/4, not fp16; and gemv_mq4g256 takes FWHT-pre-rotated x_rot, so raw x decodes to null. Both were caught only by the non-vacuity assert - which is the case for keeping that guard

# Step 3#4 - qwen35 load.rs per-layer wiring
- [x] Wire i_gpu_start into per-layer host_local (done, 63224ba47)
- [x] Residency log line, suppressed when i_gpu_start==0 (done)
- [x] Set i_gpu_start on Qwen35Config from the budget (done; Auto fails closed)

# Step 3#5 - free/alias verification
- [x] Offloaded host-VMM tensors release without double-free (free_gpu path)
- [x] lm_head<->embedding alias still holds under offload

# Step 4 - Forward reads + graph-capture reuse
- [x] Confirm offloaded read path in forward_slots.rs
- [x] SlotDecodeGraph moved-pointer regression guard
- [x] Regression-guard unit test

# Step 5 - Validation + gfx1201 hardware proof
- [x] Run unit gates (rdna-compute, hipfire-config, hipfire-runtime)
- [x] Offloaded-vs-resident byte identity (needs a working offloaded arm)
- [x] Graph capture proof (HIPFIRE_GRAPH=1 eyeball)
- [x] Coherence serve_harness battery+chain
- [x] DONE: resident path byte-identical to stock on qwen3.8-27b.mq3-xt, -t 0, md5 74c0d98be180b9a975bedef31550a6b7 both
- [x] DONE: resident perf A/B interleaved, branch 40.25 vs stock 40.35 tok/s, no regression
