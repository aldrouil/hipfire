// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Can a shader read host-located VMM — and specifically the quantized path?
//!
//! The existing host-offload checks (`vmm_tensor_smoke`, `host_offload_smoke`)
//! verify a host-located tensor with `memcpy_htod` / `memcpy_dtoh` plus a handle
//! property check. Those are satisfied by a page the copy engine can DMA to while
//! a shader still cannot dereference it. Partial offload's only real requirement
//! on the host buffer is that a kernel READS it, and that was never tested — the
//! first real offloaded load died with "Page not present or supervisor privilege".
//!
//! Two arms, each comparing a host-located buffer against a device-resident twin:
//!
//!   arm 1 — `gemv_f32` over an F32 buffer. Models the memory question alone.
//!   arm 2 — `gemv_mq4g256` over a `DType::Raw` MQ4 code blob. This is the pair
//!           production actually runs (`upload_raw_host` + the MQ family), so it
//!           is the one that localises a fault to the consumer path.
//!
//! Arm 2 uses synthetic codes: the question is whether the kernel can READ the
//! host buffer, not whether the codes are meaningful. Only the SIZE must be
//! well-formed (m * (k/256) * 136 B for MQ4G256) or the kernel reads ragged.
//!
//! Usage: cargo run --release -p rdna-compute --example host_kernel_access

use rdna_compute::{DType, Gpu};

/// FWHT sign vectors the MQ GEMV kernels take as arguments.
fn fwht_signs(seed: u32) -> Vec<f32> {
    let mut st = seed;
    (0..256)
        .map(|_| {
            st = st.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (st >> 16) & 1 == 1 { 1.0 } else { -1.0 }
        })
        .collect()
}

/// Fill an existing tensor's bytes (used for the host-located twin, which must
/// go through the mapped VA rather than a normal upload).
fn fill(gpu: &Gpu, t: &rdna_compute::GpuTensor, bytes: &[u8]) {
    gpu.hip
        .memcpy_htod(&t.buf, bytes)
        .expect("memcpy_htod into mapped VA");
}

fn arm_f32(gpu: &mut Gpu) {
    // 390*64*4 = 99840 bytes: the exact size that faulted on the first real
    // offloaded load, and NOT a multiple of the 4096 VMM granularity, so this
    // also exercises the map round-up path rather than sidestepping it.
    const M: usize = 390;
    const K: usize = 64;

    let a_host: Vec<f32> = (0..M * K).map(|i| (i % 17) as f32 * 0.5).collect();
    let x: Vec<f32> = (0..K).map(|i| (i % 7) as f32).collect();

    let a_dev = gpu.upload_f32(&a_host, &[M, K]).unwrap();
    let xd = gpu.upload_f32(&x, &[K]).unwrap();
    let mut yd = gpu.zeros(&[M], DType::F32).unwrap();
    gpu.gemv_f32(&a_dev, &xd, &yd).unwrap();
    let reference = gpu.download_f32(&yd).unwrap();

    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(a_host.as_ptr() as *const u8, a_host.len() * 4) };
    let access = [gpu.device_id];
    let a_off =
        unsafe { gpu.alloc_vmm_tensor_host(&[M, K], DType::F32, M * K * 4, &access).unwrap() };
    assert!(gpu.vmm_host_located(&a_off), "not host-located");
    fill(gpu, &a_off, bytes);

    println!("arm1: launching gemv_f32 over a host-located F32 buffer...");
    let mut yh = gpu.zeros(&[M], DType::F32).unwrap();
    match gpu.gemv_f32(&a_off, &xd, &yh) {
        Ok(()) => {
            let got = gpu.download_f32(&yh).unwrap();
            let d = reference
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("ARM1_F32 {} (max diff {d})", if d == 0.0 { "PASS" } else { "MISMATCH" });
        }
        Err(e) => println!("ARM1_F32 FAULT: {e:?}"),
    }
}

fn arm_mq4(gpu: &mut Gpu) {
    // MQ4G256 layout: 136 bytes per 256-weight group.
    const M: usize = 512;
    const K: usize = 1024;
    let nbytes = M * (K / 256) * 136;
    // Deterministic, non-uniform pattern.
    let codes: Vec<u8> = (0..nbytes).map(|i| ((i * 31 + (i >> 7)) % 251) as u8).collect();
    let x: Vec<f32> = (0..K).map(|i| ((i % 13) as f32) * 0.25).collect();

    let a_dev = gpu.upload_raw(&codes, &[codes.len()]).unwrap();
    let xd = gpu.upload_f32(&x, &[K]).unwrap();
    let s1 = gpu.upload_f32(&fwht_signs(42), &[256]).unwrap();
    let s2 = gpu.upload_f32(&fwht_signs(1042), &[256]).unwrap();
    let mut yd = gpu.zeros(&[M], DType::F32).unwrap();
    gpu.gemv_mq4g256(&a_dev, &xd, &yd, &s1, &s2, M, K).unwrap();
    let reference = gpu.download_f32(&yd).unwrap();
    println!("arm2: device-resident gemv_mq4g256 ok ({} outputs)", reference.len());

    let access = [gpu.device_id];
    let a_off = unsafe { gpu.alloc_vmm_tensor_host(&[codes.len()], DType::Raw, codes.len(), &access).unwrap() };
    assert!(gpu.vmm_host_located(&a_off), "not host-located");
    fill(gpu, &a_off, &codes);

    println!("arm2: launching gemv_mq4g256 over a host-located MQ4 code blob...");
    let mut yh = gpu.zeros(&[M], DType::F32).unwrap();
    match gpu.gemv_mq4g256(&a_off, &xd, &yh, &s1, &s2, M, K) {
        Ok(()) => {
            let got = gpu.download_f32(&yh).unwrap();
            let d = reference
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("ARM2_MQ4 {} (max diff {d})", if d == 0.0 { "PASS" } else { "MISMATCH" });
        }
        Err(e) => println!("ARM2_MQ4 FAULT: {e:?}"),
    }
}

fn main() {
    let mut gpu = Gpu::init().unwrap();
    arm_f32(&mut gpu);
    arm_mq4(&mut gpu);
}
