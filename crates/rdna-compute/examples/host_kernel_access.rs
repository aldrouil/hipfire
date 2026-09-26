// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Does a GPU SHADER actually read host-located VMM?
//!
//! Both existing host-offload checks (`vmm_tensor_smoke`, `host_offload_smoke`)
//! verify a host-located tensor with `memcpy_htod` / `memcpy_dtoh` plus a handle
//! property check. Those are satisfied by a page the GPU's copy engine can DMA to
//! while a shader still cannot dereference it. Partial offload has no other
//! requirement on the host buffer than that a kernel READS it, so that property is
//! the one that matters — and it is the one that faulted on the first real
//! offloaded load ("Page not present or supervisor privilege").
//!
//! This launches a real `gemv_f32` kernel with a host-located weight matrix `A`
//! and device-resident `x`/`y`, and compares against the identical computation with
//! a device-resident `A`. The split it produces is the useful one:
//!   * both succeed and agree  -> platform is fine; the fault was in our access setup
//!   * host faults             -> gfx1201 / ROCm 7.2 cannot do shader access to
//!                                 host-located VMM as configured, and the
//!                                 no-staging-copy design needs revisiting
//!
//! Usage: cargo run --release -p rdna-compute --features lab --example host_kernel_access

use rdna_compute::{DType, Gpu};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init()?;

    // Small, alignment-friendly shapes so the ONLY variable is memory location.
    // 390*64*4 = 99840 bytes -- NOT a multiple of the 4096 VMM granularity.
    // This is the exact size that faulted on the first real offloaded load, so the
    // experiment exercises the round-up path instead of sidestepping it.
    const M: usize = 390;
    const K: usize = 64;
    const N: usize = 4;

    let a_host: Vec<f32> = (0..M * K).map(|i| (i % 17) as f32 * 0.5).collect();
    let x_dev: Vec<f32> = (0..K).map(|i| (i % 7) as f32).collect();

    // Device-resident reference run.
    let a_dev = gpu.upload_f32(&a_host, &[M, K])?;
    let x = gpu.upload_f32(&x_dev, &[K])?;
    let mut y_dev = gpu.zeros(&[M], DType::F32)?;
    gpu.gemv_f32(&a_dev, &x, &y_dev)?;
    let reference = gpu.download_f32(&y_dev)?;
    println!("device-resident gemv: ok ({} outputs)", reference.len());

    // Same computation, weight matrix host-located.
    let access = [gpu.device_id];
    let a_off = unsafe { gpu.alloc_vmm_tensor_host(&[M, K], DType::F32, M * K * 4, &access)? };
    assert!(gpu.vmm_host_located(&a_off), "tensor is not host-located");
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(a_host.as_ptr() as *const u8, a_host.len() * 4)
    };
    gpu.hip.memcpy_htod(&a_off.buf, bytes)?;
    println!("host-located buffer written; launching kernel over it...");

    let mut y_host = gpu.zeros(&[M], DType::F32)?;
    match gpu.gemv_f32(&a_off, &x, &y_host) {
        Ok(()) => {
            let got = gpu.download_f32(&y_host)?;
            let max_diff = reference
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            if max_diff == 0.0 {
                println!("HOST_KERNEL_ACCESS PASS (shaders read host VMM; max diff {max_diff})");
            } else {
                println!("HOST_KERNEL_ACCESS MISMATCH: max diff {max_diff}");
                std::process::exit(1);
            }
        }
        Err(e) => {
            println!("HOST_KERNEL_ACCESS FAULT: {e:?}");
            println!("-> gfx1201 / ROCm 7.2 does not give shaders access to host-located VMM");
            std::process::exit(2);
        }
    }
    Ok(())
}
