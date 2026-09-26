// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Device-vs-host parity smoke for offloaded projection weights.
//!
//! Partial GPU offload loads an offloaded layer's quantized codes into
//! host-located VMM instead of VRAM. The promise is that this changes *where*
//! the bytes live and nothing else — same bytes, same dtype, same shape, so the
//! GEMV numerics are unchanged. This example proves that promise on real data
//! rather than by inspection: it loads the same tensor through both readers and
//! compares the code blobs bit-for-bit.
//!
//! It exists because the two readers share one quant-type match behind an
//! injected uploader, and a partial swap would be invisible until it OOMs or
//! silently spills to VRAM. One comparison covers every arm.
//!
//! Usage:
//!     cargo run --release -p hipfire-arch-qwen35 --example host_offload_smoke -- MODEL.hfq
//!
//! With no argument it defaults to `~/.hipfire/models/qwen3.5-9b.mq4`.

use hipfire_arch_qwen35::qwen35::load::{
    load_weight_tensor, load_weight_tensor_host, qwen35_tensor_name_candidates,
};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            std::env::var("HOME")
                .map(|h| format!("{h}/.hipfire/models/qwen3.5-9b.mq4"))
                .unwrap_or_else(|_| "qwen3.5-9b.mq4".into())
        });
    println!("model: {path}");

    let mut hfq = HfqFile::open(std::path::Path::new(&path))?;
    // Keep the mmap alive: both readers take the zero-copy mmap path first.
    let mut gpu = Gpu::init()?;

    // Probe for a real projection tensor rather than hardcoding a name: qwen3.5
    // checkpoints ship under several prefixes (`model.` / `model.language_model.`)
    // and VL builds interleave linear-attention and full-attention layers, so a
    // guessed name silently turns this into "tensor not found".
    const PROBES: &[&str] = &[
        "model.language_model.layers.3.self_attn.q_proj.weight",
        "model.layers.3.self_attn.q_proj.weight",
        "model.language_model.layers.3.self_attn.qkv_proj.weight",
        "model.layers.3.self_attn.qkv_proj.weight",
        "model.language_model.layers.4.mlp.up_proj.weight",
        "model.layers.4.mlp.up_proj.weight",
        "model.language_model.layers.4.mlp.gate_proj.weight",
        "model.layers.4.mlp.gate_proj.weight",
    ];
    let name = PROBES
        .iter()
        .copied()
        .find(|n| hfq.find_tensor_info(n).is_some())
        .unwrap_or_else(|| {
            panic!(
                "no projection tensor found; tried {PROBES:?} against {}",
                path
            )
        });
    let info = hfq.find_tensor_info(name).expect("probed above");
    // m = rows, k = columns for a 2-D projection; the loader validates these
    // against the format's own K%256 / blob-length guards.
    let m = info.shape.first().copied().unwrap_or(4096) as usize;
    let k = info.shape.get(1).copied().unwrap_or(4096) as usize;
    println!("picked {name} qt={} shape={:?}", info.quant_type, info.shape);

    let device = load_weight_tensor(
        &hfq,
        &gpu,
        name,
        m,
        k,
        qwen35_tensor_name_candidates,
    )?;
    let host = load_weight_tensor_host(
        &mut hfq,
        &mut gpu,
        name,
        m,
        k,
        qwen35_tensor_name_candidates,
    )?;

    assert_eq!(device.gpu_dtype, host.gpu_dtype, "dtype diverged");
    assert_eq!(device.m, host.m, "m diverged");
    assert_eq!(device.k, host.k, "k diverged");
    assert_eq!(device.row_stride, host.row_stride, "row_stride diverged");
    println!(
        "dtype={:?} m={} k={} bytes={}",
        device.gpu_dtype,
        device.m,
        device.k,
        device.buf.byte_size()
    );

    let dev_bytes = gpu.download_raw_bytes(&device.buf)?;
    let host_bytes = gpu.download_raw_bytes(&host.buf)?;
    assert_eq!(
        dev_bytes.len(),
        host_bytes.len(),
        "length diverged: device {} vs host {}",
        dev_bytes.len(),
        host_bytes.len()
    );
    assert!(
        dev_bytes == host_bytes,
        "code blobs differ: first mismatch at {:?}",
        dev_bytes.iter().zip(&host_bytes).position(|(a, b)| a != b)
    );

    println!("HOST_OFFLOAD_PARITY PASS ({} bytes identical)", dev_bytes.len());
    Ok(())
}
