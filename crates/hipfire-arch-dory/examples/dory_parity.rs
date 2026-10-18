// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors — see LICENSE and NOTICE in the project root.

//! Dory one-loop native parity driver: teacher-forced walk over the fixed
//! 16-token oracle sequence, dumping per-position logits + last-hidden rows.
//!
//! Load path: `load_dory_bundle(ModelSource::from_path(dir), &mut LoadCtx)`
//! (native BF16 safetensors dir; F32 activations/KV on device). Stepping:
//! `forward_token(true)` on EVERY position (full 131072-wide logits each
//! step; prefill fast-path `false` is a load-time-equivalent shortcut the
//! driver deliberately does NOT take, so every compared row is a real lm
//! head readout); `download_last_hidden` after every position for the
//! oracle cosine check. Output schema matches
//! `scripts/dory_reference.py --compare`:
//! `{"tokens","logits","hidden","loops":1}`.
//!
//! The driver always writes the dump and exits 0 on success. When the
//! reference artifact dir is present it also prints the parity report
//! (per-position max|dlogit|, top-1 agreement, hidden cosine) but NEVER
//! fails on mismatch: mismatch triage belongs to `--compare`, not the GPU
//! run (keeps device time cheap and avoids baking tolerances into the
//! example).
//!
//! Usage:
//!   HIPFIRE_DEVICES=gfx1201 cargo run -p hipfire-arch-dory --features lab --example dory_parity -- \
//!     --model /mnt/sx8200/dory --out /tmp/dory_native.json \
//!     [--artifact benchmarks/references/dory_one_loop] [--max-seq 64]
//! Device selection: `Gpu::init()` honors HIPFIRE_DEVICES (arch selector
//! `gfx1201` = first free card of that arch); the driver then refuses any
//! non-gfx1201 arch as an iGPU guard.
//! No GPU work happens at build time; all device calls are runtime-only.

use hipfire_arch_dory::{download_last_hidden, forward_token, load_dory_bundle};
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource};
use std::path::PathBuf;

/// Fixed oracle token sequence (see scripts/dory_reference.py `--run`
/// defaults): "The capital of France is Paris. The capital of Germany is
/// Berlin. The capital of Italy is" → first 16 ids.
const TOKENS: [u32; 16] = [
    1784, 8961, 1307, 5498, 1395, 6993, 1046, 1531, 8961, 1307, 10787, 1395, 10874, 1046, 1531,
    8961,
];

type R<T> = Result<T, String>;

fn arg_val(argv: &[String], name: &str) -> R<Option<String>> {
    let mut i = 1;
    while i < argv.len() {
        if argv[i] == name {
            if i + 1 >= argv.len() {
                return Err(format!("{name} needs a value"));
            }
            return Ok(Some(argv[i + 1].clone()));
        }
        i += 1;
    }
    Ok(None)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("dory_parity: ERROR: {e}");
        std::process::exit(1);
    }
}

fn run() -> R<()> {
    let argv: Vec<String> = std::env::args().collect();
    let model = arg_val(&argv, "--model")?.unwrap_or_else(|| "/mnt/sx8200/dory".into());
    let out = arg_val(&argv, "--out")?.unwrap_or_else(|| "/tmp/dory_native.json".into());
    let artifact = arg_val(&argv, "--artifact")?
        .unwrap_or_else(|| "benchmarks/references/dory_one_loop".into());
    let max_seq: usize = arg_val(&argv, "--max-seq")?
        .map(|s| s.parse().map_err(|e| format!("--max-seq: {e}")))
        .transpose()?
        .unwrap_or(64);

    let mut gpu = rdna_compute::Gpu::init().map_err(|e| format!("gpu init: {e}"))?;
    eprintln!("dory_parity: gpu arch = {}", gpu.arch);
    if !gpu.arch.contains("gfx1201") {
        return Err(format!(
            "wrong GPU arch '{}'; refusing to run parity (dGPU gfx1201 only)",
            gpu.arch
        ));
    }

    let src =
        ModelSource::from_path(&model).map_err(|e| format!("dory_parity: open {model}: {e}"))?;
    let cask = CaskConfig::default();
    let model_str: String = PathBuf::from(&model).to_string_lossy().into_owned();
    // LoadCtx: dory reads path/max_seq/gpu (+pp/cask/draft guards); the rest
    // are carrier-default (mirrors iu4_verify_kld.rs construction).
    // NOTE: `lc` borrows `gpu` AND `cask`/`model_str`; the borrow of `gpu`
    // must end before stepping. `load_dory_bundle` is the last use of `lc`,
    // so NLL releases the borrow there — stepping uses `gpu` directly.
    let mut bundle = {
        let mut lc = LoadCtx {
            path: &model_str,
            max_seq,
            sequence: None,
            deepseek4_compute_placement: Default::default(),
            deepseek4_experts_per_token: None,
            draft_path: None,
            vision_path: None,
            mtp_path: None,
            vision_mode: "off".into(),
            kv_mode_override: None,
            kv_k_override: None,
            kv_v_override: None,
            qwen_default_q8: false,
            kv_backend: Default::default(),
            kv_adaptive_override: None,
            state_quant_override: None,
            cask: &cask,
            pp: 1,
            pp_layer_counts: None,
            admitted_mesh: None,
            band_placement: None,
            spec: Default::default(),
            gpu: &mut gpu,
            gemma4_drafter_path: None,
            gemma4_draft_len: 3,
            xdna: None,
        };
        load_dory_bundle(src, &mut lc)?
    };
    eprintln!(
        "dory_parity: loaded hidden={} layers={} vocab={} max_seq={}",
        bundle.config.hidden_size,
        bundle.config.num_hidden_layers,
        bundle.config.vocab_size,
        bundle.state.max_seq
    );
    if bundle.state.max_seq < TOKENS.len() {
        return Err(format!(
            "max_seq {} < {} tokens; rerun with --max-seq N",
            bundle.state.max_seq,
            TOKENS.len()
        ));
    }

    // Teacher-forced walk: full logits at EVERY position (no `false`
    // fast path — every compared row must be a real lm-head readout).
    // Hidden snapshot after every step for the oracle cosine check.
    let mut hiddens: Vec<Vec<f32>> = Vec::with_capacity(TOKENS.len());
    let mut logits_all: Vec<Vec<f32>> = Vec::with_capacity(TOKENS.len());
    let t0 = std::time::Instant::now();
    for &tok in TOKENS.iter() {
        let row = forward_token(&mut gpu, &mut bundle, tok, true)?
            .ok_or_else(|| "dory_parity: forward_token(true) returned None".to_string())?;
        logits_all.push(row);
        hiddens.push(download_last_hidden(&mut gpu, &bundle)?);
    }
    eprintln!(
        "dory_parity: {} teacher-forced steps in {:.1}s",
        TOKENS.len(),
        t0.elapsed().as_secs_f64()
    );

    // Dump native JSON for scripts/dory_reference.py --compare.
    let dump = serde_json::json!({
        "tokens": TOKENS.to_vec(),
        "logits": logits_all,
        "hidden": hiddens,
        "loops": 1,
    });
    std::fs::write(
        &out,
        serde_json::to_string(&dump).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("write {out}: {e}"))?;
    eprintln!("dory_parity: wrote {out}");

    // Optional in-driver parity report (informational only; exit stays 0).
    let ref_logits = PathBuf::from(&artifact).join("logits_f32.npy");
    let ref_hidden = PathBuf::from(&artifact).join("hidden_f32.npy");
    if ref_logits.exists() && ref_hidden.exists() {
        match parity_report(&artifact, &TOKENS, &logits_all, &hiddens) {
            Ok(text) => eprintln!("{text}"),
            Err(e) => eprintln!("dory_parity: report skipped: {e}"),
        }
    } else {
        eprintln!("dory_parity: no artifact npy at {artifact}; dump-only run");
    }
    Ok(())
}

// --- minimal .npy reader (C-order float32, no fortran/padding surprises) ---

fn read_npy_f32_2d(path: &std::path::Path) -> R<(Vec<usize>, Vec<f32>)> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return Err(format!("{}: not npy", path.display()));
    }
    let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    if bytes.len() < 10 + hlen {
        return Err(format!("{}: truncated header", path.display()));
    }
    let header = String::from_utf8_lossy(&bytes[10..10 + hlen]).into_owned();
    if header.contains("fortran_order") && header.contains("True") {
        return Err(format!("{}: fortran order unsupported", path.display()));
    }
    if !header.contains("float32") && !header.contains("<f4") {
        return Err(format!("{}: expected float32: {header}", path.display()));
    }
    // shape tuple: first two ints in 'shape' entry.
    let shape_pos = header.find("shape").ok_or("npy: no shape")?;
    let mut dims: Vec<usize> = Vec::new();
    let mut num = String::new();
    for ch in header[shape_pos..].chars() {
        if ch.is_ascii_digit() {
            num.push(ch);
        } else if !num.is_empty() {
            dims.push(num.parse().map_err(|_| "npy: bad dim")?);
            num.clear();
            if dims.len() == 2 {
                break;
            }
        }
    }
    if dims.len() != 2 {
        return Err(format!("{}: expected 2-D shape", path.display()));
    }
    let start = 10 + hlen;
    let need = dims[0] * dims[1] * 4;
    if bytes.len() < start + need {
        return Err(format!("{}: truncated data", path.display()));
    }
    let mut out = Vec::with_capacity(dims[0] * dims[1]);
    for c in bytes[start..start + need].chunks_exact(4) {
        out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    Ok((dims, out))
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += x as f64 * y as f64;
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    d / (na.sqrt() * nb.sqrt() + 1e-30)
}

fn parity_report(
    artifact: &str,
    toks: &[u32; 16],
    logits: &[Vec<f32>],
    hidden: &[Vec<f32>],
) -> R<String> {
    let base = PathBuf::from(artifact);
    let (lshape, lref) = read_npy_f32_2d(&base.join("logits_f32.npy"))?;
    let (hshape, href) = read_npy_f32_2d(&base.join("hidden_f32.npy"))?;
    if lshape != [16, 131072] {
        return Err(format!("logits shape {lshape:?} != [16, 131072]"));
    }
    if hshape[0] != 16 || hshape[1] != hidden[0].len() {
        return Err(format!("hidden shape {hshape:?} mismatch"));
    }
    let mut s = format!("dory_parity vs {artifact} (tight aim: max|dlogit|<=1e-2, cos>=.99999):\n");
    let mut worst = 0f64;
    let mut wpos = 0;
    let mut top_same = 0;
    for i in 0..16 {
        let r = &lref[i * 131072..(i + 1) * 131072];
        let n = &logits[i];
        if n.len() != 131072 {
            return Err(format!("native logits row {i} len {}", n.len()));
        }
        let mut m = 0f64;
        let (mut ri, mut ni, mut rv, mut nv) =
            (0usize, 0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
        for j in 0..131072 {
            let d = (r[j] as f64 - n[j] as f64).abs();
            if d > m {
                m = d;
            }
            if r[j] > rv {
                rv = r[j];
                ri = j;
            }
            if n[j] > nv {
                nv = n[j];
                ni = j;
            }
        }
        if ri == ni {
            top_same += 1;
        }
        if m > worst {
            worst = m;
            wpos = i;
        }
        let cos = cosine(&href[i * hshape[1]..(i + 1) * hshape[1]], &hidden[i]);
        s.push_str(&format!(
            "  pos {i:2} tok={:6} ref_top1={:6} nat_top1={:6} max|dlogit|={m:.4} cos={cos:.6}{}\n",
            toks[i],
            ri,
            ni,
            if ri == ni { "" } else { "  <-- TOP1 DIFF" }
        ));
    }
    s.push_str(&format!(
        "worst max|dlogit|={worst:.4} at pos {wpos}; top1 {top_same}/16"
    ));
    Ok(s)
}
