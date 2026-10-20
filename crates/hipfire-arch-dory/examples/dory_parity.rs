// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors — see LICENSE and NOTICE in the project root.

//! Dory multi-loop native parity driver: teacher-forced walk over the fixed
//! 16-token oracle sequence, dumping per-position logits + last-hidden rows.
//! The walk is fully tokenwise (native prefill IS tokenwise): one
//! `forward_token(true)` per position, so every compared row is a real lm-head
//! readout. Matches the CPU oracle scheduling for BOTH cache policies —
//! `forward_token` advances `next_pos` once per token with the same absolute
//! position on every recurrent pass (see `forward.rs`).
//!
//! Same-weight rule: the canonical comparison uses the local
//! `dory-4b.q8.hfq` pack for BOTH the CPU (`--run --kv-mode f32` resolves the
//! internal F32 reference from the SAME pack) and native sides — never a BF16
//! dir mixed against a Q8 pack.
//!
//! Load path: `load_dory_bundle(ModelSource::from_path(path), &mut LoadCtx)`
//! (self-contained HFQ pack or BF16 safetensors dir) with recurrence resolved
//! BEFORE weights/state from `--loops` / `--recurrent-kv-cache-mode`
//! (`LoadCtx::{dory_loops, dory_recurrent_kv_cache_mode}`; `None` = checkpoint
//! default: 2 loops, `per_loop`). Native F32 reference flags
//! (`--kv-mode f32`, the canonical parity gate) ride the internal F32 state
//! constructor (lab-gated); packed modes (`q8`/`fwht2`/`fwht3`/`fwht4`, and
//! `auto` = carrier-resolved) ride the production carrier and write honest
//! whole-model dumps compared at the strict 0.02 max|dlogit| bar — a packed
//! FAIL is an honest known-discrepancy verdict, and `--kv-trace` replay then
//! isolates codec vs state causes.
//! `download_last_hidden` after every position feeds the oracle cosine check.
//! Output schema matches `scripts/dory_reference.py --compare`:
//! `{"tokens","logits","hidden","loops","recurrent_kv_cache_mode",
//! "kv_mode":EFFECTIVE,"model","state"}` where `kv_mode`
//! is the EFFECTIVE resolved mode (never raw `auto`), `loops`
//! and `recurrent_kv_cache_mode` are the resolved schedule, and `state`
//! carries complete metadata (`next_pos`, `max_seq`, `kv_layers`,
//! `kv_layer_mask`, stored kv mode/backend).
//!
//! `--kv-mode f32|auto|q8|fwht2|fwht3|fwht4` (default `f32` = the internal
//! numeric reference, the canonical parity gate; `auto`/packed write ordinary
//! whole-model dumps compared at the strict 0.02 bar). The dump tags EFFECTIVE
//! (resolved `f32`/`q8`/`fwht2`/`fwht3`/`fwht4` — never raw `auto`).
//! Rejected aliases (deprecated user routes like `asym*`/`turbo*`/native
//! formats the site does not admit) fail closed in the carrier, never fall
//! back silently.
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
//!     --model ./dory-4b.q8.hfq --out ./dory-native.json \
//!     [--loops 2] [--recurrent-kv-cache-mode per_loop] [--kv-mode f32] \
//!     [--artifact benchmarks/references/dory_multiloop] [--max-seq 64]
//!     [--kv-trace ./dory-native-kv-trace.json]
//! Device selection: `Gpu::init()` honors HIPFIRE_DEVICES (arch selector
//! `gfx1201` = first free card of that arch); the driver then refuses any
//! non-gfx1201 arch as an iGPU guard.
//! No GPU work happens at build time; all device calls are runtime-only.
//!
//! `--kv-trace PATH` (lab diagnostic): capture per-(position, attention
//! layer, loop_index) precache `q`/`k`/`v` + raw pre-gate attention into PATH
//! as `{"schema":"dory_kv_trace_v2","kv_mode":EFFECTIVE,"loops":N,
//! "recurrent_kv_cache_mode":MODE,...,"rows":[...]}`. Row shapes:
//! `q`/`attention` H16xD256, `k`/`v` Hkv4xD256, flattened; `layer` is the
//! PHYSICAL index, `position` the absolute token, `loop_index` the recurrent
//! pass (input/output rows use 0), `cache_layer` the virtual slot from
//! `DoryConfig::kv_layer_index`. Joins with the CPU
//! `scripts/dory_reference.py --run --kv-trace` dump on
//! (position, layer, loop_index); KV histories join by `cache_layer`.
//! Capture only; the Python KV codec helpers can independently recompute
//! attention from these exact precache inputs.

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

/// Build the internal F32 reference directly (same `--loops` /
/// `--recurrent-kv-cache-mode` resolution the carrier applies, but without
/// routing an F32 preset through the packed production carrier). Recurrence
/// overrides apply to the checkpoint config BEFORE weights/state — an invalid
/// count/mode fails with no GPU weight allocation.
fn load_f32_reference(
    src: ModelSource,
    gpu: &mut rdna_compute::Gpu,
    max_seq: usize,
    loops: Option<usize>,
    mode: Option<hipfire_config::DoryRecurrentKvCacheMode>,
) -> R<hipfire_arch_dory::DoryBundle> {
    use hipfire_arch_dory::dory::{config_from_source, load_weights_from_source, DoryState};
    let load = |source: &dyn hipfire_runtime::model_source::ModelSource,
                gpu: &mut rdna_compute::Gpu|
     -> R<hipfire_arch_dory::DoryBundle> {
        let mut config = config_from_source(source)?;
        config
            .apply_recurrence_overrides(loops, mode)
            .map_err(|e| format!("dory_parity: {e}"))?;
        let eos_tok = *config
            .eos_token_ids
            .first()
            .ok_or("missing reference EOS")?;
        let weights = load_weights_from_source(source, &config, gpu)?;
        let state = match DoryState::new_f32_reference(gpu, &config, max_seq) {
            Ok(state) => state,
            Err(error) => {
                weights.free_gpu(gpu);
                return Err(error);
            }
        };
        Ok(hipfire_arch_dory::DoryBundle {
            config,
            weights,
            state,
            eos_tok,
        })
    };
    match src {
        ModelSource::Dir(source) => load(&source, gpu),
        ModelSource::Hfq(hfq) => {
            let source = hipfire_runtime::hfq::HfqModelSource::from_hfq(hfq);
            load(&source, gpu)
        }
    }
}

fn run() -> R<()> {
    let argv: Vec<String> = std::env::args().collect();
    let model = arg_val(&argv, "--model")?.unwrap_or_else(|| "./dory-4b.q8.hfq".into());
    let out = arg_val(&argv, "--out")?.unwrap_or_else(|| "./dory-native.json".into());
    let artifact = arg_val(&argv, "--artifact")?
        .unwrap_or_else(|| "benchmarks/references/dory_multiloop".into());
    // Whole-model parity: F32 canonical plus packed honesty. `f32` rides the
    // internal reference; `auto`/packed ride the production carrier and write
    // ordinary compared dumps at the strict 0.02 bar (an honest FAIL is kept,
    // never widened; --kv-trace replay isolates why).
    let kv_mode = arg_val(&argv, "--kv-mode")?.unwrap_or_else(|| "f32".into());
    let kv_trace_path = arg_val(&argv, "--kv-trace")?;
    let loops: Option<usize> = arg_val(&argv, "--loops")?
        .map(|s| {
            s.parse::<usize>()
                .map_err(|e| format!("--loops: {e}"))
                .and_then(|n| {
                    if (1..=4).contains(&n) {
                        Ok(n)
                    } else {
                        Err(format!("--loops {n} not in 1..=4"))
                    }
                })
        })
        .transpose()?;
    let mode_raw = arg_val(&argv, "--recurrent-kv-cache-mode")?;
    let rmode: Option<hipfire_config::DoryRecurrentKvCacheMode> = mode_raw
        .as_deref()
        .map(|s| {
            s.parse::<hipfire_config::DoryRecurrentKvCacheMode>()
                .map_err(|e| format!("--recurrent-kv-cache-mode: {e}"))
        })
        .transpose()?;
    let max_seq: usize = arg_val(&argv, "--max-seq")?
        .map(|s| s.parse().map_err(|e| format!("--max-seq: {e}")))
        .transpose()?
        .unwrap_or(64);

    let mut gpu = rdna_compute::Gpu::init().map_err(|e| format!("gpu init: {e}"))?;
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
    // LoadCtx: dory reads path/max_seq/gpu + dory_loops /
    // dory_recurrent_kv_cache_mode (+pp/cask/draft guards); the rest are
    // carrier-default (mirrors iu4_verify_kld.rs construction).
    // NOTE: `lc` borrows `gpu` AND `cask`/`model_str`; the borrow of `gpu`
    // must end before stepping. `load_dory_bundle` is the last use of `lc`,
    // so NLL releases the borrow there — stepping uses `gpu` directly.
    // `f32` rides the internal reference; anything else rides the production
    // carrier and writes an ordinary compared dump (EFFECTIVE mode tagged).
    let mut bundle = if kv_mode == "f32" {
        load_f32_reference(src, &mut gpu, max_seq, loops, rmode)?
    } else {
        let mut lc = LoadCtx {
            path: &model_str,
            max_seq,
            sequence: None,
            deepseek4_compute_placement: Default::default(),
            deepseek4_experts_per_token: None,
            dory_loops: loops,
            dory_recurrent_kv_cache_mode: rmode,
            draft_path: None,
            vision_path: None,
            mtp_path: None,
            vision_mode: "off".into(),
            kv_mode_override: if kv_mode == "auto" {
                None
            } else {
                Some(kv_mode.as_str())
            },
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
    let resolved_loops = bundle.config.n_recurrent_loops;
    let resolved_mode = bundle.config.recurrent_kv_cache_mode.to_string();
    eprintln!(
        "dory_parity: loaded hidden={} layers={} vocab={} max_seq={} loops={resolved_loops} mode={resolved_mode} kv_layers={}",
        bundle.config.hidden_size,
        bundle.config.num_hidden_layers,
        bundle.config.vocab_size,
        bundle.state.max_seq,
        bundle.config.kv_layer_count(),
    );
    if bundle.state.max_seq < TOKENS.len() {
        return Err(format!(
            "max_seq {} < {} tokens; rerun with --max-seq N",
            bundle.state.max_seq,
            TOKENS.len()
        ));
    }
    // State validation: expanded per_loop KV budget must match the resolved
    // virtual cache (carrier-sized), next_pos starts at 0. A silent truncation
    // here would alias recurrent slots and poison parity.
    if bundle.state.next_pos != 0 {
        return Err(format!(
            "dory_parity: fresh state next_pos={} != 0",
            bundle.state.next_pos
        ));
    }
    if bundle.state.kv.k_gpu.len() != bundle.config.kv_layer_count()
        || bundle.state.kv.v_gpu.len() != bundle.config.kv_layer_count()
    {
        return Err(format!(
            "dory_parity: KV slots {}/{} != kv_layer_count {} (loops={resolved_loops} mode={resolved_mode})",
            bundle.state.kv.k_gpu.len(),
            bundle.state.kv.v_gpu.len(),
            bundle.config.kv_layer_count(),
        ));
    }
    // Lab trace sink: enabled ONLY when --kv-trace is present (None =
    // no capture, no downloads). The state hook records exact precache
    // q/k/v + raw attention per (position, physical layer, loop_index);
    // histories join by cache_layer. See KvTraceRow.
    if kv_trace_path.is_some() {
        bundle.state.kv_trace = Some(Vec::new());
    }
    // Teacher-forced tokenwise walk: full logits at EVERY position (no `false`
    // fast path — every compared row must be a real lm-head readout).
    // Hidden snapshot after every step for the oracle cosine check. Same
    // absolute position on every recurrent pass; next_pos advances once per
    // token inside forward_token (both policies).
    let mut hiddens: Vec<Vec<f32>> = Vec::with_capacity(TOKENS.len());
    let mut logits_all: Vec<Vec<f32>> = Vec::with_capacity(TOKENS.len());
    let t0 = std::time::Instant::now();
    for &tok in TOKENS.iter() {
        let expect = bundle.state.next_pos;
        let row = forward_token(&mut gpu, &mut bundle, tok, true)?
            .ok_or_else(|| "dory_parity: forward_token(true) returned None".to_string())?;
        logits_all.push(row);
        hiddens.push(download_last_hidden(&mut gpu, &bundle)?);
        // State check per step: next_pos must advance exactly once per token.
        if bundle.state.next_pos != expect + 1 {
            return Err(format!(
                "dory_parity: next_pos {} != {} after token (loops={resolved_loops} mode={resolved_mode})",
                bundle.state.next_pos,
                expect + 1,
            ));
        }
    }
    eprintln!(
        "dory_parity: {} teacher-forced steps in {:.1}s (loops={resolved_loops} mode={resolved_mode})",
        TOKENS.len(),
        t0.elapsed().as_secs_f64()
    );
    // EFFECTIVE resolved KV mode (never raw `auto`): `f32` for the internal
    // reference path, else the live production cache mode. Shared by the
    // trace doc and the whole-model dump below.
    let effective_mode = if kv_mode == "f32" {
        "f32".to_string()
    } else {
        let m = format!("{:?}", bundle.state.kv_mode).to_lowercase();
        match m.as_str() {
            "q8" | "fwht2" | "fwht3" | "fwht4" => m,
            _ => {
                return Err(format!(
                    "dory_parity: unsupported effective KV mode {m} (requested {kv_mode})"
                ));
            }
        }
    };
    // Lab KV trace dump; capture does not change the parity verdict.
    if let Some(trace_path) = kv_trace_path.as_deref() {
        let rows = bundle
            .state
            .kv_trace
            .take()
            .ok_or("dory_parity: --kv-trace: trace sink missing")?;
        let n_rows = rows.len();
        let doc = serde_json::json!({
            "schema": "dory_kv_trace_v2",
            "kv_mode": effective_mode,
            "loops": resolved_loops,
            "recurrent_kv_cache_mode": resolved_mode,
            "n_heads": bundle.config.num_attention_heads,
            "n_kv_heads": bundle.config.num_key_value_heads,
            "head_dim": bundle.config.head_dim,
            "sliding_window": bundle.config.sliding_window,
            "is_full_attention": bundle.config.is_full_attention,
            "input_source": {
                "oracle": "native",
                "model": model,
                "kv_mode": effective_mode,
                "loops": resolved_loops,
                "recurrent_kv_cache_mode": resolved_mode,
                "tokens": TOKENS.to_vec(),
            },
            "rows": rows,
        });
        std::fs::write(
            trace_path,
            serde_json::to_string(&doc).map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("write {trace_path}: {e}"))?;
        eprintln!(
            "dory_parity: wrote {trace_path} \
             (schema=dory_kv_trace_v2 kv_mode={effective_mode} loops={resolved_loops} mode={resolved_mode} rows={n_rows})"
        );
    }

    // Ordinary compared dump for scripts/dory_reference.py --compare (F32
    // canonical; packed honest at the strict 0.02 bar), with complete
    // metadata and state validation. EFFECTIVE mode tagged.
    let dump = serde_json::json!({
        "tokens": TOKENS.to_vec(),
        "logits": logits_all,
        "hidden": hiddens,
        "loops": resolved_loops,
        "recurrent_kv_cache_mode": resolved_mode,
        "kv_mode": effective_mode,
        "model": model,
        // Label the actual encoding, not the Q8 enum placeholder used by
        // the internal F32 constructor.
        "state": {
            "next_pos": bundle.state.next_pos,
            "max_seq": bundle.state.max_seq,
            "kv_layers": bundle.config.kv_layer_count(),
            "kv_layer_mask": bundle.config.kv_layer_mask(),
            "kv_mode": effective_mode,
            "kv_backend": bundle.state.kv_backend.to_string(),
        },
    });
    std::fs::write(
        &out,
        serde_json::to_string(&dump).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("write {out}: {e}"))?;
    eprintln!("dory_parity: wrote {out} (kv_mode={effective_mode} loops={resolved_loops} mode={resolved_mode})");

    // Optional in-driver parity report (informational only; exit stays 0).
    // Tag matches the oracle (logits_loops{N}_{mode}_{kv}.npy, EFFECTIVE kv).
    let tag = format!("loops{resolved_loops}_{resolved_mode}_{effective_mode}");
    let ref_logits = PathBuf::from(&artifact).join(format!("logits_{tag}.npy"));
    let ref_hidden = PathBuf::from(&artifact).join(format!("hidden_{tag}.npy"));
    if ref_logits.exists() && ref_hidden.exists() {
        match parity_report(&artifact, &TOKENS, &logits_all, &hiddens, &tag) {
            Ok(text) => eprintln!("{text}"),
            Err(e) => eprintln!("dory_parity: report skipped: {e}"),
        }
    } else {
        eprintln!("dory_parity: no artifact npy {tag} at {artifact}; dump-only run");
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
    tag: &str,
) -> R<String> {
    let base = PathBuf::from(artifact);
    let (lshape, lref) = read_npy_f32_2d(&base.join(format!("logits_{tag}.npy")))?;
    let (hshape, href) = read_npy_f32_2d(&base.join(format!("hidden_{tag}.npy")))?;
    if lshape != [16, 131072] {
        return Err(format!("logits shape {lshape:?} != [16, 131072]"));
    }
    if hshape[0] != 16 || hshape[1] != hidden[0].len() {
        return Err(format!("hidden shape {hshape:?} mismatch"));
    }
    // Informational only (exit stays 0): F32 aims at the tight bar, packed at the strict 0.02 bar.
    let mut s =
        format!("dory_parity vs {artifact}/{tag} (strict max|dlogit|<=2e-2, cos>=.99999):\n");
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
