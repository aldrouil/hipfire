// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors — see LICENSE and NOTICE in the project root.

//! Dory focused numeric KV probe (lab-gated): invokes the SAME production
//! helpers the forward uses — `dory_kv_prepare` (once per token/attend,
//! mapping + absolute/swa position uploads) then `dory_kv_write` /
//! `dory_kv_attend` (consume the prepared buffers, no per-layer copies) —
//! with deterministic Q/K/V (Hkv=4, D=256) and compares against the
//! independent CPU oracle (`scripts/dory_reference.py --kv-probe`).
//! Coverage: all 5 user presets (`f32` reference + `q8`/`fwht2`/`fwht3`/`fwht4`),
//! full + sliding-window attention at 1023/1024/1025 (plus overwrite of
//! 1024/1025 with different values, catching stale-tail reads), reset +
//! fresh-prefix rows 0..8, and recurrence history independence across
//! `--loops 1..4` / `--recurrent-kv-cache-mode per_loop|last_loop` (real
//! virtual-slot mapping from `DoryConfig::kv_layer_index`: per_loop disjoint
//! slots per pass, last_loop one shared slot with current-row overwrite).
//! Numerics catch wrong packed offsets / layer-index / rotation-norm /
//! Q8-fallback / cross-loop slot leaks. GPU output is NEVER used as the
//! expected cache — the consumer comparison is an independent CPU snapshot
//! (the JSON from `--kv-probe`), plus a layered channel check (write →
//! download raw bytes → independent CPU dequant vs `kv_attention_ref`).
//!
//! No model weights needed. No full benchmark.
//!
//! Usage:
//!   HIPFIRE_DEVICES=gfx1201 cargo run -p hipfire-arch-dory --features lab --example dory_kv_probe -- \
//!     --probe benchmarks/references/dory_multiloop/dory_kv_probe.json [--max-seq 2048] [--kv-backend legacy|vmm]
//!     [--loops 2] [--recurrent-kv-cache-mode per_loop]
//! Backend honors the shared storage enum (legacy contiguous default; vmm =
//! the production VMM arenas). F32 reference stays legacy and is labeled so.
//! Parent runs the SAME CPU expectations under both backends.
use hipfire_arch_dory::dory::{DoryConfig, DoryState};
use hipfire_arch_dory::kv::{dory_kv_attend, dory_kv_prepare, dory_kv_write};
use rdna_compute::DType;
use saddle_core::kv::KvMode;

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

fn parse_probe(path: &str) -> R<serde_json::Value> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    serde_json::from_str(&raw).map_err(|e| format!("parse {path}: {e}"))
}

fn flat_f32(v: &serde_json::Value) -> R<Vec<f32>> {
    // Recursive flatten: probe JSON stores k/v/q nested ([pos][kv][dim]).
    fn rec(v: &serde_json::Value, out: &mut Vec<f32>) -> R<()> {
        match v {
            serde_json::Value::Number(n) => {
                out.push(n.as_f64().unwrap_or(0.0) as f32);
            }
            serde_json::Value::Array(a) => {
                for x in a {
                    rec(x, out)?;
                }
            }
            _ => return Err("probe: expected number".to_string()),
        }
        Ok(())
    }
    let mut out = Vec::new();
    rec(v, &mut out)?;
    Ok(out)
}

fn run() -> R<()> {
    let argv: Vec<String> = std::env::args().collect();
    let probe = arg_val(&argv, "--probe")?
        .unwrap_or_else(|| "benchmarks/references/dory_multiloop/dory_kv_probe.json".into());
    let backend_raw = arg_val(&argv, "--kv-backend")?.unwrap_or_else(|| "legacy".into());
    let backend: hipfire_runtime::kv_backend::KvBackend =
        backend_raw
            .parse()
            .map_err(|e: hipfire_runtime::kv_backend::ParseKvBackendError| {
                format!("--kv-backend: {e}")
            })?;
    let max_seq: usize = arg_val(&argv, "--max-seq")?
        .map(|s| s.parse().map_err(|e| format!("--max-seq: {e}")))
        .transpose()?
        .unwrap_or(2048);
    if max_seq < 1026 {
        return Err(format!("--max-seq {max_seq} < 1026 probe rows"));
    }
    // Recurrence mapping under test: which virtual cache slots the probe
    // writes/reads (per_loop repeats the shared recurrent block per pass;
    // last_loop aliases physical layers and overwrites the current row).
    let loops: usize = arg_val(&argv, "--loops")?
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
        .transpose()?
        .unwrap_or(1);
    let rmode: hipfire_config::DoryRecurrentKvCacheMode =
        arg_val(&argv, "--recurrent-kv-cache-mode")?
            .map(|s| {
                s.parse::<hipfire_config::DoryRecurrentKvCacheMode>()
                    .map_err(|e| format!("--recurrent-kv-cache-mode: {e}"))
            })
            .transpose()?
            .unwrap_or_default();

    let mut gpu = rdna_compute::Gpu::init().map_err(|e| format!("gpu init: {e}"))?;
    eprintln!("dory_kv_probe: gpu arch = {}", gpu.arch);
    if !gpu.arch.contains("gfx1201") {
        return Err(format!(
            "wrong GPU arch '{}'; refusing to run probe (dGPU gfx1201 only)",
            gpu.arch
        ));
    }

    let doc = parse_probe(&probe)?;
    if doc["schema"] != "dory_kv_probe_v2" {
        return Err(format!(
            "probe schema {:?} != dory_kv_probe_v2",
            doc["schema"]
        ));
    }
    let hd = doc["hd"].as_u64().ok_or("probe: missing hd")? as usize;
    let n_heads = doc["nh"].as_u64().ok_or("probe: missing nh")? as usize;
    let nkv = doc["nkv"].as_u64().ok_or("probe: missing nkv")? as usize;
    let n_pos = doc["n_pos"].as_u64().ok_or("probe: missing n_pos")? as usize;
    let window = doc["window"].as_u64().ok_or("probe: missing window")? as usize;
    if hd != 256 || n_heads != 16 || nkv != 4 || n_pos != 1026 || window != 1024 {
        return Err(format!(
            "probe geometry hd={hd} nh={n_heads} nkv={nkv} n_pos={n_pos} win={window} != 256/16/4/1026/1024"
        ));
    }
    eprintln!(
        "dory_kv_probe: backend={backend} loops={loops} mode={rmode} (F32 reference always legacy)"
    );
    // Same I8/R20/O8 block split and default loop policy as production: the
    // checkpoint default is 1 loop here only because this codec probe has no
    // model weights — the mapping under test comes from --loops/--mode.
    let cfg = DoryConfig {
        hidden_size: 2560,
        num_hidden_layers: 72,
        vocab_size: 131072,
        intermediate_size: 7168,
        num_attention_heads: n_heads,
        num_key_value_heads: nkv,
        head_dim: hd,
        sliding_window: window,
        sliding_theta: 10000.0,
        full_theta: 10000000.0,
        initializer_range: 0.0177,
        is_attention: (0..72).map(|i| i % 2 == 0).collect(),
        is_full_attention: (0..72).map(|i| i % 4 == 2).collect(),
        block_split: (16, 40, 16),
        n_recurrent_loops: loops,
        recurrent_kv_cache_mode: rmode,
        alpha_scale: 0.05 / 0.0177,
        s_gate_factor: 2560f32.sqrt().sqrt() / 0.0177,
        sqk_factor: 256f32.sqrt() / 0.0177,
        suv_factor: 2560f32.sqrt(),
        eos_token_ids: vec![2],
    };

    // Parse ONCE outside the mode loop (constant inputs shared by all modes).
    let k_all = flat_f32(&doc["k"])?;
    let v_all = flat_f32(&doc["v"])?;
    let q_all = flat_f32(&doc["q"])?;
    let kv_dim = nkv * hd;
    if k_all.len() != n_pos * nkv * hd || v_all.len() != n_pos * nkv * hd {
        return Err(format!(
            "probe k/v len {} / {} != {}",
            k_all.len(),
            v_all.len(),
            n_pos * nkv * hd
        ));
    }
    if q_all.len() != n_heads * hd {
        return Err(format!("probe q len {} != {}", q_all.len(), n_heads * hd));
    }
    let ow_k1024 = flat_f32(&doc["overwrite"]["k1024"])?;
    let ow_v1024 = flat_f32(&doc["overwrite"]["v1024"])?;
    let ow_k1025 = flat_f32(&doc["overwrite"]["k1025"])?;
    let ow_v1025 = flat_f32(&doc["overwrite"]["v1025"])?;
    let rs_k = flat_f32(&doc["reset_prefix"]["k"])?;
    let rs_v = flat_f32(&doc["reset_prefix"]["v"])?;
    if ow_k1024.len() != kv_dim
        || ow_v1024.len() != kv_dim
        || ow_k1025.len() != kv_dim
        || ow_v1025.len() != kv_dim
    {
        return Err("probe: overwrite rows wrong length".into());
    }
    if rs_k.len() != 8 * kv_dim || rs_v.len() != 8 * kv_dim {
        return Err("probe: reset_prefix rows != 8".into());
    }

    let modes: &[(&str, KvMode)] = &[
        ("q8", KvMode::Q8),
        ("fwht2", KvMode::Fwht2),
        ("fwht3", KvMode::Fwht3),
        ("fwht4", KvMode::Fwht4),
        ("f32", KvMode::Q8), // F32 uses the internal reference state
    ];
    let mut failures = 0;
    for (name, mode) in modes {
        let mdoc = &doc["modes"][*name];
        let attn = mdoc
            .get("attn")
            .and_then(|v| v.as_object())
            .ok_or_else(|| format!("probe: mode {name} missing attn (schema drift?)"))?;
        let _ = attn;
        let tol_v = mdoc
            .get("tolerances")
            .and_then(|t| t.get("attn_vs_gpu"))
            .and_then(|v| v.as_f64())
            .ok_or_else(|| format!("probe: mode {name} missing tolerances.attn_vs_gpu"))?;
        let tol = tol_v as f32;
        if !(tol > 0.0) || tol != 1e-5 {
            return Err(format!(
                "probe: mode {name} tolerance {tol} != 1e-5 (schema drift?)"
            ));
        }
        // Build the production cache: F32 reference is explicitly legacy;
        // packed modes honor --kv-backend (shared storage enum, no fork).
        let use_backend = if *name == "f32" {
            hipfire_runtime::kv_backend::KvBackend::Legacy
        } else {
            backend
        };
        let mut st = if *name == "f32" {
            DoryState::new_f32_reference(&mut gpu, &cfg, max_seq)
        } else {
            DoryState::new_with_kv(&mut gpu, &cfg, max_seq, *mode, use_backend)
        }
        .map_err(|e| format!("probe {name}: state alloc: {e}"))?;
        // Probe the first RECURRENT attention layer; input layers execute
        // once and correctly ignore loop_index. per_loop must use disjoint
        // slots across recurrent passes; last_loop aliases one slot and
        // overwrites the current row each pass. A physical-vs-virtual mixup
        // shows as a cross-loop read leak (the independence probe below).
        let recurrent_layer = cfg.block_split.0;
        assert!(cfg.is_attention[recurrent_layer]);
        let cache_layers: Vec<(usize, usize)> = (0..loops)
            .map(|l| (l, cfg.kv_layer_index(recurrent_layer, l)))
            .collect::<Vec<_>>();
        let distinct: std::collections::HashSet<usize> =
            cache_layers.iter().map(|(_, c)| *c).collect();
        if rmode == hipfire_config::DoryRecurrentKvCacheMode::PerLoop {
            if distinct.len() != loops {
                return Err(format!(
                    "probe: per_loop loops={loops} expects {loops} distinct slots, got {cache_layers:?}"
                ));
            }
        } else if distinct.len() != 1 {
            return Err(format!(
                "probe: last_loop loops={loops} expects 1 shared slot, got {cache_layers:?}"
            ));
        }
        if cfg.kv_layer_count() != st.kv.k_gpu.len() {
            return Err(format!(
                "probe {name}: KV slots {} != kv_layer_count {} (loops={loops} mode={rmode})",
                st.kv.k_gpu.len(),
                cfg.kv_layer_count(),
            ));
        }
        let layer_idx = cache_layers[0].1;
        let kv_dim = nkv * hd;
        if k_all.len() != n_pos * nkv * hd || v_all.len() != n_pos * nkv * hd {
            return Err("probe: k/v wrong length".into());
        }
        if q_all.len() != n_heads * hd {
            return Err("probe: q wrong length (need 16 distinct heads)".into());
        }

        // All-16 DISTINCT queries straight from the JSON (no replication);
        // per-head outputs compare against the oracle's per-head rows, so a
        // GQA head-index bug fails LOUD (sentinel V rows amplify it).
        let q_t = gpu
            .upload_f32(&q_all, &[n_heads * hd])
            .map_err(|e| format!("probe {name}: upload q: {e:?}"))?;
        let out_t = gpu
            .alloc_tensor(&[n_heads * hd], DType::F32)
            .map_err(|e| format!("probe {name}: alloc out: {e:?}"))?;
        // Two reused GPU K/V row buffers (no alloc/free per token).
        let ks_buf = gpu
            .alloc_tensor(&[kv_dim], DType::F32)
            .map_err(|e| format!("probe {name}: alloc ks: {e:?}"))?;
        let vs_buf = gpu
            .alloc_tensor(&[kv_dim], DType::F32)
            .map_err(|e| format!("probe {name}: alloc vs: {e:?}"))?;

        // Write all positions through the PRODUCTION helpers: prepare once
        // per token (mapping + pos uploads), then the write consumes the
        // prepared buffers (no per-layer copies).
        for p in 0..n_pos {
            let k_row = &k_all[p * kv_dim..(p + 1) * kv_dim];
            let v_row = &v_all[p * kv_dim..(p + 1) * kv_dim];
            let kb: Vec<u8> = k_row.iter().flat_map(|v| v.to_ne_bytes()).collect();
            gpu.hip
                .memcpy_htod(&ks_buf.buf, &kb)
                .map_err(|e| format!("probe {name}: upload k: {e:?}"))?;
            let vb: Vec<u8> = v_row.iter().flat_map(|v| v.to_ne_bytes()).collect();
            gpu.hip
                .memcpy_htod(&vs_buf.buf, &vb)
                .map_err(|e| format!("probe {name}: upload v: {e:?}"))?;
            dory_kv_prepare(
                &mut gpu,
                &mut st.kv,
                &st.pos_buf,
                &st.swa_pos_buf,
                p,
                window,
            )
            .map_err(|e| format!("probe {name}: prepare p={p}: {e}"))?;
            dory_kv_write(
                &mut gpu,
                &mut st.kv,
                layer_idx,
                &ks_buf,
                &vs_buf,
                &st.pos_buf,
            )
            .map_err(|e| format!("probe {name}: write p={p}: {e}"))?;
        }

        // Attend at the boundary positions (full + SWA) through production.
        let tags = [
            ("full1023", true, 1023usize),
            ("full1025", true, 1025usize),
            ("swa1023", false, 1023usize),
            ("swa1024", false, 1024usize),
            ("swa1025", false, 1025usize),
        ];
        for (tag, is_full, pos) in tags {
            dory_kv_prepare(
                &mut gpu,
                &mut st.kv,
                &st.pos_buf,
                &st.swa_pos_buf,
                pos,
                window,
            )
            .map_err(|e| format!("probe {name}: prepare {tag}: {e}"))?;
            dory_kv_attend(
                &mut gpu,
                &mut st.kv,
                layer_idx,
                &q_t,
                &out_t,
                &st.attn_partials,
                &st.pos_buf,
                &st.swa_pos_buf,
                pos,
                is_full,
                window,
                n_heads,
            )
            .map_err(|e| format!("probe {name}: attend {tag}: {e}"))?;
            let got = gpu
                .download_f32(&out_t)
                .map_err(|e| format!("probe {name}: download {tag}: {e:?}"))?;
            if got.len() != n_heads * hd {
                return Err(format!("probe {name}: {tag} output len {}", got.len()));
            }
            let exp_rows = mdoc["attn"][tag]
                .as_array()
                .ok_or_else(|| format!("probe {name}: missing attn/{tag}"))?;
            if exp_rows.len() != n_heads {
                return Err(format!(
                    "probe {name}: {tag} expected heads {}",
                    exp_rows.len()
                ));
            }
            // ALL 16 heads vs the oracle's per-head rows (GQA mapping proof).
            let mut worst = 0.0f32;
            let mut worst_h = 0usize;
            for hh in 0..n_heads {
                let exp_h = flat_f32(&exp_rows[hh])?;
                if exp_h.len() != hd {
                    return Err(format!("probe {name}: {tag} head {hh} len {}", exp_h.len()));
                }
                for (a, b) in got[hh * hd..(hh + 1) * hd].iter().zip(exp_h.iter()) {
                    let d = (a - b).abs();
                    if d > worst {
                        worst = d;
                        worst_h = hh;
                    }
                }
            }
            let status = if worst <= tol { "ok" } else { "FAIL" };
            eprintln!("dory_kv_probe: {name} {tag}: max|d|={worst:.3e} @h{worst_h} tol={tol:.1e} {status}");
            if worst > tol {
                failures += 1;
            }
        }

        // REAL overwrite proof: re-write 1024/1025 with the EXACT exported
        // JSON vectors, then compare full AND SWA outputs against the oracle's
        // overwrite expectations (stale-tail reads fail the numeric bar).
        for (p, kow, vow) in [
            (1024usize, &ow_k1024, &ow_v1024),
            (1025usize, &ow_k1025, &ow_v1025),
        ] {
            let kb: Vec<u8> = kow.iter().flat_map(|v| v.to_ne_bytes()).collect();
            gpu.hip
                .memcpy_htod(&ks_buf.buf, &kb)
                .map_err(|e| format!("probe {name}: upload ow k: {e:?}"))?;
            let vb: Vec<u8> = vow.iter().flat_map(|v| v.to_ne_bytes()).collect();
            gpu.hip
                .memcpy_htod(&vs_buf.buf, &vb)
                .map_err(|e| format!("probe {name}: upload ow v: {e:?}"))?;
            dory_kv_prepare(
                &mut gpu,
                &mut st.kv,
                &st.pos_buf,
                &st.swa_pos_buf,
                p,
                window,
            )
            .map_err(|e| format!("probe {name}: prepare overwrite p={p}: {e}"))?;
            dory_kv_write(
                &mut gpu,
                &mut st.kv,
                layer_idx,
                &ks_buf,
                &vs_buf,
                &st.pos_buf,
            )
            .map_err(|e| format!("probe {name}: overwrite p={p}: {e}"))?;
        }
        for (tag, is_full) in [("swa1025_overwrite", false), ("full1025_overwrite", true)] {
            dory_kv_prepare(
                &mut gpu,
                &mut st.kv,
                &st.pos_buf,
                &st.swa_pos_buf,
                1025,
                window,
            )
            .map_err(|e| format!("probe {name}: prepare {tag}: {e}"))?;
            dory_kv_attend(
                &mut gpu,
                &mut st.kv,
                layer_idx,
                &q_t,
                &out_t,
                &st.attn_partials,
                &st.pos_buf,
                &st.swa_pos_buf,
                1025,
                is_full,
                window,
                n_heads,
            )
            .map_err(|e| format!("probe {name}: attend {tag}: {e}"))?;
            let got = gpu
                .download_f32(&out_t)
                .map_err(|e| format!("probe {name}: download {tag}: {e:?}"))?;
            let exp_rows = mdoc["attn"][tag]
                .as_array()
                .ok_or_else(|| format!("probe {name}: missing attn/{tag}"))?;
            if exp_rows.len() != n_heads {
                return Err(format!(
                    "probe {name}: {tag} expected heads {}",
                    exp_rows.len()
                ));
            }
            let mut worst = 0.0f32;
            let mut worst_h = 0usize;
            for hh in 0..n_heads {
                let exp_h = flat_f32(&exp_rows[hh])?;
                for (a, b) in got[hh * hd..(hh + 1) * hd].iter().zip(exp_h.iter()) {
                    let d = (a - b).abs();
                    if d > worst {
                        worst = d;
                        worst_h = hh;
                    }
                }
            }
            let status = if worst <= tol { "ok" } else { "FAIL" };
            eprintln!("dory_kv_probe: {name} {tag}: max|d|={worst:.3e} @h{worst_h} tol={tol:.1e} {status}");
            if worst > tol {
                failures += 1;
            }
        }
        // REAL reset proof: st.reset(), overwrite rows 0..8 with the EXPORTED
        // distinct reset vectors, then compare a fresh-prefix attend against
        // the oracle's reset_prefix8 expectation while the old tail remains
        // (stale-tail reads leak sentinels into the prefix outputs).
        st.reset();
        for p in 0..8usize {
            let kb: Vec<u8> = rs_k[p * kv_dim..(p + 1) * kv_dim]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect();
            gpu.hip
                .memcpy_htod(&ks_buf.buf, &kb)
                .map_err(|e| format!("probe {name}: upload reset k: {e:?}"))?;
            let vb: Vec<u8> = rs_v[p * kv_dim..(p + 1) * kv_dim]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect();
            gpu.hip
                .memcpy_htod(&vs_buf.buf, &vb)
                .map_err(|e| format!("probe {name}: upload reset v: {e:?}"))?;
            dory_kv_prepare(
                &mut gpu,
                &mut st.kv,
                &st.pos_buf,
                &st.swa_pos_buf,
                p,
                window,
            )
            .map_err(|e| format!("probe {name}: prepare reset p={p}: {e}"))?;
            dory_kv_write(
                &mut gpu,
                &mut st.kv,
                layer_idx,
                &ks_buf,
                &vs_buf,
                &st.pos_buf,
            )
            .map_err(|e| format!("probe {name}: reset write p={p}: {e}"))?;
        }
        dory_kv_prepare(
            &mut gpu,
            &mut st.kv,
            &st.pos_buf,
            &st.swa_pos_buf,
            7,
            window,
        )
        .map_err(|e| format!("probe {name}: prepare reset_prefix8: {e}"))?;
        dory_kv_attend(
            &mut gpu,
            &mut st.kv,
            layer_idx,
            &q_t,
            &out_t,
            &st.attn_partials,
            &st.pos_buf,
            &st.swa_pos_buf,
            7,
            true,
            window,
            n_heads,
        )
        .map_err(|e| format!("probe {name}: attend reset_prefix8: {e}"))?;
        {
            let got = gpu
                .download_f32(&out_t)
                .map_err(|e| format!("probe {name}: download reset_prefix8: {e:?}"))?;
            let exp_rows = mdoc["attn"]["reset_prefix8"]
                .as_array()
                .ok_or_else(|| format!("probe {name}: missing attn/reset_prefix8"))?;
            let mut worst = 0.0f32;
            let mut worst_h = 0usize;
            for hh in 0..n_heads {
                let exp_h = flat_f32(&exp_rows[hh])?;
                for (a, b) in got[hh * hd..(hh + 1) * hd].iter().zip(exp_h.iter()) {
                    let d = (a - b).abs();
                    if d > worst {
                        worst = d;
                        worst_h = hh;
                    }
                }
            }
            let status = if worst <= tol { "ok" } else { "FAIL" };
            eprintln!("dory_kv_probe: {name} reset_prefix8: max|d|={worst:.3e} @h{worst_h} tol={tol:.1e} {status}");
            if worst > tol {
                failures += 1;
            }
        }
        // Recurrence history independence + overwrite + reset (real slot
        // mapping, not config echo): each loop pass writes a DISTINCT
        // sentinel row to its virtual slot at position 7, then reads it
        // back; per_loop must return each pass's own row, last_loop the
        // FINAL pass's row from the shared slot. A cross-slot leak or stale
        // overwrite shifts the sentinel-dominated mean by O(1).
        for (loop_index, cache_layer) in cache_layers.iter().copied() {
            // Full-prefix rewrite (0..=7): the independence attend at 7 reads
            // the trailing window, so a single-row write would drown in the
            // earlier codec-probe rows. Rewriting the whole prefix makes the
            // slot under test dominate the mean.
            let sentinel = (loop_index as f32 + 1.0) * 8.0;
            let k_s = vec![sentinel; kv_dim];
            let v_s = vec![sentinel + 0.5; kv_dim];
            let kb: Vec<u8> = k_s.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let vb: Vec<u8> = v_s.iter().flat_map(|v| v.to_ne_bytes()).collect();
            for p in 0..8usize {
                gpu.hip
                    .memcpy_htod(&ks_buf.buf, &kb)
                    .map_err(|e| format!("probe {name}: upload sentinel k: {e:?}"))?;
                gpu.hip
                    .memcpy_htod(&vs_buf.buf, &vb)
                    .map_err(|e| format!("probe {name}: upload sentinel v: {e:?}"))?;
                dory_kv_prepare(
                    &mut gpu,
                    &mut st.kv,
                    &st.pos_buf,
                    &st.swa_pos_buf,
                    p,
                    window,
                )
                .map_err(|e| format!("probe {name}: prepare sentinel: {e}"))?;
                dory_kv_write(
                    &mut gpu,
                    &mut st.kv,
                    cache_layer,
                    &ks_buf,
                    &vs_buf,
                    &st.pos_buf,
                )
                .map_err(|e| format!("probe {name}: write sentinel: {e}"))?;
            }
        }
        for (loop_index, cache_layer) in cache_layers.iter().copied() {
            dory_kv_prepare(
                &mut gpu,
                &mut st.kv,
                &st.pos_buf,
                &st.swa_pos_buf,
                7,
                window,
            )
            .map_err(|e| format!("probe {name}: prepare reindependence: {e}"))?;
            dory_kv_attend(
                &mut gpu,
                &mut st.kv,
                cache_layer,
                &q_t,
                &out_t,
                &st.attn_partials,
                &st.pos_buf,
                &st.swa_pos_buf,
                7,
                true,
                window,
                n_heads,
            )
            .map_err(|e| format!("probe {name}: attend reindependence: {e}"))?;
            let got = gpu
                .download_f32(&out_t)
                .map_err(|e| format!("probe {name}: download reindependence: {e:?}"))?;
            // Exact (not dominated): rows 0..=7 of this slot are ALL the
            // pass's sentinel V, so any convex attention combination over the
            // full window equals the sentinel; the 1.0 bar only absorbs codec
            // dequant noise (Q8 exact, FWHT ~1e-5 scale).
            let mean = got.iter().map(|v| *v as f64).sum::<f64>() / got.len() as f64;
            let want = if rmode == hipfire_config::DoryRecurrentKvCacheMode::PerLoop {
                (loop_index as f64 + 1.0) * 8.0 + 0.5
            } else {
                (loops as f64) * 8.0 + 0.5
            };
            if (mean - want).abs() > 1.0 {
                eprintln!(
                    "dory_kv_probe: {name} rec-independence loops={loops} mode={rmode} pass={loop_index}: mean={mean:.3} want~{want:.3} FAIL"
                );
                failures += 1;
            } else {
                eprintln!(
                    "dory_kv_probe: {name} rec-independence loops={loops} mode={rmode} pass={loop_index}: mean={mean:.3} ok"
                );
            }
        }
        gpu.free_tensor(q_t)
            .map_err(|e| format!("probe {name}: free q: {e:?}"))?;
        gpu.free_tensor(out_t)
            .map_err(|e| format!("probe {name}: free out: {e:?}"))?;
        gpu.free_tensor(ks_buf)
            .map_err(|e| format!("probe {name}: free ks: {e:?}"))?;
        gpu.free_tensor(vs_buf)
            .map_err(|e| format!("probe {name}: free vs: {e:?}"))?;
        st.free_gpu(&mut gpu);
    }

    if failures > 0 {
        return Err(format!("dory_kv_probe: {failures} comparison(s) failed"));
    }
    eprintln!("dory_kv_probe: all modes passed (loops={loops} mode={rmode} backend={backend})");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("dory_kv_probe: ERROR: {e}");
        std::process::exit(1);
    }
}
