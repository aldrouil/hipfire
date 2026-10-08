// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Eager-vs-replay oracle for the compressed-serial MTP verify-graph route.
//!
//! ONE `#[ignore]`d test with two modes selected by env:
//!
//! - Orchestrator (default): spawns two sequential fresh-process workers via
//!   `current_exe` — eager (`HIPFIRE_VERIFY_GRAPH=0`) then graph
//!   (`HIPFIRE_VERIFY_GRAPH=1`, master `HIPFIRE_GRAPH` inherited) — and
//!   compares every window. Sequential so only one 35B session is resident.
//!   Controls travel as per-`Command` env only; no process-global mutation.
//! - Worker (`HIPFIRE_MTP_VERIFY_GRAPH_ORACLE_ARM=eager|graph`): loads the
//!   graded fixture with an EXPLICIT fixed offload placement (first 16 expert
//!   layers host-mapped, trunk own-weights resident — never
//!   `ModelSlot::load`, which is fully resident and OOMs), runs native k=1/2/3
//!   windows plus teacher-forced takeover windows, a reset/second generation,
//!   and an unload/reload spot-check, writing `digests.log` plus raw `.bin`
//!   sidecars per window into its own unique scratch dir.
//!
//! Byte-exact contract (no tolerance, no NaN skipping, no report-only state):
//! committed ids, accept, advance, and the FULL raw bytes of every family —
//! verify logits/hidden, `prev_hidden`, DN S codes, DN scales, DN conv, DN EF
//! residual, retained trunk KV prefix, MTP KV prefix, GDN tape prefix.
//! Formats are honored, never aliased: F32 families download as F32 and must
//! be finite; F16 EF decodes as F16 and must be finite; Q8 KV families compare
//! as raw bytes with each 34-byte block's F16 scale checked finite (int8 codes
//! are naturally finite); DN S compares as raw int8 codes with its separate
//! F32 scales checked finite. EF must be ACTIVE (non-empty) under the default
//! Q8 state build on both arms.
//!
//! Takeover windows stay inside the validated 2..4-row envelope (never the old
//! 8+ spill): full / partial / zero accept derived from a deterministic k=0
//! reference plus one wrong token, followed by native continuation windows
//! proving retained state after rollback.
//!
//! ```bash
//! HIPFIRE_MTP_BYTE_IDENTITY_MODEL=~/.hipfire/models/qwen3.6-35b-a3b.mq4p \
//! HIPFIRE_MTP_BYTE_IDENTITY_HEAD=~/.hipfire/models/qwen3.6-35b-a3b.mtp \
//! cargo test --release -p hipfire-arch-qwen35 \
//!   --test mtp_verify_graph_oracle -- --ignored --nocapture
//! ```

#![allow(clippy::all)]

use hipfire_arch_qwen35::mtp_head::{self, MtpKvMode};
use hipfire_arch_qwen35::mtp_spec::{
    prefill_trunk_and_mtp_cache, spec_step_mtp_compressed_serial_with_k,
    spec_step_mtp_compressed_serial_with_takeover_candidates, MtpPromptRoute, MtpSpecState,
};
use hipfire_arch_qwen35::qwen35::{
    config_from_hfq, host_mapped_expert_accounting, load_weights, DeltaNetState, HfqSource,
    LayerType, Layout, Qwen35Scratch, StateQuant,
};
use hipfire_arch_qwen35::speculative::{KvMode, ModelSlot, ModelSlotConfig};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::offload::{ExpertResidency, Placement};
use rdna_compute::{Gpu, GpuTensor};
use std::path::{Path, PathBuf};

const CTX: usize = 4096;
const MAX_N: usize = 3;
const VERIFY_CAP: usize = 4;
const NATIVE_WINDOWS_PER_K: usize = 5;
const REF_STEPS: usize = 24;
const HOST_EXPERT_LAYERS: usize = 16;
const REPEAT_WINDOW: usize = 128;
const ARM_ENV: &str = "HIPFIRE_MTP_VERIFY_GRAPH_ORACLE_ARM";
const OUT_ENV: &str = "HIPFIRE_MTP_VERIFY_GRAPH_ORACLE_OUT";
const OUT_ROOT_ENV: &str = "HIPFIRE_MTP_VERIFY_GRAPH_ORACLE_OUT_ROOT";
const FAMS: [&str; 10] = [
    "logits", "hidden", "prev", "dn_s", "dn_scales", "dn_conv", "dn_ef", "kv", "mtpkv", "tape",
];
const PROMPT: &str = "Write a Rust function `fn parse_kv(line: &str) -> Option<(String, String)>` that splits \
a `key=value` line on the first '=', trims both sides, and rejects an empty key. Add three unit tests.";

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn f16_to_f32(b: u16) -> f32 {
    let s = ((b >> 15) & 1) as u32;
    let mut e = ((b >> 10) & 0x1f) as i32;
    let mut m = (b & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            // Subnormal: normalize.
            e = 1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (s << 31) | (((e + 112) as u32) << 23) | (m << 13)
        }
    } else if e == 31 {
        (s << 31) | (0xff << 23) | (m << 13)
    } else {
        (s << 31) | (((e + 112) as u32) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

fn assert_all_finite_f32(tag: &str, v: &[f32]) {
    assert!(!v.is_empty(), "{tag}: empty float array");
    let bad = v.iter().filter(|x| !x.is_finite()).count();
    assert!(bad == 0, "{tag}: {bad}/{} non-finite f32", v.len());
}

fn assert_all_finite_f16(tag: &str, bytes: &[u8]) {
    assert!(!bytes.is_empty(), "{tag}: empty f16 bytes");
    assert!(bytes.len() % 2 == 0, "{tag}: ragged f16 bytes");
    let mut bad = 0usize;
    for c in bytes.chunks_exact(2) {
        if !f16_to_f32(u16::from_le_bytes([c[0], c[1]])).is_finite() {
            bad += 1;
        }
    }
    assert!(bad == 0, "{tag}: {bad} non-finite f16 lanes");
}

/// Q8_0 block layout: 2-byte f16 scale + 32 int8 codes. Codes are naturally
/// finite; every scale must decode finite. Not a float array — never read as
/// f32 lanes.
fn assert_q8_blocks(tag: &str, bytes: &[u8]) {
    assert!(!bytes.is_empty(), "{tag}: empty q8 bytes");
    assert!(bytes.len() % 34 == 0, "{tag}: ragged q8 len {}", bytes.len());
    for (i, blk) in bytes.chunks_exact(34).enumerate() {
        let s = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        assert!(s.is_finite(), "{tag} block {i}: non-finite scale");
    }
}

fn raw_prefix(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<u8> {
    assert!(n <= t.buf.size(), "prefix {n} exceeds allocation {}", t.buf.size());
    let mut out = vec![0u8; n];
    gpu.hip.memcpy_dtoh(&mut out, &t.buf).expect("dtoh prefix");
    out
}

fn raw_all(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
    raw_prefix(gpu, t, t.buf.size())
}

fn model_paths() -> Option<(PathBuf, PathBuf)> {
    let model: PathBuf = std::env::var("HIPFIRE_MTP_BYTE_IDENTITY_MODEL").ok()?.into();
    let head: PathBuf = std::env::var("HIPFIRE_MTP_BYTE_IDENTITY_HEAD")
        .map(PathBuf::from)
        .unwrap_or_else(|_| model.with_extension("mtp"));
    Some((model, head))
}

fn unique_out_dir(root: Option<PathBuf>, arm: &str) -> PathBuf {
    let base =
        root.unwrap_or_else(|| std::env::temp_dir().join("vg-oracle"));
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    base.join(format!("{}-{}-{}", ms, std::process::id(), arm))
}

/// Test-only offloaded session assembly. Mirrors `ModelSlot::load` field by
/// field, except weights go through the production placement loader with an
/// EXPLICIT fixed placement: first 16 expert layers host-mapped, every layer's
/// own weights resident. No production API edits; no `ModelSlot::load` (fully
/// resident — OOMs on this fixture).
fn load_offloaded_session(
    gpu: &mut Gpu,
    model: &Path,
    name: &str,
    lines: &mut Vec<String>,
) -> ModelSlot {
    let mut hfq = HfqFile::open(model).expect("open trunk");
    let config = config_from_hfq(&hfq).expect("trunk config");
    assert!(config.num_experts > 0, "oracle fixture must be MoE");
    let n = config.n_layers;
    assert!(n > HOST_EXPERT_LAYERS, "fixture has only {n} layers");

    let mut placement = Placement::all_device(n);
    for r in placement.expert_residency.iter_mut().take(HOST_EXPERT_LAYERS) {
        *r = ExpertResidency::HostMapped;
    }
    assert_eq!(placement.host_expert_layers(), HOST_EXPERT_LAYERS);
    assert_eq!(placement.host_layers(), 0, "trunk own-weights must stay resident");
    let layout = Layout::single(n).with_placement(placement);

    let weights = {
        let mut src = HfqSource::new(&mut hfq, &config);
        load_weights(&mut src, std::slice::from_mut(gpu), &layout).expect("load weights")
    };
    let (host_tensors, host_bytes) = host_mapped_expert_accounting(&weights.layers);
    assert!(host_tensors > 0, "fixed placement spilled 0 expert tensors");
    // Route into digests.log (not just stderr) — the orchestrator reads these.
    let host_line = format!(
        "HOST tensors={host_tensors} bytes={host_bytes} host_expert_layers={HOST_EXPERT_LAYERS}"
    );
    eprintln!("[vg-oracle] {host_line}");
    lines.push(host_line);
    let place_line = "PLACEMENT explicit-fixed-first16 not-auto".to_string();
    eprintln!("[vg-oracle] {place_line}");
    lines.push(place_line);

    let is_kv_layer: Vec<bool> = config
        .layer_types
        .iter()
        .map(|t| *t == LayerType::FullAttention)
        .collect();
    let kv_cache = KvCache::new_gpu_q8_filtered(
        gpu,
        &is_kv_layer,
        config.n_kv_heads,
        config.head_dim,
        CTX,
    )
    .expect("q8 kv");
    let dn_state =
        DeltaNetState::new_with_quant(gpu, &config, StateQuant::Q8).expect("q8 dn state");
    // Default Q8 build runs with F16 error-feedback; the orchestrator enforces
    // non-empty EF agreement across arms (both-active normally, both-empty
    // only under an explicit HIPFIRE_DN_STATE_EF=0 opt-out).
    let scratch = Qwen35Scratch::new(gpu, &config, REPEAT_WINDOW).expect("scratch");
    ModelSlot {
        name: name.to_string(),
        hfq,
        config,
        weights,
        kv_cache,
        dn_state,
        scratch,
        slot_config: ModelSlotConfig {
            max_seq: CTX,
            kv_mode: KvMode::Q8,
            repeat_window: REPEAT_WINDOW,
            state_quant: StateQuant::Q8,
        },
        dspark_extract_layers: Vec::new(),
        vision_config: None,
        vision_weights: None,
    }
}

fn free_session(gpu: &mut Gpu, slot: ModelSlot, state: MtpSpecState, head: mtp_head::Qwen35MtpHead) {
    gpu.invalidate_graph_state();
    state.free_gpu(&mut *gpu);
    gpu.invalidate_weight_caches();
    head.free_gpu(&mut *gpu);
    slot.kv_cache.free_gpu(&mut *gpu).expect("free kv");
    slot.dn_state.free_gpu(&mut *gpu);
    slot.scratch.free_gpu(&mut *gpu).expect("free scratch");
    slot.weights.free_gpu(&mut *gpu);
    gpu.drain_pool();
}

fn build_prompt(tokenizer: &hipfire_runtime::tokenizer::Tokenizer) -> Vec<u32> {
    [
        "<|im_start|>",
        "user",
        "\n",
        PROMPT,
        "<|im_end|>",
        "\n",
        "<|im_start|>",
        "assistant",
        "\n",
        "<think>\n\n</think>\n\n",
    ]
    .iter()
    .flat_map(|s| tokenizer.encode(s))
    .collect()
}

fn prefill_and_seed(
    gpu: &mut Gpu,
    slot: &mut ModelSlot,
    head: &mtp_head::Qwen35MtpHead,
    state: &mut MtpSpecState,
    prompt: &[u32],
) -> u32 {
    slot.reset_state(gpu).expect("reset trunk");
    state.reset(gpu).expect("reset head");
    prefill_trunk_and_mtp_cache(gpu, slot, head, state, prompt, 0, MtpPromptRoute::ArRoute)
        .expect("prefill");
    let logits = gpu.download_f32(&slot.scratch.logits).expect("seed logits");
    assert_all_finite_f32("prefill-logits", &logits);
    logits
        .iter()
        .enumerate()
        .fold((0u32, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i as u32, x)
            } else {
                (bi, bv)
            }
        })
        .0
}

/// Raw bytes of every compared family in `FAMS` order: verify logits/hidden
/// (live `n_verify` rows), full `prev_hidden`, full DN S codes / scales / conv
/// / EF, retained trunk KV prefix (`end_pos` rows), MTP KV prefix, GDN tape
/// prefix (`n_verify` rows). Formats checked here; equality checked by caller.
fn collect_fams(
    gpu: &Gpu,
    slot: &ModelSlot,
    state: &MtpSpecState,
    n_verify: usize,
    end_pos: usize,
) -> Vec<Vec<u8>> {
    let dim = slot.config.dim;
    let vocab = slot.config.vocab_size;
    let logits = raw_prefix(gpu, &state.verify_logits, n_verify * vocab * 4);
    let hidden = raw_prefix(gpu, &state.verify_hidden, n_verify * dim * 4);
    let prev = raw_all(gpu, &state.prev_hidden);
    assert_all_finite_f32("verify-logits", &{
        assert!(logits.len() % 4 == 0);
        logits.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<_>>()
    });
    assert_all_finite_f32("verify-hidden", &{
        hidden.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<_>>()
    });
    assert_all_finite_f32("prev-hidden", &gpu.download_f32(&state.prev_hidden).expect("prev"));

    let mut dn_s = Vec::new();
    let mut dn_scales = Vec::new();
    let mut dn_conv = Vec::new();
    let mut dn_ef = Vec::new();
    for t in &slot.dn_state.s_matrices {
        dn_s.extend_from_slice(&raw_all(gpu, t));
    }
    for t in &slot.dn_state.s_scales {
        dn_scales.extend_from_slice(&raw_all(gpu, t));
    }
    for t in &slot.dn_state.conv_states {
        dn_conv.extend_from_slice(&raw_all(gpu, t));
    }
    for t in &slot.dn_state.s_ef_residual {
        dn_ef.extend_from_slice(&raw_all(gpu, t));
    }
    // DN S is raw int8 codes (naturally finite) + separate per-row F32 scales
    // (checked below), NOT blocked Q8_0: never read as f32 lanes or 34B blocks.
    assert!(!dn_s.is_empty(), "dn-s empty");
    assert_all_finite_f32("dn-scales", &gpu_download_all(gpu, &slot.dn_state.s_scales));
    assert_all_finite_f32("dn-conv", &gpu_download_all(gpu, &slot.dn_state.conv_states));
    assert_all_finite_f16("dn-ef", &dn_ef);

    let kv = trunk_kv_prefix(gpu, slot, end_pos);
    assert_q8_blocks("trunk-kv", &kv);
    assert!(kv.iter().any(|&b| b != 0), "trunk kv prefix all zero");
    let mtpkv = mtp_kv_prefix(gpu, state, end_pos);
    assert_q8_blocks("mtp-kv", &mtpkv);
    let tape = tape_prefix(gpu, state, n_verify);
    assert_all_finite_f32("tape", &{
        tape.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<_>>()
    });
    vec![logits, hidden, prev, dn_s, dn_scales, dn_conv, dn_ef, kv, mtpkv, tape]
}

fn gpu_download_all(gpu: &Gpu, ts: &[GpuTensor]) -> Vec<f32> {
    let mut out = Vec::new();
    for t in ts {
        out.extend_from_slice(&gpu.download_f32(t).expect("download f32 fam"));
    }
    out
}

fn trunk_kv_prefix(gpu: &Gpu, slot: &ModelSlot, positions: usize) -> Vec<u8> {
    let kv = &slot.kv_cache;
    let row = kv.n_kv_heads * (kv.head_dim / 32) * 34;
    let mut out = Vec::new();
    for layer in 0..kv.k_gpu.len() {
        for buf in [&kv.k_gpu[layer], &kv.v_gpu[layer]] {
            if buf.buf.size() < row {
                continue;
            }
            out.extend_from_slice(&raw_prefix(gpu, buf, positions * row));
        }
    }
    out
}

fn mtp_kv_prefix(gpu: &Gpu, state: &MtpSpecState, positions: usize) -> Vec<u8> {
    let kv = &state.mtp_kv;
    let row = kv.n_head_kv * (kv.head_dim / 32) * 34;
    let mut out = Vec::new();
    for buf in kv.inner.k_gpu.iter().chain(kv.inner.v_gpu.iter()) {
        if buf.buf.size() < row {
            continue;
        }
        out.extend_from_slice(&raw_prefix(gpu, buf, positions * row));
    }
    out
}

fn tape_prefix(gpu: &Gpu, state: &MtpSpecState, n_verify: usize) -> Vec<u8> {
    let tape = &state.trunk_gdn_tape;
    let mut out = Vec::new();
    for t in &tape.qkv_bufs {
        out.extend_from_slice(&raw_prefix(gpu, t, n_verify * tape.qkv_dim * 4));
    }
    for t in tape.alpha_bufs.iter().chain(tape.beta_bufs.iter()) {
        out.extend_from_slice(&raw_prefix(gpu, t, n_verify * tape.n_v_heads * 4));
    }
    out
}
struct Worker {
    out_dir: PathBuf,
    lines: Vec<String>,
}

impl Worker {
    fn emit(&mut self, s: String) {
        eprintln!("[vg-oracle] {s}");
        self.lines.push(s);
    }

    #[allow(clippy::too_many_arguments)]
    fn snap(
        &mut self,
        gpu: &Gpu,
        slot: &ModelSlot,
        state: &MtpSpecState,
        tag: &str,
        k: usize,
        pos: usize,
        seed: u32,
        committed: &[u32],
        accept: usize,
        advance: usize,
        drafts: usize,
        end_pos: usize,
    ) -> Vec<Vec<u8>> {
        let n_verify = drafts + 1;
        let fams = collect_fams(gpu, slot, state, n_verify, end_pos);
        let mut h = 0xcbf29ce484222325u64;
        for f in &fams {
            h ^= fnv1a(f);
            h = h.wrapping_mul(0x100000001b3);
        }
        let toks = committed.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",");
        self.emit(format!(
            "WIN tag={tag} k={k} pos={pos} seed={seed} committed={toks} \
accept={accept} advance={advance} drafts={drafts} n={n_verify} end={end_pos} h={h:016x}"
        ));
        for (name, bytes) in FAMS.iter().zip(fams.iter()) {
            std::fs::write(self.out_dir.join(format!("{tag}-{name}.bin")), bytes)
                .expect("write sidecar");
        }
        fams
    }

    fn replay(&mut self, state: &MtpSpecState, tag: &str) {
        self.emit(format!(
            "REPLAY tag={tag} mode={} captures={} replays={}",
            state.mtp_verify_graph_mode(),
            state.mtp_verify_graph_captures(),
            state.mtp_verify_graph_replays(),
        ));
    }
}

fn run_worker(arm: &str, out_dir: PathBuf) {
    let Some((model, head_path)) = model_paths() else {
        eprintln!("skipping: HIPFIRE_MTP_BYTE_IDENTITY_MODEL not set");
        return;
    };
    assert!(model.is_file(), "trunk fixture missing: {}", model.display());
    assert!(head_path.is_file(), "mtp sidecar missing: {}", head_path.display());
    std::fs::create_dir_all(&out_dir).expect("create oracle out dir");

    let mut gpu = Gpu::init().expect("Gpu::init");
    eprintln!("[vg-oracle] arm={arm} gpu={} model={}", gpu.arch, model.display());
    assert_eq!(gpu.arch.as_str(), "gfx1201", "oracle requires single-GPU gfx1201");

    let mut w = Worker { out_dir, lines: Vec::new() };
    let mut slot = load_offloaded_session(&mut gpu, &model, "vg-oracle", &mut w.lines);
    let head = mtp_head::load_mtp_head(&head_path, &mut gpu, CTX).expect("load head");
    let tokenizer = slot.load_tokenizer().expect("tokenizer");
    let prompt = build_prompt(&tokenizer);
    let n = prompt.len();
    let eos = slot.config.eos_token;
    let vocab = slot.config.vocab_size;
    let mut state =
        MtpSpecState::new_for_slot_with_kv_mode_and_verify_capacity(&mut gpu, &slot, &head, MAX_N, VERIFY_CAP, MtpKvMode::Q8)
            .expect("state");
    if let Some(cvs) = head.weights.compressed_vocab_size {
        state.mtp_scratch.ensure_compressed_logits(&mut gpu, cvs).expect("head compressed logits");
        state.ensure_compressed_lm_logits(&mut gpu, cvs).expect("batched compressed logits");
    }
    w.emit(format!("PROMPT n={n} eos={eos}"));

    // Phase A: native windows at n_verify 2/3/4, same prefilled prompt/seed.
    let seed0 = prefill_and_seed(&mut gpu, &mut slot, &head, &mut state, &prompt);
    let mut first_k3: Option<(Vec<u32>, Vec<Vec<u8>>)> = None;
    let mut seen: std::collections::BTreeSet<(usize, usize, u32)> = std::collections::BTreeSet::new();
    for k in 1..=MAX_N {
        let mut seed = prefill_and_seed(&mut gpu, &mut slot, &head, &mut state, &prompt);
        assert_eq!(seed, seed0, "k={k}: prefill seed changed after reset");
        let mut pos = n;
        let rb = state.mtp_verify_graph_replays();
        for win in 0..NATIVE_WINDOWS_PER_K {
            let r = spec_step_mtp_compressed_serial_with_k(&mut gpu, &mut slot, &head, &mut state, pos, seed, eos, k)
                .expect("native window");
            assert!(!r.hit_eos, "phase A hit EOS");
            assert_eq!(r.drafts_generated, k);
            let end = pos + r.advance;
            let tag = format!("A-k{k}-{win}");
            let fams = w.snap(&gpu, &slot, &state, &tag, k, pos, seed, &r.committed, r.accept_count, r.advance, r.drafts_generated, end);
            w.replay(&state, &tag);
            assert!(seen.insert((k, pos, seed)), "k={k}: repeated (pos, seed) input");
            if k == MAX_N && win == 0 {
                first_k3 = Some((r.committed.clone(), fams));
            }
            seed = *r.committed.last().expect("empty commit");
            pos = end;
        }
        if arm == "graph" {
            assert!(state.mtp_verify_graph_captures() > 0, "graph: no capture at k={k}");
            assert!(state.mtp_verify_graph_replays() - rb >= 2, "graph: <2 replays at k={k}");
        }
    }
    let (caps_a, reps_a) = (state.mtp_verify_graph_captures(), state.mtp_verify_graph_replays());
    if arm == "graph" {
        assert!(caps_a >= 3, "graph: only {caps_a} captures");
        assert!(reps_a >= 6, "graph: only {reps_a} replays");
    } else {
        assert_eq!(caps_a, 0, "eager: {caps_a} captures despite optout");
        assert_eq!(reps_a, 0, "eager: {reps_a} replays despite optout");
    }

    // Phase B: deterministic k=0 reference, then full/partial/zero takeovers
    // inside the 2..4-row envelope, each followed by native continuation.
    let mut toks = vec![prefill_and_seed(&mut gpu, &mut slot, &head, &mut state, &prompt)];
    assert_eq!(toks[0], seed0);
    for i in 0..REF_STEPS {
        let r = spec_step_mtp_compressed_serial_with_k(&mut gpu, &mut slot, &head, &mut state, n + i, toks[i], eos, 0)
            .expect("ref step");
        assert!(!r.hit_eos, "reference hit EOS");
        toks.push(r.committed[0]);
    }
    let seed_b = prefill_and_seed(&mut gpu, &mut slot, &head, &mut state, &prompt);
    assert_eq!(seed_b, seed0, "re-prefill must reproduce the seed");
    let wrong = |t: u32| {
        let c = if t == 0 { 1 } else { t - 1 };
        if c == eos { (c + 1) % vocab as u32 } else { c }
    };
    // (label, n_match, append_wrong): rows = matches + wrong? + bonus...
    // drafts = candidates.len(), n_verify = drafts + 1.
    let plan: [(&str, usize, bool); 3] = [("full", 2, false), ("partial", 2, true), ("zero", 0, true)];
    let mut idx = 0usize;
    for (label, nmatch, add_wrong) in plan {
        let p = n + idx;
        let mut cands: Vec<u32> = if nmatch == 0 {
            vec![wrong(toks[idx + 1])]
        } else {
            let mut c = toks[idx + 1..=idx + nmatch].to_vec();
            if add_wrong {
                c.push(wrong(toks[idx + nmatch + 1]));
            }
            c
        };
        assert!((2..=4).contains(&(cands.len() + 1)), "{label}: outside 2..4-row envelope");
        let r = spec_step_mtp_compressed_serial_with_takeover_candidates(&mut gpu, &mut slot, &head, &mut state, p, toks[idx], eos, &cands)
            .expect("takeover window");
        assert_eq!(r.accept_count, nmatch, "{label}: accept mismatch");
        assert_eq!(&r.committed[..], &toks[idx + 1..=idx + r.advance], "{label}: committed drift");
        let end = p + r.advance;
        let tag = format!("B-{label}");
        w.snap(&gpu, &slot, &state, &tag, cands.len().saturating_sub(1).min(MAX_N).max(1), p, toks[idx], &r.committed, r.accept_count, r.advance, r.drafts_generated, end);
        w.replay(&state, &tag);
        idx += r.advance;
        // Native continuation on the retained post-rollback state.
        let c = spec_step_mtp_compressed_serial_with_k(&mut gpu, &mut slot, &head, &mut state, end, toks[idx], eos, 2)
            .expect("post-takeover continuation");
        assert!(!c.hit_eos, "{label}: continuation hit EOS");
        assert_eq!(&c.committed[..], &toks[idx + 1..=idx + c.advance], "{label}: continuation drift");
        let cend = end + c.advance;
        let ctag = format!("B-{label}-cont");
        w.snap(&gpu, &slot, &state, &ctag, 2, end, toks[idx], &c.committed, c.accept_count, c.advance, c.drafts_generated, cend);
        w.replay(&state, &ctag);
        idx += c.advance;
    }

    // Phase C: reset + second generation from the same prompt/seed.
    let seed_c = prefill_and_seed(&mut gpu, &mut slot, &head, &mut state, &prompt);
    assert_eq!(seed_c, seed0, "gen2 reset must reproduce the seed");
    let (mut pos_c, mut seed_c_mut) = (n, seed_c);
    for win in 0..4 {
        let r = spec_step_mtp_compressed_serial_with_k(&mut gpu, &mut slot, &head, &mut state, pos_c, seed_c_mut, eos, MAX_N)
            .expect("gen2 window");
        assert!(!r.hit_eos, "gen2 hit EOS");
        let end = pos_c + r.advance;
        let tag = format!("C-3-{win}");
        w.snap(&gpu, &slot, &state, &tag, MAX_N, pos_c, seed_c_mut, &r.committed, r.accept_count, r.advance, r.drafts_generated, end);
        w.replay(&state, &tag);
        seed_c_mut = *r.committed.last().unwrap();
        pos_c = end;
    }

    // Phase D: unload + reload, first k=3 window must match Phase A bytes.
    let (want_toks, want_fams) = first_k3.expect("phase A recorded no k=3 window");
    free_session(&mut gpu, slot, state, head);
    let mut slot2 = load_offloaded_session(&mut gpu, &model, "vg-oracle-reload", &mut w.lines);
    let head2 = mtp_head::load_mtp_head(&head_path, &mut gpu, CTX).expect("reload head");
    let mut state2 =
        MtpSpecState::new_for_slot_with_kv_mode_and_verify_capacity(&mut gpu, &slot2, &head2, MAX_N, VERIFY_CAP, MtpKvMode::Q8)
            .expect("reload state");
    if let Some(cvs) = head2.weights.compressed_vocab_size {
        state2.mtp_scratch.ensure_compressed_logits(&mut gpu, cvs).expect("reload head compressed logits");
        state2.ensure_compressed_lm_logits(&mut gpu, cvs).expect("reload batched compressed logits");
    }
    let seed_r = prefill_and_seed(&mut gpu, &mut slot2, &head2, &mut state2, &prompt);
    assert_eq!(seed_r, seed0, "reload prefill must reproduce the seed");
    let r = spec_step_mtp_compressed_serial_with_k(&mut gpu, &mut slot2, &head2, &mut state2, n, seed_r, eos, MAX_N)
        .expect("reload window");
    assert_eq!(r.committed, want_toks, "reload committed diverged from Phase A");
    let got = collect_fams(&gpu, &slot2, &state2, r.drafts_generated + 1, n + r.advance);
    assert_eq!(got.len(), want_fams.len(), "reload family count diverged");
    for (i, (g, want)) in got.iter().zip(want_fams.iter()).enumerate() {
        assert_eq!(g, want, "reload family {} diverged from Phase A", FAMS[i]);
    }
    w.emit(format!("RELOAD committed={} match=1", r.committed.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",")));
    free_session(&mut gpu, slot2, state2, head2);

    std::fs::write(w.out_dir.join("digests.log"), w.lines.join("\n") + "\n").expect("write digests");
    eprintln!("[vg-oracle] arm={arm} DONE lines={} out={}", w.lines.len(), w.out_dir.display());
}


fn win_field<'a>(line: &'a str, key: &str) -> &'a str {
    line.split_whitespace()
        .find(|p| p.starts_with(&format!("{key}=")))
        .unwrap_or_else(|| panic!("WIN line missing {key}: {line}"))
        .split_once('=')
        .map(|(_, v)| v)
        .unwrap()
}

fn read_sidecar(dir: &Path, tag: &str, fam: &str) -> Vec<u8> {
    std::fs::read(dir.join(format!("{tag}-{fam}.bin"))).unwrap_or_else(|_| panic!("missing {tag}-{fam}.bin"))
}


fn run_orchestrator() {
    let Some(_) = model_paths() else {
        eprintln!("skipping: HIPFIRE_MTP_BYTE_IDENTITY_MODEL not set");
        return;
    };
    let exe = std::env::current_exe().expect("current_exe");
    let root = std::env::var(OUT_ROOT_ENV).map(PathBuf::from).ok();
    let mut dirs = Vec::new();
    for (arm, vg) in [("eager", "0"), ("graph", "1")] {
        let dir = unique_out_dir(root.clone(), arm);
        eprintln!("[vg-oracle-cmp] spawning {arm} arm (HIPFIRE_VERIFY_GRAPH={vg}) ...");
        let st = std::process::Command::new(&exe)
            .arg("--exact")
            .arg("mtp_verify_graph_eager_vs_replay")
            .arg("--ignored")
            .arg("--nocapture")
            .env(ARM_ENV, arm)
            .env(OUT_ENV, &dir)
            .env("HIPFIRE_VERIFY_GRAPH", vg)
            .status()
            .expect("spawn worker");
        assert!(st.success(), "{arm} worker failed with {st}");
        dirs.push((arm, dir));
    }
    let (eager_dir, graph_dir) = (&dirs[0].1, &dirs[1].1);
    let eager = std::fs::read_to_string(eager_dir.join("digests.log")).expect("read eager digests");
    let graph = std::fs::read_to_string(graph_dir.join("digests.log")).expect("read graph digests");
    let ew: Vec<_> = eager.lines().filter(|l| l.starts_with("WIN ")).collect();
    let gw: Vec<_> = graph.lines().filter(|l| l.starts_with("WIN ")).collect();
    assert_eq!(ew.len(), gw.len(), "window count diverged eager={} graph={}", ew.len(), gw.len());
    assert!(!ew.is_empty(), "no WIN lines recorded");
    for (e, g) in ew.iter().zip(&gw) {
        for key in ["tag", "k", "pos", "seed", "committed", "accept", "advance", "drafts", "n", "end"] {
            assert_eq!(win_field(e, key), win_field(g, key), "eager/graph {key}: {e} vs {g}");
        }
        let tag = win_field(e, "tag");
        for fam in FAMS {
            let eager_bytes = read_sidecar(eager_dir, tag, fam);
            let graph_bytes = read_sidecar(graph_dir, tag, fam);
            if eager_bytes != graph_bytes {
                let offset = eager_bytes.iter().zip(&graph_bytes).position(|(a, b)| a != b)
                    .unwrap_or(eager_bytes.len().min(graph_bytes.len()));
                panic!("{tag} family {fam}: eager/graph differ at byte {offset}, lengths {} vs {}",
                    eager_bytes.len(), graph_bytes.len());
            }
        }
    }

    // EF contract: ACTIVE (non-empty) on both arms under the default Q8 build.
    let ef_len = |dir: &Path, tag: &str| read_sidecar(dir, tag, "dn_ef").len();
    let first_tag = win_field(ew[0], "tag");
    let (eager_ef, graph_ef) = (ef_len(eager_dir, first_tag), ef_len(graph_dir, first_tag));
    assert_eq!(eager_ef, graph_ef, "dn_ef length diverged across arms");
    assert!(eager_ef > 0, "dn_ef empty on both arms (HIPFIRE_DN_STATE_EF=0 set?)");

    for (name, text, is_graph) in [("eager", eager.as_str(), false), ("graph", graph.as_str(), true)] {
        let host = text.lines().find(|l| l.starts_with("HOST ")).expect("HOST line");
        assert!(host.contains("host_expert_layers=16"), "{name}: {host}");
        assert!(text.lines().any(|l| l.starts_with("PLACEMENT explicit-fixed-first16")), "{name}: placement line missing");
        assert_eq!(text.lines().filter(|l| l.starts_with("RELOAD ")).count(), 1, "{name}: reload check missing");
        let reps: Vec<&str> = text.lines().filter(|l| l.starts_with("REPLAY ")).collect();
        assert!(!reps.is_empty(), "{name}: no REPLAY lines");
        let last = reps.last().unwrap();
        let get = |k: &str| -> u64 {
            last.split_whitespace().find(|p| p.starts_with(&format!("{k}="))).and_then(|p| p.split_once('=').unwrap().1.parse().ok()).unwrap_or(u64::MAX)
        };
        let (caps, reps_n) = (get("captures"), get("replays"));
        if is_graph {
            assert!(caps >= 3, "graph: only {caps} captures");
            assert!(reps_n >= 6, "graph: only {reps_n} replays");
        } else {
            assert_eq!(caps, 0, "eager: {caps} captures despite optout");
            assert_eq!(reps_n, 0, "eager: {reps_n} replays despite optout");
        }
    }
    eprintln!(
        "[vg-oracle-cmp] PASS windows={} eager={} graph={} (tokens exact, all 10 families byte-exact, replay proven, reload match)",
        ew.len(),
        eager_dir.display(),
        graph_dir.display()
    );
}

#[test]
#[ignore = "requires real HIP GPU (gfx1201) + HIPFIRE_MTP_BYTE_IDENTITY_MODEL (qwen3.6-35b-a3b.mq4p) + MTP sidecar"]
fn mtp_verify_graph_eager_vs_replay() {
    if let Ok(arm) = std::env::var(ARM_ENV) {
        let out = std::env::var(OUT_ENV).map(PathBuf::from).unwrap_or_else(|_| {
            unique_out_dir(std::env::var(OUT_ROOT_ENV).map(PathBuf::from).ok(), &arm)
        });
        run_worker(&arm, out);
    } else {
        run_orchestrator();
    }
}
