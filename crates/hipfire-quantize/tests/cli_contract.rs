// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Black-box CLI contract of `hipfire-quantize` on tiny synthetic CPU
//! checkpoints: what gets written, and which invocations must fail.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use hipfire_runtime::hfq::HfqFile;
use safetensors::tensor::{Dtype, TensorView};

const MQ4G256: u8 = 13;
const MQ6G256: u8 = 15;
const Q8F16: u8 = 3;

fn bf16_bytes(n: usize, seed: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|i| {
            let v = (((i * 31 + seed * 17) % 97) as f32 - 48.0) * 0.01;
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}

/// A checkpoint dir with `config.json` and one BF16 shard of `[rows, 256]`
/// (or `[256]` for names containing `norm`) tensors. Dory nGPT vectors
/// (sqk/s_gate/suv_*/attn_alpha/mlp_alpha/logit_scale) are 1-D by
/// construction, so they ride the `[256]` row too.
fn checkpoint(dir: &Path, config: serde_json::Value, names: &[&str]) -> PathBuf {
    fn is_1d(name: &str) -> bool {
        name.contains("norm")
            || name.ends_with(".sqk")
            || name.ends_with(".s_gate")
            || name.ends_with(".suv_gate")
            || name.ends_with(".suv_up")
            || name.ends_with(".attn_alpha")
            || name.ends_with(".mlp_alpha")
            || name.ends_with(".logit_scale")
    }
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let data: Vec<(String, Vec<usize>, Vec<u8>)> = names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let shape = if is_1d(name) { vec![256] } else { vec![4, 256] };
            let n: usize = shape.iter().product();
            (name.to_string(), shape, bf16_bytes(n, i))
        })
        .collect();
    let views: HashMap<String, TensorView> = data
        .iter()
        .map(|(n, s, b)| {
            (
                n.clone(),
                TensorView::new(Dtype::BF16, s.clone(), b).unwrap(),
            )
        })
        .collect();
    safetensors::serialize_to_file(&views, None, &dir.join("model.safetensors")).unwrap();
    dir.to_path_buf()
}

fn llama(dir: &Path) -> PathBuf {
    checkpoint(
        dir,
        serde_json::json!({"model_type": "llama", "num_hidden_layers": 1}),
        &[
            "model.layers.0.self_attn.q_proj.weight",
            "model.norm.weight",
        ],
    )
}

fn quantize(args: &[&str], envs: &[(&str, &str)]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_hipfire-quantize"));
    cmd.args(args).env("HIPFIRE_QUANT_THREADS", "2");
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd.output().unwrap()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn qt(hfq: &Path, name: &str) -> u8 {
    HfqFile::open(hfq)
        .unwrap()
        .find_tensor_info(name)
        .unwrap()
        .quant_type
}

fn cohere(dir: &Path) -> PathBuf {
    checkpoint(
        dir,
        serde_json::json!({"model_type": "cohere2_moe", "num_hidden_layers": 1}),
        &[
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.input_layernorm.weight",
        ],
    )
}

#[test]
fn cohere2_expert_tier_follows_format() {
    let tmp = tempfile::tempdir().unwrap();
    let input = cohere(&tmp.path().join("ck"));
    let expert = "model.layers.0.mlp.experts.0.gate_proj.weight";
    for (format, want) in [
        ("mq4", MQ4G256),
        ("mq4v2", MQ4G256),
        ("mq4v1", MQ4G256),
        ("mq6", MQ6G256),
        ("q8", Q8F16),
    ] {
        let out_path = tmp.path().join(format!("{format}.hfq"));
        let out = quantize(
            &[
                "--input",
                input.to_str().unwrap(),
                "--output",
                out_path.to_str().unwrap(),
                "--format",
                format,
            ],
            &[],
        );
        assert!(out.status.success(), "{format}: {}", stderr(&out));
        assert_eq!(
            qt(&out_path, expert),
            want,
            "--format {format} expert dtype"
        );
        assert_eq!(
            qt(&out_path, "model.layers.0.self_attn.q_proj.weight"),
            Q8F16
        );
    }
    // A format the cohere2 handler has no tier for is an error, not Q8.
    let out_path = tmp.path().join("hfq4.hfq");
    let out = quantize(
        &[
            "--input",
            input.to_str().unwrap(),
            "--output",
            out_path.to_str().unwrap(),
            "--format",
            "hfq4",
        ],
        &[],
    );
    assert_eq!(out.status.code(), Some(2), "{}", stderr(&out));
    assert!(!out_path.exists());
}

#[test]
fn gemma4_unified_with_skipped_vision_does_not_claim_vision() {
    let tmp = tempfile::tempdir().unwrap();
    let input = checkpoint(
        &tmp.path().join("ck"),
        serde_json::json!({"model_type": "gemma4", "num_hidden_layers": 1}),
        &[
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.vision_tower.encoder.layers.0.mlp.fc1.weight",
        ],
    );
    let out_path = tmp.path().join("g.hfq");
    let out = quantize(
        &[
            "--input",
            input.to_str().unwrap(),
            "--output",
            out_path.to_str().unwrap(),
            "--format",
            "q8",
            "--include-vision",
        ],
        &[],
    );
    assert!(out.status.success(), "{}", stderr(&out));
    let hfq = HfqFile::open(&out_path).unwrap();
    assert!(hfq
        .find_tensor_info("model.vision_tower.encoder.layers.0.mlp.fc1.weight")
        .is_none());
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).unwrap();
    assert_ne!(meta.get("has_vision"), Some(&serde_json::json!(true)));
}

#[test]
fn unusable_spill_dir_fails_without_output() {
    let tmp = tempfile::tempdir().unwrap();
    let input = llama(&tmp.path().join("ck"));
    let out_path = tmp.path().join("o.hfq");
    let out = quantize(
        &[
            "--input",
            input.to_str().unwrap(),
            "--output",
            out_path.to_str().unwrap(),
            "--format",
            "q8",
        ],
        &[(
            "HIPFIRE_SPILL_DIR",
            tmp.path().join("missing/dir").to_str().unwrap(),
        )],
    );
    assert_eq!(out.status.code(), Some(2), "{}", stderr(&out));
    assert!(stderr(&out).contains("error: cannot create tensor spill file"));
    assert!(!out_path.exists());
}

#[cfg(unix)]
#[test]
fn failed_write_leaves_existing_output_untouched() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let input = llama(&tmp.path().join("ck"));
    let out_dir = tmp.path().join("ro");
    std::fs::create_dir(&out_dir).unwrap();
    let out_path = out_dir.join("o.hfq");
    std::fs::write(&out_path, b"previous good artifact").unwrap();
    std::fs::set_permissions(&out_dir, std::fs::Permissions::from_mode(0o555)).unwrap();
    if std::fs::write(out_dir.join("probe"), b"").is_ok() {
        return; // running as root: permissions cannot inject the failure
    }
    let spill = tmp.path().join("spill");
    std::fs::create_dir(&spill).unwrap();
    let out = quantize(
        &[
            "--input",
            input.to_str().unwrap(),
            "--output",
            out_path.to_str().unwrap(),
            "--format",
            "q8",
        ],
        &[("HIPFIRE_SPILL_DIR", spill.to_str().unwrap())],
    );
    std::fs::set_permissions(&out_dir, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert_eq!(out.status.code(), Some(2), "{}", stderr(&out));
    assert_eq!(std::fs::read(&out_path).unwrap(), b"previous good artifact");
    assert_eq!(
        std::fs::read_dir(&out_dir).unwrap().count(),
        1,
        "temp file left behind"
    );
}

#[test]
fn unknown_format_is_an_error() {
    let tmp = tempfile::tempdir().unwrap();
    let input = llama(&tmp.path().join("ck"));
    for format in ["mq4v3", "tq2"] {
        let out_path = tmp.path().join("o.hfq");
        let out = quantize(
            &[
                "--input",
                input.to_str().unwrap(),
                "--output",
                out_path.to_str().unwrap(),
                "--format",
                format,
            ],
            &[],
        );
        assert_eq!(out.status.code(), Some(2), "{format}: {}", stderr(&out));
        assert!(!out_path.exists());
    }
}

#[test]
fn documented_formats_are_accepted() {
    // docs/QUANTIZE.md format table plus scripted spellings.
    let tmp = tempfile::tempdir().unwrap();
    let input = llama(&tmp.path().join("ck"));
    for format in [
        "mq4", "mq4v2", "mq4v1", "mq4g256", "magnum", "mq4c", "mq6", "mq6v2", "mq5v2", "mq3v2",
        "mq2v2", "hf4", "hfq4", "hfq4g256", "hf6", "hfq6", "hfq6g256", "q8", "q8f16", "q4f16",
        "mq3", "hfp4", "mfp4", "mfp4e8", "f16", "bf16", "oracle",
    ] {
        let out_path = tmp.path().join(format!("{format}.hfq"));
        let out = quantize(
            &[
                "--input",
                input.to_str().unwrap(),
                "--output",
                out_path.to_str().unwrap(),
                "--format",
                format,
            ],
            &[],
        );
        assert!(out.status.success(), "--format {format}: {}", stderr(&out));
    }
}

#[test]
fn provenance_flags_are_recorded() {
    let tmp = tempfile::tempdir().unwrap();
    let input = llama(&tmp.path().join("ck"));
    let out_path = tmp.path().join("o.hfq");
    let out = quantize(
        &[
            "--input",
            input.to_str().unwrap(),
            "--output",
            out_path.to_str().unwrap(),
            "--format",
            "q8",
            "--source-url",
            "https://huggingface.co/org/model",
            "--license",
            "Apache-2.0",
        ],
        &[],
    );
    assert!(out.status.success(), "{}", stderr(&out));
    let meta: serde_json::Value =
        serde_json::from_str(&HfqFile::open(&out_path).unwrap().metadata_json).unwrap();
    let prov = &meta["hipfire_provenance"];
    assert_eq!(prov["source_url"], "https://huggingface.co/org/model");
    assert_eq!(prov["license"], "Apache-2.0");
}

#[test]
fn flags_a_route_never_reads_are_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let o = tmp.path().join("o.hfq");
    let o = o.to_str().unwrap();
    for argv in [
        &[
            "--qwen4-flash-next",
            "--input",
            "x",
            "--output",
            o,
            "--tier",
            "xt",
        ][..],
        &[
            "--format",
            "maple",
            "--input",
            "x",
            "--output",
            o,
            "--mq4v2-symmetric",
        ][..],
        &[
            "--flux-pipe",
            "p",
            "--output",
            o,
            "--fixed-tier",
            "lm_head:q8",
        ][..],
        &["--input", "x", "--output", o, "--head-only"][..],
    ] {
        let out = quantize(argv, &[]);
        assert_eq!(out.status.code(), Some(2), "{argv:?}: {}", stderr(&out));
        assert!(
            stderr(&out).contains("has no effect with"),
            "{argv:?}: {}",
            stderr(&out)
        );
    }
    // Typos in --kmap-mode no longer fall back to `alternating`.
    let out = quantize(&["--input", "x", "--output", o, "--kmap-mode", "ful"], &[]);
    assert_eq!(out.status.code(), Some(2));
}

const MQ4G256V2: u8 = 44;
const F32_QT: u8 = 2;

fn dory(dir: &Path) -> PathBuf {
    // Minimal arch-19 census: embed table, untied head, attention
    // q/k/v/o/g projections, MLP gate/up/down projections, and one vector
    // of each nGPT class. K=256 keeps every matrix on the G256 encoder path.
    checkpoint(
        dir,
        serde_json::json!({"model_type": "dory", "num_hidden_layers": 2}),
        &[
            "backbone.embeddings.weight",
            "lm_head.weight",
            "lm_head.logit_scale",
            "backbone.layers.0.mixer.q_proj.weight",
            "backbone.layers.0.mixer.k_proj.weight",
            "backbone.layers.0.mixer.v_proj.weight",
            "backbone.layers.0.mixer.o_proj.weight",
            "backbone.layers.0.mixer.g_proj.weight",
            "backbone.layers.1.mixer.gate_proj.weight",
            "backbone.layers.1.mixer.up_proj.weight",
            "backbone.layers.1.mixer.down_proj.weight",
            "backbone.layers.0.mixer.sqk",
            "backbone.layers.0.attn_alpha",
            "backbone.layers.1.mixer.suv_gate",
            "backbone.layers.1.mlp_alpha",
        ],
    )
}

/// Dory (arch 19) end-to-end disposition: exact nGPT vectors stay F32,
/// embed + untied head hold Q8, projections follow `--format`.
#[test]
fn dory_disposition_mq4_and_q8() {
    let tmp = tempfile::tempdir().unwrap();
    let input = dory(&tmp.path().join("dory"));
    let source_bytes = std::fs::read(input.join("model.safetensors")).unwrap();
    let source = safetensors::SafeTensors::deserialize(&source_bytes).unwrap();
    for (format, proj_qt) in [("mq4", MQ4G256V2), ("q8", Q8F16)] {
        let out_path = tmp.path().join(format!("dory-{format}.hfq"));
        let out = quantize(
            &[
                "--input",
                input.to_str().unwrap(),
                "--output",
                out_path.to_str().unwrap(),
                "--format",
                format,
            ],
            &[],
        );
        assert!(out.status.success(), "{format}: {}", stderr(&out));
        let hfq = HfqFile::open(&out_path).unwrap();
        assert_eq!(hfq.arch_id, 19, "{format}: arch_id");
        // Exact scales: lossless F32, never quantized or dropped.
        for name in [
            "lm_head.logit_scale",
            "backbone.layers.0.mixer.sqk",
            "backbone.layers.0.attn_alpha",
            "backbone.layers.1.mixer.suv_gate",
            "backbone.layers.1.mlp_alpha",
        ] {
            assert_eq!(qt(&out_path, name), F32_QT, "{format}: {name} must be F32");
            let original = source.tensor(name).unwrap();
            let expected: Vec<u8> = original
                .data()
                .chunks_exact(2)
                .flat_map(|value| {
                    ((u16::from_le_bytes([value[0], value[1]]) as u32) << 16).to_le_bytes()
                })
                .collect();
            let (_, actual) = hfq.tensor_data(name).unwrap();
            assert_eq!(
                actual, expected,
                "{format}: {name} must preserve every scale bit"
            );
        }
        // Lookup/output holds.
        assert_eq!(
            qt(&out_path, "backbone.embeddings.weight"),
            Q8F16,
            "{format}: embed"
        );
        assert_eq!(qt(&out_path, "lm_head.weight"), Q8F16, "{format}: head");
        // Projections follow the format body.
        assert_eq!(
            qt(&out_path, "backbone.layers.0.mixer.q_proj.weight"),
            proj_qt,
            "{format}: q_proj"
        );
        assert_eq!(
            qt(&out_path, "backbone.layers.1.mixer.gate_proj.weight"),
            proj_qt,
            "{format}: gate_proj"
        );
        // Names are verbatim HF; the artifact census closes at 15 tensors.
        assert_eq!(hfq.tensors().len(), 15, "{format}: census");
    }
}

const MQ3G256V2: u8 = 49;
const MQ5G256V2: u8 = 48;
const MQ6G256V2: u8 = 47;
const MQ8G256: u8 = 14;

/// Requested Dory body formats reach their wire qt while the exact-F32
/// vectors and the Q8 embed/head holds stay put. MQ2V2 is refused on Dory
/// (catastrophic quality; covered by `dory_mq2v2_is_refused` below) so it
/// has no row here.
#[test]
fn dory_requested_body_formats_reach_wire_qt() {
    let tmp = tempfile::tempdir().unwrap();
    let input = dory(&tmp.path().join("dory-v2"));
    // Explicit V2 spellings plus the already-supported mq8 body; the legacy
    // mq4 alias still means qt44 (covered by dory_disposition_mq4_and_q8).
    for (format, proj_qt) in [
        ("mq3v2", MQ3G256V2),
        ("mq4v2", MQ4G256V2),
        ("mq5v2", MQ5G256V2),
        ("mq6v2", MQ6G256V2),
        ("mq8", MQ8G256),
    ] {
        let out_path = tmp.path().join(format!("dory-{format}.hfq"));
        let out = quantize(
            &[
                "--input",
                input.to_str().unwrap(),
                "--output",
                out_path.to_str().unwrap(),
                "--format",
                format,
            ],
            &[],
        );
        assert!(out.status.success(), "{format}: {}", stderr(&out));
        let hfq = HfqFile::open(&out_path).unwrap();
        assert_eq!(hfq.arch_id, 19, "{format}: arch_id");
        assert_eq!(
            qt(&out_path, "backbone.layers.0.mixer.q_proj.weight"),
            proj_qt,
            "{format}: q_proj"
        );
        assert_eq!(
            qt(&out_path, "backbone.layers.1.mixer.gate_proj.weight"),
            proj_qt,
            "{format}: gate_proj"
        );
        assert_eq!(
            qt(&out_path, "backbone.embeddings.weight"),
            Q8F16,
            "{format}: embed holds Q8"
        );
        assert_eq!(
            qt(&out_path, "lm_head.weight"),
            Q8F16,
            "{format}: head holds Q8"
        );
        for name in [
            "lm_head.logit_scale",
            "backbone.layers.0.mixer.sqk",
            "backbone.layers.0.attn_alpha",
            "backbone.layers.1.mixer.suv_gate",
            "backbone.layers.1.mlp_alpha",
        ] {
            assert_eq!(qt(&out_path, name), F32_QT, "{format}: {name} must be F32");
        }
        assert_eq!(hfq.tensors().len(), 15, "{format}: census");
    }
}

/// Dory refuses MQ2V2 at export time (catastrophic quality on arch 19):
/// no partial artifact is written, and the error names a supported tier.
/// The global `--format mq2v2` allowlist is unchanged for other models.
#[test]
fn dory_mq2v2_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let input = dory(&tmp.path().join("dory-mq2v2-refused"));
    for format in ["mq2v2", "mq2g256v2"] {
        let out_path = tmp.path().join(format!("dory-{format}.hfq"));
        let out = quantize(
            &[
                "--input",
                input.to_str().unwrap(),
                "--output",
                out_path.to_str().unwrap(),
                "--format",
                format,
            ],
            &[],
        );
        assert_eq!(out.status.code(), Some(2), "{format}: {}", stderr(&out));
        assert!(
            stderr(&out).contains("mq2v2 is not supported for Dory"),
            "{format}: {}",
            stderr(&out)
        );
        assert!(stderr(&out).contains("mq3v2"), "{format}: {}", stderr(&out));
        assert!(!out_path.exists(), "{format}: partial artifact written");
    }
}

/// Dory XT tier keeps embed Q8 and exact-F32 vectors, but the untied head
/// follows the body format so XT stays distinguishable from Base.
#[test]
fn dory_xt_head_follows_body_format() {
    let tmp = tempfile::tempdir().unwrap();
    let input = dory(&tmp.path().join("dory-xt"));
    for (format, proj_qt) in [
        ("mq4v2", MQ4G256V2),
        ("mq3v2", MQ3G256V2),
        ("mq5v2", MQ5G256V2),
        ("mq6v2", MQ6G256V2),
        ("mq8", MQ8G256),
    ] {
        let tag = format!("xt-{format}");
        let out_path = tmp.path().join(format!("dory-{tag}.hfq"));
        let out = quantize(
            &[
                "--input",
                input.to_str().unwrap(),
                "--output",
                out_path.to_str().unwrap(),
                "--format",
                format,
                "--tier",
                "xt",
            ],
            &[],
        );
        assert!(out.status.success(), "{tag}: {}", stderr(&out));
        let hfq = HfqFile::open(&out_path).unwrap();
        assert_eq!(hfq.arch_id, 19, "{tag}: arch_id");
        assert_eq!(
            qt(&out_path, "lm_head.weight"),
            proj_qt,
            "{tag}: head follows body"
        );
        assert_eq!(
            qt(&out_path, "backbone.embeddings.weight"),
            Q8F16,
            "{tag}: embed holds Q8"
        );
        assert_eq!(
            qt(&out_path, "backbone.layers.0.mixer.q_proj.weight"),
            proj_qt,
            "{tag}: q_proj"
        );
        for name in [
            "lm_head.logit_scale",
            "backbone.layers.0.mixer.sqk",
            "backbone.layers.0.attn_alpha",
            "backbone.layers.1.mixer.suv_gate",
            "backbone.layers.1.mlp_alpha",
        ] {
            assert_eq!(qt(&out_path, name), F32_QT, "{tag}: {name} must be F32");
        }
        assert_eq!(hfq.tensors().len(), 15, "{tag}: census");
    }
}

/// Distinct Dory Pro: embed/head plus every o_proj and down_proj hold Q8
/// while q/k/v/g/gate/up follow the body format; XT and Base are unchanged.
#[test]
fn dory_pro_holds_o_down_q8() {
    let tmp = tempfile::tempdir().unwrap();
    let input = dory(&tmp.path().join("dory-pro"));
    for (tier, o_qt, down_qt, head_qt) in [
        ("pro", Q8F16, Q8F16, Q8F16),
        ("base", MQ4G256V2, MQ4G256V2, Q8F16),
    ] {
        let tag = format!("{tier}-mq4v2");
        let out_path = tmp.path().join(format!("dory-{tag}.hfq"));
        let out = quantize(
            &[
                "--input",
                input.to_str().unwrap(),
                "--output",
                out_path.to_str().unwrap(),
                "--format",
                "mq4v2",
                "--tier",
                tier,
            ],
            &[],
        );
        assert!(out.status.success(), "{tag}: {}", stderr(&out));
        let hfq = HfqFile::open(&out_path).unwrap();
        assert_eq!(hfq.arch_id, 19, "{tag}: arch_id");
        // Residual-write projections: Q8 only under Pro.
        assert_eq!(
            qt(&out_path, "backbone.layers.0.mixer.o_proj.weight"),
            o_qt,
            "{tag}: o_proj"
        );
        assert_eq!(
            qt(&out_path, "backbone.layers.1.mixer.down_proj.weight"),
            down_qt,
            "{tag}: down_proj"
        );
        // Neighbors always follow the body format.
        for name in [
            "backbone.layers.0.mixer.q_proj.weight",
            "backbone.layers.0.mixer.k_proj.weight",
            "backbone.layers.0.mixer.v_proj.weight",
            "backbone.layers.0.mixer.g_proj.weight",
            "backbone.layers.1.mixer.gate_proj.weight",
            "backbone.layers.1.mixer.up_proj.weight",
        ] {
            assert_eq!(qt(&out_path, name), MQ4G256V2, "{tag}: {name} follows body");
        }
        assert_eq!(
            qt(&out_path, "backbone.embeddings.weight"),
            Q8F16,
            "{tag}: embed holds Q8"
        );
        assert_eq!(qt(&out_path, "lm_head.weight"), head_qt, "{tag}: head");
        for name in [
            "lm_head.logit_scale",
            "backbone.layers.0.mixer.sqk",
            "backbone.layers.0.attn_alpha",
            "backbone.layers.1.mixer.suv_gate",
            "backbone.layers.1.mlp_alpha",
        ] {
            assert_eq!(qt(&out_path, name), F32_QT, "{tag}: {name} must be F32");
        }
        assert_eq!(hfq.tensors().len(), 15, "{tag}: census");
    }
}
