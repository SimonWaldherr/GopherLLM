//! Exercises the whole FFI round trip against a real GGUF this test
//! generates on the fly (via `go run ./cmd/gopherllm-synth-testmodel`), so
//! this proves the crate actually links and calls into libgopherllm, not
//! just that it compiles -- with no binary model checked into the repo.
//! Run with:
//!   GOPHERLLM_LIB_DIR=<repo>/build/capi cargo test
use gopherllm::{Engine, GenerationOptions, LoadOptions};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

fn repo_root() -> PathBuf {
    // This crate lives at <repo>/bindings/rust/gopherllm.
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// Materializes cmd/gopherllm-synth-testmodel's deterministic fixture once
/// per test-binary run and reuses it across every #[test] in this file.
fn synth_model_path() -> &'static Path {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let root = repo_root();
        let out = std::env::temp_dir().join("gopherllm-rust-binding-test-model.gguf");
        let status = Command::new("go")
            .args(["run", "./cmd/gopherllm-synth-testmodel", "-out"])
            .arg(&out)
            .current_dir(&root)
            .env("GO111MODULE", "on")
            .status()
            .expect("failed to run `go run ./cmd/gopherllm-synth-testmodel` -- is Go installed and on PATH?");
        assert!(status.success(), "gopherllm-synth-testmodel exited with {status}");
        out
    })
    .as_path()
}

#[test]
fn load_generate_and_stream_round_trip() {
    let model = synth_model_path();

    let engine = Engine::new();
    assert!(!engine.is_loaded());
    engine.load(model.to_str().unwrap(), &LoadOptions::default()).expect("load");
    assert!(engine.is_loaded());
    assert_eq!(engine.model_name(), "gopherllm-synth-testmodel");

    let info = engine.info().expect("info");
    assert_eq!(info.architecture, "llama");
    assert_eq!(info.vocab_size, 98);

    // Fixed weights (baked into the generator) and a fixed seed make this
    // fully deterministic.
    let opts = GenerationOptions {
        max_tokens: Some(24),
        seed: Some(1),
        ..Default::default()
    };
    let text = engine.generate("Hallo", &opts).expect("generate");
    assert_eq!(text, "$(F=bBz\".C\\[}3WeO`>i[\"cZ");

    let mut deltas = Vec::new();
    let result = engine
        .generate_stream("Hallo", &opts, |d| deltas.push(d.to_string()))
        .expect("generate_stream");
    assert_eq!(deltas.len(), 24, "every token is single-byte-ASCII, so one delta per token");
    assert_eq!(deltas.concat(), text);
    assert_eq!(result.finish_reason, "length");
    assert_eq!(result.generated_tokens, 24);

    engine.unload().expect("unload");
    assert!(!engine.is_loaded());
}

#[test]
fn double_load_is_rejected() {
    let model = synth_model_path();
    let engine = Engine::new();
    engine.load(model.to_str().unwrap(), &LoadOptions::default()).expect("first load");
    let err = engine.load(model.to_str().unwrap(), &LoadOptions::default()).unwrap_err();
    assert!(err.0.to_lowercase().contains("already loaded"), "unexpected error: {}", err.0);
}

#[test]
fn cancel_before_load_and_unload_of_unloaded_engine_are_no_ops() {
    let engine = Engine::new();
    engine.cancel();
    engine.unload().expect("unload of an unloaded engine should be a no-op, not an error");
    assert_eq!(engine.model_name(), "");
}
