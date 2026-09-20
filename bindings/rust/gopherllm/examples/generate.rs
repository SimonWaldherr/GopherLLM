//! Run with: GOPHERLLM_LIB_DIR=../../../build/capi cargo run --example generate -- <model.gguf> "<prompt>"
use gopherllm::{Engine, GenerationOptions, LoadOptions};
use std::env;

fn main() -> Result<(), gopherllm::Error> {
    let mut args = env::args().skip(1);
    let model_path = args.next().expect("usage: generate <model.gguf> [prompt]");
    let prompt = args.next().unwrap_or_else(|| "Hello!".to_string());

    let engine = Engine::new();
    engine.load(&model_path, &LoadOptions::default())?;
    println!("loaded {} ({})", engine.model_name(), engine.info()?.architecture);

    print!("streaming: ");
    let result = engine.generate_stream(
        &prompt,
        &GenerationOptions {
            max_tokens: Some(64),
            ..Default::default()
        },
        |delta| print!("{delta}"),
    )?;
    println!("\n[{} tokens, finish_reason={}]", result.generated_tokens, result.finish_reason);
    Ok(())
}
