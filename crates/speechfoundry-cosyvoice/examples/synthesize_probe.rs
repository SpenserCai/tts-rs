//! Run pretrained CosyVoice synthesis from saved public prompt features.
use candle::{DType, Device};
use serde::Serialize;
use speechfoundry_cosyvoice::{CosyVoice3, PromptFeatures, SynthesisOptions};

#[derive(Serialize)]
struct Output {
    text: String,
    sampling: String,
    max_tokens: usize,
    effective_max_tokens: usize,
    termination_reason: &'static str,
    sample_rate: u32,
    tokens: Vec<u32>,
    samples: Vec<f32>,
    elapsed_seconds: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if !(4..=7).contains(&args.len()) {
        return Err(
            "usage: synthesize_probe MODEL_DIR PROMPT_JSON OUTPUT_JSON [greedy|sample] [TEXT] [MAX_TOKENS]"
                .into(),
        );
    }
    let sampling = args.get(4).and_then(|x| x.to_str()).unwrap_or("greedy");
    let text = args.get(5).and_then(|x| x.to_str()).unwrap_or("你好。");
    let max_tokens = if let Some(value) = args.get(6) {
        value.to_str().ok_or("MAX_TOKENS must be UTF-8")?.parse()?
    } else {
        128
    };
    if !matches!(sampling, "greedy" | "sample") {
        return Err("sampling must be greedy or sample".into());
    }
    let prompt: PromptFeatures = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let started = std::time::Instant::now();
    eprintln!("loading Candle model");
    let mut model = CosyVoice3::load(&args[1], &Device::Cpu, DType::F32)?;
    let mut options = SynthesisOptions::default();
    if sampling == "greedy" {
        options.sampling.temperature = 0.0;
    }
    options.sampling.max_tokens = max_tokens;
    let tokenizer =
        tokenizers::Tokenizer::from_file(std::path::Path::new(&args[1]).join("tokenizer.json"))
            .map_err(|error| error.to_string())?;
    let text_tokens = tokenizer
        .encode(text, false)
        .map_err(|error| error.to_string())?
        .len();
    let effective_max_tokens = (text_tokens * 20).min(max_tokens);
    eprintln!("generating {sampling} probe");
    let tokens = model.generate_tokens(text, &prompt, &options)?;
    let samples = model.decode_tokens(&tokens, &prompt, options.flow_steps)?;
    // The model returns only speech tokens. An early return is caused by a
    // sampled special stop token; the API does not expose its numeric ID.
    let termination_reason = if tokens.len() < effective_max_tokens {
        "model_stop_token"
    } else {
        "token_budget"
    };
    let output = Output {
        text: text.to_owned(),
        sampling: sampling.to_owned(),
        max_tokens,
        effective_max_tokens,
        termination_reason,
        sample_rate: model.sample_rate(),
        tokens,
        samples,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    };
    std::fs::write(&args[3], serde_json::to_vec(&output)?)?;
    eprintln!(
        "{} samples, {:.2} seconds including load",
        output.samples.len(),
        output.elapsed_seconds
    );
    Ok(())
}
