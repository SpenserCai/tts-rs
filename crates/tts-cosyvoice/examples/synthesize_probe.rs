//! Run a short, greedy pretrained CosyVoice synthesis from saved prompt features.
use candle::{DType, Device};
use serde::Serialize;
use tts_cosyvoice::{CosyVoice3, PromptFeatures, SynthesisOptions};

#[derive(Serialize)]
struct Output {
    sample_rate: u32,
    samples: Vec<f32>,
    elapsed_seconds: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 4 {
        return Err("usage: synthesize_probe MODEL_DIR PROMPT_JSON OUTPUT_JSON".into());
    }
    let prompt: PromptFeatures = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let started = std::time::Instant::now();
    eprintln!("loading Candle model");
    let mut model = CosyVoice3::load(&args[1], &Device::Cpu, DType::F32)?;
    let mut options = SynthesisOptions::default();
    options.sampling.temperature = 0.0;
    options.sampling.max_tokens = 32;
    eprintln!("generating short greedy probe");
    let samples = model.synthesize("你好。", &prompt, &options)?;
    let output = Output {
        sample_rate: model.sample_rate(),
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
