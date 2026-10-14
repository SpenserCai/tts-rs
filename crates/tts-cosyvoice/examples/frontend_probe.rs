//! Run the optional real ONNX frontend against a JSON waveform fixture.
#[cfg(feature = "onnx")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candle::Device;
    use serde::Deserialize;
    #[derive(Deserialize)]
    struct Input {
        samples: Vec<f32>,
        sample_rate: u32,
        transcript: String,
    }
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 4 {
        return Err("usage: frontend_probe MODEL_DIR INPUT_JSON OUTPUT_JSON".into());
    }
    let input: Input = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let frontend = tts_cosyvoice::CosyVoice3Frontend::load(&args[1], &Device::Cpu)?;
    let prompt = frontend.extract_prompt(&input.samples, input.sample_rate, input.transcript)?;
    std::fs::write(&args[3], serde_json::to_vec_pretty(&prompt)?)?;
    Ok(())
}
#[cfg(not(feature = "onnx"))]
fn main() {
    eprintln!("frontend_probe requires --features onnx");
    std::process::exit(2);
}
