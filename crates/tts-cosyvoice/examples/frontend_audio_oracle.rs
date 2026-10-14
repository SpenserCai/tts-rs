//! Export resampled PCM and mean-normalized CAMPPlus features without ONNX models.
use candle::{Device, Tensor};
use serde::Deserialize;
use serde_json::json;
use std::error::Error;
use tts_cosyvoice::frontend::{KaldiFbank, resample};

#[derive(Deserialize)]
struct Input {
    samples: Vec<f32>,
    sample_rate: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: frontend_audio_oracle INPUT_JSON OUTPUT_JSON".into());
    }
    let input: Input = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    if input.samples.is_empty() || input.samples.iter().any(|value| !value.is_finite()) {
        return Err("expected nonempty finite mono PCM".into());
    }
    let samples = input.samples.len();
    let pcm = Tensor::from_vec(input.samples, samples, &Device::Cpu)?;
    let pcm_16k = resample(&pcm, input.sample_rate, 16_000)?;
    let fbank = KaldiFbank::new_for_campplus().forward(&pcm_16k)?;
    let normalized = fbank.broadcast_sub(&fbank.mean(0)?)?;
    let mut tensors = serde_json::Map::new();
    for (key, value) in [
        ("pcm_16k", pcm_16k),
        ("fbank", fbank),
        ("normalized", normalized),
    ] {
        tensors.insert(key.into(), json!({"shape": value.dims(), "dtype": "float32", "data": value.flatten_all()?.to_vec1::<f32>()?}));
    }
    std::fs::write(
        &args[1],
        serde_json::to_vec(&json!({"schema_version": 1, "tensors": tensors}))?,
    )?;
    Ok(())
}
