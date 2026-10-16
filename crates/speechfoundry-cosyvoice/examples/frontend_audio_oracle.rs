//! Export resampled PCM and mean-normalized CAMPPlus features without ONNX models.
use candle::{Device, Tensor};
use serde::Deserialize;
use serde_json::json;
use speechfoundry_cosyvoice::frontend::{KaldiFbank, MelSpectrogram, resample};
use std::error::Error;

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
    let pcm_24k = resample(&pcm, input.sample_rate, 24_000)?;
    let mel =
        MelSpectrogram::new_cosyvoice_speech_feat(&Device::Cpu)?.forward_cosyvoice(&pcm_24k)?;
    let whisper = MelSpectrogram::new_whisper_format(&Device::Cpu)?.forward_whisper(&pcm_16k)?;
    let floor = (whisper.flatten_all()?.max(0)? - 8.0)?;
    let whisper = ((whisper.broadcast_maximum(&floor)? + 4.0)? / 4.0)?;
    let mut tensors = serde_json::Map::new();
    for (key, value) in [
        ("pcm_16k", pcm_16k),
        ("fbank", fbank),
        ("normalized", normalized),
        ("mel", mel),
        ("whisper", whisper),
    ] {
        tensors.insert(key.into(), json!({"shape": value.dims(), "dtype": "float32", "data": value.flatten_all()?.to_vec1::<f32>()?}));
    }
    std::fs::write(
        &args[1],
        serde_json::to_vec(&json!({"schema_version": 1, "tensors": tensors}))?,
    )?;
    Ok(())
}
