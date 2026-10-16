//! Run Qwen's 24 kHz speaker log-mel frontend on a mono WAV.
use anyhow::{Result, ensure};
use candle_core::Device;
use serde_json::json;
use speechfoundry_qwen::audio::{AudioBuffer, MelSpectrogram};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 2,
        "usage: mel_oracle INPUT_24000_MONO_WAV OUTPUT_JSON"
    );
    let audio = AudioBuffer::load(&args[0])?;
    ensure!(
        audio.sample_rate == 24_000,
        "oracle requires identical 24000 Hz PCM; resample before comparing"
    );
    let frontend = MelSpectrogram::new(MelSpectrogram::speaker_encoder())?;
    let mel = frontend
        .compute_for_speaker_encoder(&audio.samples, &Device::Cpu)?
        .unsqueeze(0)?;
    let result = json!({
        "schema_version": 1,
        "metadata": {"implementation": "speechfoundry-qwen", "device": "cpu", "sample_rate": 24000},
        "tensors": {"mel": {"shape": mel.dims(), "dtype": "float32", "data": mel.flatten_all()?.to_vec1::<f32>()?}}
    });
    std::fs::write(&args[1], serde_json::to_vec_pretty(&result)?)?;
    Ok(())
}
