//! Shared band-limited resampling with exact duration and delay compensation.
use super::AudioBuffer;
use anyhow::Result;

/// Resample using the shared audio layer. For Python numerical parity, supply
/// already-resampled PCM; Rubato is not bit-identical to librosa/soxr.
pub fn resample(audio: &AudioBuffer, target_rate: u32) -> Result<AudioBuffer> {
    let audio =
        tts_core::Audio::new(audio.samples.clone(), audio.sample_rate)?.resample(target_rate)?;
    Ok(AudioBuffer::new(audio.samples, audio.sample_rate))
}

/// Convert input PCM to Qwen3-TTS's native 24 kHz.
pub fn resample_to_24k(audio: &AudioBuffer) -> Result<AudioBuffer> {
    resample(audio, 24_000)
}
