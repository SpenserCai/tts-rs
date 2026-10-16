//! Optional Qwen CPU inference from caller-supplied checkpoint bytes.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use speechfoundry_core::Audio;
use speechfoundry_qwen::{
    audio::AudioBuffer, models::SpeakerEncoderConfig, models::speaker::SpeakerEncoder,
};
use wasm_bindgen::prelude::*;

use super::{MAX_OPTIONS_BYTES, check_json_size, js_error};

/// Qwen's ECAPA speaker encoder loaded from a safetensors byte buffer.
///
/// This is the actual neural encoder. Its embeddings are specific to Qwen's
/// compatible Base checkpoint, and are not a universal speaker identity.
#[wasm_bindgen(js_name = QwenSpeakerEncoder)]
pub struct WasmQwenSpeakerEncoder {
    inner: SpeakerEncoder,
    mel_dim: usize,
}

#[wasm_bindgen(js_class = QwenSpeakerEncoder)]
impl WasmQwenSpeakerEncoder {
    /// Load a speaker-only safetensors buffer with the prefix removed.
    ///
    /// `config_json` is the official `speaker_encoder_config` object. Weights
    /// must use keys such as `blocks.0.conv.weight`, without `speaker_encoder.`.
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str, weights: Vec<u8>) -> Result<Self, JsValue> {
        check_json_size(config_json, MAX_OPTIONS_BYTES)?;
        let config: SpeakerEncoderConfig = serde_json::from_str(config_json).map_err(js_error)?;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &Device::Cpu)
            .map_err(js_error)?;
        let mel_dim = config.mel_dim;
        Ok(Self {
            inner: SpeakerEncoder::new(config, vb).map_err(js_error)?,
            mel_dim,
        })
    }

    /// Extract the unnormalized speaker vector from validated mono PCM.
    ///
    /// Qwen's native speaker frontend requires 24 kHz. Resample explicitly with
    /// `Audio.resample(24000)` when the source recording uses a different rate.
    /// Run this synchronous CPU operation in a Web Worker.
    #[wasm_bindgen(js_name = extractEmbedding)]
    pub fn extract_embedding(
        &self,
        samples: Vec<f32>,
        sample_rate: u32,
    ) -> Result<Vec<f32>, JsValue> {
        if u64::try_from(samples.len()).unwrap_or(u64::MAX) > u64::from(sample_rate) * 30 {
            return Err(js_error("reference audio must not exceed 30 seconds"));
        }
        let audio = Audio::new(samples, sample_rate).map_err(js_error)?;
        self.inner
            .encode(&AudioBuffer {
                samples: audio.samples,
                sample_rate: audio.sample_rate,
            })
            .and_then(|tensor| Ok(tensor.to_vec1::<f32>()?))
            .map_err(js_error)
    }

    /// Evaluate one `[mel_bands, frames]` log-mel tensor for numerical fixtures.
    ///
    /// Values are flattened in channel-major order. This bypasses the audio
    /// frontend so an identical Python intermediate can be compared directly.
    #[wasm_bindgen(js_name = forwardMel)]
    pub fn forward_mel(&self, values: Vec<f32>, frames: usize) -> Result<Vec<f32>, JsValue> {
        if frames == 0 || values.iter().any(|value| !value.is_finite()) {
            return Err(js_error("mel frames must be nonempty and finite"));
        }
        let mel =
            Tensor::from_vec(values, (1, self.mel_dim, frames), &Device::Cpu).map_err(js_error)?;
        self.inner
            .forward(&mel)
            .and_then(|tensor| Ok(tensor.flatten_all()?.to_vec1::<f32>()?))
            .map_err(js_error)
    }
}
