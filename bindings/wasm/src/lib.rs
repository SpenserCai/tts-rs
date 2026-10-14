//! Portable WebAssembly components, without a claim of full-model inference.

use tts_core::{Audio, SynthesisOptions, VoicePrompt};
use wasm_bindgen::prelude::*;

#[cfg(feature = "qwen")]
mod qwen;

const MAX_PROMPT_BYTES: usize = 16 * 1024 * 1024;
const MAX_OPTIONS_BYTES: usize = 64 * 1024;

fn check_json_size(value: &str, maximum: usize) -> Result<(), JsValue> {
    if value.len() > maximum {
        return Err(js_error(format!("JSON exceeds the {maximum}-byte limit")));
    }
    Ok(())
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

/// Owned, validated mono audio. Accessors copy samples across the Wasm boundary.
#[wasm_bindgen(js_name = Audio)]
pub struct WasmAudio {
    inner: Audio,
}

#[wasm_bindgen(js_class = Audio)]
impl WasmAudio {
    /// Construct from a Float32Array of finite mono samples.
    #[wasm_bindgen(constructor)]
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Result<WasmAudio, JsValue> {
        Ok(Self {
            inner: Audio::new(samples, sample_rate).map_err(js_error)?,
        })
    }

    /// Copy samples into a JavaScript Float32Array.
    #[wasm_bindgen(getter)]
    pub fn samples(&self) -> Vec<f32> {
        self.inner.samples.clone()
    }

    /// Samples per second.
    #[wasm_bindgen(getter, js_name = sampleRate)]
    pub fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }

    /// Duration in seconds.
    #[wasm_bindgen(getter, js_name = durationSeconds)]
    pub fn duration_seconds(&self) -> f64 {
        self.inner.duration_seconds()
    }

    /// Resample in Wasm. Use a Web Worker for large audio buffers.
    pub fn resample(&self, sample_rate: u32) -> Result<WasmAudio, JsValue> {
        Ok(Self {
            inner: self.inner.resample(sample_rate).map_err(js_error)?,
        })
    }
}

/// Validate a model-specific prompt envelope and return canonical compact JSON.
///
/// Envelope validation does not execute a model or validate its private payload.
#[wasm_bindgen(js_name = validateVoicePrompt)]
pub fn validate_voice_prompt(voice_prompt_json: &str) -> Result<String, JsValue> {
    check_json_size(voice_prompt_json, MAX_PROMPT_BYTES)?;
    let prompt: VoicePrompt = serde_json::from_str(voice_prompt_json).map_err(js_error)?;
    prompt.validate().map_err(js_error)?;
    serde_json::to_string(&prompt).map_err(js_error)
}

/// Parse, validate, and expand defaults for shared synthesis options.
#[wasm_bindgen(js_name = validateSynthesisOptions)]
pub fn validate_synthesis_options(options_json: &str) -> Result<String, JsValue> {
    check_json_size(options_json, MAX_OPTIONS_BYTES)?;
    let options: SynthesisOptions = serde_json::from_str(options_json).map_err(js_error)?;
    options.validate().map_err(js_error)?;
    serde_json::to_string(&options).map_err(js_error)
}

/// Report the capabilities actually present in this Wasm package.
#[wasm_bindgen(js_name = capabilities)]
pub fn capabilities() -> String {
    serde_json::json!({
        "audio_validation": true,
        "audio_resampling": true,
        "prompt_envelope_validation": true,
        "synthesis_options_validation": true,
        "voice_extraction": cfg!(feature = "qwen"),
        "speech_synthesis": false,
        "accelerators": [],
    })
    .to_string()
}
