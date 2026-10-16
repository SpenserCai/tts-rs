//! Node-API bindings. Model work runs on libuv workers, never the JS thread.

use std::sync::{Arc, Mutex};

use napi::bindgen_prelude::*;
use napi_derive::napi;
use speechfoundry::{Audio, DeviceConfig, Engine, ModelKind, SynthesisOptions, VoicePrompt};

const MAX_PROMPT_BYTES: usize = 16 * 1024 * 1024;
const MAX_OPTIONS_BYTES: usize = 64 * 1024;

/// Compiled native binding identity, independent of npm manifest metadata.
#[napi(object)]
pub struct BackendInfo {
    /// Unified workspace version.
    pub version: String,
    /// Language-wrapper protocol version.
    pub api: u32,
    /// Whether this binary includes CUDA support.
    pub cuda: bool,
    /// Whether this binary includes Metal support.
    pub metal: bool,
}

/// Describe this binary without initializing a device or loading a checkpoint.
#[napi]
pub fn backend_info() -> BackendInfo {
    BackendInfo {
        version: env!("CARGO_PKG_VERSION").into(),
        api: 1,
        cuda: cfg!(feature = "cuda"),
        metal: cfg!(feature = "metal"),
    }
}

fn check_json_size(value: &str, maximum: usize) -> Result<()> {
    if value.len() > maximum {
        return Err(invalid(format!("JSON exceeds the {maximum}-byte limit")));
    }
    Ok(())
}

fn invalid(error: impl std::fmt::Display) -> Error {
    Error::new(Status::InvalidArg, error.to_string())
}

fn failure(error: impl std::fmt::Display) -> Error {
    Error::new(Status::GenericFailure, error.to_string())
}

/// Mono PCM output. Samples are transferred as a Float32Array.
#[napi(object)]
pub struct AudioOutput {
    /// Finite mono PCM samples.
    pub samples: Float32Array,
    /// Samples per second.
    pub sample_rate: u32,
}

impl From<Audio> for AudioOutput {
    fn from(audio: Audio) -> Self {
        Self {
            samples: audio.samples.into(),
            sample_rate: audio.sample_rate,
        }
    }
}

/// A loaded engine; operations on one instance are serialized.
#[napi(js_name = "Engine")]
pub struct JsEngine {
    inner: Arc<Mutex<Engine>>,
}

#[napi]
impl JsEngine {
    /// Extract versioned voice conditioning without blocking the JS event loop.
    #[napi]
    pub fn extract_voice(
        &self,
        samples: Float32Array,
        sample_rate: u32,
        transcript: Option<String>,
    ) -> AsyncTask<ExtractTask> {
        AsyncTask::new(ExtractTask {
            engine: Arc::clone(&self.inner),
            samples: samples.to_vec(),
            sample_rate,
            transcript,
        })
    }

    /// Synthesize from matching model-specific conditioning on a worker.
    #[napi]
    pub fn synthesize(
        &self,
        text: String,
        voice_prompt_json: String,
        options_json: Option<String>,
    ) -> AsyncTask<SynthesizeTask> {
        AsyncTask::new(SynthesizeTask {
            engine: Arc::clone(&self.inner),
            text,
            voice_prompt_json,
            options_json,
        })
    }
}

/// Worker state for asynchronous model loading.
pub struct LoadTask {
    model_dir: String,
    model: String,
    device: String,
}

#[napi]
impl Task for LoadTask {
    type Output = Engine;
    type JsValue = JsEngine;

    fn compute(&mut self) -> Result<Self::Output> {
        let model: ModelKind = self.model.parse().map_err(invalid)?;
        let device: DeviceConfig = self.device.parse().map_err(invalid)?;
        Engine::load(&self.model_dir, model, device).map_err(failure)
    }

    fn resolve(&mut self, _: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(JsEngine {
            inner: Arc::new(Mutex::new(output)),
        })
    }
}

/// Load a model asynchronously. Device defaults to cpu.
#[napi]
pub fn load_engine(
    model_dir: String,
    model: String,
    device: Option<String>,
) -> AsyncTask<LoadTask> {
    AsyncTask::new(LoadTask {
        model_dir,
        model,
        device: device.unwrap_or_else(|| "cpu".into()),
    })
}

/// Worker state for voice conditioning extraction.
pub struct ExtractTask {
    engine: Arc<Mutex<Engine>>,
    samples: Vec<f32>,
    sample_rate: u32,
    transcript: Option<String>,
}

#[napi]
impl Task for ExtractTask {
    type Output = String;
    type JsValue = String;

    fn compute(&mut self) -> Result<Self::Output> {
        let mut engine = self.engine.lock().map_err(failure)?;
        let prompt = engine
            .extract_voice(&self.samples, self.sample_rate, self.transcript.as_deref())
            .map_err(failure)?;
        serde_json::to_string(&prompt).map_err(failure)
    }

    fn resolve(&mut self, _: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

/// Worker state for speech synthesis.
pub struct SynthesizeTask {
    engine: Arc<Mutex<Engine>>,
    text: String,
    voice_prompt_json: String,
    options_json: Option<String>,
}

#[napi]
impl Task for SynthesizeTask {
    type Output = Audio;
    type JsValue = AudioOutput;

    fn compute(&mut self) -> Result<Self::Output> {
        check_json_size(&self.voice_prompt_json, MAX_PROMPT_BYTES)?;
        if let Some(options) = &self.options_json {
            check_json_size(options, MAX_OPTIONS_BYTES)?;
        }
        let prompt: VoicePrompt = serde_json::from_str(&self.voice_prompt_json).map_err(invalid)?;
        prompt.validate().map_err(invalid)?;
        let options = self
            .options_json
            .as_deref()
            .map(serde_json::from_str::<SynthesisOptions>)
            .transpose()
            .map_err(invalid)?
            .unwrap_or_default();
        let mut engine = self.engine.lock().map_err(failure)?;
        engine
            .synthesize(&self.text, &prompt, &options)
            .map_err(failure)
    }

    fn resolve(&mut self, _: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output.into())
    }
}

/// Worker state for portable audio resampling.
pub struct ResampleTask {
    samples: Vec<f32>,
    source_rate: u32,
    target_rate: u32,
}

#[napi]
impl Task for ResampleTask {
    type Output = Audio;
    type JsValue = AudioOutput;

    fn compute(&mut self) -> Result<Self::Output> {
        Audio::new(std::mem::take(&mut self.samples), self.source_rate)
            .and_then(|audio| audio.resample(self.target_rate))
            .map_err(invalid)
    }

    fn resolve(&mut self, _: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output.into())
    }
}

/// Validate and resample finite mono PCM on a worker.
#[napi]
pub fn resample_audio(
    samples: Float32Array,
    source_rate: u32,
    target_rate: u32,
) -> AsyncTask<ResampleTask> {
    AsyncTask::new(ResampleTask {
        samples: samples.to_vec(),
        source_rate,
        target_rate,
    })
}

/// Worker state for prompt envelope validation.
pub struct ValidatePromptTask {
    prompt: String,
}

#[napi]
impl Task for ValidatePromptTask {
    type Output = String;
    type JsValue = String;

    fn compute(&mut self) -> Result<Self::Output> {
        check_json_size(&self.prompt, MAX_PROMPT_BYTES)?;
        let prompt: VoicePrompt = serde_json::from_str(&self.prompt).map_err(invalid)?;
        prompt.validate().map_err(invalid)?;
        serde_json::to_string(&prompt).map_err(invalid)
    }

    fn resolve(&mut self, _: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

/// Validate a voice prompt envelope and return canonical compact JSON.
#[napi]
pub fn validate_voice_prompt(voice_prompt_json: String) -> AsyncTask<ValidatePromptTask> {
    AsyncTask::new(ValidatePromptTask {
        prompt: voice_prompt_json,
    })
}
