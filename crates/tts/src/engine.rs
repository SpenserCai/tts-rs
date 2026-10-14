use std::path::Path;
#[cfg(feature = "cosyvoice")]
use std::path::PathBuf;

use candle_core::Device;

use crate::{Audio, DeviceConfig, Error, ModelKind, Result, SynthesisOptions, VoicePrompt};

enum Backend {
    #[cfg(feature = "qwen")]
    Qwen(Box<tts_qwen::Qwen3TTS>),
    #[cfg(feature = "cosyvoice")]
    CosyVoice {
        model: Box<tts_cosyvoice::CosyVoice3>,
        directory: PathBuf,
        frontend: Option<Box<tts_cosyvoice::CosyVoice3Frontend>>,
    },
}

/// One loaded model session. Mutable access serializes generation and caches.
///
/// Model weights stay local; this API performs no downloads or network calls.
/// Python and Node bindings move blocking work away from their interpreter.
pub struct Engine {
    backend: Backend,
    kind: ModelKind,
    fingerprint: String,
    device: Device,
}

fn model_error(error: impl std::fmt::Display) -> Error {
    Error::Model(error.to_string())
}

fn device(config: DeviceConfig) -> Result<Device> {
    match config {
        DeviceConfig::Cpu => Ok(Device::Cpu),
        #[cfg(feature = "cuda")]
        DeviceConfig::Cuda(index) => Device::new_cuda(index).map_err(model_error),
        #[cfg(not(feature = "cuda"))]
        DeviceConfig::Cuda(_) => Err(Error::Unsupported("build with feature cuda".into())),
        #[cfg(feature = "metal")]
        DeviceConfig::Metal(index) => Device::new_metal(index).map_err(model_error),
        #[cfg(not(feature = "metal"))]
        DeviceConfig::Metal(_) => Err(Error::Unsupported("build with feature metal".into())),
    }
}

impl Engine {
    /// Load a local checkpoint, selecting hardware explicitly.
    pub fn load(model_dir: &str, kind: ModelKind, config: DeviceConfig) -> Result<Self> {
        let directory = Path::new(model_dir);
        if !directory.is_dir() {
            return Err(Error::InvalidInput(
                "model_dir must be an existing directory".into(),
            ));
        }
        let device = device(config)?;
        let backend = match kind {
            #[cfg(feature = "qwen")]
            ModelKind::Qwen3 => Backend::Qwen(Box::new(
                tts_qwen::Qwen3TTS::from_pretrained(model_dir, device.clone())
                    .map_err(model_error)?,
            )),
            #[cfg(not(feature = "qwen"))]
            ModelKind::Qwen3 => return Err(Error::Unsupported("build with feature qwen".into())),
            #[cfg(feature = "cosyvoice")]
            ModelKind::Cosyvoice3 => Backend::CosyVoice {
                model: Box::new(
                    tts_cosyvoice::CosyVoice3::load(directory, &device, candle_core::DType::F32)
                        .map_err(model_error)?,
                ),
                directory: directory.to_path_buf(),
                frontend: None,
            },
            #[cfg(not(feature = "cosyvoice"))]
            ModelKind::Cosyvoice3 => {
                return Err(Error::Unsupported("build with feature cosyvoice".into()));
            }
        };
        let fingerprint = crate::fingerprint::model_fingerprint(directory, kind)?;
        Ok(Self {
            backend,
            kind,
            fingerprint,
            device,
        })
    }

    /// Extract model-specific conditioning. The transcript must match the audio.
    ///
    /// Qwen accepts `None` for speaker-only cloning. CosyVoice requires a
    /// transcript for zero-shot synthesis; the frontend is loaded lazily.
    pub fn extract_voice(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        transcript: Option<&str>,
    ) -> Result<VoicePrompt> {
        let audio = Audio::new(samples.to_vec(), sample_rate)?;
        if audio.duration_seconds() > 30.0 {
            return Err(Error::InvalidInput(
                "reference audio must be at most 30 seconds".into(),
            ));
        }
        if transcript.is_some_and(|text| text.trim().is_empty()) {
            return Err(Error::InvalidInput(
                "reference transcript cannot be blank".into(),
            ));
        }
        let (format, payload) = match &mut self.backend {
            #[cfg(feature = "qwen")]
            Backend::Qwen(model) => {
                let audio = audio.resample(24_000)?;
                let prompt = model
                    .create_voice_clone_prompt(
                        &tts_qwen::AudioBuffer::new(audio.samples, audio.sample_rate),
                        transcript,
                    )
                    .map_err(model_error)?;
                ("qwen3-12hz-prompt-v1", crate::qwen_prompt::encode(&prompt)?)
            }
            #[cfg(feature = "cosyvoice")]
            Backend::CosyVoice {
                directory,
                frontend,
                ..
            } => {
                let transcript = transcript.ok_or_else(|| {
                    Error::InvalidInput(
                        "CosyVoice zero-shot extraction requires the reference transcript".into(),
                    )
                })?;
                if frontend.is_none() {
                    *frontend = Some(Box::new(
                        tts_cosyvoice::CosyVoice3Frontend::load(directory, &Device::Cpu)
                            .map_err(model_error)?,
                    ));
                }
                let frontend = frontend
                    .as_ref()
                    .ok_or_else(|| Error::Model("frontend not loaded".into()))?;
                let prompt = frontend
                    .extract_prompt(&audio.samples, audio.sample_rate, transcript)
                    .map_err(model_error)?;
                ("cosyvoice3-2512-prompt-v1", serde_json::to_value(prompt)?)
            }
        };
        let prompt = VoicePrompt {
            schema_version: 1,
            model: self.kind,
            model_fingerprint: self.fingerprint.clone(),
            format: format.into(),
            payload,
        };
        prompt.validate()?;
        Ok(prompt)
    }

    /// Generate mono PCM from a compatible persisted prompt.
    pub fn synthesize(
        &mut self,
        text: &str,
        prompt: &VoicePrompt,
        options: &SynthesisOptions,
    ) -> Result<Audio> {
        options.validate()?;
        prompt.validate()?;
        if text.trim().is_empty() || text.len() > 32_768 {
            return Err(Error::InvalidInput(
                "text must contain 1..=32768 UTF-8 bytes".into(),
            ));
        }
        if prompt.model != self.kind || prompt.model_fingerprint != self.fingerprint {
            return Err(Error::InvalidInput(
                "voice prompt belongs to a different checkpoint".into(),
            ));
        }
        let audio = match &mut self.backend {
            #[cfg(feature = "qwen")]
            Backend::Qwen(model) => {
                if prompt.format != "qwen3-12hz-prompt-v1" {
                    return Err(Error::Unsupported(format!(
                        "prompt format {}",
                        prompt.format
                    )));
                }
                let voice = crate::qwen_prompt::decode(&prompt.payload, &self.device)?;
                let language = options.language.parse().map_err(model_error)?;
                let settings = tts_qwen::SynthesisOptions {
                    max_length: options.max_tokens,
                    temperature: options.temperature,
                    top_p: options.top_p,
                    top_k: options.top_k,
                    seed: Some(options.seed),
                    subtalker_temperature: options.temperature,
                    subtalker_top_p: options.top_p,
                    subtalker_top_k: options.top_k,
                    ..Default::default()
                };
                let output = model
                    .synthesize_voice_clone(text, &voice, language, Some(settings))
                    .map_err(model_error)?;
                Audio::new(output.samples, output.sample_rate)?
            }
            #[cfg(feature = "cosyvoice")]
            Backend::CosyVoice { model, .. } => {
                if prompt.format != "cosyvoice3-2512-prompt-v1" {
                    return Err(Error::Unsupported(format!(
                        "prompt format {}",
                        prompt.format
                    )));
                }
                if options.language != "auto" {
                    return Err(Error::Unsupported(
                        "CosyVoice detects language from text; use language=auto".into(),
                    ));
                }
                let voice = serde_json::from_value::<tts_cosyvoice::PromptFeatures>(
                    prompt.payload.clone(),
                )?;
                let settings = tts_cosyvoice::SynthesisOptions {
                    sampling: tts_cosyvoice::SamplingConfig {
                        seed: options.seed,
                        max_tokens: options.max_tokens,
                        top_k: options.top_k,
                        top_p: options.top_p as f32,
                        temperature: options.temperature as f32,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                Audio::new(
                    model
                        .synthesize(text, &voice, &settings)
                        .map_err(model_error)?,
                    model.sample_rate(),
                )?
            }
        };
        self.device.synchronize().map_err(model_error)?;
        Ok(audio)
    }

    /// Model family loaded in this session.
    pub fn model_kind(&self) -> ModelKind {
        self.kind
    }

    /// Content fingerprint used to bind persisted prompts to this checkpoint.
    pub fn model_fingerprint(&self) -> &str {
        &self.fingerprint
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_is_send_for_language_binding_workers() {
        fn assert_send<T: Send>() {}
        assert_send::<Engine>();
    }

    #[test]
    fn missing_model_is_an_error() {
        assert!(Engine::load("/not/a/checkpoint", ModelKind::Qwen3, DeviceConfig::Cpu).is_err());
    }
}
