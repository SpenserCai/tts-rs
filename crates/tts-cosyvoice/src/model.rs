use std::path::Path;

use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::config::{
    CosyVoice3Config, MEL_N_MELS, SAMPLE_RATE, SPEECH_TOKEN_SIZE, SPK_EMBED_DIM, SamplingConfig,
};
use crate::flow::DiT;
use crate::flow::flow_matching::CausalMaskedDiffWithDiT;
use crate::llm::CosyVoice3LM;
use crate::vocoder::CausalHiFTGenerator;

/// Portable, model-specific conditioning extracted from reference audio.
///
/// This is not an identity-verification template. The transcript must match the
/// reference audio; acoustic tokens and mel frames are aligned at a ratio of two.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptFeatures {
    /// Semantic speech tokens, each below 6561.
    pub speech_tokens: Vec<u32>,
    /// Time-major 80-channel log-mel features at 50 Hz.
    pub mel: Vec<Vec<f32>>,
    /// Raw 192-dimensional CAMPPlus embedding (normalized inside the flow).
    pub speaker_embedding: Vec<f32>,
    /// Reference transcript, without the instruction prefix.
    pub transcript: String,
}

impl PromptFeatures {
    /// Validate shapes and values before allocating device tensors.
    pub fn validate(&self) -> Result<()> {
        if self.transcript.len() > 32_768 {
            candle::bail!("reference transcript exceeds 32768 bytes");
        }
        if self.speech_tokens.is_empty() || self.speech_tokens.len() > 750 {
            candle::bail!("reference requires 1..=750 speech tokens (at most 30 seconds)");
        }
        if self
            .speech_tokens
            .iter()
            .any(|&t| t as usize >= SPEECH_TOKEN_SIZE)
        {
            candle::bail!("reference speech token is outside the CosyVoice 3 vocabulary");
        }
        if self.mel.len() != self.speech_tokens.len() * 2
            || self
                .mel
                .iter()
                .any(|row| row.len() != MEL_N_MELS || row.iter().any(|x| !x.is_finite()))
        {
            candle::bail!("reference mel must have shape [2 * tokens, 80] and finite values");
        }
        if self.speaker_embedding.len() != SPK_EMBED_DIM
            || self.speaker_embedding.iter().any(|x| !x.is_finite())
        {
            candle::bail!("speaker embedding requires 192 finite values");
        }
        Ok(())
    }
}

/// Which reference conditioning enters the language model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum SynthesisMode {
    /// Transcript and reference speech tokens condition the language model.
    #[default]
    ZeroShot,
    /// Reference audio conditions the acoustic decoder only.
    CrossLingual,
    /// Style instructions condition the language model; reference audio supplies timbre.
    Instruct(String),
}

/// Generation controls. Model configuration owns the classifier-free guidance rate.
#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    /// Language-model conditioning mode.
    pub mode: SynthesisMode,
    /// Sampling controls for speech tokens.
    pub sampling: SamplingConfig,
    /// Euler steps; upstream inference uses ten.
    pub flow_steps: usize,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            mode: SynthesisMode::default(),
            sampling: SamplingConfig::default(),
            flow_steps: 10,
        }
    }
}

/// Offline, batch-one CosyVoice 3 synthesis pipeline.
///
/// Text normalization is deliberately caller-controlled. Provide normalized text
/// for reproducibility against upstream `text_frontend=False`.
pub struct CosyVoice3 {
    llm: CosyVoice3LM,
    flow: CausalMaskedDiffWithDiT,
    vocoder: CausalHiFTGenerator,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
}

impl CosyVoice3 {
    /// Load converted `llm`, `flow`, and `hift` safetensors and a complete tokenizer.
    ///
    /// `model-config.json` is optional; absent configuration selects the official
    /// Fun-CosyVoice3-0.5B-2512 architecture. A bare BPE vocabulary is insufficient.
    pub fn load(model_dir: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<Self> {
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            candle::bail!("CosyVoice 3 requires a floating-point model dtype");
        }
        let dir = model_dir.as_ref();
        let config_path = dir.join("model-config.json");
        let config: CosyVoice3Config = if config_path.exists() {
            serde_json::from_slice(&std::fs::read(config_path).map_err(candle::Error::wrap)?)
                .map_err(candle::Error::wrap)?
        } else {
            CosyVoice3Config::default()
        };
        if config.sample_rate != SAMPLE_RATE {
            candle::bail!("this adapter targets the 24 kHz CosyVoice 3 checkpoint");
        }
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| candle::Error::Msg(format!("complete tokenizer.json is required: {e}")))?;
        if tokenizer.token_to_id("<|endofprompt|>") != Some(151646) {
            candle::bail!("tokenizer is missing the CosyVoice 3 end-of-prompt token at id 151646");
        }
        let load_weights = |name: &str| -> Result<VarBuilder<'static>> {
            let data = std::fs::read(dir.join(name)).map_err(candle::Error::wrap)?;
            VarBuilder::from_buffered_safetensors(data, dtype, device)
        };
        let llm = CosyVoice3LM::new(&config.llm, load_weights("llm.safetensors")?)?;
        let flow_vb = load_weights("flow.safetensors")?;
        let dit = DiT::new(config.flow.dit.clone(), flow_vb.pp("dit"))?;
        let noise_path = dir.join("flow-noise.safetensors");
        let noise = if noise_path.exists() {
            let vb = load_weights("flow-noise.safetensors")?;
            Some(vb.get((1, 80, 15_000), "noise")?)
        } else {
            None
        };
        let flow = CausalMaskedDiffWithDiT::new_with_rand_noise(
            config.flow.vocab_size,
            config.flow.input_size,
            config.flow.output_size,
            SPK_EMBED_DIM,
            config.flow.token_mel_ratio,
            config.flow.pre_lookahead_len,
            dit,
            config.flow.cfm,
            flow_vb,
            noise,
        )?;
        let mut vocoder = CausalHiFTGenerator::new(config.hift, load_weights("hift.safetensors")?)?;
        let source_noise = dir.join("source-noise.safetensors");
        if source_noise.exists() {
            let mut tensors = candle::safetensors::load(source_noise, device)?;
            let mut take = |key: &str| {
                tensors.remove(key).ok_or_else(|| {
                    candle::Error::Msg(format!("missing source-noise tensor: {key}"))
                })
            };
            vocoder.set_reference_noise(take("phase")?, take("sine")?, take("uv")?)?;
        }
        Ok(Self {
            llm,
            flow,
            vocoder,
            tokenizer,
            device: device.clone(),
            dtype,
        })
    }

    /// Synthesize a mono waveform at 24 kHz, using cached conditioning.
    pub fn synthesize(
        &mut self,
        text: &str,
        prompt: &PromptFeatures,
        options: &SynthesisOptions,
    ) -> Result<Vec<f32>> {
        let tokens = self.generate_tokens(text, prompt, options)?;
        self.decode_tokens(&tokens, prompt, options.flow_steps)
    }

    /// Generate semantic speech tokens with the same conditioning as synthesis.
    /// This exposes the discrete intermediate for reproducible stage comparisons.
    pub fn generate_tokens(
        &mut self,
        text: &str,
        prompt: &PromptFeatures,
        options: &SynthesisOptions,
    ) -> Result<Vec<u32>> {
        prompt.validate()?;
        if text.trim().is_empty()
            || text.len() > 32_768
            || options.flow_steps == 0
            || options.flow_steps > 100
        {
            candle::bail!(
                "text must be nonempty and at most 32768 bytes; flow_steps must be in 1..=100"
            );
        }
        let s = &options.sampling;
        if !s.temperature.is_finite()
            || s.temperature < 0.0
            || s.max_tokens == 0
            || !s.top_p.is_finite()
            || !(0.0..=1.0).contains(&s.top_p)
            || !s.repetition_penalty.is_finite()
            || s.repetition_penalty <= 0.0
        {
            candle::bail!("invalid speech-token sampling configuration");
        }
        let (prefix, use_prompt) = match &options.mode {
            SynthesisMode::ZeroShot => {
                if prompt.transcript.trim().is_empty() {
                    candle::bail!("zero-shot synthesis requires the reference transcript");
                }
                (
                    format!(
                        "You are a helpful assistant.<|endofprompt|>{}",
                        prompt.transcript
                    ),
                    true,
                )
            }
            SynthesisMode::CrossLingual => (
                "You are a helpful assistant.<|endofprompt|>".to_owned(),
                false,
            ),
            SynthesisMode::Instruct(instruction) => {
                if instruction.trim().is_empty() || instruction.len() > 32_768 {
                    candle::bail!("style instruction requires 1..=32768 bytes");
                }
                (format!("{instruction}<|endofprompt|>"), false)
            }
        };
        let text_ids = self.tokenize(text)?;
        let prefix_ids = self.tokenize(&prefix)?;
        let reference_ids = Tensor::from_slice(
            &prompt.speech_tokens,
            (1, prompt.speech_tokens.len()),
            &self.device,
        )?;
        let lm_reference = if use_prompt {
            reference_ids.clone()
        } else {
            Tensor::zeros((1, 0), DType::U32, &self.device)?
        };
        self.llm.inference(&text_ids, &prefix_ids, &lm_reference, s)
    }

    /// Decode supplied speech tokens, bypassing stochastic language-model sampling.
    /// Useful for numerical comparison using tokens exported by the Python oracle.
    pub fn decode_tokens(
        &self,
        tokens: &[u32],
        prompt: &PromptFeatures,
        steps: usize,
    ) -> Result<Vec<f32>> {
        prompt.validate()?;
        if tokens.is_empty()
            || tokens.len() + prompt.speech_tokens.len() > 7500
            || tokens.iter().any(|&x| x as usize >= SPEECH_TOKEN_SIZE)
            || steps == 0
            || steps > 100
        {
            candle::bail!("decoder requires valid nonempty speech tokens and positive step count");
        }
        let generated = Tensor::from_slice(tokens, (1, tokens.len()), &self.device)?;
        let reference = Tensor::from_slice(
            &prompt.speech_tokens,
            (1, prompt.speech_tokens.len()),
            &self.device,
        )?;
        let mel_values: Vec<f32> = prompt.mel.iter().flatten().copied().collect();
        let prompt_mel =
            Tensor::from_vec(mel_values, (1, prompt.mel.len(), MEL_N_MELS), &self.device)?
                .to_dtype(self.dtype)?;
        let embedding =
            Tensor::from_slice(&prompt.speaker_embedding, (1, SPK_EMBED_DIM), &self.device)?
                .to_dtype(self.dtype)?;
        let mel = self.flow.inference(
            &generated,
            &reference,
            &prompt_mel,
            &embedding,
            steps,
            false,
        )?;
        let audio = self
            .vocoder
            .inference(&mel, true)?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1()?;
        if audio.iter().any(|x: &f32| !x.is_finite()) {
            candle::bail!("vocoder produced non-finite samples");
        }
        Ok(audio)
    }

    /// The output waveform sampling rate, in hertz.
    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE as u32
    }

    fn tokenize(&self, text: &str) -> Result<Tensor> {
        let encoded = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| candle::Error::Msg(e.to_string()))?;
        Tensor::from_slice(encoded.get_ids(), (1, encoded.len()), &self.device)
    }
}
