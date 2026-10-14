//! Generation settings and Python-compatible defaults.
use super::*;

/// Options for speech synthesis
#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    /// Maximum autoregressive steps, including the first semantic prediction.
    /// At the length limit this yields at most `max_length - 1` complete codec frames,
    /// matching the official Python `max_new_tokens` behavior.
    pub max_length: usize,
    /// Sampling temperature (higher = more random)
    pub temperature: f64,
    /// Top-k sampling
    pub top_k: usize,
    /// Top-p (nucleus) sampling
    pub top_p: f64,
    /// Repetition penalty (1.0 = disabled, 1.05 = Python default)
    pub repetition_penalty: f64,
    /// End-of-sequence token ID (defaults to codec EOS token 2150)
    pub eos_token_id: Option<u32>,
    /// Frames per streaming chunk (default: 10 = ~800ms)
    pub chunk_frames: usize,
    /// Minimum tokens before EOS is allowed (default: 2, matching Python)
    pub min_new_tokens: usize,
    /// Acoustic predictor temperature; zero selects greedy decoding.
    pub subtalker_temperature: f64,
    /// Acoustic predictor top-k limit; zero disables filtering.
    pub subtalker_top_k: usize,
    /// Acoustic predictor nucleus probability.
    pub subtalker_top_p: f64,
    /// Random seed for repeatability within the same backend and build. `None` uses time entropy.
    pub seed: Option<u64>,
}

impl SynthesisOptions {
    /// Reject invalid numeric settings before allocating model caches.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.max_length >= 2,
            "max_length must be at least two to produce a complete audio frame"
        );
        anyhow::ensure!(self.chunk_frames > 0, "chunk_frames must be positive");
        anyhow::ensure!(
            self.temperature.is_finite() && self.temperature >= 0.0,
            "temperature must be finite and nonnegative"
        );
        anyhow::ensure!(
            self.subtalker_temperature.is_finite() && self.subtalker_temperature >= 0.0,
            "subtalker temperature must be finite and nonnegative"
        );
        anyhow::ensure!(
            self.top_p.is_finite() && self.top_p > 0.0 && self.top_p <= 1.0,
            "top_p must be in (0, 1]"
        );
        anyhow::ensure!(
            self.subtalker_top_p.is_finite()
                && self.subtalker_top_p > 0.0
                && self.subtalker_top_p <= 1.0,
            "subtalker_top_p must be in (0, 1]"
        );
        anyhow::ensure!(
            self.repetition_penalty.is_finite() && self.repetition_penalty > 0.0,
            "repetition penalty must be finite and positive"
        );
        anyhow::ensure!(
            self.eos_token_id.is_none_or(|id| id < 3072),
            "EOS token must be inside the codec vocabulary"
        );
        Ok(())
    }

    /// Convert to a [`GenerationConfig`](generation::GenerationConfig) for the generation loop.
    pub fn to_gen_config(&self) -> generation::GenerationConfig {
        generation::GenerationConfig {
            max_new_tokens: self.max_length,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            eos_token_id: self.eos_token_id,
            min_new_tokens: self.min_new_tokens,
            subtalker_temperature: self.subtalker_temperature,
            subtalker_top_k: self.subtalker_top_k,
            subtalker_top_p: self.subtalker_top_p,
        }
    }
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            max_length: 2048,
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.05,
            eos_token_id: Some(CODEC_EOS_TOKEN_ID),
            chunk_frames: 10, // ~800ms per chunk at 12.5 Hz
            min_new_tokens: 2,
            seed: None,
            subtalker_temperature: 0.9,
            subtalker_top_k: 50,
            subtalker_top_p: 1.0,
        }
    }
}
