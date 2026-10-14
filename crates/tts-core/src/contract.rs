use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// Model family. Conditioning payloads are specific to this family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    /// Qwen3-TTS 12 Hz models.
    Qwen3,
    /// CosyVoice3 models.
    Cosyvoice3,
}

impl fmt::Display for ModelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Qwen3 => "qwen3",
            Self::Cosyvoice3 => "cosyvoice3",
        })
    }
}

impl FromStr for ModelKind {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "qwen3" => Ok(Self::Qwen3),
            "cosyvoice3" => Ok(Self::Cosyvoice3),
            _ => Err(Error::InvalidInput(format!("unknown model family: {s}"))),
        }
    }
}

/// Explicit device selection; unavailable devices never silently fall back.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceConfig {
    /// Portable CPU execution.
    #[default]
    Cpu,
    /// NVIDIA GPU ordinal, requiring the `cuda` feature.
    Cuda(usize),
    /// Apple GPU ordinal, requiring the `metal` feature.
    Metal(usize),
}

impl FromStr for DeviceConfig {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        if s == "cpu" {
            return Ok(Self::Cpu);
        }
        let (kind, ordinal) = s.split_once(':').unwrap_or((s, "0"));
        let ordinal = ordinal
            .parse()
            .map_err(|_| Error::InvalidInput(format!("invalid device ordinal: {s}")))?;
        match kind {
            "cuda" => Ok(Self::Cuda(ordinal)),
            "metal" => Ok(Self::Metal(ordinal)),
            _ => Err(Error::InvalidInput(format!("unknown device: {s}"))),
        }
    }
}

/// Portable generation options. Model adapters must reject unsupported options.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SynthesisOptions {
    /// Maximum generated codec frames, not text tokens or waveform samples.
    pub max_tokens: usize,
    /// Sampling temperature; zero selects greedy generation.
    pub temperature: f64,
    /// Nucleus probability cutoff in `(0, 1]`.
    pub top_p: f64,
    /// Top-k cutoff; zero disables this filter.
    pub top_k: usize,
    /// Seed for this implementation; does not imply PyTorch RNG equivalence.
    pub seed: u64,
    /// Model language, e.g. `auto`, `chinese`, or `english`.
    pub language: String,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.9,
            top_p: 1.0,
            top_k: 50,
            seed: 0,
            language: "auto".into(),
        }
    }
}

impl SynthesisOptions {
    /// Validate before any model allocation or sampling.
    pub fn validate(&self) -> Result<()> {
        if self.max_tokens == 0 || self.max_tokens > 32_768 {
            return Err(Error::InvalidInput(
                "max_tokens must be in 1..=32768".into(),
            ));
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(Error::InvalidInput(
                "temperature must be finite and nonnegative".into(),
            ));
        }
        if !self.top_p.is_finite() || !(0.0 < self.top_p && self.top_p <= 1.0) {
            return Err(Error::InvalidInput("top_p must be in (0, 1]".into()));
        }
        if self.language.trim().is_empty() {
            return Err(Error::InvalidInput("language cannot be empty".into()));
        }
        Ok(())
    }
}

/// A persisted prompt envelope, never a universal biometric identity.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoicePrompt {
    /// Envelope version. Currently exactly `1`.
    pub schema_version: u32,
    /// Model family that produced the payload.
    pub model: ModelKind,
    /// SHA-256 of the model configuration and checkpoint files.
    pub model_fingerprint: String,
    /// Versioned format name interpreted by the owning model adapter.
    pub format: String,
    /// Model-specific conditioning data, serialized on CPU.
    pub payload: serde_json::Value,
}

impl VoicePrompt {
    /// Validate the envelope. The loaded model validates the payload itself.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(Error::Unsupported(format!(
                "prompt schema {}",
                self.schema_version
            )));
        }
        if self.model_fingerprint.len() != 64
            || !self
                .model_fingerprint
                .bytes()
                .all(|c| c.is_ascii_hexdigit())
        {
            return Err(Error::InvalidInput(
                "model_fingerprint must be a SHA-256 hex digest".into(),
            ));
        }
        if self.format.is_empty() || !self.payload.is_object() {
            return Err(Error::InvalidInput(
                "prompt requires a format and object payload".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_options_and_devices_are_rejected() {
        assert!("cuda:-1".parse::<DeviceConfig>().is_err());
        assert!("gpu".parse::<DeviceConfig>().is_err());
        assert_eq!(
            "metal:0".parse::<DeviceConfig>().unwrap(),
            DeviceConfig::Metal(0)
        );
        assert!(
            SynthesisOptions {
                top_p: f64::NAN,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(serde_json::from_str::<SynthesisOptions>(r#"{"temprature":0.5}"#).is_err());
    }
}
