//! CosyVoice 3 inference implemented with Candle tensors.
//!
//! The model separates Qwen2 speech-token generation, conditional flow matching,
//! and the HiFT vocoder. Audio feature extraction is an optional ONNX frontend.
//! Model weights are external; use the conversion script before loading.
//! Numerical equivalence with the upstream Python implementation requires shared
//! random tensors, input tokens, precision, and intermediate-output comparisons.

pub mod activations;
pub mod config;
pub mod flow;
pub mod frontend;
pub mod llm;
mod model;
pub mod vocoder;

pub use config::{CosyVoice3Config, SAMPLE_RATE, SamplingConfig};
#[cfg(feature = "onnx")]
pub use frontend::CosyVoice3Frontend;
pub use model::{CosyVoice3, PromptFeatures, SynthesisMode, SynthesisOptions};
