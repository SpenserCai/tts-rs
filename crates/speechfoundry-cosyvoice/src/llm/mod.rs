//! CosyVoice3 Language Model
//!
//! Based on Qwen2 architecture for autoregressive speech token generation.

pub mod cosyvoice3_lm;
mod sampling;

pub use cosyvoice3_lm::CosyVoice3LM;
