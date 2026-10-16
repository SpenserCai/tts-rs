//! Candle inference for Qwen3-TTS checkpoints.
//!
//! Loads local official weights and keeps model acquisition outside the inference layer.
//! See UPSTREAM.md for source provenance and validation boundaries.

pub mod audio;
pub mod generation;
pub mod models;
pub mod tokenizer;

mod engine;
pub use engine::*;
