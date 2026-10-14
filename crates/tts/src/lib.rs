//! Unified local-model speech synthesis. See the model crates for low-level APIs.

pub use tts_core::*;

#[cfg(any(feature = "qwen", feature = "cosyvoice"))]
mod engine;
#[cfg(any(feature = "qwen", feature = "cosyvoice"))]
mod fingerprint;
#[cfg(feature = "qwen")]
mod qwen_prompt;

#[cfg(any(feature = "qwen", feature = "cosyvoice"))]
pub use engine::Engine;

#[cfg(feature = "cosyvoice")]
pub use tts_cosyvoice as cosyvoice;
#[cfg(feature = "qwen")]
pub use tts_qwen as qwen;
