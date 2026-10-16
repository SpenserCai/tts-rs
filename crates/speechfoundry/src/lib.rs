//! Unified local-model speech synthesis. See the model crates for low-level APIs.

pub use speechfoundry_core::*;

#[cfg(any(feature = "qwen", feature = "cosyvoice"))]
mod engine;
#[cfg(any(feature = "qwen", feature = "cosyvoice"))]
mod fingerprint;
#[cfg(feature = "qwen")]
mod qwen_prompt;

#[cfg(any(feature = "qwen", feature = "cosyvoice"))]
pub use engine::Engine;
#[cfg(any(feature = "qwen", feature = "cosyvoice"))]
pub use fingerprint::CheckpointHasher;

#[cfg(feature = "cosyvoice")]
pub use speechfoundry_cosyvoice as cosyvoice;
#[cfg(feature = "qwen")]
pub use speechfoundry_qwen as qwen;
