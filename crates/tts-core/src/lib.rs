//! Portable input validation, audio processing, and versioned voice prompts.
//!
//! Model-specific payloads remain owned by their model crate. A speaker vector
//! from one family is not interchangeable with another model's conditioning.

mod audio;
mod contract;

pub use audio::Audio;
pub use contract::{DeviceConfig, ModelKind, SynthesisOptions, VoicePrompt};

/// An error that can cross language bindings without losing its category.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The caller supplied an invalid value.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// This build or model does not offer the requested operation.
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    /// A model could not load or run.
    #[error("model error: {0}")]
    Model(String),
    /// A filesystem operation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// A prompt or configuration is not valid JSON.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

/// Result used at public API and binding boundaries.
pub type Result<T> = std::result::Result<T, Error>;
