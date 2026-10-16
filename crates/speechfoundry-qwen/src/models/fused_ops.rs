//! Residual RMS normalization using Candle's portable backend primitives.
use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{RmsNorm, VarBuilder, rms_norm};

/// Residual RMS normalization shared by all native backends.
pub struct FusedRmsNorm {
    inner: RmsNorm,
}
impl FusedRmsNorm {
    /// Load learned scale with the checkpoint's epsilon.
    pub fn load(hidden_size: usize, eps: f64, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            inner: rms_norm(hidden_size, eps, vb)?,
        })
    }
    /// Normalize one tensor.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.inner.forward(x)?)
    }
    /// Add the residual, returning normalized and unnormalized sums.
    pub fn forward_residual(&self, x: &Tensor, residual: &Tensor) -> Result<(Tensor, Tensor)> {
        let sum = (x + residual)?;
        Ok((self.inner.forward(&sum)?, sum))
    }
}
