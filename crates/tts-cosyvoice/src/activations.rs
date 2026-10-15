//! CosyVoice3 Activation Functions
//!
//! Implements specialized activation functions used in HiFT-GAN vocoder.

use candle::{Module, Result, Tensor};
use candle_nn::VarBuilder;

/// Snake activation function: x + (1/α) * sin²(αx)
///
/// This activation is used in HiFi-GAN style vocoders for better
/// audio synthesis quality.
///
/// Reference: Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders
#[derive(Debug, Clone)]
pub struct Snake {
    alpha: Tensor,
    eps: f64,
}

impl Snake {
    /// Create a new Snake activation with learnable alpha parameter.
    ///
    /// # Arguments
    /// * `channels` - Number of channels (alpha is per-channel)
    /// * `vb` - VarBuilder for loading weights
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get((channels,), "alpha")?;
        Ok(Self { alpha, eps: 1e-9 })
    }

    /// Create Snake with initialized alpha values (for testing or default init).
    pub fn with_alpha(alpha: Tensor) -> Self {
        Self { alpha, eps: 1e-9 }
    }
}

impl Module for Snake {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, C, T]
        // alpha: [C] -> [1, C, 1]
        let alpha = self.alpha.unsqueeze(0)?.unsqueeze(2)?;

        // sin²(αx)
        let ax = x.broadcast_mul(&alpha)?;
        let sin_ax = ax.sin()?;
        let sin_sq = (&sin_ax * &sin_ax)?;

        // x + (1/α) * sin²(αx)
        let inv_alpha = (1.0 / (&alpha + self.eps)?)?;
        let term = sin_sq.broadcast_mul(&inv_alpha)?;
        x + term
    }
}

/// SnakeBeta activation: x + (1/β) * sin²(αx)
///
/// Variant where alpha and beta can be different (used in BigVGAN).
#[derive(Debug, Clone)]
pub struct SnakeBeta {
    alpha: Tensor,
    beta: Tensor,
    eps: f64,
}

impl SnakeBeta {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get((channels,), "alpha")?;
        let beta = vb.get((channels,), "beta")?;
        Ok(Self {
            alpha,
            beta,
            eps: 1e-9,
        })
    }
}

impl Module for SnakeBeta {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let alpha = self.alpha.unsqueeze(0)?.unsqueeze(2)?;
        let beta = self.beta.unsqueeze(0)?.unsqueeze(2)?;

        let ax = x.broadcast_mul(&alpha)?;
        let sin_ax = ax.sin()?;
        let sin_sq = (&sin_ax * &sin_ax)?;

        let inv_beta = (1.0 / (&beta + self.eps)?)?;
        let term = sin_sq.broadcast_mul(&inv_beta)?;
        x + term
    }
}

/// ELU activation (used in F0 predictor)
#[derive(Debug, Clone, Copy, Default)]
pub struct Elu {
    alpha: f64,
}

impl Elu {
    pub fn new() -> Self {
        Self { alpha: 1.0 }
    }

    pub fn with_alpha(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Module for Elu {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.elu(self.alpha)
    }
}

/// Swish/SiLU activation: x * sigmoid(x)
#[derive(Debug, Clone, Copy, Default)]
pub struct Swish;

impl Module for Swish {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.silu()
    }
}

/// LeakyReLU activation
#[derive(Debug, Clone, Copy)]
pub struct LeakyReLU {
    negative_slope: f64,
}

impl LeakyReLU {
    pub fn new(negative_slope: f64) -> Self {
        Self { negative_slope }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self {
            negative_slope: 0.01,
        }
    }
}

impl Module for LeakyReLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let zeros = x.zeros_like()?;
        let max_val = x.maximum(&zeros)?;
        let min_val = x.minimum(&zeros)?;
        max_val + (min_val * self.negative_slope)?
    }
}

/// Tanh activation wrapper
#[derive(Debug, Clone, Copy, Default)]
pub struct Tanh;

impl Module for Tanh {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.tanh()
    }
}

/// Mish activation: x * tanh(softplus(x))
/// Used in CausalConvPositionEmbedding
#[derive(Debug, Clone, Copy, Default)]
pub struct Mish;

impl Module for Mish {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        use candle::DType;
        let dtype = x.dtype();
        if !matches!(dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64) {
            candle::bail!("Mish requires floating-point input");
        }
        let x = match dtype {
            DType::F16 | DType::BF16 => x.to_dtype(DType::F32)?,
            _ => x.clone(),
        };
        let output = if x.device().is_cpu() {
            x.contiguous()?.apply_op1_no_bwd(&StableMish)?
        } else {
            mish_tensor(&x)?
        };
        output.to_dtype(dtype)
    }
}

// Keep accelerator values on their original device. exp(-abs(x)) cannot
// overflow. In the negative tail, log(1+y) would discard y; a fifth-order
// log1p polynomial below x=-4 has <7e-12 truncation error before tanh.
fn mish_tensor(x: &Tensor) -> Result<Tensor> {
    let y = x.abs()?.neg()?.exp()?;
    let softplus = ((&y + 1.0)?.log()? + x.maximum(&x.zeros_like()?)?)?;
    let tail = ((&y * 0.2)? - 0.25)?;
    let tail = ((tail * &y)? + (1.0 / 3.0))?;
    let tail = ((tail * &y)? - 0.5)?;
    let tail = (((tail * &y)? + 1.0)? * &y)?;
    let softplus = x.lt(-4.0)?.where_cond(&tail, &softplus)?;
    x * softplus.tanh()?
}

struct StableMish;

impl candle::CustomOp1 for StableMish {
    fn name(&self) -> &'static str {
        "cosyvoice-mish"
    }

    fn cpu_fwd(
        &self,
        storage: &candle::CpuStorage,
        layout: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        let Some((start, end)) = layout.contiguous_offsets() else {
            candle::bail!("Mish requires contiguous CPU storage");
        };
        let output = match storage {
            candle::CpuStorage::F32(values) => candle::CpuStorage::F32(
                values[start..end]
                    .iter()
                    .map(|&x| {
                        // For x > 20 the F32 result is exactly x. This also avoids
                        // an overflowing exp for finite large positive inputs.
                        if x > 20.0 {
                            x
                        } else {
                            x * x.exp().ln_1p().tanh()
                        }
                    })
                    .collect(),
            ),
            candle::CpuStorage::F64(values) => candle::CpuStorage::F64(
                values[start..end]
                    .iter()
                    .map(|&x| {
                        if x > 40.0 {
                            x
                        } else {
                            x * x.exp().ln_1p().tanh()
                        }
                    })
                    .collect(),
            ),
            _ => candle::bail!("Mish CPU storage requires float32 or float64"),
        };
        Ok((output, layout.shape().clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    #[test]
    fn mish_preserves_the_negative_tail_and_finite_positive_extremes() -> Result<()> {
        // Independent pinned PyTorch 2.3.1 nn.Mish CPU outputs. In particular,
        // log(1 + exp(x)) incorrectly becomes zero for x <= -18 in float32.
        let inputs = [
            -100f32,
            -20.,
            -17.,
            -16.,
            -10.,
            -3.,
            -1.,
            0.,
            1.,
            3.,
            10.,
            20.,
            100.,
            f32::MAX,
        ];
        let expected = [
            -3.784e-42f32,
            -4.122_307_5e-8,
            -7.037_895e-7,
            -1.800_562_7e-6,
            -4.539_890_2e-4,
            -0.145_647_47,
            -0.303_401_44,
            0.,
            0.865_098_36,
            2.986_535,
            10.,
            20.,
            100.,
            f32::MAX,
        ];
        let x = Tensor::from_slice(&inputs, (2, 7), &Device::Cpu)?;
        // Execute both the CPU kernel and the portable tensor expression on
        // CPU, independently of whether an accelerator is available in CI.
        for output in [Mish.forward(&x)?, mish_tensor(&x)?] {
            let values = output.flatten_all()?.to_vec1::<f32>()?;
            for (value, expected) in values.iter().zip(expected) {
                assert!(value.is_finite());
                assert!((value - expected).abs() <= expected.abs() * 3e-6 + f32::from_bits(1));
            }
        }
        Ok(())
    }

    #[test]
    fn test_snake_activation() -> Result<()> {
        let device = Device::Cpu;
        let channels = 4;

        // Create alpha tensor
        let alpha = Tensor::ones((channels,), DType::F32, &device)?;
        let snake = Snake::with_alpha(alpha);

        // Test input [batch=1, channels=4, time=3]
        let x = Tensor::randn(0f32, 1.0, (1, channels, 3), &device)?;
        let output = snake.forward(&x)?;

        assert_eq!(output.dims(), &[1, channels, 3]);
        Ok(())
    }

    #[test]
    fn test_leaky_relu() -> Result<()> {
        let device = Device::Cpu;
        let lrelu = LeakyReLU::new(0.1);

        let x = Tensor::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], 5, &device)?;
        let output = lrelu.forward(&x)?;
        let output_vec: Vec<f32> = output.to_vec1()?;

        // Check negative values are scaled
        assert!((output_vec[0] - (-0.2)).abs() < 1e-5);
        assert!((output_vec[1] - (-0.1)).abs() < 1e-5);
        // Check positive values unchanged
        assert!((output_vec[3] - 1.0).abs() < 1e-5);
        assert!((output_vec[4] - 2.0).abs() < 1e-5);
        Ok(())
    }
}
