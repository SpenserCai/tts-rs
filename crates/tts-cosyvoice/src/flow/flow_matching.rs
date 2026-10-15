//! Conditional Flow Matching (CFM) for CosyVoice3
//!
//! Implements the flow matching algorithm for mel spectrogram generation.
//! Uses Euler ODE solver with Classifier-Free Guidance (CFG).

use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use std::f64::consts::PI;

use super::dit::DiT;
use crate::config::CFMConfig;

/// Generate deterministic pseudo-random noise using a simple LCG (Linear Congruential Generator)
/// This fallback is deterministic but does not reproduce PyTorch RNG; use exported noise for parity.
fn generate_fixed_noise(
    shape: (usize, usize, usize),
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let (batch, channels, length) = shape;
    let total = batch * channels * length;

    // Use Box-Muller transform with deterministic sequence
    // LCG parameters (same as glibc)
    let mut state: u64 = 0; // seed = 0
    let a: u64 = 1103515245;
    let c: u64 = 12345;
    let m: u64 = 1 << 31;

    let mut values = Vec::with_capacity(total);
    for _ in 0..total.div_ceil(2) {
        // Generate two uniform values
        state = (a.wrapping_mul(state).wrapping_add(c)) % m;
        let u1 = (state as f64) / (m as f64);
        state = (a.wrapping_mul(state).wrapping_add(c)) % m;
        let u2 = (state as f64) / (m as f64);

        // Box-Muller transform
        let u1_safe = u1.max(1e-10);
        let r = (-2.0 * u1_safe.ln()).sqrt();
        let theta = 2.0 * PI * u2;

        values.push((r * theta.cos()) as f32);
        if values.len() < total {
            values.push((r * theta.sin()) as f32);
        }
    }
    values.truncate(total);

    Tensor::from_vec(values, (batch, channels, length), device)?.to_dtype(dtype)
}

/// Construct the sampling grid before applying the scheduler.
///
/// Float32 follows the pinned PyTorch CPU linspace contract: a float32 step,
/// symmetric endpoint evaluation and one rounding for each multiply-add.
/// See PyTorch's `aten/src/ATen/native/cpu/RangeFactoriesKernel.cpp` and
/// `LICENSE-PYTORCH`. Other dtypes retain the existing float32-staged grid;
/// this function does not assert PyTorch parity for those precision modes.
fn uniform_time_grid(n_steps: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    if n_steps == 0 {
        candle::bail!("flow matching requires at least one Euler step");
    }
    let points = n_steps
        .checked_add(1)
        .ok_or_else(|| candle::Error::Msg("flow time grid length overflows usize".into()))?;
    let values: Vec<f32> = if dtype == DType::F32 {
        let step = 1.0f32 / n_steps as f32;
        let halfway = points / 2;
        (0..points)
            .map(|i| {
                if i < halfway {
                    step.mul_add(i as f32, 0.0)
                } else {
                    (-step).mul_add((n_steps - i) as f32, 1.0)
                }
            })
            .collect()
    } else {
        let step = 1.0 / n_steps as f64;
        (0..points).map(|i| (i as f64 * step) as f32).collect()
    };
    Tensor::from_vec(values, points, device)?.to_dtype(dtype)
}

/// Causal Conditional Flow Matching
///
/// Uses DiT as the estimator and Euler ODE solver for sampling.
/// Implements Classifier-Free Guidance (CFG) for improved quality.
#[derive(Debug)]
pub struct CausalConditionalCFM {
    estimator: DiT,
    #[allow(dead_code)]
    sigma_min: f64,
    t_scheduler: String,
    inference_cfg_rate: f64,
    /// Pre-generated random noise for reproducibility (matches Python's set_all_random_seed(0))
    rand_noise: Option<Tensor>,
}

impl CausalConditionalCFM {
    pub fn new(estimator: DiT, config: CFMConfig) -> Self {
        Self {
            estimator,
            sigma_min: config.sigma_min,
            t_scheduler: config.t_scheduler,
            inference_cfg_rate: config.inference_cfg_rate,
            rand_noise: None,
        }
    }

    /// Create with pre-generated random noise for exact reproducibility
    pub fn new_with_rand_noise(estimator: DiT, config: CFMConfig, rand_noise: Tensor) -> Self {
        Self {
            estimator,
            sigma_min: config.sigma_min,
            t_scheduler: config.t_scheduler,
            inference_cfg_rate: config.inference_cfg_rate,
            rand_noise: Some(rand_noise),
        }
    }

    /// Sample from the flow model using Euler ODE solver with CFG
    ///
    /// # Arguments
    /// * `mu` - [B, 80, T] condition from token embedding
    /// * `mask` - [B, 1, T] mask
    /// * `n_timesteps` - Number of ODE steps (default: 10)
    /// * `spks` - [B, 80] speaker embedding
    /// * `cond` - [B, 80, T] prompt mel condition
    /// * `streaming` - Whether in streaming mode
    pub fn forward(
        &self,
        mu: &Tensor,
        mask: &Tensor,
        n_timesteps: usize,
        spks: &Tensor,
        cond: &Tensor,
        streaming: bool,
    ) -> Result<Tensor> {
        if n_timesteps == 0 {
            candle::bail!("flow matching requires at least one Euler step");
        }
        let (batch, _, seq_len) = mu.dims3()?;
        let device = mu.device();
        let dtype = mu.dtype();

        // Use pre-generated noise if available, otherwise generate fixed noise
        let z = if let Some(ref rand_noise) = self.rand_noise {
            // Use pre-generated noise (matches Python's CausalConditionalCFM)
            // rand_noise shape: [1, 80, 15000], we need [batch, 80, seq_len]
            rand_noise
                .narrow(2, 0, seq_len)?
                .to_device(device)?
                .to_dtype(dtype)?
        } else {
            // Fallback to generated noise (not recommended for exact reproducibility)
            generate_fixed_noise((batch, 80, seq_len), device, dtype)?
        };

        // Time step scheduling
        let t_span = self.get_t_span(n_timesteps, device, dtype)?;

        // Euler solver with CFG
        self.solve_euler(&z, &t_span, mu, mask, spks, cond, streaming)
    }

    /// Euler ODE solver with Classifier-Free Guidance
    ///
    /// CFG formula: dphi_dt = (1 + cfg_rate) * conditional - cfg_rate * unconditional
    #[allow(clippy::too_many_arguments)]
    fn solve_euler(
        &self,
        x: &Tensor,
        t_span: &Tensor,
        mu: &Tensor,
        mask: &Tensor,
        spks: &Tensor,
        cond: &Tensor,
        streaming: bool,
    ) -> Result<Tensor> {
        let n_steps = t_span.dim(0)? - 1;
        let device = x.device();
        let dtype = x.dtype();
        let seq_len = x.dim(2)?;

        let mut x = x.clone();
        let mut t = t_span.i(0)?;

        // Check if CFG should be used (cfg_rate > 0)
        let use_cfg = self.inference_cfg_rate > 0.0;

        for step in 0..n_steps {
            let t_next = t_span.i(step + 1)?;
            let dt = (&t_next - &t)?;

            let dphi_dt = if use_cfg {
                // Classifier-Free Guidance: run estimator with batch size 2
                // [0] = conditional (with mu, spks, cond)
                // [1] = unconditional (with zeros)

                // Create batched inputs for CFG
                let x_in = Tensor::cat(&[&x, &x], 0)?; // [2, 80, T]
                let mask_in = Tensor::cat(&[mask, mask], 0)?; // [2, 1, T]

                // Conditional uses real values, unconditional uses zeros
                let zeros_mu = Tensor::zeros((1, 80, seq_len), dtype, device)?;
                let mu_in = Tensor::cat(&[mu, &zeros_mu], 0)?; // [2, 80, T]

                let zeros_spks = Tensor::zeros((1, 80), dtype, device)?;
                let spks_in = Tensor::cat(&[spks, &zeros_spks], 0)?; // [2, 80]

                let zeros_cond = Tensor::zeros((1, 80, seq_len), dtype, device)?;
                let cond_in = Tensor::cat(&[cond, &zeros_cond], 0)?; // [2, 80, T]

                let t_in = Tensor::cat(&[&t.unsqueeze(0)?, &t.unsqueeze(0)?], 0)?; // [2]

                // Estimator forward pass with batched inputs
                let dphi_dt_combined = self.estimator.forward(
                    &x_in, &mask_in, &mu_in, &t_in, &spks_in, &cond_in, streaming,
                )?;

                // Split into conditional and unconditional predictions
                let dphi_dt_cond = dphi_dt_combined.narrow(0, 0, 1)?; // [1, 80, T]
                let dphi_dt_uncond = dphi_dt_combined.narrow(0, 1, 1)?; // [1, 80, T]

                // Apply CFG: dphi_dt = (1 + cfg_rate) * conditional - cfg_rate * unconditional
                let cfg_rate = self.inference_cfg_rate;
                ((&dphi_dt_cond * (1.0 + cfg_rate))? - (&dphi_dt_uncond * cfg_rate)?)?
            } else {
                // No CFG, just conditional prediction
                let t_batch = t.unsqueeze(0)?; // [1]
                self.estimator
                    .forward(&x, mask, mu, &t_batch, spks, cond, streaming)?
            };

            // Euler step: x = x + dphi_dt * dt
            let dt_broadcast = dt.unsqueeze(0)?.unsqueeze(0)?;
            x = (x + dphi_dt.broadcast_mul(&dt_broadcast)?)?;

            t = t_next;
        }

        Ok(x)
    }

    /// Get time span with optional cosine schedule
    fn get_t_span(&self, n_steps: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        let t = uniform_time_grid(n_steps, device, dtype)?;

        if self.t_scheduler == "cosine" {
            // t = 1 - cos(t * π/2)
            let pi_half = PI / 2.0;
            let cos_t = (t * pi_half)?.cos()?;
            (1.0 - cos_t)?.to_dtype(dtype)
        } else {
            Ok(t)
        }
    }

    /// Get the estimator (DiT)
    pub fn estimator(&self) -> &DiT {
        &self.estimator
    }
}

/// CausalMaskedDiffWithDiT - Complete Flow Decoder
///
/// Combines PreLookaheadLayer, token embedding, and CFM for mel generation.
/// Follows the official implementation structure:
/// - input_embedding: vocab_size -> input_size
/// - spk_embed_affine_layer: spk_embed_dim(192) -> output_size(80)
/// - pre_lookahead_layer: with token_mel_ratio(2x) upsampling
/// - cfm: Conditional Flow Matching with DiT estimator
#[derive(Debug)]
pub struct CausalMaskedDiffWithDiT {
    input_embedding: candle_nn::Embedding,
    spk_embed_affine_layer: candle_nn::Linear,
    pre_lookahead_layer: super::pre_lookahead::PreLookaheadLayer,
    cfm: CausalConditionalCFM,
    #[allow(dead_code)]
    vocab_size: usize,
    #[allow(dead_code)]
    token_mel_ratio: usize,
    #[allow(dead_code)]
    pre_lookahead_len: usize,
    output_size: usize,
}

impl CausalMaskedDiffWithDiT {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab_size: usize,
        input_size: usize,
        output_size: usize,
        spk_embed_dim: usize,
        token_mel_ratio: usize,
        pre_lookahead_len: usize,
        estimator: DiT,
        cfm_config: CFMConfig,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        Self::new_with_rand_noise(
            vocab_size,
            input_size,
            output_size,
            spk_embed_dim,
            token_mel_ratio,
            pre_lookahead_len,
            estimator,
            cfm_config,
            vb,
            None,
        )
    }

    /// Create with pre-generated random noise for exact reproducibility
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_rand_noise(
        vocab_size: usize,
        input_size: usize,
        output_size: usize,
        spk_embed_dim: usize,
        token_mel_ratio: usize,
        pre_lookahead_len: usize,
        estimator: DiT,
        cfm_config: CFMConfig,
        vb: candle_nn::VarBuilder,
        rand_noise: Option<Tensor>,
    ) -> Result<Self> {
        // Input embedding: vocab_size -> input_size (not output_size)
        let input_embedding =
            candle_nn::embedding(vocab_size, input_size, vb.pp("input_embedding"))?;

        // Speaker embedding affine layer: spk_embed_dim(192) -> output_size(80)
        let spk_embed_affine_layer =
            candle_nn::linear(spk_embed_dim, output_size, vb.pp("spk_embed_affine_layer"))?;

        // PreLookahead layer
        let pre_lookahead_layer = super::pre_lookahead::PreLookaheadLayer::new(
            input_size,
            1024, // hidden_dim
            output_size,
            pre_lookahead_len,
            token_mel_ratio,
            vb.pp("pre_lookahead_layer"),
        )?;

        // CFM with optional pre-generated noise
        let cfm = if let Some(noise) = rand_noise {
            CausalConditionalCFM::new_with_rand_noise(estimator, cfm_config, noise)
        } else {
            CausalConditionalCFM::new(estimator, cfm_config)
        };

        Ok(Self {
            input_embedding,
            spk_embed_affine_layer,
            pre_lookahead_layer,
            cfm,
            vocab_size,
            token_mel_ratio,
            pre_lookahead_len,
            output_size,
        })
    }

    /// Inference: speech tokens -> mel spectrogram
    ///
    /// # Arguments
    /// * `speech_tokens` - [B, T] speech token indices
    /// * `prompt_tokens` - [B, T'] prompt speech tokens
    /// * `prompt_feat` - [B, T'*2, 80] prompt mel features (already upsampled)
    /// * `embedding` - [B, 192] raw speaker embedding (will be normalized and projected)
    /// * `n_timesteps` - Number of ODE steps
    /// * `streaming` - Whether in streaming mode
    ///
    /// # Returns
    /// Generated mel spectrogram [B, 80, T_gen] (only the generated part, not including prompt)
    pub fn inference(
        &self,
        speech_tokens: &Tensor,
        prompt_tokens: &Tensor,
        prompt_feat: &Tensor,
        embedding: &Tensor,
        n_timesteps: usize,
        streaming: bool,
    ) -> Result<Tensor> {
        let (batch, _token_len) = speech_tokens.dims2()?;

        // 1. Speaker embedding: normalize and project
        // embedding = F.normalize(embedding, dim=1)
        // embedding = self.spk_embed_affine_layer(embedding)
        let embedding_norm = {
            let norm = embedding.sqr()?.sum_keepdim(1)?.sqrt()?;
            embedding.broadcast_div(&norm.clamp(1e-12, f64::INFINITY)?)? // L2 normalize
        };
        let embedding_proj = self.spk_embed_affine_layer.forward(&embedding_norm)?; // [B, 80]

        // 2. Concat prompt tokens and speech tokens
        let all_tokens = Tensor::cat(&[prompt_tokens, speech_tokens], 1)?;

        // 3. Input embedding with clamp (matching Python: torch.clamp(token, min=0))
        // Note: tokens should already be valid, but clamp for safety
        let token_emb = self.input_embedding.forward(&all_tokens)?; // [B, T_total, input_size]

        // 4. PreLookahead layer (includes projection to output_size and 2x upsampling)
        let mu = self.pre_lookahead_layer.forward(&token_emb, None)?; // [B, T_total*2, output_size]
        let mu = mu.transpose(1, 2)?; // [B, output_size, T_total*2]

        // Calculate mel lengths
        let prompt_mel_len = prompt_feat.dim(1)?; // mel_len1 in Python
        let mel_len = mu.dim(2)?; // total mel length
        let gen_mel_len = mel_len.checked_sub(prompt_mel_len).ok_or_else(|| {
            candle::Error::Msg("prompt mel exceeds the token-aligned mel length".into())
        })?;

        // 5. Prepare mask and condition
        let mask = Tensor::ones((batch, 1, mel_len), mu.dtype(), mu.device())?;

        // Prompt mel as condition (pad to same length)
        let prompt_feat = prompt_feat.transpose(1, 2)?; // [B, output_size, T']

        let cond = if prompt_mel_len < mel_len {
            let pad = Tensor::zeros(
                (batch, self.output_size, mel_len - prompt_mel_len),
                prompt_feat.dtype(),
                prompt_feat.device(),
            )?;
            Tensor::cat(&[&prompt_feat, &pad], 2)?
        } else {
            prompt_feat.narrow(2, 0, mel_len)?
        };

        // 6. CFM sampling with projected speaker embedding
        let full_mel =
            self.cfm
                .forward(&mu, &mask, n_timesteps, &embedding_proj, &cond, streaming)?;

        // 7. Return only the generated part (excluding prompt)
        // Python: feat = feat[:, :, mel_len1:]
        full_mel.narrow(2, prompt_mel_len, gen_mel_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_grid_matches_pinned_float32_linspace() -> Result<()> {
        // Independent torch 2.3.1+cpu oracle, commit d44533f9d073df13895333e70b66f81c513c1889:
        // torch.linspace(0, 1, n + 1, dtype=torch.float32).view(torch.int32)
        // Three/seven steps catch one-sided F64 generation; fifty also catches
        // separate multiply/subtract rounding. Ten preserves the default grid.
        let cases: &[(usize, &[u32])] = &[
            (1, &[0, 0x3f800000]),
            (3, &[0, 0x3eaaaaab, 0x3f2aaaaa, 0x3f800000]),
            (
                7,
                &[
                    0, 0x3e124925, 0x3e924925, 0x3edb6db8, 0x3f124924, 0x3f36db6e, 0x3f5b6db7,
                    0x3f800000,
                ],
            ),
            (
                10,
                &[
                    0, 0x3dcccccd, 0x3e4ccccd, 0x3e99999a, 0x3ecccccd, 0x3f000000, 0x3f19999a,
                    0x3f333333, 0x3f4ccccd, 0x3f666666, 0x3f800000,
                ],
            ),
            (
                50,
                &[
                    0, 0x3ca3d70a, 0x3d23d70a, 0x3d75c28f, 0x3da3d70a, 0x3dcccccc, 0x3df5c28f,
                    0x3e0f5c29, 0x3e23d70a, 0x3e3851eb, 0x3e4ccccc, 0x3e6147ae, 0x3e75c28f,
                    0x3e851eb8, 0x3e8f5c29, 0x3e999999, 0x3ea3d70a, 0x3eae147b, 0x3eb851eb,
                    0x3ec28f5c, 0x3ecccccc, 0x3ed70a3d, 0x3ee147ae, 0x3eeb851e, 0x3ef5c28f,
                    0x3f000000, 0x3f051eb8, 0x3f0a3d71, 0x3f0f5c29, 0x3f147ae1, 0x3f19999a,
                    0x3f1eb852, 0x3f23d70a, 0x3f28f5c3, 0x3f2e147b, 0x3f333333, 0x3f3851ec,
                    0x3f3d70a4, 0x3f428f5c, 0x3f47ae15, 0x3f4ccccd, 0x3f51eb85, 0x3f570a3e,
                    0x3f5c28f6, 0x3f6147ae, 0x3f666666, 0x3f6b851f, 0x3f70a3d7, 0x3f75c28f,
                    0x3f7ae148, 0x3f800000,
                ],
            ),
        ];
        for &(steps, expected) in cases {
            let grid = uniform_time_grid(steps, &Device::Cpu, DType::F32)?.to_vec1::<f32>()?;
            assert_eq!(
                grid.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected
            );
            assert_eq!(grid[0], 0.0);
            assert_eq!(grid[steps], 1.0);
            assert!(grid.windows(2).all(|pair| pair[0] < pair[1]));
        }
        Ok(())
    }

    #[test]
    fn time_grid_rejects_zero_steps_and_length_overflow() {
        for steps in [0, usize::MAX] {
            assert!(uniform_time_grid(steps, &Device::Cpu, DType::F32).is_err());
        }
    }

    #[test]
    fn time_grid_preserves_existing_low_precision_staging() -> Result<()> {
        // These frozen values preserve the previous grid construction and cast.
        // They are regression fixtures, not half-precision upstream parity claims.
        let cases = [
            (
                DType::F16,
                [
                    0.0, 0.14282227, 0.28564453, 0.4284668, 0.57128906, 0.71435547, 0.8569336, 1.0,
                ],
            ),
            (
                DType::BF16,
                [
                    0.0, 0.14257813, 0.28515625, 0.42773438, 0.5703125, 0.71484375, 0.85546875, 1.0,
                ],
            ),
        ];
        for (dtype, expected) in cases {
            let grid = uniform_time_grid(7, &Device::Cpu, dtype)?;
            assert_eq!(grid.dtype(), dtype);
            assert_eq!(grid.to_dtype(DType::F32)?.to_vec1::<f32>()?, expected);
        }
        Ok(())
    }
}
