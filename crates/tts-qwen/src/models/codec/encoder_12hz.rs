//! Speech tokenizer encoder (12Hz Mimi-based)
//!
//! Encodes raw audio into 12Hz, 16-codebook discrete codec codes for ICL
//! voice cloning.
//!
//! The Qwen3-TTS speech tokenizer uses a standard HuggingFace Mimi model
//! for encoding. We construct only the encoder path (SEANet encoder →
//! transformer → downsample → quantizer) from candle's public Mimi
//! components, skipping the decoder entirely.
//!
//! ## Weight key mapping
//!
//! The HF model wraps all encoder components under an `encoder.` prefix:
//! - `encoder.encoder.layers.*`       → SEANet convolutional encoder
//! - `encoder.encoder_transformer.*`  → Streaming transformer
//! - `encoder.downsample.*`           → 25Hz→12.5Hz frame rate conversion
//! - `encoder.quantizer.*`            → Split residual vector quantizer
//!
//! Stripping this prefix yields keys compatible with candle's Mimi format.

use anyhow::Result;
use candle_core::{DType, Device, Module, StreamingModule, Tensor};
use candle_transformers::models::mimi;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

use crate::audio::AudioBuffer;

fn interleave_rope_weight(weight: &Tensor, heads: usize) -> Result<Tensor> {
    let (output, input) = weight.dims2()?;
    anyhow::ensure!(
        heads > 0 && output % heads == 0 && (output / heads).is_multiple_of(2),
        "RoPE projection must contain an even number of channels per attention head"
    );
    Ok(weight
        .reshape((heads, 2, output / heads / 2, input))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((output, input))?)
}

/// Mimi-based encoder producing 12Hz, 16-codebook codes from raw audio.
///
/// Constructs the encoder-only path from candle's Mimi components (no
/// decoder). Used by ICL voice cloning to tokenize reference audio.
pub struct Encoder12Hz {
    encoder: mimi::seanet::SeaNetEncoder,
    // RefCell because ProjectedTransformer::forward/reset_state need &mut self
    encoder_transformer: RefCell<mimi::transformer::ProjectedTransformer>,
    downsample: mimi::conv::ConvDownsample1d,
    quantizer: mimi::quantization::SplitResidualVectorQuantizer,
    device: Device,
}

impl Encoder12Hz {
    /// Load from a safetensors file (e.g., `speech_tokenizer/model.safetensors`).
    pub fn from_safetensors(path: &Path, device: &Device) -> Result<Self> {
        let raw_tensors: HashMap<String, Tensor> = candle_core::safetensors::load(path, device)?;
        Self::from_weights(&raw_tensors, device)
    }

    /// Load from pre-loaded weight tensors.
    ///
    /// Expects HF-format keys with `encoder.*` prefix. Strips the prefix
    /// and constructs encoder-only components.
    pub fn from_weights(weights: &HashMap<String, Tensor>, device: &Device) -> Result<Self> {
        // The HF Qwen3-TTS speech tokenizer wraps all encoder components
        // under an "encoder." prefix. Strip it to match candle's Mimi format:
        //   encoder.encoder.layers.0.conv.weight      → encoder.layers.0.conv.weight
        //   encoder.encoder_transformer.layers.0.*     → encoder_transformer.layers.0.*
        //   encoder.quantizer.*                        → quantizer.*
        //   encoder.downsample.conv.weight             → downsample.conv.weight
        let mut encoder_weights: HashMap<String, Tensor> = weights
            .iter()
            .filter_map(|(k, v)| {
                k.strip_prefix("encoder.")
                    .map(|stripped| (stripped.to_string(), v.clone()))
            })
            .collect();

        if encoder_weights.is_empty() {
            anyhow::bail!("No encoder keys found (expected keys starting with 'encoder.')");
        }

        let cfg = mimi::Config::v0_1(Some(16));
        // Hugging Face stores RoPE channels as [real_half, imaginary_half],
        // while Candle Mimi rotates adjacent pairs. Reorder only Q/K output
        // channels within each attention head; V and output projections stay
        // in their original layout because the attention scores are invariant
        // under this shared Q/K permutation.
        for layer in 0..cfg.transformer.num_layers {
            for projection in ["q_proj", "k_proj"] {
                let name =
                    format!("encoder_transformer.layers.{layer}.self_attn.{projection}.weight");
                let weight = encoder_weights
                    .get(&name)
                    .ok_or_else(|| anyhow::anyhow!("missing encoder weight {name}"))?;
                let permuted = interleave_rope_weight(weight, cfg.transformer.num_heads)?;
                encoder_weights.insert(name, permuted);
            }
        }
        let vb = candle_nn::VarBuilder::from_tensors(encoder_weights, DType::F32, device);

        let encoder = mimi::seanet::SeaNetEncoder::new(&cfg.seanet, vb.pp("encoder"))?;
        let encoder_transformer = mimi::transformer::ProjectedTransformer::new(
            cfg.seanet.dimension,
            &[cfg.seanet.dimension],
            &cfg.transformer,
            vb.pp("encoder_transformer"),
        )?;

        // SEANet produces 25Hz (24000 / (8*6*5*4) = 25), downsample to 12.5Hz
        let encoder_frame_rate =
            cfg.sample_rate / cfg.seanet.ratios.iter().product::<usize>() as f64;
        let downsample_stride = (encoder_frame_rate / cfg.frame_rate) as usize;
        let downsample = mimi::conv::ConvDownsample1d::new(
            downsample_stride,
            cfg.seanet.dimension,
            /* causal */ true,
            /* learnt */ true,
            vb.pp("downsample"),
        )?;

        let quantizer = mimi::quantization::SplitResidualVectorQuantizer::new(
            cfg.quantizer_dim,
            Some(cfg.seanet.dimension),
            Some(cfg.seanet.dimension),
            cfg.quantizer_n_q,
            cfg.quantizer_bins,
            vb.pp("quantizer"),
        )?;

        Ok(Self {
            encoder,
            encoder_transformer: RefCell::new(encoder_transformer),
            downsample,
            quantizer,
            device: device.clone(),
        })
    }

    /// Encode audio to discrete codec codes.
    ///
    /// Input: `AudioBuffer` at 24kHz.
    /// Output: `Tensor` of shape `[T_frames, 16]` containing discrete codes (u32).
    pub fn encode(&self, audio: &AudioBuffer) -> Result<Tensor> {
        let [_, _, features] = self.features(audio)?;
        self.encode_features(&features)
    }

    /// Export encoder stages for pinned Python numerical comparisons.
    /// Continuous outputs use `[batch, channels, time]`; codes use `[time, 16]`.
    pub fn encode_with_intermediates(
        &self,
        audio: &AudioBuffer,
    ) -> Result<Vec<(&'static str, Tensor)>> {
        let [seanet, transformer, downsample] = self.features(audio)?;
        let codes = self.encode_features(&downsample)?;
        Ok(vec![
            ("seanet", seanet),
            ("transformer", transformer),
            ("downsample", downsample),
            ("reference_audio_codes", codes),
        ])
    }

    fn features(&self, audio: &AudioBuffer) -> Result<[Tensor; 3]> {
        anyhow::ensure!(
            audio.sample_rate == 24_000
                && !audio.samples.is_empty()
                && audio.samples.iter().all(|value| value.is_finite()),
            "codec encoder requires finite nonempty mono PCM at 24000 Hz"
        );
        let samples = &audio.samples;

        // Mimi expects [batch, channels, samples] = [1, 1, N]
        let input = Tensor::from_vec(samples.to_vec(), (1, 1, samples.len()), &self.device)?;
        let input = input.to_dtype(DType::F32)?;

        // SEANet encoder: [1, 1, N] → [1, 512, T_25hz]
        let seanet = self.encoder.forward(&input)?;

        // Transformer: [1, 512, T_25hz] → [1, T_25hz, 512] (conv_layout transposes internally)
        let mut transformer = self.encoder_transformer.borrow_mut();
        transformer.reset_state();
        let xs = transformer.forward(&seanet)?;
        let xs = &xs[0];

        // Downsample 25Hz → 12.5Hz
        let downsample = xs.apply(&self.downsample)?;

        Ok([seanet, xs.clone(), downsample])
    }

    fn encode_features(&self, features: &Tensor) -> Result<Tensor> {
        // Quantize: [1, 512, T_12hz] → [1, 16, T_12hz]
        let codes = self.quantizer.encode(features)?;

        // Transpose to [T_frames, 16]
        let codes = codes.squeeze(0)?.transpose(0, 1)?;
        Ok(codes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_projection_permutation_preserves_rotary_attention() -> Result<()> {
        let device = Device::Cpu;
        let weights = Tensor::arange(1f32, 9f32, &device)?.reshape((8, 1))?;
        let split = weights.reshape((1, 2, 1, 4))?;
        let paired = interleave_rope_weight(&weights, 2)?.reshape((1, 2, 1, 4))?;
        let angles = Tensor::new(&[[0.4f32, 0.9]], &device)?;
        let cosine = angles.cos()?;
        let sine = angles.sin()?;
        let split_rotated = candle_nn::rotary_emb::rope(&split, &cosine, &sine)?;
        let paired_rotated = candle_nn::rotary_emb::rope_i(&paired, &cosine, &sine)?;
        let expected = interleave_rope_weight(&split_rotated.reshape((8, 1))?, 2)?;
        let difference = (paired_rotated.flatten_all()? - expected.flatten_all()?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(difference < 1e-6);
        Ok(())
    }

    #[test]
    fn test_strip_encoder_prefix() {
        let mut weights = HashMap::new();
        let dummy = Tensor::zeros((1,), DType::F32, &Device::Cpu).unwrap();
        weights.insert(
            "encoder.encoder.layers.0.conv.weight".to_string(),
            dummy.clone(),
        );
        weights.insert(
            "encoder.encoder_transformer.layers.0.self_attn.q_proj.weight".to_string(),
            dummy.clone(),
        );
        weights.insert(
            "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum"
                .to_string(),
            dummy.clone(),
        );
        weights.insert("encoder.downsample.conv.weight".to_string(), dummy.clone());
        // Decoder keys should be excluded
        weights.insert("decoder.decoder.0.conv.weight".to_string(), dummy);

        let stripped: HashMap<String, Tensor> = weights
            .iter()
            .filter_map(|(k, v)| {
                k.strip_prefix("encoder.")
                    .map(|s| (s.to_string(), v.clone()))
            })
            .collect();

        assert!(stripped.contains_key("encoder.layers.0.conv.weight"));
        assert!(stripped.contains_key("encoder_transformer.layers.0.self_attn.q_proj.weight"));
        assert!(stripped.contains_key(
            "quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum"
        ));
        assert!(stripped.contains_key("downsample.conv.weight"));
        assert!(!stripped.contains_key("decoder.decoder.0.conv.weight"));
        assert_eq!(stripped.len(), 4);
    }
}
