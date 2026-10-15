//! Stable, unweighted normalization for the acoustic transformer.
//!
//! The CPU moment reduction is adapted from PyTorch 2.3.1, commit
//! d44533f9d073df13895333e70b66f81c513c1889, in
//! `aten/src/ATen/native/cpu/moments_utils.h` and `layer_norm_kernel.cpp`.
//! See `LICENSE-PYTORCH` for the retained copyright notices and BSD license.

use candle::{CpuStorage, D, DType, Layout, Result, Shape, Tensor};

const EPSILON: f64 = 1e-6;
const LANES: usize = 8;
const CHUNK_VECTORS: usize = 16;

/// Normalize the last dimension without learned scale or bias.
pub(super) fn layer_norm_no_affine(x: &Tensor) -> Result<Tensor> {
    if x.dim(D::Minus1)? == 0 {
        candle::bail!("flow layer normalization requires a nonempty last dimension");
    }
    if !matches!(
        x.dtype(),
        DType::F16 | DType::BF16 | DType::F32 | DType::F64
    ) {
        candle::bail!("flow layer normalization requires floating-point input");
    }
    if x.device().is_cpu() && x.dtype() == DType::F32 {
        return x.contiguous()?.apply_op1_no_bwd(&StableLayerNorm);
    }

    // Accelerator tensors remain on their original device. Half precision
    // accumulates in float32; this path makes no CPU numerical-parity claim.
    let original_dtype = x.dtype();
    let x = match original_dtype {
        DType::F16 | DType::BF16 => x.to_dtype(DType::F32)?,
        _ => x.clone(),
    };
    let mean = x.mean_keepdim(D::Minus1)?;
    let centered = x.broadcast_sub(&mean)?;
    let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
    let reciprocal_std = (variance + EPSILON)?.sqrt()?.recip()?;
    centered
        .broadcast_mul(&reciprocal_std)?
        .to_dtype(original_dtype)
}

struct StableLayerNorm;

impl candle::CustomOp1 for StableLayerNorm {
    fn name(&self) -> &'static str {
        "cosyvoice-layer-norm"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let CpuStorage::F32(values) = storage else {
            candle::bail!("flow CPU layer normalization requires float32 storage");
        };
        let Some((start, end)) = layout.contiguous_offsets() else {
            candle::bail!("flow CPU layer normalization requires contiguous input");
        };
        let width = layout.dim(D::Minus1)?;
        if width == 0 {
            candle::bail!("flow layer normalization requires a nonempty last dimension");
        }
        let mut output = Vec::with_capacity(end - start);
        for row in values[start..end].chunks_exact(width) {
            let (mean, variance) = row_moments(row);
            let reciprocal_std = 1.0 / (variance + EPSILON as f32).sqrt();
            if !mean.is_finite() || !variance.is_finite() || !reciprocal_std.is_finite() {
                candle::bail!("flow layer normalization encountered non-finite moments");
            }
            output.extend(row.iter().map(|value| (value - mean) * reciprocal_std));
        }
        Ok((CpuStorage::F32(output), layout.shape().clone()))
    }
}

#[derive(Clone, Copy, Default)]
struct Moments {
    count: usize,
    mean: f32,
    squared_deviation: f32,
}

impl Moments {
    fn merge(&mut self, other: Self) {
        let count = self.count + other.count;
        let weight = if count == 0 {
            0.0
        } else {
            other.count as f32 / count as f32
        };
        let delta = other.mean - self.mean;
        self.mean = weight.mul_add(delta, self.mean);
        self.squared_deviation +=
            other.squared_deviation + delta * delta * weight * self.count as f32;
        self.count = count;
    }
}

/// Eight scalar lanes specify a stable reduction order, not an ISA dependency.
/// Bounded Welford chunks and a cascade avoid a long, error-prone naive sum.
fn row_moments(row: &[f32]) -> (f32, f32) {
    let vectors = row.len() / LANES;
    let chunks = vectors.div_ceil(CHUNK_VECTORS);
    let depth = (usize::BITS - chunks.saturating_sub(1).leading_zeros()).max(1) as usize;
    let mut stack = vec![[Moments::default(); LANES]; depth];

    for (index, chunk) in row[..vectors * LANES]
        .chunks(LANES * CHUNK_VECTORS)
        .enumerate()
    {
        let mut moments = [Moments::default(); LANES];
        for (step, vector) in chunk.as_chunks::<LANES>().0.iter().enumerate() {
            let count = step + 1;
            let reciprocal_count = 1.0 / count as f32;
            for (moment, &value) in moments.iter_mut().zip(vector) {
                let delta = value - moment.mean;
                moment.mean = delta.mul_add(reciprocal_count, moment.mean);
                moment.squared_deviation =
                    delta.mul_add(value - moment.mean, moment.squared_deviation);
                moment.count = count;
            }
        }
        for (destination, source) in stack[0].iter_mut().zip(moments) {
            destination.merge(source);
        }

        let mut mask = index + 1;
        for level in 1..depth {
            if mask & 1 != 0 {
                break;
            }
            let previous = std::mem::take(&mut stack[level - 1]);
            for (destination, source) in stack[level].iter_mut().zip(previous) {
                destination.merge(source);
            }
            mask >>= 1;
        }
    }

    for level in 1..depth {
        let previous = stack[level];
        for (destination, source) in stack[0].iter_mut().zip(previous) {
            destination.merge(source);
        }
    }

    let mut moments = Moments::default();
    for &value in &row[vectors * LANES..] {
        let delta = value - moments.mean;
        moments.count += 1;
        moments.mean += delta / moments.count as f32;
        moments.squared_deviation = delta.mul_add(value - moments.mean, moments.squared_deviation);
    }
    for lane in stack[0] {
        moments.merge(lane);
    }
    (moments.mean, moments.squared_deviation / row.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn matches_pinned_torch_moments_across_chunks_and_tails() -> Result<()> {
        // Independent PyTorch 2.3.1 CPU oracle: native_layer_norm(x, [n],
        // None, None, 1e-6). Integer/dyadic inputs avoid libm RNG differences.
        let cases = [
            (7, [0xbfc0_0000, 0xbf80_0000, 0x0000_0000, 0x3fc0_0000]),
            (9, [0xbfc6_4bf1, 0xbf94_b8f5, 0x0000_0000, 0x3fc6_4bf1]),
            (129, [0xbfd6_a499, 0xbfb6_c427, 0x3f0e_c5b1, 0xbf22_35e8]),
            (1031, [0xbfdd_0145, 0xbfbc_c27f, 0xbe5c_4507, 0x3fa5_f003]),
        ];
        for (width, expected) in cases {
            let values: Vec<f32> = (0..width)
                .map(|index| ((index * 37 % 509) as f32 - 254.0) * 0.125 + 10000.0)
                .collect();
            let x = Tensor::from_vec(values, (1, 1, width), &Device::Cpu)?;
            let output = layer_norm_no_affine(&x)?.flatten_all()?.to_vec1::<f32>()?;
            for (index, bits) in [0, 1, width / 2, width - 1].into_iter().zip(expected) {
                assert_eq!(
                    output[index].to_bits(),
                    bits,
                    "width={width}, index={index}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn preserves_small_variation_at_a_large_offset() -> Result<()> {
        let values: Vec<f32> = (0..1031)
            .map(|index| 10000.0 + index as f32 / 1024.0)
            .collect();
        let x = Tensor::from_vec(values, 1031, &Device::Cpu)?;
        let output = layer_norm_no_affine(&x)?.to_vec1::<f32>()?;
        for (index, bits) in
            [0, 1, 515, 1030]
                .into_iter()
                .zip([0xbfdd_7cef, 0xbfdd_0ed6, 0x0000_0000, 0x3fdd_7cef])
        {
            assert_eq!(output[index].to_bits(), bits);
        }
        Ok(())
    }

    #[test]
    fn honors_transposed_views_and_storage_offsets() -> Result<()> {
        let x = Tensor::arange(0f32, 60f32, &Device::Cpu)?
            .reshape((3, 4, 5))?
            .narrow(0, 1, 1)?
            .transpose(1, 2)?;
        let output = layer_norm_no_affine(&x)?.flatten_all()?.to_vec1::<f32>()?;
        // Each row is four equally spaced values, independent of its offset.
        let expected = [-1.341_640_8f32, -0.447_213_6, 0.447_213_6, 1.341_640_8];
        for row in output.as_chunks::<4>().0 {
            for (&value, &target) in row.iter().zip(&expected) {
                assert!((value - target).abs() < 1e-6);
            }
        }
        Ok(())
    }

    #[test]
    fn constant_rows_are_zero_and_invalid_input_is_rejected() -> Result<()> {
        for width in [1, 7, 128, 1031] {
            let x = Tensor::full(10000f32, (2, width), &Device::Cpu)?;
            let values = layer_norm_no_affine(&x)?.flatten_all()?.to_vec1::<f32>()?;
            assert!(values.iter().all(|value| *value == 0.0));
        }
        let empty = Tensor::zeros((1, 0), DType::F32, &Device::Cpu)?;
        assert!(layer_norm_no_affine(&empty).is_err());
        let integers = Tensor::new(&[1u32, 2], &Device::Cpu)?;
        assert!(layer_norm_no_affine(&integers).is_err());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let x = Tensor::new(&[value, 1.0], &Device::Cpu)?;
            assert!(layer_norm_no_affine(&x).is_err());
        }
        Ok(())
    }
}
