//! HiFT phase arithmetic for the pinned PyTorch CPU float32 path.
//!
//! True division and a float64 scan accumulator prevent small frequency errors
//! from accumulating into audible-rate phase drift. See `UPSTREAM.md` and
//! `LICENSE-PYTORCH` for the reviewed operations and attribution.

use candle::{CpuStorage, DType, Layout, Result, Shape, Tensor};

pub(super) fn divide_frequency(x: &Tensor, sample_rate: usize) -> Result<Tensor> {
    if sample_rate == 0 {
        candle::bail!("source sampling rate must be positive");
    }
    if x.device().is_cpu() && x.dtype() == DType::F32 {
        return x
            .contiguous()?
            .apply_op1_no_bwd(&FrequencyDivision(sample_rate as f32));
    }
    // Preserve the existing non-CPU/non-F32 path, with no device transfers or
    // unsupported claim of accelerator parity.
    x / sample_rate as f64
}

pub(super) fn accumulate_phase(x: &Tensor) -> Result<Tensor> {
    x.dims3()?;
    if x.device().is_cpu() && x.dtype() == DType::F32 {
        return x.contiguous()?.apply_op1_no_bwd(&PhaseScan);
    }
    x.cumsum(1)
}

fn contiguous_f32<'a>(storage: &'a CpuStorage, layout: &Layout) -> Result<&'a [f32]> {
    let CpuStorage::F32(values) = storage else {
        candle::bail!("source CPU phase operation requires float32 storage");
    };
    let Some((start, end)) = layout.contiguous_offsets() else {
        candle::bail!("source CPU phase operation requires contiguous input");
    };
    Ok(&values[start..end])
}

struct FrequencyDivision(f32);

impl candle::CustomOp1 for FrequencyDivision {
    fn name(&self) -> &'static str {
        "cosyvoice-source-frequency-division"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        // Tensor/scalar division in Candle 0.9.2 instead multiplies by the
        // rounded reciprocal. These are different float32 operations.
        let output = contiguous_f32(storage, layout)?
            .iter()
            .map(|value| value / self.0)
            .collect();
        Ok((CpuStorage::F32(output), layout.shape().clone()))
    }
}

struct PhaseScan;

impl candle::CustomOp1 for PhaseScan {
    fn name(&self) -> &'static str {
        "cosyvoice-source-phase-scan"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (batch, frames, harmonics) = layout.shape().dims3()?;
        let input = contiguous_f32(storage, layout)?;
        let mut output = vec![0.0; input.len()];
        for b in 0..batch {
            for h in 0..harmonics {
                let mut sum = 0.0_f64;
                for t in 0..frames {
                    let index = (b * frames + t) * harmonics + h;
                    sum += f64::from(input[index]);
                    output[index] = sum as f32;
                }
            }
        }
        Ok((CpuStorage::F32(output), layout.shape().clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Fixture {
        division: Division,
        scan: Scan,
    }

    #[derive(Deserialize)]
    struct Division {
        input: Vec<f32>,
        sample_rate: usize,
        expected_bits: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct Scan {
        input: Vec<f32>,
        shape: (usize, usize, usize),
        expected_bits: Vec<u32>,
    }

    fn bits(x: &Tensor) -> Result<Vec<u32>> {
        Ok(x.flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }

    #[test]
    fn matches_pinned_torch_cpu_bit_patterns_and_strided_views() -> Result<()> {
        let fixture: Fixture =
            serde_json::from_str(include_str!("../../tests/fixtures/source-phase.json"))
                .map_err(candle::Error::wrap)?;
        let Division {
            input,
            sample_rate,
            expected_bits,
        } = fixture.division;
        let x = Tensor::from_slice(&input, input.len(), &Device::Cpu)?;
        assert_eq!(bits(&divide_frequency(&x, sample_rate)?)?, expected_bits);
        // A contiguous slice with a nonzero storage offset must not read its prefix.
        let slice = x.narrow(0, 1, input.len() - 1)?;
        assert_eq!(
            bits(&divide_frequency(&slice, sample_rate)?)?,
            expected_bits[1..]
        );

        let Scan {
            input,
            shape,
            expected_bits,
        } = fixture.scan;
        let x = Tensor::from_slice(&input, shape, &Device::Cpu)?;
        assert_eq!(bits(&accumulate_phase(&x)?)?, expected_bits);
        let strided = x.transpose(1, 2)?.contiguous()?.transpose(1, 2)?;
        assert!(!strided.is_contiguous());
        assert_eq!(bits(&accumulate_phase(&strided)?)?, expected_bits);
        let slice = x.narrow(0, 1, shape.0 - 1)?;
        assert_eq!(
            bits(&accumulate_phase(&slice)?)?,
            expected_bits[shape.1 * shape.2..]
        );
        Ok(())
    }

    #[test]
    fn validates_rate_and_rank_and_preserves_other_dtypes() -> Result<()> {
        let x = Tensor::from_slice(&[1.0_f64, 2.0, 3.0], (1, 3, 1), &Device::Cpu)?;
        assert!(divide_frequency(&x, 0).is_err());
        assert!(accumulate_phase(&x.flatten_all()?).is_err());
        assert_eq!(divide_frequency(&x, 2)?.dtype(), DType::F64);
        assert_eq!(
            accumulate_phase(&x)?.flatten_all()?.to_vec1::<f64>()?,
            [1.0, 3.0, 6.0]
        );
        let empty = Tensor::zeros((1, 0, 2), DType::F32, &Device::Cpu)?;
        assert_eq!(accumulate_phase(&empty)?.dims(), [1, 0, 2]);
        Ok(())
    }
}
