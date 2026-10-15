//! Model-local categorical sampling; independent of the RNG and tensor backend.

use candle::Result;

pub(super) fn probability_mass(weights: &[f32]) -> Result<f64> {
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        candle::bail!("categorical weights must be finite and nonnegative");
    }
    let total: f64 = weights.iter().map(|&weight| f64::from(weight)).sum();
    if total <= 0.0 || !total.is_finite() {
        candle::bail!("categorical weights must have positive finite mass");
    }
    Ok(total)
}

/// Inverse CDF over the *actual* mass, not an assumed float32 sum of one.
/// Keep the caller's float32 uniform/RNG consumption; accumulate in float64 so
/// low-probability vocabulary tails are not lost to float32 CDF rounding.
pub(super) fn categorical(weights: &[f32], uniform: f32) -> Result<usize> {
    if !(0.0..1.0).contains(&uniform) {
        candle::bail!("categorical uniform must be finite and in [0, 1)");
    }
    let target = f64::from(uniform) * probability_mass(weights)?;
    let mut cumulative = 0.0_f64;
    let mut last_positive = None;
    for (index, &weight) in weights.iter().enumerate() {
        if weight > 0.0 {
            last_positive = Some(index);
            cumulative += f64::from(weight);
            if target < cumulative {
                return Ok(index);
            }
        }
    }
    // Rounding must never manufacture a masked/zero-probability stop token.
    last_positive.ok_or_else(|| candle::Error::Msg("categorical weights have no support".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_distributions_and_uniforms() {
        for weights in [
            &[][..],
            &[0.0, 0.0],
            &[-1.0, 2.0],
            &[f32::NAN],
            &[f32::INFINITY],
        ] {
            assert!(categorical(weights, 0.5).is_err());
        }
        for uniform in [-0.1, 1.0, f32::NAN, f32::INFINITY] {
            assert!(categorical(&[1.0], uniform).is_err());
        }
    }

    #[test]
    fn respects_support_and_half_open_boundaries() -> Result<()> {
        let weights = [0.0, 1.0, 0.0, 3.0, 0.0];
        assert_eq!(categorical(&weights, 0.0)?, 1);
        assert_eq!(categorical(&weights, 0.25)?, 3);
        assert_eq!(
            categorical(&weights, f32::from_bits(1.0f32.to_bits() - 1))?,
            3
        );
        Ok(())
    }

    #[test]
    fn mass_drift_cannot_select_a_masked_stop_token() -> Result<()> {
        // RAS can leave 100 equiprobable tokens and a masked vocabulary tail.
        // The old float32 CDF ended below the largest legal uniform and returned
        // the last vocabulary entry (6760), despite that entry having zero mass.
        let mut weights = vec![0.0; 6761];
        weights[..101].fill(0.01);
        weights[24] = 0.0;
        let uniform = f32::from_bits(1.0f32.to_bits() - 1);
        assert!(weights.iter().copied().sum::<f32>() < uniform);
        assert_eq!(categorical(&weights, uniform)?, 100);
        Ok(())
    }

    #[test]
    fn preserves_small_tail_weights() -> Result<()> {
        let mut weights = vec![0.5];
        weights.extend(std::iter::repeat_n(1.0 / 33_554_432.0, 4096));
        weights.push(0.0);
        let uniform = f32::from_bits(1.0f32.to_bits() - 1);
        assert_eq!(categorical(&weights, uniform)?, 4095);
        Ok(())
    }

    #[test]
    fn scale_invariant_without_float32_renormalization() -> Result<()> {
        for uniform in [0.0, 0.124, 0.125, 0.499, 0.5, 0.999] {
            assert_eq!(
                categorical(&[1.0, 3.0, 4.0], uniform)?,
                categorical(&[0.125, 0.375, 0.5], uniform)?
            );
        }
        Ok(())
    }
}
