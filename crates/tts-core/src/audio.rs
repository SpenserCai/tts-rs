use std::io::{Read, Seek, Write};

use audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{
    Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction, audioadapter::Adapter,
};
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// Mono, finite floating-point PCM. Samples are not silently normalized.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Audio {
    /// Waveform samples in time order.
    pub samples: Vec<f32>,
    /// Samples per second, in `1..=384000`.
    pub sample_rate: u32,
}

impl Audio {
    /// Construct validated mono PCM.
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Result<Self> {
        let audio = Self {
            samples,
            sample_rate,
        };
        audio.validate()?;
        Ok(audio)
    }

    /// Validate data received from a binding or deserializer.
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate == 0 || self.sample_rate > 384_000 {
            return Err(Error::InvalidInput(
                "sample_rate must be in 1..=384000".into(),
            ));
        }
        if self.samples.is_empty() || self.samples.iter().any(|s| !s.is_finite()) {
            return Err(Error::InvalidInput(
                "audio must contain finite, nonempty mono samples".into(),
            ));
        }
        Ok(())
    }

    /// Duration represented by the sample count and sample rate.
    pub fn duration_seconds(&self) -> f64 {
        self.samples.len() as f64 / f64::from(self.sample_rate)
    }

    /// Band-limited resampling with delay compensation and a complete tail.
    ///
    /// Output length is `round(input_len * target_rate / input_rate)`. This
    /// utility is not claimed bit-identical to librosa/soxr. For strict model
    /// parity, supply the same PCM at the model's native sample rate.
    pub fn resample(&self, target_rate: u32) -> Result<Self> {
        self.validate()?;
        if target_rate == 0 || target_rate > 384_000 {
            return Err(Error::InvalidInput(
                "target rate must be in 1..=384000".into(),
            ));
        }
        if self.sample_rate == target_rate {
            return Ok(self.clone());
        }
        let ratio = f64::from(target_rate) / f64::from(self.sample_rate);
        let wanted = (self.samples.len() as f64 * ratio).round() as usize;
        if wanted == 0 {
            return Err(Error::InvalidInput(
                "audio is too short for the target rate".into(),
            ));
        }
        let parameters = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler =
            Async::<f32>::new_sinc(ratio, 1.0, &parameters, 1024, 1, FixedAsync::Input)
                .map_err(|e| Error::InvalidInput(e.to_string()))?;
        let delay = resampler.output_delay();
        let mut output = Vec::with_capacity(wanted + delay + 1024);
        let mut position = 0;
        while output.len() < wanted + delay {
            let count = resampler.input_frames_next();
            let mut block = vec![0.0; count];
            let available = self.samples.len().saturating_sub(position).min(count);
            block[..available].copy_from_slice(&self.samples[position..position + available]);
            position += available;
            let channels = vec![block];
            let input = SequentialSliceOfVecs::new(&channels, 1, count)
                .map_err(|e| Error::InvalidInput(e.to_string()))?;
            let result = resampler
                .process(&input, 0, None)
                .map_err(|e| Error::InvalidInput(e.to_string()))?;
            for frame in 0..result.frames() {
                output.push(
                    result.read_sample(0, frame).ok_or_else(|| {
                        Error::Model("resampler returned an invalid frame".into())
                    })?,
                );
            }
        }
        Self::new(output[delay..delay + wanted].to_vec(), target_rate)
    }

    /// Read WAV PCM, averaging channels explicitly into mono.
    pub fn read_wav<R: Read>(reader: R) -> Result<Self> {
        let mut wav =
            hound::WavReader::new(reader).map_err(|e| Error::InvalidInput(e.to_string()))?;
        let spec = wav.spec();
        if spec.channels == 0 {
            return Err(Error::InvalidInput("WAV has no channels".into()));
        }
        let samples = match spec.sample_format {
            hound::SampleFormat::Float => wav
                .samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| Error::InvalidInput(e.to_string()))?,
            hound::SampleFormat::Int => {
                let scale = 2.0_f32.powi(i32::from(spec.bits_per_sample) - 1);
                wav.samples::<i32>()
                    .map(|s| s.map(|s| s as f32 / scale))
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| Error::InvalidInput(e.to_string()))?
            }
        };
        let channels = usize::from(spec.channels);
        if samples.len() % channels != 0 {
            return Err(Error::InvalidInput(
                "incomplete interleaved WAV frame".into(),
            ));
        }
        let mono = samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect();
        Self::new(mono, spec.sample_rate)
    }

    /// Write float32 WAV without clipping or hidden gain changes.
    pub fn write_wav<W: Write + Seek>(&self, writer: W) -> Result<()> {
        self.validate()?;
        let mut wav = hound::WavWriter::new(
            writer,
            hound::WavSpec {
                channels: 1,
                sample_rate: self.sample_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .map_err(|e| Error::InvalidInput(e.to_string()))?;
        for &sample in &self.samples {
            wav.write_sample(sample)
                .map_err(|e| Error::InvalidInput(e.to_string()))?;
        }
        wav.finalize()
            .map_err(|e| Error::InvalidInput(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn resampling_preserves_duration_and_impulse_time() {
        let mut samples = vec![0.0; 4_801];
        samples[2400] = 1.0;
        let result = Audio::new(samples, 48_000)
            .unwrap()
            .resample(24_000)
            .unwrap();
        assert_eq!(result.samples.len(), 2401);
        let peak = result
            .samples
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
            .unwrap()
            .0;
        assert!(peak.abs_diff(1200) <= 1);
    }

    #[test]
    fn wav_round_trip_preserves_float_pcm() {
        let source = Audio::new(vec![-0.5, 0.0, 0.8], 24_000).unwrap();
        let mut bytes = Cursor::new(Vec::new());
        source.write_wav(&mut bytes).unwrap();
        bytes.set_position(0);
        let decoded = Audio::read_wav(bytes).unwrap();
        assert_eq!(decoded.samples, source.samples);
        assert_eq!(decoded.sample_rate, source.sample_rate);
    }

    #[test]
    fn invalid_pcm_never_reaches_a_resampler() {
        assert!(Audio::new(vec![f32::NAN], 24_000).is_err());
        assert!(Audio::new(vec![0.0], 0).is_err());
        assert!(Audio::new(Vec::new(), 24_000).is_err());
    }
}
