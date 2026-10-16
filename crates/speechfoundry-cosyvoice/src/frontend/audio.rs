//! Audio Processing for CosyVoice3
//!
//! Implements mel spectrogram extraction and Kaldi-compatible Fbank features.

use candle::{DType, Device, Result, Tensor};
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;

fn validate_samples(samples: &[f32]) -> Result<()> {
    if samples.is_empty() || samples.iter().any(|value| !value.is_finite()) {
        candle::bail!("expected nonempty finite mono PCM");
    }
    Ok(())
}

fn validate_mel_dimensions(
    sample_rate: usize,
    n_fft: usize,
    hop: usize,
    n_mels: usize,
) -> Result<()> {
    if sample_rate == 0
        || n_fft < 2
        || hop == 0
        || hop > n_fft
        || n_mels == 0
        || n_mels.checked_mul(n_fft / 2 + 1).is_none()
        || n_mels.checked_add(2).is_none()
    {
        candle::bail!(
            "invalid mel dimensions: positive sample rate/bands and 0 < hop <= FFT required"
        );
    }
    Ok(())
}

/// Float32 operation order of torch.hann_window(periodic=True).
fn periodic_hann(size: usize) -> Vec<f32> {
    let step = (2.0 * PI / size as f64) as f32;
    (0..size)
        .map(|i| 0.5f32 - 0.5f32 * (i as f32 * step).cos())
        .collect()
}

/// PyTorch reflection excludes the boundary sample and requires pad < length.
fn reflect_pad(samples: &[f32], padding: usize) -> Result<Vec<f32>> {
    if samples.len() <= padding {
        candle::bail!("reflection padding requires more than {padding} samples");
    }
    let mut padded = Vec::with_capacity(samples.len() + 2 * padding);
    padded.extend(samples[1..=padding].iter().rev().copied());
    padded.extend_from_slice(samples);
    padded.extend(
        samples[samples.len() - padding - 1..samples.len() - 1]
            .iter()
            .rev()
            .copied(),
    );
    Ok(padded)
}

/// Mel Spectrogram Extractor
///
/// Fully CPU-based implementation, output is automatically transferred to target device.
#[derive(Debug, Clone)]
pub struct MelSpectrogram {
    #[allow(dead_code)]
    sample_rate: usize,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    mel_filters: Vec<f32>,
    hann_window: Vec<f32>,
    target_device: Device,
}

impl MelSpectrogram {
    /// Create CosyVoice Speech Feat configuration
    ///
    /// Uses librosa-compatible mel filters matching Matcha-TTS mel_spectrogram.
    /// Padding, window dtype and filter normalization follow the pinned Python implementation.
    pub fn new_cosyvoice_speech_feat(device: &Device) -> Result<Self> {
        // Matcha-TTS uses: librosa_mel_fn(sr=24000, n_fft=1920, n_mels=80, fmin=0, fmax=None)
        Self::new_with_librosa_filters(24000, 1920, 480, 80, device)
    }

    /// Create MelSpectrogram with librosa-compatible mel filters
    fn new_with_librosa_filters(
        sample_rate: usize,
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
        device: &Device,
    ) -> Result<Self> {
        validate_mel_dimensions(sample_rate, n_fft, hop_length, n_mels)?;
        let hann_window = periodic_hann(n_fft);

        // Use librosa-compatible mel filters (Slaney normalization)
        let mel_filters = Self::create_librosa_mel_filters(sample_rate, n_fft, n_mels);

        Ok(Self {
            sample_rate,
            n_fft,
            hop_length,
            n_mels,
            mel_filters,
            hann_window,
            target_device: device.clone(),
        })
    }

    /// Create Whisper format configuration (for speech tokenizer)
    ///
    /// Uses librosa-compatible mel filters matching OpenAI Whisper's mel_filters.npz.
    /// Feature extraction follows the pinned Python implementation.
    pub fn new_whisper_format(device: &Device) -> Result<Self> {
        Self::new_with_librosa_filters(16000, 400, 160, 128, device)
    }

    pub fn new(
        sample_rate: usize,
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
        device: &Device,
    ) -> Result<Self> {
        // Pre-compute Hann window
        validate_mel_dimensions(sample_rate, n_fft, hop_length, n_mels)?;
        let hann_window = periodic_hann(n_fft);

        // Pre-compute Mel filter bank
        let mel_filters = Self::create_mel_filters(sample_rate, n_fft, n_mels);

        Ok(Self {
            sample_rate,
            n_fft,
            hop_length,
            n_mels,
            mel_filters,
            hann_window,
            target_device: device.clone(),
        })
    }

    /// Create librosa-compatible mel filter bank with Slaney normalization
    ///
    /// This matches `librosa.filters.mel(sr, n_fft, n_mels)` which is used by OpenAI Whisper.
    /// Key differences from HTK-style:
    /// 1. Uses Slaney formula for mel scale (not HTK)
    /// 2. Applies area normalization (Slaney norm)
    fn create_librosa_mel_filters(sample_rate: usize, n_fft: usize, n_mels: usize) -> Vec<f32> {
        let n_freqs = n_fft / 2 + 1;
        let mut filters = vec![0.0f32; n_mels * n_freqs];

        // Slaney-style mel conversion (librosa default)
        // f_mel = 15 * log2(1 + f / 700) for f >= 1000
        // f_mel = 3 * f / 200 for f < 1000
        let f_min = 0.0f64;
        let f_max = sample_rate as f64 / 2.0;

        // Compute mel breakpoint
        let min_log_hz = 1000.0f64;
        let min_log_mel = 15.0f64; // 3 * 1000 / 200
        let logstep = 6.4f64.ln() / 27.0; // log(6.4) / 27 ≈ 0.0687

        let hz_to_mel_slaney = |hz: f64| -> f64 {
            if hz >= min_log_hz {
                min_log_mel + (hz / min_log_hz).ln() / logstep
            } else {
                3.0 * hz / 200.0
            }
        };

        let mel_to_hz_slaney = |mel: f64| -> f64 {
            if mel >= min_log_mel {
                min_log_hz * ((mel - min_log_mel) * logstep).exp()
            } else {
                200.0 * mel / 3.0
            }
        };

        let mel_min = hz_to_mel_slaney(f_min);
        let mel_max = hz_to_mel_slaney(f_max);

        // Create n_mels + 2 equally spaced points in mel space
        let mel_points: Vec<f64> = (0..=n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
            .collect();

        // Convert back to Hz
        let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz_slaney(m)).collect();

        // Convert to FFT bin indices (using floor like librosa)
        let fft_freqs: Vec<f64> = (0..n_freqs)
            .map(|i| i as f64 * sample_rate as f64 / n_fft as f64)
            .collect();

        // Build triangular filters
        for m in 0..n_mels {
            let left_hz = hz_points[m];
            let center_hz = hz_points[m + 1];
            let right_hz = hz_points[m + 2];

            // Compute filter weights for each frequency bin
            for k in 0..n_freqs {
                let freq = fft_freqs[k];

                let weight = if freq >= left_hz && freq < center_hz {
                    (freq - left_hz) / (center_hz - left_hz).max(1e-10)
                } else if freq >= center_hz && freq <= right_hz {
                    (right_hz - freq) / (right_hz - center_hz).max(1e-10)
                } else {
                    0.0
                };

                filters[m * n_freqs + k] = weight as f32;
            }

            // Apply Slaney normalization: normalize by bandwidth
            // enorm = 2.0 / (hz_points[m+2] - hz_points[m])
            let enorm = 2.0 / (right_hz - left_hz).max(1e-10);
            for k in 0..n_freqs {
                filters[m * n_freqs + k] = (filters[m * n_freqs + k] as f64 * enorm) as f32;
            }
        }

        filters
    }

    fn create_mel_filters(sample_rate: usize, n_fft: usize, n_mels: usize) -> Vec<f32> {
        let n_freqs = n_fft / 2 + 1;
        let mut filters = vec![0.0f32; n_mels * n_freqs];

        // Hz to Mel conversion (HTK style)
        let hz_to_mel = |hz: f64| -> f64 { 2595.0 * (1.0 + hz / 700.0).log10() };
        let mel_to_hz = |mel: f64| -> f64 { 700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0) };

        let mel_low = hz_to_mel(0.0);
        let mel_high = hz_to_mel(sample_rate as f64 / 2.0);

        // Evenly distributed points on Mel scale
        let mel_points: Vec<f64> = (0..=n_mels + 1)
            .map(|i| mel_low + (mel_high - mel_low) * i as f64 / (n_mels + 1) as f64)
            .collect();

        let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&hz| ((n_fft as f64 + 1.0) * hz / sample_rate as f64).floor() as usize)
            .collect();

        for m in 0..n_mels {
            let left = bin_points[m];
            let center = bin_points[m + 1];
            let right = bin_points[m + 2];

            for k in left..center {
                if k < n_freqs {
                    filters[m * n_freqs + k] = (k - left) as f32 / (center - left).max(1) as f32;
                }
            }
            for k in center..right {
                if k < n_freqs {
                    filters[m * n_freqs + k] = (right - k) as f32 / (right - center).max(1) as f32;
                }
            }
        }

        filters
    }

    /// Extract Mel Spectrogram
    ///
    /// # Arguments
    /// * `audio` - `[samples]` or [1, samples] audio waveform
    ///
    /// # Returns
    /// * `mel` - [1, n_mels, T] Mel spectrogram
    #[allow(clippy::needless_range_loop)]
    pub fn forward(&self, audio: &Tensor) -> Result<Tensor> {
        self.forward_internal(audio, false)
    }

    /// Extract Mel Spectrogram with CosyVoice/Matcha-TTS style padding
    ///
    /// This matches the Matcha-TTS mel_spectrogram function which adds
    /// reflect padding of (n_fft - hop_size) / 2 on each side before STFT.
    ///
    /// # Arguments
    /// * `audio` - `[samples]` or [1, samples] audio waveform
    ///
    /// # Returns
    /// * `mel` - [1, n_mels, T] Mel spectrogram
    #[allow(clippy::needless_range_loop)]
    pub fn forward_cosyvoice(&self, audio: &Tensor) -> Result<Tensor> {
        self.forward_internal(audio, true)
    }

    #[allow(clippy::needless_range_loop)]
    fn forward_internal(&self, audio: &Tensor, use_matcha_padding: bool) -> Result<Tensor> {
        // Ensure 1D input
        let audio = if audio.dims().len() == 2 {
            audio.squeeze(0)?
        } else {
            audio.clone()
        };

        // Convert to CPU f32 for processing
        let samples: Vec<f32> = audio
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec1()?;

        validate_samples(&samples)?;
        let padded_samples = if use_matcha_padding {
            reflect_pad(&samples, (self.n_fft - self.hop_length) / 2)?
        } else {
            samples
        };

        // Calculate number of frames
        let n_samples = padded_samples.len();
        let n_frames = if n_samples >= self.n_fft {
            (n_samples - self.n_fft) / self.hop_length + 1
        } else {
            0
        };

        if n_frames == 0 {
            return Tensor::zeros((1, self.n_mels, 0), DType::F32, &self.target_device);
        }

        let n_freqs = self.n_fft / 2 + 1;
        let mut mel_output = vec![0.0f32; self.n_mels * n_frames];

        let fft = FftPlanner::<f32>::new().plan_fft_forward(self.n_fft);
        let mut fft_input = vec![Complex::new(0.0, 0.0); self.n_fft];
        let mut fft_scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;

            // Apply window + FFT
            for (i, value) in fft_input.iter_mut().enumerate() {
                *value = Complex::new(padded_samples[start + i] * self.hann_window[i], 0.0);
            }
            fft.process_with_scratch(&mut fft_input, &mut fft_scratch);

            // Calculate magnitude spectrum (not power!)
            // Matcha-TTS uses: spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
            let mut magnitude_spectrum = vec![0.0f32; n_freqs];
            for k in 0..n_freqs {
                let re = fft_input[k].re;
                let im = fft_input[k].im;
                magnitude_spectrum[k] = (re * re + im * im + 1e-9).sqrt();
            }

            // Apply Mel filters
            for m in 0..self.n_mels {
                let mut sum = 0.0f32;
                for k in 0..n_freqs {
                    sum += magnitude_spectrum[k] * self.mel_filters[m * n_freqs + k];
                }
                // Log-mel (natural log, matching PyTorch's .log())
                // spectral_normalize_torch: torch.log(torch.clamp(x, min=1e-5))
                mel_output[m * n_frames + frame_idx] = (sum.max(1e-5)).ln();
            }
        }

        // Transfer to target device
        let mel = Tensor::from_vec(mel_output, (self.n_mels, n_frames), &Device::Cpu)?;
        mel.unsqueeze(0)?.to_device(&self.target_device)
    }

    /// Extract Mel Spectrogram matching OpenAI Whisper's implementation
    ///
    /// This method follows the STFT and mel stages of `whisper.log_mel_spectrogram()`:
    /// 1. Uses center=True padding (like torch.stft default)
    /// 2. Drops the last frame (magnitudes[..., :-1])
    /// 3. Uses librosa-compatible mel filters (Slaney normalization)
    ///
    /// # Arguments
    /// * `audio` - `[samples]` or [1, samples] audio waveform at 16kHz
    ///
    /// # Returns
    /// * `mel` - [1, n_mels, T] Mel spectrogram matching whisper output
    #[allow(clippy::needless_range_loop)]
    pub fn forward_whisper(&self, audio: &Tensor) -> Result<Tensor> {
        // Ensure 1D input
        let audio = if audio.dims().len() == 2 {
            audio.squeeze(0)?
        } else {
            audio.clone()
        };

        // Convert to CPU f32 for processing
        let samples: Vec<f32> = audio
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec1()?;
        validate_samples(&samples)?;
        let n_samples = samples.len();
        let padded_samples = reflect_pad(&samples, self.n_fft / 2)?;

        // Calculate number of frames (torch.stft formula with center=True)
        // n_frames = 1 + n_samples // hop_length
        let n_frames_stft = 1 + n_samples / self.hop_length;

        // Whisper drops the last frame: magnitudes[..., :-1]
        let n_frames = if n_frames_stft > 0 {
            n_frames_stft - 1
        } else {
            0
        };

        if n_frames == 0 {
            return Tensor::zeros((1, self.n_mels, 0), DType::F32, &self.target_device);
        }

        let n_freqs = self.n_fft / 2 + 1;
        let mut mel_output = vec![0.0f32; self.n_mels * n_frames];

        let fft = FftPlanner::<f32>::new().plan_fft_forward(self.n_fft);
        let mut fft_input = vec![Complex::new(0.0, 0.0); self.n_fft];
        let mut fft_scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;

            // Apply window + FFT
            for (i, value) in fft_input.iter_mut().enumerate() {
                *value = Complex::new(padded_samples[start + i] * self.hann_window[i], 0.0);
            }
            fft.process_with_scratch(&mut fft_input, &mut fft_scratch);

            // Calculate power spectrum (|STFT|^2)
            let mut power_spectrum = vec![0.0f32; n_freqs];
            for k in 0..n_freqs {
                let re = fft_input[k].re;
                let im = fft_input[k].im;
                power_spectrum[k] = re * re + im * im;
            }

            // Apply Mel filters
            for m in 0..self.n_mels {
                let mut sum = 0.0f32;
                for k in 0..n_freqs {
                    sum += power_spectrum[k] * self.mel_filters[m * n_freqs + k];
                }
                // Log-mel (log10)
                mel_output[m * n_frames + frame_idx] = (sum.max(1e-10)).log10();
            }
        }

        // Transfer to target device
        let mel = Tensor::from_vec(mel_output, (self.n_mels, n_frames), &Device::Cpu)?;
        mel.unsqueeze(0)?.to_device(&self.target_device)
    }
}

/// Kaldi-compatible Fbank feature extractor
///
/// Equivalent to torchaudio.compliance.kaldi.fbank.
/// Used for CAMPPlus speaker embedding extraction.
#[derive(Debug, Clone)]
pub struct KaldiFbank {
    #[allow(dead_code)]
    sample_rate: usize,
    n_mels: usize,
    preemphasis: f64,
    window_size: usize,
    hop_length: usize,
    padded_window_size: usize,
    povey_window: Vec<f32>,
    mel_filters: Vec<f32>,
}

impl KaldiFbank {
    /// Create CosyVoice CAMPPlus configuration
    pub fn new_for_campplus() -> Self {
        Self::new(16000, 80, 25.0, 10.0, 0.97, 20.0, 0.0)
    }

    pub fn new(
        sample_rate: usize,
        n_mels: usize,
        frame_length_ms: f64,
        frame_shift_ms: f64,
        preemphasis: f64,
        low_freq: f64,
        high_freq: f64,
    ) -> Self {
        let window_size = (sample_rate as f64 * frame_length_ms * 0.001) as usize;
        let hop_length = (sample_rate as f64 * frame_shift_ms * 0.001) as usize;
        let padded_window_size = window_size.next_power_of_two();

        let actual_high_freq = if high_freq <= 0.0 {
            sample_rate as f64 / 2.0 + high_freq
        } else {
            high_freq
        };

        let povey_window = Self::create_povey_window(window_size);
        let mel_filters = Self::create_mel_filters(
            sample_rate,
            padded_window_size,
            n_mels,
            low_freq,
            actual_high_freq,
        );

        Self {
            sample_rate,
            n_mels,
            preemphasis,
            window_size,
            hop_length,
            padded_window_size,
            povey_window,
            mel_filters,
        }
    }

    /// Create Povey window matching torchaudio.compliance.kaldi
    ///
    /// torchaudio uses: torch.hann_window(size, periodic=False).pow(0.85)
    /// The non-periodic (symmetric) hann window formula is:
    ///   w`[i]` = 0.5 * (1 - cos(2*pi*i / (N-1)))
    /// Note: denominator is (N-1), not N
    fn create_povey_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                // Non-periodic hann window: denominator is (size - 1)
                let angle = i as f32 * (2.0 * PI / (size - 1) as f64) as f32;
                let hann = 0.5f32 - 0.5f32 * angle.cos();
                hann.powf(0.85)
            })
            .collect()
    }

    fn create_mel_filters(
        sample_rate: usize,
        padded_window_size: usize,
        n_mels: usize,
        low_freq: f64,
        high_freq: f64,
    ) -> Vec<f32> {
        // Kaldi/torchaudio uses n_fft/2 bins (excluding Nyquist)
        // The Nyquist bin is handled separately by padding with 0
        let n_fft_bins = padded_window_size / 2;
        // We store n_fft_bins + 1 to include the Nyquist bin (which will be 0)
        let mut filters = vec![0.0f32; n_mels * (n_fft_bins + 1)];

        // Kaldi/HTK mel scale
        let hz_to_mel = |hz: f64| -> f64 { 1127.0 * (1.0 + hz / 700.0).ln() };

        let mel_low = hz_to_mel(low_freq);
        let mel_high = hz_to_mel(high_freq);

        // Mel frequency delta between filter centers
        let mel_freq_delta = (mel_high - mel_low) / (n_mels + 1) as f64;

        // FFT bin width in Hz
        let fft_bin_width = sample_rate as f64 / padded_window_size as f64;

        // Build triangular mel filters (matching torchaudio/Kaldi exactly)
        // Filter m has:
        //   left_mel = mel_low + m * mel_freq_delta
        //   center_mel = mel_low + (m + 1) * mel_freq_delta
        //   right_mel = mel_low + (m + 2) * mel_freq_delta
        for m in 0..n_mels {
            // Scalar endpoints are computed in Python float, then tensor arithmetic
            // uses float32 in torchaudio.compliance.kaldi.get_mel_banks.
            let left_mel = mel_low as f32 + m as f32 * mel_freq_delta as f32;
            let center_mel = mel_low as f32 + (m + 1) as f32 * mel_freq_delta as f32;
            let right_mel = mel_low as f32 + (m + 2) as f32 * mel_freq_delta as f32;

            for k in 0..n_fft_bins {
                // Convert bin to mel: mel(k * fft_bin_width)
                let hz = k as f32 * fft_bin_width as f32;
                let k_mel = 1127.0f32 * (1.0f32 + hz / 700.0f32).ln();

                // Compute triangular filter weight in mel domain
                // up_slope = (mel - left_mel) / (center_mel - left_mel)
                // down_slope = (right_mel - mel) / (right_mel - center_mel)
                // weight = max(0, min(up_slope, down_slope))
                let up_slope = (k_mel - left_mel) / (center_mel - left_mel);
                let down_slope = (right_mel - k_mel) / (right_mel - center_mel);
                let weight = up_slope.min(down_slope).max(0.0);

                filters[m * (n_fft_bins + 1) + k] = weight;
            }
            // Nyquist bin (k = n_fft_bins) is left as 0
        }

        filters
    }

    /// Extract Fbank features
    ///
    /// Follows the torchaudio.compliance.kaldi.fbank operation order:
    /// DC removal → pre-emphasis → window → FFT. CPU FFT and reduction kernels
    /// can differ numerically from PyTorch despite matching float32 staging.
    #[allow(clippy::needless_range_loop)]
    pub fn forward(&self, audio: &Tensor) -> Result<Tensor> {
        let samples: Vec<f32> = audio.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;

        // Calculate number of frames
        let n_frames = if samples.len() >= self.window_size {
            1 + (samples.len() - self.window_size) / self.hop_length
        } else {
            0
        };

        if n_frames == 0 {
            return Tensor::zeros((0, self.n_mels), DType::F32, audio.device());
        }

        // rfft output has n_fft/2 + 1 bins (DC to Nyquist)
        let n_fft_bins = self.padded_window_size / 2 + 1;
        let mut fbank = vec![0.0f32; n_frames * self.n_mels];
        let alpha = self.preemphasis as f32;
        let fft = FftPlanner::<f32>::new().plan_fft_forward(self.padded_window_size);
        let mut fft_input = vec![Complex::new(0.0, 0.0); self.padded_window_size];
        let mut fft_scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;

            // 1. Extract raw frame
            let mut frame: Vec<f32> = (0..self.window_size)
                .map(|i| samples.get(start + i).copied().unwrap_or(0.0))
                .collect();

            // 2. DC offset removal (per-frame, before pre-emphasis)
            let mean: f32 = frame.iter().sum::<f32>() / frame.len() as f32;
            for x in frame.iter_mut() {
                *x -= mean;
            }

            // 3. Pre-emphasis (per-frame, replicate padding style like torchaudio)
            // frame[i] = frame[i] - alpha * frame[i-1], with frame[-1] = frame[0] (replicate)
            if alpha != 0.0 {
                let mut prev = frame[0]; // replicate padding: frame[-1] = frame[0]
                for i in 0..self.window_size {
                    let current = frame[i];
                    frame[i] = current - alpha * prev;
                    prev = current;
                }
            }

            // 4. Apply Povey window
            for (i, x) in frame.iter_mut().enumerate() {
                *x *= self.povey_window[i];
            }

            // 5. Zero padding
            frame.resize(self.padded_window_size, 0.0);

            // 6. FFT (returns n_fft_bins complex values as interleaved [re, im, re, im, ...])
            for (value, &sample) in fft_input.iter_mut().zip(&frame) {
                *value = Complex::new(sample, 0.0);
            }
            fft.process_with_scratch(&mut fft_input, &mut fft_scratch);

            // 7. Power spectrum from rfft output (all n_fft_bins including Nyquist)
            let mut power = vec![0.0f32; n_fft_bins];
            for k in 0..n_fft_bins {
                let re = fft_input[k].re;
                let im = fft_input[k].im;
                // Kaldi applies complex abs, then squares the rounded magnitude.
                let magnitude = re.hypot(im);
                power[k] = magnitude * magnitude;
            }

            // 8. Mel filter + log
            for m in 0..self.n_mels {
                let mut sum = 0.0f32;
                for k in 0..n_fft_bins {
                    sum += power[k] * self.mel_filters[m * n_fft_bins + k];
                }
                fbank[frame_idx * self.n_mels + m] = sum.max(f32::EPSILON).ln();
            }
        }

        Tensor::from_vec(fbank, (n_frames, self.n_mels), audio.device())
    }
}

/// Convenience function: Extract Kaldi-style Fbank
pub fn kaldi_fbank(audio: &Tensor, n_mels: usize) -> Result<Tensor> {
    let extractor = KaldiFbank::new(16000, n_mels, 25.0, 10.0, 0.97, 20.0, 0.0);
    extractor.forward(audio)
}

/// High-quality audio resampling using windowed sinc interpolation
///
/// Follows torchaudio.transforms.Resample with the sinc_interp_hann method.
/// Kernels are designed in float64, cached as float32, and applied in float32.
/// The algorithm uses bandlimited interpolation with a Hann-windowed sinc function.
///
/// # Algorithm
/// The sinc interpolation formula reconstructs the continuous signal:
///   x(t) = sum_i x`[i]` * sinc(π * orig_freq * (i/orig_freq - t))
///
/// We then sample at the new rate:
///   y`[j]` = x(j / new_freq)
///
/// A Hann window is applied to limit the filter width and reduce ringing artifacts.
///
/// # Parameters
/// - `lowpass_filter_width`: Controls sharpness (default: 6, matching torchaudio)
/// - `rolloff`: Anti-aliasing rolloff frequency (default: 0.99)
#[allow(clippy::needless_range_loop)]
pub fn resample(audio: &Tensor, from_sr: usize, to_sr: usize) -> Result<Tensor> {
    resample_sinc(audio, from_sr, to_sr, 6, 0.99)
}

/// Sinc interpolation resampling with configurable parameters
///
/// # Arguments
/// * `audio` - Input audio tensor `[samples]`
/// * `from_sr` - Original sample rate
/// * `to_sr` - Target sample rate
/// * `lowpass_filter_width` - Filter width in zero crossings (higher = sharper but slower)
/// * `rolloff` - Anti-aliasing rolloff (0.0-1.0, lower = more anti-aliasing)
#[allow(clippy::needless_range_loop)]
pub fn resample_sinc(
    audio: &Tensor,
    from_sr: usize,
    to_sr: usize,
    lowpass_filter_width: usize,
    rolloff: f64,
) -> Result<Tensor> {
    if from_sr == 0
        || to_sr == 0
        || lowpass_filter_width == 0
        || !rolloff.is_finite()
        || !(0.0..=1.0).contains(&rolloff)
        || rolloff == 0.0
    {
        candle::bail!("resampling rates, filter width, and rolloff must be positive and finite");
    }
    if from_sr == to_sr {
        return Ok(audio.clone());
    }

    let samples: Vec<f32> = audio.to_dtype(DType::F32)?.to_vec1()?;
    let device = audio.device();

    // Compute GCD to reduce the resampling ratio
    let gcd = gcd(from_sr, to_sr);
    let orig_freq = from_sr / gcd;
    let new_freq = to_sr / gcd;

    // Base frequency for anti-aliasing
    let base_freq = (orig_freq.min(new_freq) as f64) * rolloff;

    // Filter width (number of input samples to consider on each side)
    let width = ((lowpass_filter_width as f64 * orig_freq as f64 / base_freq).ceil()) as usize;

    // Build resampling kernels for each phase
    // We need new_freq different kernels (one for each output phase)
    let kernel_size = 2 * width + orig_freq;
    // CosyVoice uses torchaudio.transforms.Resample: design in float64,
    // then cache the kernel and run convolution in float32.
    let mut kernels: Vec<Vec<f32>> = Vec::with_capacity(new_freq);

    for phase in 0..new_freq {
        let mut kernel = Vec::with_capacity(kernel_size);
        // The phase arange is integer in torchaudio and true division first
        // produces float32, before adding the float64 kernel positions.
        let phase_offset = (-(phase as f32) / new_freq as f32) as f64;

        for k in 0..kernel_size {
            let idx = (k as i64 - width as i64) as f64 / orig_freq as f64;
            let t = (phase_offset + idx) * base_freq;

            // Clamp t for numerical stability
            let t_clamped = t.clamp(-(lowpass_filter_width as f64), lowpass_filter_width as f64);

            // Hann window: cos²(π * t / (2 * width))
            let window = (t_clamped * PI / lowpass_filter_width as f64 / 2.0)
                .cos()
                .powi(2);

            // Sinc function: sin(π * t) / (π * t), with sinc(0) = 1
            let t_pi = t_clamped * PI;
            let sinc = if t_pi == 0.0 { 1.0 } else { t_pi.sin() / t_pi };

            // Combined kernel with scaling
            let scale = base_freq / orig_freq as f64;
            kernel.push((sinc * (window * scale)) as f32);
        }
        kernels.push(kernel);
    }

    // Pad input signal
    let padded_len = samples.len() + 2 * width + orig_freq;
    let mut padded = vec![0.0f32; padded_len];
    for (i, &s) in samples.iter().enumerate() {
        padded[width + i] = s;
    }

    // Calculate output length
    let output_len = (samples.len() * new_freq).div_ceil(orig_freq);

    // Apply resampling via strided convolution
    let mut resampled = Vec::with_capacity(output_len);

    for out_idx in 0..output_len {
        // Determine which kernel to use (phase)
        let phase = out_idx % new_freq;
        let kernel = &kernels[phase];

        // Determine input position (stride by orig_freq in the rational resampling sense)
        let in_base = (out_idx / new_freq) * orig_freq;

        // Convolve
        let mut sum = 0.0f32;
        for (k, &w) in kernel.iter().enumerate() {
            let in_idx = in_base + k;
            if in_idx < padded_len {
                sum += padded[in_idx] * w;
            }
        }
        resampled.push(sum);
    }

    let len = resampled.len();
    Tensor::from_vec(resampled, len, device)
}

/// Greatest Common Divisor using Euclidean algorithm
fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_frontend_features_include_both_boundaries_and_fractional_tail() -> Result<()> {
        // Generated by scripts/reference/export_cosyvoice_frontend.py from
        // public deterministic PCM, torch/torchaudio 2.3.1, float32 CPU.
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/frontend.json"))
                .expect("valid checked-in frontend fixture");
        let mut state = 0x1357_9bdfu32;
        let samples: Vec<f32> = (0..2401)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as i32 - 8_388_608) as f32 / 33_554_432.0
            })
            .collect();
        let pcm = Tensor::from_vec(samples, 2401, &Device::Cpu)?;
        let pcm_16k = resample(&pcm, 24_000, 16_000)?;
        let fbank = KaldiFbank::new_for_campplus().forward(&pcm_16k)?;
        let normalized = fbank.broadcast_sub(&fbank.mean(0)?)?;
        let mel =
            MelSpectrogram::new_cosyvoice_speech_feat(&Device::Cpu)?.forward_cosyvoice(&pcm)?;
        let whisper =
            MelSpectrogram::new_whisper_format(&Device::Cpu)?.forward_whisper(&pcm_16k)?;
        let floor = (whisper.flatten_all()?.max(0)? - 8.0)?;
        let whisper = ((whisper.broadcast_maximum(&floor)? + 4.0)? / 4.0)?;
        for (name, tensor) in [
            ("pcm_16k", pcm_16k),
            ("fbank", fbank),
            ("normalized", normalized),
            ("mel", mel),
            ("whisper", whisper),
        ] {
            let expected = &fixture["tensors"][name];
            let shape: Vec<usize> =
                serde_json::from_value(expected["shape"].clone()).expect("valid expected shape");
            assert_eq!(tensor.dims(), shape, "{name} shape");
            let values = tensor.flatten_all()?.to_vec1::<f32>()?;
            let reference: Vec<f32> =
                serde_json::from_value(expected["data"].clone()).expect("valid reference values");
            assert_eq!(values.len(), reference.len());
            for (index, (&actual, &expected)) in values.iter().zip(&reference).enumerate() {
                assert!(
                    (actual - expected).abs() <= 1e-5 + 1e-4 * expected.abs(),
                    "{name}[{index}]: actual={actual}, expected={expected}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn reflection_and_input_validation_match_stft_contract() -> Result<()> {
        assert_eq!(
            reflect_pad(&[1.0, 2.0, 3.0, 4.0], 2)?,
            vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
        );
        assert!(reflect_pad(&[1.0, 2.0], 2).is_err());
        assert_eq!(reflect_pad(&[1.0], 0)?, vec![1.0]);
        let whisper = MelSpectrogram::new_whisper_format(&Device::Cpu)?;
        for samples in [
            vec![],
            vec![0.0; 200],
            vec![f32::NAN; 401],
            vec![f32::INFINITY; 401],
        ] {
            let pcm = Tensor::from_vec(samples.clone(), samples.len(), &Device::Cpu)?;
            assert!(whisper.forward_whisper(&pcm).is_err());
        }
        let cosy = MelSpectrogram::new_cosyvoice_speech_feat(&Device::Cpu)?;
        assert!(
            cosy.forward_cosyvoice(&Tensor::zeros(720, DType::F32, &Device::Cpu)?)
                .is_err()
        );
        for (rate, fft, hop, bins) in [
            (0, 400, 160, 80),
            (16000, 0, 160, 80),
            (16000, 400, 0, 80),
            (16000, 400, 401, 80),
            (16000, 400, 160, 0),
        ] {
            assert!(MelSpectrogram::new(rate, fft, hop, bins, &Device::Cpu).is_err());
        }
        Ok(())
    }

    #[test]
    fn test_mel_spectrogram() -> Result<()> {
        let device = Device::Cpu;

        let mel_extractor = MelSpectrogram::new_cosyvoice_speech_feat(&device)?;

        // Create 1 second test audio (24kHz)
        let samples: Vec<f32> = (0..24000)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / 24000.0).sin() as f32)
            .collect();
        let audio = Tensor::from_vec(samples, 24000, &device)?;

        let mel = mel_extractor.forward(&audio)?;

        // Check shape: [1, 80, T]
        assert_eq!(mel.dim(0)?, 1);
        assert_eq!(mel.dim(1)?, 80);
        assert!(mel.dim(2)? > 0);

        Ok(())
    }

    #[test]
    fn resample_matches_pinned_torchaudio_and_keeps_fractional_tail() -> Result<()> {
        // torchaudio 2.3.1 transforms.Resample, float32 on CPU. Fractional
        // output lengths round up; the single-sample case must not become empty.
        let input = Tensor::from_slice(&[1.0f32, -0.5, 0.25, 0.125], 4, &Device::Cpu)?;
        for (from, to, expected) in [
            (
                24_000,
                16_000,
                vec![0.494_285_82, -0.128_965_17, 0.204_626_05],
            ),
            (
                16_000,
                24_000,
                vec![
                    0.984_078_05,
                    -0.036_942_02,
                    -0.502_387_46,
                    0.236_456_65,
                    0.337_781_5,
                    -0.013_394_259,
                ],
            ),
        ] {
            let actual = resample(&input, from, to)?.to_vec1::<f32>()?;
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 2e-7);
            }
        }
        let single = Tensor::from_slice(&[1.0f32], 1, &Device::Cpu)?;
        let actual = resample(&single, 48_000, 16_000)?.to_vec1::<f32>()?;
        assert_eq!(actual.len(), 1);
        assert!((actual[0] - 0.33).abs() < 2e-7);
        Ok(())
    }

    #[test]
    fn kaldi_povey_window_follows_float32_reference() {
        // torchaudio 2.3.1: non-periodic Hann.pow(0.85).
        // F64 evaluation followed by a final cast changes these constants.
        let window = KaldiFbank::create_povey_window(400);
        for (index, expected) in [
            (1, 0.000_265_134_56f32),
            (17, 0.032_588_01),
            (99, 0.549_210_4),
            (198, 0.999_881_45),
            (300, 0.549_210_55),
        ] {
            assert!((window[index] - expected).abs() < 2e-7);
        }
    }

    #[test]
    fn test_resample() -> Result<()> {
        let device = Device::Cpu;

        let audio = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], 4, &device)?;
        let resampled = resample(&audio, 2, 4)?;

        assert_eq!(resampled.dim(0)?, 8);
        Ok(())
    }
}
