//! Native ONNX Runtime adapter for the two auxiliary models released upstream.
//!
//! The `onnx` feature requires ONNX Runtime 1.24 or newer. Set `ORT_DYLIB_PATH`
//! to its shared library, or call [`CosyVoice3Frontend::load_with_runtime`]. The
//! language model, flow, and vocoder remain Candle implementations.

use std::{path::Path, sync::Mutex};

use candle::{DType, Device, Result, Tensor};
use ort::{environment::Environment, session::Session, value::Tensor as OrtTensor};

/// Audio conditioning frontend; sessions run on CPU as an independent component.
///
/// Session access is serialized because ONNX Runtime inference requires exclusive
/// session access. Build a separate frontend per worker for parallel requests.
pub struct CosyVoice3Frontend {
    speech_tokenizer: Mutex<Session>,
    campplus: Mutex<Session>,
    whisper_mel: super::audio::MelSpectrogram,
    kaldi_fbank: super::audio::KaldiFbank,
    speech_feat_mel: super::audio::MelSpectrogram,
    device: Device,
}

impl CosyVoice3Frontend {
    /// Load official graphs using the runtime path in `ORT_DYLIB_PATH`.
    pub fn load(model_dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let runtime = std::env::var_os("ORT_DYLIB_PATH").ok_or_else(|| {
            candle::Error::Msg("set ORT_DYLIB_PATH to the ONNX Runtime shared library (>= 1.24), or call load_with_runtime".into())
        })?;
        Self::load_with_runtime(model_dir, runtime, device)
    }

    /// Load an explicitly supplied ONNX Runtime shared library and official graphs.
    pub fn load_with_runtime(
        model_dir: impl AsRef<Path>,
        runtime_library: impl AsRef<Path>,
        device: &Device,
    ) -> Result<Self> {
        let environment = ort::init_from(runtime_library.as_ref())
            .map_err(ort_error)?
            .with_telemetry(false)
            .build()
            .map_err(ort_error)?;
        let model_dir = model_dir.as_ref();
        let load = |name: &str, preserve_pool_padding: bool| {
            load_session(&environment, &model_dir.join(name), preserve_pool_padding)
        };
        Ok(Self {
            speech_tokenizer: Mutex::new(load("speech_tokenizer_v3.onnx", false)?),
            campplus: Mutex::new(load("campplus.onnx", true)?),
            whisper_mel: super::audio::MelSpectrogram::new_whisper_format(&Device::Cpu)?,
            kaldi_fbank: super::audio::KaldiFbank::new_for_campplus(),
            speech_feat_mel: super::audio::MelSpectrogram::new_cosyvoice_speech_feat(&Device::Cpu)?,
            device: device.clone(),
        })
    }

    /// Extract portable conditioning without loading the synthesis weights.
    /// Reference clips require 0.1 through 30 seconds of finite mono PCM.
    pub fn extract_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        transcript: impl Into<String>,
    ) -> Result<crate::PromptFeatures> {
        if !(8000..=384000).contains(&sample_rate)
            || audio.len() < sample_rate as usize / 10
            || audio.len() > sample_rate as usize * 30
            || audio.iter().any(|x| !x.is_finite())
        {
            candle::bail!("reference requires 0.1..=30 seconds of finite mono PCM");
        }
        let transcript = transcript.into();
        if transcript.len() > 32_768 {
            candle::bail!("reference transcript exceeds 32768 bytes");
        }
        let audio = Tensor::from_slice(audio, audio.len(), &Device::Cpu)?;
        let (tokens, mel, embedding) =
            self.extract_prompt_features(&audio, sample_rate as usize)?;
        let result = crate::PromptFeatures {
            speech_tokens: tokens.flatten_all()?.to_dtype(DType::U32)?.to_vec1()?,
            mel: mel.squeeze(0)?.to_dtype(DType::F32)?.to_vec2()?,
            speaker_embedding: embedding.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?,
            transcript,
        };
        result.validate()?;
        Ok(result)
    }

    /// Run the released speech tokenizer on a mono waveform already sampled at 16 kHz.
    pub fn extract_speech_tokens(&self, audio_16k: &Tensor) -> Result<Tensor> {
        let mel = self.whisper_mel.forward_whisper(audio_16k)?;
        let floor = (mel.flatten_all()?.max(0)? - 8.0)?;
        let mel = ((mel.broadcast_maximum(&floor)? + 4.0)? / 4.0)?;
        let frames = mel.dim(2)?;
        let feat =
            OrtTensor::from_array((vec![1, 128, frames], mel.flatten_all()?.to_vec1::<f32>()?))
                .map_err(ort_error)?;
        let length = OrtTensor::from_array(([1], vec![frames as i32])).map_err(ort_error)?;
        let mut session = self
            .speech_tokenizer
            .lock()
            .map_err(|_| candle::Error::Msg("speech tokenizer session lock is poisoned".into()))?;
        let outputs = session.run(ort::inputs![feat, length]).map_err(ort_error)?;
        let tokens: Vec<u32> = if let Ok((_, values)) = outputs[0].try_extract_tensor::<i64>() {
            values
                .iter()
                .map(|&x| u32::try_from(x).map_err(ort_error))
                .collect::<Result<_>>()?
        } else {
            let (_, values) = outputs[0].try_extract_tensor::<i32>().map_err(ort_error)?;
            values
                .iter()
                .map(|&x| u32::try_from(x).map_err(ort_error))
                .collect::<Result<_>>()?
        };
        let length = tokens.len();
        Tensor::from_vec(tokens, (1, length), &self.device)
    }

    /// Run CAMPPlus on 80-bin, mean-normalized Kaldi filter-bank features.
    pub fn extract_speaker_embedding(&self, audio_16k: &Tensor) -> Result<Tensor> {
        let fbank = self.kaldi_fbank.forward(audio_16k)?;
        let fbank = fbank.broadcast_sub(&fbank.mean(0)?)?;
        let frames = fbank.dim(0)?;
        let feat =
            OrtTensor::from_array((vec![1, frames, 80], fbank.flatten_all()?.to_vec1::<f32>()?))
                .map_err(ort_error)?;
        let mut session = self
            .campplus
            .lock()
            .map_err(|_| candle::Error::Msg("speaker session lock is poisoned".into()))?;
        let outputs = session.run(ort::inputs![feat]).map_err(ort_error)?;
        let (_, values) = outputs[0].try_extract_tensor::<f32>().map_err(ort_error)?;
        Tensor::from_slice(values, (1, 192), &self.device)
    }

    /// Extract Matcha-style log mel at 24 kHz, returned as `[1, time, 80]`.
    pub fn extract_speech_feat(&self, audio_24k: &Tensor) -> Result<Tensor> {
        self.speech_feat_mel
            .forward_cosyvoice(audio_24k)?
            .transpose(1, 2)?
            .to_device(&self.device)
    }

    /// Return aligned semantic tokens, acoustic mel, and speaker embedding tensors.
    pub fn extract_prompt_features(
        &self,
        audio: &Tensor,
        sample_rate: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        if sample_rate == 0 {
            candle::bail!("sample rate must be positive");
        }
        let audio_16k = super::audio::resample(audio, sample_rate, 16_000)?;
        let audio_24k = super::audio::resample(audio, sample_rate, 24_000)?;
        let tokens = self.extract_speech_tokens(&audio_16k)?;
        let embedding = self.extract_speaker_embedding(&audio_16k)?;
        let mel = self.extract_speech_feat(&audio_24k)?;
        let length = tokens.dim(1)?.min(mel.dim(1)? / 2);
        Ok((
            tokens.narrow(1, 0, length)?,
            mel.narrow(1, 0, length * 2)?,
            embedding,
        ))
    }
}

fn load_session(
    environment: &Environment,
    path: &Path,
    preserve_pool_padding: bool,
) -> Result<Session> {
    let mut builder = Session::builder(environment)
        .map_err(ort_error)?
        .with_intra_threads(1)
        .map_err(ort_error)?;
    if preserve_pool_padding {
        // CAMPPlus has a Pad -> AveragePool(ceil_mode=1) boundary. Even
        // when the Pad widths are zero, ORT 1.24.4's Pad_Fusion changes
        // count_include_pad to 1 and divides the final partial window by the
        // full kernel length. Preserve this operator boundary; keep all
        // other graph optimizations enabled and the supplied graph unchanged.
        builder = builder
            .with_disabled_optimizers("Pad_Fusion")
            .map_err(ort_error)?;
    }
    builder.commit_from_file(path).map_err(ort_error)
}

fn ort_error(error: impl std::fmt::Display) -> candle::Error {
    candle::Error::Msg(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn campplus_preserves_explicit_padding_in_average_pool() -> Result<()> {
        // A reduced graph for CAMPPlus's Pad -> AveragePool boundary. Pad
        // widths are zero. Ceil pooling's partial [1,2,3] window averages to
        // 2.0; the faulty fusion divides by the four-wide kernel, yielding 1.5.
        // This exercises the production session configuration, not a copied
        // option in the test. ONNX-enabled tests require the documented runtime.
        let runtime = std::env::var_os("ORT_DYLIB_PATH")
            .ok_or_else(|| candle::Error::Msg("ONNX tests require ORT_DYLIB_PATH".into()))?;
        let environment = ort::init_from(runtime)
            .map_err(ort_error)?
            .with_telemetry(false)
            .build()
            .map_err(ort_error)?;
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/explicit_padding_average.onnx");
        let mut session = load_session(&environment, &path, true)?;
        let input =
            OrtTensor::from_array(([1, 1, 3], vec![1.0f32, 2.0, 3.0])).map_err(ort_error)?;
        let output = session.run(ort::inputs![input]).map_err(ort_error)?;
        let (_, values) = output[0].try_extract_tensor::<f32>().map_err(ort_error)?;
        assert_eq!(values, &[2.0]);
        Ok(())
    }
}
