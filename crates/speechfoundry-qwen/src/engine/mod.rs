//! Model session orchestration.

mod loading;
mod options;
mod streaming;
mod synthesis;
mod voice_clone;
pub use options::SynthesisOptions;
pub use streaming::StreamingSession;

use crate::{audio, generation, tokenizer};
use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use crate::models::AnyKVCache;
use crate::models::codec::{Decoder12Hz, Encoder12Hz};
use crate::models::speaker::SpeakerEncoder;

// SynthesisTiming is defined above, already public.

/// Re-exports for convenience
pub use crate::audio::AudioBuffer;
pub use crate::models::config::Qwen3TTSConfig;
// StreamingSession is defined in this module, exported as top-level type
pub use crate::generation::SamplingContext;
pub use crate::models::talker::{Language, Speaker, codec_tokens, special_tokens, tts_tokens};
pub use crate::models::{
    CodePredictor, CodePredictorConfig, ModelType, ParsedModelConfig, SpeakerEncoderConfig,
    TalkerConfig, TalkerModel,
};

/// A sequence of codec frames, where each frame contains 16 codebook values
/// (1 semantic + 15 acoustic, formatted as `[semantic, acoustic_0..14]`).
pub type FrameCodes = Vec<Vec<u32>>;

/// Reference audio prompt for voice cloning.
///
/// Holds the speaker embedding and optional ICL (in-context learning) data.
/// Created via [`Qwen3TTS::create_voice_clone_prompt`].
pub struct VoiceClonePrompt {
    /// Speaker embedding from the ECAPA-TDNN encoder, shape `[enc_dim]` (typically 1024).
    pub speaker_embedding: Tensor,
    /// Reference audio codec codes for ICL mode, shape `[T, 16]`. `None` = x_vector_only mode.
    pub ref_codes: Option<Tensor>,
    /// Tokenized reference text for ICL mode.
    pub ref_text_ids: Option<Vec<u32>>,
}

/// Per-stage timing breakdown from a synthesis run.
#[derive(Debug, Clone, Serialize)]
pub struct SynthesisTiming {
    /// Time spent in the prefill phase (ms).
    pub prefill_ms: f64,
    /// Time spent in the autoregressive generation loop (ms).
    pub generation_ms: f64,
    /// Number of codec frames generated.
    pub generation_frames: usize,
    /// Time spent decoding codec frames to audio (ms).
    pub decode_ms: f64,
}

/// Main TTS interface using proper autoregressive pipeline.
///
/// Supports all 5 Qwen3-TTS model variants. Use [`model_type()`](Self::model_type)
/// to check which variant was loaded and [`supports_voice_cloning()`](Self::supports_voice_cloning)
/// / [`supports_preset_speakers()`](Self::supports_preset_speakers) to check capabilities.
pub struct Qwen3TTS {
    /// Talker model for semantic token generation
    talker: TalkerModel,
    /// Code predictor for acoustic token generation
    code_predictor: CodePredictor,
    /// 12Hz decoder for audio synthesis
    decoder: Decoder12Hz,
    /// Text tokenizer
    text_tokenizer: tokenizer::TextTokenizer,
    /// Speaker encoder for voice cloning (loaded when weights are present)
    speaker_encoder: Option<SpeakerEncoder>,
    /// Speech tokenizer encoder for ICL voice cloning (encodes reference audio → codes)
    speech_encoder: Option<Encoder12Hz>,
    /// Detected model variant (None if loaded without config.json)
    model_type: Option<ModelType>,
    /// Device to run inference on
    device: Device,
    /// Compute dtype for talker + code predictor (BF16 on CUDA, F32 otherwise)
    compute_dtype: DType,
}

impl Qwen3TTS {
    fn generation_caches(&self, steps: usize, prefix_lengths: &[usize]) -> Result<Vec<AnyKVCache>> {
        let capacity = prefix_lengths.iter().try_fold(steps, |total, &length| {
            total
                .checked_add(length)
                .ok_or_else(|| anyhow::anyhow!("generation context length overflow"))
        })?;
        self.talker.new_kv_caches(capacity)
    }
}

/// Convert a slice of codec frames into a tensor of shape `[1, 16, T]`.
///
/// Each frame must contain exactly 16 codebook values. The output layout is
/// `[q0_f0, q0_f1, ...], [q1_f0, q1_f1, ...]` matching the decoder's expectation.
pub fn codes_to_tensor(codes: &[Vec<u32>], device: &Device) -> Result<Tensor> {
    let num_frames = codes.len();
    if num_frames == 0 {
        return Ok(Tensor::zeros((1, 16, 0), DType::I64, device)?);
    }

    anyhow::ensure!(
        codes.iter().all(|frame| frame.len() == 16),
        "every codec frame must contain exactly 16 codebook values"
    );
    anyhow::ensure!(
        codes.iter().flatten().all(|&code| code < 2048),
        "codec values must be in 0..2048"
    );
    let mut data = vec![0i64; 16 * num_frames];
    for (frame, frame_codes) in codes.iter().enumerate() {
        for (q, &code) in frame_codes.iter().enumerate() {
            data[q * num_frames + frame] = code as i64;
        }
    }

    Ok(Tensor::from_vec(data, (1, 16, num_frames), device)?)
}

/// Return the recommended compute dtype for the given device.
///
/// Returns `BF16` for CUDA/Metal (lower memory, faster attention) and `F32` for CPU.
pub fn compute_dtype_for_device(device: &Device) -> DType {
    if device.is_cuda() || device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    }
}

/// Force the GPU to complete all pending work before returning.
///
/// On CUDA/Metal, GPU operations are asynchronous — `Instant::now()` would
/// measure submission time, not completion time. This helper forces a sync
/// by creating a tiny tensor and reading it back to the CPU.
///
/// On CPU this is a no-op.
pub fn sync_device(device: &Device) -> Result<()> {
    match device {
        Device::Cpu => Ok(()),
        _ => {
            // Force a GPU→CPU sync by reading a scalar back
            let _: Vec<f32> = Tensor::zeros(1, DType::F32, device)?.to_vec1()?;
            Ok(())
        }
    }
}

/// The codec end-of-sequence token ID (2150).
///
/// Generation stops when this token is sampled. This is in the codec vocabulary
/// `[0, 3072)`, not the text vocabulary.
pub const CODEC_EOS_TOKEN_ID: u32 = codec_tokens::CODEC_EOS;

/// Number of audio samples per codec frame at 24kHz (1920 = 80ms per frame at 12Hz).
pub const SAMPLES_PER_FRAME: usize = 1920;

/// Select the best available compute device for inference.
///
/// Checks for available hardware in order: CUDA → Metal → CPU.
/// Falls back to CPU if no GPU acceleration is available.
///
/// # Feature Flags
///
/// - `cuda`: Enables NVIDIA GPU support
/// - `metal`: Enables Apple Silicon GPU support
///
/// # Example
///
/// ```no_run
/// use speechfoundry_qwen::Qwen3TTS;
/// # fn main() -> anyhow::Result<()> {
/// let device = speechfoundry_qwen::auto_device()?;
/// let _model = Qwen3TTS::from_pretrained("path/to/model", device)?;
/// # Ok(())
/// # }
/// ```
pub fn auto_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::cuda_if_available(0)
            && device.is_cuda()
        {
            tracing::info!("Using CUDA device");
            return Ok(device);
        }
    }

    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            tracing::info!("Using Metal device");
            return Ok(device);
        }
    }

    tracing::info!("Using CPU device");
    Ok(Device::Cpu)
}

/// Parse a device string into a [`Device`].
///
/// Supported formats:
/// - `"auto"` — select best available via [`auto_device`]
/// - `"cpu"` — force CPU
/// - `"cuda"` or `"cuda:0"` — CUDA device 0
/// - `"cuda:N"` — CUDA device N
/// - `"metal"` — Apple Silicon GPU
///
/// # Errors
///
/// Returns an error if the device string is unrecognized, the requested
/// backend wasn't compiled in, or hardware initialization fails.
pub fn parse_device(device_str: &str) -> Result<Device> {
    match device_str.to_lowercase().as_str() {
        "auto" => auto_device(),
        "cpu" => Ok(Device::Cpu),
        s if s == "cuda" || s.starts_with("cuda:") => {
            #[cfg(feature = "cuda")]
            {
                let ordinal: usize = if s == "cuda" {
                    0
                } else if let Some(idx) = s.strip_prefix("cuda:") {
                    idx.parse()
                        .map_err(|e| anyhow::anyhow!("invalid CUDA device index: {e}"))?
                } else {
                    0
                };
                Device::new_cuda(ordinal)
                    .map_err(|e| anyhow::anyhow!("failed to init CUDA device {ordinal}: {e}"))
            }
            #[cfg(not(feature = "cuda"))]
            anyhow::bail!("CUDA support not compiled in. Rebuild with: cargo build --features cuda")
        }
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0)
                    .map_err(|e| anyhow::anyhow!("failed to init Metal device: {e}"))
            }
            #[cfg(not(feature = "metal"))]
            anyhow::bail!(
                "Metal support not compiled in. Rebuild with: cargo build --features metal"
            )
        }
        other => {
            anyhow::bail!("unknown device '{other}'. Supported: auto, cpu, cuda, cuda:N, metal")
        }
    }
}

/// Human-readable label for a [`Device`].
pub fn device_info(device: &Device) -> String {
    match device {
        Device::Cpu => "CPU".to_string(),
        Device::Cuda(_) => "CUDA".to_string(),
        Device::Metal(_) => "Metal".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesis_options_default() {
        let options = SynthesisOptions::default();
        assert_eq!(options.max_length, 2048);
        assert!((options.temperature - 0.9).abs() < 1e-6);
        assert_eq!(options.top_k, 50);
        assert!((options.top_p - 1.0).abs() < 1e-6);
        assert!((options.repetition_penalty - 1.05).abs() < 1e-6);
        assert_eq!(options.eos_token_id, Some(CODEC_EOS_TOKEN_ID));
        assert_eq!(options.chunk_frames, 10);
        assert_eq!(options.min_new_tokens, 2);
    }

    #[test]
    fn test_synthesis_options_custom() {
        let options = SynthesisOptions {
            max_length: 512,
            temperature: 0.5,
            top_k: 10,
            top_p: 0.8,
            repetition_penalty: 1.2,
            eos_token_id: Some(CODEC_EOS_TOKEN_ID),
            chunk_frames: 5,
            min_new_tokens: 0,
            seed: Some(42),
            ..Default::default()
        };
        assert_eq!(options.max_length, 512);
        assert!((options.temperature - 0.5).abs() < 1e-6);
        assert_eq!(options.eos_token_id, Some(CODEC_EOS_TOKEN_ID));
        assert_eq!(options.chunk_frames, 5);
    }

    #[test]
    fn test_synthesis_options_clone() {
        let options = SynthesisOptions::default();
        let cloned = options.clone();
        assert_eq!(cloned.max_length, options.max_length);
        assert_eq!(cloned.top_k, options.top_k);
    }

    #[test]
    fn test_synthesis_options_debug() {
        let options = SynthesisOptions::default();
        let debug_str = format!("{:?}", options);
        assert!(debug_str.contains("max_length"));
        assert!(debug_str.contains("2048"));
    }

    #[test]
    fn test_auto_device() {
        // Should always succeed on CPU
        let device = auto_device().unwrap();
        // Just verify it returns a valid device
        assert!(
            matches!(device, Device::Cpu)
                || matches!(device, Device::Cuda(_))
                || matches!(device, Device::Metal(_))
        );
    }

    #[test]
    fn test_audio_buffer_reexport() {
        // Verify re-exports work
        let buffer = AudioBuffer::new(vec![0.0f32; 100], 24000);
        assert_eq!(buffer.sample_rate, 24000);
    }

    #[test]
    fn test_config_reexport() {
        // Verify config re-export works
        let config = Qwen3TTSConfig::default();
        assert_eq!(config.model_type, "qwen3_tts");
    }

    #[test]
    fn test_codes_to_tensor_empty() {
        let device = Device::Cpu;
        let codes: Vec<Vec<u32>> = vec![];
        let tensor = codes_to_tensor(&codes, &device).unwrap();
        assert_eq!(tensor.dims(), &[1, 16, 0]);
    }

    #[test]
    fn test_codes_to_tensor_single_frame() {
        let device = Device::Cpu;
        let codes = vec![vec![0u32; 16]];
        let tensor = codes_to_tensor(&codes, &device).unwrap();
        assert_eq!(tensor.dims(), &[1, 16, 1]);
    }

    #[test]
    fn test_codes_to_tensor_layout() {
        let device = Device::Cpu;
        // 2 frames, each with 16 codebooks
        let codes = vec![
            (0..16).map(|i| i as u32).collect::<Vec<_>>(), // frame 0
            (100..116).map(|i| i as u32).collect::<Vec<_>>(), // frame 1
        ];
        let tensor = codes_to_tensor(&codes, &device).unwrap();
        assert_eq!(tensor.dims(), &[1, 16, 2]);

        // Verify layout: tensor[0, q, frame] = codes[frame][q]
        let vals: Vec<i64> = tensor.flatten_all().unwrap().to_vec1().unwrap();
        // q=0: [frame0_q0, frame1_q0] = [0, 100]
        assert_eq!(vals[0], 0);
        assert_eq!(vals[1], 100);
        // q=1: [frame0_q1, frame1_q1] = [1, 101]
        assert_eq!(vals[2], 1);
        assert_eq!(vals[3], 101);
    }

    #[test]
    fn test_parse_device_cpu() {
        let device = parse_device("cpu").unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_parse_device_auto() {
        let device = parse_device("auto").unwrap();
        // Should succeed regardless of hardware
        assert!(
            matches!(device, Device::Cpu)
                || matches!(device, Device::Cuda(_))
                || matches!(device, Device::Metal(_))
        );
    }

    #[test]
    fn test_parse_device_unknown() {
        let result = parse_device("tpu");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_device_case_insensitive() {
        let device = parse_device("CPU").unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_device_info() {
        assert_eq!(device_info(&Device::Cpu), "CPU");
    }

    #[test]
    fn test_compute_dtype_for_device() {
        let dtype = compute_dtype_for_device(&Device::Cpu);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_update_penalty_mask() {
        let device = Device::Cpu;
        let vocab_size = 3072;
        let mut mask = Tensor::zeros((1, vocab_size), DType::F32, &device).unwrap();

        Qwen3TTS::update_penalty_mask(&mut mask, 42, vocab_size).unwrap();

        let vals: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(vals[42], 1.0);
        // Neighboring positions should be untouched
        assert_eq!(vals[41], 0.0);
        assert_eq!(vals[43], 0.0);
    }

    #[test]
    fn test_update_penalty_mask_out_of_range() {
        let device = Device::Cpu;
        let vocab_size = 3072;
        let mut mask = Tensor::zeros((1, vocab_size), DType::F32, &device).unwrap();

        // Token beyond vocab_size should be a no-op (no panic)
        Qwen3TTS::update_penalty_mask(&mut mask, 9999, vocab_size).unwrap();

        let sum: f32 = mask.sum_all().unwrap().to_scalar().unwrap();
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn test_suppression_mask_deterministic() {
        let device = Device::Cpu;
        let vocab = codec_tokens::CODEC_VOCAB_SIZE;
        let mask1 = generation::build_suppression_mask(vocab, CODEC_EOS_TOKEN_ID, &device).unwrap();
        let mask2 = generation::build_suppression_mask(vocab, CODEC_EOS_TOKEN_ID, &device).unwrap();

        // Apply both masks to uniform logits and verify identical output
        let logits = Tensor::ones((1, vocab), DType::F32, &device).unwrap();
        let out1 = generation::apply_token_suppression_with_mask(&logits, &mask1).unwrap();
        let out2 = generation::apply_token_suppression_with_mask(&logits, &mask2).unwrap();
        let v1: Vec<f32> = out1.flatten_all().unwrap().to_vec1().unwrap();
        let v2: Vec<f32> = out2.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v1, v2);
    }
}
