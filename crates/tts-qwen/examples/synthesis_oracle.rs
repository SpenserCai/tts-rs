//! Deterministic full-checkpoint probe for numerical and end-to-end verification.
use anyhow::{Result, ensure};
use candle_core::{DType, Device};
use serde_json::json;
use std::{path::PathBuf, time::Instant};
use tts_qwen::{AudioBuffer, Language, Qwen3TTS, SynthesisOptions};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        (5..=7).contains(&args.len()),
        "usage: synthesis_oracle MODEL_DIR REFERENCE_WAV TEXT OUTPUT_JSON MAX_NEW_TOKENS [LANGUAGE] [REFERENCE_TEXT]"
    );
    let start = Instant::now();
    let model = Qwen3TTS::from_pretrained(&args[0], Device::Cpu)?;
    let load_seconds = start.elapsed().as_secs_f64();
    let reference = AudioBuffer::load(&args[1])?;
    let reference_text = args.get(6).map(String::as_str);
    let prompt = model.create_voice_clone_prompt(&reference, reference_text)?;
    let options = SynthesisOptions {
        max_length: args[4].parse()?,
        temperature: 0.0,
        subtalker_temperature: 0.0,
        seed: Some(42),
        ..Default::default()
    };
    let language = args
        .get(5)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(Language::Auto);
    let start = Instant::now();
    let (audio, codes) =
        model.synthesize_voice_clone_debug(&args[2], &prompt, language, Some(options))?;
    ensure!(
        !audio.samples.is_empty() && audio.samples.iter().all(|x| x.is_finite()),
        "synthesis returned empty or nonfinite PCM"
    );
    let inference_seconds = start.elapsed().as_secs_f64();
    let output = PathBuf::from(&args[3]);
    audio.save(output.with_extension("wav"))?;
    if let Some(reference_codes) = &prompt.ref_codes {
        let reference = json!({
            "schema_version": 1,
            "metadata": {"implementation": "tts-qwen", "device": "cpu"},
            "tensors": {
                "reference_audio_codes": {
                    "shape": reference_codes.dims(),
                    "dtype": "uint32",
                    "data": reference_codes.to_dtype(DType::U32)?.flatten_all()?.to_vec1::<u32>()?
                }
            }
        });
        std::fs::write(
            output.with_extension("reference-codes.json"),
            serde_json::to_vec_pretty(&reference)?,
        )?;
    }
    let frames = codes.len();
    let result = json!({
        "schema_version": 1,
        "metadata": {"implementation": "tts-qwen", "device": "cpu", "dtype": "float32", "load_seconds": load_seconds, "inference_seconds": inference_seconds, "sample_rate": audio.sample_rate},
        "tensors": {
            "speaker_embedding": {"shape": prompt.speaker_embedding.dims(), "dtype": "float32", "data": prompt.speaker_embedding.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?},
            "audio_codes": {"shape": [frames, 16], "dtype": "uint32", "data": codes.into_iter().flatten().collect::<Vec<_>>()},
            "waveform": {"shape": [audio.samples.len()], "dtype": "float32", "data": audio.samples}
        }
    });
    std::fs::write(output, serde_json::to_vec_pretty(&result)?)?;
    eprintln!("loaded in {load_seconds:.3}s; generated {frames} frames in {inference_seconds:.3}s");
    Ok(())
}
