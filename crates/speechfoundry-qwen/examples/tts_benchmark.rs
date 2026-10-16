//! Resident-model CPU benchmark with explicit extraction and synthesis stages.
use anyhow::{Result, ensure};
use candle_core::Device;
use serde::Serialize;
use serde_json::json;
use speechfoundry_qwen::{AudioBuffer, Language, Qwen3TTS, SynthesisOptions};
use std::{path::Path, time::Instant};

#[derive(Serialize)]
struct Sample {
    run: usize,
    extract_seconds: f64,
    prefill_seconds: f64,
    generate_seconds: f64,
    decode_seconds: f64,
    synthesis_seconds: f64,
    request_seconds: f64,
    audio_seconds: f64,
    audio_samples: usize,
    sample_rate: u32,
    codec_frames: usize,
    reached_token_budget: bool,
    synthesis_rtf: f64,
    request_rtf: f64,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 8,
        "usage: tts_benchmark MODEL_DIR REFERENCE_WAV TEXT OUTPUT_JSON MAX_NEW_TOKENS WARMUPS RUNS LANGUAGE"
    );
    let max_tokens: usize = args[4].parse()?;
    let warmups: usize = args[5].parse()?;
    let runs: usize = args[6].parse()?;
    ensure!((1..=100).contains(&runs), "runs must be in 1..=100");
    ensure!(warmups <= 20, "warmups must be at most 20");
    let language: Language = args[7].parse()?;
    let options = SynthesisOptions {
        max_length: max_tokens,
        temperature: 0.0,
        subtalker_temperature: 0.0,
        seed: Some(42),
        ..Default::default()
    };
    options.validate()?;
    let reference = AudioBuffer::load(&args[1])?;
    let started = Instant::now();
    let model = Qwen3TTS::from_pretrained(&args[0], Device::Cpu)?;
    let load_seconds = started.elapsed().as_secs_f64();
    let mut measured = Vec::with_capacity(runs);
    let mut first_audio: Option<Vec<f32>> = None;
    for index in 0..warmups + runs {
        let request_started = Instant::now();
        let prompt = model.create_voice_clone_prompt(&reference, None)?;
        let extract_seconds = request_started.elapsed().as_secs_f64();
        let synthesis_started = Instant::now();
        let (audio, timing) = model.synthesize_voice_clone_with_timing(
            &args[2],
            &prompt,
            language,
            Some(options.clone()),
        )?;
        let synthesis_seconds = synthesis_started.elapsed().as_secs_f64();
        let request_seconds = request_started.elapsed().as_secs_f64();
        ensure!(
            audio.sample_rate > 0
                && !audio.samples.is_empty()
                && audio.samples.iter().all(|sample| sample.is_finite()),
            "benchmark produced invalid or empty audio"
        );
        if let Some(first) = &first_audio {
            ensure!(first == &audio.samples, "greedy repeated outputs changed");
        } else {
            first_audio = Some(audio.samples.clone());
        }
        let audio_seconds = audio.samples.len() as f64 / f64::from(audio.sample_rate);
        eprintln!(
            "{} {}: {:.3}s request / {:.3}s audio; RTF {:.3}",
            if index < warmups { "warmup" } else { "sample" },
            index,
            request_seconds,
            audio_seconds,
            request_seconds / audio_seconds,
        );
        if index >= warmups {
            measured.push(Sample {
                run: index - warmups,
                extract_seconds,
                prefill_seconds: timing.prefill_ms / 1000.0,
                generate_seconds: timing.generation_ms / 1000.0,
                decode_seconds: timing.decode_ms / 1000.0,
                synthesis_seconds,
                request_seconds,
                audio_seconds,
                audio_samples: audio.samples.len(),
                sample_rate: audio.sample_rate,
                codec_frames: timing.generation_frames,
                reached_token_budget: timing.generation_frames == max_tokens - 1,
                synthesis_rtf: synthesis_seconds / audio_seconds,
                request_rtf: request_seconds / audio_seconds,
            });
        }
    }
    let document = json!({
        "schema_version": 1,
        "passed": true,
        "implementation": "speechfoundry-qwen",
        "crate_version": env!("CARGO_PKG_VERSION"),
        "device": "cpu",
        "dtype": "float32",
        "rayon_threads": rayon::current_num_threads(),
        "conditioning": "x_vector_only",
        "load_seconds": load_seconds,
        "warmups": warmups,
        "runs": runs,
        "samples": measured,
        "greedy_outputs_identical": true,
        "reference_audio_seconds": reference.samples.len() as f64 / f64::from(reference.sample_rate),
        "options": {"max_new_tokens": max_tokens, "temperature": 0, "subtalker_temperature": 0, "repetition_penalty": options.repetition_penalty, "min_new_tokens": options.min_new_tokens, "seed": options.seed},
        "timing": {
            "load": "one local checkpoint load; includes text tokenizer and all model components; filesystem caches are not flushed",
            "extract": "speaker feature extraction and prompt construction repeated per request; source WAV I/O excluded",
            "prefill": "text tokenization, cache allocation and prefill forward pass",
            "generate": "autoregressive semantic and acoustic codec prediction",
            "decode": "codec-to-PCM forward pass",
            "synthesis_rtf": "synthesis wall time / actual generated audio duration; excludes load and extraction",
            "request_rtf": "extraction plus synthesis wall time / actual generated audio duration; excludes load",
            "warmups": "complete extraction and synthesis runs on the resident model, excluded from measured samples",
            "scope": "bounded short-request CPU throughput probe; not time-to-first-audio, natural-EOS latency or perceptual quality"
        }
    });
    if let Some(parent) = Path::new(&args[3]).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args[3], serde_json::to_vec_pretty(&document)?)?;
    Ok(())
}
