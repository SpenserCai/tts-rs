//! Run an official-layout speaker checkpoint against a portable tensor fixture.
use anyhow::{Context, Result, ensure};
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use serde_json::json;
use speechfoundry_qwen::{SpeakerEncoderConfig, models::speaker::SpeakerEncoder};
use std::{collections::HashMap, path::PathBuf};

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    tensors: HashMap<String, Values>,
}

#[derive(Deserialize)]
struct Values {
    shape: Vec<usize>,
    dtype: String,
    data: Vec<f32>,
}

fn main() -> Result<()> {
    let paths: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    ensure!(
        paths.len() == 4 || paths.len() == 5,
        "usage: speaker_oracle CONFIG WEIGHTS INPUT_JSON OUTPUT_JSON [cpu|cuda|metal]"
    );
    let requested = paths
        .get(4)
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "cpu".to_owned());
    ensure!(
        requested == "cpu"
            || requested == "metal"
            || requested == "cuda"
            || requested.starts_with("cuda:"),
        "oracle requires an explicit cpu/cuda/metal backend"
    );
    let device = speechfoundry_qwen::parse_device(&requested)?;
    let config: SpeakerEncoderConfig = serde_json::from_slice(&std::fs::read(&paths[0])?)?;
    let input: Fixture = serde_json::from_slice(&std::fs::read(&paths[2])?)?;
    ensure!(input.schema_version == 1, "unsupported fixture version");
    let mel = input.tensors.get("mel").context("missing mel tensor")?;
    ensure!(mel.dtype == "float32", "mel dtype must be float32");
    ensure!(
        mel.shape.len() == 3 && mel.shape.iter().all(|&n| n > 0),
        "mel must have nonempty [batch, channels, time] dimensions"
    );
    ensure!(
        mel.shape[1] == config.mel_dim,
        "mel channel count differs from config"
    );
    ensure!(mel.data.iter().all(|x| x.is_finite()), "mel must be finite");
    let weights = candle_core::safetensors::load(&paths[1], &device)?;
    let encoder = SpeakerEncoder::new(
        config,
        VarBuilder::from_tensors(weights, DType::F32, &device),
    )?;
    let tensor = Tensor::from_vec(mel.data.clone(), mel.shape.as_slice(), &device)?;
    let embedding = encoder.forward(&tensor)?;
    let data: Vec<f32> = embedding.flatten_all()?.to_vec1()?;
    ensure!(
        data.iter().all(|x| x.is_finite()),
        "output contains nonfinite values"
    );
    let output = json!({
        "schema_version": 1,
        "metadata": {"implementation": "speechfoundry-qwen", "device": requested, "dtype": "float32"},
        "tensors": {"speaker_embedding": {"shape": embedding.dims(), "dtype": "float32", "data": data}}
    });
    std::fs::write(&paths[3], serde_json::to_vec_pretty(&output)?)?;
    Ok(())
}
