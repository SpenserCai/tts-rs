//! Numerical oracle for official CosyVoice diffusion conditioning modules.
use candle::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use serde_json::json;
use speechfoundry_cosyvoice::flow::dit::FeedForward;
use speechfoundry_cosyvoice::flow::embeddings::{
    AdaLayerNormZero, AdaLayerNormZeroFinal, TimestepEmbedding,
};
use std::{collections::HashMap, error::Error, path::PathBuf};

#[derive(Deserialize)]
struct Config {
    dim: usize,
}

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

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(4..=5).contains(&args.len()) {
        return Err("usage: embeddings_oracle CONFIG WEIGHTS INPUT OUTPUT [cpu|cuda|metal]".into());
    }
    let device = match args.get(4).map_or("cpu", String::as_str) {
        "cpu" => Device::Cpu,
        "cuda" => Device::new_cuda(0)?,
        "metal" => Device::new_metal(0)?,
        _ => return Err("unknown device".into()),
    };
    let config: Config = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    if fixture.schema_version != 1 {
        return Err("unsupported fixture version".into());
    }
    let mut tensors = HashMap::new();
    for (name, values) in fixture.tensors {
        if values.dtype != "float32" || !values.data.iter().all(|value| value.is_finite()) {
            return Err("fixture tensors must be finite float32".into());
        }
        tensors.insert(
            name,
            Tensor::from_vec(values.data, values.shape.as_slice(), &device)?,
        );
    }
    let time = tensors.get("time").ok_or("missing time input")?;
    let x = tensors.get("x").ok_or("missing x input")?;
    let emb = tensors.get("emb").ok_or("missing emb input")?;
    let weights = candle::safetensors::load(PathBuf::from(&args[1]), &device)?;
    let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
    let time_result = TimestepEmbedding::new(config.dim, vb.pp("time"))?.forward(time)?;
    let ada = AdaLayerNormZero::new(config.dim, vb.pp("ada"))?.forward(x, emb)?;
    let final_result = AdaLayerNormZeroFinal::new(config.dim, vb.pp("final"))?.forward(x, emb)?;
    let feedforward = FeedForward::new(config.dim, 2, vb.pp("feedforward"))?.forward(x)?;
    let outputs = [
        ("time", time_result),
        ("ada.x", ada.0),
        ("ada.gate_msa", ada.1),
        ("ada.shift_mlp", ada.2),
        ("ada.scale_mlp", ada.3),
        ("ada.gate_mlp", ada.4),
        ("final", final_result),
        ("feedforward", feedforward),
    ];
    let mut results = serde_json::Map::new();
    for (name, tensor) in outputs {
        let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
        if !data.iter().all(|value| value.is_finite()) {
            return Err("nonfinite oracle output".into());
        }
        results.insert(
            name.into(),
            json!({"shape": tensor.dims(), "dtype": "float32", "data": data}),
        );
    }
    std::fs::write(
        &args[3],
        serde_json::to_vec_pretty(&json!({
            "schema_version": 1,
            "metadata": {"implementation": "speechfoundry-cosyvoice", "device": args.get(4).map_or("cpu", String::as_str)},
            "tensors": results,
        }))?,
    )?;
    Ok(())
}
