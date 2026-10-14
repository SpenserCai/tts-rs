//! Compare the pretrained vocoder independently of language-model sampling.
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use serde_json::json;
use std::{collections::HashMap, error::Error, path::Path};
use tts_cosyvoice::{
    config::HiFTConfig,
    vocoder::{CausalConvRNNF0Predictor, CausalHiFTGenerator, SourceModuleHnNSF},
};

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
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 4 {
        return Err("usage: hift_oracle MODEL_DIR INPUT_JSON SOURCE_NOISE OUTPUT_JSON".into());
    }
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let values = fixture.tensors.get("mel").ok_or("missing mel input")?;
    if fixture.schema_version != 1
        || values.dtype != "float32"
        || !values.data.iter().all(|x| x.is_finite())
    {
        return Err("expected version 1 finite float32 fixture".into());
    }
    let device = Device::Cpu;
    let mel = Tensor::from_slice(&values.data, values.shape.as_slice(), &device)?;
    let weights = std::fs::read(Path::new(&args[0]).join("hift.safetensors"))?;
    let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)?;
    let mut hift = CausalHiFTGenerator::new(HiFTConfig::default(), vb.clone())?;
    let mut noise = candle::safetensors::load(&args[2], &device)?;
    let mut take = |key: &str| noise.remove(key).ok_or_else(|| format!("missing {key}"));
    let (phase, sine, uv) = (take("phase")?, take("sine")?, take("uv")?);
    hift.set_reference_noise(phase.clone(), sine.clone(), uv.clone())?;
    let waveform = hift.inference(&mel, true)?;
    let samples = waveform.flatten_all()?.to_vec1::<f32>()?;
    if !samples.iter().all(|x| x.is_finite()) {
        return Err("nonfinite vocoder output".into());
    }
    std::fs::write(
        &args[3],
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "metadata": {"implementation": "tts-cosyvoice", "device": "cpu"},
            "tensors": {"waveform": {
                "shape": waveform.dims(), "dtype": "float32", "data": samples,
            }},
        }))?,
    )?;
    // Stage outputs localize regressions without changing the inference path.
    drop(hift);
    let predictor = CausalConvRNNF0Predictor::new(80, 512, true, vb.pp("f0_predictor"))?;
    let f0 = predictor.forward(&mel.to_dtype(DType::F64)?, true)?;
    let (batch, frames) = f0.dims2()?;
    let upsampled = f0
        .to_dtype(DType::F32)?
        .unsqueeze(2)?
        .unsqueeze(3)?
        .broadcast_as((batch, frames, 480, 1))?
        .reshape((batch, frames * 480, 1))?;
    let mut source = SourceModuleHnNSF::new(24000, 480, 8, 0.1, 0.003, vb.pp("m_source"))?;
    source.set_reference_noise(phase, sine, uv)?;
    let source = source.forward(&upsampled)?.0.transpose(1, 2)?;
    std::fs::write(
        Path::new(&args[3]).with_extension("intermediates.json"),
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "tensors": {
                "f0": {"shape": f0.dims(), "dtype": "float64", "data": f0.flatten_all()?.to_vec1::<f64>()?},
                "source": {"shape": source.dims(), "dtype": "float32", "data": source.flatten_all()?.to_vec1::<f32>()?},
            },
        }))?,
    )?;
    Ok(())
}
