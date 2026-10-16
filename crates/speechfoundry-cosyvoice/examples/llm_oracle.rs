//! Compare first logits and greedy speech tokens with the official language model.
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use serde_json::json;
use speechfoundry_cosyvoice::{
    config::{CosyVoice3LMConfig, SamplingConfig},
    llm::CosyVoice3LM,
};
use std::{collections::HashMap, error::Error, path::Path};

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    tensors: HashMap<String, Values>,
}

#[derive(Deserialize)]
struct Values {
    shape: Vec<usize>,
    dtype: String,
    data: Vec<u32>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: llm_oracle MODEL_DIR INPUT_JSON OUTPUT_JSON".into());
    }
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    if fixture.schema_version != 1 {
        return Err("expected version 1 fixture".into());
    }
    let device = Device::Cpu;
    let tensor = |key: &str| -> Result<Tensor, Box<dyn Error>> {
        let v = fixture
            .tensors
            .get(key)
            .ok_or_else(|| format!("missing {key}"))?;
        if v.dtype != "uint32" {
            return Err("expected uint32 token fixture".into());
        }
        Ok(Tensor::from_slice(&v.data, v.shape.as_slice(), &device)?)
    };
    let target = tensor("text_tokens")?;
    let prefix = tensor("prompt_text_tokens")?;
    let speech = tensor("prompt_tokens")?;
    let weights = std::fs::read(Path::new(&args[0]).join("llm.safetensors"))?;
    let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)?;
    let mut model = CosyVoice3LM::new(&CosyVoice3LMConfig::default(), vb)?;
    let logits = model.next_token_logits(&Tensor::cat(&[&prefix, &target], 1)?, &speech)?;
    let data = logits.flatten_all()?.to_vec1::<f32>()?;
    if !data.iter().all(|x| x.is_finite()) {
        return Err("nonfinite language model logits".into());
    }
    let options = SamplingConfig {
        max_tokens: 128,
        temperature: 0.,
        ..Default::default()
    };
    let generated = model.inference(&target, &prefix, &speech, &options)?;
    let continuation = if fixture.tensors.contains_key("continuation_tokens") {
        tensor("continuation_tokens")?
    } else {
        Tensor::from_slice(
            std::slice::from_ref(generated.first().ok_or("no generated speech token")?),
            (1, 1),
            &device,
        )?
    };
    let teacher_forced = model.next_token_logits(
        &Tensor::cat(&[&prefix, &target], 1)?,
        &Tensor::cat(&[&speech, &continuation], 1)?,
    )?;
    std::fs::write(
        Path::new(&args[2]).with_extension("diagnostics.json"),
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "metadata": {"scope": "next logits after a fixed speech continuation, recomputing full context"},
            "tensors": {
                "full_context_logits_0": {
                    "shape": teacher_forced.dims(), "dtype": "float32",
                    "data": teacher_forced.flatten_all()?.to_vec1::<f32>()?,
                },
                "continuation_tokens": {
                    "shape": continuation.dims(), "dtype": "uint32",
                    "data": continuation.flatten_all()?.to_vec1::<u32>()?,
                },
            },
        }))?,
    )?;
    std::fs::write(
        &args[2],
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "metadata": {"implementation": "speechfoundry-cosyvoice", "device": "cpu"},
            "tensors": {
                "first_logits": {"shape": logits.dims(), "dtype": "float32", "data": data},
                "generated_tokens": {"shape": [generated.len()], "dtype": "uint32", "data": generated},
            },
        }))?,
    )?;
    Ok(())
}
