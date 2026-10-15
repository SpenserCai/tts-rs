//! Isolate pretrained DiT operators using frozen official block inputs.

use candle::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use serde_json::json;
use std::{collections::HashMap, error::Error, path::Path};
use tts_cosyvoice::{
    config::FlowConfig,
    flow::dit::{Attention, DiTBlock, FeedForward},
    flow::embeddings::AdaLayerNormZero,
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
    data: Vec<f64>,
}

impl Fixture {
    fn tensor(&self, name: &str, device: &Device) -> Result<Tensor, Box<dyn Error>> {
        let values = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("missing {name}"))?;
        if !values.data.iter().all(|x| x.is_finite()) {
            return Err("nonfinite fixture".into());
        }
        Ok(match values.dtype.as_str() {
            "float32" => Tensor::from_vec(
                values.data.iter().map(|x| *x as f32).collect::<Vec<_>>(),
                values.shape.as_slice(),
                device,
            )?,
            _ => return Err("expected a float32 block fixture".into()),
        })
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 4 {
        return Err("usage: flow_block_oracle MODEL_DIR INPUT INDEX OUTPUT".into());
    }
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    if fixture.schema_version != 1 {
        return Err("expected fixture version 1".into());
    }
    let device = Device::Cpu;
    let config = FlowConfig::default().dit;
    let index: usize = args[2].to_str().ok_or("non-UTF8 block index")?.parse()?;
    if index >= config.depth {
        return Err("block index exceeds the model depth".into());
    }
    let weights = std::fs::read(Path::new(&args[0]).join("flow.safetensors"))?;
    let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)?
        .pp(format!("dit.transformer_blocks.{}", index));
    let get = |name: &str| fixture.tensor(name, &device);
    let mut tensors = std::collections::BTreeMap::new();
    tensors.insert(
        "attn_norm.linear".to_string(),
        candle_nn::linear(config.dim, config.dim * 6, vb.pp("attn_norm.linear"))?
            .forward(&get("attn_norm.linear.input")?)?,
    );
    tensors.insert("attn_norm.linear.input".to_string(), get("time")?.silu()?);
    tensors.insert(
        "attn.to_q.input".to_string(),
        AdaLayerNormZero::new(config.dim, vb.pp("attn_norm"))?
            .forward(&get("input")?, &get("time")?)?
            .0,
    );

    for name in ["attn.to_q", "attn.to_k", "attn.to_v"] {
        tensors.insert(
            name.to_string(),
            candle_nn::linear(config.dim, config.dim, vb.pp(name))?
                .forward(&get(&format!("{name}.input"))?)?,
        );
    }
    tensors.insert(
        "attn".to_string(),
        Attention::new(config.dim, config.heads, config.dim_head, vb.pp("attn"))?.forward(
            &get("attn.to_q.input")?,
            None,
            &get("rope")?,
        )?,
    );

    tensors.insert(
        "ff".to_string(),
        FeedForward::new(config.dim, config.ff_mult, vb.pp("ff"))?.forward(&get("ff.input")?)?,
    );
    tensors.insert(
        "output".to_string(),
        DiTBlock::new(&config, vb)?.forward(&get("input")?, &get("time")?, None, &get("rope")?)?,
    );
    let mut result = serde_json::Map::new();
    for (name, tensor) in tensors {
        result.insert(name,json!({"shape":tensor.dims(),"dtype":"float32","data":tensor.flatten_all()?.to_vec1::<f32>()?}));
    }
    std::fs::write(
        &args[3],
        serde_json::to_vec(&json!({"schema_version":1,"tensors":result}))?,
    )?;
    Ok(())
}
