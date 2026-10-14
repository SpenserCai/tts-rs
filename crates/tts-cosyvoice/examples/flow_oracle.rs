//! Compare the pretrained flow decoder using fixed tokens and shared noise.
use candle::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use serde_json::json;
use std::{collections::HashMap, error::Error, path::Path};
use tts_cosyvoice::{
    config::FlowConfig,
    flow::dit::DiTBlock,
    flow::embeddings::{AdaLayerNormZeroFinal, InputEmbedding, RotaryEmbedding, TimestepEmbedding},
    flow::{DiT, flow_matching::CausalMaskedDiffWithDiT, pre_lookahead::PreLookaheadLayer},
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
            "uint32"
                if values
                    .data
                    .iter()
                    .all(|x| *x >= 0.0 && *x < 6561.0 && x.fract() == 0.0) =>
            {
                Tensor::from_vec(
                    values.data.iter().map(|x| *x as u32).collect::<Vec<_>>(),
                    values.shape.as_slice(),
                    device,
                )?
            }
            _ => return Err("unexpected fixture dtype or token range".into()),
        })
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if !(4..=5).contains(&args.len()) || args.get(4).is_some_and(|flag| flag != "--layerwise") {
        return Err(
            "usage: flow_oracle MODEL_DIR INPUT_JSON FLOW_NOISE OUTPUT_JSON [--layerwise]".into(),
        );
    }
    let layerwise = args.len() == 5;
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    if fixture.schema_version != 1 {
        return Err("expected version 1 fixture".into());
    }
    let device = Device::Cpu;
    let config = FlowConfig::default();
    let weights = std::fs::read(Path::new(&args[0]).join("flow.safetensors"))?;
    let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)?;
    let dit = DiT::new(config.dit.clone(), vb.pp("dit"))?;
    let mut noise = candle::safetensors::load(&args[2], &device)?;
    let noise = noise.remove("noise").ok_or("missing noise")?;
    let prompt_tokens = fixture.tensor("prompt_tokens", &device)?;
    let target_tokens = fixture.tensor("tokens", &device)?;
    let prompt_mel = fixture.tensor("prompt_mel", &device)?;
    let speaker = fixture.tensor("speaker_embedding", &device)?;
    let all_tokens = Tensor::cat(&[&prompt_tokens, &target_tokens], 1)?;
    let embedded =
        candle_nn::embedding(6561, 80, vb.pp("input_embedding"))?.forward(&all_tokens)?;
    let pre = PreLookaheadLayer::new(80, 1024, 80, 3, 1, vb.pp("pre_lookahead_layer"))?
        .forward(&embedded, None)?;
    let (_, token_frames, channels) = pre.dims3()?;
    let mu = pre
        .unsqueeze(2)?
        .broadcast_as((1, token_frames, 2, channels))?
        .reshape((1, token_frames * 2, channels))?
        .transpose(1, 2)?;
    let norm = speaker
        .sqr()?
        .sum_keepdim(1)?
        .sqrt()?
        .clamp(1e-12, f64::INFINITY)?;
    let speaker = candle_nn::linear(192, 80, vb.pp("spk_embed_affine_layer"))?
        .forward(&speaker.broadcast_div(&norm)?)?;
    let frames = token_frames * 2;
    let zeros_mu = Tensor::zeros((1, 80, frames), DType::F32, &device)?;
    let condition =
        prompt_mel
            .transpose(1, 2)?
            .pad_with_zeros(2, 0, frames - prompt_mel.dim(1)?)?;
    let first = dit.forward(
        &noise.narrow(2, 0, frames)?.repeat((2, 1, 1))?,
        &Tensor::ones((2, 1, frames), DType::F32, &device)?,
        &Tensor::cat(&[&mu, &zeros_mu], 0)?,
        &Tensor::zeros(2, DType::F32, &device)?,
        &Tensor::cat(&[&speaker, &speaker.zeros_like()?], 0)?,
        &Tensor::cat(&[&condition, &zeros_mu], 0)?,
        false,
    )?;
    let mut intermediate = serde_json::Map::new();
    if layerwise {
        // Replay the first estimator step with the public component modules. This
        // isolates accumulated errors by layer without production tracing hooks.
        let dit_vb = vb.pp("dit");
        let time = TimestepEmbedding::new(config.dit.dim, dit_vb.pp("time_embed"))?
            .forward(&Tensor::zeros(2, DType::F32, &device)?)?;
        let mut hidden = InputEmbedding::new(80, 80, config.dit.dim, dit_vb.pp("input_embed"))?
            .forward(
                &noise
                    .narrow(2, 0, frames)?
                    .repeat((2, 1, 1))?
                    .transpose(1, 2)?,
                &Tensor::cat(&[&condition, &zeros_mu], 0)?.transpose(1, 2)?,
                &Tensor::cat(&[&mu, &zeros_mu], 0)?.transpose(1, 2)?,
                &Tensor::cat(&[&speaker, &speaker.zeros_like()?], 0)?,
            )?;
        let rope = RotaryEmbedding::new(config.dit.dim_head, 10000.0).forward(
            frames,
            &device,
            DType::F32,
        )?;
        let mut trace = vec![
            ("decoder.estimator.time_embed".to_string(), time.clone()),
            ("decoder.estimator.input_embed".to_string(), hidden.clone()),
        ];
        for index in 0..config.dit.depth {
            hidden = DiTBlock::new(
                &config.dit,
                dit_vb.pp(format!("transformer_blocks.{index}")),
            )?
            .forward(&hidden, &time, None, &rope)?;
            trace.push((
                format!("decoder.estimator.transformer_blocks.{index}"),
                hidden.clone(),
            ));
        }
        let normalized = AdaLayerNormZeroFinal::new(config.dit.dim, dit_vb.pp("norm_out"))?
            .forward(&hidden, &time)?;
        let replay = candle_nn::linear(config.dit.dim, 80, dit_vb.pp("proj_out"))?
            .forward(&normalized)?
            .transpose(1, 2)?;
        if replay.flatten_all()?.to_vec1::<f32>()? != first.flatten_all()?.to_vec1::<f32>()? {
            return Err("diagnostic replay differs from production estimator".into());
        }
        trace.push(("decoder.estimator.norm_out".to_string(), normalized));
        for (key, tensor) in trace {
            intermediate.insert(key, json!({"shape": tensor.dims(), "dtype": "float32", "data": tensor.flatten_all()?.to_vec1::<f32>()?}));
        }
    }
    for (key, tensor) in [
        ("pre_lookahead_layer", pre),
        ("spk_embed_affine_layer", speaker),
        ("decoder.estimator", first),
    ] {
        intermediate.insert(key.into(), json!({"shape": tensor.dims(), "dtype": "float32", "data": tensor.flatten_all()?.to_vec1::<f32>()?}));
    }
    std::fs::write(
        Path::new(&args[3]).with_extension("intermediates.json"),
        serde_json::to_vec(&json!({"schema_version": 1, "tensors": intermediate}))?,
    )?;
    let model = CausalMaskedDiffWithDiT::new_with_rand_noise(
        config.vocab_size,
        config.input_size,
        config.output_size,
        192,
        config.token_mel_ratio,
        config.pre_lookahead_len,
        dit,
        config.cfm,
        vb,
        Some(noise),
    )?;
    let mel = model.inference(
        &fixture.tensor("tokens", &device)?,
        &fixture.tensor("prompt_tokens", &device)?,
        &fixture.tensor("prompt_mel", &device)?,
        &fixture.tensor("speaker_embedding", &device)?,
        10,
        false,
    )?;
    let values = mel.flatten_all()?.to_vec1::<f32>()?;
    if !values.iter().all(|x| x.is_finite()) {
        return Err("nonfinite flow output".into());
    }
    std::fs::write(
        &args[3],
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "metadata": {"implementation": "tts-cosyvoice", "device": "cpu"},
            "tensors": {"mel": {"shape": mel.dims(), "dtype": "float32", "data": values}},
        }))?,
    )?;
    Ok(())
}
