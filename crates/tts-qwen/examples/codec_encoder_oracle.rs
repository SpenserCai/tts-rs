//! Encoder-only checkpoint probe; no language-model weights are loaded.
use anyhow::{Result, ensure};
use candle_core::{DType, Device};
use serde_json::{Map, json};
use std::path::Path;
use tts_qwen::{AudioBuffer, models::codec::Encoder12Hz};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 3,
        "usage: codec_encoder_oracle SPEECH_TOKENIZER_WEIGHTS INPUT_24000_WAV OUTPUT_JSON"
    );
    let model = Encoder12Hz::from_safetensors(Path::new(&args[0]), &Device::Cpu)?;
    let audio = AudioBuffer::load(&args[1])?;
    let mut tensors = Map::new();
    for (name, tensor) in model.encode_with_intermediates(&audio)? {
        let data = if tensor.dtype() == DType::U32 {
            json!(tensor.flatten_all()?.to_vec1::<u32>()?)
        } else {
            json!(
                tensor
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
            )
        };
        let dtype = if tensor.dtype() == DType::U32 {
            "uint32"
        } else {
            "float32"
        };
        tensors.insert(
            name.to_string(),
            json!({"shape": tensor.dims(), "dtype": dtype, "data": data}),
        );
    }
    let document = json!({"schema_version": 1, "metadata": {"device": "cpu", "implementation": "tts-qwen"}, "tensors": tensors});
    std::fs::write(&args[2], serde_json::to_vec_pretty(&document)?)?;
    Ok(())
}
