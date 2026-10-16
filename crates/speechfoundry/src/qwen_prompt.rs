use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// Qwen conditioning is kept separate from the portable envelope.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    speaker_embedding: Vec<f32>,
    ref_codes: Option<Vec<Vec<u32>>>,
    ref_text_ids: Option<Vec<u32>>,
}

fn model_error(error: impl std::fmt::Display) -> Error {
    Error::Model(error.to_string())
}

pub(crate) fn encode(prompt: &speechfoundry_qwen::VoiceClonePrompt) -> Result<serde_json::Value> {
    let embedding = prompt
        .speaker_embedding
        .to_dtype(DType::F32)
        .and_then(|t| t.to_vec1())
        .map_err(model_error)?;
    let codes = prompt
        .ref_codes
        .as_ref()
        .map(|t| {
            t.to_dtype(DType::U32)
                .and_then(|t| t.to_vec2::<u32>())
                .map_err(model_error)
        })
        .transpose()?;
    Ok(serde_json::to_value(Payload {
        speaker_embedding: embedding,
        ref_codes: codes,
        ref_text_ids: prompt.ref_text_ids.clone(),
    })?)
}

pub(crate) fn decode(
    value: &serde_json::Value,
    device: &Device,
) -> Result<speechfoundry_qwen::VoiceClonePrompt> {
    let payload: Payload = serde_json::from_value(value.clone())?;
    if payload.speaker_embedding.len() != 1024
        || payload.speaker_embedding.iter().any(|v| !v.is_finite())
    {
        return Err(Error::InvalidInput(
            "Qwen prompt requires 1024 finite speaker values".into(),
        ));
    }
    if payload.ref_codes.is_some() != payload.ref_text_ids.is_some() {
        return Err(Error::InvalidInput(
            "Qwen ICL requires both codes and transcript tokens".into(),
        ));
    }
    if let Some(ids) = &payload.ref_text_ids
        && (ids.is_empty() || ids.len() > 8192 || ids.iter().any(|&v| v >= 151_936))
    {
        return Err(Error::InvalidInput("invalid Qwen transcript tokens".into()));
    }
    let codes = payload
        .ref_codes
        .map(|frames| {
            if frames.is_empty()
                || frames.len() > 376
                || frames
                    .iter()
                    .any(|frame| frame.len() != 16 || frame.iter().any(|&v| v >= 2048))
            {
                return Err(Error::InvalidInput(
                    "Qwen codes must have shape [1..376, 16] and values <2048".into(),
                ));
            }
            let len = frames.len();
            let values: Vec<u32> = frames.into_iter().flatten().collect();
            Tensor::from_vec(values, (len, 16), device).map_err(model_error)
        })
        .transpose()?;
    Ok(speechfoundry_qwen::VoiceClonePrompt {
        speaker_embedding: Tensor::from_vec(payload.speaker_embedding, 1024, device)
            .map_err(model_error)?,
        ref_codes: codes,
        ref_text_ids: payload.ref_text_ids,
    })
}
