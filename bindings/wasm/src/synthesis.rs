//! Thin explicit-byte adapters; all model math and prompt validation stay in Rust crates.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use speechfoundry::{CheckpointHasher, Engine, ModelKind};
use speechfoundry_qwen::{ParsedModelConfig, Qwen3TTS, tokenizer::TextTokenizer};
use wasm_bindgen::prelude::*;

use super::{MAX_OPTIONS_BYTES, MAX_PROMPT_BYTES, WasmAudio, check_json_size, js_error};

/// Streaming SHA-256 identity, identical to the native filesystem loader.
#[wasm_bindgen(js_name = CheckpointHasher)]
pub struct WasmCheckpointHasher {
    inner: CheckpointHasher,
}

#[wasm_bindgen(js_class = CheckpointHasher)]
impl WasmCheckpointHasher {
    /// Start a Qwen checkpoint digest.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: CheckpointHasher::new(ModelKind::Qwen3),
        }
    }

    /// Canonical ordered paths as JSON. Missing files must be recorded explicitly.
    #[wasm_bindgen(js_name = fileNames)]
    pub fn file_names(&self) -> String {
        // Serializing static string slices cannot fail.
        serde_json::json!(self.inner.file_names()).to_string()
    }

    /// Begin the next present file with its exact BigInt byte length.
    #[wasm_bindgen(js_name = beginFile)]
    pub fn begin_file(&mut self, name: &str, length: u64) -> Result<(), JsValue> {
        self.inner.begin_file(name, Some(length)).map_err(js_error)
    }

    /// Record that the next canonical file is absent.
    #[wasm_bindgen(js_name = skipFile)]
    pub fn skip_file(&mut self, name: &str) -> Result<(), JsValue> {
        self.inner.begin_file(name, None).map_err(js_error)
    }

    /// Hash the next bytes without retaining them. Limit each transfer to 16 MiB.
    pub fn update(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        if bytes.len() > 16 * 1024 * 1024 {
            return Err(js_error("digest chunks must not exceed 16 MiB"));
        }
        self.inner.update(bytes).map_err(js_error)
    }

    /// Consume the hasher and return the validated checkpoint digest.
    pub fn finish(self) -> Result<String, JsValue> {
        self.inner.finish().map_err(js_error)
    }
}

impl Default for WasmCheckpointHasher {
    fn default() -> Self {
        Self::new()
    }
}

/// Assemble one checkpoint with bounded, per-tensor transfers.
///
/// The JavaScript loader parses safetensors metadata and submits each tensor
/// separately. No complete multi-gigabyte file crosses a JavaScript typed-array
/// boundary. Use this class in a Worker and call `free` on failed loads.
#[wasm_bindgen(js_name = QwenModelBuilder)]
pub struct WasmQwenModelBuilder {
    config: ParsedModelConfig,
    tokenizer: TextTokenizer,
    model: HashMap<String, Tensor>,
    codec: HashMap<String, Tensor>,
}

#[wasm_bindgen(js_class = QwenModelBuilder)]
impl WasmQwenModelBuilder {
    /// Parse official configuration and a serialized Hugging Face tokenizer.
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str, tokenizer_json: &str) -> Result<Self, JsValue> {
        if usize::BITS < 64 {
            return Err(js_error(
                "full Qwen inference requires the memory64 package",
            ));
        }
        check_json_size(config_json, MAX_OPTIONS_BYTES)?;
        check_json_size(tokenizer_json, 32 * 1024 * 1024)?;
        let config = ParsedModelConfig::from_json_str(config_json).map_err(js_error)?;
        let tokenizer =
            tokenizers::Tokenizer::from_bytes(tokenizer_json.as_bytes()).map_err(js_error)?;
        Ok(Self {
            config,
            tokenizer: TextTokenizer::from_tokenizer(tokenizer).map_err(js_error)?,
            model: HashMap::new(),
            codec: HashMap::new(),
        })
    }

    /// Construct directly from an official vocab/merges checkpoint without preprocessing.
    #[wasm_bindgen(js_name = fromVocabAndMerges)]
    pub fn from_vocab_and_merges(
        config_json: &str,
        vocab_json: &str,
        merges: &str,
        tokenizer_config: Option<String>,
    ) -> Result<Self, JsValue> {
        if usize::BITS < 64 {
            return Err(js_error(
                "full Qwen inference requires the memory64 package",
            ));
        }
        check_json_size(config_json, MAX_OPTIONS_BYTES)?;
        check_json_size(vocab_json, 32 * 1024 * 1024)?;
        check_json_size(merges, 16 * 1024 * 1024)?;
        if let Some(config) = &tokenizer_config {
            check_json_size(config, 1024 * 1024)?;
        }
        Ok(Self {
            config: ParsedModelConfig::from_json_str(config_json).map_err(js_error)?,
            tokenizer: TextTokenizer::from_vocab_and_merges_data(
                vocab_json,
                merges,
                tokenizer_config.as_deref(),
            )
            .map_err(js_error)?,
            model: HashMap::new(),
            codec: HashMap::new(),
        })
    }

    /// Insert one safetensors tensor. `section` is `model` or `codec`.
    ///
    /// Float weights become CPU float32 immediately, so constructing the model
    /// reuses storage instead of retaining both complete BF16 and F32 checkpoints.
    #[wasm_bindgen(js_name = addTensor)]
    pub fn add_tensor(
        &mut self,
        section: &str,
        name: &str,
        dtype: &str,
        shape_json: &str,
        bytes: &[u8],
    ) -> Result<(), JsValue> {
        if name.is_empty() || name.len() > 512 {
            return Err(js_error("invalid tensor name"));
        }
        check_json_size(shape_json, 1024)?;
        let shape: Vec<usize> = serde_json::from_str(shape_json).map_err(js_error)?;
        if shape.is_empty() || shape.len() > 8 || shape.contains(&0) {
            return Err(js_error("invalid tensor shape"));
        }
        let dtype = match dtype {
            "BF16" => DType::BF16,
            "F16" => DType::F16,
            "F32" => DType::F32,
            _ => {
                return Err(js_error(
                    "Qwen checkpoint tensors must be BF16, F16, or F32",
                ));
            }
        };
        let expected = shape
            .iter()
            .try_fold(dtype.size_in_bytes(), |n, &dim| n.checked_mul(dim))
            .ok_or_else(|| js_error("tensor shape overflow"))?;
        if bytes.len() != expected || expected > 2 * 1024 * 1024 * 1024 {
            return Err(js_error(
                "tensor byte length mismatch or transfer exceeds 2 GiB",
            ));
        }
        let tensors = match section {
            "model" => &mut self.model,
            "codec" => &mut self.codec,
            _ => return Err(js_error("tensor section must be model or codec")),
        };
        if tensors.contains_key(name) {
            return Err(js_error("duplicate tensor name"));
        }
        let tensor = Tensor::from_raw_buffer(bytes, dtype, &shape, &Device::Cpu)
            .and_then(|tensor| tensor.to_dtype(DType::F32))
            .map_err(js_error)?;
        tensors.insert(name.to_owned(), tensor);
        Ok(())
    }

    /// Consume the builder and load the complete Qwen pipeline.
    ///
    /// Supply the digest produced from the same files by `CheckpointHasher`.
    pub fn finish(self, fingerprint: String) -> Result<WasmQwenModel, JsValue> {
        let model = Qwen3TTS::from_weights_with_config(
            &self.model,
            &self.codec,
            self.tokenizer,
            &self.config,
            &Device::Cpu,
        )
        .map_err(js_error)?;
        Ok(WasmQwenModel {
            inner: Engine::from_qwen(model, fingerprint).map_err(js_error)?,
        })
    }
}

/// Loaded Qwen model: voice conditioning, text-to-code generation and PCM decoding.
#[wasm_bindgen(js_name = QwenModel)]
pub struct WasmQwenModel {
    inner: Engine,
}

#[wasm_bindgen(js_class = QwenModel)]
impl WasmQwenModel {
    /// Extract a portable prompt. A matching transcript enables ICL conditioning.
    #[wasm_bindgen(js_name = extractVoice)]
    pub fn extract_voice(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        transcript: Option<String>,
    ) -> Result<String, JsValue> {
        self.inner
            .extract_voice(samples, sample_rate, transcript.as_deref())
            .and_then(|prompt| serde_json::to_string(&prompt).map_err(Into::into))
            .map_err(js_error)
    }

    /// Synthesize text with a compatible portable prompt and shared option JSON.
    pub fn synthesize(
        &mut self,
        text: &str,
        prompt_json: &str,
        options_json: &str,
    ) -> Result<WasmAudio, JsValue> {
        check_json_size(prompt_json, MAX_PROMPT_BYTES)?;
        check_json_size(options_json, MAX_OPTIONS_BYTES)?;
        let prompt = serde_json::from_str(prompt_json).map_err(js_error)?;
        let options = serde_json::from_str(options_json).map_err(js_error)?;
        Ok(WasmAudio {
            inner: self
                .inner
                .synthesize(text, &prompt, &options)
                .map_err(js_error)?,
        })
    }

    /// The checkpoint identity shared with native Rust, Python and Node bindings.
    #[wasm_bindgen(getter, js_name = modelFingerprint)]
    pub fn model_fingerprint(&self) -> String {
        self.inner.model_fingerprint().to_owned()
    }
}
