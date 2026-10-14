//! loading operations.
use super::*;

impl Qwen3TTS {
    /// Load an official checkpoint from a local directory.
    ///
    /// Auto-detects the model variant (0.6B/1.7B, Base/CustomVoice/VoiceDesign)
    /// from `config.json` if present, falling back to weight inspection.
    ///
    /// Text tokenizer files come from the same directory. This method never
    /// downloads files. Use `tokenizer_id` to override the local tokenizer path.
    pub fn from_pretrained(model_id: &str, device: Device) -> Result<Self> {
        Self::from_pretrained_with_tokenizer(model_id, None, device)
    }

    /// Load a model with an explicit tokenizer source.
    ///
    /// `tokenizer_id` is a local directory or tokenizer JSON file. If omitted,
    /// tokenizer files must be present in the model directory.
    pub fn from_pretrained_with_tokenizer(
        model_id: &str,
        tokenizer_id: Option<&str>,
        device: Device,
    ) -> Result<Self> {
        tracing::info!("Loading Qwen3-TTS from: {}", model_id);
        tracing::info!("Compute dtype: {:?}", compute_dtype_for_device(&device));

        // Try to parse config.json for auto-detection
        let config_path = Path::new(model_id).join("config.json");
        let parsed_config = if config_path.exists() {
            Some(ParsedModelConfig::from_file(&config_path)?)
        } else {
            None
        };

        // Load text tokenizer
        let tok_source = tokenizer_id.unwrap_or(model_id);
        let text_tokenizer = tokenizer::TextTokenizer::from_pretrained(tok_source)?;

        // Load model weights
        let model_path = Path::new(model_id).join("model.safetensors");
        if !model_path.exists() {
            anyhow::bail!(
                "Model weights not found at {}. Please download the model first.",
                model_path.display()
            );
        }
        let weights = Self::load_weights(&model_path, &device)?;

        // Load speech tokenizer for decoder
        let st_path = Path::new(model_id).join("speech_tokenizer/model.safetensors");
        let st_weights = Self::load_weights(&st_path, &device)?;

        Self::build_from_components(
            &weights,
            &st_weights,
            text_tokenizer,
            parsed_config.as_ref(),
            &device,
        )
    }

    /// Load from pre-loaded weight tensors.
    ///
    /// Uses weight inspection for auto-detection. For config.json-based
    /// detection, use [`from_pretrained`](Self::from_pretrained) instead.
    pub fn from_weights(
        model_weights: &HashMap<String, Tensor>,
        decoder_weights: &HashMap<String, Tensor>,
        text_tokenizer: tokenizer::TextTokenizer,
        device: &Device,
    ) -> Result<Self> {
        Self::build_from_components(model_weights, decoder_weights, text_tokenizer, None, device)
    }

    /// Construct from in-memory weights with the checkpoint's parsed configuration.
    /// This entry point does not read files and is suitable for language bindings.
    pub fn from_weights_with_config(
        model_weights: &HashMap<String, Tensor>,
        decoder_weights: &HashMap<String, Tensor>,
        text_tokenizer: tokenizer::TextTokenizer,
        config: &ParsedModelConfig,
        device: &Device,
    ) -> Result<Self> {
        Self::build_from_components(
            model_weights,
            decoder_weights,
            text_tokenizer,
            Some(config),
            device,
        )
    }

    /// Shared builder: assembles all model components from pre-loaded weights.
    ///
    /// When `parsed_config` is `Some`, uses config.json dimensions and model type.
    /// When `None`, auto-detects the model variant from weight shapes.
    pub(super) fn build_from_components(
        model_weights: &HashMap<String, Tensor>,
        decoder_weights: &HashMap<String, Tensor>,
        text_tokenizer: tokenizer::TextTokenizer,
        parsed_config: Option<&ParsedModelConfig>,
        device: &Device,
    ) -> Result<Self> {
        let compute_dtype = compute_dtype_for_device(device);

        // Build TalkerModel
        let talker_config = if let Some(cfg) = parsed_config {
            TalkerConfig::from_parsed(cfg)
        } else {
            Self::detect_talker_config(model_weights)?
        };
        let talker = TalkerModel::from_weights_with_config_dtype(
            model_weights,
            talker_config,
            device,
            compute_dtype,
        )?;

        // Build CodePredictor
        let cp_config = if let Some(cfg) = parsed_config {
            CodePredictorConfig::from_parsed(cfg)
        } else {
            let talker_hidden = talker.config().hidden_size;
            if talker_hidden != 1024 {
                CodePredictorConfig {
                    codec_embed_dim: Some(talker_hidden),
                    ..CodePredictorConfig::default()
                }
            } else {
                CodePredictorConfig::default()
            }
        };
        let cp_weights = Self::filter_weights(model_weights, "talker.code_predictor.");
        let cp_vb = candle_nn::VarBuilder::from_tensors(cp_weights, compute_dtype, device);
        let code_predictor = CodePredictor::new(cp_config, cp_vb)?;

        // Decoder (always F32 — convolutional, no attention)
        let decoder = Decoder12Hz::from_weights(decoder_weights, Default::default())?;

        // Speaker encoder (always F32, only present in Base models)
        let se_config = parsed_config.and_then(|c| c.speaker_encoder_config.clone());
        let speaker_encoder =
            Self::try_load_speaker_encoder(model_weights, se_config.as_ref(), device)?;

        // Speech encoder for ICL voice cloning
        let speech_encoder = Self::try_load_speech_encoder(decoder_weights, device)?;

        let model_type = parsed_config.map(|c| c.model_type);

        Ok(Self {
            talker,
            code_predictor,
            decoder,
            text_tokenizer,
            speaker_encoder,
            speech_encoder,
            model_type,
            device: device.clone(),
            compute_dtype,
        })
    }

    /// Detect talker config from weight shapes (fallback when no config.json).
    pub(super) fn detect_talker_config(weights: &HashMap<String, Tensor>) -> Result<TalkerConfig> {
        let norm_weight = weights
            .get("talker.model.norm.weight")
            .ok_or_else(|| anyhow::anyhow!("Missing talker.model.norm.weight"))?;
        let hidden_size = norm_weight.dim(0)?;
        Ok(if hidden_size == 2048 {
            TalkerConfig::custom_voice()
        } else {
            TalkerConfig::default()
        })
    }

    // ── Private helpers ─────────────────────────────────────────────────

    /// Attempt to load the speaker encoder from model weights.
    ///
    /// Returns `Ok(Some(encoder))` if `speaker_encoder.*` keys are found,
    /// `Ok(None)` if they are absent. When `config` is provided, uses the
    /// parsed enc_dim; otherwise falls back to defaults (enc_dim=1024).
    pub(super) fn try_load_speaker_encoder(
        weights: &HashMap<String, Tensor>,
        config: Option<&SpeakerEncoderConfig>,
        device: &Device,
    ) -> Result<Option<SpeakerEncoder>> {
        let has_se_weights = weights.keys().any(|k| k.starts_with("speaker_encoder."));
        if !has_se_weights {
            return Ok(None);
        }

        let config = config.cloned().unwrap_or_default();
        tracing::info!(
            "Loading speaker encoder (ECAPA-TDNN, enc_dim={}) for voice cloning...",
            config.enc_dim
        );
        let se_weights = Self::filter_weights(weights, "speaker_encoder.");
        let se_vb = candle_nn::VarBuilder::from_tensors(se_weights, DType::F32, device);
        let encoder = SpeakerEncoder::new(config, se_vb)?;
        Ok(Some(encoder))
    }

    /// Attempt to load the speech encoder (Mimi) from speech tokenizer weights.
    ///
    /// The speech encoder encodes raw audio to 12Hz codec codes, needed for
    /// ICL voice cloning. Returns `Ok(None)` if encoder keys are absent or
    /// loading fails (non-fatal — ICL mode just won't be available).
    pub(super) fn try_load_speech_encoder(
        weights: &HashMap<String, Tensor>,
        device: &Device,
    ) -> Result<Option<Encoder12Hz>> {
        // Check for encoder-related keys (either HF or candle format)
        let has_encoder_keys = weights
            .keys()
            .any(|k| k.starts_with("encoder.") || k.starts_with("encoder_transformer."));
        if !has_encoder_keys {
            return Ok(None);
        }

        tracing::debug!("Attempting to load speech encoder (Mimi) for ICL voice cloning...");
        Ok(Some(Encoder12Hz::from_weights(weights, device)?))
    }

    /// Load weights from safetensors file.
    ///
    /// Tensors are loaded in their native dtype (typically BF16 for Qwen3-TTS).
    /// Each component's VarBuilder handles casting to its target dtype.
    pub(super) fn load_weights(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
        Ok(candle_core::safetensors::load(path, device)?)
    }

    /// Filter weights by prefix, removing the prefix from keys.
    pub(crate) fn filter_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> HashMap<String, Tensor> {
        weights
            .iter()
            .filter_map(|(k, v)| {
                k.strip_prefix(prefix)
                    .map(|stripped| (stripped.to_string(), v.clone()))
            })
            .collect()
    }
}
