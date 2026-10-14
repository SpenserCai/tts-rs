//! synthesis operations.
use super::*;

impl Qwen3TTS {
    /// Returns the detected model type, or `None` if loaded without config.json.
    pub fn model_type(&self) -> Option<&ModelType> {
        self.model_type.as_ref()
    }

    /// Whether this model supports voice cloning (Base models with speaker encoder).
    pub fn supports_voice_cloning(&self) -> bool {
        self.speaker_encoder.is_some()
    }

    /// Whether this model supports preset speaker selection (CustomVoice models).
    ///
    /// Returns `true` for CustomVoice, `false` for Base and VoiceDesign.
    /// When `model_type` is unknown (loaded without config.json), returns `true`
    /// as a permissive default.
    pub fn supports_preset_speakers(&self) -> bool {
        match &self.model_type {
            Some(ModelType::CustomVoice) => true,
            Some(ModelType::Base) | Some(ModelType::VoiceDesign) => false,
            None => true, // permissive when unknown
        }
    }

    /// Whether this model supports voice design (text-described voice conditioning).
    ///
    /// Returns `true` for VoiceDesign, `false` for all other variants.
    pub fn supports_voice_design(&self) -> bool {
        matches!(&self.model_type, Some(ModelType::VoiceDesign))
    }

    /// Synthesize speech from text with default voice (Ryan, English).
    ///
    /// Convenience wrapper around [`synthesize_with_voice`](Self::synthesize_with_voice).
    pub fn synthesize(&self, text: &str, options: Option<SynthesisOptions>) -> Result<AudioBuffer> {
        self.synthesize_with_voice(text, Speaker::Ryan, Language::English, options)
    }

    /// Synthesize speech with per-stage timing breakdown.
    ///
    /// Same as [`synthesize_with_voice`](Self::synthesize_with_voice) but also
    /// returns a [`SynthesisTiming`] with prefill, generation, and decode durations.
    /// Uses [`sync_device`] at timing boundaries for accurate GPU measurements.
    pub fn synthesize_with_timing(
        &self,
        text: &str,
        speaker: Speaker,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<(AudioBuffer, SynthesisTiming)> {
        anyhow::ensure!(
            self.supports_preset_speakers(),
            "preset voices require a CustomVoice checkpoint"
        );
        let options = options.unwrap_or_default();
        options.validate()?;
        let mut sampling_ctx = generation::SamplingContext::new(options.seed);
        anyhow::ensure!(!text.trim().is_empty(), "target text must not be empty");
        let input_ids = self.text_tokenizer.encode(text)?;
        let gen_config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            self.build_trailing_text(&input_ids)?;

        // -- Prefill --

        sync_device(&self.device)?;
        let t_prefill = Instant::now();

        let mut kv_caches = self
            .talker
            .new_kv_caches(gen_config.max_new_tokens + input_ids.len() + 16);
        let (hidden, logits) =
            self.talker
                .prefill_custom_voice(&input_ids, speaker, language, &mut kv_caches)?;
        let prefill_len = hidden.dim(1)?;
        let offset = prefill_len;
        let last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

        sync_device(&self.device)?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

        // -- Generation --
        let t_gen = Instant::now();

        let all_codes = self.generate_codes(
            &gen_config,
            &mut sampling_ctx,
            &mut kv_caches,
            offset,
            last_hidden,
            &logits,
            &trailing_text_hidden,
            trailing_text_len,
            &tts_pad_embed,
        )?;

        sync_device(&self.device)?;
        let generation_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
        let generation_frames = all_codes.len();

        // -- Decode --

        let t_decode = Instant::now();
        let audio = self.decode_codes(&all_codes)?;

        sync_device(&self.device)?;
        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;

        let timing = SynthesisTiming {
            prefill_ms,
            generation_ms,
            generation_frames,
            decode_ms,
        };

        Ok((audio, timing))
    }

    /// Build trailing text embeddings from input token IDs.
    ///
    /// Returns (trailing_text_hidden, trailing_text_len, tts_pad_embed).
    /// The trailing text is: remaining text tokens (all except first) projected + tts_eos.
    /// After trailing text is exhausted, tts_pad is used for each subsequent step.
    pub(super) fn build_trailing_text(&self, input_ids: &[u32]) -> Result<(Tensor, usize, Tensor)> {
        let trailing_text_hidden = if input_ids.len() > 1 {
            let remaining_proj = self.talker.get_projected_text_embeddings(&input_ids[1..])?;
            let tts_eos_embed = self.talker.get_tts_eos_embed()?;
            Tensor::cat(&[&remaining_proj, &tts_eos_embed], 1)?
        } else {
            self.talker.get_tts_eos_embed()?
        };
        let trailing_text_len = trailing_text_hidden.dim(1)?;
        let tts_pad_embed = self.talker.get_tts_pad_embed()?;
        Ok((trailing_text_hidden, trailing_text_len, tts_pad_embed))
    }

    /// Core generation loop shared by all synthesis methods.
    ///
    /// Runs the autoregressive generation loop: for each frame, check EOS,
    /// generate acoustic codes via CodePredictor, build the residual VQ sum
    /// with trailing text fusion, and sample the next semantic token.
    ///
    /// Callers handle prefill (which varies by model variant) and post-processing
    /// (decode, ICL ref_codes prepending, etc.).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate_codes(
        &self,
        gen_config: &generation::GenerationConfig,
        sampling_ctx: &mut generation::SamplingContext,
        kv_caches: &mut [AnyKVCache],
        mut offset: usize,
        mut last_hidden: Tensor,
        initial_logits: &Tensor,
        trailing_text_hidden: &Tensor,
        trailing_text_len: usize,
        tts_pad_embed: &Tensor,
    ) -> Result<FrameCodes> {
        // Pre-build the token suppression mask once (reused every frame)
        let suppression_mask = generation::build_suppression_mask(
            codec_tokens::CODEC_VOCAB_SIZE,
            CODEC_EOS_TOKEN_ID,
            &self.device,
        )?;

        // GPU-side repetition penalty mask: [1, vocab] — updated incrementally
        // instead of transferring all generated tokens to CPU each frame.
        let vocab_size = codec_tokens::CODEC_VOCAB_SIZE;
        let mut penalty_mask = Tensor::zeros((1, vocab_size), DType::F32, &self.device)?;

        // Pre-allocate code predictor KV caches (reused + reset each frame)
        let mut cp_kv_caches = self.code_predictor.new_kv_caches();

        // Sample first semantic token
        let logits_2d = initial_logits.squeeze(1)?;
        let logits_2d = self.apply_generation_penalties_gpu(
            &logits_2d,
            &penalty_mask,
            gen_config,
            0,
            Some(&suppression_mask),
        )?;
        let mut semantic_token_tensor = generation::sample(&logits_2d, gen_config, sampling_ctx)?;
        tracing::trace!(target: "gpu_sync", "to_vec1 in generate_codes first token");
        let mut semantic_token: u32 = semantic_token_tensor.flatten_all()?.to_vec1::<u32>()?[0];
        // Update penalty mask with this token (O(1) CPU work)
        Self::update_penalty_mask(&mut penalty_mask, semantic_token, vocab_size)?;
        let token_count: usize = 1;

        // Accumulate frames as GPU tensors: Vec of [16] U32 tensors
        // Deferred to_vec1 at the end eliminates per-frame acoustic code sync.
        let mut gpu_frames: Vec<Tensor> = Vec::new();

        for (token_count, frame_idx) in
            (token_count..).zip(0..gen_config.max_new_tokens.saturating_sub(1))
        {
            if let Some(eos_id) = gen_config.eos_token_id
                && semantic_token == eos_id
            {
                break;
            }

            // Embedding lookup using GPU-resident token tensor (no CPU→GPU roundtrip)
            let semantic_embed = self
                .talker
                .get_codec_embedding_from_tensor(&semantic_token_tensor)?;

            let acoustic_codes_tensor = self.code_predictor.generate_acoustic_codes(
                &last_hidden,
                &semantic_embed,
                &mut cp_kv_caches,
                &gen_config.acoustic(),
                sampling_ctx,
            )?;

            // Build [16] frame tensor on GPU: [semantic_token, acoustic_0..14]
            let frame_tensor = Tensor::cat(
                &[&semantic_token_tensor.reshape(1)?, &acoustic_codes_tensor],
                0,
            )?;
            gpu_frames.push(frame_tensor);

            // Use GPU tensor directly for embedding lookup (avoids 15 CPU→GPU transfers)
            let acoustic_embed_sum = self
                .code_predictor
                .get_acoustic_embeddings_sum_from_tensor(&acoustic_codes_tensor)?;
            let summed = semantic_embed.add(&acoustic_embed_sum)?;

            let text_addition = if frame_idx < trailing_text_len {
                trailing_text_hidden.i((.., frame_idx..frame_idx + 1, ..))?
            } else {
                tts_pad_embed.clone()
            };
            let step_input = summed.add(&text_addition)?;

            let (h, new_logits) =
                self.talker
                    .generate_step_with_embed(&step_input, kv_caches, offset)?;
            offset += 1;
            last_hidden = h;

            let logits_2d = new_logits.squeeze(1)?;
            let logits_2d = self.apply_generation_penalties_gpu(
                &logits_2d,
                &penalty_mask,
                gen_config,
                token_count,
                Some(&suppression_mask),
            )?;
            semantic_token_tensor = generation::sample(&logits_2d, gen_config, sampling_ctx)?;
            tracing::trace!(target: "gpu_sync", "to_vec1 in generate_codes sampling");
            semantic_token = semantic_token_tensor.flatten_all()?.to_vec1::<u32>()?[0];
            Self::update_penalty_mask(&mut penalty_mask, semantic_token, vocab_size)?;
        }

        // Single GPU→CPU transfer: convert all accumulated GPU frames to FrameCodes
        self.gpu_frames_to_frame_codes(&gpu_frames)
    }

    /// Update the GPU-side penalty mask for a single token ID.
    ///
    /// Sets `penalty_mask[0, token_id] = 1.0` using slice_assign with a
    /// pre-built scalar. This is O(1) CPU work (no GPU→CPU transfer).
    pub(super) fn update_penalty_mask(
        penalty_mask: &mut Tensor,
        token_id: u32,
        vocab_size: usize,
    ) -> Result<()> {
        let idx = token_id as usize;
        if idx < vocab_size {
            let one = Tensor::ones((1, 1), DType::F32, penalty_mask.device())?;
            *penalty_mask = penalty_mask.slice_assign(&[0..1, idx..idx + 1], &one)?;
        }
        Ok(())
    }

    /// Convert accumulated GPU frame tensors to FrameCodes via a single bulk transfer.
    pub(super) fn gpu_frames_to_frame_codes(&self, gpu_frames: &[Tensor]) -> Result<FrameCodes> {
        if gpu_frames.is_empty() {
            return Ok(Vec::new());
        }
        // Stack all frames into [n_frames, 16], then single to_vec1
        let stacked = Tensor::stack(gpu_frames, 0)?; // [n_frames, 16]
        let n_frames = stacked.dim(0)?;
        let flat: Vec<u32> = stacked.flatten_all()?.to_vec1()?;
        let mut result = Vec::with_capacity(n_frames);
        for f in 0..n_frames {
            let start = f * 16;
            result.push(flat[start..start + 16].to_vec());
        }
        Ok(result)
    }

    /// Synthesize speech with a specific voice and language.
    ///
    /// Uses the correct generation loop: CustomVoice prefill, autoregressive
    /// semantic tokens, per-frame acoustic code prediction via CodePredictor,
    /// residual VQ summation, and trailing text fusion.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to synthesize
    /// * `speaker` - Predefined speaker voice
    /// * `language` - Target language
    /// * `options` - Synthesis options (temperature, top_k, etc.)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use tts_qwen::{Qwen3TTS, Speaker, Language, SynthesisOptions};
    ///
    /// let audio = model.synthesize_with_voice(
    ///     "Hello, world!",
    ///     Speaker::Ryan,
    ///     Language::English,
    ///     None,
    /// )?;
    /// audio.save("output.wav")?;
    /// ```
    pub fn synthesize_with_voice(
        &self,
        text: &str,
        speaker: Speaker,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<AudioBuffer> {
        if let Some(ModelType::Base) = &self.model_type {
            tracing::warn!(
                "Using preset speaker {:?} on a Base model. Base models are trained for \
                 voice cloning, not preset speakers — output will have an unpredictable voice. \
                 Use synthesize_voice_clone() with reference audio instead.",
                speaker
            );
        } else if let Some(ModelType::VoiceDesign) = &self.model_type {
            tracing::warn!(
                "Using preset speaker {:?} on a VoiceDesign model. VoiceDesign models \
                 are trained for text-described voice creation, not preset speakers.",
                speaker
            );
        }

        anyhow::ensure!(
            self.supports_preset_speakers(),
            "preset voices require a CustomVoice checkpoint"
        );
        let options = options.unwrap_or_default();
        options.validate()?;
        let mut sampling_ctx = generation::SamplingContext::new(options.seed);
        anyhow::ensure!(!text.trim().is_empty(), "target text must not be empty");
        let input_ids = self.text_tokenizer.encode(text)?;

        let gen_config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            self.build_trailing_text(&input_ids)?;

        // Prefill with CustomVoice format

        let mut kv_caches = self
            .talker
            .new_kv_caches(gen_config.max_new_tokens + input_ids.len() + 16);
        let (hidden, logits) =
            self.talker
                .prefill_custom_voice(&input_ids, speaker, language, &mut kv_caches)?;
        let prefill_len = hidden.dim(1)?;
        let offset = prefill_len;
        let last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

        let all_codes = self.generate_codes(
            &gen_config,
            &mut sampling_ctx,
            &mut kv_caches,
            offset,
            last_hidden,
            &logits,
            &trailing_text_hidden,
            trailing_text_len,
            &tts_pad_embed,
        )?;

        // Decode to audio

        self.decode_codes(&all_codes)
    }

    /// Synthesize speech using a text-described voice (VoiceDesign model).
    ///
    /// Uses the same generation loop as [`Self::synthesize_with_voice`] but runs the
    /// VoiceDesign prefill instead of the predefined-speaker prefill. The voice
    /// is conditioned on a natural language description (e.g., "A cheerful young
    /// female voice with high pitch and energetic tone").
    ///
    /// The instruct text is tokenized with ChatML framing:
    /// `<|im_start|>user\n{instruct}<|im_end|>\n`
    ///
    /// # Arguments
    ///
    /// * `text` - Text to synthesize
    /// * `instruct` - Natural language voice description
    /// * `language` - Target language
    /// * `options` - Synthesis options (temperature, top_k, etc.)
    pub fn synthesize_voice_design(
        &self,
        text: &str,
        instruct: &str,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<AudioBuffer> {
        anyhow::ensure!(
            self.supports_voice_design(),
            "voice design requires a VoiceDesign checkpoint"
        );

        let options = options.unwrap_or_default();
        options.validate()?;
        let mut sampling_ctx = generation::SamplingContext::new(options.seed);
        anyhow::ensure!(!text.trim().is_empty(), "target text must not be empty");
        let input_ids = self.text_tokenizer.encode(text)?;

        // Tokenize instruct with ChatML user framing: <|im_start|>user\n{instruct}<|im_end|>\n
        let instruct_text = format!("<|im_start|>user\n{}<|im_end|>\n", instruct);
        let instruct_ids = self.text_tokenizer.encode(&instruct_text)?;

        let gen_config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            self.build_trailing_text(&input_ids)?;

        // Prefill with VoiceDesign format

        let mut kv_caches = self
            .talker
            .new_kv_caches(gen_config.max_new_tokens + input_ids.len() + instruct_ids.len() + 16);
        let (hidden, logits) = self.talker.prefill_voice_design(
            &input_ids,
            &instruct_ids,
            language,
            &mut kv_caches,
        )?;
        let prefill_len = hidden.dim(1)?;
        let offset = prefill_len;
        let last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

        let all_codes = self.generate_codes(
            &gen_config,
            &mut sampling_ctx,
            &mut kv_caches,
            offset,
            last_hidden,
            &logits,
            &trailing_text_hidden,
            trailing_text_len,
            &tts_pad_embed,
        )?;

        // Decode to audio

        self.decode_codes(&all_codes)
    }

    /// Convert list of frame codes to tensor [batch, 16, num_frames]
    pub fn codes_to_tensor(&self, codes: &[Vec<u32>]) -> Result<Tensor> {
        codes_to_tensor(codes, &self.device)
    }

    /// Decode raw frame codes to audio.
    ///
    /// Takes a slice of frames (each frame is a `Vec<u32>` of 16 codebook values)
    /// and runs the 12Hz decoder to produce an audio waveform at 24kHz.
    pub fn decode_codes(&self, codes: &[Vec<u32>]) -> Result<AudioBuffer> {
        let tensor = self.codes_to_tensor(codes)?;
        self.decode_tensor(&tensor)
    }

    /// Decode a codes tensor `[1, 16, T]` to audio.
    pub(super) fn decode_tensor(&self, codes: &Tensor) -> Result<AudioBuffer> {
        let waveform = self.decoder.decode(codes)?;
        AudioBuffer::from_tensor(waveform, 24000)
    }

    /// Get the device this model is running on
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Apply repetition penalty, token suppression, and min_new_tokens EOS suppression
    /// using a pre-built `[1, vocab]` penalty mask on GPU.
    ///
    /// The mask is updated incrementally via [`update_penalty_mask`] after each
    /// sampled token, eliminating the O(n) GPU→CPU transfer that grows with
    /// each frame.
    pub(super) fn apply_generation_penalties_gpu(
        &self,
        logits: &Tensor,
        penalty_mask: &Tensor,
        config: &generation::GenerationConfig,
        token_count: usize,
        suppression_mask: Option<&generation::SuppressionMask>,
    ) -> Result<Tensor> {
        let logits = logits.to_dtype(DType::F32)?;

        // 1. Repetition penalty via pre-built GPU mask
        let logits = if config.repetition_penalty != 1.0 {
            generation::apply_repetition_penalty_with_mask(
                &logits,
                penalty_mask,
                config.repetition_penalty,
            )?
        } else {
            logits
        };

        // 2. Token suppression
        let logits = if let Some(mask) = suppression_mask {
            generation::apply_token_suppression_with_mask(&logits, mask)?
        } else {
            generation::apply_token_suppression(
                &logits,
                codec_tokens::CODEC_VOCAB_SIZE,
                CODEC_EOS_TOKEN_ID,
            )?
        };

        // 3. Min new tokens EOS suppression
        if token_count < config.min_new_tokens
            && let Some(eos_id) = config.eos_token_id
        {
            let vocab = logits.dim(1)?;
            let batch = logits.dim(0)?;
            let mut mask_data = vec![0.0f32; vocab];
            mask_data[eos_id as usize] = 1.0;
            let eos_mask = Tensor::new(mask_data.as_slice(), logits.device())?
                .unsqueeze(0)?
                .broadcast_as((batch, vocab))?;
            let neg_inf =
                Tensor::new(&[f32::NEG_INFINITY], logits.device())?.broadcast_as((batch, vocab))?;
            let zeros = Tensor::zeros((batch, vocab), DType::F32, logits.device())?;
            let is_eos = eos_mask.gt(&zeros)?;
            return Ok(is_eos.where_cond(&neg_inf, &logits)?);
        }

        Ok(logits)
    }

    /// Returns `true` if a speech encoder is loaded (ICL voice cloning is available).
    pub fn has_speech_encoder(&self) -> bool {
        self.speech_encoder.is_some()
    }
}
