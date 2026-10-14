//! streaming operations.
use super::*;

impl Qwen3TTS {
    /// Create a streaming synthesis session with a specific voice and language.
    ///
    /// Returns an iterator that yields audio chunks as they are generated.
    /// Each chunk contains approximately `chunk_frames` frames worth of audio
    /// (default: 10 frames = ~800ms at 12.5 Hz frame rate).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use tts_qwen::{Qwen3TTS, Speaker, Language, SynthesisOptions};
    /// # fn main() -> anyhow::Result<()> {
    /// let model = Qwen3TTS::from_pretrained("path/to/custom-voice", candle_core::Device::Cpu)?;
    /// let options = SynthesisOptions::default();
    /// for chunk in model.synthesize_streaming("Hello!", Speaker::Ryan, Language::English, options)? {
    ///     let _audio = chunk?;
    ///     // Pass this chunk to the application's audio output.
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn synthesize_streaming(
        &self,
        text: &str,
        speaker: Speaker,
        language: Language,
        options: SynthesisOptions,
    ) -> Result<StreamingSession<'_>> {
        anyhow::ensure!(
            self.supports_preset_speakers(),
            "preset voices require a CustomVoice checkpoint"
        );
        anyhow::ensure!(!text.trim().is_empty(), "target text must not be empty");
        options.validate()?;
        let input_ids = self.text_tokenizer.encode(text)?;
        StreamingSession::new(self, &input_ids, speaker, language, options)
    }

    /// Synthesize speech using a text-described voice (VoiceDesign model), streaming.
    ///
    /// Same as [`Self::synthesize_voice_design`] but returns a streaming session
    /// that yields audio chunks as they are generated.
    ///
    /// The instruct text is tokenized with ChatML framing:
    /// `<|im_start|>user\n{instruct}<|im_end|>\n`
    ///
    /// # Arguments
    ///
    /// * `text` - Text to synthesize
    /// * `instruct` - Natural language voice description (e.g., "A cheerful young female voice")
    /// * `language` - Target language
    /// * `options` - Synthesis options (temperature, top_k, chunk_frames, etc.)
    pub fn synthesize_voice_design_streaming(
        &self,
        text: &str,
        instruct: &str,
        language: Language,
        options: SynthesisOptions,
    ) -> Result<StreamingSession<'_>> {
        anyhow::ensure!(
            self.supports_voice_design(),
            "voice design requires a VoiceDesign checkpoint"
        );

        anyhow::ensure!(!text.trim().is_empty(), "target text must not be empty");
        options.validate()?;
        let input_ids = self.text_tokenizer.encode(text)?;

        // Tokenize instruct with ChatML user framing: <|im_start|>user\n{instruct}<|im_end|>\n
        let instruct_text = format!("<|im_start|>user\n{}<|im_end|>\n", instruct);
        let instruct_ids = self.text_tokenizer.encode(&instruct_text)?;

        StreamingSession::new_voice_design(self, &input_ids, &instruct_ids, language, options)
    }
}

/// Streaming synthesis session.
///
/// Yields audio chunks as they are generated. Use with
/// [`Qwen3TTS::synthesize_streaming`].
pub struct StreamingSession<'a> {
    model: &'a Qwen3TTS,
    config: generation::GenerationConfig,
    sampling_ctx: generation::SamplingContext,
    kv_caches: Vec<AnyKVCache>,
    offset: usize,
    last_hidden: Tensor,
    current_token: Option<u32>,
    current_token_tensor: Option<Tensor>,
    frames_generated: usize,
    frame_buffer: FrameCodes,
    // The official decoder uses 25 preceding codec frames at chunk boundaries.
    decoder_context: FrameCodes,
    chunk_frames: usize,
    done: bool,
    // Trailing text state for residual VQ + text fusion
    trailing_text_hidden: Tensor,
    trailing_text_len: usize,
    tts_pad_embed: Tensor,
    // GPU-side repetition penalty mask [1, vocab] — updated incrementally
    penalty_mask: Tensor,
    token_count: usize,
    // Pre-built suppression mask (reused every frame)
    suppression_mask: generation::SuppressionMask,
    // Pre-allocated code predictor KV caches (reused + reset each frame)
    cp_kv_caches: Vec<AnyKVCache>,
}

impl<'a> StreamingSession<'a> {
    pub(super) fn new(
        model: &'a Qwen3TTS,
        input_ids: &[u32],
        speaker: Speaker,
        language: Language,
        options: SynthesisOptions,
    ) -> Result<Self> {
        let sampling_ctx = generation::SamplingContext::new(options.seed);
        let config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            model.build_trailing_text(input_ids)?;

        let mut kv_caches =
            model.generation_caches(config.max_new_tokens, &[input_ids.len(), 16])?;
        let prefill_result =
            model
                .talker
                .prefill_custom_voice(input_ids, speaker, language, &mut kv_caches)?;

        Self::from_prefill(
            model,
            config,
            sampling_ctx,
            kv_caches,
            prefill_result,
            trailing_text_hidden,
            trailing_text_len,
            tts_pad_embed,
            options.chunk_frames,
        )
    }

    /// Create a streaming session using voice design (text-described voice).
    ///
    /// Uses `prefill_voice_design` instead of `prefill_custom_voice` to condition
    /// on a natural language voice description rather than a predefined speaker.
    pub(super) fn new_voice_design(
        model: &'a Qwen3TTS,
        input_ids: &[u32],
        instruct_ids: &[u32],
        language: Language,
        options: SynthesisOptions,
    ) -> Result<Self> {
        let sampling_ctx = generation::SamplingContext::new(options.seed);
        let config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            model.build_trailing_text(input_ids)?;

        let mut kv_caches = model.generation_caches(
            config.max_new_tokens,
            &[input_ids.len(), instruct_ids.len(), 16],
        )?;
        let prefill_result =
            model
                .talker
                .prefill_voice_design(input_ids, instruct_ids, language, &mut kv_caches)?;

        Self::from_prefill(
            model,
            config,
            sampling_ctx,
            kv_caches,
            prefill_result,
            trailing_text_hidden,
            trailing_text_len,
            tts_pad_embed,
            options.chunk_frames,
        )
    }

    /// Shared post-prefill constructor.
    ///
    /// Extracts `last_hidden` from the prefill result, builds the suppression and
    /// penalty masks, samples the first semantic token, and assembles the session.
    #[allow(clippy::too_many_arguments)]
    fn from_prefill(
        model: &'a Qwen3TTS,
        config: generation::GenerationConfig,
        mut sampling_ctx: generation::SamplingContext,
        kv_caches: Vec<AnyKVCache>,
        prefill_result: (Tensor, Tensor),
        trailing_text_hidden: Tensor,
        trailing_text_len: usize,
        tts_pad_embed: Tensor,
        chunk_frames: usize,
    ) -> Result<Self> {
        let (hidden, logits) = prefill_result;
        let prefill_len = hidden.dim(1)?;
        let last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

        // Build suppression mask once for reuse across all frames
        let suppression_mask = generation::build_suppression_mask(
            codec_tokens::CODEC_VOCAB_SIZE,
            CODEC_EOS_TOKEN_ID,
            &model.device,
        )?;

        // Sample first token with full penalty pipeline
        let vocab_size = codec_tokens::CODEC_VOCAB_SIZE;
        let mut penalty_mask = Tensor::zeros((1, vocab_size), DType::F32, &model.device)?;
        let logits_2d = logits.squeeze(1)?;
        let logits_2d = model.apply_generation_penalties_gpu(
            &logits_2d,
            &penalty_mask,
            &config,
            0,
            Some(&suppression_mask),
        )?;
        let first_token = generation::sample(&logits_2d, &config, &mut sampling_ctx)?;
        let first_token_id: u32 = first_token.flatten_all()?.to_vec1::<u32>()?[0];
        Qwen3TTS::update_penalty_mask(&mut penalty_mask, first_token_id, vocab_size)?;

        let done = config.eos_token_id == Some(first_token_id);
        let cp_kv_caches = model.code_predictor.new_kv_caches();

        Ok(Self {
            model,
            config,
            sampling_ctx,
            kv_caches,
            offset: prefill_len,
            last_hidden,
            current_token: if done { None } else { Some(first_token_id) },
            current_token_tensor: if done { None } else { Some(first_token) },
            frames_generated: 0,
            frame_buffer: Vec::new(),
            decoder_context: Vec::new(),
            chunk_frames,
            done,
            trailing_text_hidden,
            trailing_text_len,
            tts_pad_embed,
            penalty_mask,
            token_count: 1,
            suppression_mask,
            cp_kv_caches,
        })
    }

    /// Generate the next chunk of audio.
    ///
    /// Returns `Some(AudioBuffer)` for each chunk, or `None` when generation is complete.
    pub fn next_chunk(&mut self) -> Result<Option<AudioBuffer>> {
        if self.done {
            // Flush remaining buffer
            if !self.frame_buffer.is_empty() {
                return self.decode_buffered_frames().map(Some);
            }
            return Ok(None);
        }

        // Generate frames until we have enough for a chunk
        while self.frame_buffer.len() < self.chunk_frames
            && self.frames_generated < self.config.max_new_tokens.saturating_sub(1)
        {
            let (token_id, token_tensor) =
                match (self.current_token, self.current_token_tensor.take()) {
                    (Some(id), Some(t)) => (id, t),
                    _ => {
                        self.done = true;
                        break;
                    }
                };

            // Embedding lookup using GPU-resident token tensor (no CPU→GPU roundtrip)
            let semantic_embed = self
                .model
                .talker
                .get_codec_embedding_from_tensor(&token_tensor)?;

            // Generate 15 acoustic codes (stays on GPU)
            let acoustic_codes_tensor = self.model.code_predictor.generate_acoustic_codes(
                &self.last_hidden,
                &semantic_embed,
                &mut self.cp_kv_caches,
                &self.config.acoustic(),
                &mut self.sampling_ctx,
            )?;

            // Build frame on GPU, then transfer for frame_buffer
            let semantic_t = Tensor::new(&[token_id], self.model.device())?;
            let frame_tensor = Tensor::cat(&[&semantic_t, &acoustic_codes_tensor], 0)?;
            let frame_codes: Vec<u32> = frame_tensor.to_vec1()?;
            self.frame_buffer.push(frame_codes);

            let frame_idx = self.frames_generated;
            self.frames_generated += 1;

            // Build residual VQ sum + trailing text for next step
            let acoustic_embed_sum = self
                .model
                .code_predictor
                .get_acoustic_embeddings_sum_from_tensor(&acoustic_codes_tensor)?;
            let summed = semantic_embed.add(&acoustic_embed_sum)?;

            let text_addition = if frame_idx < self.trailing_text_len {
                self.trailing_text_hidden
                    .i((.., frame_idx..frame_idx + 1, ..))?
            } else {
                self.tts_pad_embed.clone()
            };
            let step_input = summed.add(&text_addition)?;

            // Run talker step with fused embedding
            let (h, new_logits) = self.model.talker.generate_step_with_embed(
                &step_input,
                &mut self.kv_caches,
                self.offset,
            )?;
            self.offset += 1;
            self.last_hidden = h;

            // Sample next semantic token with repetition penalty + token suppression + min_new_tokens
            let logits_2d = new_logits.squeeze(1)?;
            let logits_2d = self.model.apply_generation_penalties_gpu(
                &logits_2d,
                &self.penalty_mask,
                &self.config,
                self.token_count,
                Some(&self.suppression_mask),
            )?;
            let next_token_tensor =
                generation::sample(&logits_2d, &self.config, &mut self.sampling_ctx)?;
            let next_token_id: u32 = next_token_tensor.flatten_all()?.to_vec1::<u32>()?[0];
            Qwen3TTS::update_penalty_mask(
                &mut self.penalty_mask,
                next_token_id,
                codec_tokens::CODEC_VOCAB_SIZE,
            )?;
            self.token_count += 1;

            if self.config.eos_token_id == Some(next_token_id) {
                self.current_token = None;
                self.current_token_tensor = None;
                self.done = true;
            } else {
                self.current_token = Some(next_token_id);
                self.current_token_tensor = Some(next_token_tensor);
            }
        }

        // Decode the buffered frames
        if self.frame_buffer.is_empty() {
            return Ok(None);
        }

        self.decode_buffered_frames().map(Some)
    }

    fn decode_buffered_frames(&mut self) -> Result<AudioBuffer> {
        const LEFT_CONTEXT_FRAMES: usize = 25;
        let context_frames = self.decoder_context.len();
        let mut frames = std::mem::take(&mut self.decoder_context);
        frames.append(&mut self.frame_buffer);
        let codes = self.model.codes_to_tensor(&frames)?;
        let decoded = self.model.decoder.decode(&codes)?;
        let mut audio = AudioBuffer::from_tensor(decoded, 24_000)?;
        let cut = context_frames * SAMPLES_PER_FRAME;
        anyhow::ensure!(
            cut <= audio.samples.len(),
            "decoder returned fewer samples than its context"
        );
        audio.samples.drain(..cut);
        self.decoder_context = frames[frames.len().saturating_sub(LEFT_CONTEXT_FRAMES)..].to_vec();
        Ok(audio)
    }

    /// Returns the total number of frames generated so far.
    pub fn frames_generated(&self) -> usize {
        self.frames_generated
    }

    /// Returns true if generation is complete.
    pub fn is_done(&self) -> bool {
        self.done && self.frame_buffer.is_empty()
    }
}

impl<'a> Iterator for StreamingSession<'a> {
    type Item = Result<AudioBuffer>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_chunk() {
            Ok(Some(audio)) => Some(Ok(audio)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}
