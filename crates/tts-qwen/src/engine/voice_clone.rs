//! voice clone operations.
use super::*;

impl Qwen3TTS {
    /// Synthesize speech using a cloned voice, returning raw codes alongside audio.
    ///
    /// Identical to [`synthesize_voice_clone`](Self::synthesize_voice_clone) but also
    /// returns the raw generated codes (`Vec<Vec<u32>>`) for debugging.
    /// Each inner `Vec<u32>` is one frame: `[semantic, acoustic_0..14]` (16 values).
    pub fn synthesize_voice_clone_debug(
        &self,
        text: &str,
        prompt: &VoiceClonePrompt,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<(AudioBuffer, FrameCodes)> {
        self.voice_clone_run(text, prompt, language, options, false)
            .map(|(audio, codes, _)| (audio, codes))
    }

    /// Synthesize a voice-clone prompt with synchronized stage timings.
    ///
    /// Prefill includes text tokenization and cache setup. Generation covers
    /// autoregressive codec prediction; decode includes reference-context trimming.
    /// Model loading and reference feature extraction are separate operations.
    pub fn synthesize_voice_clone_with_timing(
        &self,
        text: &str,
        prompt: &VoiceClonePrompt,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<(AudioBuffer, SynthesisTiming)> {
        self.voice_clone_run(text, prompt, language, options, true)
            .map(|(audio, _, timing)| (audio, timing))
    }

    fn voice_clone_run(
        &self,
        text: &str,
        prompt: &VoiceClonePrompt,
        language: Language,
        options: Option<SynthesisOptions>,
        timed: bool,
    ) -> Result<(AudioBuffer, FrameCodes, SynthesisTiming)> {
        let options = options.unwrap_or_default();
        options.validate()?;
        if timed {
            sync_device(&self.device)?;
        }
        let prefill_started = Instant::now();
        let mut sampling_ctx = generation::SamplingContext::new(options.seed);
        anyhow::ensure!(!text.trim().is_empty(), "target text must not be empty");
        let input_ids = self.text_tokenizer.encode(text)?;

        let gen_config = options.to_gen_config();

        // Cast speaker embedding to compute dtype (speaker encoder produces F32)
        let speaker_embed = prompt.speaker_embedding.to_dtype(self.compute_dtype)?;

        // Voice clone prefill (9 positions for ICL, 10 for x_vector_only)

        let ref_frames = prompt
            .ref_codes
            .as_ref()
            .map(|codes| codes.dim(0))
            .transpose()?
            .unwrap_or(0);
        let ref_tokens = prompt.ref_text_ids.as_ref().map_or(0, Vec::len);
        // ICL aligns reference+target text against BOS+reference codec frames.
        // Reserve for the longer prefix, the role/control prefix and every output step.
        let icl_len = (input_ids.len() + ref_tokens + 1).max(ref_frames + 1);
        let capacity = gen_config
            .max_new_tokens
            .checked_add(icl_len)
            .and_then(|n| n.checked_add(16))
            .ok_or_else(|| anyhow::anyhow!("KV cache length overflow"))?;
        let mut kv_caches = self.talker.new_kv_caches(capacity)?;
        let (hidden, logits, trailing_text_hidden) = if let (Some(ref_codes), Some(ref_text_ids)) =
            (&prompt.ref_codes, &prompt.ref_text_ids)
        {
            let ref_codec_embeds = self.sum_ref_codec_embeddings(ref_codes)?;
            self.talker.prefill_voice_clone_icl(
                &input_ids,
                &speaker_embed,
                language,
                ref_text_ids,
                &ref_codec_embeds,
                &mut kv_caches,
            )?
        } else {
            let (hidden, logits) = self.talker.prefill_voice_clone(
                &input_ids,
                &speaker_embed,
                language,
                false,
                &mut kv_caches,
            )?;
            let trailing = self.build_default_trailing_text(&input_ids)?;
            (hidden, logits, trailing)
        };
        let offset = hidden.dim(1)?;
        let last_hidden = hidden.i((.., offset - 1..offset, ..))?;

        let trailing_text_len = trailing_text_hidden.dim(1)?;
        let tts_pad_embed = self.talker.get_tts_pad_embed()?;

        if timed {
            sync_device(&self.device)?;
        }
        let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;
        let generation_started = Instant::now();
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

        if timed {
            sync_device(&self.device)?;
        }
        let generation_ms = generation_started.elapsed().as_secs_f64() * 1000.0;
        let decode_started = Instant::now();
        // Prepend ref_codes for ICL decoder context (same fix as synthesize_voice_clone)

        let audio = if let Some(ref_codes) = &prompt.ref_codes {
            let ref_frames = self.tensor_to_frame_codes(ref_codes)?;
            let ref_len = ref_frames.len();
            let mut combined = ref_frames;
            combined.extend(all_codes.iter().cloned());

            let mut audio = self.decode_codes(&combined)?;
            let total_frames = combined.len();
            // Proportional cut: matches official Qwen3-TTS Python implementation
            // cut = ref_len / total_len * wav.shape[0]
            let cut_samples = ref_len * audio.len() / total_frames.max(1);
            tracing::debug!(
                "ICL decode: ref_frames={}, gen_frames={}, total_samples={}, cut_samples={}",
                ref_len,
                all_codes.len(),
                audio.len(),
                cut_samples,
            );
            audio.samples = audio.samples[cut_samples.min(audio.len())..].to_vec();
            audio
        } else {
            self.decode_codes(&all_codes)?
        };
        if timed {
            sync_device(&self.device)?;
        }
        let timing = SynthesisTiming {
            prefill_ms,
            generation_ms,
            generation_frames: all_codes.len(),
            decode_ms: decode_started.elapsed().as_secs_f64() * 1000.0,
        };
        Ok((audio, all_codes, timing))
    }

    // ── Voice cloning API ─────────────────────────────────────────────────

    /// Create a voice clone prompt from reference audio.
    ///
    /// When `ref_text` is `None`, produces an **x_vector_only** prompt (speaker
    /// embedding only). When `Some`, produces an **ICL** prompt (speaker embedding
    /// + reference audio codes + reference text) — requires a speech encoder.
    ///
    /// # Errors
    ///
    /// Returns an error if the speaker encoder is not loaded.
    pub fn create_voice_clone_prompt(
        &self,
        ref_audio: &AudioBuffer,
        ref_text: Option<&str>,
    ) -> Result<VoiceClonePrompt> {
        anyhow::ensure!(
            ref_audio.sample_rate > 0
                && !ref_audio.samples.is_empty()
                && ref_audio.samples.iter().all(|v| v.is_finite()),
            "reference audio must be nonempty finite PCM with a nonzero sample rate"
        );
        anyhow::ensure!(
            ref_text.is_none_or(|text| !text.trim().is_empty()),
            "ICL reference transcript must not be empty"
        );
        let encoder = self.speaker_encoder.as_ref().ok_or_else(|| {
            let hint = match &self.model_type {
                Some(ModelType::CustomVoice) => {
                    " CustomVoice models use preset speakers (synthesize_with_voice), \
                     not voice cloning. Use a Base model for voice cloning."
                }
                Some(ModelType::VoiceDesign) => {
                    " VoiceDesign models use text-described voices, not voice cloning. \
                     Use a Base model for voice cloning."
                }
                _ => {
                    " Ensure model weights contain `speaker_encoder.*` keys \
                     (only Base models include a speaker encoder)."
                }
            };
            anyhow::anyhow!("Speaker encoder not available.{}", hint)
        })?;

        // Resample to 24kHz if needed — both encoders assume 24kHz input
        let ref_audio_24k;
        let ref_audio = if ref_audio.sample_rate != 24000 {
            tracing::info!(
                "Resampling reference audio from {}Hz to 24000Hz",
                ref_audio.sample_rate
            );
            ref_audio_24k = audio::resample_to_24k(ref_audio)?;
            &ref_audio_24k
        } else {
            ref_audio
        };

        let speaker_embedding = encoder.encode(ref_audio)?; // [enc_dim]

        // ICL data: encode reference audio to codes and tokenize reference text
        let (ref_codes, ref_text_ids) = if let Some(text) = ref_text {
            let speech_enc = self.speech_encoder.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "ICL voice cloning requires a speech encoder, but it was not loaded. \
                     Ensure the speech tokenizer weights contain encoder keys, or use \
                     x_vector_only mode by passing ref_text=None."
                )
            })?;

            let codes = speech_enc.encode(ref_audio)?; // [T_frames, 16]
            let text_ids = self.text_tokenizer.encode(text)?;

            (Some(codes), Some(text_ids))
        } else {
            (None, None)
        };

        Ok(VoiceClonePrompt {
            speaker_embedding,
            ref_codes,
            ref_text_ids,
        })
    }

    /// Synthesize speech using a cloned voice.
    ///
    /// Uses the same generation loop as [`Self::synthesize_with_voice`] but runs the
    /// voice-clone prefill instead of the predefined-speaker prefill.
    ///
    /// When the prompt contains ICL data (ref_codes + ref_text_ids), the model
    /// is conditioned on reference audio/text to better reproduce the speaker's voice.
    pub fn synthesize_voice_clone(
        &self,
        text: &str,
        prompt: &VoiceClonePrompt,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<AudioBuffer> {
        self.synthesize_voice_clone_debug(text, prompt, language, options)
            .map(|(audio, _codes)| audio)
    }

    /// Convert a ref_codes tensor `[T_frames, 16]` to `Vec<Vec<u32>>` frame format.
    pub(super) fn tensor_to_frame_codes(&self, codes: &Tensor) -> Result<FrameCodes> {
        let (n_frames, n_codebooks) = codes.dims2()?;
        let codes_u32 = codes.to_dtype(DType::U32)?;
        let mut frames = Vec::with_capacity(n_frames);
        for f in 0..n_frames {
            let frame_tensor = codes_u32.i(f)?; // [16]
            let frame_vec: Vec<u32> = frame_tensor.to_vec1()?;
            debug_assert_eq!(frame_vec.len(), n_codebooks);
            frames.push(frame_vec);
        }
        Ok(frames)
    }

    /// Sum reference codec embeddings across all 16 codebook groups.
    ///
    /// For each frame:
    /// - Group 0 (semantic): `talker.codec_embedding(ref_codes[:, 0])`
    /// - Groups 1–15 (acoustic): `code_predictor.codec_embeddings[i-1](ref_codes[:, i])`
    /// - Sum all 16 → single embedding per frame
    ///
    /// # Arguments
    /// * `ref_codes` — shape `[T_frames, 16]` of i64 codes
    ///
    /// # Returns
    /// Tensor of shape `[1, T_frames, hidden_size]`
    pub(super) fn sum_ref_codec_embeddings(&self, ref_codes: &Tensor) -> Result<Tensor> {
        // Group 0: semantic codes → talker.codec_embedding
        let semantic_codes = ref_codes.i((.., 0))?; // [T_frames]
        let semantic_codes = semantic_codes.to_dtype(candle_core::DType::U32)?;
        let summed = self.talker.get_codec_embedding_batch(&semantic_codes)?; // [1, T, hidden]

        // Groups 1-15: acoustic codes → code_predictor.embed_codes_for_group
        let mut summed = summed;
        for group in 1..16 {
            let group_codes = ref_codes.i((.., group))?; // [T_frames]
            let group_codes = group_codes.to_dtype(candle_core::DType::U32)?;
            let group_embed = self
                .code_predictor
                .embed_codes_for_group(group - 1, &group_codes)?; // [1, T, embed_dim]
            summed = summed.add(&group_embed)?;
        }

        Ok(summed)
    }

    /// Build default trailing text embeddings (for non-ICL mode).
    pub(super) fn build_default_trailing_text(&self, input_ids: &[u32]) -> Result<Tensor> {
        let (hidden, _len, _pad) = self.build_trailing_text(input_ids)?;
        Ok(hidden)
    }
}
