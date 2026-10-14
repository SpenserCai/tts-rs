# Source provenance

The Qwen inference implementation is adapted from [TrevorS/qwen3-tts-rs](https://github.com/TrevorS/qwen3-tts-rs) at commit `711ceee07cad92673f86de8997bdf54c30caa49f` (package `qwen3-tts` 0.3.0, MIT as declared in its Cargo.toml and README). Copyright remains with the upstream authors and contributors. The upstream snapshot contains no separate LICENSE file; its license declaration is recorded here without fabricating a copyright year.

The authoritative Python implementation is [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) at `022e286b98fbec7e1e916cb940cdf532cd9f488e` (Apache-2.0). Model weights have their own model-card license. No weights or voice recordings are distributed by this crate.

Changes here:

- Rust 2024 and a thin public API over separate loading, synthesis, voice-clone, streaming and options modules.
- Local-only checkpoint loading with one deterministic speech-tokenizer directory; no CLI, download client, global trace subscriber or custom PTX RMSNorm.
- Official attentive-pooling variance floor (`clamp(min=1e-12)`), explicit regex pre-tokenization, automatic language prefix and dynamic prefix lengths.
- Independent acoustic-predictor sampling with official defaults; no hidden ICL repetition or maximum-length overrides.
- Reference-length-aware KV capacity, validated generation settings and shared delay-compensated resampling from `tts-core`.
- Portable speaker, mel and full synthesis oracle executables for comparison with the pinned Python implementation.

An upstream port is not evidence of numerical parity. Fixtures and full-checkpoint validation record the exact Python revision, weight digest, dtype, device and observed errors. CPU checks do not certify CUDA or Metal parity.
