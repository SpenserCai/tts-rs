<div align="center">

# tts-rs

**Local voices. Explicit models. One Rust interface.**

[![Rust 2024](https://img.shields.io/badge/Rust-2024-DEA584?logo=rust)](Cargo.toml)
[![Candle](https://img.shields.io/badge/inference-Candle-FFD21E)](https://github.com/huggingface/candle)
[![Manual CI](https://img.shields.io/badge/CI-manual_dispatch-2563EB)](.github/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-22C55E)](LICENSE)

Qwen3-TTS · CosyVoice3 · Python · Node.js · WebAssembly

[Architecture](docs/architecture.md) · [Validation](docs/validation.md) · [Contributing](AGENTS.md)

</div>

`candle-tts` is a modular, local-inference speech library built with Hugging Face
Candle. It separates model-specific voice conditioning and speech synthesis from
audio utilities, application interfaces, and language runtimes.

The repository is named **tts-rs**. The Rust package is **candle-tts**, because the
`tts-rs` registry name belongs to a different project. This is a pre-release
implementation; consult the validation evidence before selecting a deployment.
No model weights or voice recordings are included.

Measured CPU comparisons are recorded in the
[validation baseline](tests/fixtures/validation-baseline.json), with the exact
scope, upstream revision, tolerances and errors. Component checks, full synthesis
checks and voice quality are distinct; the support table below states which
platforms have actually been exercised.

Qwen's bounded CPU x-vector and ICL synthesis comparisons pass the strict tensor
gate. CosyVoice's official-runtime language-model and HiFT stage comparisons also
pass; its float32 flow and coupled acoustic path still fail the same elementwise
threshold. The baseline records those failures alongside the official Python
attention-kernel variation. Do not treat them as full CosyVoice waveform parity.

## What belongs where

| Package | Responsibility |
| --- | --- |
| `candle-tts` | Unified `Engine`, checkpoint identity, persisted voice prompts |
| `tts-core` | Mono PCM, WAV, band-limited resampling, portable contracts |
| `tts-qwen` | Qwen3 speaker encoder, Talker, code predictor and 12 Hz codec |
| `tts-cosyvoice` | CosyVoice3 frontend, Qwen2, conditional flow and HiFT |
| Python / Node | Native bindings with blocking inference off the interpreter |
| WASM | Explicit portable components and optional model capabilities |

Voice prompts retain their model family, format version and checkpoint content
fingerprint. A 1024-dimensional Qwen speaker vector is not interchangeable with a
192-dimensional CosyVoice embedding. Full reference conditioning can also contain
audio tokens, mel features and a matching transcript.

## Rust

Until a registry release is published, use the repository dependency:

```toml
[dependencies]
candle-tts = { git = "https://github.com/SpenserCai/tts-rs", branch = "feat/modular-candle-tts" }
```

```rust,no_run
use candle_tts::{Audio, DeviceConfig, Engine, ModelKind, SynthesisOptions};

fn main() -> candle_tts::Result<()> {
    let reference = Audio::read_wav(std::fs::File::open("reference.wav")?)?;
    let mut engine = Engine::load("models/qwen-base", ModelKind::Qwen3, DeviceConfig::Cpu)?;
    let voice = engine.extract_voice(&reference.samples, reference.sample_rate, None)?;
    let speech = engine.synthesize("Hello from Rust.", &voice, &SynthesisOptions::default())?;
    speech.write_wav(std::fs::File::create("speech.wav")?)?;
    Ok(())
}
```

For Qwen, `None` selects speaker-only conditioning. Supply the **actual reference
transcript** for in-context voice cloning. CosyVoice zero-shot synthesis requires
the transcript. Reference audio is finite mono PCM, up to 30 seconds. Generated
audio is 24 kHz mono. Text normalization remains caller-controlled.

The low-level `qwen` and `cosyvoice` modules expose model-specific controls. For
example, Qwen CustomVoice presets and VoiceDesign differ from Base-model cloning;
the unified voice-prompt API targets cloning, not every model-specific mode.

## Bindings

Build instructions and executable smoke tests live next to each binding:

- [Python](bindings/python/README.md): PyO3 + maturin; import `candle_tts`.
- [Node.js](bindings/node/README.md): napi-rs; asynchronous native model work.
- [WASM](bindings/wasm/README.md): wasm-bindgen; query `capabilities()` rather than
  assuming that a browser build has native accelerator or full-model support.

CPU is the default. `--features cuda` requires an NVIDIA CUDA toolchain;
`--features metal` targets Apple Silicon. Device selection is explicit and errors
if the requested accelerator is unavailable. Do not enable all hardware features
at once. CUDA/Metal parity must be measured on that hardware.

| Target | Implemented boundary | Validation boundary |
| --- | --- | --- |
| Linux x64 CPU | Both native model pipelines; Python and Node | Local build, package smoke and scoped model comparisons |
| Windows x64 CPU | Native Rust / Python / Node build configuration | Manual matrix; requires a Windows run |
| macOS ARM CPU / Metal | Native build configuration and explicit Metal feature | Manual matrix; numerical GPU checks require hardware |
| Linux CUDA | Explicit native feature | Opt-in CUDA runner; not certified by CPU tests |
| WASM | Audio utilities and optional Qwen speaker extraction | Real speaker-weight CPU comparison; no full synthesis |

The Linux Node release build targets Ubuntu 24.04 / glibc 2.39 or newer. Python
Linux release wheels target manylinux 2.28. ONNX Runtime is an explicit external
dependency for CosyVoice reference extraction, not an undeclared bundled library.

## Models and reproducibility

Qwen expects a local official Base checkpoint with its speech tokenizer under
`speech_tokenizer/`. CosyVoice uses converted safetensors plus its complete text
tokenizer and ONNX frontend graphs. Its conversion tool is
[`crates/tts-cosyvoice/scripts/convert.py`](crates/tts-cosyvoice/scripts/convert.py).
Keep source revisions and weight digests with every experiment.

The reference suite installs pinned official Python implementations, exports
intermediate tensors, checks finite values and shapes, and compares them with
Candle. Stochastic waveform identity is not inferred from equal random seeds.
See [validation commands and boundaries](docs/validation.md).

## Development and manual workflows

```sh
python scripts/ci.py rust
python scripts/ci.py bindings --kind python
python scripts/ci.py bindings --kind node
python scripts/ci.py bindings --kind wasm
```

Rust uses edition 2024 and a pinned toolchain. Formatting, Clippy and documentation
are strict: zero errors, zero warnings. CosyVoice's auxiliary ONNX frontend uses
ONNX Runtime 1.24 or newer. Set `ORT_DYLIB_PATH` to its shared library before
extracting reference conditioning; the Candle synthesis models remain native Rust.

| Workflow | Purpose |
| --- | --- |
| Manual CI | Cheap Linux gate, optional Windows x64 / Linux x64 / macOS ARM matrix |
| Benchmark | Criterion audio benchmarks and machine metadata |
| Model evaluation | Tensor parity, Qwen CPU waveform E2E, component timings and opt-in resident Qwen TTS benchmark |
| Release | Build and test artifacts; dry run by default; opt-in draft or registry publish |

Every workflow uses `workflow_dispatch`. Heavy model jobs and accelerator runners
are opt-in. Registry publishing requires configured identities and a matching
existing release tag; creating this repository does not publish any package.

## Attribution

New project code is MIT licensed. Imported implementations retain their upstream
licenses and attribution: [Qwen provenance](crates/tts-qwen/UPSTREAM.md),
[CosyVoice provenance](crates/tts-cosyvoice/UPSTREAM.md). The authoritative models
are [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) and
[CosyVoice](https://github.com/QwenAudio/CosyVoice). Model weights have separate
model-card licenses.
