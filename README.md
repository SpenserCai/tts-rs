<div align="center">

# tts-rs

**Local voices. Explicit models. One Rust interface.**

[![crates.io](https://img.shields.io/crates/v/candle-tts?logo=rust)](https://crates.io/crates/candle-tts)
[![PyPI](https://img.shields.io/pypi/v/candle-tts?logo=python)](https://pypi.org/project/candle-tts/)
[![Node.js](https://img.shields.io/npm/v/%40spensercai%2Ftts-rs?logo=npm&label=Node.js)](https://www.npmjs.com/package/@spensercai/tts-rs)
[![WASM32](https://img.shields.io/npm/v/%40spensercai%2Ftts-wasm?label=WASM32)](https://www.npmjs.com/package/@spensercai/tts-wasm)
[![WASM64](https://img.shields.io/npm/v/%40spensercai%2Ftts-wasm64?label=WASM64)](https://www.npmjs.com/package/@spensercai/tts-wasm64)

[![Rust 2024](https://img.shields.io/badge/Rust-2024-DEA584?logo=rust)](Cargo.toml)
[![Candle](https://img.shields.io/badge/inference-Candle-FFD21E)](https://github.com/huggingface/candle)
[![Release](https://github.com/SpenserCai/tts-rs/actions/workflows/release.yml/badge.svg)](https://github.com/SpenserCai/tts-rs/actions/workflows/release.yml)
[![License](https://img.shields.io/badge/license-MIT-22C55E)](LICENSE)

Qwen3-TTS · CosyVoice3 · Python · Node.js · WebAssembly

[Architecture](docs/architecture.md) · [Validation](docs/validation.md) · [Contributing](AGENTS.md)

</div>

`candle-tts` is a modular, local-inference speech library built with Hugging Face
Candle. It separates model-specific voice conditioning and speech synthesis from
audio utilities, application interfaces, and language runtimes.

The repository is named **tts-rs**; the Rust and Python package is **candle-tts**.
Models run locally, and model weights and recordings stay outside the packages.
Choose a model and runtime from the support matrix below; numerical evidence is
recorded per component and device.

Package names and installation commands are listed below. Published versions and
downloadable distributions are listed on the
[Releases page](https://github.com/SpenserCai/tts-rs/releases). Source-build commands
are available in the [Python](bindings/python/README.md),
[Node](bindings/node/README.md) and [WASM](bindings/wasm/README.md) guides.

Measured CPU comparisons are recorded in the
[validation baseline](tests/fixtures/validation-baseline.json), with the exact
scope, upstream revision, tolerances and errors. Component checks, full synthesis
checks and voice quality are distinct; the support table below states which
platforms have actually been exercised.

Both native pipelines implement reference conditioning and speech synthesis.
Qwen's bounded CPU x-vector and ICL comparisons pass the strict tensor gate.
CosyVoice's bounded frontend, language-model and short HiFT comparisons pass. Strict
full-Flow parity remains unresolved: three of four latest analytic cases still
fail the unchanged tolerance, and improvements are not uniform across inputs.
See [numerical validation](docs/validation.md) for the tested boundaries; complete
CosyVoice waveform equivalence is not claimed.

**CosyVoice3 remains experimental.** Version 0.1.1 fixes categorical-sampling
rounding that could select a zero-probability token, and improves CPU float32
HiFT phase arithmetic. Fixed-seed output may change because the sampling bug is
corrected. A longer same-input HiFT control improves from 47.45 to 101.08 dB error
SNR, but still fails the strict waveform threshold at 38 of 100,800 values.
See the [0.1.1 numerical regression](docs/validation.md#cosyvoice-011-numerical-regression).
These corrections do not establish that reported leading sounds or elongated
endings are eliminated. Evaluate the output for the intended use; full-Flow
differences and listening quality remain separate from the passing component
checks, and no numerical tolerance has been relaxed.

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

```toml
[dependencies]
candle-tts = "0.1"
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

CUDA/Metal **preview wheels and Node tarballs** are distributed as explicit
GitHub Release assets. They use separate package/import
names and can coexist with the CPU packages. See the
[accelerator matrix and installation guide](docs/accelerators.md) for runtime
requirements and the build-only validation boundary. They are not GPU-certified
replacements for the default packages; browser WASM packages remain CPU-only.

Install the native CPU packages:

```sh
python -m pip install candle-tts
npm install @spensercai/tts-rs
```

Python requires CPython 3.10+; Node.js requires 22+. Both expose the same model
families, explicit device selection and portable voice-prompt JSON:

```python
from candle_tts import Engine

engine = Engine.load("models/qwen-base", "qwen3", "cpu")
# reference_pcm: mono floating-point samples at the stated sample rate.
voice = engine.extract_voice(reference_pcm, 24000, None)
speech = engine.synthesize("Hello from Python.", voice)
samples, sample_rate = speech.samples, speech.sample_rate
```

```js
import { loadEngine } from '@spensercai/tts-rs'

const engine = await loadEngine('models/qwen-base', 'qwen3', 'cpu')
// referencePcm: Float32Array of mono PCM at the stated sample rate.
const voice = await engine.extractVoice(referencePcm, 24000)
const speech = await engine.synthesize('Hello from Node.js.', voice)
// speech.samples: Float32Array; speech.sampleRate: number
```

Python releases the GIL while model work runs; Node dispatches it to worker
threads. Keep the loaded engine and extracted voice prompt for repeated requests.
Use `cosyvoice3` with a converted model and the reference transcript to select
CosyVoice. Prompts are specific to the model checkpoint that created them.

Runtime details, source builds and executable smoke tests live next to each binding:

- [Python](bindings/python/README.md): PyO3 + maturin; import `candle_tts`.
- [Node.js](bindings/node/README.md): napi-rs; asynchronous native model work.
- [WASM](bindings/wasm/README.md): wasm-bindgen; query `capabilities()` rather than
  assuming that a browser build has native accelerator or full-model support.

WebAssembly has two explicit packages: `@spensercai/tts-wasm` provides portable
audio operations and Qwen speaker extraction using 32-bit memory;
`@spensercai/tts-wasm64` provides Qwen synthesis using WebAssembly Memory64.
The latter requires a Memory64-capable runtime, Node.js 24+ for Node use, and
enough physical memory for weights, caches and activations. Neither package
bundles weights or implements CosyVoice's ONNX frontend. Load model bytes
explicitly and run inference in a worker; see the [WASM guide](bindings/wasm/README.md).

CPU is the default. `--features cuda` requires an NVIDIA CUDA toolchain;
`--features metal` targets Apple Silicon. Device selection is explicit and errors
if the requested accelerator is unavailable. Do not enable all hardware features
at once. CUDA/Metal parity must be measured on that hardware.

| Target | Implemented boundary | Validation boundary |
| --- | --- | --- |
| Linux x64 CPU | Both native model pipelines; Python and Node | Local build, package smoke and scoped model comparisons |
| Windows x64 CPU | Native Rust / Python / Node | Release wheel/addon builds and final package installation smoke tests |
| macOS ARM CPU / Metal | Native packages and explicit Metal feature | CPU release wheel/addon builds and package installation smoke tests; Metal numerical checks remain unverified |
| Linux CUDA | Explicit native feature | Opt-in CUDA runner; not certified by CPU tests |
| WASM32 | Audio utilities and Qwen speaker extraction | Real speaker-weight CPU comparison |
| WASM64 | Memory64 Qwen synthesis with explicit model bytes | Node 24 full bounded Qwen waveform parity; real JS/Rust transfer above 4 GiB |

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

## Development and releases

For native tests, first run `python scripts/install_onnxruntime.py` (Python 3.12+)
and set `ORT_DYLIB_PATH` to the library path it reports. This verified external
runtime also runs the small CosyVoice frontend operator regression; no model
weights are needed for the ordinary Rust gate.

```sh
python scripts/ci.py rust
python scripts/ci.py bindings --kind python
python scripts/ci.py bindings --kind node
python scripts/ci.py bindings --kind wasm
python scripts/wasm64.py prepare
python scripts/wasm64.py build
python scripts/wasm64.py check
python scripts/version.py check
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
| Release | Publish a new unified version after a verified PR merge; manual artifact builds and publication |

CI, benchmarks and model evaluation use `workflow_dispatch`. Heavy model jobs and
accelerator runners remain opt-in. All distributions share the Cargo workspace
version. Use `python scripts/version.py set X.Y.Z` to update release metadata
together before merging a release change.

Merging a PR into `main` starts the **Release** workflow. A lightweight planner
verifies that the exact commit belongs to a merged PR and reads its workspace
version. An already published version is a no-op, so a documentation update does
not rebuild packages. Direct pushes do not publish automatically.

The workflow builds and tests Windows x64, Linux x64 and macOS ARM wheels and Node
addons, packages both WebAssembly targets, and verifies the distribution inventory
and checksums. Only then does it create the matching immutable `vX.Y.Z` tag and
publish through `CARGO_REGISTRY_TOKEN`, `PYPI_API_TOKEN` and `NPM_TOKEN`.
The complete GitHub release becomes public after every registry upload and release
asset check succeeds. Tags and versions do not need to be entered separately.

The manual entry point supports building artifacts without publishing, or
publishing a commit already merged into `main`:

```sh
gh workflow run release.yml --repo SpenserCai/tts-rs --ref main -f destination=artifacts
gh workflow run release.yml --repo SpenserCai/tts-rs --ref main -f destination=registries
```

After a partial publication, use **Re-run failed jobs** on the original Actions
run. It reuses the verified packages and checks registry digests before skipping
identical uploads. Do not rebuild a reserved version or move its tag. See the
[maintenance requirements](AGENTS.md) for versioning and retry rules.

## Attribution

New project code is MIT licensed. Imported implementations retain their upstream
licenses and attribution: [Qwen provenance](crates/tts-qwen/UPSTREAM.md),
[CosyVoice provenance](crates/tts-cosyvoice/UPSTREAM.md). The authoritative models
are [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) and
[CosyVoice](https://github.com/QwenAudio/CosyVoice). Model weights have separate
model-card licenses.
