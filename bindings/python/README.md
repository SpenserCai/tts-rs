# SpeechFoundry for Python

Thin [PyO3](https://pyo3.rs/) bindings over the same native Rust engine. Model loading,
voice extraction, synthesis, and resampling release the Python interpreter lock.
Concurrent operations on one `Engine` are serialized; create separate engines when
independent inference state is needed.

CosyVoice3 is experimental: full-Flow numerical parity remains unresolved, and
reported output artifacts include an unrelated leading sound and elongated
endings. See the [project validation status](https://github.com/SpenserCai/tts-rs#speechfoundry)
before using that model. The binding uses the same model implementation as Rust.

```sh
python -m pip install speechfoundry
```

```python
import json
from speechfoundry import Engine

engine = Engine.load("/models/Qwen3-TTS-12Hz-0.6B-Base", "qwen3", "cpu")
# reference_pcm is mono float32 PCM; pass the actual recording sample rate.
voice = engine.extract_voice(reference_pcm, 24000, "Reference transcript.")
audio = engine.synthesize("Hello.", voice, json.dumps({"seed": 42}))
samples, sample_rate = audio.samples, audio.sample_rate
```

`voice` is model-specific conditioning, not a universal biometric identity or a
newly trained voice model. Keep its model and schema fields intact when storing it.
`options_json` uses the shared Rust synthesis options and defaults.

`Audio(samples, sample_rate)` validates finite PCM samples. Accessing `.samples`
copies them to a Python list; no NumPy runtime dependency is required.

The default wheel is CPU-only. Build from source with `--features cuda` or
`--features metal` for a matching toolchain and target. A feature-enabled binary
does not imply a compatible accelerator exists on the current machine.

For prebuilt GPU previews, install the same-version backend wheel from GitHub
Releases **alongside** this package. The import remains unchanged:

```python
from speechfoundry import Engine, installed_backends

print(installed_backends())  # Installed builds, not a hardware readiness test.
gpu = Engine.load("models/qwen-base", "qwen3", "cuda:0")
# With several CUDA builds installed, add backend="cu124".
# On Apple Silicon use device="metal:0" with the matching Metal preview.
```

Backend distribution names identify CUDA builds, but are not application import
names. Missing/ambiguous backends and version mismatches are explicit errors.
CPU imports never load accelerator libraries. See the
[accelerator guide](https://github.com/SpenserCai/tts-rs/blob/main/docs/accelerators.md)
for installation and native runtime requirements. GPU previews remain build-verified,
not GPU-parity-certified. Version 0.2 replaces the old `candle_tts` import;
existing voice-prompt JSON retains its schema and checkpoint identity.

CosyVoice3's official frontend additionally requires the **CPU ONNX Runtime
shared library, API 24 or newer**. Set `ORT_DYLIB_PATH` to its absolute path before
loading a CosyVoice3 model: `libonnxruntime.so` on Linux,
`libonnxruntime.dylib` on macOS, or `onnxruntime.dll` on Windows. Its adjacent
runtime dependencies must remain available. Provision the pinned official native
runtime from the repository root:

```sh
python scripts/install_onnxruntime.py --output .cache/onnxruntime
```

This repository development tool requires Python 3.12+ for safe archive extraction. This does not change the wheel's CPython 3.10+ requirement.

Then set the environment before constructing a CosyVoice3 engine:

```python
import json
import os
from pathlib import Path

os.environ["ORT_DYLIB_PATH"] = json.loads(
    Path(".cache/onnxruntime/runtime.json").read_text()
)["library"]
```

An `onnxruntime` Python wheel may also contain a compatible native library, but
its internal layout varies by release and platform; do not assume a fixed path.
Inference itself does not call the Python implementation. The import smoke test does not
verify this external runtime or execute a model. Candle's CUDA/Metal features
accelerate the neural synthesis backend; the ONNX frontend currently stays on CPU.

Prompt JSON is limited to 16 MiB and options JSON to 64 KiB before decoding.

CPython 3.10+ uses the stable `abi3` ABI; free-threaded CPython wheels are not
included in this distribution. Model support and numerical validation status are
documented in the repository root.

## Build from source

```sh
python -m pip install 'maturin>=1.12,<2'
python -m maturin develop --release --manifest-path bindings/python/Cargo.toml
python -m unittest discover -s bindings/python/tests -p '*.py'
```
