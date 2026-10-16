# SpeechFoundry for Node.js

Thin [NAPI-RS](https://napi.rs/) bindings. Loading, voice extraction, synthesis,
prompt validation, and resampling run through `AsyncTask` on libuv workers.
These methods return promises; model inference does not block the JavaScript
event loop. Calls on one engine are serialized to protect its inference state.

CosyVoice3 is experimental: full-Flow numerical parity remains unresolved, and
reported output artifacts include an unrelated leading sound and elongated
endings. See the [project validation status](https://github.com/SpenserCai/tts-rs#speechfoundry)
before using that model. The binding uses the same model implementation as Rust.

```sh
npm install @spensercai/speechfoundry
```

```js
import { loadEngine } from '@spensercai/speechfoundry'

const engine = await loadEngine('/models/Qwen3-TTS-12Hz-0.6B-Base', 'qwen3', 'cpu')
// referencePcm is mono Float32Array; provide its actual sample rate.
const voice = await engine.extractVoice(referencePcm, 24000, 'Reference transcript.')
const audio = await engine.synthesize('Hello.', voice, JSON.stringify({ seed: 42 }))
// audio.samples: Float32Array; audio.sampleRate: number
```

Voice conditioning is model-specific JSON with a versioned envelope. It is not
a universal voice identity or a newly trained model. Options use the shared
Rust schema, including validation and defaults.

Node.js 22+ is the supported baseline. `index.js` and `index.d.ts` are the public
facade; the build generates private `_native.js` and `_native.d.ts` files. Each
target gets its own native addon. The declared targets are
Linux x64 GNU, Windows x64 MSVC, and macOS arm64. Accelerator builds require
`--features cuda` or `--features metal` and matching native dependencies.

Prebuilt GPU previews install alongside this package from GitHub Release
tarballs. They must match the facade version; public imports never change:

```js
import { loadEngine, installedBackends } from '@spensercai/speechfoundry'

console.log(installedBackends()) // Installed builds, not a hardware probe.
const gpu = await loadEngine('models/qwen-base', 'qwen3', 'cuda:0')
// With several CUDA builds installed, pass { backend: 'cu124' } as argument four.
// Apple Silicon uses 'metal:0' and the matching Metal preview.
```

Missing or ambiguous backends and version mismatches are errors, never CPU
fallback. CPU imports do not load accelerator libraries. Consult the
[accelerator guide](https://github.com/SpenserCai/tts-rs/blob/main/docs/accelerators.md)
for install/runtime details and the build-only verification boundary. Version 0.2
replaces `@spensercai/tts-rs`; existing voice-prompt JSON remains compatible with
identical model files.

Build and runtime validation are separate: consult the root support matrix
for the tested model/device combinations and Linux libc floor. Cross-platform
archives must include all corresponding `.node` files before npm publication.
This package is separate from the WebAssembly component binding.

CosyVoice3 additionally loads its official frontend using the **CPU ONNX Runtime
shared library, API 24 or newer**. Before loading the model, set `ORT_DYLIB_PATH`
to an absolute `libonnxruntime.so` (Linux), `libonnxruntime.dylib` (macOS), or
`onnxruntime.dll` (Windows) path, keeping its adjacent runtime dependencies
available. The addon does not bundle that runtime or need a Python interpreter.
For development, provision the pinned official archive from the repository root:

```sh
python scripts/install_onnxruntime.py --output .cache/onnxruntime
```

This repository development tool requires Python 3.12+ for safe archive extraction. Node inference itself does not require Python.

The installer writes the resolved library path to `.cache/onnxruntime/runtime.json`:

```js
process.env.ORT_DYLIB_PATH = JSON.parse(
  require('node:fs').readFileSync('.cache/onnxruntime/runtime.json', 'utf8')
).library
```

Set this before `loadEngine`; production deployment may provision the same
official archive directly without Python.
Its import smoke test does not verify model inference. Candle CUDA/Metal affects
the synthesis backend; the ONNX frontend currently runs on CPU.

Prompt JSON is limited to 16 MiB and options JSON to 64 KiB before decoding.

## Build from source

```sh
cd bindings/node
npm ci
npm run build -- --release
npm test
```
