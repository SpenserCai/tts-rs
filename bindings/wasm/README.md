# Candle TTS for WebAssembly

Two explicit WebAssembly targets share the Rust model implementation and portable
voice-prompt format. Check `JSON.parse(capabilities())` for the loaded package.

| npm package | Target | Inference |
|---|---|---|
| `@spensercai/tts-wasm` | wasm32 | Audio utilities and Qwen speaker extraction |
| `@spensercai/tts-wasm64` | memory64 | Complete Qwen Base voice extraction and speech synthesis |

The memory64 package runs Qwen's speaker encoder, optional reference audio codec,
text generation, acoustic prediction, and waveform decoder through Candle CPU.
It supports speaker-only and transcript-conditioned ICL prompts. CosyVoice and
CUDA/Metal/WebGPU are not exposed by these WebAssembly packages. Model files stay
external; neither package downloads weights or includes them in its archive.

## Memory64 speech synthesis

Use a memory64-capable JavaScript runtime with sufficient physical memory. Node
24 is the validated host. A browser must independently support memory64 and the
required allocation; a larger address space does not increase device RAM. Run
loading and inference in a dedicated Worker. The original Qwen 0.6B Base tensors
occupy 4,340,801,796 bytes after CPU float32 expansion, before caches, activations
and transfer buffers. Wasm32 cannot address this complete pipeline.

```js
// Inside a module Worker. `files` maps original relative checkpoint filenames
// to File/Blob objects supplied by the application (for example a directory picker).
import init, * as bindings from '@spensercai/tts-wasm64'
import { loadQwenModel } from '@spensercai/tts-wasm64/loader.mjs'

await init()
const model = await loadQwenModel(bindings, files)
try {
  const promptJson = model.extractVoice(referencePcm, 24000, referenceTranscript)
  // Omit referenceTranscript for speaker-only conditioning.
  const audio = model.synthesize('Hello, welcome.', promptJson, JSON.stringify({
    language: 'english', seed: 42, max_tokens: 256,
  }))
  try {
    const samples = audio.samples
    self.postMessage({ samples, sampleRate: audio.sampleRate }, [samples.buffer])
  } finally { audio.free() }
} finally { model.free() }
```

In Node, initialize the same published web module from explicit Wasm bytes:

```js
import { readFile } from 'node:fs/promises'
await init({ module_or_path: await readFile(new URL(
  import.meta.resolve('@spensercai/tts-wasm64/tts_wasm_bg.wasm'),
)) })
```

Node's `openAsBlob` can supply local model files to the same `loadQwenModel` API;
no alternate model implementation is used.

Supply the original files: `config.json`, `model.safetensors`,
`speech_tokenizer/model.safetensors`, and either `tokenizer.json` or the official
`vocab.json`/`merges.txt` pair. Include `tokenizer_config.json` and
`speech_tokenizer/config.json` when present. The loader hashes the exact original
file set with the same SHA-256 identity protocol as native Rust/Python/Node, so a
persisted prompt from that same checkpoint can be used directly with `synthesize`.
Adding or changing tokenizer files changes checkpoint identity.

`loadQwenModel` reads bounded file slices: at most 8 MiB for hashing and one tensor
for loading (at most 2 GiB per tensor). It never constructs a whole-checkpoint
ArrayBuffer. Tensors immediately become float32 so assembling the model shares
existing tensor storage. Configurations, shapes, offsets, byte counts, duplicate
tensors and prompt compatibility are validated. Missing components produce an
error rather than silently reducing the model's capabilities.

The low-level `QwenModelBuilder` and `CheckpointHasher` exports support other
explicit-byte hosts. Keep their input contents consistent; the high-level loader
coordinates both. Calls are synchronous CPU work, except for the JavaScript
loader's Blob reads. Release sessions with `free()` when no longer needed.

## Wasm32 audio and speaker components

```js
import init, { Audio, QwenSpeakerEncoder } from '@spensercai/tts-wasm'
await init()
const audio = new Audio(referencePcm, 24000)
const encoder = new QwenSpeakerEncoder(speakerConfigJson, speakerWeights)
try {
  const embedding = encoder.extractEmbedding(audio.samples, audio.sampleRate)
  // Qwen's unnormalized 1024-dimensional vector; not a universal speaker identity.
} finally { encoder.free(); audio.free() }
```

The speaker-only safetensors buffer uses keys with `speaker_encoder.` removed;
`speakerConfigJson` is the checkpoint's official `speaker_encoder_config` object.
Call `Audio.resample(24000)` for other recording rates. Reference recordings are
limited to 30 seconds. `forwardMel` accepts a shared reference mel tensor for
numerical comparison. Default source builds without `qwen` contain audio,
prompt-envelope and synthesis-options validation only.

## Reproducible builds

All distributions use the root Cargo workspace version. The standard wasm32
package uses the pinned stable toolchain and `wasm-pack`:

```sh
wasm-pack build bindings/wasm --target web --out-dir pkg-web --no-opt --features qwen --locked
```

Rust's memory64 target is tier 3 and does not ship a prebuilt standard library.
The memory64 build therefore pins a separate nightly toolchain and rebuilds std;
it uses the existing wasm-bindgen ABI rather than a custom pointer bridge.

```sh
rustup toolchain install nightly-2026-10-13 --profile minimal --component rust-src --component clippy --component rustfmt
python scripts/wasm64.py prepare
python scripts/wasm64.py build
python scripts/wasm64.py check
python scripts/wasm64.py package
```

`prepare` installs the Cargo.lock-matched wasm-bindgen CLI; on Linux x64 it verifies
the official binary's SHA-256. `build` produces web (`pkg-memory64`) and Node test
(`pkg-memory64-node`) modules from the same Wasm binary, then executes smoke tests.
`check` runs strict target Clippy/rustdoc. A low-resident-memory test reserves pages
past 4 GiB, creates a Rust audio object above that address and verifies actual
JavaScript typed-array round trips. `package` packs the already tested web output.
None of these commands publishes a registry package.

An optional real-model test accepts a JSON request containing `samples`,
`sample_rate`, `text`, and shared `options` (plus optional `transcript`):

```sh
python scripts/reference/offline_exec.py -- node bindings/wasm/tests/model-memory64.mjs \
  /models/qwen /fixtures/request.json /results/memory64.json
```

The output records full-model timing, waveform samples and checkpoint identity.
Use a prepared model and the same finite PCM/options as the native reference.
Downloads and compilation happen before the Linux offline execution boundary.

The complete web package was exercised with Qwen 0.6B Base on Node 24.19.0,
Linux x64: an analytic one-second conditioning signal, `Hello.`, greedy decoding
and a 20-token budget produced 36,480 finite samples. The waveform passed the
unchanged `atol=1e-5, rtol=1e-4` gate against both official Python and native Rust
(maximum errors `5.271e-9` and `5.180e-9`). Peak process RSS was 6,248,228 KiB.
This bounded speaker-only numerical probe does not establish natural EOS,
perceptual quality, browser execution or ICL parity in memory64. See the
[scoped validation record](../../tests/fixtures/qwen-memory64-validation.json).

The target and ABI references are the [Rust memory64 target documentation](https://doc.rust-lang.org/rustc/platform-support/wasm64-unknown-unknown.html)
and [wasm-bindgen's memory64 implementation](https://github.com/wasm-bindgen/wasm-bindgen/pull/5764).
