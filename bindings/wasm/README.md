# tts-rs WebAssembly components

The default package provides mono PCM validation/resampling, voice-prompt
**envelope** validation, and synthesis-option validation. The optional `qwen`
feature adds Qwen3-TTS's real Candle ECAPA speaker encoder, loaded from bytes.
`capabilities()` reports the compiled features. Full speech synthesis and
CosyVoice3 inference are not supported in this Wasm package.

The release workflow builds the web package with `--features qwen`, so release
artifacts include `QwenSpeakerEncoder`. Model weights stay external and are never
bundled. Source builds without that feature keep the smaller utility-only API.

```sh
cd bindings/wasm
wasm-pack build --target nodejs --out-dir pkg
node --test tests/smoke.cjs
wasm-pack build --target web --out-dir pkg-web
```

```js
import init, { Audio, validateSynthesisOptions } from './pkg-web/tts_wasm.js'

await init()
const audio = new Audio(new Float32Array([0, 0.25, -0.25, 0]), 24000)
try {
  console.log(audio.sampleRate, audio.durationSeconds)
  console.log(JSON.parse(validateSynthesisOptions('{"seed":42}')))
} finally {
  audio.free()
}
```

Typed arrays are copied across the Wasm boundary. Run expensive processing in a
Web Worker to keep the browser responsive. There is no filesystem/network model
loader and no CUDA/Metal/WebGPU backend. Prompt JSON is limited to 16 MiB and
options JSON to 64 KiB before decoding.

## Qwen speaker encoder

```sh
wasm-pack build --target nodejs --out-dir pkg-qwen --features qwen
node tests/qwen.cjs /path/to/speaker-fixture
```

```js
const { QwenSpeakerEncoder } = require('./pkg-qwen/tts_wasm.js')
// Supply the official speaker_encoder_config object and a speaker-only
// safetensors checkpoint whose speaker_encoder. key prefix has been removed.
const encoder = new QwenSpeakerEncoder(configJson, weightsUint8Array)
try {
  const embedding = encoder.extractEmbedding(monoPcmFloat32Array, 24000)
  // The output is the model's unnormalized speaker vector, not universal identity.
} finally {
  encoder.free()
}
```

From the repository root, run
`python scripts/reference/export_qwen.py --mode pretrained --output /path/to/speaker-fixture`
in the pinned official Python environment to generate the compatible
`config.json`, `speaker.safetensors`, `input.json`, and `reference.json` files. The Wasm probe compares the identical Python log-mel
intermediate through `forwardMel` and separately executes the PCM frontend.
Call `Audio.resample(24000)` explicitly for recordings at other sample rates.
Reference audio is limited to 30 seconds; configuration JSON is limited to 64 KiB.

## Full model memory boundary

The official Qwen3-TTS 0.6B Base checkpoint and speech codec contain
**4,340,801,796 bytes** of tensors when expanded to float32, already greater than
Wasm32's 4 GiB address space (**4,294,967,296 bytes**). Cache, activations, and byte
transfer copies need additional memory. The current CPU loader keeps those
components together, so exposing it here would produce allocation failures.

The speaker-only encoder fits and is exercised against real pretrained weights.
Full Wasm speech generation needs a separate memory design: validated quantized
operators, staged component loading, or a supported Wasm64 runtime. A compiled
model crate alone is not evidence that full inference is viable in a browser.

Recompute a checkpoint's tensor storage without reading its data payload:

```sh
python scripts/model_memory.py /models/qwen/model.safetensors \
  /models/qwen/speech_tokenizer/model.safetensors --output memory.json
```
