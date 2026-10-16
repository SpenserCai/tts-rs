# Architecture

SpeechFoundry is a local speech inference library with independent model and
task boundaries. Synthesis is implemented; ASR/STT is a future extension, not a
capability advertised by this release.

The synthesis boundary is a local model session: load a checkpoint, extract or import
a compatible voice prompt, synthesize PCM, and release the session. Model math
stays outside language bindings. Model downloads are explicit tooling operations.

```mermaid
flowchart TD
    R["Rust application"] --> E["speechfoundry Engine"]
    P["Python / PyO3"] --> E
    N["Node / napi-rs workers"] --> E
    E --> Q["speechfoundry-qwen"]
    E --> C["speechfoundry-cosyvoice"]
    Q --> T["Candle tensors"]
    C --> T
    C --> O["ONNX Runtime: reference frontend"]
    E --> A["speechfoundry-core: audio and contracts"]
    W["wasm32 / Worker"] --> A
    W --> QS["Optional Qwen speaker encoder"]
    QS --> T
    W64["wasm64 / Worker"] --> L["Streamed checkpoint loader"]
    L --> E
```

The browser package reports its actual capabilities. A native feature named
`cuda` or `metal` does not describe browser GPU execution. Large model buffers,
WASM's address space and copying across JavaScript boundaries require their own
validation; successful target compilation alone is not a browser performance test.
The wasm32 package exposes audio utilities and optional speaker extraction from
explicit configuration and weight bytes. The memory64 package uses the same
binding layer and wraps a complete in-memory Qwen model in `Engine`. It imports
checkpoint tensors individually, avoiding a contiguous multi-gigabyte JavaScript
buffer, and computes the same content fingerprint through a streaming hasher.
Memory64 must be supported by the host engine; it does not supply browser GPU
acceleration. CosyVoice's native ONNX frontend is not part of this WebAssembly
boundary. Consult the binding's capability report for the loaded build.

## Model-specific conditioning

```mermaid
flowchart TD
    PCM["Reference PCM + optional transcript"] --> QE["Qwen: ECAPA + 12 Hz encoder"]
    PCM --> CE["CosyVoice: CAMPPlus + token/mel frontend"]
    QE --> QP["Speaker vector + optional ICL codes/text"]
    CE --> CP["Speaker vector + tokens + mel + text"]
    QP --> V["Versioned VoicePrompt envelope"]
    CP --> V
    V --> G["Checkpoint and payload validation"]
    G --> S["Model-specific synthesis"]
```

The envelope has a schema version, model family, payload format and SHA-256
checkpoint content fingerprint. A model adapter owns shape and value validation.
Fingerprinting reads checkpoint contents once on load; moving identical files
does not invalidate prompts. Changing model files does. It is a compatibility
check, not a signature or an assertion of ownership of a voice.

## Shared code

Share finite PCM validation, file I/O, resampling, binding ownership and error
handling. Keep model mel parameters, windowing, normalization, codec layouts,
attention masks and sampling policies separate unless their equations match.
Resampling uses Rubato and explicitly compensates delay and flushes the tail.
For exact Python comparisons, feed identical PCM already at the native rate.

Qwen's sequence is text prefill → Talker code → acoustic code predictor → codec
decoder. CosyVoice's sequence is speech-token LM → conditional flow → HiFT. Their
common use of transformers does not make their caches or conditioning equivalent.

## Ownership and execution

Python and Node expose stable facade imports. The default native binary provides
CPU execution and portable utilities; accelerator previews register separate
private binaries. Python discovers the `speechfoundry.backends` distribution entry
points; Node resolves backend package manifests for the host platform. Neither
discovery path imports a GPU binary or probes hardware. Distribution versions,
compiled workspace versions and binding protocol 1 must agree before model loading.

```mermaid
flowchart TD
    A["Public Python / Node API"] --> S["Explicit device and build selection"]
    S --> C["Bundled CPU binary"]
    S --> M["Installed backend metadata"]
    M --> V["Version and protocol validation"]
    V --> G["One selected CUDA / Metal binary"]
    C --> E["Same Rust engine and model code"]
    G --> E
```

One matching installed backend is selected automatically; multiple CUDA builds
require an explicit build identifier. A process reserves its first CUDA build
even after a failed import, because native libraries can load partially; another
CUDA build requires a fresh process. CPU remains independently usable. Source
builds can use accelerators compiled into their main binary without an extra
preview package. These choices do not establish hardware execution or model parity.

The Python facade owns the native engine and normalizes returned PCM into its
public `Audio` type; it never casts objects between extension modules. Node owns
the native engine behind a single public wrapper and returns owned typed arrays.
No model math is implemented in either loader or wrapper.

`Engine` owns a model and its device. Mutable synthesis access prevents concurrent
cache mutation. Python releases its GIL around model work; Node uses worker tasks
and serializes calls on one engine. The application owns scheduling, cancellation
policy and audio playback. No host application state or agent runtime is embedded.

Raw samples and prompt data cross bindings as owned values. Checkpoints stay local.
Unsupported model features and missing ONNX operators are errors, not fallback
audio, fabricated embeddings or an implicit switch to another model.

CosyVoice's released CAMPPlus and speech tokenizer graphs execute through the
optional native ONNX Runtime frontend, matching the upstream deployment boundary.
The language model, flow and vocoder use Candle. The runtime is an explicit shared
library dependency, not bundled implicitly into Python wheels or Node packages.
Within HiFT, the small F0 predictor always runs on CPU in float64 to preserve the
upstream inference precision, including when the other stages use CUDA or Metal.
Its result returns to the selected model device. Account for this transfer when
measuring accelerator performance.

## Extension rule

Add a model crate with a documented checkpoint format and executable Python
oracle. Integrate its typed payload through an adapter. Add a binding only when
it can call the same tested Rust API; do not fork numerical logic into a language
wrapper. Add a shared acoustic primitive after two real callers demonstrate
identical required behavior.

A numerical primitive that needs different semantics from Candle belongs beside
the model that requires it. Keep its input shape, dtype, device and accumulation
rules explicit, and test the primitive against the pinned upstream operation
before checking the composed model. Prefer existing tensor operations; a local
kernel is justified by a measured semantic or numerical gap, not by the name of
an upstream operator. Such extensions do not require a Candle fork or a second
inference backend. CPU-only preprocessing remains at the frontend boundary;
model operations must not silently move accelerator tensors to the host.
