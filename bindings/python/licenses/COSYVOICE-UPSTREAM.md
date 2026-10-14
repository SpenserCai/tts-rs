# Source provenance

The initial tensor implementations were adapted from the CosyVoice module in
[SpenserCai/candle](https://github.com/SpenserCai/candle/tree/6058747b7a5c3f62a03e74d6f052285477f8430c/candle-transformers/src/models/cosyvoice),
commit `6058747b7a5c3f62a03e74d6f052285477f8430c`, under Candle's MIT OR Apache-2.0
license. The Python bindings from `cosyvoice3.rs` are not part of this crate.

The architecture and checkpoint defaults target
`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`. The reviewed Python implementation is
[QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice/tree/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc),
commit `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`:

- `examples/libritts/cosyvoice3/conf/cosyvoice3.yaml`: released architecture.
- `cosyvoice/cli/frontend.py`: reference conditioning and alignment.
- `cosyvoice/llm/llm.py`: Qwen2 and CosyVoice3LM inference.
- `cosyvoice/flow/flow_matching.py`: cosine Euler schedule and guidance.
- `cosyvoice/flow/DiT/`: acoustic transformer.
- `cosyvoice/hifigan/generator.py`: causal source and waveform synthesis.
- `cosyvoice/tokenizer/tokenizer.py`: required special-token registration.

Changes include standalone modules and Rust APIs, corrected guidance and HiFT
defaults, full-tokenizer validation, conditioning validation, and explicit
shared-noise import. An identical numeric seed in different RNG implementations
does not imply identical noise. Model weights are not distributed with this crate.
The F0 predictor uses float64 on CPU, including with an accelerated main model.
Its normalized weights must be materialized from float64 gain/direction tensors;
promoting weights already folded in float32 loses measurable phase precision.
The converter uses the same weight-normalization primitive as the pinned Python
reference. Linear source interpolation and reflective STFT padding follow the
official operations without substituting average pooling or zero padding.

RoPE evaluates both the inverse-frequency power and reciprocal in float32, as
the pinned implementation does. DiT applies the input padding mask in offline
and chunked attention and clears masked query outputs after projection.

The audio frontend follows the pinned torchaudio 2.3.1 runtime: resampling
uses float64-designed, float32-cached sinc kernels and a ceil output length;
Kaldi Povey windows and mel-bank tensor operations use float32 staging.
Backend logarithm, FFT and reduction kernels can still differ in rounding.

## Validation boundary

Component tests and compilation do not establish pretrained synthesis parity.
Use the Python reference suite with identical weights, token IDs, precision, flow
noise and vocoder noise. Compare intermediates before comparing waveforms. The
native ONNX Runtime adapter uses the official auxiliary graphs unchanged. The
`onnx` feature pins `ort` 2.0.0-rc.14 and dynamically loads ONNX Runtime >= 1.24
from `ORT_DYLIB_PATH` (or an explicit path). This avoids the standard Candle ONNX
evaluator's unsupported integer input declarations and constant-padding graphs.
The native runtime is optional and is not part of the WebAssembly build.
The current public synthesis API is offline and batch one. Streaming state is
not exposed as a supported end-to-end mode.
