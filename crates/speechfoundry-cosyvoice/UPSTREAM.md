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

The frontend uses RustFFT (the same dependency as the Qwen model crate) with
float32 buffers and one reused plan/scratch buffer per call. Whisper's centered
STFT uses reflection, and Matcha/Whisper periodic Hann windows use float32
operation order. Librosa mel normalization retains its float64 scale until the
float32 filter assignment. These are model-local preprocessing operations; they
do not change Candle or impose a shared mel definition on unrelated models.

CAMPPlus sessions disable only the ONNX Runtime `Pad_Fusion` rewrite. In runtime
1.24.4, it removes the graph's zero-width `Pad` and changes subsequent
`AveragePool(ceil_mode=1)` from `count_include_pad=0` to `1`, changing the divisor
for incomplete final windows. This is a semantic error, not float32 rounding.
The remaining graph optimizations and speech tokenizer settings stay enabled.
See the authored minimal graph in `tests/fixtures/explicit_padding_average.onnx`:
the correct mean of `[1,2,3]` is `2.0`; the faulty rewrite returns `1.5`.
The upstream rule name is defined in
[`pad_fusion.h` at v1.24.4](https://github.com/microsoft/onnxruntime/blob/v1.24.4/onnxruntime/core/optimizer/pad_fusion.h).

The model-local CPU float32 LayerNorm in `src/flow/normalization.rs` adapts the
Welford/cascade reduction and normalization order from PyTorch 2.3.1, commit
[`d44533f9d073df13895333e70b66f81c513c1889`](https://github.com/pytorch/pytorch/tree/d44533f9d073df13895333e70b66f81c513c1889):
`aten/src/ATen/native/cpu/moments_utils.h` and
`aten/src/ATen/native/cpu/layer_norm_kernel.cpp`.
It preserves the float32 accumulation and fused multiply-add stages without
depending on libtorch or modifying Candle. The complete upstream copyright
notices and BSD conditions are retained in [LICENSE-PYTORCH](LICENSE-PYTORCH)
and included in the native language distributions.

Mish follows the same pinned PyTorch source's `x * tanh(log1p(exp(x)))`
definition in `aten/src/ATen/native/cpu/Activation.cpp`. The local CPU kernel
uses `ln_1p` to preserve the negative tail; large positive inputs return the
already-rounded identity without overflowing `exp`. Accelerator tensors retain
their device and use a stable softplus identity with a bounded negative-tail
series. This is a numerical formulation of Mish, not a change to model weights
or a claim of accelerator parity.

The float32 Euler grid follows the same pinned PyTorch CPU source's
`aten/src/ATen/native/cpu/RangeFactoriesKernel.cpp` linspace operation: a
float32 step, symmetric evaluation from both endpoints, and fused multiply-add
rounding. Independent `torch.linspace` bit-pattern fixtures cover non-default
step counts as well as the unchanged ten-step default. Other precision modes
retain their existing float32-staged grid; these fixtures do not claim
half-precision or accelerator parity. The PyTorch attribution and license above
also apply to this local grid construction.

## Validation boundary

HiFT source phase uses model-local CPU float32 primitives in `src/vocoder/phase.rs`.
They preserve true division (rather than multiplication by a rounded reciprocal)
and the pinned PyTorch CPU cumsum's float64 accumulator with float32 output at
each step. The reviewed PyTorch files are `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp`,
`ReduceOpsKernel.cpp`, and `aten/src/ATen/AccumulateType.h` at the revision above;
the existing PyTorch attribution and license also cover this scan adaptation.
This avoids Candle 0.9.2's scalar reciprocal multiplication and triangular-matmul
cumsum without changing Candle. Other dtypes/devices retain their existing paths,
without hidden device transfers. `scripts/reference/export_cosyvoice_phase.py`
regenerates the small authored bit-pattern fixtures from the pinned runtime.

CosyVoice's top-p cutoff and RAS rejection policy retain their upstream semantics.
Categorical selection accumulates float32 weights in float64 and scales the
uniform draw by their actual mass. A rounding fallback can only select a positive
weight, never a masked vocabulary-tail stop token. This corrects the former
float32 inverse-CDF implementation; it does not reproduce `torch.multinomial`'s
RNG algorithm. Fixed seeds can therefore produce different utterances across
versions or frameworks even with equivalent probability distributions.

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
