# Reproducible validation

Treat build checks, component parity, pretrained end-to-end parity, binding
installation and perceptual quality as separate claims. A successful import or
finite waveform is useful evidence but does not establish numerical equivalence.

## Ordinary changes

Prepare the pinned external frontend runtime with
`python scripts/install_onnxruntime.py` (Python 3.12+) and set `ORT_DYLIB_PATH` to
the reported library. The default workspace includes the native facade and its
ONNX frontend feature, so the small pooling regression requires that runtime.
The runtime installer verifies the official archive digest; it does not download
model weights. CI performs this preparation explicitly before the Rust gate.

```sh
python scripts/ci.py rust
python scripts/ci.py bindings --kind python
python scripts/ci.py bindings --kind node
python scripts/ci.py bindings --kind wasm
```

The Rust gate checks formatting, Clippy with warnings denied, tests and rustdoc.
It also checks the facade without a model and with each model independently.
Native binding checks build real packages, install/load them and run smoke tests.
The normal test suite does not download weights. Use the manual full CI profile
for Windows x64 and macOS ARM; Linux success does not certify those platforms.

## Official Python oracle

`scripts/reference/pins.json` is the source of truth for official Python and
checkpoint revisions. `prepare_env.py` checks out those revisions and rejects a
dirty reference tree. It pins the core runtime dependencies and records the full
resolved environment. It does not pretend that all transitive packages are locked.
Each model family has its own virtual environment: Qwen uses Torch 2.10, while
CosyVoice uses the upstream Torch/torchaudio 2.3.1 and Transformers 4.51.3
contract. In particular, changing Transformers can change cached attention-mask
behavior without changing any checkpoint tensor. A different runtime is a new
oracle configuration and must be recorded and validated separately.

On Linux, `scripts/reference/offline_exec.py -- COMMAND ...` runs an already
prepared command with a libseccomp policy that denies socket creation and network
transmission across child processes. It fails closed if that policy cannot be
installed. Download dependencies and checkpoints beforehand; environment flags
such as `HF_HUB_OFFLINE` alone do not enforce this boundary.
Use `run.py --offline-reference` to apply that policy to both model processes
while leaving downloads, environment preparation and Cargo compilation outside.
Cargo needs local IPC; wrapping the entire build pipeline is not equivalent to
isolating model execution. Linux model-evaluation workflows select this flag.

```sh
python scripts/reference/run.py --suite parity --model qwen --device cpu \
  --prepare-env --fixture-kind synthetic --output artifacts/qwen-small

python scripts/reference/run.py --suite parity --model cosyvoice --device cpu \
  --prepare-env --fixture-kind synthetic --output artifacts/cosy-small

python scripts/reference/run.py --suite e2e --model qwen --device cpu \
  --prepare-env --output target/qwen-e2e

python scripts/reference/run.py --suite parity --model cosyvoice --device cpu \
  --prepare-env --fixture-kind pretrained --component hift --output target/cosy-hift
```

The Qwen synthetic check runs the complete speaker encoder with small deterministic
weights. The CosyVoice synthetic check covers DiT conditioning primitives and
the official tanh-GELU feed-forward network; it is
not a full-model test. Pretrained Qwen speaker testing can fetch only the speaker
tensor ranges, avoiding a full speech model download. See `--help` on
`download_speaker.py` and `export_qwen.py` for explicit paths and revisions.

For full-model checks, provide matching weights, tokenization, reference PCM,
precision, and decoding settings. Use greedy decoding or exported shared random
tensors. Equal seeds in Rust and PyTorch do not imply identical draws. CosyVoice
also requires shared flow noise and vocoder noise for waveform comparisons.

Automatic Qwen E2E runs two public numerical smoke cases in sequence: x-vector
conditioning from an analytic tone, then ICL conditioned on that case's generated
audio and known text. The ICL gate also compares every reference codec ID. A
20-token cap bounds runtime; it is not an assertion of natural EOS or voice quality.

Pretrained CosyVoice CPU stages are selected with `--component hift`, `flow`,
`llm` or `all` in both `run.py` and the manual model-evaluation workflow. The
default input is analytic, with no recording. Reuse reviewed public features
with `--prompt-features`; reuse checkpoints with `--model-directory` and
`--converted-model-directory`, and a checkout with `--cosyvoice-source`.
Each stage executes official Python and Candle; shared randomness is serialized,
not inferred from seed equality. Failed stages remain failures. Full chained
CosyVoice E2E requires a reviewed manifest and separate evidence.

The [observed validation baseline](../tests/fixtures/validation-baseline.json)
contains public scalar metrics, pins and exact scopes. CPU Qwen speaker and full
bounded x-vector/ICL outputs passed `atol=1e-5, rtol=1e-4`; both ICL reference and
generated code sequences matched exactly (304 IDs each). Under CosyVoice's pinned
runtime, LM first logits passed and all 60 bounded greedy speech tokens matched;
the same runtime's HiFT waveform passed on 15,360 samples, with maximum absolute
error `2.504e-6`. These
observations do not certify every intermediate tensor, input, language or device.

The [frontend regression record](../tests/fixtures/cosyvoice-frontend-validation.json)
uses 2,401 samples of public deterministic PCM, including both STFT boundaries
and a fractional resampling tail. Resampled PCM, Kaldi Fbank, mean-normalized
Fbank, Matcha mel, Whisper mel and the 192-value CAMPPlus embedding all pass the
same strict threshold. The embedding maximum absolute error is `1.812e-5`.
Small official feature tensors are checked into the model crate for ordinary
tests; pretrained CAMPPlus weights remain external.

This comparison exposed a semantic ONNX Runtime 1.24.4 optimizer defect:
`Pad_Fusion` changes the divisor in CAMPPlus's final partial average-pooling
windows. The production CAMPPlus session disables only that rewrite. An authored
small ONNX graph reproduces the defect without weights and exercises the same
session configuration in the default workspace gate. RustFFT, float32 Hann
windows and reflection padding also align the frontend with the pinned oracle.
See the model's [source provenance](../crates/speechfoundry-cosyvoice/UPSTREAM.md) for the
affected operations. These results do not establish full synthesis parity.

CosyVoice's complete native Flow path is implemented, but **strict full-Flow
parity remains unresolved**. The latest
[revalidation](../tests/fixtures/cosyvoice-flow-revalidation.json) compares four
slices of public analytic conditioning, with identical exported noise and
unchanged `atol=1e-5, rtol=1e-4`:

| Prompt + target tokens | Failing mel values before | Failing mel values after |
| --- | --- | --- |
| 8 + 8 | 18 / 1,280 | 4 / 1,280 |
| 4 + 4 | 2 / 640 | 2 / 640 |
| 5 + 11 | 37 / 1,760 | 56 / 1,760 |
| 1 + 1 | 0 / 160 | 0 / 160 |

The retained LayerNorm and Mish changes have independent operator evidence:
Welford moments preserve small variance at large offsets, and `log1p` preserves
Mish's negative tail. They do not guarantee smaller complete-model error; the
asymmetric case above worsens. Both extensions stay in the model crate without
a Candle fork, whole-model float64 promotion or accelerator-to-CPU fallback.

The float32 Euler time grid also follows the pinned CPU `torch.linspace`
rounding contract. Independent bit-pattern fixtures cover 1, 3, 7, 10 and 50
steps; the previous one-sided float64 construction differed at non-default
counts. The ten-step default is bit-for-bit unchanged, so this grid correction
does not change the full-Flow results above. Low-precision grid behavior is
preserved without claiming new half-precision parity.

Frozen block diagnostics show that Q/K/V and adaptive projections individually
pass the strict threshold, while their small differences amplify through the
checkpoint's large attention logits. The complete block still fails. This
evidence supports a floating-point reduction sensitivity; it does not establish
a missing operator or justify a parity claim. A projection experiment that
reproduced the pinned MKL reduction on frozen inputs was removed because it
added substantial cost without stable complete-model gains.

A later isolated experiment used standard MKL 2022.2 SGEMM with the pinned
Python Linear's bias/flattening semantics. Although frozen conditioning and
Q/K/V projections matched bit-for-bit, the complete 8+8 case still failed:
12 / 1,280 values, maximum error `5.779e-4`, RMSE `8.724e-5`. Smaller maximum
error and RMSE did not imply strict success. That experimental backend is not
part of the library. Frozen attention checks likewise ruled out changing AV
normalization order as a sufficient repair. Keep these operator-level and
composed-model conclusions separate when evaluating another backend.

Use `export_cosyvoice_flow.py --layerwise` and `flow_oracle ... --layerwise` for
first-step layer traces. Use `--block-trace INDEX` with the exporter and
`flow_block_oracle` for independent frozen operator inputs; reproduction commands
are included in the revalidation record. Normal evaluation avoids tracing cost.

The earlier [flow diagnosis](../tests/fixtures/cosyvoice-flow-diagnostics.json)
remains historical evidence. Its inputs have different hashes, so its 51 / 1,280
failure count is not the before value for the table above. That record also
shows numerical variation between the official CPU SDPA and MATH kernels;
variation in the oracle does not turn a failed candidate comparison into a pass.

The current [coupled waveform check](../tests/fixtures/cosyvoice-coupled-waveform-validation.json)
feeds each implementation's own Flow mel into HiFT with identical serialized
source noise. Both current inputs fail the unchanged waveform threshold:

| Prompt + target tokens | Failing waveform values | RMSE | Error SNR |
| --- | --- | --- | --- |
| 8 + 8 | 6,195 / 7,680 | `6.845e-4` | 29.39 dB |
| 5 + 11 | 6,469 / 10,560 | `2.677e-4` | 34.71 dB |

The same-mel controls, which give Rust HiFT the official Flow output, both pass;
their maximum absolute errors are `1.188e-6` and `1.233e-6`. This isolates a
measurable downstream effect from the remaining Flow differences. It does not
establish listening quality or permit a full synthesis parity claim.

A separate [official attention control](../tests/fixtures/cosyvoice-attention-backend-control.json)
uses the same current inputs and noise, changing only the official Flow attention
backend from default to MATH before running official HiFT. Both resulting
waveforms also fail elementwise parity, with SNR 35.17 dB and 34.76 dB respectively.
The Rust result is worse on the first case and close on the second; this control
demonstrates sensitivity to arithmetic order, not equivalence or an exemption
from the candidate's failing gate. Earlier coupled metrics belong to different
input hashes and are retained only in the historical diagnosis.

## CosyVoice 0.1.1 numerical regression

The [0.1.1 regression record](../tests/fixtures/cosyvoice-0.1.1-regression.json)
separates two model-local corrections from the unresolved Flow measurements
above. All CPU comparisons retain `atol=1e-5, rtol=1e-4`.

- Categorical sampling now uses a float64 CDF and the actual weight sum. The
  zero-mass vocabulary-tail regression, masked weights, small tails, invalid
  distributions and half-open uniform boundaries run without model downloads.
  Top-p candidate selection and RAS rejection policy are unchanged. Neither
  seeded backward identity nor `torch.multinomial` RNG identity is promised.
- CPU float32 source division and phase scans match pinned PyTorch bit-pattern
  fixtures, including strided tensors and nonzero storage offsets. Regenerate
  these authored fixtures with `export_cosyvoice_phase.py --output PATH` inside
  the pinned CosyVoice environment and `offline_exec.py` boundary.
- A 210-frame fixed-mel control replays the exact same phase/sine/UV tensors.
  Official Python re-execution matches the earlier oracle bit for bit. Source
  error falls from `7.729e-5` to `7.451e-9` (zero failing source values). Waveform
  RMSE falls from `5.878e-4` to `1.224e-6`; error SNR rises from 47.45 to 101.08 dB.
  **Waveform parity still fails:** 38 / 100,800 values exceed the unchanged gate.
  The first and last 0.5 seconds pass, but this is not evidence that semantic
  leading sounds or elongated endings have been removed.

The longer control uses private conditioning and publishes only scalar metrics
and digests, never audio, voice features or token traces. Reproduction requires
the retained private inputs. Ordinary CI uses the separate authored operation
fixtures. Accelerator arithmetic and full-Flow semantics are unchanged; this
record does not claim CUDA/Metal parity or a perceptual quality score.

All four existing analytic Flow controls were rerun: their inputs, official
outputs and Rust outputs are bitwise unchanged from the previous baseline
(including its three strict failures). The short 15,360-sample HiFT gate and
DiT conditioning module comparisons still pass. Five predetermined complete
text-to-waveform requests returned finite audio without hitting token budgets;
this smoke test is not a listening-quality assessment. Linux native/binding
checks and WASM32/Memory64 checks pass; Qwen pretrained waveform and accelerator
model validation were not rerun for this model-local patch.

## Strict comparisons

```sh
python scripts/reference/compare_tensors.py --help
python scripts/reference/run.py --suite e2e --model qwen \
  --e2e-manifest /path/to/manifest.json --output artifacts/e2e
```

The manifest records source revision, hashes of inputs and checkpoint files,
commands for both implementations, output paths and the stochastic-input strategy.
Commands are argument arrays rather than shell strings. Stale output files are
rejected. Both implementations must produce a real, nonempty waveform tensor and
the same positive sample rate. Missing tensors, shape changes and non-finite
values fail the check. Integer code tensors must match exactly.

For floating tensors, a value passes when
`abs(candidate - reference) <= atol + rtol * abs(reference)`. Reports include
maximum and mean absolute error, RMSE, failing element counts and cosine
similarity. Defaults are `atol=1e-5`, `rtol=1e-4` for current CPU float32 fixtures;
document a mathematical reason and evidence before changing a tolerance.
Waveform reports also include reference RMS/peak and error SNR. SNR is `null`
when its reference or error energy is zero; `identical_waveform` distinguishes
the exact-match case. These diagnostics never override the strict pass result.

For coupled flow/vocoder diagnostics, `export_cosyvoice_stages.py --mel-input`
accepts a flow tensor document in place of `--prompt-features`. Export the
reference waveform and its `source-noise.safetensors`, then pass that same noise
to `hift_oracle` with the candidate flow document. Use `--source-noise` to replay
it in another Python backend run. Explicit `--frames` must match the tested mel
length. Compare the resulting waveform documents with `compare_tensors.py`;
retain both the strict failures and the RMS/SNR diagnostics.

Compare intermediates in order: frontend features → speaker/conditioning → LM
logits or fixed codes → acoustic decoder → waveform. This distinguishes a token
boundary error from a numeric kernel difference. A waveform test from one short
utterance does not certify every language, model size, sampling mode or stream.

## Separate speech-quality audits

The official [CV3-Eval protocol](https://github.com/QwenAudio/CV3-Eval/tree/68b7682a410d048308d1a12f2fbc266770cd23d5)
measures content consistency with CER/WER, speaker similarity with ERes2Net,
and audio quality with DNSMOS. Its specified ASR models are Paraformer for
Chinese and Whisper large-v3 for English. These measure different properties
from tensor parity; none provides a cross-backend floating-point tolerance or
turns a failed numerical comparison into a passing one.

The pinned official [`export_onnx.py`](https://github.com/QwenAudio/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/cosyvoice/bin/export_onnx.py)
does perform a local numerical check: ten randomly sized single-estimator
inputs use `atol=1e-4, rtol=1e-2`. That export check does not cover the complete
Euler solver, HiFT or final waveform. It is a separate scope and does not replace
this port's recorded full-Flow comparison contract.

For a separate paired audit, freeze the public sample manifest, reference
transcripts, text normalization, model/evaluator revisions, weight hashes and
decoding settings before generating either implementation's output. First
compare acoustic decoding with identical speech tokens and actual noise; then
evaluate the complete synthesis pipelines. Record per-sample scores for both
implementations, their paired differences and the sample counts. Missing output,
an evaluator exception or a missing score must fail the requested audit rather
than silently shrink its denominator.

Preserve the evaluator's primary aggregation; filtered or clipped error rates
are additional diagnostics. Speech-quality samples must reach a natural end of
utterance. A token-capped numerical fixture or a short analytic waveform is not
a quality sample. Greedy decoding is useful for port diagnosis but differs from
the official default sampling configuration and must be labeled accordingly.

Use an independent speaker evaluator rather than the conditioning encoder alone.
Do not call a short transcription smoke check the full official benchmark, or
treat one short utterance as statistical evidence of equal quality. Thresholds,
sample coverage and any non-inferiority margin need an explicit justification
before inspecting candidate scores. Quality evaluation supplements the existing
strict gate; it does not relax it. No complete CV3-Eval quality audit is claimed
by the numerical records above.

## Benchmarking

```sh
cargo bench -p speechfoundry-core --bench audio
python scripts/reference/run.py --suite benchmark --model qwen --device cpu \
  --prepare-env --output artifacts/speaker-benchmark

python scripts/reference/benchmark_tts.py --model-dir /path/to/qwen-base \
  --output target/tts-benchmark --warmups 1 --runs 3 --threads 1

python scripts/reference/run.py --suite benchmark --model qwen --device cpu \
  --benchmark-kind tts --prepare-env --runs 3 --output artifacts/tts-benchmark
```

Criterion measures audio operations, not TTS real-time factor. Model evaluation
separates process/load overhead from synthesis where supported, records machine,
runtime, source revision and device, and uses warm-up plus repeated runs. GPU
timing requires synchronization. Report peak memory and output duration alongside
latency/RTF; do not interpret shared CI runner timing as a stable performance gate.
The observed resident CPU probe used one thread, one warmup and two measured
runs on a shared AMD EPYC host. It generated 1.52 seconds per request, with a
median request RTF of 5.484 and a separately measured 5.935-second model load.
These are throughput observations, not real-time or perceptual-quality claims;
the exact workload and individual samples are retained in the baseline.

## Runtime boundaries

CPU is portable; CUDA and Metal require their native runtimes and dedicated
execution evidence. CosyVoice's reference frontend additionally requires ONNX
Runtime 1.24+ through `ORT_DYLIB_PATH`. Missing runtime libraries fail explicitly.

WASM is validated separately. Query its capability report. The official Qwen
0.6B checkpoint expands to more than 4 GiB of float32 parameters across the main
model and codec, before caches and activations. The wasm32 package provides audio
and speaker components; the separate memory64 package loads and executes the
full Qwen pipeline using explicit model bytes.

The [memory64 validation record](../tests/fixtures/qwen-memory64-validation.json)
documents execution of the published web module on Node 24.19.0/Linux x64. The
same analytic tone, text and greedy 20-token budget used by native x-vector E2E
produced 36,480 finite samples. Maximum waveform error was `5.271e-9` against
official Python and `5.180e-9` against native Rust; every value passed the existing
gate. Peak process RSS was 6,248,228 KiB. A separate test allocated a Rust object
above 4 GiB and round-tripped its samples through JavaScript. This evidence does
not certify browser engines, memory64 ICL, natural EOS, voice quality or other
checkpoints; the full-model fixture deliberately stops at a 20-token budget.

Voice recordings and extracted personal prompts stay outside this public
repository. CI fixtures use synthetic signals or explicitly documented public
inputs. Measured verification summaries omit personal audio, embeddings and paths.
