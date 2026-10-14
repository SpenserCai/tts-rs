# Reproducible validation

Treat build checks, component parity, pretrained end-to-end parity, binding
installation and perceptual quality as separate claims. A successful import or
finite waveform is useful evidence but does not establish numerical equivalence.

## Ordinary changes

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

CosyVoice flow does not pass the current acceptance gate: the official-runtime
default comparison failed on 124 of 1,280 mel values (`max_abs=1.201e-3`). The
official implementation itself also fails that strict gate when switching only
its CPU attention backend from SDPA to MATH. Both failures are retained rather
than widening the tolerance from one example.

A separate coupled check feeds each implementation's own flow mel into HiFT with
identical serialized source noise. On this 7,680-sample public fixture, Candle
versus official SDPA measured waveform RMSE `1.624e-4` and SNR `53.97 dB`; official
SDPA versus MATH measured RMSE `1.557e-3` and SNR `34.33 dB`. This bounds one
observed downstream effect, not perceptual quality or universal equivalence.
The coupled waveform still fails strict elementwise parity.

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

## Benchmarking

```sh
cargo bench -p tts-core --bench audio
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
model and codec, before caches and activations. Full-model browser synthesis
therefore needs a memory-saving implementation such as quantization; successful
speaker extraction does not imply complete TTS fits in wasm32.

Voice recordings and extracted personal prompts stay outside this public
repository. CI fixtures use synthetic signals or explicitly documented public
inputs. Measured verification summaries omit personal audio, embeddings and paths.
