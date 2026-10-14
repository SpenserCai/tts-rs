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

```sh
python scripts/reference/run.py --suite parity --model qwen --device cpu \
  --prepare-env --fixture-kind synthetic --output artifacts/qwen-small

python scripts/reference/run.py --suite parity --model cosyvoice --device cpu \
  --prepare-env --fixture-kind synthetic --output artifacts/cosy-small
```

The Qwen synthetic check runs the complete speaker encoder with small deterministic
weights. The CosyVoice synthetic check covers DiT conditioning primitives; it is
not a full-model test. Pretrained Qwen speaker testing can fetch only the speaker
tensor ranges, avoiding a full speech model download. See `--help` on
`download_speaker.py` and `export_qwen.py` for explicit paths and revisions.

For full-model checks, provide matching weights, tokenization, reference PCM,
precision, and decoding settings. Use greedy decoding or exported shared random
tensors. Equal seeds in Rust and PyTorch do not imply identical draws. CosyVoice
also requires shared flow noise and vocoder noise for waveform comparisons.

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

Compare intermediates in order: frontend features → speaker/conditioning → LM
logits or fixed codes → acoustic decoder → waveform. This distinguishes a token
boundary error from a numeric kernel difference. A waveform test from one short
utterance does not certify every language, model size, sampling mode or stream.

## Benchmarking

```sh
cargo bench -p tts-core --bench audio
python scripts/reference/run.py --suite benchmark --model qwen --device cpu \
  --prepare-env --output artifacts/speaker-benchmark
```

Criterion measures audio operations, not TTS real-time factor. Model evaluation
separates process/load overhead from synthesis where supported, records machine,
runtime, source revision and device, and uses warm-up plus repeated runs. GPU
timing requires synchronization. Report peak memory and output duration alongside
latency/RTF; do not interpret shared CI runner timing as a stable performance gate.

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
