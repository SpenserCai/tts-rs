# Official reference evaluation

These scripts run the upstream Python implementation and compare its actual
outputs with Candle. Source/model revisions and top-level dependencies live in
[`pins.json`](pins.json). The prepared environment records transitive packages;
it is not a fully locked Python dependency graph.
Model downloads are explicit preparation steps. Reference inference requests
offline Hugging Face loading; this setting is not a network security boundary.
Tensor-only helpers do not initialize unrelated runtimes. Where verified offline
execution is required, enforce it outside Python rather than relying on telemetry
preferences. Upstream CosyVoice imports ONNX Runtime even in its LM module.

Preparation creates separate `qwen-venv` and `cosyvoice-venv` environments. The
CosyVoice oracle pins its upstream Torch/torchaudio 2.3.1 and Transformers 4.51.3
runtime rather than inheriting Qwen's newer dependencies. Optional serving and
training stacks are not installed; selected upstream inference pins and the
resolved environment are recorded.
Pretrained CosyVoice exporters reject mismatched core dependency versions and
checkpoint digests before loading models. The pinned legacy Whisper build and
Lightning runtime require the recorded setuptools version that still provides
`pkg_resources`.

`run.py --offline-reference` applies the Linux libseccomp network-denial policy
to both Python and Rust model children. Downloads and Cargo compilation happen
outside it; missing kernel/library support fails explicitly. The standalone
`e2e.py` and resident `benchmark_tts.py` support the same flag. This avoids
confusing a build's local IPC requirements with a model's network permissions.

Run from the repository root with Python 3.12 and a working Rust toolchain:

```sh
python scripts/reference/run.py --suite parity --model qwen --device cpu \
  --fixture-kind synthetic --prepare-env --output target/parity-qwen-small

python scripts/reference/run.py --suite parity --model qwen --device cpu \
  --fixture-kind pretrained --prepare-env --output target/parity-qwen-weights

python scripts/reference/run.py --suite parity --model cosyvoice --device cpu \
  --fixture-kind synthetic --prepare-env --output target/parity-cosyvoice

python scripts/reference/run.py --suite e2e --model qwen --device cpu \
  --prepare-env --output target/e2e-qwen

python scripts/reference/run.py --suite parity --model cosyvoice --device cpu \
  --fixture-kind pretrained --component hift --prepare-env --output target/parity-hift

python scripts/reference/run.py --suite benchmark --model qwen --device cpu \
  --fixture-kind pretrained --prepare-env --runs 5 --output target/benchmark-qwen
```

The Qwen speaker fixture covers the complete ECAPA encoder, including reflection
padding, dilated TDNN/Res2Net, squeeze/excitation and attentive statistics pooling.
`pretrained` downloads only the speaker tensor ranges from the real checkpoint
and converts its stored weights to float32 for both runtimes. It does not claim
waveform or acoustic frontend parity. The synthetic variant uses small, analytic
weights, serialized once and loaded by both runtimes.

The small CosyVoice fixture covers official DiT timestep embedding, two adaptive
normalization modules and the tanh-GELU feed-forward network. Pretrained CPU
stages use `--component hift|flow|llm|all`: pinned checkpoints are downloaded and
converted only for the requested stages, then Python and Candle run sequentially with identical serialized
inputs. The automatic fixture uses analytic features; `--prompt-features` accepts
a reviewed public feature file. `--model-directory`, `--converted-model-directory`
and `--cosyvoice-source` reuse existing downloads. HiFT phase/sine/UV and flow
noise tensors are exported from the actual Python execution and replayed by Rust.
Stage success does not establish full chained CosyVoice waveform parity.
`export_cosyvoice_stages.py --mel-input FLOW_JSON --frames N` can consume a flow
result directly; reuse its exact `source-noise.safetensors` in `hift_oracle` with
the candidate flow document. Python backend comparisons replay it through
`--source-noise`. The comparator reports waveform RMS, peak and SNR as
diagnostics while retaining the strict elementwise pass/fail result.

Automatic Qwen E2E downloads the full pinned Base model, creates a one-second
analytic tone as conditioning, and runs both implementations sequentially using
greedy decoding. It runs x-vector conditioning followed by ICL using the first
case's generated audio and known text. It compares exact generated codec IDs,
ICL reference codec IDs and every waveform sample. `--qwen-mode xvector` selects
only the first case; `icl` still runs its required public conditioning precursor.
This is a numerical smoke test, not a perceptual or voice-identity evaluation.
For a separately reviewed test, `--e2e-manifest PATH` accepts two argument-array
commands, hashed input/checkpoint files and an explicit noise/token replay or
deterministic strategy. Existing output paths are rejected to prevent stale
results from passing a failed invocation. Use a fresh output directory.

`cuda` and `metal` parity select that device in both runtimes and fail if it is
unavailable. Metal uses PyTorch MPS. Full automatic synthesis E2E currently runs
on CPU only; these flags do not imply accelerator parity has been established.

Comparison rejects missing/extra tensors, shape/dtype mismatch, empty payloads,
non-finite values and truncated flattened data. Floating tensors use the
elementwise rule `abs(candidate-reference) <= atol + rtol*abs(reference)`;
integer codec IDs require exact equality. Default float32 gates are `atol=1e-5`,
`rtol=1e-4`; changes require a reviewed numerical justification. Reports include
maximum/mean absolute error, RMSE, cosine similarity and failing element counts.
Sharing an RNG seed is not cross-runtime numerical evidence.

Observed CPU results and their scope are recorded in
[`validation-baseline.json`](../../tests/fixtures/validation-baseline.json).
Reports are historical observations, not substitutes for rerunning the gate.

Component benchmarking runs only after parity succeeds. It records
fresh-process wall time, including model loading and startup. Its warmups only
prime caches; results are not steady-state kernel latency or time to first audio.
It reports each sample, median, p95, spread, environment and workload scope.
For resident full-model Qwen timing, use `benchmark_tts.py --model-dir MODEL
--output target/tts-benchmark --warmups 1 --runs 3 --threads 1`; it separates load,
prompt extraction, generation and decoding and reports actual output duration/RTF.
The same resident benchmark is available through `run.py --suite benchmark
--model qwen --device cpu --benchmark-kind tts`; `--model-directory` reuses a
checkpoint, otherwise the exact pinned checkpoint is downloaded into the shared
cache. This explicit, more expensive mode measures throughput independently of
the numerical gate and does not imply numerical equivalence. The default
`--benchmark-kind component` retains the smaller workload.

Private audio and extracted speaker conditioning belong outside the repository.
`extract_speaker.py` produces a raw vector and provenance; this is neither a
trained voice model nor a complete reference-code/transcript ICL prompt. Public
fixtures contain no personal audio or speaker vectors.

The comparator's negative regression tests require no ML dependencies:

```sh
python -m unittest discover -s scripts/reference -p 'test_*.py'
```
