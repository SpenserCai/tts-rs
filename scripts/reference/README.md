# Official reference evaluation

These scripts run the upstream Python implementation and compare its actual
outputs with Candle. Source/model revisions and top-level dependencies live in
[`pins.json`](pins.json). The prepared environment records transitive packages;
it is not a fully locked Python dependency graph.
Model downloads are explicit preparation steps. Reference inference requests
offline Hugging Face loading and disables ONNX Runtime's optional telemetry.

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

python scripts/reference/run.py --suite benchmark --model qwen --device cpu \
  --fixture-kind pretrained --prepare-env --runs 5 --output target/benchmark-qwen
```

The Qwen speaker fixture covers the complete ECAPA encoder, including reflection
padding, dilated TDNN/Res2Net, squeeze/excitation and attentive statistics pooling.
`pretrained` downloads only the speaker tensor ranges from the real checkpoint
and converts its stored weights to float32 for both runtimes. It does not claim
waveform or acoustic frontend parity. The synthetic variant uses small, analytic
weights, serialized once and loaded by both runtimes.

The CosyVoice fixture currently covers the official DiT timestep embedding and
two adaptive normalization modules. Full DiT, flow and vocoder parity are separate
gates. Asking for an unavailable suite returns nonzero, never a successful skip.

Automatic Qwen E2E downloads the full pinned Base model, creates a one-second
analytic tone as conditioning, and runs both implementations sequentially using
greedy decoding. It compares exact codec token IDs and every waveform sample.
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

Benchmarking runs only after parity succeeds. The included benchmark records
fresh-process wall time, including model loading and startup. Its warmups only
prime caches; results are not steady-state kernel latency or time to first audio.
It reports each sample, median, p95, spread, environment and workload scope.

Private audio and extracted speaker conditioning belong outside the repository.
`extract_speaker.py` produces a raw vector and provenance; this is neither a
trained voice model nor a complete reference-code/transcript ICL prompt. Public
fixtures contain no personal audio or speaker vectors.

The comparator's negative regression tests require no ML dependencies:

```sh
python -m unittest discover -s scripts/reference -p 'test_*.py'
```
