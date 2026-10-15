# Frontend regression fixtures

`frontend.json` contains float32 CPU outputs from the pinned CosyVoice Python
environment. The input is 2,401 samples at 24 kHz, generated from the integer LCG
in `scripts/reference/export_cosyvoice_frontend.py`. It includes both STFT
boundaries and a fractional resampling tail. It contains no recording or model
weights. Regenerate with the prepared reference environment:

```bash
python scripts/reference/offline_exec.py -- .reference/cosyvoice/bin/python \
  scripts/reference/export_cosyvoice_frontend.py \
  --source .reference/CosyVoice \
  --output crates/tts-cosyvoice/tests/fixtures/frontend.json
```

The exporter also writes a sibling `*-input.json` for `frontend_audio_oracle`;
the unit test regenerates that input, so the sibling file need not be committed.
The shared acceptance rule is `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`.

`explicit_padding_average.onnx` is an authored 11-opset, IR-8 graph without
learned weights: input `[1,1,3]` → constant `Pad([0,0,0,0,0,0])` →
`AveragePool(kernel=4, stride=4, ceil_mode=1, count_include_pad=0)`.
For `[1,2,3]`, the expected output is `2.0`. ONNX Runtime 1.24.4's `Pad_Fusion`
changes the divisor and returns `1.5`. The ONNX-enabled regression exercises the
production CAMPPlus session settings with this graph; set `ORT_DYLIB_PATH` to
the supported native runtime. Missing runtime is an error, not a skipped test.
