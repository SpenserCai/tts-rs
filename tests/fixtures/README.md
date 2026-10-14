# Numerical fixtures

Generate fixtures with `scripts/reference/run.py`; large pretrained weights and
private audio do not belong in Git. The small checked-in JSON schema is a
transport contract, not proof of model parity.

Each tensor JSON has `schema_version: 1`, a `metadata` object and a nonempty
`tensors` object. A tensor has `shape` (nonnegative integer dimensions), `dtype`
(`float32`, `int64`, etc.) and flattened row-major `data`. Numerical comparisons
reject missing/extra tensors, shape/dtype mismatch, empty tensors and NaN/Inf.
Integer token IDs always require exact equality.

Qwen speaker fixtures contain `[batch, mel_channels, time]` input `mel` and
`[batch, embedding_dim]` output `speaker_embedding`. The exporter transposes the
same input for the official Python module's `[batch, time, mel_channels]` API.
It serializes actual weights and input values; shared RNG seeds are not used as
a cross-framework equivalence guarantee.

[`validation-baseline.json`](validation-baseline.json) records observed CPU
results, their exact scope and source/weight provenance. It contains scalar
metrics only, not audio, model weights or speaker vectors. It is historical
evidence; rerun the evaluation scripts to validate a changed implementation.
