# Maintaining tts-rs

## Contract

This repository implements local speech inference. `candle-tts` is the public
Rust package; `tts-rs` is the repository name. Preserve a small portable core,
independent model crates, and thin language bindings. Model conditioning is not
a universal speaker identity. Never cast one model's voice prompt into another.

Before changing a module, read its nearest source-provenance document and the
pinned official Python implementation in `scripts/reference/pins.json`. A port
or model card is not a substitute for reading the executed Python path.

## Boundaries

- `tts-core`: finite mono PCM, audio I/O/resampling, validated contracts. No
  model weights, Candle tensors, Python, Node, network clients, or GPU runtime.
- `tts-qwen`, `tts-cosyvoice`: inference, configuration, preprocessing, cache and
  model-specific payloads. Share a primitive only when its math and semantics
  actually match. Do not force unrelated codec or mel pipelines together.
- `candle-tts`: public session orchestration, checkpoint identity and adapters.
- `bindings/*`: conversion, ownership, error translation and worker dispatch.
  Do not reimplement model math here. Python releases the GIL; Node moves model
  work to workers. Browser inference must use explicit bytes and a Worker.
- `scripts/reference`: pinned upstream oracle, reproducible tensor exports,
  model validation and benchmarks. Downloads and recordings are not source.

## Rust quality gate

Use edition 2024 and the pinned toolchain. Run `python scripts/ci.py rust` before
handing off a change. It runs fmt, strict Clippy, tests and strict rustdoc.
Every supported build must have zero errors and zero warnings. Do not add
crate-wide `allow(warnings)`, silence a failed test, or change numeric tolerances
just to obtain a green result. Targeted allowances need a local technical reason.
Do not use `--all-features`: CUDA and Metal have different system prerequisites.

Public entry points validate shapes, ranges, finite values, sample rates,
configuration versions and checkpoint compatibility. Invalid inputs return
errors; no panic or silent hardware/model fallback. Keep unsafe code out unless
required by a well-reviewed FFI boundary, with a documented safety argument.

## Numerical changes

Pin Python source revision, checkpoint revision/digests, dtype, device and input
hash. Compare layer outputs before judging waveforms. Shared seeds do not make
PyTorch and Rust random numbers identical: export random tensors or use greedy
teacher-forced inputs. Test tails, padding, normalization, RoPE, masks and cache
offsets when changing those operations. Missing weights/fixtures/operators fail
the requested model test; they must not turn into a passing skipped test.

Keep reference environments separate by model family and verify their pinned
runtime dependencies before importing model code. A newer Transformers version
can change cached attention semantics without changing checkpoint weights. For
prepared Linux reference runs, use `scripts/reference/offline_exec.py` to deny
network syscalls at the kernel boundary; download dependencies and weights before
entering that boundary. Offline environment variables alone are insufficient.

Record measured error and tolerance per component. Speaker parity does not prove
codec, language model, full waveform, CUDA or Metal parity. State untested support
precisely in capability tables and verification artifacts. Do not fabricate a
benchmark or call an empty benchmark run successful.

## CI and releases

`[workspace.package].version` in the root Cargo manifest is the single source of
the release version. Every Rust crate and binding uses `version.workspace = true`;
Python obtains its dynamic version from Cargo. Node's manifest and root lockfile
entries are synchronized copies. Both Wasm targets and native platform npm
packages inherit that same version at packaging time. Publish one stable
`MAJOR.MINOR.PATCH` version across crates.io, PyPI, npm and the GitHub `vVERSION`
tag; never independently bump a binding or silently reuse a published version.

Use `python scripts/version.py set X.Y.Z` for a release update and commit its
manifest and lockfile changes together. Run `python scripts/version.py check`
locally and in CI. The version command does not tag, merge, dispatch or publish.
The initial release remains 0.1.0 until released; subsequent changes follow
Semantic Versioning, with breaking changes before 1.0 incrementing the minor.

All GitHub workflows remain `workflow_dispatch` only. Keep a cheap Linux gate,
an explicit full platform matrix, and separate expensive model evaluation.
Pin external actions to verified commit SHAs. Use minimal permissions, cache
keys tied to dependency/model revisions, timeouts and bounded artifact retention.
Never add push/PR/schedule triggers without a maintainer request.

Release defaults to a dry run. Real registry publication requires the existing
matching tag, complete checks and configured registry identities. Build once,
test the built Python/npm packages, then reuse those exact artifacts. Cargo
performs its standard package verification again when publishing the same tagged
sources, after checking that repackaged archives match the verified artifacts.
Registry jobs use repository secrets `CARGO_REGISTRY_TOKEN`, `NPM_TOKEN`
and `PYPI_API_TOKEN`. Preflight may check presence only; do not print or export
credential values. Only the corresponding publisher uses each credential.
After all registry uploads succeed, publish the GitHub release with checksums and
the tested distributions. Partial registry failures remain failed releases.
Retry failed publisher jobs from the same workflow run so they reuse the original
artifacts: publishers skip
only archives whose registry digest exactly matches and reject conflicting or
yanked files. Do not rebuild an already uploaded version and blindly skip it;
new build metadata can change wheel/npm hashes even for the same source commit.
Do not create release tags, merge branches, publish packages or configure secrets as a side effect of
routine development. A requested development task may be delivered in a branch
or draft PR without publication.

## Review and documentation

Keep README examples executable and architecture diagrams in Mermaid. Prefer
stable boundaries and commands to duplicate lists of implementation details.
Update source attribution when importing code; model licenses remain separate.
Never commit personal recordings, extracted voice vectors, credentials, large
weights or unrelated project context. Do not put such data in workflow outputs.

For a handoff, describe behavior, measured checks, numerical evidence and concrete
remaining gaps. Distinguish implemented, compiled and executed on each platform.
