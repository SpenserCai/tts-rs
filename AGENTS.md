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
Prepare the pinned frontend runtime with `python scripts/install_onnxruntime.py`
and set `ORT_DYLIB_PATH` to the reported library before running native tests.
The small ONNX operator regression is part of the default workspace gate; a
missing runtime must fail rather than skip that regression. CI prepares it explicitly.
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
After the initial 0.1.0 release, changes follow Semantic Versioning: bug fixes
increment the patch; breaking changes before 1.0 increment the minor. Numerical
bug fixes may change seeded synthesis output; disclose this without claiming
cross-version or cross-framework RNG equivalence.

Release runs after a PR is merged into `main`. Its `push` event is only a signal:
the lightweight planner must verify the exact triggering SHA against GitHub's PR
association API (`closed`, `merged_at`, matching `merge_commit_sha`, base repository
and `main` branch), and confirm that commit belongs to `origin/main`. Direct pushes
do not publish. Every job checks out that immutable SHA, never a PR head or a moving
branch. Do not replace this with a privileged unmerged-PR trigger. Completed versions
are cheap no-ops with an Actions summary; a new release requires a unified version
update. Other CI, benchmarks and model evaluation remain `workflow_dispatch` only.
Keep a cheap Linux gate, an explicit full platform matrix, and separate expensive
model evaluation.
Pin external actions to verified commit SHAs. Use minimal permissions, cache
keys tied to dependency/model revisions, timeouts and bounded artifact retention.
Never add push/PR/schedule triggers without a maintainer request.

The manual release entry point defaults to `artifacts`; choose `registries` to
publish a commit already in `main` history. Both automatic and manual release derive
the version from the workspace, with no independent tag/version input. After complete
checks and package installation tests, reserve an annotated `vVERSION` tag for this
exact commit and workflow run. Never move or overwrite an existing tag. Build once,
test the built Python/npm packages, then reuse those exact artifacts. Cargo
performs its standard package verification again when publishing the same tagged
sources, after checking that repackaged archives match the verified artifacts.
Registry jobs use repository secrets `CARGO_REGISTRY_TOKEN`, `NPM_TOKEN`
and `PYPI_API_TOKEN`. Preflight may check presence only; do not print or export
credential values. Only the corresponding publisher uses each credential.
After all registry uploads succeed, upload the tested distributions to a GitHub draft,
verify every asset digest, then publish the complete release with checksums. A retry
may add only missing identical assets, never replace existing ones. Partial registry
failures remain failed releases. Use **Re-run failed jobs** from the same workflow
run, not **Re-run all jobs**, so publishers reuse the original artifacts. Publishers
skip only archives whose registry digest exactly matches and reject conflicting or
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
