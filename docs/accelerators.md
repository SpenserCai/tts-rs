# Accelerator preview packages

The GPU packages are **build-verified previews**, pending user hardware tests.
They are downloadable wheel and npm tarball assets on GitHub Releases, not
published under the default PyPI/npm names. CPU packages remain independently
installable and usable without a GPU runtime. No model weights, NVIDIA driver,
CUDA redistributables or ONNX Runtime library are bundled.

## Choosing a package

| Platform | Backend | Python backend distribution | Node backend tarball package |
| --- | --- | --- | --- |
| Linux x64 | CUDA 11.8 | `speechfoundry-cu118` | `@spensercai/speechfoundry-cu118-linux-x64` |
| Linux x64 | CUDA 12.4 | `speechfoundry-cu124` | `@spensercai/speechfoundry-cu124-linux-x64` |
| Linux x64 | CUDA 12.8 | `speechfoundry-cu128` | `@spensercai/speechfoundry-cu128-linux-x64` |
| Windows x64 | CUDA 12.4 | `speechfoundry-cu124` | `@spensercai/speechfoundry-cu124-win32-x64` |
| macOS ARM64 | Metal | `speechfoundry-metal` | `@spensercai/speechfoundry-metal-darwin-arm64` |

These are installation identities, not application imports. Every native device
uses `from speechfoundry import Engine` or the `@spensercai/speechfoundry` Node
entry point. Install the facade and the selected backend at the **same version**:

Download the matching `.whl` and/or `.tgz`, and verify its SHA-256 against the
release's `SHA256SUMS`. Install the local filename explicitly:

```sh
python -m pip install 'speechfoundry==0.2.0' ./speechfoundry_cu124-0.2.0-cp310-abi3-linux_x86_64.whl
npm install @spensercai/speechfoundry@0.2.0 ./spensercai-speechfoundry-cu124-linux-x64-0.2.0.tgz
```

Replace the version and platform tag with the actual asset filename. Rust/npm
release candidates use `X.Y.Z-rc.N`; Python's equivalent spelling is `X.Y.ZrcN`.
These are one logical workspace version, not separately maintained versions.
For a matching CPU comparison, explicitly install the same candidate version
(`pip install speechfoundry==X.Y.ZrcN` and
`npm install @spensercai/speechfoundry@X.Y.Z-rc.N`). Ordinary unpinned installs remain
on the stable channel. Rust users select
`speechfoundry = { version = "=X.Y.Z-rc.N", features = ["cuda"] }` or `metal`;
Cargo builds the selected backend from source using the local toolkit.

```python
from speechfoundry import Engine, installed_backends

print(installed_backends())  # Metadata only; not a GPU execution test.
cpu = Engine.load("models/qwen-base", "qwen3", "cpu")
gpu = Engine.load("models/qwen-base", "qwen3", "cuda:0")
# Multiple CUDA builds installed: add backend="cu124".
# Apple Silicon: Engine.load("models/qwen-base", "qwen3", "metal:0")
```

```js
import { loadEngine } from '@spensercai/speechfoundry'
const engine = await loadEngine('models/qwen-base', 'qwen3', 'cuda:0')
// Multiple CUDA builds installed: pass { backend: 'cu124' } as argument four.
```

The facade always uses its ordinary CPU binary for `cpu`. GPU libraries load only
when a matching device is explicitly requested. One installed backend is selected
automatically; multiple CUDA builds require a build choice, not a different import.
Discovery does not inspect drivers or guess the newest compatible CUDA version.
Missing runtimes, ambiguous selection and mismatched versions fail explicitly.

Different CUDA packages coexist on disk, but the facade prevents selecting more
than one CUDA build in a process, even after a failed native import. Test another
CUDA version in a new process: the OS loader can reuse same-SONAME CUDA libraries,
and Python/Node namespaces cannot isolate native ABIs. CPU and the selected GPU
remain independently usable. Source-built accelerator-enabled facade binaries
can use their compiled `cuda` / `metal` backend without an additional preview.

## Runtime prerequisites and validation

- CUDA previews compile PTX for compute capability 8.0 (Ampere or newer). Use a
  compatible NVIDIA driver and matching toolkit/runtime libraries. A newer card
  may require a newer driver/toolkit; the cu118 build is not a claim of universal
  GPU compatibility. No driver stubs are shipped or accepted as runtime validation.
- Linux previews are built on Ubuntu 22.04 and use a `linux_x86_64` wheel tag,
  not an unverified manylinux compatibility claim. Supply CUDA's library paths
  through the OS loader configuration or `LD_LIBRARY_PATH` before launch.
- Windows CUDA previews need the matching runtime DLLs discoverable by the loader
  and the Visual C++ runtime. For Python 3.10+, retain an
  `os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin")`
  handle **before** loading a CUDA engine. Node uses the configured DLL search
  path, including `PATH`. Do not copy arbitrary DLLs into the package.
- Metal previews require Apple Silicon macOS with a usable Metal device.
- CosyVoice reference extraction still requires external CPU ONNX Runtime via
  `ORT_DYLIB_PATH`. F0 and the current iSTFT implementation also run on CPU. This
  is a hybrid inference path, not an all-GPU pipeline.

Release jobs check strict accelerator compilation/Clippy/rustdoc, package names,
versions, namespaces and hashes. The Metal job exercises portable binding APIs;
this does not establish Metal execution. Fresh-install jobs co-install previews
alongside the CPU packages and rerun CPU smoke tests. Each preview's JSON report
explicitly records `gpu_execution: false` and `model_parity: false`.

For hardware testing, first load the selected device, then use the same local
checkpoint/reference PCM/transcript to extract conditioning and synthesize a short
sentence. Test Qwen and CosyVoice separately. Record package version, build-report
commit, OS, GPU, driver, CUDA/Metal, model digests, elapsed time and any complete
error. Compare shape, finite samples, duration and listening quality. Equal seeds
do not imply identical CPU/GPU random numbers or waveform samples. Do not relax
the existing tensor tolerances to declare GPU parity.

## Distribution design

[PyTorch](https://pytorch.org/get-started/previous-versions/) selects CPU/CUDA wheels
through separate indexes while retaining one `torch` import namespace. That does
not give two independent installations in the same environment.
[TensorFlow.js](https://github.com/tensorflow/tfjs/blob/master/tfjs-node/README.md)
uses distinct native CPU and GPU packages. Here, accelerator distribution names
and private native namespaces are distinct, behind one stable application API.
Python uses standard distribution entry points; Node reads the installed backend
manifest. The loader checks distribution version, compiled version and wrapper
protocol before model loading. All packages are generated from the same tracked
Rust models; no model implementation is forked.

WASM32 and WASM64 remain separately packaged CPU targets. Memory64 increases the
address space; it does not add WebGPU. No CUDA or Metal feature is enabled for
WebAssembly. Browser GPU support requires a separately validated tensor backend.

Previews do not change the stable npm `latest` tag. Default package release
candidates use npm `next`, PyPI prerelease versions and a GitHub prerelease.
Accelerator archives stay on GitHub until hardware feedback supports promotion.
