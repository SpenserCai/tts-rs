#!/usr/bin/env python3
"""Explicitly install and verify the pinned external CPU runtime for CosyVoice's frontend.

This developer/CI utility is never invoked by package import or a build script.
The archive remains separate from Python wheels and Node packages.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import tarfile
import urllib.request
import zipfile

VERSION = "1.24.4"
# Digests are the official microsoft/onnxruntime GitHub release asset digests.
ASSETS = {
    ("Linux", "x86_64"): (
        "onnxruntime-linux-x64-1.24.4.tgz",
        "3a211fbea252c1e66290658f1b735b772056149f28321e71c308942cdb54b747",
        "lib/libonnxruntime.so.1.24.4",
    ),
    ("Darwin", "arm64"): (
        "onnxruntime-osx-arm64-1.24.4.tgz",
        "93787795f47e1eee369182e43ed51b9e5da0878ab0346aecf4258979b8bba989",
        "lib/libonnxruntime.1.24.4.dylib",
    ),
    ("Windows", "x86_64"): (
        "onnxruntime-win-x64-1.24.4.zip",
        "d2319fddfb6ea4db99ccc4b60c85c517bcd855721f5daa6a06d40d7cb2ee2357",
        "lib/onnxruntime.dll",
    ),
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_runtime(library: Path) -> str:
    """Check native dependencies, runtime version and C API 24, without claiming model parity."""
    convention = ctypes.WINFUNCTYPE if os.name == "nt" else ctypes.CFUNCTYPE

    class ApiBase(ctypes.Structure):
        _fields_ = [
            ("GetApi", convention(ctypes.c_void_p, ctypes.c_uint32)),
            ("GetVersionString", convention(ctypes.c_char_p)),
        ]

    loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
    runtime = loader(str(library))
    runtime.OrtGetApiBase.restype = ctypes.POINTER(ApiBase)
    runtime.OrtGetApiBase.argtypes = []
    base = runtime.OrtGetApiBase().contents
    if not base.GetApi(24):
        raise RuntimeError("ONNX Runtime does not expose C API version 24")
    version = base.GetVersionString().decode("ascii")
    if version != VERSION:
        raise RuntimeError(f"Expected ONNX Runtime {VERSION}, loaded {version}")
    return version


def install(destination: Path) -> dict[str, object]:
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    key = (platform.system(), machine)
    if key not in ASSETS:
        raise RuntimeError(f"No reviewed CPU runtime archive for {key}")
    asset, digest, relative_library = ASSETS[key]
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / asset
    url = f"https://github.com/microsoft/onnxruntime/releases/download/v{VERSION}/{asset}"
    if not archive_path.exists():
        temporary = archive_path.with_suffix(archive_path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        if sha256(temporary) != digest:
            temporary.unlink()
            raise RuntimeError("Downloaded ONNX Runtime archive SHA-256 mismatch")
        temporary.replace(archive_path)
    if sha256(archive_path) != digest:
        raise RuntimeError("Cached ONNX Runtime archive SHA-256 mismatch; remove it and retry")
    directory = asset.removesuffix(".tgz").removesuffix(".zip")
    if asset.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
    else:
        with tarfile.open(archive_path) as archive:
            archive.extractall(destination, filter="data")
    library = destination / directory / relative_library
    if not library.is_file():
        raise RuntimeError(f"Expected runtime library missing from the verified archive: {library}")
    version = verify_runtime(library)
    return {
        "schema_version": 1,
        "source": url,
        "archive_sha256": digest,
        "version": version,
        "execution_provider": "CPU",
        "library": str(library),
        "validation": "Native library load and C API 24; does not test model graphs or synthesis",
        "distribution_policy": "External runtime; not bundled in Python or npm packages",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".cache/onnxruntime"))
    parser.add_argument("--github-env", action="store_true", help="Export path to later Actions steps")
    args = parser.parse_args()
    result = install(args.output)
    (args.output / "runtime.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.github_env:
        environment_file = os.environ.get("GITHUB_ENV")
        if not environment_file:
            raise SystemExit("--github-env requires GITHUB_ENV")
        with open(environment_file, "a", encoding="utf-8") as output:
            output.write(f"ORT_DYLIB_PATH={result['library']}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
