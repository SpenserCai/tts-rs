#!/usr/bin/env python3
"""Build, verify and package the explicit memory64 WebAssembly target."""
from __future__ import annotations

import argparse
import hashlib
import platform
import tarfile
import urllib.request
import json
import os
from pathlib import Path
import shutil
import subprocess
import tomllib

from package_licenses import stage
from version import check as check_version

ROOT = Path(__file__).resolve().parents[1]
BINDING = ROOT / "bindings/wasm"
TOOLCHAIN = "nightly-2026-10-13"
TARGET = "wasm64-unknown-unknown"
WEB = BINDING / "pkg-memory64"
NODE = BINDING / "pkg-memory64-node"


def run(*command: str, env: dict[str, str] | None = None, cwd: Path = ROOT) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env["CARGO_INCREMENTAL"] = "0"
    env.setdefault("CARGO_BUILD_JOBS", "2")
    # Cargo gives encoded/global flags precedence over target-specific flags.
    # Own the complete cross-build flag set so CI's global -D warnings cannot
    # accidentally erase the required getrandom backend selection.
    env.pop("CARGO_ENCODED_RUSTFLAGS", None)
    env.pop("CARGO_TARGET_WASM64_UNKNOWN_UNKNOWN_RUSTFLAGS", None)
    env["RUSTFLAGS"] = '--cfg getrandom_backend="custom" -D warnings'
    env["RUSTDOCFLAGS"] = "-D warnings"
    return env


def cargo(command: str, *extra: str) -> None:
    run("cargo", f"+{TOOLCHAIN}", command, "-Z", "build-std=std,panic_abort", "--locked", "--target", TARGET,
        "-p", "tts-wasm", "--features", "qwen-synthesis", *extra, env=environment())


def package_metadata(directory: Path, web: bool) -> None:
    version = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    package = {
        "name": "@spensercai/tts-wasm64", "version": version,
        "description": "Memory64 Candle Qwen3-TTS inference with explicit checkpoint bytes",
        "license": "MIT", "repository": "github:SpenserCai/tts-rs",
        "type": "module" if web else "commonjs", "main": "tts_wasm.js", "types": "tts_wasm.d.ts",
        "files": ["tts_wasm.js", "tts_wasm.d.ts", "tts_wasm_bg.wasm", "tts_wasm_bg.wasm.d.ts", "loader.mjs", "snippets", "README.md", "licenses"],
        "sideEffects": False, "engines": {"node": ">=24"},
    }
    (directory / "package.json").write_text(json.dumps(package, indent=2) + "\n")
    shutil.copyfile(BINDING / "loader.mjs", directory / "loader.mjs")
    shutil.copyfile(BINDING / "README.md", directory / "README.md")
    stage("wasm", directory, npm=True)


def bindgen_version() -> str:
    lock = tomllib.loads((ROOT / "Cargo.lock").read_text())
    return next(item["version"] for item in lock["package"] if item["name"] == "wasm-bindgen")


def bindgen_path() -> str:
    version = bindgen_version()
    candidates = [ROOT / ".cache/wasm-bindgen" / version / f"wasm-bindgen-{version}-x86_64-unknown-linux-musl/wasm-bindgen"]
    if found := shutil.which("wasm-bindgen"): candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file() and subprocess.check_output([str(candidate), "--version"], text=True).strip() == f"wasm-bindgen {version}":
            return str(candidate)
    raise SystemExit("Run python scripts/wasm64.py prepare to install the locked wasm-bindgen CLI")


def prepare() -> None:
    version = bindgen_version()
    try:
        print(bindgen_path())
        return
    except SystemExit:
        pass
    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        run("cargo", "install", "wasm-bindgen-cli", "--locked", "--version", version)
        print(bindgen_path())
        return
    # Official GitHub release asset digest, verified via its release API metadata.
    digests = {"0.2.129": "82d12bb940e2d4e72e0d5605387fc1b8ca179044e012b620f0ce4e7440e8320e"}
    if version not in digests: raise SystemExit("Update the reviewed wasm-bindgen release digest for the new lockfile")
    output = ROOT / ".cache/wasm-bindgen" / version
    output.mkdir(parents=True, exist_ok=True)
    filename = f"wasm-bindgen-{version}-x86_64-unknown-linux-musl.tar.gz"
    archive = output / filename
    with urllib.request.urlopen(f"https://github.com/wasm-bindgen/wasm-bindgen/releases/download/{version}/{filename}", timeout=60) as response, archive.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != digests[version]:
            archive.unlink()
            raise SystemExit("wasm-bindgen release archive digest mismatch")
    with tarfile.open(archive) as source: source.extractall(output, filter="data")
    print(bindgen_path())


def build() -> None:
    bindgen = bindgen_path()
    cargo("build", "--release")
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    if not target.is_absolute(): target = ROOT / target
    binary = target / TARGET / "release/tts_wasm.wasm"
    for directory, platform in [(WEB, "web"), (NODE, "nodejs")]:
        if directory.exists(): shutil.rmtree(directory)
        run(bindgen, str(binary), "--out-dir", str(directory), "--target", platform)
        package_metadata(directory, platform == "web")
    smoke()


def smoke() -> None:
    run("node", "--test", str(BINDING / "tests/memory64.cjs"), str(BINDING / "tests/loader.mjs"), str(BINDING / "tests/memory64-address.mjs"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "build", "check", "smoke", "package"))
    args = parser.parse_args()
    check_version()
    if args.command == "prepare": prepare()
    elif args.command == "build": build()
    elif args.command == "check":
        cargo("clippy", "--", "-D", "warnings")
        cargo("doc", "--no-deps")
        smoke()
    elif args.command == "smoke": smoke()
    else:
        if not (WEB / "tts_wasm_bg.wasm").is_file(): raise SystemExit("Build and test memory64 before packaging")
        manifest = json.loads((WEB / "package.json").read_text())
        version = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"]["version"]
        if manifest["version"] != version: raise SystemExit("Stale memory64 package version")
        run("npm", "pack", "--ignore-scripts", cwd=WEB)


if __name__ == "__main__": main()
