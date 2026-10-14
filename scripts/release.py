#!/usr/bin/env python3
"""Validate and assemble release artifacts; publication always requires an explicit command."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib

from package_licenses import stage

ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT / "bindings/node"
EXCLUDED = ("tts-python", "tts-node", "tts-wasm")


def run(*args: str, cwd: Path = ROOT) -> None:
    command = list(args)
    command[0] = shutil.which(command[0]) or command[0]
    print("+ " + " ".join(args), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def capture(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def preflight(tag: str, publish: bool) -> None:
    manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    version = manifest["workspace"]["package"]["version"]
    if tag != f"v{version}" or not re.fullmatch(r"v\d+\.\d+\.\d+(?:-[a-zA-Z0-9.-]+)?", tag):
        raise SystemExit(f"Tag must equal the workspace version: v{version}")
    for member in manifest["workspace"]["members"]:
        for path in ROOT.glob(f"{member}/Cargo.toml"):
            package = tomllib.loads(path.read_text(encoding="utf-8"))["package"]
            declared = package["version"]
            package_version = version if declared == {"workspace": True} else declared
            if package_version != version:
                raise SystemExit(f"Rust package {package['name']} must use the workspace version {version}")
    node_version = json.loads((NODE / "package.json").read_text(encoding="utf-8"))["version"]
    pyproject = tomllib.loads((ROOT / "bindings/python/pyproject.toml").read_text(encoding="utf-8"))
    python_version = pyproject["project"].get("version", version)
    if node_version != version or python_version != version:
        raise SystemExit("Cargo, Python and Node package versions must match before release")
    if publish:
        commit = capture("git", "rev-parse", "HEAD")
        tag_commit = capture("git", "rev-parse", f"refs/tags/{tag}^{{commit}}")
        if tag_commit != commit:
            raise SystemExit("The existing release tag must point at the checked-out commit")
        if capture("git", "status", "--porcelain", "--untracked-files=no"):
            raise SystemExit("Refusing to publish modified tracked sources")
    print(f"Validated {tag}; publication requested: {publish}")


def rust_packages(publish: bool) -> None:
    excludes = [arg for name in EXCLUDED for arg in ("--exclude", name)]
    args = ["cargo", "publish", "--workspace", *excludes, "--locked"]
    if not publish:
        args.append("--dry-run")
    run(*args)
    if not publish:
        collect_crates()


def collect_crates() -> None:
    """Collect this commit's verified archives, including Cargo's workspace staging layout."""
    workspace = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))["workspace"]
    expected = {}
    for member in workspace["members"]:
        for path in ROOT.glob(f"{member}/Cargo.toml"):
            package = tomllib.loads(path.read_text(encoding="utf-8"))["package"]
            if package["name"] in EXCLUDED or package.get("publish") is False:
                continue
            version = package["version"]
            if version == {"workspace": True}:
                version = workspace["package"]["version"]
            expected[f"{package['name']}-{version}.crate"] = f"{package['name']}-{version}"
    if not expected:
        raise SystemExit("No publishable workspace crates found")
    target = Path(os.environ.get("CARGO_TARGET_DIR", str(ROOT / "target")))
    if not target.is_absolute():
        target = ROOT / target
    output = target / "release-crates"
    if set(path.name for path in output.glob("*.crate")) - set(expected):
        raise SystemExit("Unexpected stale archives in target/release-crates; select a fresh output directory")
    commit = capture("git", "rev-parse", "HEAD")
    selected = {}
    for filename, prefix in expected.items():
        candidates = {}
        for path in (target / "package").rglob(filename):
            with tarfile.open(path) as archive:
                try:
                    stream = archive.extractfile(f"{prefix}/.cargo_vcs_info.json")
                except KeyError:
                    continue
                if stream is None or json.load(stream).get("git", {}).get("sha1") != commit:
                    continue
            candidates[hashlib.sha256(path.read_bytes()).hexdigest()] = path
        if len(candidates) != 1:
            raise SystemExit(f"Expected one unique verified archive for {filename} at {commit}; found {len(candidates)}")
        selected[filename] = next(iter(candidates.values()))
    output.mkdir(parents=True, exist_ok=True)
    for filename, source in selected.items():
        shutil.copyfile(source, output / filename)
    print(f"Collected {len(selected)} Cargo archives for {commit} in {output}")


def collect_node(artifacts: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    run("npm", "ci", cwd=NODE)
    run("npx", "--no-install", "napi", "create-npm-dirs", cwd=NODE)
    run("npx", "--no-install", "napi", "artifacts", "--output-dir", str(artifacts.resolve()), cwd=NODE)
    # Explicitly disable the two external side effects while preparing exact versions.
    run("npx", "--no-install", "napi", "pre-publish", "--tag-style", "npm",
        "--no-gh-release", "--skip-optional-publish", cwd=NODE)
    manifest = json.loads((NODE / "package.json").read_text(encoding="utf-8"))
    packages = sorted((NODE / "npm").glob("*/package.json"))
    if len(packages) != len(manifest["napi"]["targets"]):
        raise SystemExit("The Node platform package count does not match napi.targets")
    for package in packages:
        if len(list(package.parent.glob("*.node"))) != 1:
            raise SystemExit(f"Missing or ambiguous native binary in {package.parent}")
        stage("node", package.parent, npm=True)
        run("npm", "pack", "--ignore-scripts", "--pack-destination", str(output.resolve()), cwd=package.parent)
    # Ship binaries only in their platform packages, not redundantly in the root tarball.
    for binary in NODE.glob("*.node"):
        binary.unlink()
    run("npm", "pack", "--ignore-scripts", "--pack-destination", str(output.resolve()), cwd=NODE)


def checksums(directory: Path) -> None:
    lines = []
    for file in sorted(directory.rglob("*")):
        if file.is_file() and file.name != "SHA256SUMS":
            with file.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            lines.append(f"{checksum}  {file.relative_to(directory).as_posix()}")
    if not lines:
        raise SystemExit("No release artifacts to checksum")
    (directory / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def runtime_manifest(output: Path) -> None:
    """Ship deployment requirements as evidence, without pretending to bundle a runtime."""
    from install_onnxruntime import ASSETS, VERSION

    document = {
        "schema_version": 1,
        "default_native_backend": "CPU",
        "wasm_package": {
            "name": "@spensercai/tts-wasm",
            "qwen_speaker_extraction": True,
            "model_weights_included": False,
            "speech_synthesis": False,
            "native_accelerators": False,
        },
        "accelerators": "CUDA and Metal require separate source builds and matching hardware",
        "cosyvoice_frontend": {
            "runtime": "ONNX Runtime",
            "minimum_c_api": 24,
            "execution_provider": "CPU",
            "bundled_in_wheel_or_npm": False,
            "environment_variable": "ORT_DYLIB_PATH",
            "reviewed_runtime_version": VERSION,
            "archives": [
                {"os": target[0], "architecture": target[1], "name": asset[0], "sha256": asset[1]}
                for target, asset in ASSETS.items()
            ],
            "requirements": "Supply the matching shared library and its native dependencies; Windows needs the Visual C++ runtime",
        },
        "validation_scope": "Binding smoke covers package import and portable APIs, not full model inference",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def publish_npm(directory: Path) -> None:
    """Publish platform tarballs first and the facade last, without lifecycle hooks."""
    packages = []
    for path in directory.glob("*.tgz"):
        with tarfile.open(path) as archive:
            stream = archive.extractfile("package/package.json")
            if stream is None:
                raise SystemExit(f"Missing package.json in {path}")
            manifest = json.load(stream)
        packages.append((bool(manifest.get("optionalDependencies")), path))
    if not packages:
        raise SystemExit("No npm tarballs found")
    for _, path in sorted(packages):
        run("npm", "publish", str(path.resolve()), "--ignore-scripts", "--access", "public")


def github_release(tag: str, directory: Path) -> None:
    files = [str(path) for path in sorted(directory.rglob("*")) if path.is_file()]
    if not files:
        raise SystemExit("No release artifacts found")
    run("gh", "release", "create", tag, "--draft", "--verify-tag", "--generate-notes", *files)


def test_wheel(directory: Path) -> None:
    wheels = list(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"Expected one native wheel; got {len(wheels)}")
    run(sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheels[0]))
    run(sys.executable, str(ROOT / "bindings/python/tests/smoke.py"))


def test_node_packages(directory: Path) -> None:
    """Exercise the packed facade and this host's optional dependency in a clean install."""
    node_os = {"Linux": "linux", "Darwin": "darwin", "Windows": "win32"}[platform.system()]
    node_cpu = "arm64" if platform.machine().lower() in {"arm64", "aarch64"} else "x64"
    selected: list[Path] = []
    root_name = None
    for path in directory.glob("*.tgz"):
        with tarfile.open(path) as archive:
            stream = archive.extractfile("package/package.json")
            if stream is None:
                raise SystemExit(f"Missing package manifest: {path}")
            manifest = json.load(stream)
        if manifest.get("optionalDependencies"):
            root_name = manifest["name"]
            selected.append(path)
        elif node_os in manifest.get("os", []) and node_cpu in manifest.get("cpu", []):
            selected.append(path)
    if root_name is None or len(selected) != 2:
        raise SystemExit("Expected the root npm package and exactly one matching native package")
    with tempfile.TemporaryDirectory(prefix="tts-package-") as temporary:
        prefix = Path(temporary)
        (prefix / "package.json").write_text('{"name":"package-smoke","private":true}\n', encoding="utf-8")
        run("npm", "install", "--ignore-scripts", "--no-audit", "--no-fund",
            *(str(path.resolve()) for path in selected), cwd=prefix)
        source = (NODE / "tests/smoke.cjs").read_text(encoding="utf-8")
        if "require('../index.js')" not in source:
            raise SystemExit("Update the package smoke adapter for the changed Node smoke entry point")
        source = source.replace("require('../index.js')", f"require({json.dumps(root_name)})")
        (prefix / "smoke.cjs").write_text(source, encoding="utf-8")
        run("node", "--test", "smoke.cjs", cwd=prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("preflight")
    pre.add_argument("--tag", default=os.getenv("RELEASE_TAG"), required=not os.getenv("RELEASE_TAG"))
    pre.add_argument("--publish", action="store_true")
    rust = commands.add_parser("rust")
    rust.add_argument("--publish", action="store_true")
    commands.add_parser("collect-crates", help="Collect current-commit Cargo archives after a successful dry-run")
    node = commands.add_parser("collect-node")
    node.add_argument("--artifacts", type=Path, required=True)
    node.add_argument("--output", type=Path, required=True)
    sums = commands.add_parser("checksums")
    sums.add_argument("directory", type=Path)
    runtime = commands.add_parser("runtime-manifest")
    runtime.add_argument("--output", type=Path, required=True)
    npm = commands.add_parser("publish-npm")
    npm.add_argument("directory", type=Path)
    wheel = commands.add_parser("test-wheel")
    wheel.add_argument("directory", type=Path)
    node_test = commands.add_parser("test-node")
    node_test.add_argument("directory", type=Path)
    github = commands.add_parser("github")
    github.add_argument("--tag", default=os.getenv("RELEASE_TAG"), required=not os.getenv("RELEASE_TAG"))
    github.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args.tag, args.publish)
    elif args.command == "rust":
        rust_packages(args.publish)
    elif args.command == "collect-crates":
        collect_crates()
    elif args.command == "collect-node":
        collect_node(args.artifacts, args.output)
    elif args.command == "checksums":
        checksums(args.directory)
    elif args.command == "runtime-manifest":
        runtime_manifest(args.output)
    elif args.command == "publish-npm":
        publish_npm(args.directory)
    elif args.command == "test-wheel":
        test_wheel(args.directory)
    elif args.command == "test-node":
        test_node_packages(args.directory)
    else:
        github_release(args.tag, args.directory)


if __name__ == "__main__":
    main()
