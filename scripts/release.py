#!/usr/bin/env python3
"""Validate and assemble release artifacts; publication always requires an explicit command."""

from __future__ import annotations

import argparse
import base64
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from package_licenses import stage
from version import check as check_version

ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT / "bindings/node"
EXCLUDED = ("tts-python", "tts-node", "tts-wasm")


def check_secrets() -> None:
    """Check presence only; credentials never become log output or artifacts."""
    required = ("CARGO_REGISTRY_TOKEN", "NPM_TOKEN", "PYPI_API_TOKEN")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit("Missing registry secret(s): " + ", ".join(missing))
    print("All three required registry credentials are configured")


def run(*args: str, cwd: Path = ROOT) -> None:
    command = list(args)
    command[0] = shutil.which(command[0]) or command[0]
    print("+ " + " ".join(args), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def capture(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def registry_document(url: str) -> dict | None:
    """Read public registry metadata; only an explicit 404 means unpublished."""
    request = urllib.request.Request(url, headers={"User-Agent": "candle-tts-release"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def preflight(tag: str, publish: bool) -> None:
    version = check_version()
    if tag != f"v{version}":
        raise SystemExit(f"Tag must equal the workspace version: v{version}")
    if publish:
        commit = capture("git", "rev-parse", "HEAD")
        tag_commit = capture("git", "rev-parse", f"refs/tags/{tag}^{{commit}}")
        if tag_commit != commit:
            raise SystemExit("The existing release tag must point at the checked-out commit")
        if capture("git", "status", "--porcelain", "--untracked-files=no"):
            raise SystemExit("Refusing to publish modified tracked sources")
    print(f"Validated {tag}; publication requested: {publish}")


def rust_packages(publish: bool, artifacts: Path | None = None) -> None:
    if publish:
        tag = os.environ.get("RELEASE_TAG")
        if not tag:
            raise SystemExit("RELEASE_TAG is required for registry publication")
        preflight(tag, True)
        if artifacts is None:
            raise SystemExit("--artifacts must identify the verified release bundle")
        verify_distributions(artifacts)
        # Cargo has no "upload this archive" CLI. Repackage without recompiling,
        # verify byte identity, then let Cargo perform its standard publication.
        excludes = [arg for name in EXCLUDED for arg in ("--exclude", name)]
        run("cargo", "package", "--workspace", *excludes, "--locked", "--no-verify")
        collect_crates()
        target = Path(os.environ.get("CARGO_TARGET_DIR", str(ROOT / "target")))
        if not target.is_absolute():
            target = ROOT / target
        for verified in (artifacts / "crates").glob("*.crate"):
            repackaged = target / "release-crates" / verified.name
            if hashlib.sha256(repackaged.read_bytes()).digest() != hashlib.sha256(verified.read_bytes()).digest():
                raise SystemExit(f"Repackaged Cargo archive differs from the verified artifact: {verified.name}")
        pending = []
        for path in sorted((artifacts / "crates").glob("*.crate")):
            name, version = path.stem.rsplit("-", 1)
            document = registry_document(f"https://crates.io/api/v1/crates/{name}/{version}")
            if document is None:
                pending.extend(("--package", name))
            elif document["version"].get("yanked") or document["version"].get("checksum") != hashlib.sha256(path.read_bytes()).hexdigest():
                raise SystemExit(f"Registry already contains different or yanked {name} {version}; refusing to skip it")
            else:
                print(f"Already published, identical Cargo archive: {name} {version}")
        if not pending:
            return
        selection = pending
    else:
        selection = ["--workspace", *(arg for name in EXCLUDED for arg in ("--exclude", name))]
    args = ["cargo", "publish", *selection, "--locked"]
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


def verify_distributions(directory: Path) -> None:
    """Reject missing, mismatched or stale distributions before publication."""
    version = check_version()
    expected_files = set()
    for line in (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise SystemExit(f"Invalid artifact path: {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest or name in expected_files:
            raise SystemExit(f"Artifact checksum mismatch or duplicate: {name}")
        expected_files.add(name)
    actual_files = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
                    if path.is_file() and path.name != "SHA256SUMS"}
    if expected_files != actual_files or not expected_files:
        raise SystemExit("Artifact inventory differs from SHA256SUMS")
    runtime = json.loads((directory / "runtime-requirements.json").read_text(encoding="utf-8"))
    commit = capture("git", "rev-parse", "HEAD")
    if runtime.get("version") != version or runtime.get("source_commit") != commit:
        raise SystemExit("Release artifacts must belong to this exact version and source commit")
    wheels = list((directory / "wheels").glob("*.whl"))
    if len(wheels) != 3:
        raise SystemExit("Expected three native Python wheels")
    for path in wheels:
        with zipfile.ZipFile(path) as wheel:
            names = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
            if len(names) != 1:
                raise SystemExit(f"Invalid wheel metadata: {path.name}")
            metadata = BytesParser().parsebytes(wheel.read(names[0]))
        if metadata["Name"] != "candle-tts" or metadata["Version"] != version:
            raise SystemExit(f"Wheel version/name mismatch: {path.name}")
    npm_packages = {}
    for path in (directory / "npm").glob("*.tgz"):
        with tarfile.open(path) as archive:
            stream = archive.extractfile("package/package.json")
            if stream is None:
                raise SystemExit(f"Missing npm metadata: {path.name}")
            metadata = json.load(stream)
        if metadata.get("version") != version or metadata["name"] in npm_packages:
            raise SystemExit(f"npm version/name mismatch: {path.name}")
        npm_packages[metadata["name"]] = metadata
    facade = json.loads((NODE / "package.json").read_text(encoding="utf-8"))
    platforms = npm_packages.get(facade["name"], {}).get("optionalDependencies", {})
    required = {facade["name"], "@spensercai/tts-wasm", "@spensercai/tts-wasm64", *platforms}
    if len(platforms) != len(facade["napi"]["targets"]) or any(value != version for value in platforms.values()):
        raise SystemExit("Node optional packages must use the unified exact version")
    if set(npm_packages) != required:
        raise SystemExit("The npm distribution set is incomplete or contains unexpected packages")
    workspace = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))["workspace"]
    expected_crates = {f"{package['name']}-{version}.crate"
                       for member in workspace["members"] for path in ROOT.glob(f"{member}/Cargo.toml")
                       for package in [tomllib.loads(path.read_text(encoding="utf-8"))["package"]]
                       if package.get("publish") is not False}
    if {path.name for path in (directory / "crates").glob("*.crate")} != expected_crates:
        raise SystemExit("The Cargo distribution set is incomplete or contains unexpected packages")
    for path in (directory / "crates").glob("*.crate"):
        with tarfile.open(path) as archive:
            stream = archive.extractfile(f"{path.stem}/.cargo_vcs_info.json")
            if stream is None or json.load(stream).get("git", {}).get("sha1") != commit:
                raise SystemExit(f"Cargo archive source commit mismatch: {path.name}")
    print(f"Verified {len(expected_files)} release artifacts at unified version {version}")


def runtime_manifest(output: Path) -> None:
    """Ship deployment requirements as evidence, without pretending to bundle a runtime."""
    from install_onnxruntime import ASSETS, VERSION

    document = {
        "schema_version": 1,
        "version": check_version(),
        "source_commit": capture("git", "rev-parse", "HEAD"),
        "default_native_backend": "CPU",
        "wasm_packages": [
            {
                "name": "@spensercai/tts-wasm",
                "target": "wasm32-unknown-unknown",
                "qwen_speaker_extraction": True,
                "speech_synthesis_models": [],
                "model_weights_included": False,
                "native_accelerators": False,
            },
            {
                "name": "@spensercai/tts-wasm64",
                "target": "wasm64-unknown-unknown",
                "qwen_speaker_extraction": True,
                "speech_synthesis_models": ["qwen3"],
                "model_weights_included": False,
                "native_accelerators": False,
                "runtime": "Memory64-capable WebAssembly engine; Node.js 24+; sufficient host memory for weights and activations",
                "validation": "See tests/fixtures/validation-baseline.json; an address-space probe does not establish full model parity",
            },
        ],
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
        packages.append((bool(manifest.get("optionalDependencies")), path, manifest))
    if not packages:
        raise SystemExit("No npm tarballs found")
    for _, path, manifest in sorted(packages):
        name = urllib.parse.quote(manifest["name"], safe="")
        document = registry_document(f"https://registry.npmjs.org/{name}/{manifest['version']}")
        if document is not None:
            integrity = "sha512-" + base64.b64encode(hashlib.sha512(path.read_bytes()).digest()).decode("ascii")
            if document.get("dist", {}).get("integrity") != integrity:
                raise SystemExit(f"npm already contains a different archive for {manifest['name']} {manifest['version']}")
            print(f"Already published, identical npm archive: {manifest['name']} {manifest['version']}")
            continue
        run("npm", "publish", str(path.resolve()), "--ignore-scripts", "--access", "public")


def prepare_pypi(directory: Path, output: Path) -> None:
    """Resume a partial PyPI upload only when existing files have identical hashes."""
    version = check_version()
    document = registry_document(f"https://pypi.org/pypi/candle-tts/{version}/json")
    existing = {item["filename"]: item for item in (document or {}).get("urls", [])}
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit("PyPI staging directory must be empty")
    pending = 0
    wheels = sorted(directory.glob("*.whl"))
    if not wheels:
        raise SystemExit("No Python wheels to prepare")
    for path in wheels:
        previous = existing.get(path.name)
        if previous is not None:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if previous.get("yanked") or previous.get("digests", {}).get("sha256") != digest:
                raise SystemExit(f"PyPI already contains a different or yanked artifact: {path.name}")
            print(f"Already published, identical Python wheel: {path.name}")
        else:
            shutil.copyfile(path, output / path.name)
            pending += 1
    if os.environ.get("GITHUB_OUTPUT"):
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
            stream.write(f"has_uploads={'true' if pending else 'false'}\n")
    print(f"Python wheels remaining to publish: {pending}")


def github_release(tag: str, directory: Path, draft: bool) -> None:
    verify_distributions(directory)
    files = [str(path) for path in sorted(directory.rglob("*")) if path.is_file()]
    if not files:
        raise SystemExit("No release artifacts found")
    run("gh", "release", "create", tag, *(('--draft',) if draft else ()),
        "--verify-tag", "--generate-notes", *files)


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
    commands.add_parser("check-secrets", help="Check registry credential presence without logging values")
    pre = commands.add_parser("preflight")
    pre.add_argument("--tag", default=os.getenv("RELEASE_TAG"), required=not os.getenv("RELEASE_TAG"))
    pre.add_argument("--publish", action="store_true")
    rust = commands.add_parser("rust")
    rust.add_argument("--publish", action="store_true")
    rust.add_argument("--artifacts", type=Path, help="Verified release bundle, required for publication")
    commands.add_parser("collect-crates", help="Collect current-commit Cargo archives after a successful dry-run")
    node = commands.add_parser("collect-node")
    node.add_argument("--artifacts", type=Path, required=True)
    node.add_argument("--output", type=Path, required=True)
    sums = commands.add_parser("checksums")
    sums.add_argument("directory", type=Path)
    verify = commands.add_parser("verify-distributions", help="Validate checksums, versions and package inventory")
    verify.add_argument("directory", type=Path)
    runtime = commands.add_parser("runtime-manifest")
    runtime.add_argument("--output", type=Path, required=True)
    npm = commands.add_parser("publish-npm")
    npm.add_argument("directory", type=Path)
    pypi = commands.add_parser("prepare-pypi", help="Stage missing wheels after verifying published file digests")
    pypi.add_argument("directory", type=Path)
    pypi.add_argument("--output", type=Path, required=True)
    wheel = commands.add_parser("test-wheel")
    wheel.add_argument("directory", type=Path)
    node_test = commands.add_parser("test-node")
    node_test.add_argument("directory", type=Path)
    github = commands.add_parser("github")
    github.add_argument("--tag", default=os.getenv("RELEASE_TAG"), required=not os.getenv("RELEASE_TAG"))
    github.add_argument("--directory", type=Path, required=True)
    github.add_argument("--draft", action="store_true", help="Create a reviewable draft instead of a public release")
    args = parser.parse_args()
    if args.command == "check-secrets":
        check_secrets()
    elif args.command == "preflight":
        preflight(args.tag, args.publish)
    elif args.command == "rust":
        rust_packages(args.publish, args.artifacts)
    elif args.command == "collect-crates":
        collect_crates()
    elif args.command == "collect-node":
        collect_node(args.artifacts, args.output)
    elif args.command == "checksums":
        checksums(args.directory)
    elif args.command == "verify-distributions":
        verify_distributions(args.directory)
    elif args.command == "runtime-manifest":
        runtime_manifest(args.output)
    elif args.command == "publish-npm":
        publish_npm(args.directory)
    elif args.command == "prepare-pypi":
        prepare_pypi(args.directory, args.output)
    elif args.command == "test-wheel":
        test_wheel(args.directory)
    elif args.command == "test-node":
        test_node_packages(args.directory)
    else:
        github_release(args.tag, args.directory, args.draft)


if __name__ == "__main__":
    main()
