#!/usr/bin/env python3
"""Build opt-in, co-installable accelerator previews from the unchanged Rust models.

These wheels/tarballs are GitHub Release assets, not default registry packages.
Compilation and package inspection do not certify device execution or model parity.
"""

from __future__ import annotations

import argparse
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
import zipfile

from version import check, python_version

ROOT = Path(__file__).resolve().parents[1]
# Preserve the legacy build matrix, but never imply CUDA versions are interchangeable.
MATRIX = {
    "cu118-linux-x64": ("cu118", "cuda", "Linux", "linux", "x64", "11.8"),
    "cu124-linux-x64": ("cu124", "cuda", "Linux", "linux", "x64", "12.4"),
    "cu128-linux-x64": ("cu128", "cuda", "Linux", "linux", "x64", "12.8"),
    "cu124-win32-x64": ("cu124", "cuda", "Windows", "win32", "x64", "12.4"),
    "metal-darwin-arm64": ("metal", "metal", "Darwin", "darwin", "arm64", None),
}


def names(variant: str) -> tuple[str, str, str]:
    backend = MATRIX[variant][0]
    return f"candle-tts-{backend}", f"candle_tts_{backend}", f"@spensercai/tts-rs-{variant}"


def run(*args: str, cwd: Path, env: dict | None = None) -> None:
    print("+ " + " ".join(args), flush=True)
    executable = shutil.which(args[0])
    if executable is None:
        raise SystemExit(f"Required executable missing: {args[0]}")
    subprocess.run([executable, *args[1:]], cwd=cwd, env=env, check=True)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stage_workspace(destination: Path, variant: str) -> None:
    """Stage tracked sources only; change binding identity, never model implementation."""
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    for relative in filter(None, paths):
        source = ROOT / relative
        if source.is_symlink() or not source.is_file():
            raise SystemExit(f"Expected a regular tracked source: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    distribution, namespace, npm_name = names(variant)
    python = destination / "bindings/python"
    project = python / "pyproject.toml"
    source = project.read_text(encoding="utf-8")
    source = source.replace('name = "candle-tts"', f'name = "{distribution}"')
    source = source.replace('module-name = "candle_tts._native"', f'module-name = "{namespace}._native"')
    project.write_text(source, encoding="utf-8")
    (python / "python/candle_tts").rename(python / "python" / namespace)
    binding = python / "src/lib.rs"
    source = binding.read_text(encoding="utf-8")
    if source.count('module = "candle_tts"') != 2:
        raise SystemExit("Review Python class metadata before packaging accelerator namespaces")
    binding.write_text(source.replace('module = "candle_tts"', f'module = "{namespace}"'), encoding="utf-8")
    node = destination / "bindings/node"
    for package in (python, node):
        shutil.copyfile(destination / "docs/accelerators.md", package / "README.md")
    for filename in ("package.json", "package-lock.json"):
        path = node / filename
        value = json.loads(path.read_text(encoding="utf-8"))
        value["name"] = npm_name
        if filename == "package-lock.json":
            value["packages"][""]["name"] = npm_name
        else:
            value["os"] = [MATRIX[variant][3]]
            value["cpu"] = [MATRIX[variant][4]]
            value["private"] = True  # Downloadable preview; no accidental npm publish.
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def inspect_packages(directory: Path, variant: str, version: str) -> list[Path]:
    distribution, namespace, npm_name = names(variant)
    wheels, tarballs = list(directory.glob("*.whl")), list(directory.glob("*.tgz"))
    if len(wheels) != 1 or len(tarballs) != 1:
        raise SystemExit(f"{variant}: expected one wheel and one npm tarball")
    with zipfile.ZipFile(wheels[0]) as archive:
        entries = archive.namelist()
        metadata_files = [p for p in entries if p.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            raise SystemExit("Expected one Python distribution metadata file")
        metadata = BytesParser().parsebytes(archive.read(metadata_files[0]))
        if metadata["Name"] != distribution or metadata["Version"] != python_version(version):
            raise SystemExit("Accelerator Python distribution identity mismatch")
        if any(p.startswith("candle_tts/") for p in entries) or f"{namespace}/__init__.py" not in entries:
            raise SystemExit("Accelerator wheel must not overwrite the CPU namespace")
        extensions = [p for p in entries if p.startswith(namespace + "/_native") and p.endswith((".so", ".pyd"))]
        if len(extensions) != 1 or not any("LICENSE" in p for p in entries):
            raise SystemExit("Accelerator wheel is missing its extension or licenses")
    with tarfile.open(tarballs[0]) as archive:
        stream = archive.extractfile("package/package.json")
        if stream is None:
            raise SystemExit("Missing npm metadata")
        metadata = json.load(stream)
        if metadata.get("name") != npm_name or metadata.get("version") != version:
            raise SystemExit("Accelerator npm distribution identity mismatch")
        if metadata.get("os") != [MATRIX[variant][3]] or metadata.get("cpu") != [MATRIX[variant][4]]:
            raise SystemExit("Accelerator npm platform mismatch")
        if metadata.get("optionalDependencies") or metadata.get("private") is not True:
            raise SystemExit("Preview packages must be explicit, non-publishing, self-contained tarballs")
        entries = archive.getnames()
        if len([p for p in entries if p.endswith(".node")]) != 1 or "package/index.d.ts" not in entries:
            raise SystemExit("Missing accelerator Node binary or TypeScript declarations")
    return wheels + tarballs


def build(variant: str, output: Path) -> None:
    version = check()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT).strip():
        raise SystemExit("Commit the packaging sources before generating a source-identified preview")
    backend, feature, host, _, architecture, cuda = MATRIX[variant]
    machine = "arm64" if platform.machine().lower() in {"arm64", "aarch64"} else "x64"
    if platform.system() != host or machine != architecture:
        raise SystemExit(f"{variant} must build on {host}/{architecture}")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit("Use an empty accelerator output directory")
    env = os.environ.copy()
    env["RUSTFLAGS"] = f"{env.get('RUSTFLAGS', '')} -D warnings".strip()
    env["RUSTDOCFLAGS"] = "-D warnings"
    # Python, Node, lint and docs share artifacts; no repeated kernel compilation.
    env["CARGO_TARGET_DIR"] = str(ROOT / "target" / "accelerators" / variant)
    with tempfile.TemporaryDirectory(prefix="tts-accelerator-") as temporary:
        staged = Path(temporary)
        stage_workspace(staged, variant)
        run("cargo", "clippy", "-p", "candle-tts", "-p", "tts-python", "-p", "tts-node",
            "--all-targets", "--features", f"candle-tts/{feature},tts-python/{feature},tts-node/{feature}",
            "--locked", "--", "-D", "warnings", cwd=staged, env=env)
        run("cargo", "doc", "-p", "candle-tts", "--features", feature,
            "--no-deps", "--locked", cwd=staged, env=env)
        # External CUDA libraries are intentional. These are platform wheels for
        # direct installation, never mislabeled as self-contained manylinux wheels.
        audit = ("--auditwheel", "skip", "--compatibility", "linux") if host == "Linux" else ()
        run(sys.executable, "-m", "maturin", "build", "--release", "--locked", "--features", feature,
            "--manifest-path", "bindings/python/Cargo.toml", "--out", str(output), *audit, cwd=staged, env=env)
        node = staged / "bindings/node"
        run("npm", "ci", cwd=node, env=env)
        run("npm", "run", "build", "--", "--release", "--features", feature, cwd=node, env=env)
        if host == "Darwin":
            # Portable APIs in the Metal-linked binary; this is NOT a GPU execution test.
            run("npm", "test", cwd=node, env=env)
        run("npm", "pack", "--ignore-scripts", "--pack-destination", str(output), cwd=node, env=env)
    files = inspect_packages(output, variant, version)
    if host == "Darwin":
        with tempfile.TemporaryDirectory(prefix="tts-metal-import-") as temporary:
            run(sys.executable, "-m", "pip", "install", "--no-deps", "--target", temporary,
                str(files[0]), cwd=ROOT)
            env["PYTHONPATH"] = temporary
            smoke = (ROOT / "bindings/python/tests/smoke.py").read_text(encoding="utf-8")
            smoke = smoke.replace("from candle_tts import", f"from {names(variant)[1]} import")
            run(sys.executable, "-c", smoke, cwd=ROOT, env=env)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report = {
        "schema_version": 1, "version": version, "source_commit": commit,
        "variant": variant, "backend": backend, "cuda_toolkit": cuda,
        "cuda_compute_cap": os.environ.get("CUDA_COMPUTE_CAP") if cuda else None,
        "validation": {"compiled": True, "strict_clippy": True, "rustdoc": True,
                       "package_identity": True, "gpu_execution": False, "model_parity": False},
        "runtime": "Matching NVIDIA driver and CUDA runtime/cuBLAS/cuRAND/NVRTC shared libraries" if cuda else "Apple Silicon macOS with Metal",
        "bundled_gpu_runtime": False,
        "files": {path.name: sha256(path) for path in files},
    }
    (output / f"{variant}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def verify(directory: Path, version: str, commit: str) -> None:
    expected = set()
    for variant in MATRIX:
        report_name = f"{variant}.json"
        report = json.loads((directory / report_name).read_text(encoding="utf-8"))
        if (report.get("version"), report.get("source_commit"), report.get("variant")) != (version, commit, variant):
            raise SystemExit("Accelerator report source/version mismatch")
        validation = report.get("validation", {})
        if not all(validation.get(k) is True for k in ("compiled", "strict_clippy", "rustdoc", "package_identity")):
            raise SystemExit("Accelerator build verification is incomplete")
        if validation.get("gpu_execution") is not False or validation.get("model_parity") is not False:
            raise SystemExit("Build-only preview must not claim GPU execution/parity")
        files = report.get("files", {})
        if len(files) != 2 or any(Path(name).name != name for name in files):
            raise SystemExit("Invalid accelerator artifact inventory")
        # Inspect each pair independently while keeping the flat release inventory.
        with tempfile.TemporaryDirectory(prefix="tts-inspect-") as temporary:
            pair = Path(temporary)
            for name, digest in files.items():
                source = directory / name
                if sha256(source) != digest or name in expected:
                    raise SystemExit("Accelerator digest mismatch or duplicate artifact")
                shutil.copyfile(source, pair / name)
            inspect_packages(pair, variant, version)
        expected.update((report_name, *files))
    if {path.name for path in directory.iterdir()} != expected:
        raise SystemExit("Unexpected or incomplete accelerator release inventory")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=MATRIX)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.variant, args.output)


if __name__ == "__main__":
    main()
