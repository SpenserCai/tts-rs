#!/usr/bin/env python3
"""Run the same fail-fast, portable checks locally and in manual CI."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap

from package_licenses import stage, verify
from version import check as check_version

ROOT = Path(__file__).resolve().parents[1]
BINDINGS = ("speechfoundry-python", "speechfoundry-node", "speechfoundry-wasm")


def run(*args: str, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    """Execute without a shell, preserving argument boundaries on every OS."""
    command = list(args)
    executable = shutil.which(command[0])
    if executable is None:
        raise SystemExit(f"Required executable not found: {command[0]}")
    command[0] = executable
    print("+ " + " ".join(args), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def strict_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["RUSTFLAGS"] = f"{env.get('RUSTFLAGS', '')} -D warnings".strip()
    env["RUSTDOCFLAGS"] = f"{env.get('RUSTDOCFLAGS', '')} -D warnings".strip()
    env["CARGO_TERM_COLOR"] = "always" if os.getenv("CI") else "auto"
    return env


def rust_checks(device: str) -> None:
    env = strict_environment()
    run(sys.executable, "-m", "unittest", "discover", "-s", "scripts/tests", env=env)
    verify()
    run("cargo", "fmt", "--all", "--check", env=env)
    for feature in (None, "qwen", "cosyvoice"):
        features = ("--features", feature) if feature is not None else ()
        run("cargo", "check", "-p", "speechfoundry", "--no-default-features",
            *features, "--locked", env=env)
    run("cargo", "clippy", "--workspace", "--all-targets", "--locked", "--", "-D", "warnings", env=env)
    excludes = [arg for package in BINDINGS for arg in ("--exclude", package)]
    run("cargo", "test", "--workspace", *excludes, "--locked", env=env)
    run("cargo", "doc", "--workspace", "--no-deps", "--locked", env=env)
    if device != "cpu":
        # This checks backend compilation. Real GPU evaluation has its own workflow.
        run("cargo", "clippy", "-p", "speechfoundry", "--all-targets", "--features", device,
            "--locked", "--", "-D", "warnings", env=env)


def python_binding() -> None:
    dist = ROOT / "target" / "binding-wheels"
    dist.mkdir(parents=True, exist_ok=True)
    for wheel in dist.glob("*.whl"):
        wheel.unlink()
    run(sys.executable, "-m", "maturin", "build", "--release", "--locked",
        "--manifest-path", "bindings/python/Cargo.toml", "--out", str(dist))
    wheels = list(dist.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"Expected exactly one wheel; found {len(wheels)}")
    run(sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheels[0]))
    run(sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "bindings/python/tests"), "-p", "*.py")


def node_binding() -> None:
    cwd = ROOT / "bindings/node"
    run("npm", "ci", cwd=cwd)
    run("npm", "run", "build", "--", "--release", cwd=cwd)
    run("npm", "test", cwd=cwd)


def build_wasm_package(destination: Path, *options: str) -> None:
    """Build in a fresh directory; wasm-pack treats existing package.json as dependencies."""
    cwd = ROOT / "bindings/wasm"
    if destination not in (cwd / "pkg", cwd / "pkg-web") or destination.is_symlink():
        raise ValueError("Expected an owned generated WASM output directory")
    with tempfile.TemporaryDirectory(prefix="speechfoundry-wasm-") as temporary:
        output = Path(temporary)
        run("wasm-pack", "build", "--out-dir", str(output), "--no-opt", *options, "--locked", cwd=cwd)
        stage("wasm", output, npm=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(output, destination)


def wasm_binding() -> None:
    cwd = ROOT / "bindings/wasm"
    env = strict_environment()
    run("rustup", "target", "add", "wasm32-unknown-unknown")
    run("cargo", "clippy", "-p", "speechfoundry-wasm", "--target", "wasm32-unknown-unknown",
        "--locked", "--", "-D", "warnings", env=env)
    run("cargo", "clippy", "-p", "speechfoundry-wasm", "--target", "wasm32-unknown-unknown",
        "--features", "qwen", "--locked", "--", "-D", "warnings", env=env)
    # Keep LLVM's release optimizations, without wasm-pack downloading a separate Binaryen tool.
    build_wasm_package(cwd / "pkg", "--target", "nodejs")
    run("node", "--test", "tests/smoke.cjs", cwd=cwd)
    build_wasm_package(cwd / "pkg", "--target", "nodejs", "--features", "qwen")
    run("node", "-e", "require('node:assert/strict').equal(typeof require('./pkg/speechfoundry_wasm.js').QwenSpeakerEncoder, 'function')", cwd=cwd)
    run("node", "--test", "tests/smoke.cjs", cwd=cwd)
    build_wasm_package(cwd / "pkg-web", "--target", "web", "--scope", "spensercai", "--features", "qwen")


def offline_reference(install_if_missing: bool) -> None:
    """Fail closed unless the kernel policy blocks sockets and survives child exec."""
    if not sys.platform.startswith("linux"):
        raise SystemExit("Offline reference isolation is supported only on Linux")
    library = ctypes.util.find_library("seccomp")
    if library is None:
        if not install_if_missing:
            raise SystemExit("The Linux CUDA runner must provide libseccomp (Debian/Ubuntu: libseccomp2) and permit seccomp filters")
        run("sudo", "apt-get", "update")
        run("sudo", "apt-get", "install", "--yes", "--no-install-recommends", "libseccomp2")
        library = ctypes.util.find_library("seccomp")
    if library is None:
        raise SystemExit("libseccomp is unavailable after installation; refusing to run model evaluation")
    ctypes.CDLL(library)
    probe = textwrap.dedent("""\
        import errno
        import socket
        for family in (socket.AF_INET, socket.AF_INET6, socket.AF_UNIX):
            try:
                connection = socket.socket(family, socket.SOCK_STREAM)
            except OSError as error:
                if error.errno != errno.EPERM:
                    raise RuntimeError(f'Expected EPERM for socket family {family}') from error
            else:
                connection.close()
                raise RuntimeError(f'Offline policy allowed socket family {family}')
        """)
    inherited = probe + f"\nimport subprocess, sys\nsubprocess.run([sys.executable, '-c', {probe!r}], check=True)\n"
    run(sys.executable, str(ROOT / "scripts/reference/offline_exec.py"), "--",
        sys.executable, "-c", inherited)
    print("Validated Linux socket denial for IPv4, IPv6 and Unix sockets, including child exec")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    rust = commands.add_parser("rust", help="Formatting, lint, test and rustdoc")
    rust.add_argument("--device", choices=("cpu", "cuda", "metal"), default="cpu")
    bindings = commands.add_parser("bindings", help="Build and import the actual language package")
    bindings.add_argument("--kind", choices=("python", "node", "wasm"), required=True)
    offline = commands.add_parser("offline-reference", help="Verify the Linux kernel model-network boundary")
    offline.add_argument("--install-if-missing", action="store_true",
                         help="Explicitly install libseccomp2 with apt on the hosted Ubuntu runner")
    args = parser.parse_args()
    check_version()
    if args.command == "rust":
        rust_checks(args.device)
    elif args.command == "offline-reference":
        offline_reference(args.install_if_missing)
    else:
        {"python": python_binding, "node": node_binding, "wasm": wasm_binding}[args.kind]()


if __name__ == "__main__":
    main()
