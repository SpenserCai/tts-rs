#!/usr/bin/env python3
"""Install the verified Linux x64 wasm-pack binary used by the manual Actions jobs."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import urllib.request

VERSION = "0.15.0"
DIRECTORY = f"wasm-pack-v{VERSION}-x86_64-unknown-linux-musl"
ASSET = f"{DIRECTORY}.tar.gz"
SHA256 = "c09f971ecaed9a2efc80fdcea7a00ef6b53c7fadc8c57d1f61b53a6aa66b668a"
URL = f"https://github.com/wasm-bindgen/wasm-pack/releases/download/v{VERSION}/{ASSET}"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    if sys.version_info < (3, 12):
        raise SystemExit("This repository setup utility requires Python 3.12 or newer")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".cache/wasm-pack"))
    parser.add_argument("--github-path", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise SystemExit("This CI utility supports Linux x64; install wasm-pack normally on other hosts")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / ASSET
    if not archive_path.exists():
        temporary = archive_path.with_suffix(".part")
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        if digest(temporary) != SHA256:
            temporary.unlink()
            raise SystemExit("Downloaded wasm-pack archive SHA-256 mismatch")
        temporary.replace(archive_path)
    if digest(archive_path) != SHA256:
        raise SystemExit("Cached wasm-pack archive SHA-256 mismatch; remove it and retry")
    with tarfile.open(archive_path) as archive:
        archive.extractall(output, filter="data")
    binary = output / DIRECTORY / "wasm-pack"
    actual = subprocess.check_output([str(binary), "--version"], text=True).strip()
    if actual != f"wasm-pack {VERSION}":
        raise SystemExit(f"Unexpected wasm-pack version: {actual}")
    if args.github_path:
        path_file = os.environ.get("GITHUB_PATH")
        if not path_file:
            raise SystemExit("--github-path requires GITHUB_PATH")
        with open(path_file, "a", encoding="utf-8") as stream:
            stream.write(f"{binary.parent}\n")
    print(binary)


if __name__ == "__main__":
    main()
