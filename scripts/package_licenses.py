#!/usr/bin/env python3
"""Keep source provenance in language packages synchronized with the canonical sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("python", "node", "wasm")


def sources(kind: str) -> dict[str, Path]:
    files = {"LICENSE": ROOT / "LICENSE"}
    models = ("qwen",) if kind == "wasm" else ("qwen", "cosyvoice")
    for model in models:
        for filename in ("LICENSE-MIT", "UPSTREAM.md"):
            files[f"{model.upper()}-{filename}"] = ROOT / "crates" / f"tts-{model}" / filename
    if "cosyvoice" in models:
        files["COSYVOICE-LICENSE-PYTORCH"] = ROOT / "crates/tts-cosyvoice/LICENSE-PYTORCH"
    return files


def stage(kind: str, directory: Path, *, npm: bool = False) -> None:
    destination = directory / "licenses"
    destination.mkdir(parents=True, exist_ok=True)
    for name, source in sources(kind).items():
        shutil.copyfile(source, destination / name)
    if npm:
        path = directory / "package.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        files = manifest.setdefault("files", [])
        if "licenses" not in files:
            files.append("licenses")
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def verify() -> None:
    for kind in KINDS:
        directory = ROOT / "bindings" / kind / "licenses"
        for name, source in sources(kind).items():
            copied = directory / name
            if not copied.is_file() or copied.read_bytes() != source.read_bytes():
                raise SystemExit(f"Stale package license or provenance: {copied}; run python scripts/package_licenses.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        verify()
    else:
        for kind in KINDS:
            stage(kind, ROOT / "bindings" / kind)


if __name__ == "__main__":
    main()
