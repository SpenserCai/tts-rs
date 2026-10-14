"""Create an isolated official-reference environment at pinned source revisions."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import venv
from pathlib import Path

from common import PINS, ROOT, revision, write_json


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def prepare(destination: Path, model: str, device: str) -> Path:
    pin = PINS.get(model)
    if not pin or not pin.get("revision"):
        raise ValueError(f"No reviewed, pinned official reference for {model}")
    destination = destination.resolve()
    environment_path = destination / "venv"
    python = environment_path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(environment_path)
    source = destination / model
    if not source.exists():
        run(["git", "clone", "--filter=blob:none", pin["repository"], str(source)])
    run(["git", "-C", str(source), "fetch", "origin", pin["revision"]])
    run(["git", "-C", str(source), "checkout", "--detach", pin["revision"]])
    if revision(source) != pin["revision"]:
        raise RuntimeError("Reference checkout does not match the pinned source revision")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    )
    if dirty:
        raise RuntimeError("Reference checkout is dirty; refuse an unreviewed oracle")
    torch_command = [str(python), "-m", "pip", "install",
                     f"torch=={PINS['torch']}", f"torchaudio=={PINS['torchaudio']}"]
    if device == "cpu":
        torch_command += ["--index-url", "https://download.pytorch.org/whl/cpu"]
    run(torch_command)
    if model == "qwen":
        run([str(python), "-m", "pip", "install", "-e", str(source),
             f"safetensors=={PINS['safetensors']}"])
    else:
        # DiT conditioning oracle imports a source module directly. Installing
        # unrelated inference backends is unnecessary for this scoped check.
        run([str(python), "-m", "pip", "install", f"safetensors=={PINS['safetensors']}",
             f"x-transformers=={PINS['x_transformers']}"])
    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
    write_json(destination / "environment.json", {
        "schema_version": 1, "model": model, "source": str(source),
        "revision": pin["revision"], "python": str(python),
        "requested_device": device, "pip_freeze": freeze.splitlines(),
        "note": "Top-level runtime/source pins; transitive dependency versions are recorded, not fully locked.",
    })
    return python


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["qwen", "cosyvoice"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "metal"], default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / ".reference")
    args = parser.parse_args()
    try:
        python = prepare(args.output, args.model, args.device)
        print(json.dumps({"python": str(python)}))
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
