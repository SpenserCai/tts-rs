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


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def prepare(destination: Path, model: str, device: str, pretrained: bool = False) -> Path:
    pin = PINS.get(model)
    if not pin or not pin.get("revision"):
        raise ValueError(f"No reviewed, pinned official reference for {model}")
    destination = destination.resolve()
    # Qwen and CosyVoice require different Transformers/Torch contracts. Never
    # mutate one family's oracle by installing the other into a shared venv.
    environment_path = destination / (model + "-venv")
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
    runtime = PINS["runtimes"][model]
    torch_command = [str(python), "-m", "pip", "install",
                     f"torch=={runtime['torch']}", f"torchaudio=={runtime['torchaudio']}"]
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
        if pretrained:
            run(["git", "-C", str(source), "submodule", "update", "--init", "--depth", "1", "third_party/Matcha-TTS"])
            if revision(source / "third_party/Matcha-TTS") != pin["matcha_revision"]:
                raise RuntimeError("Matcha source revision does not match the reviewed pin")
            # The pinned 2023 Whisper source build imports pkg_resources.
            # Propagate this constraint to pip's isolated build subprocess,
            # preserving the actual upstream package rather than upgrading it.
            build_constraint = Path(__file__).with_name("cosyvoice-build-constraints.txt").resolve()
            install_env = dict(os.environ, PIP_CONSTRAINT=str(build_constraint))
            run([str(python), "-m", "pip", "install", *PINS["cosyvoice_reference_packages"]],
                env=install_env)
    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
    write_json(destination / (model + "-environment.json"), {
        "schema_version": 1, "model": model, "source": str(source),
        "revision": pin["revision"], "python": str(python),
        "requested_device": device, "runtime_pins": runtime, "pip_freeze": freeze.splitlines(),
        "note": "Top-level runtime/source pins; transitive dependency versions are recorded, not fully locked.",
    })
    return python


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["qwen", "cosyvoice"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "metal"], default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / ".reference")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    try:
        python = prepare(args.output, args.model, args.device, args.pretrained)
        print(json.dumps({"python": str(python)}))
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
