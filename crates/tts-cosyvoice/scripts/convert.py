#!/usr/bin/env python3
"""Convert the official CosyVoice 3 checkpoint without loading executable pickle objects.

Install the pinned upstream reference environment first. Example:
  python convert.py --input MODEL --output CONVERTED --source COSYVOICE_CHECKOUT
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


UPSTREAM_REVISION = "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"


def fuse_weight_norm(state):
    """Materialize dim=0 weight normalization used by the upstream HiFT model."""
    import torch

    converted = {}
    consumed = set()
    for key, gain in state.items():
        if key.endswith(".parametrizations.weight.original0"):
            stem = key.removesuffix(".parametrizations.weight.original0")
            direction_key = stem + ".parametrizations.weight.original1"
        elif key.endswith(".weight_g"):
            stem = key.removesuffix(".weight_g")
            direction_key = stem + ".weight_v"
        else:
            continue
        direction = state[direction_key].float()
        norm = torch.linalg.vector_norm(
            direction, dim=tuple(range(1, direction.ndim)), keepdim=True
        )
        if torch.any(norm == 0):
            raise ValueError(f"zero weight-norm direction: {direction_key}")
        converted[stem + ".weight"] = (gain.float() * direction / norm).contiguous()
        consumed.update((key, direction_key))
    for key, value in state.items():
        if key not in consumed:
            if ".parametrizations." in key or key.endswith((".weight_v", ".weight_g")):
                raise ValueError(f"unmatched weight parametrization: {key}")
            if key in converted:
                raise ValueError(f"duplicate materialized weight: {key}")
            converted[key] = value.float().clone().contiguous()
    return converted


def convert(args):
    import torch
    from safetensors.torch import save_file

    source = args.source.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"expected upstream {UPSTREAM_REVISION}, found {revision}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "upstream_revision": revision, "files": {}}
    for stage in ("llm", "flow", "hift"):
        source_path = args.input / f"{stage}.pt"
        state = torch.load(source_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise TypeError(f"expected a tensor state dictionary: {source_path}")
        if stage == "hift":
            converted = fuse_weight_norm(state)
        else:
            converted = {}
            for key, value in state.items():
                if stage == "llm":
                    if key.startswith("llm.model.model."):
                        key = "llm." + key.removeprefix("llm.model.model.")
                    elif key.startswith("llm.model."):
                        key = "llm." + key.removeprefix("llm.model.")
                elif key.startswith("decoder.estimator."):
                    key = "dit." + key.removeprefix("decoder.estimator.")
                if key in converted:
                    raise ValueError(f"weight name collision: {key}")
                converted[key] = value.float().clone().contiguous()
        target = args.output / f"{stage}.safetensors"
        save_file(converted, str(target), metadata={"source_revision": revision})
        manifest["files"][target.name] = {
            "tensors": len(converted),
            "source_sha256": sha256(source_path),
            "sha256": sha256(target),
        }
        del state, converted

    # The Python model creates this nonpersistent tensor after resetting seed=0.
    generator = torch.Generator(device="cpu").manual_seed(0)
    save_file({"noise": torch.randn(1, 80, 15000, generator=generator)},
              str(args.output / "flow-noise.safetensors"))

    # Import the versioned upstream tokenizer so every phoneme/control token and
    # Qwen pre-tokenization rule is preserved. Bare vocab/merges are insufficient.
    sys.path.insert(0, str(source))
    from cosyvoice.tokenizer.tokenizer import CosyVoice3Tokenizer

    tokenizer = CosyVoice3Tokenizer(str(args.input / "CosyVoice-BlankEN")).tokenizer
    if not tokenizer.is_fast:
        raise RuntimeError("a fast upstream tokenizer is required for tokenizer.json export")
    if tokenizer.convert_tokens_to_ids("<|endofprompt|>") != 151646:
        raise ValueError("unexpected CosyVoice 3 special-token mapping")
    tokenizer.backend_tokenizer.save(str(args.output / "tokenizer.json"))
    for name in ("campplus.onnx", "speech_tokenizer_v3.onnx"):
        if (args.input / name).is_file():
            shutil.copyfile(args.input / name, args.output / name)
    (args.output / "conversion.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
