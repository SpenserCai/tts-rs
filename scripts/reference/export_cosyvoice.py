"""Export official CosyVoice DiT conditioning modules with tiny deterministic weights."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
from pathlib import Path

from common import PINS, environment, revision, sha256, tensor_document, write_json


def export(source: Path, output: Path, device: str = "cpu") -> None:
    import torch
    from safetensors.torch import save_file

    source = source.resolve()
    if revision(source) != PINS["cosyvoice"]["revision"]:
        raise RuntimeError("CosyVoice official source revision does not match pin")
    module_path = source / "cosyvoice" / "flow" / "DiT" / "modules.py"
    if subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--", str(module_path)]):
        raise RuntimeError("Official module source has uncommitted changes")
    spec = importlib.util.spec_from_file_location("official_cosyvoice_dit_modules", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load official DiT source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch_device = {"cpu": "cpu", "cuda": "cuda", "metal": "mps"}[device]
    torch.set_num_threads(1)
    dim = 16
    modules = {
        "time": module.TimestepEmbedding(dim),
        "ada": module.AdaLayerNormZero(dim),
        "final": module.AdaLayerNormZero_Final(dim),
    }
    weights = {}
    for module_index, (prefix, model) in enumerate(modules.items()):
        state = {}
        for index, (key, value) in enumerate(model.state_dict().items()):
            values = torch.arange(value.numel(), dtype=torch.float64)
            state[key] = (0.04 * torch.sin(values * 0.133 + index * 0.41 + module_index)).reshape(value.shape).float()
        model.load_state_dict(state, strict=True)
        model.to(torch_device).eval()
        weights.update({prefix + "." + key: value.contiguous() for key, value in state.items()})
    time = torch.tensor([0.0, 0.37, 1.0], dtype=torch.float32)
    x = torch.sin(torch.arange(3 * 5 * dim, dtype=torch.float32) * 0.05).reshape(3, 5, dim)
    emb = torch.cos(torch.arange(3 * dim, dtype=torch.float32) * 0.11).reshape(3, dim)
    with torch.inference_mode():
        time_result = modules["time"](time.to(torch_device))
        ada = modules["ada"](x.to(torch_device), emb.to(torch_device))
        final = modules["final"](x.to(torch_device), emb.to(torch_device))
    output.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(output / "conditioning.safetensors"))
    metadata = {
        **environment(), "model": "cosyvoice3-dit-conditioning", "fixture_kind": "synthetic",
        "source_revision": PINS["cosyvoice"]["revision"], "source_file_sha256": sha256(module_path),
        "weights_sha256": sha256(output / "conditioning.safetensors"),
        "torch": torch.__version__, "dtype": "float32", "device": device, "threads": 1,
        "scope": "TimestepEmbedding, AdaLayerNormZero and AdaLayerNormZero_Final; excludes full DiT/flow/vocoder",
    }
    write_json(output / "config.json", {"dim": dim})
    write_json(output / "input.json", tensor_document({"time": time, "x": x, "emb": emb}, metadata))
    results = {"time": time_result, "final": final}
    results.update(dict(zip(["ada.x", "ada.gate_msa", "ada.shift_mlp", "ada.scale_mlp", "ada.gate_mlp"], ada)))
    write_json(output / "reference.json", tensor_document(results, metadata))
    write_json(output / "manifest.json", metadata)
    print(f"Exported official CosyVoice conditioning oracle to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "metal"], default="cpu")
    args = parser.parse_args()
    export(args.source, args.output, args.device)
