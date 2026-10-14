"""Export the pinned official Qwen speaker encoder, weights and forward tensors.

Synthetic fixtures use analytic, serialized weights, not an RNG seed as a claim
of parity. Pretrained fixtures use the real checkpoint's speaker encoder only.
These tests cover the speaker module, not waveform synthesis or speaker quality.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import subprocess
from pathlib import Path

from common import PINS, environment, offline_reference_runtime, revision, sha256, tensor_document, write_json


def export(output: Path, mode: str, weights: Path | None, device: str, frames: int) -> None:
    offline_reference_runtime()
    import torch
    from safetensors.torch import load_file, save_file
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder

    source_file = Path(inspect.getfile(Qwen3TTSSpeakerEncoder)).resolve()
    source_root = source_file.parents[3]
    actual_revision = revision(source_root)
    if actual_revision != PINS["qwen"]["revision"]:
        raise RuntimeError(f"Official source revision {actual_revision} does not match pins.json")
    if subprocess.check_output(["git", "-C", str(source_root), "diff", "HEAD", "--", str(source_file)]):
        raise RuntimeError("Official encoder source has uncommitted changes")
    if frames < 9:
        raise ValueError("At least 9 frames are required by dilated reflection padding")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; CPU fallback is forbidden for this test")
    if device == "metal" and not torch.backends.mps.is_available():
        raise RuntimeError("Metal requested but unavailable; CPU fallback is forbidden for this test")
    torch_device = {"cpu": "cpu", "cuda": "cuda", "metal": "mps"}[device]
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    if mode == "synthetic":
        config = Qwen3TTSSpeakerEncoderConfig(
            mel_dim=8, enc_dim=16, enc_channels=[8, 8, 8, 8, 24],
            enc_kernel_sizes=[5, 3, 3, 3, 1], enc_dilations=[1, 2, 3, 4, 1],
            enc_attention_channels=8, enc_res2net_scale=4, enc_se_channels=4,
        )
        model = Qwen3TTSSpeakerEncoder(config).eval()
        state = {}
        for index, (name, tensor) in enumerate(model.state_dict().items()):
            values = torch.arange(tensor.numel(), dtype=torch.float64)
            values = 0.07 * torch.sin(values * 0.173 + index * 0.619)
            state[name] = values.reshape(tensor.shape).to(torch.float32)
        model.load_state_dict(state, strict=True)
    else:
        if weights is None:
            from download_speaker import download
            weights = download(output / "download")
        download_metadata = json.loads((weights.parent / "download.json").read_text())
        if (download_metadata["model_revision"] != PINS["qwen"]["model_revision"]
                or download_metadata["subset_sha256"] != sha256(weights)):
            raise RuntimeError("Speaker weights differ from pinned download provenance")
        config_path = weights.parent / "model-config.json"
        config_values = json.loads(config_path.read_text())["speaker_encoder_config"]
        config = Qwen3TTSSpeakerEncoderConfig(**config_values)
        model = Qwen3TTSSpeakerEncoder(config).eval()
        state = {name.removeprefix("speaker_encoder."): value.float()
                 for name, value in load_file(str(weights), device="cpu").items()}
        model.load_state_dict(state, strict=True)
    config_fields = ["mel_dim", "enc_dim", "enc_channels", "enc_kernel_sizes", "enc_dilations",
                     "enc_attention_channels", "enc_res2net_scale", "enc_se_channels", "sample_rate"]
    write_json(output / "config.json", {name: getattr(config, name) for name in config_fields})
    state_path = output / "speaker.safetensors"
    save_file({name: value.contiguous() for name, value in model.state_dict().items()}, str(state_path))
    # Smooth but non-periodically identical channels/time axes exercise statistics.
    values = torch.arange(config.mel_dim * frames, dtype=torch.float64)
    mel = (0.5 * torch.sin(values * 0.071) + 0.2 * torch.cos(values * 0.113))
    mel = mel.reshape(1, config.mel_dim, frames).float()
    metadata = {
        **environment(), "model": "qwen3-tts-speaker-encoder", "fixture_kind": mode,
        "source_revision": actual_revision, "source_file_sha256": sha256(source_file),
        "weights_sha256": sha256(state_path), "torch": torch.__version__,
        "safetensors": importlib.metadata.version("safetensors"),
        "device": device, "dtype": "float32", "threads": 1,
        "scope": "mel -> ECAPA-TDNN -> speaker embedding; excludes acoustic frontend and synthesis",
        "model_revision": PINS["qwen"]["model_revision"] if mode == "pretrained" else None,
    }
    write_json(output / "input.json", tensor_document({"mel": mel}, metadata))
    intermediates = {}
    handles = []
    for name in ["blocks.0", "blocks.1", "blocks.2", "blocks.3", "mfa", "asp", "fc"]:
        module = model.get_submodule(name)
        handles.append(module.register_forward_hook(
            lambda _module, _inputs, result, key=name: intermediates.__setitem__(key, result.detach().cpu())
        ))
    with torch.inference_mode():
        model = model.to(torch_device)
        result = model(mel.transpose(1, 2).to(torch_device))
    for handle in handles:
        handle.remove()
    write_json(output / "reference.json", tensor_document({"speaker_embedding": result}, metadata))
    write_json(output / "intermediates.json", tensor_document(intermediates, metadata))
    write_json(output / "manifest.json", {
        **metadata, "artifacts": {name: sha256(output / name) for name in [
            "config.json", "speaker.safetensors", "input.json", "reference.json", "intermediates.json"
        ]},
    })
    print(f"Exported official {mode} speaker oracle to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "pretrained"], default="synthetic")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "metal"], default="cpu")
    parser.add_argument("--frames", type=int, default=41)
    args = parser.parse_args()
    export(args.output, args.mode, args.weights, args.device, args.frames)


if __name__ == "__main__":
    main()
