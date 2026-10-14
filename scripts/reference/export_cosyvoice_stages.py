"""Export a real official CosyVoice HiFT oracle with shared source noise tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import PINS, environment, offline_reference_runtime, reference_runtime, revision, sha256, tensor_document, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--prompt-features", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    if revision(source) != PINS["cosyvoice"]["revision"]:
        raise RuntimeError("CosyVoice source revision differs from the reviewed pin")
    sys.path.insert(0, str(source))
    runtime = reference_runtime("cosyvoice")
    offline_reference_runtime()
    import torch
    from safetensors.torch import save_file
    from cosyvoice.hifigan.generator import CausalHiFTGenerator
    from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor

    torch.set_num_threads(1)
    prompt = json.loads(args.prompt_features.read_text())
    if args.frames < 9 or args.frames > len(prompt["mel"]):
        raise ValueError("Requested frames must be between 9 and available prompt frames")
    mel = torch.tensor(prompt["mel"][:args.frames], dtype=torch.float32).T.unsqueeze(0).contiguous()
    model = CausalHiFTGenerator(
        in_channels=80, base_channels=512, nb_harmonics=8, sampling_rate=24000,
        nsf_alpha=0.1, nsf_sigma=0.003, nsf_voiced_threshold=10,
        upsample_rates=[8, 5, 3], upsample_kernel_sizes=[16, 11, 7],
        istft_params={"n_fft": 16, "hop_len": 4}, resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5]] * 3, source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5]] * 3, lrelu_slope=0.1,
        audio_limit=0.99, conv_pre_look_right=4,
        f0_predictor=CausalConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512),
    ).eval()
    state = torch.load(args.weights, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict({key.removeprefix("generator."): value for key, value in state.items()}, strict=True)
    intermediates = {}
    handle = model.f0_predictor.register_forward_hook(
        lambda _module, _inputs, result: intermediates.__setitem__("f0", result.detach().cpu())
    )
    with torch.inference_mode():
        waveform, source_signal = model.inference(mel, finalize=True)
    handle.remove()
    intermediates["source"] = source_signal
    samples = source_signal.shape[-1]
    noise = {
        "phase": model.m_source.l_sin_gen.rand_ini.contiguous(),
        "sine": model.m_source.l_sin_gen.sine_waves[:, :samples].contiguous(),
        "uv": model.m_source.uv[:, :samples].contiguous(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(noise, str(args.output / "source-noise.safetensors"))
    metadata = {
        **environment(), "model": "cosyvoice3-hift", "source_revision": PINS["cosyvoice"]["revision"],
        "source_file_sha256": sha256(source / "cosyvoice/hifigan/generator.py"),
        "weights_sha256": sha256(args.weights), "prompt_features_sha256": sha256(args.prompt_features),
        "source_noise_sha256": sha256(args.output / "source-noise.safetensors"),
        "device": "cpu", "dtype": "float32", "f0_dtype": "float64", "torch": torch.__version__,
        "runtime": runtime,
        "sample_rate": 24000, "scope": "official HiFT waveform decoder from fixed mel; excludes LLM and flow",
        "randomness": "actual PyTorch phase, sine and UV buffers exported; no shared-seed equivalence claim",
    }
    write_json(args.output / "input.json", tensor_document({"mel": mel}, metadata))
    write_json(args.output / "reference.json", tensor_document({"waveform": waveform}, metadata))
    write_json(args.output / "intermediates.json", tensor_document(intermediates, metadata))
    write_json(args.output / "manifest.json", metadata)
    print(f"Exported official HiFT: {mel.shape[-1]} mel frames, {waveform.numel()} audio samples")


if __name__ == "__main__":
    main()
