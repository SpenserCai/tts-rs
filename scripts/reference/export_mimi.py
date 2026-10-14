"""Export official Qwen 12 Hz reference-encoder internals for numeric diagnosis."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

from common import PINS, environment, offline_reference_runtime, revision, sha256, tensor_document, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    offline_reference_runtime()
    import torch
    import transformers
    from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

    source = Path(inspect.getfile(Qwen3TTSTokenizer)).resolve().parents[2]
    if revision(source) != PINS["qwen"]["revision"]:
        raise RuntimeError("Qwen source revision differs from pin")
    torch.set_num_threads(1)
    tokenizer = Qwen3TTSTokenizer.from_pretrained(str(args.model), dtype=torch.float32,
                                                device_map="cpu", attn_implementation="eager")
    encoder = tokenizer.model.encoder
    tensors, handles = {}, []
    names = {
        "seanet": "encoder",
        "transformer": "encoder_transformer",
        "downsample": "downsample",
        "semantic_projection": "quantizer.semantic_residual_vector_quantizer.input_proj",
        "acoustic_projection": "quantizer.acoustic_residual_vector_quantizer.input_proj",
    }

    def capture(name, result):
        if hasattr(result, "last_hidden_state"):
            result = result.last_hidden_state
        elif isinstance(result, tuple):
            result = result[0]
        if name == "transformer":
            result = result.transpose(1, 2)
        tensors[name] = result.detach().cpu()

    for name, path in names.items():
        module = encoder.get_submodule(path)
        handles.append(module.register_forward_hook(
            lambda _module, _inputs, result, key=name: capture(key, result)
        ))
    with torch.inference_mode():
        result = tokenizer.encode(str(args.audio))
    for handle in handles:
        handle.remove()
    tensors["reference_audio_codes"] = result.audio_codes[0].to(torch.uint32)
    metadata = {
        **environment(), "model": "qwen3-tts-reference-encoder", "device": "cpu", "dtype": "float32",
        "source_revision": PINS["qwen"]["revision"], "model_revision": PINS["qwen"]["model_revision"],
        "torch": torch.__version__, "transformers": transformers.__version__,
        "weights_sha256": sha256(args.model / "model.safetensors"), "audio_sha256": sha256(args.audio),
        "scope": "official SEANet -> encoder transformer -> downsample -> semantic/acoustic RVQ projections",
    }
    projections = {name: value for name, value in tensors.items() if name.endswith("_projection")}
    main_tensors = {name: value for name, value in tensors.items() if name not in projections}
    write_json(args.output, tensor_document(main_tensors, metadata))
    write_json(args.output.with_name(args.output.stem + "-projections.json"), tensor_document(projections, metadata))
    print({name: list(tensor.shape) for name, tensor in tensors.items()})


if __name__ == "__main__":
    main()
