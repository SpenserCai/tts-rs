"""Run real official Qwen Base x-vector synthesis and export tokens and waveform."""

from __future__ import annotations

import argparse
import inspect
import time
from pathlib import Path

from common import PINS, ROOT, environment, offline_reference_runtime, revision, sha256, tensor_document, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--reference-text")
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    offline_reference_runtime()

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    if (args.output.resolve() == ROOT or ROOT in args.output.resolve().parents) and (ROOT / "target") not in args.output.resolve().parents:
        raise ValueError("Synthesis evaluation containing voice artifacts must be outside the source repository")
    source = Path(inspect.getfile(Qwen3TTSModel)).resolve().parents[2]
    if revision(source) != PINS["qwen"]["revision"]:
        raise RuntimeError("Official source revision does not match the pin")
    model_sha = sha256(args.model / "model.safetensors")
    metadata_path = args.model / ".cache" / "huggingface" / "download" / "model.safetensors.metadata"
    download_metadata = metadata_path.read_text().splitlines()
    if download_metadata[:2] != [PINS["qwen"]["model_revision"], model_sha]:
        raise RuntimeError("Model revision or bytes differ from the pinned Hugging Face download metadata")
    torch.set_num_threads(1)
    started = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        str(args.model), device_map="cpu", dtype=torch.float32, attn_implementation="eager"
    )
    load_seconds = time.perf_counter() - started
    captures = {}
    original_generate = model.model.generate

    def capture_generate(*positional, **keyword):
        result = original_generate(*positional, **keyword)
        captures["audio_codes"] = result[0][0].to(torch.uint32)
        return result

    model.model.generate = capture_generate
    started = time.perf_counter()
    prompt = model.create_voice_clone_prompt(
        ref_audio=str(args.audio), ref_text=args.reference_text,
        x_vector_only_mode=args.reference_text is None,
    )
    wavs, sample_rate = model.generate_voice_clone(
        text=args.text, language=args.language, voice_clone_prompt=prompt,
        do_sample=False, subtalker_dosample=False, max_new_tokens=args.max_frames,
        min_new_tokens=2, repetition_penalty=1.05,
    )
    inference_seconds = time.perf_counter() - started
    tensors = {
        "speaker_embedding": prompt[0].ref_spk_embedding.float(),
        "audio_codes": captures["audio_codes"],
        "waveform": torch.from_numpy(wavs[0]).float(),
    }
    metadata = {
        **environment(), "implementation": "official-qwen-python", "source_revision": revision(source),
        "model_revision": PINS["qwen"]["model_revision"], "audio_sha256": sha256(args.audio),
        "model_weights_sha256": model_sha,
        "sample_rate": sample_rate, "device": "cpu", "dtype": "float32", "torch": torch.__version__,
        "threads": 1, "max_frames": args.max_frames, "min_new_tokens": 2, "repetition_penalty": 1.05,
        "decoding": "greedy main and subtalker; no RNG equivalence claim",
        "conditioning": "x_vector_only" if args.reference_text is None else "icl", "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "text_sha256": __import__("hashlib").sha256(args.text.encode()).hexdigest(),
    }
    write_json(args.output, tensor_document(tensors, metadata))
    if prompt[0].ref_code is not None:
        write_json(args.output.with_name(args.output.stem + "-reference-codes.json"),
                   tensor_document({"reference_audio_codes": prompt[0].ref_code.to(torch.uint32)}, metadata))
    sf.write(args.output.with_suffix(".wav"), wavs[0], sample_rate, subtype="PCM_16")
    print(f"Official synthesis produced {len(wavs[0])} samples at {sample_rate} Hz")


if __name__ == "__main__":
    main()
