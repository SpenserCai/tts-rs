"""Extract a raw Qwen speaker conditioning vector outside the source repository.

The vector is not a trained/cloned TTS model or a complete ICL voice-clone
prompt: reference codec tokens and a transcript are separate conditioning data.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from common import PINS, ROOT, environment, offline_reference_runtime, revision, sha256, tensor_document, write_json


def extract(audio: Path, weights: Path, output: Path) -> None:
    offline_reference_runtime()
    import numpy as np
    import soundfile as sf
    import torch
    from safetensors.torch import load_file, save_file
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder, mel_spectrogram

    output = output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Personal voice artifacts must be written outside the source repository")
    source = Path(inspect.getfile(Qwen3TTSSpeakerEncoder)).resolve()
    if revision(source.parents[3]) != PINS["qwen"]["revision"]:
        raise RuntimeError("Official Python source revision does not match pin")
    pcm, sample_rate = sf.read(audio, dtype="float32", always_2d=True)
    if sample_rate != 24000 or pcm.shape[1] != 1:
        raise ValueError("Provide preprocessed 24 kHz mono audio; this tool does not silently resample")
    if len(pcm) < 1024 or not np.isfinite(pcm).all():
        raise ValueError("Audio must contain at least 1024 finite samples")
    values = json.loads((weights.parent / "model-config.json").read_text())["speaker_encoder_config"]
    download_metadata = json.loads((weights.parent / "download.json").read_text())
    if (download_metadata["model_revision"] != PINS["qwen"]["model_revision"]
            or download_metadata["subset_sha256"] != sha256(weights)):
        raise RuntimeError("Speaker weights differ from pinned download provenance")
    config = Qwen3TTSSpeakerEncoderConfig(**values)
    model = Qwen3TTSSpeakerEncoder(config).float().eval()
    model.load_state_dict({key.removeprefix("speaker_encoder."): value.float()
                           for key, value in load_file(str(weights)).items()}, strict=True)
    torch.set_num_threads(1)
    with torch.inference_mode():
        mel = mel_spectrogram(torch.from_numpy(pcm[:, 0]).unsqueeze(0), n_fft=1024,
                              num_mels=128, sampling_rate=24000, hop_size=256,
                              win_size=1024, fmin=0, fmax=12000)
        vector = model(mel.transpose(1, 2)).cpu().contiguous()
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "speaker-embedding.npy", vector[0].numpy(), allow_pickle=False)
    save_file({"speaker_embedding": vector[0]}, str(output / "speaker-embedding.safetensors"))
    metadata = {
        **environment(), "schema_version": 1, "model_id": PINS["qwen"]["model_id"],
        "model_revision": PINS["qwen"]["model_revision"], "source_revision": PINS["qwen"]["revision"],
        "audio_sha256": sha256(audio), "weights_sha256": sha256(weights),
        "sample_rate": sample_rate, "duration_seconds": len(pcm) / sample_rate,
        "embedding_dim": config.enc_dim, "dtype": "float32", "device": "cpu",
        "normalization": "none; raw official speaker encoder output",
        "conditioning_mode": "x_vector_only", "torch": torch.__version__,
        "limitations": ["Not a trained voice model", "No reference codec tokens or transcript included",
                        "No perceptual voice-clone quality claim", "Frontend uses the official Python implementation"],
    }
    write_json(output / "manifest.json", metadata)
    write_json(output / "input.json", tensor_document({"mel": mel}, metadata))
    write_json(output / "reference.json", tensor_document({"speaker_embedding": vector}, metadata))
    # Save only dimensions, not private reference input, in stdout.
    print(f"Extracted {config.enc_dim}-dimensional raw speaker conditioning vector")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract(args.audio, args.weights, args.output)
