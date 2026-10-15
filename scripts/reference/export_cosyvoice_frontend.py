"""Export pinned CosyVoice frontend features from public deterministic PCM.

Run with the prepared CosyVoice Python environment under offline_exec.py.
No recording or model weights are needed unless --models is specified.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

from common import PINS, reference_runtime, revision, sha256, tensor_document, write_json


def analytic_pcm(length: int) -> list[float]:
    """Exact binary fractions from an integer LCG; portable to the Rust test."""
    state = 0x13579BDF
    samples = []
    for _ in range(length):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        samples.append(((state >> 8) - 8388608) / 33554432)
    return samples


def export(source: Path, output: Path, length: int, models: Path | None) -> None:
    reference_runtime("cosyvoice")
    source = source.resolve()
    matcha = source / "third_party/Matcha-TTS"
    if revision(source) != PINS["cosyvoice"]["revision"] or revision(matcha) != PINS["cosyvoice"]["matcha_revision"]:
        raise RuntimeError("CosyVoice/Matcha source revision does not match pins")
    sys.path.insert(0, str(matcha))
    import torch
    import torchaudio
    import torchaudio.compliance.kaldi as kaldi
    import whisper
    from matcha.utils.audio import mel_spectrogram

    torch.set_num_threads(1)
    samples = analytic_pcm(length)
    pcm = torch.tensor(samples, dtype=torch.float32)
    pcm_16k = torchaudio.transforms.Resample(24000, 16000)(pcm)
    fbank = kaldi.fbank(pcm_16k.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
    normalized = fbank - fbank.mean(0)
    tensors = {
        "pcm_16k": pcm_16k,
        "fbank": fbank,
        "normalized": normalized,
        "mel": mel_spectrogram(pcm.unsqueeze(0), 1920, 80, 24000, 480, 1920, 0, None, False),
        "whisper": whisper.log_mel_spectrogram(pcm_16k.unsqueeze(0), n_mels=128),
    }
    metadata = {
        "source_revision": PINS["cosyvoice"]["revision"],
        "matcha_revision": PINS["cosyvoice"]["matcha_revision"],
        "torch": torch.__version__, "dtype": "float32", "device": "cpu", "threads": 1,
        "input": {
            "kind": "lcg32", "seed": 0x13579BDF, "samples": length, "sample_rate": 24000,
            "encoding": "float32 little-endian mono PCM",
            "sha256": hashlib.sha256(pcm.numpy().astype("<f4", copy=False).tobytes()).hexdigest(),
        },
        "scope": "public PCM resampling, Kaldi Fbank, Matcha and Whisper mel; no synthesis",
    }
    if models is not None:
        import onnxruntime as ort
        if ort.__version__ != "1.18.0":
            raise RuntimeError("CAMPPlus oracle requires the pinned official ONNX Runtime 1.18.0")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        checkpoint = models / "campplus.onnx"
        session = ort.InferenceSession(str(checkpoint), options, providers=["CPUExecutionProvider"])
        vector = session.run(None, {session.get_inputs()[0].name: normalized.unsqueeze(0).numpy()})[0]
        tensors["speaker_embedding"] = torch.from_numpy(vector)
        metadata.update({"onnxruntime": ort.__version__, "campplus_sha256": sha256(checkpoint)})
    write_json(output, tensor_document(tensors, metadata))
    write_json(output.with_name(output.stem + "-input.json"), {
        "samples": samples, "sample_rate": 24000, "transcript": "Public numerical fixture.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2401)
    parser.add_argument("--models", type=Path)
    args = parser.parse_args()
    if not 2401 <= args.samples <= 720000:
        parser.error("--samples must be within 2401..720000")
    export(args.source, args.output, args.samples, args.models)
