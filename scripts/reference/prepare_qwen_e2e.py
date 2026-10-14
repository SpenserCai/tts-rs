"""Build a reproducible, public-data-free Qwen synthesis numerical smoke test.

The analytic tone is conditioning data for a numerical smoke test. It is not
speech and this check makes no claim about voice identity or perceptual quality.
"""

from __future__ import annotations

import math
import os
import struct
import subprocess
import sys
import wave
from pathlib import Path

from common import PINS, ROOT, sha256, write_json


def prepare(output: Path, model_dir: Path | None = None) -> Path:
    from huggingface_hub import snapshot_download

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if model_dir is None:
        cache_root = Path(os.environ.get("HF_HOME", str(ROOT / ".reference" / "models")))
        model_dir = cache_root / "oracle-snapshots" / "qwen3-tts-0.6b-base"
        snapshot_download(PINS["qwen"]["model_id"], revision=PINS["qwen"]["model_revision"],
                          local_dir=model_dir, allow_patterns=["*.json", "*.txt", "*.safetensors", "speech_tokenizer/*"])
    audio = output / "conditioning-tone.wav"
    sample_rate = 24000
    pcm = []
    for index in range(sample_rate):
        time = index / sample_rate
        envelope = math.sin(math.pi * time) ** 2
        value = envelope * (0.16 * math.sin(2 * math.pi * 220 * time) + 0.05 * math.sin(2 * math.pi * 351 * time))
        pcm.append(round(value * 32767))
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(struct.pack(f"<{len(pcm)}h", *pcm))
    subprocess.run(["cargo", "build", "--locked", "--release", "-p", "tts-qwen", "--example", "synthesis_oracle"],
                   cwd=ROOT, check=True)
    target = Path(os.environ.get("CARGO_TARGET_DIR", str(ROOT / "target")))
    binary = target / "release" / "examples" / ("synthesis_oracle.exe" if os.name == "nt" else "synthesis_oracle")
    text = "Hello."
    manifest = {
        "schema_version": 1, "model": "qwen", "device": "cpu", "source_revision": PINS["qwen"]["revision"],
        "model_revision": PINS["qwen"]["model_revision"],
        "scope": "CPU float32 full synthesis with analytic non-speech x-vector conditioning; no quality claim",
        "inputs": [{"path": str(path), "sha256": sha256(path)}
                   for path in [audio, model_dir / "model.safetensors", model_dir / "speech_tokenizer/model.safetensors"]],
        "stochastic_inputs": "greedy main and subtalker, deterministic decoder; no seed equivalence assumption",
        "atol": 1e-5, "rtol": 1e-4,
        "reference": {
            "command": [sys.executable, str(Path(__file__).with_name("export_qwen_synthesis.py")),
                        "--model", str(model_dir), "--audio", str(audio), "--text", text,
                        "--language", "English", "--max-frames", "20", "--output", str(output / "python.json")],
            "output": str(output / "python.json"),
        },
        "candidate": {
            "command": [str(binary), str(model_dir), str(audio), text, str(output / "rust.json"), "20", "english"],
            "output": str(output / "rust.json"),
        },
    }
    path = output / "e2e-manifest.json"
    write_json(path, manifest)
    return path
