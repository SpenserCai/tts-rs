"""Reproducible pretrained stage gates, with serialized stochastic inputs."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

from common import PINS, ROOT, model_command, pinned_checkpoint_sha256, sha256, write_json
from compare_tensors import compare


def analytic_features(path: Path) -> None:
    """No recording or personal voice is used by the automatic fixture."""
    write_json(path, {
        "speech_tokens": [(index * 137 + 19) % 6561 for index in range(32)],
        "mel": [[-4.0 + math.sin(frame * 0.17 + channel * 0.11)
                 for channel in range(80)] for frame in range(64)],
        "speaker_embedding": [math.sin(index * 0.07) / math.sqrt(192) for index in range(192)],
        "transcript": "This is a numerical test.",
        "provenance": "analytic non-speech conditioning; no perceptual or voice-identity claim",
    })


def run(output: Path, source: Path, model: Path | None, converted: Path | None,
        features: Path | None, component: str, offline: bool = False) -> dict:
    from huggingface_hub import snapshot_download

    output, source = output.resolve(), source.resolve()
    output.mkdir(parents=True, exist_ok=True)
    components = ["llm", "flow", "hift"] if component == "all" else [component]
    if model is None:
        model = Path(os.environ.get("HF_HOME", str(ROOT / ".reference/models"))) / "oracle-snapshots/cosyvoice3"
        patterns = [stage + ".pt" for stage in components]
        if "llm" in components:
            patterns += ["CosyVoice-BlankEN/*.json", "CosyVoice-BlankEN/*.txt"]
        snapshot_download(PINS["cosyvoice"]["model_id"], revision=PINS["cosyvoice"]["model_revision"],
                          local_dir=model, allow_patterns=patterns)
    model = model.resolve()
    if converted is None:
        converted = model.with_name(model.name + "-candle")
        conversion = converted / "conversion.json"
        existing = json.loads(conversion.read_text())["files"] if conversion.exists() else {}
        missing = [stage for stage in components if stage + ".safetensors" not in existing]
        if missing:
            command = [sys.executable, str(ROOT / "crates/tts-cosyvoice/scripts/convert.py"),
                            "--source", str(source), "--input", str(model), "--output", str(converted),
                            "--stages", *missing]
            subprocess.run(model_command(command, offline), check=True)
    converted = converted.resolve()
    conversion = json.loads((converted / "conversion.json").read_text())
    if conversion["upstream_revision"] != PINS["cosyvoice"]["revision"]:
        raise ValueError("Converted checkpoint source revision mismatch")
    for stage in components:
        record = conversion["files"][stage + ".safetensors"]
        if pinned_checkpoint_sha256("cosyvoice", model / (stage + ".pt")) != record["source_sha256"]:
            raise ValueError(f"Original checkpoint hash mismatch: {stage}")
        if sha256(converted / (stage + ".safetensors")) != record["sha256"]:
            raise ValueError(f"Converted checkpoint hash mismatch: {stage}")
    if features is None:
        features = output / "analytic-features.json"
        analytic_features(features)
    features = features.resolve()
    target = Path(os.environ.get("CARGO_TARGET_DIR", str(ROOT / "target")))
    if not target.is_absolute():
        target = ROOT / target
    reports = {}
    for stage in components:
        directory = output / stage
        if (directory / "reference.json").exists() or (directory / "candidate.json").exists():
            raise ValueError("Refuse stale stage outputs; select a fresh evaluation directory")
        directory.mkdir(parents=True, exist_ok=True)
        script = {"llm": "export_cosyvoice_llm.py", "flow": "export_cosyvoice_flow.py",
                  "hift": "export_cosyvoice_stages.py"}[stage]
        command = [sys.executable, str(Path(__file__).with_name(script)), "--source", str(source),
                   "--prompt-features", str(features), "--output", str(directory)]
        command += ["--model", str(model)] if stage == "llm" else ["--weights", str(model / (stage + ".pt"))]
        with (directory / "reference.log").open("w") as log:
            subprocess.run(model_command(command, offline), stdout=log, stderr=subprocess.STDOUT, check=True)
        example = stage + "_oracle"
        subprocess.run(["cargo", "build", "--release", "--locked", "-p", "tts-cosyvoice", "--example", example],
                       cwd=ROOT, check=True)
        binary = target / "release/examples" / (example + (".exe" if os.name == "nt" else ""))
        command = [str(binary), str(converted), str(directory / "input.json")]
        if stage != "llm":
            command += [str(directory / ("flow-noise.safetensors" if stage == "flow" else "source-noise.safetensors"))]
        command += [str(directory / "candidate.json")]
        with (directory / "candidate.log").open("w") as log:
            subprocess.run(model_command(command, offline), stdout=log, stderr=subprocess.STDOUT, check=True)
        report = compare(directory / "reference.json", directory / "candidate.json", 1e-5, 1e-4)
        write_json(directory / "parity.json", report)
        reports[stage] = report
    report = {"schema_version": 1, "passed": all(value["passed"] for value in reports.values()),
              "device": "cpu", "scope": "separate pretrained stage gates; not full chained waveform parity",
              "components": reports, "requested_components": components,
              "offline_reference": offline,
              "source_revision": PINS["cosyvoice"]["revision"], "input_sha256": sha256(features)}
    write_json(output / "parity.json", report)
    return report
