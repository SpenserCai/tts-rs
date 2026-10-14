"""Run both implementations from an explicit manifest and compare real outputs.

Commands are argument arrays, never shell strings. Each implementation must
write tensor JSON with a nonempty waveform tensor. The manifest binds input
files, checkpoints and reference revision by SHA-256/revision, and must declare
how stochastic draws/tokens are shared; equal seeds alone are insufficient.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from common import PINS, environment, model_command, sha256, write_json
from compare_tensors import compare, read_document


def run(manifest_path: Path, output: Path, expected_device: str = "cpu", offline: bool = False) -> dict:
    manifest = json.loads(manifest_path.read_text())
    model = manifest["model"]
    if manifest.get("device") != expected_device:
        raise ValueError("Manifest device does not match the requested evaluation device")
    if model not in PINS or manifest.get("source_revision") != PINS[model]["revision"]:
        raise ValueError("E2E source revision must match the reviewed pin")
    if manifest.get("schema_version") != 1 or not manifest.get("inputs"):
        raise ValueError("E2E manifest requires schema_version=1 and hashed input files")
    if not manifest.get("stochastic_inputs") or manifest["stochastic_inputs"] == "same_seed":
        raise ValueError("Declare shared tokens/noise or a deterministic decoding strategy; same seed is insufficient")
    for item in manifest["inputs"]:
        path = Path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Input hash mismatch: {path}")
    output.mkdir(parents=True, exist_ok=True)
    for sidecar in manifest.get("sidecars", []):
        for implementation in ["reference", "candidate"]:
            if Path(sidecar[implementation]).exists():
                raise ValueError("Refuse stale E2E sidecar; choose a fresh output directory")
    results = {}
    for implementation in ["reference", "candidate"]:
        command = manifest[implementation]["command"]
        if not isinstance(command, list) or not command or not all(isinstance(v, str) for v in command):
            raise ValueError("Each command must be a nonempty array of argument strings")
        tensor_path = Path(manifest[implementation]["output"])
        if tensor_path.exists():
            raise ValueError(f"Refuse stale output {tensor_path}; choose a fresh path")
        with (output / f"{implementation}.log").open("w") as log:
            subprocess.run(model_command(command, offline), check=True, stdout=log, stderr=subprocess.STDOUT)
        document = read_document(tensor_path)
        if document.get("metadata", {}).get("device") != expected_device:
            raise ValueError(f"{implementation} did not run on the requested device")
        if "waveform" not in document["tensors"]:
            raise ValueError(f"{implementation} did not produce a waveform tensor")
        sample_rate = document.get("metadata", {}).get("sample_rate")
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ValueError(f"{implementation} output needs a positive sample_rate")
        results[implementation] = (tensor_path, sample_rate)
    if results["reference"][1] != results["candidate"][1]:
        raise ValueError("Generated sample rates differ")
    report = compare(results["reference"][0], results["candidate"][0],
                     float(manifest["atol"]), float(manifest["rtol"]))
    report.update(environment=environment(), manifest_sha256=sha256(manifest_path),
                  source_revision=manifest["source_revision"], sample_rate=results["reference"][1],
                  offline_reference=offline,
                  scope="strict numerical end-to-end waveform comparison")
    report["sidecars"] = {}
    for sidecar in manifest.get("sidecars", []):
        sidecar_report = compare(Path(sidecar["reference"]), Path(sidecar["candidate"]),
                                 float(manifest["atol"]), float(manifest["rtol"]))
        report["sidecars"][sidecar["name"]] = sidecar_report
        report["passed"] = report["passed"] and sidecar_report["passed"]
    write_json(output / "e2e.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "metal"], default="cpu")
    parser.add_argument("--offline-reference", action="store_true")
    args = parser.parse_args()
    try:
        report = run(args.manifest, args.output, args.device, args.offline_reference)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        write_json(args.output / "e2e.json", {"schema_version": 1, "passed": False, "error": str(error)})
        print(str(error))
        return 1
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
