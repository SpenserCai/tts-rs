"""Benchmark real Qwen CPU synthesis with a resident model and public analytic conditioning.

The deterministic tone exercises the full synthesis path; it is not speech or a
voice-quality test. RTF uses measured output duration, not a caller's estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tomllib
from pathlib import Path

from benchmark import percentile
from common import PINS, ROOT, environment, revision, sha256, write_json
from prepare_qwen_e2e import write_conditioning_tone


def version(command: list[str]) -> str:
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.runs <= 100 or not 0 <= args.warmups <= 20 or args.threads < 1:
        parser.error("runs must be 1..100, warmups 0..20, and threads positive")
    if not 2 <= args.max_new_tokens <= 4096:
        parser.error("max-new-tokens must be 2..4096")
    output = args.output.resolve()
    model = args.model_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "passed": False, "environment": environment(),
              "scope": "real resident-model Qwen CPU synthesis throughput; analytic conditioning; no audio quality claim",
              "resource_isolation": "not established; shared-machine activity and filesystem caches can affect timings"}
    try:
        reference = output / "conditioning-tone.wav"
        write_conditioning_tone(reference)
        lock = tomllib.loads((ROOT / "Cargo.lock").read_text())
        diff = subprocess.check_output(["git", "diff", "HEAD", "--", "crates/tts-qwen", "crates/tts-core"], cwd=ROOT)
        report.update(source_revision=revision(ROOT), source_diff_sha256=hashlib.sha256(diff).hexdigest(),
                      source_dirty=bool(diff), cargo_lock_sha256=sha256(ROOT / "Cargo.lock"),
                      rustc=version(["rustc", "--version", "--verbose"]), cargo=version(["cargo", "--version"]),
                      dependencies={package["name"]: package["version"] for package in lock["package"]
                                    if package["name"] in {"candle-core", "candle-nn", "candle-transformers", "rayon"}},
                      reference_implementation_pin=PINS["qwen"], requested_threads=args.threads)
        source_files = sorted(path for crate in ("tts-qwen", "tts-core")
                              for path in (ROOT / "crates" / crate).rglob("*")
                              if path.suffix == ".rs" or path.name == "Cargo.toml")
        report["source_files_sha256"] = {str(path.relative_to(ROOT)): sha256(path) for path in source_files}
        inputs = [reference, model / "model.safetensors", model / "speech_tokenizer/model.safetensors"]
        inputs.extend(model / name for name in ["config.json", "tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json"]
                      if (model / name).exists())
        report["inputs"] = [{"path": str(path), "sha256": sha256(path)} for path in inputs]
        metadata = model / ".cache/huggingface/download/model.safetensors.metadata"
        report["download_metadata_revision"] = metadata.read_text().splitlines()[0] if metadata.exists() else None
        with (output / "build.log").open("w") as stream:
            subprocess.run(["cargo", "build", "--locked", "--release", "-p", "tts-qwen", "--example", "tts_benchmark"],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        target = Path(os.environ.get("CARGO_TARGET_DIR", str(ROOT / "target")))
        if not target.is_absolute():
            target = ROOT / target
        binary = target / "release/examples" / ("tts_benchmark.exe" if os.name == "nt" else "tts_benchmark")
        raw = output / "samples.json"
        command = [str(binary), str(model), str(reference), "Hello.", str(raw), str(args.max_new_tokens),
                   str(args.warmups), str(args.runs), "English"]
        child_env = dict(os.environ, RAYON_NUM_THREADS=str(args.threads), OMP_NUM_THREADS=str(args.threads))
        report.update(command=command, binary_sha256=sha256(binary), text="Hello.", language="English")
        with (output / "inference.log").open("w") as stream:
            subprocess.run(command, cwd=ROOT, env=child_env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        result = json.loads(raw.read_text())
        if result.get("passed") is not True or len(result.get("samples", [])) != args.runs:
            raise ValueError("Benchmark did not return every requested successful sample")
        metrics = ["extract_seconds", "prefill_seconds", "generate_seconds", "decode_seconds",
                   "synthesis_seconds", "request_seconds", "audio_seconds", "synthesis_rtf", "request_rtf"]
        summary = {}
        for metric in metrics:
            values = [sample[metric] for sample in result["samples"]]
            if not all(math.isfinite(value) and value > 0 for value in values):
                raise ValueError(f"Invalid measurements for {metric}")
            summary[metric] = {"mean": statistics.mean(values), "median": statistics.median(values),
                               "min": min(values), "max": max(values), "p95": percentile(values, 0.95),
                               "standard_deviation": statistics.stdev(values) if len(values) > 1 else None}
        report.update(passed=True, measurements=result, summary=summary)
        write_json(output / "benchmark.json", report)
        print(json.dumps({"passed": True, "load_seconds": result["load_seconds"],
                          "median_request_rtf": summary["request_rtf"]["median"],
                          "report": str(output / "benchmark.json")}))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError, KeyError) as error:
        report["error"] = str(error)
        write_json(output / "benchmark.json", report)
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
