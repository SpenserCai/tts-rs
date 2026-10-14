"""Manual evaluation entry point shared by local development and Actions."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from common import ROOT, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["parity", "e2e", "benchmark"], required=True)
    parser.add_argument("--model", choices=["qwen", "cosyvoice"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "metal"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-kind", choices=["synthetic", "pretrained"], default="synthetic")
    parser.add_argument("--prepare-env", action="store_true")
    parser.add_argument("--environment", type=Path, default=ROOT / ".reference")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--e2e-manifest", type=Path)
    parser.add_argument("--cosyvoice-source", type=Path)
    parser.add_argument("--model-directory", type=Path)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.prepare_env:
        from prepare_env import prepare
        python = prepare(args.environment, args.model, args.device)
        command = [str(python), str(Path(__file__).resolve())]
        command += [arg for arg in sys.argv[1:] if arg != "--prepare-env"]
        return subprocess.run(command, check=False).returncode
    try:
        if args.suite == "e2e":
            if args.e2e_manifest is None:
                if args.model != "qwen" or args.device != "cpu":
                    raise ValueError("Automatic E2E currently requires Qwen CPU; other paths require a reviewed --e2e-manifest")
                from prepare_qwen_e2e import prepare
                args.e2e_manifest = prepare(args.output, args.model_directory)
            from e2e import run
            report = run(args.e2e_manifest, args.output, args.device)
            return 0 if report["passed"] else 1
        from compare_tensors import compare
        if args.model == "qwen":
            from export_qwen import export
            export(args.output, args.fixture_kind, args.weights, args.device, 41)
            package, example, weight_file = "tts-qwen", "speaker_oracle", "speaker.safetensors"
            scope = "Qwen ECAPA speaker encoder, excludes frontend and synthesis"
        else:
            if args.fixture_kind != "synthetic":
                raise ValueError("CosyVoice pretrained parity has not been implemented; this gate fails explicitly")
            from export_cosyvoice import export
            source = args.cosyvoice_source or args.environment / "cosyvoice"
            export(source, args.output, args.device)
            package, example, weight_file = "tts-cosyvoice", "embeddings_oracle", "conditioning.safetensors"
            scope = "CosyVoice DiT conditioning modules; excludes full DiT, flow and vocoder"
        cargo = ["cargo", "build", "--locked", "-p", package, "--example", example, "--release"]
        if args.device != "cpu":
            cargo += ["--features", args.device]
        subprocess.run(cargo, cwd=ROOT, check=True)
        target = Path(os.environ.get("CARGO_TARGET_DIR", str(ROOT / "target")))
        binary = target / "release" / "examples" / (example + (".exe" if os.name == "nt" else ""))
        command = [str(binary), str(args.output / "config.json"),
                   str(args.output / weight_file), str(args.output / "input.json"),
                   str(args.output / "candidate.json"), args.device]
        subprocess.run(command, check=True)
        report = compare(args.output / "reference.json", args.output / "candidate.json", 1e-5, 1e-4)
        report.update(model=args.model, device=args.device, fixture_kind=args.fixture_kind,
                      scope=scope)
        write_json(args.output / "parity.json", report)
        if not report["passed"]:
            return 1
        if args.suite == "benchmark":
            from benchmark import benchmark
            benchmark(command, args.output / "benchmark", args.runs, 1,
                      scope + "; process startup + weights load + forward")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        write_json(args.output / "failure.json", {"schema_version": 1, "passed": False, "error": str(error)})
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
