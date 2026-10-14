"""Measure an explicit command; report process wall time without hiding failures.

Every sample launches a new process. This is not steady-state kernel latency or
time-to-first-audio. Warmups prime filesystem/runtime caches only. For inference
RTF, the caller must provide the duration of the actual generated output audio.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

from common import environment, write_json


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def benchmark(command: list[str], output: Path, runs: int, warmups: int,
              scope: str, audio_duration: float | None = None) -> dict:
    if not command or runs < 1 or warmups < 0:
        raise ValueError("A command, positive run count, and non-negative warmups are required")
    if audio_duration is not None and (not math.isfinite(audio_duration) or audio_duration <= 0):
        raise ValueError("Audio duration must be finite and positive")
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, **environment(), "command": command, "scope": scope,
              "method": "fresh process wall time including load and startup; caches may be warm",
              "resource_isolation": "not established; shared-machine load can affect measurements",
              "warmups": warmups, "requested_runs": runs, "passed": False, "samples_seconds": []}
    for index in range(warmups + runs):
        started = time.perf_counter()
        with (output / f"run-{index:03d}.log").open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        elapsed = time.perf_counter() - started
        if process.returncode:
            report.update(failed_run=index, exit_code=process.returncode)
            write_json(output / "benchmark.json", report)
            raise RuntimeError(f"Benchmark command failed in run {index}; exit {process.returncode}")
        if index >= warmups:
            report["samples_seconds"].append(elapsed)
    samples = report["samples_seconds"]
    report.update(passed=True, mean_seconds=statistics.mean(samples),
                  median_seconds=statistics.median(samples), p95_seconds=percentile(samples, 0.95),
                  min_seconds=min(samples), max_seconds=max(samples),
                  standard_deviation_seconds=statistics.stdev(samples) if len(samples) > 1 else None)
    if audio_duration is not None:
        report.update(audio_duration_seconds=audio_duration,
                      real_time_factor_samples=[value / audio_duration for value in samples])
    write_json(output / "benchmark.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--audio-duration", type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        report = benchmark(command, args.output, args.runs, args.warmups, args.scope, args.audio_duration)
    except (ValueError, OSError, RuntimeError) as error:
        print(str(error))
        return 1
    print(json.dumps({"passed": report["passed"], "median_seconds": report["median_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
