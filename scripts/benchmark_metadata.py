#!/usr/bin/env python3
"""Record environment metadata alongside Criterion results, without claiming model RTF."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                              capture_output=True, check=False)
    changes = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True,
                             capture_output=True, check=False)
    document = {
        "schema_version": 1,
        "kind": "cpu-microbenchmark",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "working_tree_dirty": bool(changes.stdout) if changes.returncode == 0 else None,
        "rustc": command("rustc", "--version", "--verbose"),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "comparison_policy": "Informational on shared runners; compare regressions on identical hardware.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
