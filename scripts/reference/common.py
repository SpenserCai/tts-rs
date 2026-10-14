"""Small, dependency-free artifact and provenance helpers."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PINS = json.loads(Path(__file__).with_name("pins.json").read_text())


def reference_runtime(model: str) -> dict[str, str]:
    """Reject a different model family's runtime before importing ML modules."""
    expected = dict(PINS["runtimes"][model])
    actual = {name: version(name) for name in expected}
    mismatched = {name: {"expected": expected[name], "actual": value}
                  for name, value in actual.items() if value.split("+", 1)[0] != expected[name]}
    if mismatched:
        raise RuntimeError(f"{model} reference runtime mismatch: {mismatched}")
    return actual


def offline_reference_runtime() -> None:
    """Request local Hugging Face loading without importing unrelated runtimes.

    These environment settings are not a network security boundary. Run an
    oracle under an external network policy when verified offline execution is
    required. Tensor-only oracles must not initialize ONNX Runtime merely to
    change its telemetry setting.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def model_command(command: list[str], offline: bool) -> list[str]:
    """Apply the offline boundary only to model execution, never preparation."""
    if not offline:
        return command
    if not sys.platform.startswith("linux"):
        raise ValueError("--offline-reference requires Linux and libseccomp")
    return [sys.executable, str(Path(__file__).with_name("offline_exec.py")), "--", *command]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_checkpoint_sha256(model: str, path: Path) -> str:
    """Bind a local checkpoint to the claimed upstream model revision."""
    expected = PINS[model]["weights_sha256"].get(path.name)
    if expected is None:
        raise ValueError(f"No reviewed checkpoint digest for {model}/{path.name}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Checkpoint does not match pinned {model} revision: {path.name}")
    return actual


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def environment() -> dict[str, Any]:
    info = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
    return info


def revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def tensor_document(tensors: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Detach tensors to CPU before serializing. Preserve source dtype explicitly."""
    result = {}
    for name, tensor in tensors.items():
        tensor = tensor.detach().cpu().contiguous()
        data = tensor.flatten().tolist()
        if not all(math.isfinite(value) for value in data):
            raise ValueError(f"Non-finite reference tensor: {name}")
        result[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "data": data,
        }
    return {"schema_version": 1, "metadata": metadata, "tensors": result}
