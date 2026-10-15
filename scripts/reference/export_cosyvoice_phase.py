"""Export small, authored CPU phase-operation fixtures from pinned PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from common import PINS, offline_reference_runtime, reference_runtime, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime = reference_runtime("cosyvoice")
    offline_reference_runtime()
    import torch

    torch.set_num_threads(1)
    # Frequencies include fractional, low and high harmonics. The scan includes
    # cancellation and increments below float32 accumulator resolution, across
    # independent batch/harmonic lanes. No recording or checkpoint is involved.
    frequencies = torch.tensor([0.0, 0.1, 10.1, 110.3, 220.1, 440.2,
                                660.3, 880.4, 1234.567, 23999.0, 24000.0, 48000.1])
    scan = torch.tensor([
        [[16777216.0, 0.1], [1.0, 0.2], [1.0, 0.3], [-16777216.0, 0.4]],
        [[0.5, 1.0], [2.0 ** -25, -1.0], [2.0 ** -25, 0.00001], [2.0 ** -25, 0.00002]],
    ], dtype=torch.float32)
    inputs = {"frequencies": frequencies.tolist(), "scan": scan.tolist(), "sample_rate": 24000}

    def bits(value):
        return [int(item) & 0xffffffff for item in value.contiguous().view(torch.int32).flatten().tolist()]

    write_json(args.output, {
        "schema_version": 1,
        "metadata": {
            "cosyvoice_revision": PINS["cosyvoice"]["revision"],
            "pytorch_revision": "d44533f9d073df13895333e70b66f81c513c1889",
            "runtime": runtime, "device": "cpu", "dtype": "float32",
            "input_sha256": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
            "scope": "True scalar division and dim-1 cumsum only; not full source or waveform parity",
            "generator": "scripts/reference/export_cosyvoice_phase.py",
        },
        "division": {"input": frequencies.tolist(), "sample_rate": 24000,
                     "expected_bits": bits(frequencies / 24000)},
        "scan": {"input": scan.flatten().tolist(), "shape": list(scan.shape),
                 "expected_bits": bits(torch.cumsum(scan, dim=1))},
    })


if __name__ == "__main__":
    main()
