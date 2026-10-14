#!/usr/bin/env python3
"""Estimate float32 tensor residency from safetensors headers, without loading weights."""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

MAX_HEADER_BYTES = 64 * 1024 * 1024
WASM32_ADDRESS_BYTES = 4 * 1024**3


def inspect_checkpoint(path: Path) -> dict:
    """Read a bounded header and count the model's projected float32 storage."""
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path.name}: missing safetensors header length")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_size <= min(MAX_HEADER_BYTES, file_size - 8):
            raise ValueError(f"{path.name}: invalid or oversized safetensors header")
        header_bytes = stream.read(header_size)
    header = json.loads(header_bytes)
    if not isinstance(header, dict):
        raise ValueError(f"{path.name}: safetensors header must be an object")

    elements = 0
    tensor_count = 0
    payload_size = file_size - 8 - header_size
    dtypes: dict[str, int] = {}
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(tensor, dict):
            raise ValueError(f"{path.name}: malformed tensor descriptor")
        shape = tensor.get("shape")
        offsets = tensor.get("data_offsets")
        dtype = tensor.get("dtype")
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f"{path.name}: invalid tensor shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(n) is not int for n in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= payload_size
        ):
            raise ValueError(f"{path.name}: invalid tensor data offsets")
        if not isinstance(dtype, str):
            raise ValueError(f"{path.name}: missing tensor dtype")
        count = math.prod(shape)
        elements += count
        tensor_count += 1
        dtypes[dtype] = dtypes.get(dtype, 0) + count

    if not tensor_count:
        raise ValueError(f"{path.name}: no tensors found")
    return {
        "filename": path.name,
        "file_bytes": file_size,
        "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
        "tensor_count": tensor_count,
        "elements_by_dtype": dtypes,
        "float32_tensor_bytes": elements * 4,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-wasm32-fit", action="store_true")
    args = parser.parse_args()
    try:
        checkpoints = [inspect_checkpoint(path) for path in args.checkpoints]
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    total = sum(item["float32_tensor_bytes"] for item in checkpoints)
    report = {
        "schema_version": 1,
        "assumption": "all listed tensor elements resident together as float32",
        "checkpoints": checkpoints,
        "float32_tensor_bytes": total,
        "wasm32_address_bytes": WASM32_ADDRESS_BYTES,
        "weights_alone_fit_wasm32": total < WASM32_ADDRESS_BYTES,
        "additional_memory_not_counted": [
            "model and tokenizer structures",
            "KV cache",
            "intermediate activations",
            "checkpoint transfer and dtype-conversion copies",
            "runtime and allocator overhead",
        ],
        "note": "A true weights_alone_fit_wasm32 result does not prove full-model feasibility.",
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.require_wasm32_fit and not report["weights_alone_fit_wasm32"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
