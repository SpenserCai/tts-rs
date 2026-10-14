"""Strict tensor oracle comparison, with no NumPy dependency and no skip path."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from common import write_json

FLOAT_DTYPES = {"float16", "bfloat16", "float32", "float64"}
INTEGER_DTYPES = {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}


def read_document(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported or absent schema_version")
    tensors = document.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError(f"{path}: non-empty tensors object required")
    for name, tensor in tensors.items():
        shape, data, dtype = tensor.get("shape"), tensor.get("data"), tensor.get("dtype")
        if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
            raise ValueError(f"{path}: invalid shape for {name}")
        if dtype not in FLOAT_DTYPES | INTEGER_DTYPES | {"bool"}:
            raise ValueError(f"{path}: unsupported dtype {dtype!r} for {name}")
        if not isinstance(data, list) or math.prod(shape) != len(data) or not data:
            raise ValueError(f"{path}: incorrect or empty flattened data for {name}")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in data):
            raise ValueError(f"{path}: non-finite or non-numeric data for {name}")
        if dtype in INTEGER_DTYPES and any(type(value) is not int for value in data):
            raise ValueError(f"{path}: non-integer data for integer tensor {name}")
        if dtype in INTEGER_DTYPES:
            unsigned = dtype.startswith("uint")
            bits = int(dtype.removeprefix("uint" if unsigned else "int"))
            minimum, maximum = (0, 2**bits - 1) if unsigned else (-2**(bits-1), 2**(bits-1) - 1)
            if any(value < minimum or value > maximum for value in data):
                raise ValueError(f"{path}: integer outside {dtype} range for {name}")
        if dtype == "bool" and any(type(value) is not bool for value in data):
            raise ValueError(f"{path}: bool tensor must contain JSON booleans for {name}")
    return document


def compare(reference: Path, candidate: Path, atol: float, rtol: float) -> dict[str, Any]:
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be finite and non-negative")
    expected = read_document(reference)["tensors"]
    actual = read_document(candidate)["tensors"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "reference": str(reference),
        "candidate": str(candidate),
        "atol": atol,
        "rtol": rtol,
        "missing": sorted(set(expected) - set(actual)),
        "unexpected": sorted(set(actual) - set(expected)),
        "tensors": {},
        "passed": set(expected) == set(actual),
    }
    for name in sorted(set(expected) & set(actual)):
        left, right = expected[name], actual[name]
        item: dict[str, Any] = {"passed": False}
        if left["shape"] != right["shape"] or left["dtype"] != right["dtype"]:
            item.update(reason="shape or dtype mismatch", reference_shape=left["shape"],
                        candidate_shape=right["shape"], reference_dtype=left["dtype"],
                        candidate_dtype=right["dtype"])
        else:
            pairs = list(zip(left["data"], right["data"], strict=True))
            errors = [abs(a - b) for a, b in pairs]
            integer = left["dtype"] not in FLOAT_DTYPES
            failures = [index for index, (a, b) in enumerate(pairs)
                        if (a != b if integer else abs(a - b) > atol + rtol * abs(a))]
            l2_left = math.sqrt(sum(a * a for a, _ in pairs))
            l2_right = math.sqrt(sum(b * b for _, b in pairs))
            item.update(
                passed=not failures,
                elements=len(errors),
                max_absolute_error=max(errors),
                mean_absolute_error=sum(errors) / len(errors),
                root_mean_square_error=math.sqrt(sum(e * e for e in errors) / len(errors)),
                max_relative_error=max(e / max(abs(a), 1e-30) for e, (a, _) in zip(errors, pairs)),
                cosine_similarity=(sum(a * b for a, b in pairs) / (l2_left * l2_right)
                                   if l2_left and l2_right else None),
                failing_elements=len(failures),
                first_failing_indices=failures[:16],
                comparison="exact" if integer else "abs(candidate-reference) <= atol + rtol*abs(reference)",
            )
        report["tensors"][name] = item
        report["passed"] &= item["passed"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare(args.reference, args.candidate, args.atol, args.rtol)
    except (ValueError, OSError, TypeError, KeyError) as error:
        report = {"schema_version": 1, "passed": False, "error": str(error)}
    write_json(args.output, report)
    print(json.dumps({"passed": report["passed"], "report": str(args.output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
