"""Regression cases for false-positive numerical validation reports."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from compare_tensors import compare


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.reference = Path(self.directory.name) / "reference.json"
        self.candidate = Path(self.directory.name) / "candidate.json"
        self.document = {"schema_version": 1, "tensors": {
            "x": {"shape": [2], "dtype": "float32", "data": [0.0, 1.0]},
        }}
        self.reference.write_text(json.dumps(self.document))

    def check_document(self, document, atol=1e-5):
        self.candidate.write_text(json.dumps(document))
        return compare(self.reference, self.candidate, atol, 1e-4)

    def test_identical_passes(self):
        self.assertTrue(self.check_document(self.document)["passed"])

    def test_tolerance_applied_elementwise(self):
        document = copy.deepcopy(self.document)
        document["tensors"]["x"]["data"] = [0.00002, 1.0]
        report = self.check_document(document)
        self.assertFalse(report["passed"])
        self.assertEqual(report["tensors"]["x"]["first_failing_indices"], [0])

    def test_missing_tensor_is_failure(self):
        document = copy.deepcopy(self.document)
        document["tensors"]["wrong"] = document["tensors"].pop("x")
        self.assertFalse(self.check_document(document)["passed"])

    def test_shape_and_dtype_are_not_coerced(self):
        for field, value in [("shape", [1, 2]), ("dtype", "float64")]:
            document = copy.deepcopy(self.document)
            document["tensors"]["x"][field] = value
            self.assertFalse(self.check_document(document)["passed"])

    def test_nonfinite_rejected_even_on_both_sides(self):
        document = copy.deepcopy(self.document)
        document["tensors"]["x"]["data"][0] = float("nan")
        self.reference.write_text(json.dumps(document))
        with self.assertRaises(ValueError):
            self.check_document(document)

    def test_empty_and_truncated_rejected(self):
        for values in [[], [1.0]]:
            document = copy.deepcopy(self.document)
            document["tensors"]["x"]["data"] = values
            with self.assertRaises(ValueError):
                self.check_document(document)

    def test_integer_tokens_require_exact_match(self):
        document = copy.deepcopy(self.document)
        document["tensors"]["x"].update(dtype="int64", data=[1, 2])
        self.reference.write_text(json.dumps(document))
        document["tensors"]["x"]["data"] = [1, 3]
        self.assertFalse(self.check_document(document, atol=100)["passed"])

    def test_unsigned_values_do_not_wrap(self):
        document = copy.deepcopy(self.document)
        document["tensors"]["x"].update(dtype="uint32", data=[-1, 2])
        with self.assertRaises(ValueError):
            self.check_document(document)


if __name__ == "__main__":
    unittest.main()
