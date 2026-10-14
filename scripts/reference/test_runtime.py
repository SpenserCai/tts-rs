"""Prevent one model family's dependency installation from becoming its oracle."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from common import PINS, pinned_checkpoint_sha256, reference_runtime


class RuntimeContractTests(unittest.TestCase):
    def test_cpu_build_suffix_preserves_pinned_runtime(self):
        versions = dict(PINS["runtimes"]["cosyvoice"])
        versions.update(torch="2.3.1+cpu", torchaudio="2.3.1+cpu")
        with patch("common.version", side_effect=versions.__getitem__):
            self.assertEqual(reference_runtime("cosyvoice"), versions)

    def test_newer_transformers_is_not_an_equivalent_oracle(self):
        versions = dict(PINS["runtimes"]["cosyvoice"], transformers="4.57.3")
        with patch("common.version", side_effect=versions.__getitem__):
            with self.assertRaisesRegex(RuntimeError, "transformers"):
                reference_runtime("cosyvoice")

    def test_newer_torch_is_not_an_equivalent_oracle(self):
        versions = dict(PINS["runtimes"]["cosyvoice"], torch="2.10.0+cpu")
        with patch("common.version", side_effect=versions.__getitem__):
            with self.assertRaisesRegex(RuntimeError, "torch"):
                reference_runtime("cosyvoice")

    def test_arbitrary_local_weights_cannot_claim_the_official_revision(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "hift.pt"
            checkpoint.write_bytes(b"not the pinned checkpoint")
            with self.assertRaisesRegex(ValueError, "does not match pinned"):
                pinned_checkpoint_sha256("cosyvoice", checkpoint)


if __name__ == "__main__":
    unittest.main()
