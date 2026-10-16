"""Repeatable wasm-pack staging without touching source manifests or prior good output."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ci


class WasmPackTests(unittest.TestCase):
    def test_repeated_builds_never_reuse_package_json(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(ci, "stage"):
            root = Path(temporary)
            (root / "bindings/wasm").mkdir(parents=True)
            output = root / "bindings/wasm/pkg"

            def build(*command, **kwargs):
                staging = Path(command[command.index("--out-dir") + 1])
                self.assertEqual(list(staging.iterdir()), [])
                (staging / "package.json").write_text('{"repository":{"type":"git"}}')

            with patch.object(ci, "ROOT", root), patch.object(ci, "run", side_effect=build):
                ci.build_wasm_package(output, "--target", "nodejs")
                (output / "stale.js").write_text("old generated module")
                ci.build_wasm_package(output, "--target", "nodejs", "--features", "qwen")
                self.assertFalse((output / "stale.js").exists())
                self.assertTrue((output / "package.json").is_file())

    def test_failed_build_keeps_previous_output_and_rejects_other_destinations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "bindings/wasm/pkg"
            output.mkdir(parents=True)
            marker = output / "package.json"
            marker.write_text("previous good package")
            with patch.object(ci, "ROOT", root), patch.object(ci, "run", side_effect=RuntimeError("build failed")):
                with self.assertRaises(RuntimeError):
                    ci.build_wasm_package(output, "--target", "nodejs")
                self.assertEqual(marker.read_text(), "previous good package")
                with self.assertRaises(ValueError):
                    ci.build_wasm_package(root, "--target", "nodejs")
