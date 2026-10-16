"""Offline packaging regressions; synthetic archives are not executable GPU tests."""

import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import accelerators
import release
from version import python_version


class AcceleratorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def pair(self, variant, bad_namespace=False):
        version = "0.1.2-rc.1"
        distribution, namespace, npm_name = accelerators.names(variant)
        wheel = self.root / f"{variant}.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(f"{distribution}.dist-info/METADATA",
                             f"Name: {distribution}\nVersion: {python_version(version)}\n")
            archive.writestr(f"{distribution}.dist-info/entry_points.txt",
                             f"[speechfoundry.backends]\n{accelerators.MATRIX[variant][0]} = {namespace}._native\n")
            archive.writestr(f"{namespace}/__init__.py", "")
            archive.writestr(f"{namespace}/_native.abi3.so", "fixture, not machine code")
            archive.writestr("licenses/LICENSE", "fixture")
            if bad_namespace:
                archive.writestr("speechfoundry/__init__.py", "must never overwrite CPU")
        tgz = self.root / f"{variant}.tgz"
        metadata = {"name": npm_name, "version": version, "private": True,
                    "speechfoundryBackend": accelerators.MATRIX[variant][0],
                    "main": "_native.js", "types": "_native.d.ts",
                    "os": [accelerators.MATRIX[variant][3]], "cpu": [accelerators.MATRIX[variant][4]]}
        with tarfile.open(tgz, "w:gz") as archive:
            for name, data in {"package/package.json": json.dumps(metadata),
                               "package/_native.js": "", "package/_native.d.ts": "", "package/binding.node": "fixture"}.items():
                content = data.encode()
                item = tarfile.TarInfo(name)
                item.size = len(content)
                archive.addfile(item, io.BytesIO(content))
        return version

    def test_all_backend_names_are_distinct_from_default_packages(self):
        for variant in accelerators.MATRIX:
            distribution, namespace, npm = accelerators.names(variant)
            self.assertNotEqual(distribution, "speechfoundry")
            self.assertNotEqual(namespace, "speechfoundry")
            self.assertNotEqual(npm, "@spensercai/speechfoundry")

    def test_inspection_accepts_the_isolated_pair(self):
        version = self.pair("cu118-linux-x64")
        self.assertEqual(len(accelerators.inspect_packages(self.root, "cu118-linux-x64", version)), 2)

    def test_inspection_rejects_cpu_namespace_collision(self):
        version = self.pair("cu124-linux-x64", True)
        with self.assertRaisesRegex(SystemExit, "CPU namespace"):
            accelerators.inspect_packages(self.root, "cu124-linux-x64", version)

    def test_inspection_rejects_wrong_backend(self):
        version = self.pair("cu128-linux-x64")
        with self.assertRaisesRegex(SystemExit, "identity mismatch"):
            accelerators.inspect_packages(self.root, "cu118-linux-x64", version)

    def test_candidate_npm_does_not_move_latest(self):
        self.pair("metal-darwin-arm64")
        with patch.object(release, "registry_document", return_value=None), patch.object(release, "run") as run:
            release.publish_npm(self.root)
        self.assertEqual(run.call_args.args[-2:], ("--tag", "next"))

    def test_staging_changes_only_binding_identity(self):
        target = self.root / "staged"
        accelerators.stage_workspace(target, "cu124-linux-x64")
        for source in (accelerators.ROOT / "crates").rglob("*.rs"):
            relative = source.relative_to(accelerators.ROOT)
            self.assertEqual(source.read_bytes(), (target / relative).read_bytes())
        self.assertFalse((target / "bindings/python/python/speechfoundry").exists())
        self.assertTrue((target / "bindings/python/python/_speechfoundry_cu124/__init__.py").exists())
        self.assertIn('module = "_speechfoundry_cu124"', (target / "bindings/python/src/lib.rs").read_text())
        self.assertIn('[project.entry-points."speechfoundry.backends"]', (target / "bindings/python/pyproject.toml").read_text())
        self.assertNotIn("_backends", (target / "bindings/python/python/_speechfoundry_cu124/__init__.py").read_text())


if __name__ == "__main__":
    unittest.main()
