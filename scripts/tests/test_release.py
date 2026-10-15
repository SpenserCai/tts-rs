"""Release invariants and immutable-registry retry behavior; no credentials/network."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release
import version


def write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class VersionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        write(self.root, "Cargo.toml", '[workspace]\nresolver = "3"\nmembers = ["crates/a"]\n'
              '[workspace.package]\nversion = "0.1.0"\nedition = "2024"\n'
              '[workspace.dependencies]\na = { path = "crates/a", version = "0.1.0" }\n')
        write(self.root, "crates/a/Cargo.toml", '[package]\nname = "a"\nversion.workspace = true\nedition.workspace = true\n')
        write(self.root, "crates/a/src/lib.rs", "//! Version fixture.\n")
        write(self.root, "Cargo.lock", 'version = 4\n[[package]]\nname = "a"\nversion = "0.1.0"\n')
        write(self.root, "bindings/python/pyproject.toml", '[project]\nname = "candle-tts"\ndynamic = ["version"]\n')
        write(self.root, "bindings/node/package.json", '{"name":"fixture", "version":"0.1.0"}')
        write(self.root, "bindings/node/package-lock.json", '{"version":"0.1.0", "packages":{"":{"version":"0.1.0"}}}')

    def test_updates_manifests_and_cargo_lock_together(self) -> None:
        version.set_version("0.2.0", self.root)
        self.assertEqual(version.check(self.root), "0.2.0")

    def test_rejects_independent_crate_version(self) -> None:
        write(self.root, "crates/a/Cargo.toml", '[package]\nname = "a"\nversion = "0.1.0"\n')
        with self.assertRaisesRegex(ValueError, "version.workspace"):
            version.check(self.root)

    def test_rejects_stale_cargo_lock(self) -> None:
        write(self.root, "Cargo.lock", '[[package]]\nname = "a"\nversion = "0.0.9"\n')
        with self.assertRaisesRegex(ValueError, "Cargo.lock"):
            version.check(self.root)

    def test_rejects_stale_npm_lock(self) -> None:
        write(self.root, "bindings/node/package-lock.json", '{"version":"0.1.0", "packages":{"":{"version":"0.0.9"}}}')
        with self.assertRaisesRegex(ValueError, "npm lock root"):
            version.check(self.root)

    def test_rejects_python_version_override(self) -> None:
        write(self.root, "bindings/python/pyproject.toml", '[project]\nversion = "0.1.0"\n')
        with self.assertRaisesRegex(ValueError, "dynamic"):
            version.check(self.root)

    def test_rejects_nonportable_version_before_writing(self) -> None:
        with self.assertRaisesRegex(ValueError, "MAJOR"):
            version.set_version("0.1.0+cuda", self.root)
        self.assertEqual(version.check(self.root), "0.1.0")

    def test_release_candidate_has_one_version_and_normalized_python_spelling(self) -> None:
        version.set_version("0.1.1-rc.1", self.root)
        self.assertEqual(version.check(self.root), "0.1.1-rc.1")
        self.assertEqual(version.python_version(version.check(self.root)), "0.1.1rc1")


class PublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def npm(self) -> Path:
        path = self.root / "fixture.tgz"
        data = json.dumps({"name": "@example/fixture", "version": "0.1.0"}).encode()
        with tarfile.open(path, "w:gz") as archive:
            item = tarfile.TarInfo("package/package.json")
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
        return path

    def test_npm_skips_only_identical_published_tarball(self) -> None:
        path = self.npm()
        digest = "sha512-" + base64.b64encode(hashlib.sha512(path.read_bytes()).digest()).decode()
        with patch.object(release, "registry_document", return_value={"dist": {"integrity": digest}}), patch.object(release, "run") as run:
            release.publish_npm(self.root)
            run.assert_not_called()
        with patch.object(release, "registry_document", return_value={"dist": {"integrity": "different"}}), patch.object(release, "run") as run:
            with self.assertRaisesRegex(SystemExit, "different archive"):
                release.publish_npm(self.root)
            run.assert_not_called()

    def test_pypi_stages_only_missing_exact_files(self) -> None:
        published = self.root / "published.whl"
        published.write_bytes(b"published wheel")
        missing = self.root / "missing.whl"
        missing.write_bytes(b"missing wheel")
        document = {"urls": [{"filename": published.name, "digests": {"sha256": hashlib.sha256(published.read_bytes()).hexdigest()}}]}
        with patch.object(release, "check_version", return_value="0.1.0"), patch.object(release, "registry_document", return_value=document):
            release.prepare_pypi(self.root, self.root / "upload")
            self.assertEqual([path.name for path in (self.root / "upload").iterdir()], [missing.name])
            published.write_bytes(b"changed wheel")
            with self.assertRaisesRegex(SystemExit, "different or yanked"):
                release.prepare_pypi(self.root, self.root / "conflict")

    def test_secret_presence_never_prints_values(self) -> None:
        values = {name: "test-do-not-log" for name in ("CARGO_REGISTRY_TOKEN", "NPM_TOKEN", "PYPI_API_TOKEN")}
        output = io.StringIO()
        with patch.dict(release.os.environ, values, clear=True), contextlib.redirect_stdout(output):
            release.check_secrets()
        self.assertNotIn("test-do-not-log", output.getvalue())
        with patch.dict(release.os.environ, {}, clear=True):
            with self.assertRaisesRegex(SystemExit, "CARGO_REGISTRY_TOKEN"):
                release.check_secrets()

    def test_checksum_changes_stop_publication(self) -> None:
        (self.root / "file").write_bytes(b"original")
        release.checksums(self.root)
        (self.root / "file").write_bytes(b"changed")
        with patch.object(release, "check_version", return_value="0.1.0"):
            with self.assertRaisesRegex(SystemExit, "checksum mismatch"):
                release.verify_distributions(self.root)


if __name__ == "__main__":
    unittest.main()
