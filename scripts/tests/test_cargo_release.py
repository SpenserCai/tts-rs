"""Cargo archive identity and the real publication-selection path, without uploads."""

from __future__ import annotations

import contextlib
import hashlib
import io
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release


class CargoPublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.artifacts = self.root / "artifacts"
        (self.artifacts / "crates").mkdir(parents=True)
        (self.root / "target/release-crates").mkdir(parents=True)

    def crate(self, name: str = "speechfoundry", version: str = "0.1.2-rc.1", *,
              manifest: str | None = None, copies: int = 1, link: bool = False) -> Path:
        path = self.artifacts / "crates" / f"{name}-{version}.crate"
        if manifest is None:
            manifest = f'[package]\nname = "{name}"\nversion = "{version}"\n'
        with tarfile.open(path, "w:gz") as archive:
            for _ in range(copies):
                item = tarfile.TarInfo(f"{path.stem}/Cargo.toml")
                data = manifest.encode()
                if link:
                    item.type = tarfile.SYMTYPE
                    item.linkname = "Cargo.toml.orig"
                else:
                    item.size = len(data)
                archive.addfile(item, None if link else io.BytesIO(data))
            # Only Cargo.toml is authoritative, not the original workspace manifest.
            original = tarfile.TarInfo(f"{path.stem}/Cargo.toml.orig")
            data = b'[package]\nname = "wrong"\nversion.workspace = true\n'
            original.size = len(data)
            archive.addfile(original, io.BytesIO(data))
        shutil.copyfile(path, self.root / "target/release-crates" / path.name)
        return path

    @contextlib.contextmanager
    def publisher(self, version: str, document: dict | None):
        # Keep archive reading, digest comparison, URL construction and selection real.
        with patch.object(release, "ROOT", self.root), \
             patch.dict(release.os.environ, {"CARGO_TARGET_DIR": str(self.root / "target")}), \
             patch.object(release, "check_version", return_value=version), \
             patch.object(release, "preflight"), \
             patch.object(release, "verify_distributions"), \
             patch.object(release, "collect_crates"), \
             patch.object(release, "registry_document", return_value=document) as registry, \
             patch.object(release, "run") as run:
            yield registry, run

    def test_normalized_manifest_supports_hyphens_and_semver(self) -> None:
        for name in ("speechfoundry", "speechfoundry-core", "speechfoundry-cosyvoice", "speechfoundry-qwen"):
            for version in ("0.1.3", "0.1.2-rc.1", "1.2.3-alpha-beta.2+build.4"):
                with self.subTest(name=name, version=version):
                    self.assertEqual(release.crate_identity(self.crate(name, version)), (name, version))

    def test_rejects_missing_duplicate_or_linked_manifest(self) -> None:
        for options in ({"copies": 0}, {"copies": 2}, {"link": True}):
            with self.subTest(options=options), self.assertRaisesRegex(SystemExit, "regular Cargo manifest"):
                release.crate_identity(self.crate(**options))

    def test_rejects_unresolved_or_missing_identity(self) -> None:
        for manifest in ('[package]\nname = "speechfoundry"\nversion.workspace = true\n',
                         '[package]\nversion = "0.1.2-rc.1"\n',
                         '[package]\nname = ""\nversion = ""\n'):
            with self.subTest(manifest=manifest), self.assertRaisesRegex(SystemExit, "resolved package"):
                release.crate_identity(self.crate(manifest=manifest))

    def test_rejects_filename_manifest_disagreement(self) -> None:
        for name, version in (("other-crate", "0.1.2-rc.1"), ("speechfoundry", "0.1.3")):
            with self.subTest(name=name, version=version), self.assertRaisesRegex(SystemExit, "filename/manifest"):
                release.crate_identity(self.crate(manifest=f'[package]\nname="{name}"\nversion="{version}"\n'))

    def test_publisher_queries_and_selects_stable_and_prerelease_crates(self) -> None:
        # These are the exact names/versions misparsed by the old rsplit implementation.
        for version in ("0.1.3", "0.1.2-rc.1"):
            with self.subTest(version=version):
                path = self.crate(version=version)
                with self.publisher(version, None) as (registry, run):
                    release.rust_packages(True, self.artifacts)
                    registry.assert_called_once_with(f"https://crates.io/api/v1/crates/speechfoundry/{version}")
                    self.assertEqual(run.call_args.args, ("cargo", "publish", "--package", "speechfoundry", "--locked"))
                path.unlink()

    def test_identical_prerelease_retry_does_not_upload(self) -> None:
        path = self.crate()
        document = {"version": {"checksum": hashlib.sha256(path.read_bytes()).hexdigest(), "yanked": False}}
        with self.publisher("0.1.2-rc.1", document) as (registry, run):
            release.rust_packages(True, self.artifacts)
            registry.assert_called_once_with("https://crates.io/api/v1/crates/speechfoundry/0.1.2-rc.1")
            self.assertFalse(any(call.args[:2] == ("cargo", "publish") for call in run.call_args_list))

    def test_conflicting_or_yanked_archive_is_never_skipped_or_uploaded(self) -> None:
        path = self.crate()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for checksum, yanked in (("different", False), (digest, True)):
            with self.subTest(checksum=checksum, yanked=yanked):
                with self.publisher("0.1.2-rc.1", {"version": {"checksum": checksum, "yanked": yanked}}) as (_, run):
                    with self.assertRaisesRegex(SystemExit, "different or yanked"):
                        release.rust_packages(True, self.artifacts)
                    self.assertFalse(any(call.args[:2] == ("cargo", "publish") for call in run.call_args_list))

    def test_manifest_mismatch_stops_before_registry_query(self) -> None:
        self.crate(manifest='[package]\nname="wrong"\nversion="0.1.2-rc.1"\n')
        with self.publisher("0.1.2-rc.1", None) as (registry, run):
            with self.assertRaisesRegex(SystemExit, "filename/manifest"):
                release.rust_packages(True, self.artifacts)
            registry.assert_not_called()
            self.assertFalse(any(call.args[:2] == ("cargo", "publish") for call in run.call_args_list))


if __name__ == "__main__":
    unittest.main()
