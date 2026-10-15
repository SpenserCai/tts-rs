"""Release trust, immutable versions and restart behavior without network or credentials."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release
import release_plan as plan

COMMIT = "a" * 40
OTHER = "b" * 40
REPOSITORY = "example/speech"


class ReleasePlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.event = {"ref": "refs/heads/main", "after": COMMIT, "deleted": False}
        (self.root / "event.json").write_text(json.dumps(self.event), encoding="utf-8")
        self.environment = self.enterContext(patch.dict(plan.os.environ, {
            "GITHUB_REPOSITORY": REPOSITORY, "GITHUB_SHA": COMMIT, "GITHUB_RUN_ID": "123",
            "GITHUB_EVENT_NAME": "push", "GITHUB_EVENT_PATH": str(self.root / "event.json"),
            "GITHUB_OUTPUT": str(self.root / "outputs"), "GITHUB_STEP_SUMMARY": str(self.root / "summary"),
        }, clear=True))
        self.capture = self.enterContext(patch.object(plan, "capture", return_value=COMMIT))
        self.ancestor = self.enterContext(patch.object(plan, "is_ancestor", return_value=True))
        self.api = self.enterContext(patch.object(plan, "github_api"))
        self.enterContext(patch.object(plan.time, "sleep"))
        self.pull = {"state": "closed", "merged_at": "2026-10-15T00:00:00Z", "merge_commit_sha": COMMIT,
                     "base": {"ref": "main", "repo": {"full_name": REPOSITORY}}}

    def test_accepts_only_exact_merged_main_pr(self) -> None:
        self.api.return_value = [self.pull]
        self.assertTrue(plan.source_is_eligible("push", self.event, COMMIT, "registries"))
        self.ancestor.assert_called_once_with(COMMIT, "refs/remotes/origin/main")
        for path, value in [("state", "open"), ("merged_at", None), ("merge_commit_sha", OTHER),
                            ("base.ref", "develop"), ("base.repo.full_name", "someone/fork")]:
            candidate = deepcopy(self.pull)
            fields = path.split(".")
            item = candidate
            for field in fields[:-1]:
                item = item[field]
            item[fields[-1]] = value
            self.api.return_value = [candidate]
            with self.subTest(path=path):
                self.assertFalse(plan.source_is_eligible("push", self.event, COMMIT, "registries"))

    def test_waits_briefly_for_merged_pr_metadata(self) -> None:
        self.api.side_effect = [[], [{**self.pull, "merged_at": None}], [self.pull]]
        self.assertTrue(plan.source_is_eligible("push", self.event, COMMIT, "registries"))
        self.assertEqual(self.api.call_count, 3)

    def test_wrong_checkout_branch_or_event_cannot_publish(self) -> None:
        self.capture.return_value = OTHER
        with self.assertRaisesRegex(SystemExit, "Checkout"):
            plan.source_is_eligible("push", self.event, COMMIT, "registries")
        self.capture.return_value = COMMIT
        with self.assertRaisesRegex(SystemExit, "main push"):
            plan.source_is_eligible("push", {**self.event, "ref": "refs/heads/feature"}, COMMIT, "registries")
        with self.assertRaisesRegex(SystemExit, "Only a verified"):
            plan.source_is_eligible("pull_request_target", {}, COMMIT, "registries")
        self.api.assert_not_called()

    def test_manual_publication_requires_main_ancestry(self) -> None:
        self.ancestor.return_value = False
        with self.assertRaisesRegex(SystemExit, "main history"):
            plan.source_is_eligible("workflow_dispatch", {}, COMMIT, "registries")
        self.assertTrue(plan.source_is_eligible("workflow_dispatch", {}, COMMIT, "artifacts"))

    def test_direct_push_is_a_cheap_noop(self) -> None:
        self.api.return_value = []
        with patch.object(plan, "check_version") as version:
            plan.plan("artifacts")
        version.assert_not_called()
        self.assertIn("proceed=false", (self.root / "outputs").read_text())

    def test_unified_version_is_validated_before_remote_release_lookup(self) -> None:
        with patch.object(plan, "source_is_eligible", return_value=True), \
             patch.object(plan, "check_version", side_effect=ValueError("stable version required")):
            with self.assertRaisesRegex(ValueError, "stable version"):
                plan.plan("registries")
        self.api.assert_not_called()

    def test_completed_version_skips_build_for_later_docs_commit(self) -> None:
        self.api.return_value = {"draft": False, "published_at": "2026-10-15", "body": plan.completion_marker(OTHER)}
        with patch.object(plan, "source_is_eligible", return_value=True), \
             patch.object(plan, "check_version", return_value="0.1.0"), patch.object(plan, "preflight"), \
             patch.object(plan, "remote_tag", return_value=(OTHER, "previous workflow")):
            plan.plan("registries")
        self.assertIn("proceed=false", (self.root / "outputs").read_text())
        self.assertIn("unified workspace version", (self.root / "summary").read_text())

    def test_unfinished_version_blocks_new_build_even_for_same_commit(self) -> None:
        self.api.return_value = None
        for previous in (COMMIT, OTHER):
            with self.subTest(previous=previous), patch.object(plan, "source_is_eligible", return_value=True), \
                 patch.object(plan, "check_version", return_value="0.1.0"), patch.object(plan, "preflight"), \
                 patch.object(plan, "remote_tag", return_value=(previous, "original workflow")):
                with self.assertRaisesRegex(SystemExit, "Re-run failed jobs"):
                    plan.plan("registries")

    def test_claim_tag_is_immutable_and_retries_only_its_own_reservation(self) -> None:
        self.capture.side_effect = lambda *args: "" if args[1] == "status" else COMMIT
        with patch.object(plan, "verify_distributions") as verify, \
             patch.object(plan, "check_version", return_value="0.1.0"), patch.object(plan, "preflight"), \
             patch.object(plan, "remote_tag", return_value=None) as tag:
            self.api.side_effect = [{"sha": "c" * 40}, {"ref": "refs/tags/v0.1.0"}]
            plan.claim_tag(self.root)
            verify.assert_called_once_with(self.root)
            self.assertEqual(self.api.call_args_list, [
                call("git/tags", {"tag": "v0.1.0", "message": plan.run_marker(COMMIT), "object": COMMIT, "type": "commit"}),
                call("git/refs", {"ref": "refs/tags/v0.1.0", "sha": "c" * 40}),
            ])
            self.api.reset_mock()
            tag.return_value = (COMMIT, plan.run_marker(COMMIT))
            plan.claim_tag(self.root)
            self.api.assert_not_called()
            for previous in [(OTHER, plan.run_marker(OTHER)), (COMMIT, "another run")]:
                tag.return_value = previous
                with self.assertRaisesRegex(SystemExit, "another commit or workflow run"):
                    plan.claim_tag(self.root)
            self.api.assert_not_called()

    def test_failed_package_verification_never_creates_a_tag(self) -> None:
        with patch.object(plan, "verify_distributions", side_effect=SystemExit("checksum mismatch")):
            with self.assertRaisesRegex(SystemExit, "checksum"):
                plan.claim_tag(self.root)
        self.api.assert_not_called()

    def test_drafts_use_paginated_release_list(self) -> None:
        draft = {"tag_name": "v0.1.0", "draft": True}
        self.api.side_effect = [None, [{"tag_name": "old"}] * 100, [draft]]
        self.assertEqual(plan.release_document("v0.1.0"), draft)
        self.assertEqual(self.api.call_args_list[-1], call("releases?per_page=100&page=2"))


class GitHubPublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.file = self.root / "fixture.whl"
        self.file.write_bytes(b"verified distribution")
        self.asset = {"name": self.file.name, "digest": "sha256:" + hashlib.sha256(self.file.read_bytes()).hexdigest()}
        release.checksums(self.root)
        self.assets = [self.asset, {"name": "SHA256SUMS", "digest": "sha256:" + hashlib.sha256(
            (self.root / "SHA256SUMS").read_bytes()).hexdigest()}]
        self.enterContext(patch.dict(plan.os.environ, {"GITHUB_RUN_ID": "123"}))
        self.enterContext(patch.object(release, "verify_distributions"))
        self.enterContext(patch.object(release, "check_version", return_value="0.1.0"))
        self.enterContext(patch.object(release, "capture", return_value=COMMIT))
        self.enterContext(patch.object(plan, "remote_tag", return_value=(COMMIT, plan.run_marker(COMMIT))))
        self.run = self.enterContext(patch.object(release, "run"))
        self.document = {"draft": True, "body": plan.completion_marker(COMMIT) + "\n" + plan.run_marker(COMMIT), "assets": []}

    def test_publish_only_after_draft_assets_match_verified_digest(self) -> None:
        with patch.object(plan, "release_document", side_effect=[None, self.document, {**self.document, "assets": self.assets}]):
            release.github_release("v0.1.0", self.root, False)
        self.assertEqual(self.run.call_args_list[0].args[:5], ("gh", "release", "create", "v0.1.0", "--draft"))
        self.assertEqual(self.run.call_args_list[1], call("gh", "release", "upload", "v0.1.0", str(self.file)))
        self.assertEqual(Path(self.run.call_args_list[2].args[-1]).name, "SHA256SUMS")
        self.assertEqual(self.run.call_args_list[3], call("gh", "release", "edit", "v0.1.0", "--draft=false"))

    def test_retry_skips_identical_assets_and_already_published_release(self) -> None:
        document = {**self.document, "draft": False, "assets": self.assets}
        with patch.object(plan, "release_document", return_value=document):
            release.github_release("v0.1.0", self.root, False)
        self.run.assert_not_called()

    def test_conflicting_assets_or_owner_are_never_overwritten(self) -> None:
        for document in [{**self.document, "assets": [{**self.asset, "digest": "different"}]},
                         {**self.document, "body": "another release"}, {**self.document, "draft": False}]:
            with self.subTest(document=document), patch.object(plan, "release_document", return_value=document):
                with self.assertRaises(SystemExit):
                    release.github_release("v0.1.0", self.root, False)
            self.run.assert_not_called()

    def test_flat_checksums_verify_downloads_and_remain_stable_on_retry(self) -> None:
        (self.root / "wheels").mkdir()
        self.file.rename(self.root / "wheels" / self.file.name)
        (self.root / "npm").mkdir()
        (self.root / "npm" / "fixture.tgz").write_bytes(b"verified npm archive")
        release.checksums(self.root)
        original = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        manifests = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as download:
                files = release.github_asset_files(self.root, Path(staging))
                for path in files:
                    shutil.copyfile(path, Path(download) / path.name)
                manifest = (Path(download) / "SHA256SUMS").read_bytes()
                manifests.append(manifest)
                entries = manifest.decode("utf-8").splitlines()
                self.assertEqual(len(entries), 2)
                for entry in entries:
                    digest, name = entry.split("  ", 1)
                    self.assertEqual(Path(name).name, name)
                    self.assertEqual(hashlib.sha256((Path(download) / name).read_bytes()).hexdigest(), digest)
        self.assertEqual(manifests[0], manifests[1])
        self.assertIn(b"wheels/fixture.whl", original[self.root / "SHA256SUMS"])
        for path, content in original.items():
            self.assertEqual(path.read_bytes(), content)

    def test_duplicate_basenames_fail_before_any_publication(self) -> None:
        (self.root / "nested").mkdir()
        (self.root / "nested" / self.file.name).write_bytes(b"another archive")
        with self.assertRaisesRegex(SystemExit, "names must be unique"):
            release.github_release("v0.1.0", self.root, False)
        self.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
