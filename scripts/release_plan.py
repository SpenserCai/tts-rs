#!/usr/bin/env python3
"""Plan merged-PR releases and reserve an immutable tag after package verification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request

from release import ROOT, capture, preflight, verify_distributions
from version import check as check_version


def repository() -> str:
    value = os.environ.get("GITHUB_REPOSITORY", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        raise SystemExit("GITHUB_REPOSITORY must identify the release repository")
    return value


def github_api(path: str, payload: dict | None = None) -> dict | list | None:
    """Only an explicit GET 404 means absent; authentication failures stay failures."""
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise SystemExit("GH_TOKEN is required for release metadata access")
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository()}/{path}",
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}",
                 "X-GitHub-Api-Version": "2022-11-28", "Content-Type": "application/json",
                 "User-Agent": "speechfoundry-release"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404 and payload is None:
            return None
        raise SystemExit(f"GitHub release metadata request failed: HTTP {error.code}") from None


def is_ancestor(older: str, newer: str) -> bool:
    result = subprocess.run(["git", "merge-base", "--is-ancestor", older, newer], cwd=ROOT,
                            check=False, capture_output=True)
    if result.returncode not in (0, 1):
        raise SystemExit("Cannot establish release commit ancestry")
    return result.returncode == 0


def source_is_eligible(event_name: str, event: dict, commit: str, destination: str) -> bool:
    """Accept only the exact merged main commit, or an explicit manual invocation."""
    if not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", commit):
        raise SystemExit("Release source must be a full commit SHA")
    if capture("git", "rev-parse", "HEAD") != commit:
        raise SystemExit("Checkout must equal the triggering source commit")
    if event_name == "push":
        if event.get("ref") != "refs/heads/main" or event.get("deleted") or event.get("after") != commit:
            raise SystemExit("Automatic releases require the exact main push commit")
        eligible = False
        # The push event can arrive just before PR association metadata converges.
        for attempt in range(3):
            pulls = github_api(f"commits/{commit}/pulls?per_page=100")
            if not isinstance(pulls, list):
                raise SystemExit("Cannot verify the merged pull request associated with this commit")
            eligible = any(
                pull.get("state") == "closed" and pull.get("merged_at")
                and pull.get("merge_commit_sha") == commit
                and pull.get("base", {}).get("ref") == "main"
                and pull.get("base", {}).get("repo", {}).get("full_name") == repository()
                for pull in pulls
            )
            if eligible:
                break
            if attempt < 2:
                time.sleep(2)
        if not eligible:
            return False
    elif event_name != "workflow_dispatch":
        raise SystemExit("Only a verified merged PR or workflow_dispatch can release")
    if destination == "registries" and not is_ancestor(commit, "refs/remotes/origin/main"):
        raise SystemExit("Publication source must belong to the repository's main history")
    return True


def remote_tag(tag: str) -> tuple[str, str] | None:
    reference = github_api(f"git/ref/tags/{tag}")
    if reference is None:
        return None
    target = reference["object"]
    if target["type"] == "commit":
        return target["sha"], ""
    if target["type"] != "tag":
        raise SystemExit("Release tag must point to a commit")
    annotated = github_api(f"git/tags/{target['sha']}")
    if not annotated or annotated["object"]["type"] != "commit":
        raise SystemExit("Release annotation must point directly to a commit")
    return annotated["object"]["sha"], annotated["message"].strip()


def release_document(tag: str) -> dict | None:
    """Find published releases by tag, and drafts through their documented list API."""
    published = github_api(f"releases/tags/{tag}")
    if published is not None:
        return published
    page = 1
    while True:
        releases = github_api(f"releases?per_page=100&page={page}")
        if not isinstance(releases, list):
            raise SystemExit("Cannot inspect existing GitHub release drafts")
        matches = [release for release in releases if release.get("tag_name") == tag]
        if len(matches) > 1:
            raise SystemExit("More than one GitHub release uses the requested tag")
        if matches:
            return matches[0]
        if len(releases) < 100:
            return None
        page += 1


def run_marker(commit: str) -> str:
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if not run_id.isdecimal():
        raise SystemExit("A GitHub Actions run ID is required for tag ownership")
    return f"speechfoundry release: {commit}; run: {run_id}"


def completion_marker(commit: str) -> str:
    return f"<!-- speechfoundry release complete: {commit} -->"


def report(outputs: dict[str, str], message: str) -> None:
    print(message)
    if os.environ.get("GITHUB_OUTPUT"):
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
            stream.writelines(f"{key}={value}\n" for key, value in outputs.items())
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")


def plan(destination: str) -> None:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    commit = os.environ["GITHUB_SHA"]
    event_name = os.environ["GITHUB_EVENT_NAME"]
    if event_name == "push":
        destination = "registries"
    if not source_is_eligible(event_name, event, commit, destination):
        report({"proceed": "false"}, "No release: this main push is not an exact merged PR commit.")
        return
    version = check_version()
    tag = f"v{version}"
    outputs = {"proceed": "true", "tag": tag, "destination": destination, "commit": commit}
    preflight(tag, False)
    if destination == "registries":
        existing = remote_tag(tag)
        published = github_api(f"releases/tags/{tag}")
        if published and not published.get("draft") and published.get("published_at"):
            if not existing or not is_ancestor(existing[0], commit):
                raise SystemExit("Published release tag is missing or outside this commit's history")
            if completion_marker(existing[0]) not in (published.get("body") or ""):
                raise SystemExit("Existing release has no verified multi-registry completion record")
            outputs["proceed"] = "false"
            report(outputs, f"{tag} is already published. No packages were rebuilt. "
                   "Update the unified workspace version in a PR to publish another release.")
            return
        if existing or published:
            message = (f"{tag} is reserved by an unfinished release. Use 'Re-run failed jobs' in the "
                       "original workflow run, not 'Re-run all jobs', to reuse its verified artifacts. "
                       "Do not rebuild or move the tag.")
            report({"proceed": "false"}, message)
            raise SystemExit(message)
    report(outputs, f"Release {tag} from {commit}; destination: {destination}.")


def claim_tag(directory: Path) -> None:
    """Reserve the version only after every distributable has been built and tested."""
    verify_distributions(directory)
    tag = f"v{check_version()}"
    preflight(tag, False)
    commit = capture("git", "rev-parse", "HEAD")
    if commit != os.environ.get("GITHUB_SHA") or not is_ancestor(commit, "refs/remotes/origin/main"):
        raise SystemExit("Refusing to tag sources outside this run's exact main commit")
    if capture("git", "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Refusing to tag modified tracked sources")
    marker = run_marker(commit)
    existing = remote_tag(tag)
    if existing is not None:
        if existing != (commit, marker):
            raise SystemExit("Release tag is already owned by another commit or workflow run")
        print(f"{tag} already belongs to this workflow run; no tag was changed")
        return
    annotated = github_api("git/tags", {"tag": tag, "message": marker, "object": commit, "type": "commit"})
    # Creating refs is intentionally non-overwriting. A concurrent reservation fails.
    github_api("git/refs", {"ref": f"refs/tags/{tag}", "sha": annotated["sha"]})
    print(f"Reserved immutable {tag} at {commit}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planned = commands.add_parser("plan")
    planned.add_argument("--destination", choices=("artifacts", "registries"), default="artifacts")
    claim = commands.add_parser("claim-tag")
    claim.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        plan(args.destination)
    else:
        claim_tag(args.directory)


if __name__ == "__main__":
    main()
