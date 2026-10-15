#!/usr/bin/env python3
"""Keep every distribution on the Cargo workspace release version."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
RELEASE_VERSION = re.compile(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:-rc\.[1-9]\d*)?")


def python_version(version: str) -> str:
    """PEP 440 spelling of the same release candidate, not an independent version."""
    if RELEASE_VERSION.fullmatch(version) is None:
        raise ValueError("Expected MAJOR.MINOR.PATCH or MAJOR.MINOR.PATCH-rc.N")
    return version.replace("-rc.", "rc")


def workspace(root: Path = ROOT) -> dict:
    return tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))["workspace"]


def packages(root: Path = ROOT) -> dict[str, dict]:
    return {
        document["package"]["name"]: document["package"]
        for member in workspace(root)["members"]
        for path in root.glob(f"{member}/Cargo.toml")
        for document in [tomllib.loads(path.read_text(encoding="utf-8"))]
    }


def check(root: Path = ROOT) -> str:
    """Validate source and lockfiles without building or contacting a registry."""
    ws = workspace(root)
    version = ws["package"]["version"]
    python_version(version)
    members = packages(root)
    for name, package in members.items():
        if package.get("version") != {"workspace": True}:
            raise ValueError(f"{name}: use version.workspace = true")
    for name, dependency in ws.get("dependencies", {}).items():
        if isinstance(dependency, dict) and "path" in dependency and dependency.get("version") != version:
            raise ValueError(f"Workspace dependency {name} must declare version {version}")
    lock = tomllib.loads((root / "Cargo.lock").read_text(encoding="utf-8"))
    local = {package["name"]: package["version"] for package in lock["package"] if "source" not in package}
    for name in members:
        if local.get(name) != version:
            raise ValueError(f"Cargo.lock: {name} must have version {version}")
    python = tomllib.loads((root / "bindings/python/pyproject.toml").read_text(encoding="utf-8"))["project"]
    if "version" in python or "version" not in python.get("dynamic", []):
        raise ValueError("Python must inherit Cargo's version through project.dynamic")
    node = json.loads((root / "bindings/node/package.json").read_text(encoding="utf-8"))
    node_lock = json.loads((root / "bindings/node/package-lock.json").read_text(encoding="utf-8"))
    for label, actual in (("Node package", node.get("version")), ("npm lock", node_lock.get("version")),
                          ("npm lock root", node_lock.get("packages", {}).get("", {}).get("version"))):
        if actual != version:
            raise ValueError(f"{label} must use workspace version {version}")
    return version


def set_version(version: str, root: Path = ROOT) -> None:
    """Update only release metadata; Cargo refreshes its own workspace lock entries."""
    python_version(version)
    check(root)
    manifest = root / "Cargo.toml"
    source = manifest.read_text(encoding="utf-8")
    source, changed = re.subn(r'(\[workspace\.package\]\s*\nversion = ")[^"]+("\n)',
                             lambda match: f"{match[1]}{version}{match[2]}", source)
    if changed != 1:
        raise ValueError("Could not locate the single workspace version declaration")
    for name, dependency in workspace(root).get("dependencies", {}).items():
        if isinstance(dependency, dict) and "path" in dependency:
            source, changed = re.subn(rf'(^\s*{re.escape(name)}\s*=\s*\{{[^\n]*\bversion\s*=\s*")[^"]+("[^\n]*\}})',
                                     lambda match: f"{match[1]}{version}{match[2]}", source, flags=re.MULTILINE)
            if changed != 1:
                raise ValueError(f"Could not update workspace dependency {name}; preserve its inline version declaration")
    manifest.write_text(source, encoding="utf-8")
    for relative in ("bindings/node/package.json", "bindings/node/package-lock.json"):
        path = root / relative
        value = json.loads(path.read_text(encoding="utf-8"))
        value["version"] = version
        if "packages" in value:
            value["packages"][""]["version"] = version
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    subprocess.run(["cargo", "metadata", "--offline", "--format-version", "1"], cwd=root,
                   stdout=subprocess.DEVNULL, check=True)
    check(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Reject release-version drift")
    update = commands.add_parser("set", help="Update all release metadata together; no tag or upload")
    update.add_argument("version")
    args = parser.parse_args()
    try:
        if args.command == "set":
            set_version(args.version)
        print(f"Unified release version: {check()}")
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
