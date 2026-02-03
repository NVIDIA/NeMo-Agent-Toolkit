#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare dependency licenses between the current and base `uv.lock`.

This script fetches the base lockfile from the GitHub repository and compares it
to the local `uv.lock`. It prints added, removed, and changed third-party
packages and includes license data where possible.

The output is intended for human review during CI checks, not as a machine-
parsable report.
"""

import argparse
import json
import tomllib
import urllib.request


def pypi_license(name: str, version: str | None = None) -> str:
    """Resolve a package license from PyPI metadata.

    Args:
        name: Distribution name on PyPI.
        version: Optional version pin used to query version-specific metadata.

    Returns:
        A best-effort license string from the available metadata fields.
    """
    # Use version-specific metadata when available to avoid mismatches.
    try:
        url = f"https://pypi.org/pypi/{name}/json" if version is None else f"https://pypi.org/pypi/{name}/{version}/json"
        with urllib.request.urlopen(url) as r:
            data = json.load(r)
    except Exception:
        return "(License not found)"

    info = data.get("info", {})
    candidates = []
    lic = (info.get("license_expression") or "").strip()
    if lic:
        candidates.append(lic)
    classifiers = info.get("classifiers") or []
    lic_cls = [c for c in classifiers if c.startswith("License ::")]
    if lic_cls:
        candidates.append("; ".join(lic_cls))
    lic = (info.get("license") or "").strip()
    if lic:
        candidates.append(lic)

    if candidates:
        return min(candidates, key=len)
    return "(License not found)"


def main(base_branch: str) -> None:
    """Compare the local `uv.lock` against a base branch lockfile.

    Args:
        base_branch: Git branch name used to locate the base `uv.lock` file.
    """
    # Read the current lockfile from the workspace.
    with open("uv.lock", "rb") as f:
        head = tomllib.load(f)

    # Fetch the reference lockfile from GitHub for comparison.
    try:
        with urllib.request.urlopen(
                f"https://raw.githubusercontent.com/NVIDIA/NeMo-Agent-Toolkit/{base_branch}/uv.lock") as f:
            base = tomllib.load(f)
    except Exception:
        print(f"Failed to fetch base lockfile from GitHub: {base_branch}")
        return

    # Index package metadata by name for easy diffing.
    head_packages = {pkg["name"]: pkg for pkg in head["package"]}
    base_packages = {pkg["name"]: pkg for pkg in base["package"]}

    added = head_packages.keys() - base_packages.keys()
    removed = base_packages.keys() - head_packages.keys()
    intersection = head_packages.keys() & base_packages.keys()

    # Track third-party dependency changes only (skip internal `nvidia-nat*`).
    added_packages = {pkg: head_packages[pkg] for pkg in added}
    removed_packages = {pkg: base_packages[pkg] for pkg in removed}
    changed_packages = {pkg: head_packages[pkg] for pkg in intersection if not pkg.startswith("nvidia-nat")}

    if added_packages:
        print("Added packages:")
        for pkg in sorted(added_packages.keys()):
            try:
                version = head_packages[pkg]["version"]
                license = pypi_license(pkg, version)
                print(f"- {pkg} {version} {license}")
            except KeyError:
                # "Source" entries lack pinned versions (VCS or local path).
                print(f"- {pkg} (source)")

    if removed_packages:
        print("Removed packages:")
        for pkg in sorted(removed_packages.keys()):
            try:
                version = base_packages[pkg]["version"]
                print(f"- {pkg} {version}")
            except KeyError:
                print(f"- {pkg} (source)")

    printed_header = False
    for pkg in sorted(changed_packages.keys()):
        try:
            head_version = head_packages[pkg]["version"]
            base_version = base_packages[pkg]["version"]
            if head_version == base_version:
                # Only report version or license changes.
                continue
            head_license = pypi_license(pkg, head_version)
            base_license = pypi_license(pkg, base_version)
            if not printed_header:
                print("Changed packages:")
                printed_header = True
            if head_license != base_license:
                print(f"- {pkg} {base_version} -> {head_version} ({base_license} -> {head_license})")
            else:
                print(f"- {pkg} {base_version} -> {head_version}")
        except KeyError:
            if not printed_header:
                print("Changed packages:")
                printed_header = True
            print(f"- {pkg} (source)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Report third-party dependency license changes between lockfiles.")
    parser.add_argument("--base-branch", type=str, default="develop")
    args = parser.parse_args()
    main(args.base_branch)
