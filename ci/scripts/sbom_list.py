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
"""Generate a tab-separated list of dependency licenses from `uv.lock`.

The output is stored as `sbom_list.tsv` and includes package name, version, and
license metadata from PyPI. This is intended for lightweight SBOM checks in CI.
"""

import csv
import json
import tomllib
import urllib.request
from pathlib import Path

from tqdm import tqdm


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


def process_uvlock(uvlock: dict, base_name: str) -> Path:
    """Write a generic license table from a loaded `uv.lock` structure.

    Args:
        uvlock: Parsed `uv.lock` content.
        base_name: Logical label for the source data (kept for compatibility).

    Returns:
        Path to the generated `licenses.tsv` file.
    """
    # Keep packages ordered to make diffs stable between runs.
    sorted_packages = sorted(uvlock["package"], key=lambda x: x["name"])

    with open("licenses.tsv", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Name", "Version", "License"])
        for pkg in tqdm(sorted_packages, desc="Checking licenses", unit="packages"):
            try:
                name = pkg["name"]
                version = pkg["version"]
                license = pypi_license(name, version)
                writer.writerow([name, version, license])
            except KeyError:
                # Skip entries that do not have name/version info.
                pass
    return Path("licenses.tsv")


def main() -> None:
    """Create `sbom_list.tsv` for third-party license reporting."""
    # Load the lockfile that captures the dependency graph.
    with open("uv.lock", "rb") as f:
        head = tomllib.load(f)

    # Index packages by name for quick lookups.
    pkgs = {pkg["name"]: pkg for pkg in head["package"]}

    sbom_list = []
    for pkg in tqdm(pkgs.keys(), desc="Processing packages", unit="packages"):
        try:
            sbom_list.append({
                "name": pkg,
                "version": pkgs[pkg]["version"],
                "license": pypi_license(pkg, pkgs[pkg]["version"]),
            })
        except KeyError:
            # Skip entries that do not contain a version field.
            pass

    # Write the final SBOM table in a TSV format to keep it spreadsheet-friendly.
    with open("sbom_list.tsv", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Name", "Version", "License"])
        for pkg in sbom_list:
            writer.writerow([pkg["name"], pkg["version"], pkg["license"].replace("\n", "\\n")])


if __name__ == "__main__":
    main()
