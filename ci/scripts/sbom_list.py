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
import argparse
import csv
import tomllib
import typing
from pathlib import Path

from package_utils import UvLock
from package_utils import pypi_license
from tqdm import tqdm


class SbomEntry(typing.TypedDict):
    name: str
    version: str
    license: str


def process_uvlock(uvlock: UvLock, output_path: Path) -> None:
    """Write a generic license table from a loaded `uv.lock` structure.

    Args:
        uvlock: Parsed `uv.lock` content.
        output_path: Path to the output file.
    """
    # Keep packages ordered to make diffs stable between runs.
    sbom_entries: dict[tuple[str, str], SbomEntry] = {}
    for pkg in tqdm(uvlock["package"], desc="Processing packages", unit="packages"):
        try:
            name = pkg["name"]
            version = pkg["version"]
        except KeyError:
            # Skip entries that do not contain a version field.
            continue
        key = (name, version)
        if key in sbom_entries:
            continue
        sbom_entries[key] = SbomEntry(
            name=name,
            version=version,
            license=pypi_license(name, version),
        )

    sbom_list: list[SbomEntry] = sorted(sbom_entries.values(), key=lambda entry: (entry["name"], entry["version"]))

    # Write the final SBOM table in a TSV format to keep it spreadsheet-friendly.
    with open(output_path, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Name", "Version", "License"])
        for pkg in sbom_list:
            writer.writerow([pkg["name"], pkg["version"], pkg["license"].replace("\n", "\\n")])


def main(uvlock_path: Path, output_path: Path) -> None:
    """Create SBOM list for third-party license reporting."""
    # Load the lockfile that captures the dependency graph.
    with open(uvlock_path, "rb") as f:
        head: UvLock = typing.cast(UvLock, tomllib.load(f))

    process_uvlock(head, output_path)

    print(f"SBOM list written successfully to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create SBOM list for third-party license reporting.")
    parser.add_argument("--uvlock",
                        type=Path,
                        help="Path to the lockfile to process. Defaults to 'uv.lock'.",
                        default="uv.lock")
    parser.add_argument("--output",
                        type=Path,
                        help="Path to the output file. Defaults to 'sbom_list.tsv'.",
                        default="sbom_list.tsv")
    args = parser.parse_args()
    main(uvlock_path=args.uvlock, output_path=args.output)
