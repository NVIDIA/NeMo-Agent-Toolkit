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
import itertools
import tomllib
import typing
import urllib.request
from collections.abc import Iterator
from operator import itemgetter

from package_utils import Package
from package_utils import UvLock
from package_utils import package_variant_key
from package_utils import pypi_license


def main(base_branch: str) -> None:
    """Compare the local `uv.lock` against a base branch lockfile.

    Args:
        base_branch: Git branch name used to locate the base `uv.lock` file.
    """
    # Read the current lockfile from the workspace.
    with open("uv.lock", "rb") as f:
        head: UvLock = typing.cast(UvLock, tomllib.load(f))

    # Fetch the reference lockfile from GitHub for comparison.
    try:
        with urllib.request.urlopen(
                f"https://raw.githubusercontent.com/NVIDIA/NeMo-Agent-Toolkit/{base_branch}/uv.lock") as f:
            base: UvLock = typing.cast(UvLock, tomllib.load(f))
    except Exception:
        print(f"Failed to fetch base lockfile from GitHub: {base_branch}")
        return

    # packages to filter out from the diff
    FILTERED_PACKAGE_PREFIXES = ["nvidia-nat"]

    # Index package metadata by name and variant for easy diffing.
    head_by_name: Iterator[tuple[str, Iterator[Package]]] = itertools.groupby(head["package"], key=itemgetter("name"))
    base_by_name: Iterator[tuple[str, Iterator[Package]]] = itertools.groupby(base["package"], key=itemgetter("name"))

    # grouped entries based on add/removed/changed
    added_entries: list[Package] = []
    removed_entries: list[Package] = []
    changed_entries: list[tuple[Package, Package]] = []

    # iterators over the grouped entries
    heads: Iterator[tuple[str, Iterator[Package]]] = iter(head_by_name)
    bases: Iterator[tuple[str, Iterator[Package]]] = iter(base_by_name)

    # cursors over the grouped entries
    current_head: tuple[str, Iterator[Package]] | None = next(heads, None)
    current_base: tuple[str, Iterator[Package]] | None = next(bases, None)

    # single-pass iteration over the grouped entries
    while current_head is not None or current_base is not None:

        if current_head is not None and (current_base is None or current_head[0] < current_base[0]):
            # head package is before base package; add it to the added entries
            name, group = current_head
            if not any(str(name).startswith(prefix) for prefix in FILTERED_PACKAGE_PREFIXES):
                added_entries.extend(group)
            current_head = next(heads, None)
            continue

        if current_base is not None and (current_head is None or current_base[0] < current_head[0]):
            # base package is before head package; add it to the removed entries
            name, group = current_base
            if not any(str(name).startswith(prefix) for prefix in FILTERED_PACKAGE_PREFIXES):
                removed_entries.extend(group)
            current_base = next(bases, None)
            continue

        # same name in both; add it to the changed entries
        assert current_head is not None and current_base is not None
        name, head_group = current_head
        _, base_group = current_base
        head_pkgs: list[Package] = list(head_group)
        base_pkgs: list[Package] = list(base_group)
        current_head = next(heads, None)
        current_base = next(bases, None)

        if any(str(name).startswith(prefix) for prefix in FILTERED_PACKAGE_PREFIXES):
            continue

        head_variants: dict[tuple[str, str], Package] = {package_variant_key(pkg): pkg for pkg in head_pkgs}
        base_variants: dict[tuple[str, str], Package] = {package_variant_key(pkg): pkg for pkg in base_pkgs}

        added: set[tuple[str, str]] = set(head_variants.keys()) - set(base_variants.keys())
        removed: set[tuple[str, str]] = set(base_variants.keys()) - set(head_variants.keys())

        if added and removed and len(added) == len(removed):
            for b, h in zip(removed, added, strict=True):
                changed_entries.append((base_variants[b], head_variants[h]))
        else:
            added_entries.extend(head_variants[k] for k in added)
            removed_entries.extend(base_variants[k] for k in removed)

    if added_entries:
        print("Added packages:")
        for pkg in added_entries:
            name = pkg["name"]
            if (version := pkg.get("version")):
                print(f"- {name} {version} {pypi_license(name, version)}")
            else:
                print(f"- {name} (source)")

    if removed_entries:
        print("Removed packages:")
        for pkg in removed_entries:
            print(f"- {pkg['name']} {pkg.get('version', '(source)')}")

    if changed_entries:
        print("Changed packages:")
        for base_pkg, head_pkg in changed_entries:
            try:
                pkg_name = head_pkg["name"]
                base_version = base_pkg.get("version", None)
                head_version = head_pkg.get("version", None)
                if (head_license := pypi_license(pkg_name, head_version)) \
                    != (base_license := pypi_license(pkg_name, base_version)):
                    print(f"- {pkg_name} {base_version} -> {head_version} ({base_license} -> {head_license})")
                else:
                    print(f"- {pkg_name} {base_version} -> {head_version}")
            except KeyError:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Report third-party dependency license changes between lockfiles.")
    parser.add_argument("base_branch",
                        type=str,
                        nargs='?',
                        default="develop",
                        help="The base branch to compare against. Defaults to 'develop'.")
    args = parser.parse_args()
    main(args.base_branch)
