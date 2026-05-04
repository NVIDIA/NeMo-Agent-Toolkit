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
"""Helpers for checking package Python-version compatibility in CI scripts."""

from __future__ import annotations

import argparse
import email
import re
import tomllib
import zipfile
from pathlib import Path

Version = tuple[int, int, int]


def _parse_version(value: str) -> Version:
    parts = [int(part) for part in re.findall(r"\d+", value)[:3]]
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def _compatible_release_bounds(version: Version, raw_version: str) -> tuple[Version, Version]:
    release_parts = [int(part) for part in re.findall(r"\d+", raw_version)]
    lower = version
    if len(release_parts) <= 2:
        upper = (version[0] + 1, 0, 0)
    else:
        upper = (version[0], version[1] + 1, 0)
    return lower, upper


def specifier_contains(specifier: str | None, python_version: str) -> bool:
    """Return whether a simple PEP 440 specifier set contains *python_version*.

    CI only needs the common `Requires-Python` operators used by local
    pyprojects and wheel metadata. Unknown clauses are treated conservatively as
    compatible so this helper does not accidentally hide valid test coverage.
    """
    if not specifier:
        return True

    version = _parse_version(python_version)
    for raw_clause in specifier.split(","):
        clause = raw_clause.strip()
        if not clause:
            continue

        match = re.match(r"(===|==|!=|<=|>=|<|>|~=)\s*([^\s,]+)", clause)
        if not match:
            continue

        operator, raw_bound = match.groups()
        if raw_bound.endswith(".*"):
            prefix = tuple(int(part) for part in raw_bound[:-2].split("."))
            version_prefix = version[:len(prefix)]
            if operator == "==" and version_prefix != prefix:
                return False
            if operator == "!=" and version_prefix == prefix:
                return False
            continue

        bound = _parse_version(raw_bound)
        if operator in {"==", "==="} and version != bound:
            return False
        if operator == "!=" and version == bound:
            return False
        if operator == ">=" and version < bound:
            return False
        if operator == ">" and version <= bound:
            return False
        if operator == "<=" and version > bound:
            return False
        if operator == "<" and version >= bound:
            return False
        if operator == "~=":
            lower, upper = _compatible_release_bounds(bound, raw_bound)
            if version < lower or version >= upper:
                return False

    return True


def project_requires_python(project_dir: Path) -> str | None:
    pyproject = project_dir / "pyproject.toml"
    with pyproject.open("rb") as file:
        data = tomllib.load(file)
    project = data.get("project", {})
    value = project.get("requires-python")
    return value if isinstance(value, str) else None


def project_supports_python(project_dir: Path, python_version: str) -> bool:
    return specifier_contains(project_requires_python(project_dir), python_version)


def wheel_requires_python(wheel_path: Path) -> str | None:
    with zipfile.ZipFile(wheel_path) as wheel:
        metadata_names = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if not metadata_names:
            return None
        metadata = email.message_from_bytes(wheel.read(metadata_names[0]))
    value = metadata.get("Requires-Python")
    return value if isinstance(value, str) else None


def wheel_supports_python(wheel_path: Path, python_version: str) -> bool:
    return specifier_contains(wheel_requires_python(wheel_path), python_version)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Python compatibility for a project or wheel.")
    parser.add_argument("kind", choices=["project", "wheel"])
    parser.add_argument("path", type=Path)
    parser.add_argument("python_version")
    args = parser.parse_args()

    if args.kind == "project":
        compatible = project_supports_python(args.path, args.python_version)
    else:
        compatible = wheel_supports_python(args.path, args.python_version)
    return 0 if compatible else 1


if __name__ == "__main__":
    raise SystemExit(main())
