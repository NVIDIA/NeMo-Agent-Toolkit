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

import json
import typing
from urllib import request


class Package(typing.TypedDict):
    name: str
    version: typing.NotRequired[str]
    source: str


class UvLock(typing.TypedDict):
    package: list[Package]


def package_variant_key(pkg: Package) -> tuple[str, str]:
    return pkg["name"], pkg.get("version", "(source)")


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
        with request.urlopen(url) as r:
            data = json.load(r)
    except Exception:
        return "(License not found)"

    info = data.get("info", {})
    candidates: list[str] = []
    if (lic := (info.get("license_expression") or "").strip()):
        candidates.append(lic)
    if (lic := [c for c in (info.get("classifiers") or []) if c.startswith("License ::")]):
        candidates.append("; ".join(lic))
    if (lic := (info.get("license") or "").strip()):
        candidates.append(lic)

    text = typing.cast(str, min(candidates, key=len, default="(License not found)"))

    # Escape dangerous characters
    dangerous_chars = ('=', '+', '-', '@', '\t', '\r')
    if text.startswith(dangerous_chars):
        text = f"'{text}"
    return text
