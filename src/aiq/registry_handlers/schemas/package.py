# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class WheelData(BaseModel):
    """Data model containing a built python wheel and its corresponding metadata.

    Args:
        package_root (str): The path to the package root directory containing the pyproject.toml file.
        package_name (str): The name of the python package.
        toml_project (dict): A dictionary containing data about the python project.
        toml_dependencies (set): The list of dependencies provided in the pyproject.toml file.
        toml_aiq_packages (set): The AgentIQ plugins listed in the pyproject.toml.
        union_dependencies (set): The union of toml_dependencies and toml_aiq_packages.
        whl_path (str): The path to the package wheel file.
        whl_base64 (str): Base64 encoded string of the wheel file.
        whl_version (str): The version representing the wheel file.
    """

    package_root: str
    package_name: str
    toml_project: dict
    toml_dependencies: set
    toml_aiq_packages: set
    union_dependencies: set
    whl_path: str
    whl_base64: str
    whl_version: str


class PackageNameVersion(BaseModel):
    """Represents a data model containing a package name and version.

    Args:
        name (str): Package name, excluding the version.
        version (str | None): The package version, excluding the name. Defaults to None.
    """

    name: str
    version: str | None = None


class PackageNameVersionList(BaseModel):
    """Represents a data model containing a list of `PackageNameVersion` packages.

    Args:
        packages (list[PackageNameVersion]): A list of `PackageNameVersion` models. Defaults to `[]`
    """

    packages: list[PackageNameVersion] = []
