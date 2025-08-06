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

import base64
import importlib.metadata
import logging
import os
import subprocess
from functools import lru_cache

from aiq.data_models.component import AIQComponentEnum
from aiq.data_models.discovery_metadata import DiscoveryMetadata
from aiq.registry_handlers.schemas.package import WheelData
from aiq.registry_handlers.schemas.publish import AIQArtifact
from aiq.runtime.loader import PluginTypes
from aiq.runtime.loader import discover_entrypoints

# pylint: disable=redefined-outer-name
logger = logging.getLogger(__name__)


@lru_cache
def get_module_name_from_distribution(distro_name: str) -> str | None:
    """Return the first top-level module name for a given distribution name."""
    if not distro_name:
        return None

    try:
        # Read 'top_level.txt' which contains the module(s) provided by the package
        dist = importlib.metadata.distribution(distro_name)
        # will reading a file set of vun scan?
        top_level = dist.read_text('top_level.txt')

        if top_level:
            module_names = top_level.strip().split()
            # return firs module name
            return module_names[0]
    except importlib.metadata.PackageNotFoundError:
        # Distribution not found
        return None
    except FileNotFoundError:
        # 'top_level.txt' might be missing
        return None

    return None


def build_wheel(package_root: str) -> WheelData:
    """Builds a Python .whl for the specified package and saves to disk, sets self._whl_path, and returned as bytes.

    Args:
        package_root (str): Path to the local package repository.

    Returns:
        WheelData: Data model containing a built python wheel and its corresponding metadata.
    """

    import importlib.util
    import re
    import tomllib

    from pkginfo import Wheel

    pyproject_toml_path = os.path.join(package_root, "pyproject.toml")

    if not os.path.exists(pyproject_toml_path):
        raise ValueError("Invalid package path, does not contain a pyproject.toml file.")

    with open(pyproject_toml_path, "rb") as f:
        data = tomllib.load(f)

    toml_project: dict = data.get("project", {})
    toml_project_name = toml_project.get("name", None)

    assert toml_project_name is not None, f"Package name '{toml_project_name}' not found in pyproject.toml"
    # replace "aiqtoolkit" substring with "aiq" to get the import name
    module_name = get_module_name_from_distribution(toml_project_name)
    assert module_name is not None, f"No modules found for package name '{toml_project_name}'"

    assert importlib.util.find_spec(module_name) is not None, (f"Package {module_name} not "
                                                               "installed, cannot discover components.")

    toml_packages = set(i for i in data.get("project", {}).get("entry-points", {}).get("aiq.plugins", {}))
    toml_dependencies = set(
        re.search(r"[a-zA-Z][a-zA-Z\d_-]*", package_name).group(0)
        for package_name in toml_project.get("dependencies", []))

    union_dependencies = toml_dependencies.union(toml_packages)
    union_dependencies.add(toml_project_name)

    working_dir = os.getcwd()
    os.chdir(package_root)

    result = subprocess.run(["uv", "build", "--wheel"], check=True)
    result.check_returncode()

    whl_file = sorted(os.listdir("dist"), reverse=True)[0]
    whl_file_path = os.path.join("dist", whl_file)

    with open(whl_file_path, "rb") as whl:
        whl_bytes = whl.read()
        whl_base64 = base64.b64encode(whl_bytes).decode("utf-8")

    whl_path = os.path.join(os.getcwd(), whl_file_path)

    os.chdir(working_dir)

    whl_version = Wheel(whl_path).version

    return WheelData(package_root=package_root,
                     package_name=toml_project_name,
                     toml_project=toml_project,
                     toml_dependencies=toml_dependencies,
                     toml_aiq_packages=toml_packages,
                     union_dependencies=union_dependencies,
                     whl_path=whl_path,
                     whl_base64=whl_base64,
                     whl_version=whl_version)


def build_package_metadata(wheel_data: WheelData | None) -> dict[AIQComponentEnum, list[dict | DiscoveryMetadata]]:
    """Loads discovery metadata for all registered AIQ Toolkit components included in this Python package.

    Args:
        wheel_data (WheelData): Data model containing a built python wheel and its corresponding metadata.

    Returns:
        dict[AIQComponentEnum, list[typing.Union[dict, DiscoveryMetadata]]]: List containing each components discovery
        metadata.
    """

    from aiq.cli.type_registry import GlobalTypeRegistry
    from aiq.registry_handlers.metadata_factory import ComponentDiscoveryMetadata
    from aiq.runtime.loader import discover_and_register_plugins

    discover_and_register_plugins(PluginTypes.ALL)

    registry = GlobalTypeRegistry.get()

    aiq_plugins = discover_entrypoints(PluginTypes.ALL)

    if (wheel_data is not None):
        registry.register_package(package_name=wheel_data.package_name, package_version=wheel_data.whl_version)
        for entry_point in aiq_plugins:
            package_name = entry_point.dist.name
            if (package_name == wheel_data.package_name):
                continue
            if (package_name in wheel_data.union_dependencies):
                registry.register_package(package_name=package_name)

    else:
        for entry_point in aiq_plugins:
            registry.register_package(package_name=entry_point.dist.name)

    discovery_metadata = {}
    for component_type in AIQComponentEnum:

        if (component_type == AIQComponentEnum.UNDEFINED):
            continue
        component_metadata = ComponentDiscoveryMetadata.from_package_component_type(wheel_data=wheel_data,
                                                                                    component_type=component_type)
        component_metadata.load_metadata()
        discovery_metadata[component_type] = component_metadata.get_metadata_items()

    return discovery_metadata


def build_aiq_artifact(package_root: str) -> AIQArtifact:
    """Builds a complete AIQ Toolkit Artifact that can be published for discovery and reuse.

    Args:
        package_root (str): Path to root of python package

    Returns:
        AIQArtifact: An publishabla AIQArtifact containing package wheel and discovery metadata.
    """

    from aiq.registry_handlers.schemas.publish import BuiltAIQArtifact

    wheel_data = build_wheel(package_root=package_root)
    metadata = build_package_metadata(wheel_data=wheel_data)
    built_artifact = BuiltAIQArtifact(whl=wheel_data.whl_base64, metadata=metadata)

    return AIQArtifact(artifact=built_artifact, whl_path=wheel_data.whl_path)
