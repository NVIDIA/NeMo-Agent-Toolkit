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

from packaging.requirements import Requirement

from nat.data_models.component import ComponentEnum
from nat.data_models.discovery_metadata import DiscoveryMetadata
from nat.registry_handlers.schemas.package import WheelData
from nat.registry_handlers.schemas.publish import Artifact
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_entrypoints

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


def parse_requirement(requirement: str) -> str:
    """Extract the base package name from a requirement string.

    This function extracts only the package name, ignoring extras, version specifiers,
    and environment markers.

    Args:
        requirement (str): A requirement string like 'numpy>=1.20.0' or 'requests[security]~=2.28.0'

    Returns:
        str: The base package name (e.g., 'numpy' from 'numpy>=1.20.0',
        'requests' from 'requests[security]~=2.28.0')
    """
    # Handle inline comments by splitting on '#' and taking the first part
    clean_requirement = requirement.split('#')[0].strip()
    if not clean_requirement:
        return ""

    try:
        parsed = Requirement(clean_requirement)
        return parsed.name.lower()
    except Exception as e:
        logger.warning("Failed to parse requirement '%s': %s. Skipping this dependency.", requirement, e)
        return ""


def resolve_extras_to_packages(package_name: str, extras: list[str]) -> set[str]:
    """Resolve package extras to their actual package dependencies.

    Args:
        package_name (str): The base package name (e.g., 'nvidia-nat')
        extras (list[str]): List of extra names (e.g., ['langchain', 'telemetry'])

    Returns:
        set[str]: Set of additional package names that the extras resolve to
        (e.g., {'nvidia-nat-langchain', 'nvidia-nat-opentelemetry', 'nvidia-nat-phoenix',
        'nvidia-nat-weave', 'nvidia-nat-ragaai'})
    """
    resolved_packages = set()

    try:
        # Get the distribution metadata for the package
        dist = importlib.metadata.distribution(package_name)

        # Parse all requirements to find optional dependencies
        requires = dist.requires or []

        for requirement_str in requires:
            try:
                req = Requirement(requirement_str)

                # Check if this requirement has a marker that matches our extras
                if req.marker:
                    for extra in extras:
                        # Try marker evaluation first
                        try:
                            if req.marker.evaluate({'extra': extra}):
                                resolved_packages.add(req.name.lower())
                                break
                        except Exception:
                            # Fallback to simple string check
                            marker_str = str(req.marker)
                            if f'extra == "{extra}"' in marker_str or f"extra == '{extra}'" in marker_str:
                                resolved_packages.add(req.name.lower())
                                break

            except Exception as e:
                logger.warning("Failed to parse requirement '%s' for extras resolution: %s", requirement_str, e)

    except importlib.metadata.PackageNotFoundError:
        logger.warning("Package '%s' not found for extras resolution", package_name)
    except Exception as e:
        logger.warning("Failed to resolve extras for package '%s': %s", package_name, e)

    return resolved_packages


def extract_dependencies_with_extras_resolved(pyproject_path: str) -> set[str]:
    """Extract dependency names from pyproject.toml with extras properly resolved.

    This function not only extracts the base package names but also resolves
    any extras (e.g., package[extra1,extra2]) to their actual package dependencies.

    Args:
        pyproject_path (str): Path to the pyproject.toml file

    Returns:
        set[str]: Set of all dependency names including those resolved from extras

    Example:
        For a dependency like "nat[langchain,telemetry]~=1.2", this will return:
        {'nvidia-nat', 'nvidia-nat-langchain', 'nvidia-nat-opentelemetry', 'nvidia-nat-phoenix', ...}

    Raises:
        FileNotFoundError: If the pyproject.toml file doesn't exist
        ValueError: If the file cannot be parsed
    """
    import tomllib

    if not os.path.exists(pyproject_path):
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")

    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse pyproject.toml: {e}") from e

    project_data = data.get("project", {})
    all_dependencies = set()

    def _process_dependency(dep_spec: str):
        """Process a single dependency specification and resolve extras."""
        # Handle inline comments
        clean_req = dep_spec.split('#')[0].strip()
        if not clean_req:
            return

        try:
            parsed = Requirement(clean_req)
            base_name = parsed.name.lower()
            all_dependencies.add(base_name)

            # If there are extras, try to resolve them
            if parsed.extras:
                resolved_extras = resolve_extras_to_packages(base_name, list(parsed.extras))
                all_dependencies.update(resolved_extras)

        except Exception as e:
            logger.warning("Failed to process dependency '%s': %s", dep_spec, e)

    # Process main dependencies
    for dep_spec in project_data.get("dependencies", []):
        _process_dependency(dep_spec)

    # Process optional dependencies
    optional_deps = project_data.get("optional-dependencies", {})
    for _group_name, group_deps in optional_deps.items():
        for dep_spec in group_deps:
            _process_dependency(dep_spec)

    return all_dependencies


@lru_cache
def get_distributions() -> list[importlib.metadata.Distribution]:
    """Get all installed distributions. This is an expensive operation and should be cached."""
    return list(importlib.metadata.distributions())


def find_distribution_name(name: str) -> str | None:
    """Try to find the correct distribution name for a given package name.

    Uses dynamic discovery through importlib.metadata to find distributions
    that provide the requested module/package name.

    Args:
        name (str): Package name to search for.

    Returns:
        str | None: The correct distribution name if found, None otherwise.
    """
    # First try the name as-is
    try:
        importlib.metadata.distribution(name)
        return name
    except importlib.metadata.PackageNotFoundError:
        pass

    # Try common case variations
    variations = [
        name.lower(),
        name.upper(),
        name.replace('-', '_'),
        name.replace('_', '-'),
    ]

    # Try each variation
    for variation in variations:
        if variation != name:  # Skip the original name we already tried
            try:
                importlib.metadata.distribution(variation)
                return variation
            except importlib.metadata.PackageNotFoundError:
                continue

    # Search through all installed distributions to find one that provides this module
    try:
        for dist in get_distributions():
            dist_name = dist.metadata['Name']

            # Check top-level packages provided by this distribution
            try:
                # Try to get top-level packages from metadata
                top_level_txt = dist.read_text('top_level.txt')
                if top_level_txt:
                    top_level_packages = set(top_level_txt.strip().split('\n'))
                    if name in top_level_packages:
                        return dist_name
            except (FileNotFoundError, AttributeError):
                # top_level.txt doesn't exist, try alternative method
                pass

            # Fallback: check file paths for top-level modules
            try:
                if hasattr(dist, 'files') and dist.files:
                    top_level_from_files = {
                        f.parts[0]
                        for f in dist.files if len(f.parts) > 0 and not f.parts[0].endswith('.dist-info')
                    }
                    if name in top_level_from_files:
                        return dist_name
            except Exception:
                # Some distributions might not have files info or it might be inaccessible
                continue

    except Exception as e:
        logger.debug("Error searching distributions for %s: %s", name, e)

    return None


def get_transitive_dependencies(distribution_names: list[str]) -> dict[str, set[str]]:
    """Get transitive dependencies from a list of Python distribution names.

    This function recursively resolves all dependencies for the given distribution names,
    returning a mapping of each package to its complete set of transitive dependencies.
    This is useful when publishing plugins to remote registries that contain with nested dependencies,
    ensuring that all dependencies are included in the Artifact's metadata.

    Args:
        distribution_names (list[str]): List of Python distribution names (package names) to analyze.

    Returns:
        dict[str, set[str]]: Dictionary mapping each distribution name to its set of transitive dependencies.
        The dependencies include both direct and indirect dependencies.
    """
    result: dict[str, set[str]] = {}
    processing: set[str] = set()  # Track packages currently being processed (cycle detection)
    completed: set[str] = set()  # Track packages that have been fully processed

    def _get_dependencies_recursive(dist_name: str, path: set[str]) -> set[str]:
        """Recursively get all dependencies for a distribution.

        Args:
            dist_name: The distribution name to process
            path: Set of packages in the current dependency path (for cycle detection)
        """
        # If we've already computed this package's dependencies, return them
        if dist_name in completed:
            return result.get(dist_name, set())

        # If we encounter this package in the current path, we have a cycle
        if dist_name in path:
            logger.debug("Cycle detected in dependency chain: %s", " -> ".join(list(path) + [dist_name]))
            return set()

        # If we're currently processing this package in another branch, return empty
        # to avoid duplicate work (we'll get the full result when that branch completes)
        if dist_name in processing:
            return set()

        processing.add(dist_name)
        new_path = path | {dist_name}
        dependencies = set()

        try:
            dist = importlib.metadata.distribution(dist_name)
            requires = dist.requires or []

            for requirement in requires:
                # Skip requirements with extra markers (optional dependencies)
                # These should only be included if the extra is explicitly requested
                if 'extra ==' in requirement:
                    continue

                # Parse the requirement to get the package name
                dep_name = parse_requirement(requirement)

                # Skip self-references and empty names
                if not dep_name or dep_name == dist_name.lower():
                    continue

                dependencies.add(dep_name)

                # Recursively get dependencies of this dependency
                try:
                    transitive_deps = _get_dependencies_recursive(dep_name, new_path)
                    dependencies.update(transitive_deps)
                except importlib.metadata.PackageNotFoundError:
                    # Check if this is likely a conditional dependency (has markers)
                    is_conditional = any(marker in requirement for marker in [
                        'python_version', 'sys_platform', 'platform_system', 'platform_machine', 'implementation_name',
                        'implementation_version'
                    ])

                    if is_conditional:
                        # This is expected - conditional dependencies aren't always installed
                        logger.debug("Conditional dependency %s of %s is not installed: %s",
                                     dep_name,
                                     dist_name,
                                     requirement)
                    else:
                        # This might be a real issue - a non-conditional dependency is missing
                        logger.warning("Dependency %s of %s is not installed", dep_name, dist_name)
                    continue

        except importlib.metadata.PackageNotFoundError:
            # Transitive dependencies that aren't found are usually conditional (platform/version specific)
            # and this is expected behavior
            logger.debug("Distribution %s not found (likely conditional dependency)", dist_name)
            # Don't raise - just return empty dependencies for missing distributions
        finally:
            processing.remove(dist_name)

        result[dist_name] = dependencies
        completed.add(dist_name)
        return dependencies

    # Process each distribution name
    for dist_name in distribution_names:
        if dist_name not in completed:
            try:
                _get_dependencies_recursive(dist_name.lower(), set())
            except importlib.metadata.PackageNotFoundError:
                # Try to find the correct distribution name
                correct_name = find_distribution_name(dist_name)
                if correct_name:
                    logger.debug("Found distribution '%s' for requested name '%s'", correct_name, dist_name)
                    try:
                        _get_dependencies_recursive(correct_name.lower(), set())
                        # Map the original name to the results of the correct name
                        if correct_name.lower() in result:
                            result[dist_name] = result[correct_name.lower()]
                        continue
                    except importlib.metadata.PackageNotFoundError:
                        pass

                logger.error("Distribution %s not found (tried common variations)", dist_name, exc_info=True)
                result[dist_name] = set()

    return result


def get_all_transitive_dependencies(distribution_names: list[str]) -> set[str]:
    """Get all unique transitive dependencies from a list of Python distribution names.

    Returns a flattened set of all unique dependencies across all the provided distribution names.
    This is useful when publishing plugins to remote registries that contain with nested dependencies,
    ensuring that all dependencies are included in the Artifact's metadata.

    Args:
        distribution_names: List of Python distribution names (package names) to analyze

    Returns:
        set[str]: Set of all unique transitive dependency names
    """
    deps_map = get_transitive_dependencies(distribution_names)
    all_deps = set()

    for deps in deps_map.values():
        all_deps.update(deps)

    return all_deps


def build_wheel(package_root: str) -> WheelData:
    """Builds a Python .whl for the specified package and saves to disk, sets self._whl_path, and returned as bytes.

    Args:
        package_root (str): Path to the local package repository.

    Returns:
        WheelData: Data model containing a built python wheel and its corresponding metadata.
    """

    import tomllib

    from pkginfo import Wheel

    pyproject_toml_path = os.path.join(package_root, "pyproject.toml")

    if not os.path.exists(pyproject_toml_path):
        raise ValueError("Invalid package path, does not contain a pyproject.toml file.")

    with open(pyproject_toml_path, "rb") as f:
        data = tomllib.load(f)

    toml_project: dict = data.get("project", {})
    toml_project_name = toml_project.get("name", None)
    toml_packages = set(i for i in data.get("project", {}).get("entry-points", {}).get("nat.plugins", {}))

    # Extract dependencies using the robust requirement parser with extras resolution
    try:
        toml_dependencies = extract_dependencies_with_extras_resolved(pyproject_toml_path)
        logger.debug("Extracted dependencies with extras resolved: %s", toml_dependencies)
    except Exception as e:
        logger.warning("Failed to extract dependencies with extras resolution, falling back to basic extraction: %s", e)
        # Fallback to basic extraction
        toml_dependencies = set()
        for dep_spec in toml_project.get("dependencies", []):
            try:
                dep_name = parse_requirement(dep_spec)
                if dep_name:
                    toml_dependencies.add(dep_name)
            except Exception as e:
                logger.warning("Failed to parse dependency '%s': %s", dep_spec, e)

    toml_dependencies_transitive = get_all_transitive_dependencies(list(toml_dependencies))
    union_dependencies = toml_dependencies.union(toml_packages)
    union_dependencies.update(toml_dependencies_transitive)

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

    whl_version = Wheel(whl_path).version or "unknown"

    return WheelData(package_root=package_root,
                     package_name=toml_project_name,
                     toml_project=toml_project,
                     toml_dependencies=toml_dependencies,
                     toml_nat_packages=toml_packages,
                     union_dependencies=union_dependencies,
                     whl_path=whl_path,
                     whl_base64=whl_base64,
                     whl_version=whl_version)


def build_package_metadata(wheel_data: WheelData | None) -> dict[ComponentEnum, list[dict | DiscoveryMetadata]]:
    """Loads discovery metadata for all registered NAT components included in this Python package.

    Args:
        wheel_data (WheelData): Data model containing a built python wheel and its corresponding metadata.

    Returns:
        dict[ComponentEnum, list[typing.Union[dict, DiscoveryMetadata]]]: List containing each components discovery
        metadata.
    """

    from nat.cli.type_registry import GlobalTypeRegistry
    from nat.registry_handlers.metadata_factory import ComponentDiscoveryMetadata
    from nat.runtime.loader import discover_and_register_plugins

    discover_and_register_plugins(PluginTypes.ALL)

    registry = GlobalTypeRegistry.get()

    nat_plugins = discover_entrypoints(PluginTypes.ALL)

    if (wheel_data is not None):
        registry.register_package(package_name=wheel_data.package_name, package_version=wheel_data.whl_version)
        for entry_point in nat_plugins:
            package_name = entry_point.dist.name
            if (package_name == wheel_data.package_name):
                continue
            if (package_name in wheel_data.union_dependencies):
                registry.register_package(package_name=package_name)

    else:
        for entry_point in nat_plugins:
            registry.register_package(package_name=entry_point.dist.name)

    discovery_metadata = {}
    for component_type in ComponentEnum:

        if (component_type == ComponentEnum.UNDEFINED):
            continue
        component_metadata = ComponentDiscoveryMetadata.from_package_component_type(wheel_data=wheel_data,
                                                                                    component_type=component_type)
        component_metadata.load_metadata()
        discovery_metadata[component_type] = component_metadata.get_metadata_items()

    return discovery_metadata


def build_artifact(package_root: str) -> Artifact:
    """Builds a complete NeMo Agent toolkit Artifact that can be published for discovery and reuse.

    Args:
        package_root (str): Path to root of python package

    Returns:
        Artifact: A publishable Artifact containing package wheel and discovery metadata.
    """

    from nat.registry_handlers.schemas.publish import BuiltArtifact

    wheel_data = build_wheel(package_root=package_root)
    metadata = build_package_metadata(wheel_data=wheel_data)
    built_artifact = BuiltArtifact(whl=wheel_data.whl_base64, metadata=metadata)

    return Artifact(artifact=built_artifact, whl_path=wheel_data.whl_path)


# Compatibility alias
build_aiq_artifact = build_artifact
