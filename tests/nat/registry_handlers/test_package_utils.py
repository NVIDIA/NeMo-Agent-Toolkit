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

import os
import tempfile
import textwrap
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from nat.data_models.component import ComponentEnum
from nat.data_models.discovery_metadata import DiscoveryMetadata
from nat.registry_handlers.package_utils import build_artifact
from nat.registry_handlers.package_utils import build_package_metadata
from nat.registry_handlers.package_utils import build_wheel
from nat.registry_handlers.package_utils import extract_dependencies_with_extras_resolved
from nat.registry_handlers.package_utils import get_all_transitive_dependencies
from nat.registry_handlers.package_utils import get_transitive_dependencies
from nat.registry_handlers.package_utils import parse_requirement
from nat.registry_handlers.package_utils import resolve_extras_to_packages
from nat.registry_handlers.schemas.package import WheelData
from nat.registry_handlers.schemas.publish import Artifact


def test_build_wheel():

    package_root = "."

    wheel_data = build_wheel(package_root=package_root)

    assert isinstance(wheel_data, WheelData)
    assert wheel_data.package_root == package_root


@pytest.mark.parametrize("use_wheel_data", [
    (True),
    (False),
])
def test_build_package_metadata(use_wheel_data):

    wheel_data: WheelData | None = None
    if (use_wheel_data):
        wheel_data = WheelData(package_root=".",
                               package_name="nat",
                               toml_project={},
                               toml_dependencies=set(),
                               toml_nat_packages=set(),
                               union_dependencies=set(),
                               whl_path="whl/path.whl",
                               whl_base64="",
                               whl_version="")

    discovery_metadata = build_package_metadata(wheel_data=wheel_data)

    assert isinstance(discovery_metadata, dict)

    for component_type, discovery_metadatas in discovery_metadata.items():
        assert isinstance(component_type, ComponentEnum)

        for discovery_metadata in discovery_metadatas:
            DiscoveryMetadata(**discovery_metadata)


def test_build_nat_artifact():

    package_root = "."

    nat_artifact = build_artifact(package_root=package_root)

    assert isinstance(nat_artifact, Artifact)


class TestParseRequirement:
    """Test the parse_requirement function."""

    def test_simple_package_name(self):
        """Test parsing simple package names."""
        assert parse_requirement("numpy") == "numpy"
        assert parse_requirement("requests") == "requests"
        assert parse_requirement("Django") == "django"  # Should be lowercase

    def test_package_with_version_specifier(self):
        """Test parsing packages with version specifiers."""
        assert parse_requirement("numpy>=1.20.0") == "numpy"
        assert parse_requirement("requests~=2.28.0") == "requests"
        assert parse_requirement("pydantic==2.10.*") == "pydantic"

    def test_package_with_extras(self):
        """Test parsing packages with extras."""
        assert parse_requirement("requests[security]") == "requests"
        assert parse_requirement("uvicorn[standard]~=0.32.0") == "uvicorn"
        assert parse_requirement("nvidia-nat[langchain,telemetry]~=1.2") == "nvidia-nat"

    def test_package_with_comments(self):
        """Test parsing packages with inline comments."""
        assert parse_requirement("numpy>=1.20.0 # required for calculations") == "numpy"
        assert parse_requirement("requests # HTTP library") == "requests"

    def test_package_with_environment_markers(self):
        """Test parsing packages with environment markers."""
        assert parse_requirement("pytest ; python_version >= '3.8'") == "pytest"
        assert parse_requirement("sphinx ; extra == 'docs'") == "sphinx"

    def test_empty_or_invalid_requirements(self):
        """Test parsing empty or invalid requirements."""
        assert parse_requirement("") == ""
        assert parse_requirement("   ") == ""
        assert parse_requirement("# just a comment") == ""

    def test_whitespace_handling(self):
        """Test proper whitespace handling."""
        assert parse_requirement("  numpy  ") == "numpy"
        assert parse_requirement("\tnumpy\n") == "numpy"


class TestResolveExtrasToPackages:
    """Test the resolve_extras_to_packages function."""

    @patch('nat.registry_handlers.package_utils.importlib.metadata.distribution')
    def test_resolve_simple_extras(self, mock_distribution):
        """Test resolving simple extras."""
        # Mock the distribution metadata
        mock_dist = Mock()
        mock_dist.requires = [
            'package-a ; extra == "extra1"',
            'package-b ; extra == "extra2"',
            'package-c',  # No extra marker
        ]
        mock_distribution.return_value = mock_dist

        result = resolve_extras_to_packages("test-package", ["extra1"])
        assert result == {"package-a"}

        result = resolve_extras_to_packages("test-package", ["extra2"])
        assert result == {"package-b"}

        result = resolve_extras_to_packages("test-package", ["extra1", "extra2"])
        assert result == {"package-a", "package-b"}

    @patch('nat.registry_handlers.package_utils.importlib.metadata.distribution')
    def test_resolve_nonexistent_extras(self, mock_distribution):
        """Test resolving non-existent extras."""
        mock_dist = Mock()
        mock_dist.requires = [
            'package-a ; extra == "extra1"',
        ]
        mock_distribution.return_value = mock_dist

        result = resolve_extras_to_packages("test-package", ["nonexistent"])
        assert result == set()

    @patch('nat.registry_handlers.package_utils.importlib.metadata.distribution')
    def test_package_not_found(self, mock_distribution):
        """Test behavior when package is not found."""
        from importlib.metadata import PackageNotFoundError
        mock_distribution.side_effect = PackageNotFoundError("Package not found")

        result = resolve_extras_to_packages("nonexistent-package", ["extra1"])
        assert result == set()


class TestExtractDependenciesWithExtrasResolved:
    """Test the extract_dependencies_with_extras_resolved function."""

    @patch('nat.registry_handlers.package_utils.resolve_extras_to_packages')
    def test_extract_with_extras_resolution(self, mock_resolve_extras):
        """Test extracting dependencies with extras resolution."""
        mock_resolve_extras.return_value = {"resolved-package-1", "resolved-package-2"}

        content = textwrap.dedent("""
            [project]
            name = "test-package"
            dependencies = [
                "base-package[extra1,extra2]~=1.0",
                "simple-package"
            ]
            """)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(content)
            f.flush()

            try:
                deps = extract_dependencies_with_extras_resolved(f.name)

                # Should include base package, simple package, and resolved extras
                expected = {"base-package", "simple-package", "resolved-package-1", "resolved-package-2"}
                assert deps == expected

                # Verify resolve_extras_to_packages was called correctly
                mock_resolve_extras.assert_called_once()
                call_args = mock_resolve_extras.call_args
                assert call_args[0][0] == "base-package"  # First argument: package name
                assert set(call_args[0][1]) == {"extra1", "extra2"}  # Second argument: extras (order doesn't matter)

            finally:
                os.unlink(f.name)


class TestGetTransitiveDependencies:
    """Test the get_transitive_dependencies function."""

    @patch('nat.registry_handlers.package_utils.importlib.metadata.distribution')
    def test_simple_transitive_dependencies(self, mock_distribution):
        """Test getting simple transitive dependencies."""

        def mock_dist_side_effect(name):
            mock_dist = Mock()
            if name == "package-a":
                mock_dist.requires = ["package-b>=1.0", "package-c"]
            elif name == "package-b":
                mock_dist.requires = ["package-d"]
            elif name == "package-c":
                mock_dist.requires = []
            elif name == "package-d":
                mock_dist.requires = []
            else:
                from importlib.metadata import PackageNotFoundError
                raise PackageNotFoundError(f"Package {name} not found")
            return mock_dist

        mock_distribution.side_effect = mock_dist_side_effect

        result = get_transitive_dependencies(["package-a"])

        assert "package-a" in result
        expected_deps = {"package-b", "package-c", "package-d"}
        assert result["package-a"] == expected_deps

    @patch('nat.registry_handlers.package_utils.importlib.metadata.distribution')
    def test_cycle_detection(self, mock_distribution):
        """Test that cycles are properly detected and handled."""

        def mock_dist_side_effect(name):
            mock_dist = Mock()
            if name == "package-a":
                mock_dist.requires = ["package-b"]
            elif name == "package-b":
                mock_dist.requires = ["package-a"]  # Creates a cycle
            else:
                from importlib.metadata import PackageNotFoundError
                raise PackageNotFoundError(f"Package {name} not found")
            return mock_dist

        mock_distribution.side_effect = mock_dist_side_effect

        # Should not hang due to cycle detection
        result = get_transitive_dependencies(["package-a"])

        assert "package-a" in result
        # Should include package-b despite the cycle
        assert "package-b" in result["package-a"]

    @patch('nat.registry_handlers.package_utils.importlib.metadata.distribution')
    def test_missing_package(self, mock_distribution):
        """Test behavior with missing packages."""
        from importlib.metadata import PackageNotFoundError
        mock_distribution.side_effect = PackageNotFoundError("Package not found")

        result = get_transitive_dependencies(["nonexistent-package"])

        assert result == {"nonexistent-package": set()}


class TestGetAllTransitiveDependencies:
    """Test the get_all_transitive_dependencies function."""

    @patch('nat.registry_handlers.package_utils.get_transitive_dependencies')
    def test_flatten_dependencies(self, mock_get_transitive):
        """Test flattening of transitive dependencies."""
        mock_get_transitive.return_value = {
            "package-a": {"dep1", "dep2", "dep3"}, "package-b": {"dep2", "dep4", "dep5"}
        }

        result = get_all_transitive_dependencies(["package-a", "package-b"])

        expected = {"dep1", "dep2", "dep3", "dep4", "dep5"}
        assert result == expected
