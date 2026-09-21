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

import pathlib
import tomllib

from packaging.requirements import Requirement
from packaging.version import Version


def test_adk_dependencies_cap_opentelemetry_exporter_at_compatible_version():
    """Verify the ADK package prevents an incompatible split OpenTelemetry stack."""
    pyproject_path = pathlib.Path(__file__).parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject_file:
        dependencies = tomllib.load(pyproject_file)["tool"]["setuptools_dynamic_dependencies"]["dependencies"]

    parsed_requirements = (Requirement(dependency) for dependency in dependencies if "{version}" not in dependency)
    exporter_requirement = next(requirement for requirement in parsed_requirements
                                if requirement.name == "opentelemetry-exporter-otlp")

    assert Version("1.41.1") in exporter_requirement.specifier
    for incompatible_version in ("1.42.0", "1.44.0", "2.0.0"):
        assert Version(incompatible_version) not in exporter_requirement.specifier
