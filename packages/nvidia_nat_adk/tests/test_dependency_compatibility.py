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

from importlib import metadata

import pytest
from packaging.requirements import Requirement


@pytest.mark.parametrize("package", ["google-adk", "opentelemetry-exporter-otlp", "opentelemetry-sdk"])
def test_installed_opentelemetry_dependencies_are_compatible(package):
    """Check the installed stack against upstream requirements, including ADK 2.x.

    ADK 1.x needed an exporter cap below 1.42. ADK 2.x supports newer exporters;
    retaining the old cap conflicts with current framework integrations.
    """
    requirements = [Requirement(dependency) for dependency in metadata.requires(package) or []]
    telemetry_requirements = [
        requirement for requirement in requirements if requirement.name.startswith("opentelemetry-") and (
            requirement.marker is None or requirement.marker.evaluate({"extra": ""}))
    ]
    assert telemetry_requirements, f"No OpenTelemetry requirements found for {package}"
    for requirement in telemetry_requirements:
        assert metadata.version(requirement.name) in requirement.specifier, f"{package} requires {requirement}"
