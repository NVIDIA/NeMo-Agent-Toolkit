# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

"""Smoke tests for A365 telemetry exporter plugin registration and discovery."""

import pytest

from nat.cli.type_registry import GlobalTypeRegistry
from nat.plugins.a365.telemetry.register import A365TelemetryExporter
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins


@pytest.fixture(autouse=True)
def discover_plugins():
    """Discover and register all plugins before each test."""
    discover_and_register_plugins(PluginTypes.ALL)


def test_a365_telemetry_exporter_discovered():
    """Smoke test: Verify A365 telemetry exporter plugin is discovered and registered.

    Similar to test_mcp_plugin_discovered() - just checks plugin discovery works.
    """
    registry = GlobalTypeRegistry.get()

    registered_exporters = registry.get_registered_telemetry_exporters()
    exporter_types = [exporter.config_type for exporter in registered_exporters]

    assert A365TelemetryExporter in exporter_types, (
        f"A365TelemetryExporter not found in registered exporters. "
        f"Found: {[exporter.config_type.__name__ for exporter in registered_exporters]}"
    )


def test_a365_telemetry_exporter_config_accepts_auth_ref():
    """Test that A365TelemetryExporter accepts AuthenticationRef for token_resolver."""
    from nat.data_models.component_ref import AuthenticationRef
    
    # Pydantic will coerce strings to AuthenticationRef automatically
    # So both string and AuthenticationRef should work
    config1 = A365TelemetryExporter(
        agent_id="test-agent",
        tenant_id="test-tenant",
        token_resolver="test_auth",  # String is coerced to AuthenticationRef
    )
    assert isinstance(config1.token_resolver, AuthenticationRef)
    assert str(config1.token_resolver) == "test_auth"
    
    config2 = A365TelemetryExporter(
        agent_id="test-agent",
        tenant_id="test-tenant",
        token_resolver=AuthenticationRef("test_auth"),  # Explicit AuthenticationRef
    )
    assert isinstance(config2.token_resolver, AuthenticationRef)
    assert str(config2.token_resolver) == "test_auth"
