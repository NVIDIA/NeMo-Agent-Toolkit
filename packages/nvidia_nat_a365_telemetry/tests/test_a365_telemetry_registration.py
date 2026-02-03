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
from nat.plugins.a365_telemetry.register import A365TelemetryExporter, _resolve_token_resolver
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


def test_resolve_token_resolver_none():
    """Test that None token resolver path returns None."""
    assert _resolve_token_resolver(None) is None


def test_resolve_token_resolver_empty_string():
    """Test that empty string token resolver path raises ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        _resolve_token_resolver("")


def test_resolve_token_resolver_invalid_format():
    """Test that invalid format token resolver path raises ValueError."""
    with pytest.raises(ValueError, match="Invalid token_resolver path format"):
        _resolve_token_resolver("not_a_valid_path")


def test_resolve_token_resolver_nonexistent_module():
    """Test that nonexistent module raises ValueError."""
    with pytest.raises(ValueError, match="Failed to import module"):
        _resolve_token_resolver("nonexistent.module.function")


def test_resolve_token_resolver_nonexistent_function():
    """Test that nonexistent function in module raises AttributeError."""
    with pytest.raises(AttributeError, match="not found in module"):
        _resolve_token_resolver("logging.nonexistent_function")


def test_resolve_token_resolver_not_callable():
    """Test that non-callable attribute raises ValueError."""
    with pytest.raises(ValueError, match="is not callable"):
        _resolve_token_resolver("logging.INFO")  # logging.INFO is a constant, not callable


def test_resolve_token_resolver_success():
    """Test successful token resolver resolution."""
    # Use a simple built-in function for testing
    resolver = _resolve_token_resolver("builtins.str")

    assert resolver is not None
    assert callable(resolver)
    # Verify it's actually the str function
    assert resolver("test") == "test"


def test_resolve_token_resolver_custom_function():
    """Test resolving a custom function from a test module."""
    # Create a simple test module inline
    import types

    # Create a mock module
    test_module = types.ModuleType("test_token_resolver_module")

    def mock_token_resolver(agent_id: str, tenant_id: str) -> str:
        return f"token_for_{agent_id}_{tenant_id}"

    test_module.mock_token_resolver = mock_token_resolver

    # Import it into sys.modules so importlib can find it
    import sys
    sys.modules["test_token_resolver_module"] = test_module

    try:
        resolver = _resolve_token_resolver("test_token_resolver_module.mock_token_resolver")

        assert resolver is not None
        assert callable(resolver)
        # Test the resolver function
        token = resolver("agent-123", "tenant-456")
        assert token == "token_for_agent-123_tenant-456"
    finally:
        # Clean up
        if "test_token_resolver_module" in sys.modules:
            del sys.modules["test_token_resolver_module"]
