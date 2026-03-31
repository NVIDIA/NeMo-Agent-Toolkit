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

"""Tests for A365 front-end plugin registration and discovery."""

import logging
import os

import pytest

from nat.cli.type_registry import GlobalTypeRegistry
from nat.data_models.common import get_secret_value
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.plugins.a365.front_end.front_end_config import A365FrontEndConfig
from nat.plugins.a365.front_end.plugin import A365FrontEndPlugin
from nat.plugins.a365.front_end.register import a365_front_end
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins
from nat.test.functions import EchoFunctionConfig


@pytest.fixture(autouse=True)
def discover_plugins():
    """Discover and register all plugins before each test."""
    discover_and_register_plugins(PluginTypes.ALL)


def test_a365_frontend_discovered():
    """Test that A365 front-end plugin is discovered and registered."""
    registry = GlobalTypeRegistry.get()

    registered_front_ends = registry.get_registered_front_ends()
    front_end_types = [fe.config_type for fe in registered_front_ends]

    assert A365FrontEndConfig in front_end_types, (
        f"A365FrontEndConfig not found in registered front ends. "
        f"Found: {[fe.config_type.__name__ for fe in registered_front_ends]}"
    )


def test_a365_frontend_config_validation():
    """Test that A365FrontEndConfig validates required fields."""
    # Should fail without required fields (app_id is required)
    with pytest.raises(Exception):  # Pydantic validation error
        A365FrontEndConfig()

    # Should succeed with required fields
    # Note: app_password can be None in config, but registration function will validate it
    config = A365FrontEndConfig(
        app_id="test-app-id",
        app_password="test-app-password"
    )
    assert config.app_id == "test-app-id"
    assert get_secret_value(config.app_password) == "test-app-password"
    assert config.host == "localhost"  # default
    assert config.port == 3978  # default
    assert config.enable_notifications is True  # default


def test_a365_frontend_config_optional_fields():
    """Test that optional fields work correctly."""
    config = A365FrontEndConfig(
        app_id="test-app-id",
        app_password="test-app-password",
        tenant_id="test-tenant-id",
        host="0.0.0.0",
        port=8080,
        enable_notifications=False,
        notification_workflow="custom-workflow"
    )
    assert config.tenant_id == "test-tenant-id"
    assert config.host == "0.0.0.0"
    assert config.port == 8080
    assert config.enable_notifications is False
    assert config.notification_workflow == "custom-workflow"


def test_a365_frontend_config_port_validation():
    """Test that port validation works."""
    # Valid port
    config = A365FrontEndConfig(
        app_id="test-app-id",
        app_password="test-app-password",
        port=8080
    )
    assert config.port == 8080

    # Invalid port (too high)
    with pytest.raises(Exception):  # Pydantic validation error
        A365FrontEndConfig(
            app_id="test-app-id",
            app_password="test-app-password",
            port=70000
        )

    # Invalid port (negative)
    with pytest.raises(Exception):  # Pydantic validation error
        A365FrontEndConfig(
            app_id="test-app-id",
            app_password="test-app-password",
            port=-1
        )


@pytest.mark.asyncio
async def test_register_a365_front_end():
    """Test that the register_a365_front_end function returns the correct plugin."""
    # Create configuration objects
    a365_config = A365FrontEndConfig(
        app_id="test-app-id",
        app_password="test-app-password"
    )

    # Use a real Config with a proper workflow
    full_config = Config(
        general=GeneralConfig(front_end=a365_config),
        workflow=EchoFunctionConfig()
    )

    # Use the context manager pattern since a365_front_end
    # returns an AsyncGeneratorContextManager
    async with a365_front_end(a365_config, full_config) as plugin:
        # Verify that the plugin is of the correct type and has the right config
        assert isinstance(plugin, A365FrontEndPlugin)
        assert plugin.full_config is full_config
        assert plugin.front_end_config is a365_config


@pytest.mark.asyncio
async def test_register_a365_front_end_with_env_var():
    """Test that app_password can be loaded from environment variable."""
    # Set environment variable
    os.environ["A365_APP_PASSWORD"] = "env-password-123"
    
    try:
        # Create config without app_password
        a365_config = A365FrontEndConfig(
            app_id="test-app-id"
            # app_password not provided - should load from env
        )
        
        full_config = Config(
            general=GeneralConfig(front_end=a365_config),
            workflow=EchoFunctionConfig()
        )
        
        # Registration should succeed and load password from env
        async with a365_front_end(a365_config, full_config) as plugin:
            assert isinstance(plugin, A365FrontEndPlugin)
            assert get_secret_value(plugin.front_end_config.app_password) == "env-password-123"
    finally:
        # Clean up environment variable
        os.environ.pop("A365_APP_PASSWORD", None)


@pytest.mark.asyncio
async def test_register_a365_front_end_missing_password():
    """Test that registration fails if app_password is not provided."""
    # Ensure env var is not set
    os.environ.pop("A365_APP_PASSWORD", None)
    
    # Create config without app_password
    a365_config = A365FrontEndConfig(
        app_id="test-app-id"
        # app_password not provided
    )
    
    full_config = Config(
        general=GeneralConfig(front_end=a365_config),
        workflow=EchoFunctionConfig()
    )
    
    # Registration should fail with ValueError
    with pytest.raises(ValueError, match="app_password must be provided"):
        async with a365_front_end(a365_config, full_config) as plugin:
            pass


def test_security_configuration_validation_non_localhost(caplog):
    """Test that security warnings are logged for non-localhost bindings."""
    with caplog.at_level(logging.WARNING):
        config = A365FrontEndConfig(
            app_id="test-app-id",
            app_password="test-password",
            host="0.0.0.0"  # Non-localhost
        )
    
    # Check that warning was logged
    assert "non-localhost interface" in caplog.text.lower()
    assert "0.0.0.0" in caplog.text


def test_security_configuration_validation_default_port(caplog):
    """Test that security warnings are logged for default port on non-localhost."""
    with caplog.at_level(logging.WARNING):
        config = A365FrontEndConfig(
            app_id="test-app-id",
            app_password="test-password",
            host="192.168.1.100",  # Non-localhost
            port=3978  # Default port
        )
    
    # Check that warnings were logged
    assert "non-localhost interface" in caplog.text.lower()
    assert "default port" in caplog.text.lower() or "port 3978" in caplog.text


def test_security_configuration_validation_localhost_no_warning(caplog):
    """Test that no warnings are logged for localhost bindings."""
    with caplog.at_level(logging.WARNING):
        config = A365FrontEndConfig(
            app_id="test-app-id",
            app_password="test-password",
            host="localhost"  # Localhost - should not warn
        )
    
    # Check that no security warnings were logged
    assert "non-localhost interface" not in caplog.text.lower()
    assert "default port" not in caplog.text.lower()
