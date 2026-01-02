# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from nat.builder.builder import Builder
from nat.plugins.memmachine.memory import MemMachineMemoryClientConfig, memmachine_memory_client


@pytest.fixture(name="mock_builder")
def mock_builder_fixture():
    """Fixture to provide a mocked Builder instance."""
    return Mock(spec=Builder)


@pytest.fixture(name="config")
def config_fixture():
    """Fixture to provide a MemMachineMemoryClientConfig instance."""
    return MemMachineMemoryClientConfig(
        base_url="http://localhost:8080",
        org_id="test_org",
        project_id="test_project",
        timeout=30,
        max_retries=3
    )


@pytest.fixture(name="config_minimal")
def config_minimal_fixture():
    """Fixture to provide a minimal MemMachineMemoryClientConfig instance."""
    return MemMachineMemoryClientConfig(
        base_url="http://localhost:8080"
    )


@pytest.fixture(name="mock_memmachine_client")
def mock_memmachine_client_fixture():
    """Fixture to provide a mocked MemMachineClient."""
    mock_client = Mock()
    mock_client.base_url = "http://localhost:8080"
    return mock_client


@pytest.fixture(name="mock_project")
def mock_project_fixture():
    """Fixture to provide a mocked Project instance."""
    mock_project = Mock()
    mock_project.org_id = "test_org"
    mock_project.project_id = "test_project"
    return mock_project


async def test_memmachine_memory_client_success(
    config: MemMachineMemoryClientConfig,
    mock_builder: Mock,
    mock_memmachine_client: Mock,
    mock_project: Mock
):
    """Test successful initialization of memmachine memory client."""
    mock_memmachine_client.create_project.return_value = mock_project
    
    # Patch where the import happens - inside the function
    with patch("memmachine.MemMachineClient", return_value=mock_memmachine_client):
        # @register_memory wraps the function with asynccontextmanager, so use async with
        async with memmachine_memory_client(config, mock_builder) as editor:
            assert editor is not None
            # Verify client was initialized correctly
            mock_memmachine_client.create_project.assert_called_once_with(
                org_id="test_org",
                project_id="test_project",
                description="NeMo Agent Toolkit project: test_project"
            )


async def test_memmachine_memory_client_minimal_config(
    config_minimal: MemMachineMemoryClientConfig,
    mock_builder: Mock,
    mock_memmachine_client: Mock
):
    """Test initialization with minimal config (no org_id/project_id)."""
    with patch("memmachine.MemMachineClient", return_value=mock_memmachine_client):
        # @register_memory wraps the function with asynccontextmanager, so use async with
        async with memmachine_memory_client(config_minimal, mock_builder) as editor:
            assert editor is not None
            # Should not create project if org_id/project_id not provided
            mock_memmachine_client.create_project.assert_not_called()


async def test_memmachine_memory_client_import_error(
    config: MemMachineMemoryClientConfig,
    mock_builder: Mock
):
    """Test that ImportError is raised when memmachine package is not installed."""
    # Mock the import to raise ImportError
    # We need to patch the import inside the function, so patch where it's imported from
    import builtins
    original_import = builtins.__import__
    
    def import_side_effect(name, *args, **kwargs):
        if name == "memmachine":
            raise ImportError("No module named 'memmachine'")
        return original_import(name, *args, **kwargs)
    
    with patch("builtins.__import__", side_effect=import_side_effect):
        with pytest.raises(ImportError, match="Could not import MemMachineClient"):
            async with memmachine_memory_client(config, mock_builder):
                pass


async def test_memmachine_memory_client_initialization_error(
    config: MemMachineMemoryClientConfig,
    mock_builder: Mock
):
    """Test that RuntimeError is raised when client initialization fails."""
    with patch("memmachine.MemMachineClient", side_effect=ValueError("base_url is required")):
        with pytest.raises(RuntimeError, match="Failed to initialize MemMachineClient"):
            async with memmachine_memory_client(config, mock_builder):
                pass


async def test_memmachine_memory_client_project_creation_failure(
    config: MemMachineMemoryClientConfig,
    mock_builder: Mock,
    mock_memmachine_client: Mock
):
    """Test that editor still works if project creation fails."""
    mock_memmachine_client.create_project.side_effect = Exception("Project creation failed")
    
    with patch("memmachine.MemMachineClient", return_value=mock_memmachine_client):
        # Should not raise exception, should fall back to using client directly
        # @register_memory wraps the function with asynccontextmanager, so use async with
        async with memmachine_memory_client(config, mock_builder) as editor:
            assert editor is not None
            # Project creation should have been attempted
            mock_memmachine_client.create_project.assert_called_once()




async def test_memmachine_memory_client_config_validation():
    """Test that MemMachineMemoryClientConfig validates required fields."""
    # base_url is required
    with pytest.raises(Exception):  # Pydantic validation error
        MemMachineMemoryClientConfig()
    
    # Should work with base_url
    config = MemMachineMemoryClientConfig(base_url="http://localhost:8080")
    assert config.base_url == "http://localhost:8080"
    assert config.timeout == 30
    assert config.max_retries == 3


async def test_memmachine_memory_client_with_retry_mixin(
    config: MemMachineMemoryClientConfig,
    mock_builder: Mock,
    mock_memmachine_client: Mock,
    mock_project: Mock
):
    """Test that retry mixin is applied when config has retry settings."""
    mock_memmachine_client.create_project.return_value = mock_project
    
    # Add retry configuration
    config.num_retries = 5
    config.retry_on_status_codes = [500, 502, 503]
    config.retry_on_errors = ["ConnectionError"]
    
    with patch("memmachine.MemMachineClient", return_value=mock_memmachine_client):
        with patch("nat.plugins.memmachine.memory.patch_with_retry") as mock_patch:
            mock_patch.return_value = Mock()
            # @register_memory wraps the function with asynccontextmanager, so use async with
            async with memmachine_memory_client(config, mock_builder) as editor:
                assert editor is not None
                # Verify patch_with_retry was called
                mock_patch.assert_called_once()
                call_kwargs = mock_patch.call_args.kwargs
                assert call_kwargs["retries"] == 5
                assert call_kwargs["retry_codes"] == [500, 502, 503]
                assert call_kwargs["retry_on_messages"] == ["ConnectionError"]
