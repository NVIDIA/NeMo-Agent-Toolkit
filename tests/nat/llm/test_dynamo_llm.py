# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for the Dynamo LLM provider."""

from unittest.mock import MagicMock

import pytest

from nat.llm.dynamo_llm import DynamoModelConfig
from nat.llm.dynamo_llm import DynamoPrefixContext
from nat.llm.dynamo_llm import _create_dynamo_request_hook
from nat.llm.dynamo_llm import create_httpx_client_with_dynamo_hooks

# ---------------------------------------------------------------------------
# DynamoModelConfig Tests
# ---------------------------------------------------------------------------


class TestDynamoModelConfig:
    """Tests for DynamoModelConfig configuration class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = DynamoModelConfig(model_name="test-model")

        assert config.model_name == "test-model"
        assert config.prefix_template == "nat-dynamo-{uuid}"  # Enabled by default
        assert config.prefix_total_requests == 10
        assert config.prefix_osl == "MEDIUM"
        assert config.prefix_iat == "MEDIUM"
        assert config.request_timeout == 600.0

    def test_custom_prefix_values(self):
        """Test custom prefix parameter values."""
        config = DynamoModelConfig(
            model_name="test-model",
            prefix_template="session-{uuid}",
            prefix_total_requests=20,
            prefix_osl="HIGH",
            prefix_iat="LOW",
            request_timeout=300.0,
        )

        assert config.prefix_template == "session-{uuid}"
        assert config.prefix_total_requests == 20
        assert config.prefix_osl == "HIGH"
        assert config.prefix_iat == "LOW"
        assert config.request_timeout == 300.0

    def test_disable_prefix_headers(self):
        """Test that prefix headers can be disabled by setting prefix_template to None."""
        config = DynamoModelConfig(
            model_name="test-model",
            prefix_template=None,  # Explicitly disable prefix headers
        )

        assert config.prefix_template is None

    def test_prefix_total_requests_validation(self):
        """Test that prefix_total_requests validates bounds."""
        # Valid range
        config = DynamoModelConfig(model_name="test-model", prefix_total_requests=1)
        assert config.prefix_total_requests == 1

        config = DynamoModelConfig(model_name="test-model", prefix_total_requests=50)
        assert config.prefix_total_requests == 50

        # Invalid: below minimum
        with pytest.raises(ValueError):
            DynamoModelConfig(model_name="test-model", prefix_total_requests=0)

        # Invalid: above maximum
        with pytest.raises(ValueError):
            DynamoModelConfig(model_name="test-model", prefix_total_requests=51)

    def test_prefix_level_validation(self):
        """Test that prefix_osl and prefix_iat only accept valid values."""
        # Valid values
        for level in ["LOW", "MEDIUM", "HIGH"]:
            config = DynamoModelConfig(model_name="test-model", prefix_osl=level, prefix_iat=level)
            assert config.prefix_osl == level
            assert config.prefix_iat == level

        # Invalid values
        with pytest.raises(ValueError):
            DynamoModelConfig(model_name="test-model", prefix_osl="INVALID")

        with pytest.raises(ValueError):
            DynamoModelConfig(model_name="test-model", prefix_iat="INVALID")

    def test_request_timeout_validation(self):
        """Test that request_timeout validates positive values."""
        config = DynamoModelConfig(model_name="test-model", request_timeout=1.0)
        assert config.request_timeout == 1.0

        with pytest.raises(ValueError):
            DynamoModelConfig(model_name="test-model", request_timeout=0.0)

        with pytest.raises(ValueError):
            DynamoModelConfig(model_name="test-model", request_timeout=-1.0)

    def test_inherits_openai_config_fields(self):
        """Test that DynamoModelConfig inherits OpenAI fields."""
        config = DynamoModelConfig(
            model_name="test-model",
            base_url="http://localhost:8000/v1",
            temperature=0.7,
            top_p=0.9,
        )

        assert config.base_url == "http://localhost:8000/v1"
        assert config.temperature == 0.7
        assert config.top_p == 0.9

    def test_get_dynamo_field_names(self):
        """Test that get_dynamo_field_names returns the correct field set."""
        field_names = DynamoModelConfig.get_dynamo_field_names()

        expected = frozenset({
            "prefix_template",
            "prefix_total_requests",
            "prefix_osl",
            "prefix_iat",
            "request_timeout",
        })

        assert field_names == expected
        assert isinstance(field_names, frozenset)  # Ensure immutability


# ---------------------------------------------------------------------------
# Context Variable Tests
# ---------------------------------------------------------------------------


class TestDynamoPrefixContext:
    """Tests for DynamoPrefixContext singleton class."""

    def test_set_and_get_prefix_id(self):
        """Test setting and getting prefix ID."""
        # Ensure clean state
        DynamoPrefixContext.clear()
        assert DynamoPrefixContext.get() is None

        # Set and get
        DynamoPrefixContext.set("test-prefix-123")
        assert DynamoPrefixContext.get() == "test-prefix-123"

        # Clean up
        DynamoPrefixContext.clear()

    def test_clear_prefix_id(self):
        """Test clearing prefix ID."""
        DynamoPrefixContext.set("test-prefix-456")
        assert DynamoPrefixContext.get() == "test-prefix-456"

        DynamoPrefixContext.clear()
        assert DynamoPrefixContext.get() is None

    def test_overwrite_prefix_id(self):
        """Test that setting a new prefix ID overwrites the old one."""
        DynamoPrefixContext.clear()

        DynamoPrefixContext.set("first-prefix")
        assert DynamoPrefixContext.get() == "first-prefix"

        DynamoPrefixContext.set("second-prefix")
        assert DynamoPrefixContext.get() == "second-prefix"

        DynamoPrefixContext.clear()

    def test_scope_context_manager(self):
        """Test the scope context manager for automatic cleanup."""
        DynamoPrefixContext.clear()
        assert DynamoPrefixContext.get() is None

        with DynamoPrefixContext.scope("scoped-prefix-789"):
            assert DynamoPrefixContext.get() == "scoped-prefix-789"

        # Should be cleared after exiting context
        assert DynamoPrefixContext.get() is None

    def test_scope_context_manager_cleanup_on_exception(self):
        """Test that scope context manager clears prefix ID even on exception."""
        DynamoPrefixContext.clear()

        with pytest.raises(ValueError):
            with DynamoPrefixContext.scope("error-prefix"):
                assert DynamoPrefixContext.get() == "error-prefix"
                raise ValueError("Test exception")

        # Should still be cleared after exception
        assert DynamoPrefixContext.get() is None

    def test_scope_nested_replaces_then_clears(self):
        """Test that nested scopes work but outer scope is lost after inner exits."""
        DynamoPrefixContext.clear()

        with DynamoPrefixContext.scope("outer"):
            assert DynamoPrefixContext.get() == "outer"
            with DynamoPrefixContext.scope("inner"):
                assert DynamoPrefixContext.get() == "inner"
            # After inner scope exits, it clears - outer value is lost
            assert DynamoPrefixContext.get() is None

        # Still None after outer exits
        assert DynamoPrefixContext.get() is None


# ---------------------------------------------------------------------------
# Request Hook Tests
# ---------------------------------------------------------------------------


class TestDynamoRequestHook:
    """Tests for the Dynamo request hook that injects headers."""

    @pytest.fixture(autouse=True)
    def clean_context(self):
        """Ensure clean context before and after each test."""
        DynamoPrefixContext.clear()
        yield
        DynamoPrefixContext.clear()

    @pytest.mark.asyncio
    async def test_hook_injects_headers(self):
        """Test that the hook injects all required Dynamo headers."""
        hook = _create_dynamo_request_hook(
            prefix_template="test-{uuid}",
            total_requests=15,
            osl="HIGH",
            iat="LOW",
        )

        # Create a mock request
        mock_request = MagicMock()
        mock_request.headers = {}

        await hook(mock_request)

        assert "x-prefix-id" in mock_request.headers
        assert mock_request.headers["x-prefix-id"].startswith("test-")
        assert mock_request.headers["x-prefix-total-requests"] == "15"
        assert mock_request.headers["x-prefix-osl"] == "HIGH"
        assert mock_request.headers["x-prefix-iat"] == "LOW"

    @pytest.mark.asyncio
    async def test_hook_uses_context_prefix_id(self):
        """Test that the hook uses context variable prefix ID when set."""
        hook = _create_dynamo_request_hook(
            prefix_template="template-{uuid}",
            total_requests=10,
            osl="MEDIUM",
            iat="MEDIUM",
        )

        # Set context prefix ID
        DynamoPrefixContext.set("context-prefix-abc")

        mock_request = MagicMock()
        mock_request.headers = {}

        await hook(mock_request)

        # Should use context prefix ID, not generate from template
        assert mock_request.headers["x-prefix-id"] == "context-prefix-abc"

    @pytest.mark.asyncio
    async def test_hook_uses_same_id_for_all_requests(self):
        """Test that the hook uses the same prefix ID for all requests from the same client.

        This ensures Dynamo's KV cache optimization works across multi-turn conversations.
        All requests from the same client (created with the same hook) should share
        the same prefix ID to enable KV cache reuse.
        """
        hook = _create_dynamo_request_hook(
            prefix_template="session-{uuid}",
            total_requests=10,
            osl="MEDIUM",
            iat="MEDIUM",
        )

        prefix_ids = set()
        for _ in range(10):
            mock_request = MagicMock()
            mock_request.headers = {}
            await hook(mock_request)
            prefix_ids.add(mock_request.headers["x-prefix-id"])

        # All requests should share the SAME prefix ID (for KV cache optimization)
        assert len(prefix_ids) == 1
        # And it should start with our template
        assert list(prefix_ids)[0].startswith("session-")

    @pytest.mark.asyncio
    async def test_different_hooks_have_different_ids(self):
        """Test that different hooks (different clients) get different prefix IDs."""
        prefix_ids = set()
        for _ in range(5):
            hook = _create_dynamo_request_hook(
                prefix_template="session-{uuid}",
                total_requests=10,
                osl="MEDIUM",
                iat="MEDIUM",
            )
            mock_request = MagicMock()
            mock_request.headers = {}
            await hook(mock_request)
            prefix_ids.add(mock_request.headers["x-prefix-id"])

        # Different hooks should have different prefix IDs
        assert len(prefix_ids) == 5

    @pytest.mark.asyncio
    async def test_hook_default_prefix_template(self):
        """Test that the hook uses default prefix format when template is None."""
        hook = _create_dynamo_request_hook(
            prefix_template=None,
            total_requests=10,
            osl="MEDIUM",
            iat="MEDIUM",
        )

        mock_request = MagicMock()
        mock_request.headers = {}

        await hook(mock_request)

        # Should use default "nat-dynamo-{id}" format
        assert mock_request.headers["x-prefix-id"].startswith("nat-dynamo-")

    @pytest.mark.asyncio
    async def test_hook_normalizes_case(self):
        """Test that OSL and IAT values are uppercased."""
        hook = _create_dynamo_request_hook(
            prefix_template=None,
            total_requests=10,
            osl="low",
            iat="high",
        )

        mock_request = MagicMock()
        mock_request.headers = {}

        await hook(mock_request)

        assert mock_request.headers["x-prefix-osl"] == "LOW"
        assert mock_request.headers["x-prefix-iat"] == "HIGH"

    @pytest.mark.asyncio
    async def test_hook_template_without_uuid_placeholder(self):
        """Test that a template without {uuid} placeholder uses template as-is."""
        hook = _create_dynamo_request_hook(
            prefix_template="static-prefix-no-uuid",
            total_requests=10,
            osl="MEDIUM",
            iat="MEDIUM",
        )

        mock_request = MagicMock()
        mock_request.headers = {}
        await hook(mock_request)

        # Template used as-is when no {uuid} placeholder
        assert mock_request.headers["x-prefix-id"] == "static-prefix-no-uuid"


# ---------------------------------------------------------------------------
# HTTPX Client Creation Tests
# ---------------------------------------------------------------------------


class TestCreateHttpxClient:
    """Tests for create_httpx_client_with_dynamo_hooks."""

    def test_creates_client_with_event_hooks(self):
        """Test that the function creates an httpx client with event hooks."""
        client = create_httpx_client_with_dynamo_hooks(
            prefix_template="test-{uuid}",
            total_requests=10,
            osl="MEDIUM",
            iat="MEDIUM",
        )

        # Check that event hooks are configured
        assert "request" in client.event_hooks
        assert len(client.event_hooks["request"]) == 1

    def test_uses_custom_timeout(self):
        """Test that the function uses the provided timeout."""
        client = create_httpx_client_with_dynamo_hooks(
            prefix_template=None,
            total_requests=10,
            osl="MEDIUM",
            iat="MEDIUM",
            timeout=120.0,
        )

        assert client.timeout.connect == 120.0
        assert client.timeout.read == 120.0
        assert client.timeout.write == 120.0

    def test_uses_default_timeout(self):
        """Test that the function uses default timeout when not specified."""
        client = create_httpx_client_with_dynamo_hooks(
            prefix_template=None,
            total_requests=10,
            osl="MEDIUM",
            iat="MEDIUM",
        )

        assert client.timeout.connect == 600.0


# ---------------------------------------------------------------------------
# Provider Registration Tests
# ---------------------------------------------------------------------------


class TestDynamoLLMProvider:
    """Tests for the dynamo_llm provider registration."""

    def test_dynamo_model_config_type_name(self):
        """Test that DynamoModelConfig has the correct type name."""
        assert DynamoModelConfig.static_type() == "dynamo"

    def test_dynamo_model_config_full_type(self):
        """Test that DynamoModelConfig has the correct full type."""
        assert DynamoModelConfig.static_full_type() == "nat.llm/dynamo"
