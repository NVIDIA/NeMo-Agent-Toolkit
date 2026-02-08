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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from nat.llm.dynamo_llm import DynamoModelConfig
from nat.llm.dynamo_llm import DynamoPrefixContext
from nat.llm.dynamo_llm import create_httpx_client_with_dynamo_hooks
from nat.llm.utils.constants import LLMHeaderPrefix

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
            "prediction_trie_path",
        })

        assert field_names == expected
        assert isinstance(field_names, frozenset)  # Ensure immutability


# ---------------------------------------------------------------------------
# Context Variable Tests
# ---------------------------------------------------------------------------


class TestDynamoPrefixContext:
    """Tests for DynamoPrefixContext singleton class."""

    def test_auto_generates_depth_based_prefix(self):
        """Test that get() auto-generates a depth-based prefix when no override is set."""
        DynamoPrefixContext.clear()

        # get() always returns a value - auto-generated if no override
        prefix = DynamoPrefixContext.get()
        assert prefix is not None
        assert "-d0" in prefix  # Depth 0 at root level

    def test_set_and_get_override_prefix_id(self):
        """Test setting and getting an override prefix ID."""
        DynamoPrefixContext.clear()

        # Set override
        DynamoPrefixContext.set("test-prefix-123")
        assert DynamoPrefixContext.get() == "test-prefix-123"

        # Clean up
        DynamoPrefixContext.clear()

    def test_clear_removes_override_but_auto_generates(self):
        """Test that clear() removes override but get() still returns auto-generated value."""
        DynamoPrefixContext.set("test-prefix-456")
        assert DynamoPrefixContext.get() == "test-prefix-456"

        DynamoPrefixContext.clear()
        # After clear, get() returns auto-generated depth-based prefix
        prefix = DynamoPrefixContext.get()
        assert prefix is not None
        assert prefix != "test-prefix-456"
        assert "-d0" in prefix

    def test_overwrite_prefix_id(self):
        """Test that setting a new prefix ID overwrites the old one."""
        DynamoPrefixContext.clear()

        DynamoPrefixContext.set("first-prefix")
        assert DynamoPrefixContext.get() == "first-prefix"

        DynamoPrefixContext.set("second-prefix")
        assert DynamoPrefixContext.get() == "second-prefix"

        DynamoPrefixContext.clear()

    def test_scope_context_manager(self):
        """Test the scope context manager with override prefix."""
        DynamoPrefixContext.clear()

        with DynamoPrefixContext.scope("scoped-prefix-789"):
            assert DynamoPrefixContext.get() == "scoped-prefix-789"

        # After exiting scope, returns to auto-generated
        prefix = DynamoPrefixContext.get()
        assert prefix != "scoped-prefix-789"
        assert "-d0" in prefix

    def test_scope_context_manager_cleanup_on_exception(self):
        """Test that scope context manager restores state even on exception."""
        DynamoPrefixContext.clear()

        with pytest.raises(ValueError):
            with DynamoPrefixContext.scope("error-prefix"):
                assert DynamoPrefixContext.get() == "error-prefix"
                raise ValueError("Test exception")

        # After exception, returns to auto-generated
        prefix = DynamoPrefixContext.get()
        assert prefix != "error-prefix"
        assert "-d0" in prefix

    def test_scope_nested_restores_outer(self):
        """Test that nested scopes properly restore outer scope value."""
        DynamoPrefixContext.clear()

        with DynamoPrefixContext.scope("outer"):
            assert DynamoPrefixContext.get() == "outer"
            with DynamoPrefixContext.scope("inner"):
                assert DynamoPrefixContext.get() == "inner"
            # After inner scope exits, outer value is restored
            assert DynamoPrefixContext.get() == "outer"

        # After outer scope exits, returns to auto-generated
        prefix = DynamoPrefixContext.get()
        assert prefix != "outer"
        assert "-d0" in prefix

    def test_is_set_always_true(self):
        """Test that is_set() always returns True since IDs are auto-generated."""
        DynamoPrefixContext.clear()
        assert DynamoPrefixContext.is_set() is True


# ---------------------------------------------------------------------------
# HTTPX Client Creation Tests
# ---------------------------------------------------------------------------


class TestCreateHttpxClient:
    """Tests for create_httpx_client_with_dynamo_hooks."""

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

    def test_creates_client_with_custom_transport(self):
        """Test that create_httpx_client_with_dynamo_hooks uses _DynamoTransport."""
        from nat.llm.dynamo_llm import _DynamoTransport

        client = create_httpx_client_with_dynamo_hooks(
            prefix_template="test-{uuid}",
            total_requests=7,
            osl="HIGH",
            iat="LOW",
            timeout=120.0,
            prediction_lookup=None,
        )

        # Verify client uses custom transport
        assert isinstance(client._transport, _DynamoTransport)

        # Verify transport has correct values
        assert client._transport._total_requests == 7
        assert client._transport._osl == "HIGH"
        assert client._transport._iat == "LOW"

        # Verify timeout
        assert client.timeout.read == 120.0


# ---------------------------------------------------------------------------
# _DynamoTransport Tests
# ---------------------------------------------------------------------------


class TestDynamoTransport:
    """Tests for _DynamoTransport custom transport wrapper."""

    async def test_transport_injects_headers(self):
        """Test that _DynamoTransport injects HTTP headers."""
        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport

        # Create mock base transport
        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        # Create transport with static values
        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=15,
            osl="HIGH",
            iat="LOW",
            prediction_lookup=None,
        )

        # Set prefix ID via context
        DynamoPrefixContext.set("test-prefix-123")

        # Create a request
        request = httpx.Request("POST", "https://api.example.com/chat")

        # Handle request (should inject headers)
        await transport.handle_async_request(request)

        # Get the request that was passed to mock transport
        call_args = mock_transport.handle_async_request.call_args
        modified_request = call_args[0][0]

        # Verify headers were injected
        prefix = f"{LLMHeaderPrefix.DYNAMO.value}"
        assert modified_request.headers[f"{prefix}-id"] == "test-prefix-123"
        assert modified_request.headers[f"{prefix}-total-requests"] == "15"
        assert modified_request.headers[f"{prefix}-osl"] == "HIGH"
        assert modified_request.headers[f"{prefix}-iat"] == "LOW"

        # Cleanup
        DynamoPrefixContext.clear()

    async def test_transport_injects_nvext_annotations(self):
        """Test that _DynamoTransport injects nvext.annotations in request body."""
        import json

        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport

        # Create mock base transport
        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        # Create transport
        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=10,
            osl="MEDIUM",
            iat="HIGH",
            prediction_lookup=None,
        )

        # Set prefix ID
        DynamoPrefixContext.set("eval-q001")

        # Create a POST request with JSON body
        original_body = {"model": "test", "messages": []}
        request = httpx.Request(
            "POST",
            "https://api.example.com/chat",
            json=original_body,
        )

        # Handle request
        await transport.handle_async_request(request)

        # Get the request that was passed to mock transport
        call_args = mock_transport.handle_async_request.call_args
        modified_request = call_args[0][0]

        # Parse the modified body
        body = json.loads(modified_request.content.decode("utf-8"))

        # Verify nvext.annotations was injected
        assert "nvext" in body
        assert "annotations" in body["nvext"]
        annotations = body["nvext"]["annotations"]

        assert "prefix_id:eval-q001" in annotations
        assert "total_requests:10" in annotations
        assert "osl:MEDIUM" in annotations
        assert "iat:HIGH" in annotations

        # Cleanup
        DynamoPrefixContext.clear()

    async def test_transport_merges_existing_annotations(self):
        """Test that existing nvext.annotations are preserved (non-conflicting)."""
        import json

        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport

        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=5,
            osl="LOW",
            iat="MEDIUM",
            prediction_lookup=None,
        )

        DynamoPrefixContext.set("merge-test")

        # Create request with existing nvext.annotations
        original_body = {
            "model": "test",
            "nvext": {
                "annotations": [
                    "custom_key:custom_value",
                    "iat:SHOULD_BE_REPLACED",  # Should be overridden
                ]
            }
        }
        request = httpx.Request("POST", "https://api.example.com/chat", json=original_body)

        # Handle request
        await transport.handle_async_request(request)

        # Get modified request
        modified_request = mock_transport.handle_async_request.call_args[0][0]
        body = json.loads(modified_request.content.decode("utf-8"))

        annotations = body["nvext"]["annotations"]

        # Our annotations should be first
        assert annotations[0] == "prefix_id:merge-test"
        assert annotations[1] == "total_requests:5"
        assert annotations[2] == "osl:LOW"
        assert annotations[3] == "iat:MEDIUM"

        # Custom annotation preserved
        assert "custom_key:custom_value" in annotations

        # Old conflicting annotation should NOT be present
        assert "iat:SHOULD_BE_REPLACED" not in annotations

        DynamoPrefixContext.clear()

    async def test_transport_handles_non_json_gracefully(self):
        """Test that non-JSON bodies don't cause failures."""
        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport

        mock_response = httpx.Response(200, text="ok")
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=1,
            osl="LOW",
            iat="LOW",
            prediction_lookup=None,
        )

        DynamoPrefixContext.set("non-json-test")

        # Create request with non-JSON content
        request = httpx.Request("POST", "https://api.example.com/chat", content=b"plain text")

        # Should not raise
        await transport.handle_async_request(request)

        # Headers should still be injected
        modified_request = mock_transport.handle_async_request.call_args[0][0]
        prefix = f"{LLMHeaderPrefix.DYNAMO.value}"
        assert modified_request.headers[f"{prefix}-id"] == "non-json-test"
        assert modified_request.headers[f"{prefix}-total-requests"] == "1"
        assert modified_request.headers[f"{prefix}-osl"] == "LOW"
        assert modified_request.headers[f"{prefix}-iat"] == "LOW"

        DynamoPrefixContext.clear()

    async def test_transport_uses_prediction_override(self):
        """Test that prediction lookup overrides static config values."""
        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport
        from nat.profiler.prediction_trie.data_models import LLMCallPrediction
        from nat.profiler.prediction_trie.data_models import PredictionMetrics

        # Create mock prediction lookup
        mock_prediction = LLMCallPrediction(
            remaining_calls=PredictionMetrics(mean=25.0, p50=25.0, p90=30.0),
            output_tokens=PredictionMetrics(mean=2000.0, p50=2000.0, p90=2500.0),  # HIGH
            interarrival_ms=PredictionMetrics(mean=50.0, p50=50.0, p90=70.0),  # LOW
        )

        mock_lookup = MagicMock()
        mock_lookup.find = MagicMock(return_value=mock_prediction)

        # Create mock base transport
        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        # Create transport with static values that should be overridden
        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=10,  # Should be overridden to 25
            osl="MEDIUM",  # Should be overridden to HIGH
            iat="MEDIUM",  # Should be overridden to LOW
            prediction_lookup=mock_lookup,
        )

        # Set prefix ID
        DynamoPrefixContext.set("prediction-test")

        # Create a POST request
        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})

        # Handle request
        await transport.handle_async_request(request)

        # Get the modified request
        modified_request = mock_transport.handle_async_request.call_args[0][0]

        # Verify prediction values were used instead of static config in headers
        prefix = f"{LLMHeaderPrefix.DYNAMO.value}"
        assert modified_request.headers[f"{prefix}-total-requests"] == "25"  # from prediction
        assert modified_request.headers[f"{prefix}-osl"] == "HIGH"  # from prediction (2500 tokens)
        assert modified_request.headers[f"{prefix}-iat"] == "LOW"  # from prediction (50ms)

        # Verify prediction values also used in nvext.annotations
        import json
        body = json.loads(modified_request.content.decode("utf-8"))
        annotations = body["nvext"]["annotations"]

        assert "total_requests:25" in annotations
        assert "osl:HIGH" in annotations
        assert "iat:LOW" in annotations

        # Verify lookup was called
        assert mock_lookup.find.called

        DynamoPrefixContext.clear()


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
