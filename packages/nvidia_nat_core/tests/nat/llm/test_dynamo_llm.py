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
        assert config.prefix_osl == 512
        assert config.prefix_iat == 250
        assert config.prefix_use_raw_values is True
        assert config.disable_headers is True
        assert config.request_timeout == 600.0

    def test_custom_prefix_values(self):
        """Test custom prefix parameter values."""
        config = DynamoModelConfig(
            model_name="test-model",
            prefix_template="session-{uuid}",
            prefix_total_requests=20,
            prefix_osl=2048,
            prefix_iat=50,
            request_timeout=300.0,
        )

        assert config.prefix_template == "session-{uuid}"
        assert config.prefix_total_requests == 20
        assert config.prefix_osl == 2048
        assert config.prefix_iat == 50
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

    def test_prefix_osl_iat_accept_integers(self):
        """Test that prefix_osl and prefix_iat accept integer values."""
        config = DynamoModelConfig(model_name="test-model", prefix_osl=1024, prefix_iat=100)
        assert config.prefix_osl == 1024
        assert config.prefix_iat == 100

    def test_prefix_osl_iat_reject_invalid(self):
        """Test that prefix_osl and prefix_iat reject invalid values."""
        with pytest.raises(ValueError):
            DynamoModelConfig(model_name="test-model", prefix_osl=0)

        with pytest.raises(ValueError):
            DynamoModelConfig(model_name="test-model", prefix_iat=0)

        with pytest.raises(ValueError):
            DynamoModelConfig(model_name="test-model", prefix_osl="INVALID")

    def test_backward_compat_categorical_strings(self):
        """Test that categorical string values (LOW/MEDIUM/HIGH) are coerced to integers."""
        config = DynamoModelConfig(model_name="test-model", prefix_osl="LOW", prefix_iat="LOW")
        assert config.prefix_osl == 128
        assert config.prefix_iat == 50

        config = DynamoModelConfig(model_name="test-model", prefix_osl="MEDIUM", prefix_iat="MEDIUM")
        assert config.prefix_osl == 512
        assert config.prefix_iat == 250

        config = DynamoModelConfig(model_name="test-model", prefix_osl="HIGH", prefix_iat="HIGH")
        assert config.prefix_osl == 2048
        assert config.prefix_iat == 750

    def test_backward_compat_case_insensitive(self):
        """Test that categorical coercion is case-insensitive."""
        config = DynamoModelConfig(model_name="test-model", prefix_osl="low", prefix_iat="high")
        assert config.prefix_osl == 128
        assert config.prefix_iat == 750

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
            "prefix_use_raw_values",
            "request_timeout",
            "prediction_trie_path",
            "disable_headers",
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
            osl=512,
            iat=250,
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
            osl=512,
            iat=250,
        )

        assert client.timeout.connect == 600.0

    def test_creates_client_with_custom_transport(self):
        """Test that create_httpx_client_with_dynamo_hooks uses _DynamoTransport."""
        from nat.llm.dynamo_llm import _DynamoTransport

        client = create_httpx_client_with_dynamo_hooks(
            prefix_template="test-{uuid}",
            total_requests=7,
            osl=2048,
            iat=50,
            timeout=120.0,
            prediction_lookup=None,
        )

        # Verify client uses custom transport
        assert isinstance(client._transport, _DynamoTransport)

        # Verify transport has correct values
        assert client._transport._total_requests == 7
        assert client._transport._osl == 2048
        assert client._transport._iat == 50
        assert client._transport._use_raw_values is True
        assert client._transport._disable_headers is True

        # Verify timeout
        assert client.timeout.read == 120.0


# ---------------------------------------------------------------------------
# _DynamoTransport Tests
# ---------------------------------------------------------------------------


class TestDynamoTransport:
    """Tests for _DynamoTransport custom transport wrapper."""

    async def test_transport_injects_raw_headers_by_default(self):
        """Test that _DynamoTransport injects raw integer values in HTTP headers by default."""
        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport

        # Create mock base transport
        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        # Create transport with integer values (default use_raw_values=True)
        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=15,
            osl=2048,
            iat=50,
            prediction_lookup=None,
            disable_headers=False,
        )

        # Set prefix ID via context
        DynamoPrefixContext.set("test-prefix-123")

        # Create a request
        request = httpx.Request("POST", "https://api.example.com/chat")

        # Handle request (should inject raw integer headers)
        await transport.handle_async_request(request)

        # Get the request that was passed to mock transport
        call_args = mock_transport.handle_async_request.call_args
        modified_request = call_args[0][0]

        # Verify headers were injected with raw integer values
        prefix = f"{LLMHeaderPrefix.DYNAMO}"
        assert modified_request.headers[f"{prefix}-id"] == "test-prefix-123"
        assert modified_request.headers[f"{prefix}-total-requests"] == "15"
        assert modified_request.headers[f"{prefix}-osl"] == "2048"
        assert modified_request.headers[f"{prefix}-iat"] == "50"

        # Cleanup
        DynamoPrefixContext.clear()

    async def test_transport_injects_categorical_headers_when_raw_disabled(self):
        """Test that _DynamoTransport converts to categorical values when use_raw_values=False."""
        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport

        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        # osl=2048 -> HIGH (>= 1024), iat=50 -> LOW (< 100)
        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=15,
            osl=2048,
            iat=50,
            prediction_lookup=None,
            use_raw_values=False,
            disable_headers=False,
        )

        DynamoPrefixContext.set("test-categorical")

        request = httpx.Request("POST", "https://api.example.com/chat")
        await transport.handle_async_request(request)

        modified_request = mock_transport.handle_async_request.call_args[0][0]
        prefix = f"{LLMHeaderPrefix.DYNAMO}"
        assert modified_request.headers[f"{prefix}-osl"] == "HIGH"
        assert modified_request.headers[f"{prefix}-iat"] == "LOW"

        DynamoPrefixContext.clear()

    async def test_transport_injects_nvext_agent_hints(self):
        """Test that _DynamoTransport injects nvext.agent_hints in request body."""
        import json

        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport

        # Create mock base transport
        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        # Create transport with raw values (default)
        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=10,
            osl=512,
            iat=750,
            prediction_lookup=None,
            disable_headers=False,
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

        # Verify nvext.agent_hints was injected with raw integer values
        assert "nvext" in body
        assert "agent_hints" in body["nvext"]
        agent_hints = body["nvext"]["agent_hints"]

        assert agent_hints["prefix_id"] == "eval-q001"
        assert agent_hints["total_requests"] == 10
        assert agent_hints["osl"] == 512
        assert agent_hints["iat"] == 750

        # Cleanup
        DynamoPrefixContext.clear()

    async def test_transport_merges_existing_agent_hints(self):
        """Test that existing nvext.agent_hints are preserved (non-conflicting)."""
        import json

        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport

        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=5,
            osl=128,
            iat=250,
            prediction_lookup=None,
            disable_headers=False,
        )

        DynamoPrefixContext.set("merge-test")

        # Create request with existing nvext.agent_hints
        original_body = {
            "model": "test",
            "nvext": {
                "agent_hints": {
                    "custom_key": "custom_value",
                    "iat": "SHOULD_BE_REPLACED",  # Should be overridden
                }
            }
        }
        request = httpx.Request("POST", "https://api.example.com/chat", json=original_body)

        # Handle request
        await transport.handle_async_request(request)

        # Get modified request
        modified_request = mock_transport.handle_async_request.call_args[0][0]
        body = json.loads(modified_request.content.decode("utf-8"))

        agent_hints = body["nvext"]["agent_hints"]

        # Our hints should be present (raw integers)
        assert agent_hints["prefix_id"] == "merge-test"
        assert agent_hints["total_requests"] == 5
        assert agent_hints["osl"] == 128
        assert agent_hints["iat"] == 250

        # Custom hint preserved
        assert agent_hints["custom_key"] == "custom_value"

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
            osl=128,
            iat=50,
            prediction_lookup=None,
            disable_headers=False,
        )

        DynamoPrefixContext.set("non-json-test")

        # Create request with non-JSON content
        request = httpx.Request("POST", "https://api.example.com/chat", content=b"plain text")

        # Should not raise
        await transport.handle_async_request(request)

        # Headers should still be injected with raw values
        modified_request = mock_transport.handle_async_request.call_args[0][0]
        prefix = f"{LLMHeaderPrefix.DYNAMO}"
        assert modified_request.headers[f"{prefix}-id"] == "non-json-test"
        assert modified_request.headers[f"{prefix}-total-requests"] == "1"
        assert modified_request.headers[f"{prefix}-osl"] == "128"
        assert modified_request.headers[f"{prefix}-iat"] == "50"

        DynamoPrefixContext.clear()

    async def test_transport_uses_prediction_override_raw(self):
        """Test that prediction lookup overrides static config with raw values by default."""
        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport
        from nat.profiler.prediction_trie.data_models import LLMCallPrediction
        from nat.profiler.prediction_trie.data_models import PredictionMetrics

        # Create mock prediction lookup
        mock_prediction = LLMCallPrediction(
            remaining_calls=PredictionMetrics(mean=25.0, p50=25.0, p90=30.0),
            output_tokens=PredictionMetrics(mean=2000.0, p50=2000.0, p90=2500.0),
            interarrival_ms=PredictionMetrics(mean=50.0, p50=50.0, p90=70.0),
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
            total_requests=10,
            osl=512,
            iat=250,
            prediction_lookup=mock_lookup,
            disable_headers=False,
        )

        # Set prefix ID
        DynamoPrefixContext.set("prediction-test")

        # Create a POST request
        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})

        # Handle request
        await transport.handle_async_request(request)

        # Get the modified request
        modified_request = mock_transport.handle_async_request.call_args[0][0]

        # Verify raw prediction values in headers
        prefix = f"{LLMHeaderPrefix.DYNAMO}"
        assert modified_request.headers[f"{prefix}-total-requests"] == "25"
        assert modified_request.headers[f"{prefix}-osl"] == "2500"  # raw output_tokens.p90
        assert modified_request.headers[f"{prefix}-iat"] == "50"  # raw interarrival_ms.mean

        # Verify raw prediction values in nvext.agent_hints
        import json
        body = json.loads(modified_request.content.decode("utf-8"))
        agent_hints = body["nvext"]["agent_hints"]

        assert agent_hints["total_requests"] == 25
        assert agent_hints["osl"] == 2500
        assert agent_hints["iat"] == 50

        # Verify lookup was called
        assert mock_lookup.find.called

        DynamoPrefixContext.clear()

    async def test_transport_uses_prediction_override_categorical(self):
        """Test that prediction lookup converts to categories when use_raw_values=False."""
        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport
        from nat.profiler.prediction_trie.data_models import LLMCallPrediction
        from nat.profiler.prediction_trie.data_models import PredictionMetrics

        mock_prediction = LLMCallPrediction(
            remaining_calls=PredictionMetrics(mean=25.0, p50=25.0, p90=30.0),
            output_tokens=PredictionMetrics(mean=2000.0, p50=2000.0, p90=2500.0),  # >= 1024 -> HIGH
            interarrival_ms=PredictionMetrics(mean=50.0, p50=50.0, p90=70.0),  # < 100 -> LOW
        )

        mock_lookup = MagicMock()
        mock_lookup.find = MagicMock(return_value=mock_prediction)

        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=10,
            osl=512,
            iat=250,
            prediction_lookup=mock_lookup,
            use_raw_values=False,
            disable_headers=False,
        )

        DynamoPrefixContext.set("prediction-categorical")

        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
        await transport.handle_async_request(request)

        modified_request = mock_transport.handle_async_request.call_args[0][0]
        prefix = f"{LLMHeaderPrefix.DYNAMO}"
        assert modified_request.headers[f"{prefix}-osl"] == "HIGH"
        assert modified_request.headers[f"{prefix}-iat"] == "LOW"

        import json
        body = json.loads(modified_request.content.decode("utf-8"))
        agent_hints = body["nvext"]["agent_hints"]
        assert agent_hints["osl"] == "HIGH"
        assert agent_hints["iat"] == "LOW"

        DynamoPrefixContext.clear()

    async def test_transport_suppresses_headers_when_disabled(self):
        """Test that headers are NOT injected when disable_headers=True but nvext.agent_hints still are."""
        import json

        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport

        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        # disable_headers=True (default) -> no HTTP headers injected
        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=10,
            osl=512,
            iat=250,
            prediction_lookup=None,
            disable_headers=True,
        )

        DynamoPrefixContext.set("test-no-headers")

        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test", "messages": []})
        await transport.handle_async_request(request)

        modified_request = mock_transport.handle_async_request.call_args[0][0]
        prefix = f"{LLMHeaderPrefix.DYNAMO}"

        # HTTP headers should NOT be present
        assert f"{prefix}-id" not in modified_request.headers
        assert f"{prefix}-total-requests" not in modified_request.headers
        assert f"{prefix}-osl" not in modified_request.headers
        assert f"{prefix}-iat" not in modified_request.headers
        assert f"{prefix}-latency-sensitivity" not in modified_request.headers

        # nvext.agent_hints should still be present
        body = json.loads(modified_request.content.decode("utf-8"))
        assert "nvext" in body
        agent_hints = body["nvext"]["agent_hints"]
        assert agent_hints["prefix_id"] == "test-no-headers"
        assert agent_hints["total_requests"] == 10
        assert agent_hints["osl"] == 512
        assert agent_hints["iat"] == 250

        DynamoPrefixContext.clear()

    async def test_transport_injects_latency_sensitivity_header(self):
        """Test that _DynamoTransport injects latency-sensitivity HTTP header."""
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
            osl=512,
            iat=750,
            prediction_lookup=None,
            disable_headers=False,
        )

        # Set prefix ID
        DynamoPrefixContext.set("test-latency-123")

        # Create a request
        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})

        # Handle request (should inject latency-sensitivity header)
        await transport.handle_async_request(request)

        # Get the request that was passed to mock transport
        call_args = mock_transport.handle_async_request.call_args
        modified_request = call_args[0][0]

        # Verify latency-sensitivity header was injected with default MEDIUM
        prefix = f"{LLMHeaderPrefix.DYNAMO}"
        assert modified_request.headers[f"{prefix}-latency-sensitivity"] == "MEDIUM"

        # Cleanup
        DynamoPrefixContext.clear()

    async def test_transport_injects_latency_sensitivity_in_agent_hints(self):
        """Test that _DynamoTransport injects latency-sensitivity in nvext.agent_hints."""
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
            osl=512,
            iat=750,
            prediction_lookup=None,
            disable_headers=False,
        )

        # Set prefix ID
        DynamoPrefixContext.set("test-latency-ann")

        # Create a POST request with JSON body
        request = httpx.Request(
            "POST",
            "https://api.example.com/chat",
            json={
                "model": "test", "messages": []
            },
        )

        # Handle request
        await transport.handle_async_request(request)

        # Get the request that was passed to mock transport
        call_args = mock_transport.handle_async_request.call_args
        modified_request = call_args[0][0]

        # Parse the modified body
        body = json.loads(modified_request.content.decode("utf-8"))

        # Verify latency-sensitivity in agent_hints with default MEDIUM
        assert "nvext" in body
        assert "agent_hints" in body["nvext"]
        agent_hints = body["nvext"]["agent_hints"]
        assert agent_hints["latency_sensitivity"] == "MEDIUM"

        # Cleanup
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
