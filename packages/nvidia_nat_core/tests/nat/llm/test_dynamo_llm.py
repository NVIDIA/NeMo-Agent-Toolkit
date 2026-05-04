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

from nat.llm.dynamo_llm import CacheControlMode
from nat.llm.dynamo_llm import CachePinType
from nat.llm.dynamo_llm import DynamoModelConfig
from nat.llm.dynamo_llm import DynamoPrefixContext
from nat.llm.dynamo_llm import _create_httpx_client_with_dynamo_hooks

# ---------------------------------------------------------------------------
# DynamoModelConfig Tests
# ---------------------------------------------------------------------------


class TestDynamoModelConfig:
    """Tests for DynamoModelConfig configuration class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = DynamoModelConfig(model_name="test-model")

        assert config.model_name == "test-model"
        assert config.nvext_prefix_id_template == "nat-dynamo-{uuid}"  # Enabled by default
        assert config.nvext_prefix_total_requests == 10
        assert config.nvext_prefix_osl == 512
        assert config.nvext_prefix_iat == 250
        assert config.request_timeout == 600.0
        assert config.nvext_cache_pin_type == CachePinType.EPHEMERAL
        assert config.nvext_cache_control_mode == CacheControlMode.ALWAYS
        assert config.enable_nvext_hints is False
        assert config.nvext_max_sensitivity == 1000

    def test_enable_nvext_hints_toggle(self):
        """Test that enable_nvext_hints can be set to True."""
        config = DynamoModelConfig(model_name="test-model", enable_nvext_hints=True)
        assert config.enable_nvext_hints is True

        config = DynamoModelConfig(model_name="test-model", enable_nvext_hints=False)
        assert config.enable_nvext_hints is False

    def test_custom_prefix_values(self):
        """Test custom prefix parameter values."""
        config = DynamoModelConfig(
            model_name="test-model",
            nvext_prefix_id_template="session-{uuid}",
            nvext_prefix_total_requests=20,
            nvext_prefix_osl=2048,
            nvext_prefix_iat=50,
            request_timeout=300.0,
        )

        assert config.nvext_prefix_id_template == "session-{uuid}"
        assert config.nvext_prefix_total_requests == 20
        assert config.nvext_prefix_osl == 2048
        assert config.nvext_prefix_iat == 50
        assert config.request_timeout == 300.0

    def test_prefix_template_none_does_not_toggle_hints(self):
        """Test that setting nvext_prefix_id_template to None only clears the template value.

        Hint injection is controlled by enable_nvext_hints, not by this field,
        so this assignment does not affect whether hints are enabled or disabled.
        """
        config = DynamoModelConfig(
            model_name="test-model",
            nvext_prefix_id_template=None,
        )

        assert config.nvext_prefix_id_template is None

    def test_prefix_total_requests_validation(self):
        """Test that prefix_total_requests validates bounds."""
        # Valid range
        config = DynamoModelConfig(model_name="test-model", nvext_prefix_total_requests=1)
        assert config.nvext_prefix_total_requests == 1

        config = DynamoModelConfig(model_name="test-model", nvext_prefix_total_requests=50)
        assert config.nvext_prefix_total_requests == 50

        # Invalid: below minimum
        with pytest.raises(ValueError):
            DynamoModelConfig(model_name="test-model", nvext_prefix_total_requests=0)

        # Invalid: above maximum
        with pytest.raises(ValueError):
            DynamoModelConfig(model_name="test-model", nvext_prefix_total_requests=51)

    def test_prefix_osl_iat_accept_integers(self):
        """Test that prefix_osl and prefix_iat accept integer values."""
        config = DynamoModelConfig(model_name="test-model", nvext_prefix_osl=1024, nvext_prefix_iat=100)
        assert config.nvext_prefix_osl == 1024
        assert config.nvext_prefix_iat == 100

    def test_prefix_osl_iat_reject_invalid(self):
        """Test that prefix_osl and prefix_iat reject invalid values."""
        with pytest.raises(ValueError):
            DynamoModelConfig(model_name="test-model", nvext_prefix_osl=0)

        with pytest.raises(ValueError):
            DynamoModelConfig(model_name="test-model", nvext_prefix_iat=0)

        with pytest.raises(ValueError):
            DynamoModelConfig(model_name="test-model", nvext_prefix_osl="INVALID")

        with pytest.raises(ValueError):
            DynamoModelConfig(model_name="test-model", nvext_prefix_iat="INVALID")

    def test_backward_compat_categorical_strings(self):
        """Test that categorical string values (LOW/MEDIUM/HIGH) are coerced to integers."""
        config = DynamoModelConfig(model_name="test-model", nvext_prefix_osl="LOW", nvext_prefix_iat="LOW")
        assert config.nvext_prefix_osl == 128
        assert config.nvext_prefix_iat == 50

        config = DynamoModelConfig(model_name="test-model", nvext_prefix_osl="MEDIUM", nvext_prefix_iat="MEDIUM")
        assert config.nvext_prefix_osl == 512
        assert config.nvext_prefix_iat == 250

        config = DynamoModelConfig(model_name="test-model", nvext_prefix_osl="HIGH", nvext_prefix_iat="HIGH")
        assert config.nvext_prefix_osl == 2048
        assert config.nvext_prefix_iat == 750

    def test_backward_compat_case_insensitive(self):
        """Test that categorical coercion is case-insensitive."""
        config = DynamoModelConfig(model_name="test-model", nvext_prefix_osl="low", nvext_prefix_iat="high")
        assert config.nvext_prefix_osl == 128
        assert config.nvext_prefix_iat == 750

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

    def test_cache_pin_type_none_disables(self):
        """Test that cache_pin_type can be set to None to disable cache control."""
        config = DynamoModelConfig(model_name="test-model", nvext_cache_pin_type=None)
        assert config.nvext_cache_pin_type is None

    def test_cache_pin_type_accepts_enum(self):
        """Test that cache_pin_type accepts CachePinType enum values."""
        config = DynamoModelConfig(model_name="test-model", nvext_cache_pin_type=CachePinType.EPHEMERAL)
        assert config.nvext_cache_pin_type == CachePinType.EPHEMERAL

    def test_cache_pin_type_accepts_string(self):
        """Test that cache_pin_type accepts string values matching enum."""
        config = DynamoModelConfig(model_name="test-model", nvext_cache_pin_type="ephemeral")
        assert config.nvext_cache_pin_type == CachePinType.EPHEMERAL

    def test_cache_pin_type_rejects_invalid_string(self):
        """Test that cache_pin_type rejects invalid string values."""
        with pytest.raises(ValueError):
            DynamoModelConfig(model_name="test-model", nvext_cache_pin_type="invalid")

    def test_cache_control_mode_default(self):
        """Test that cache_control_mode defaults to ALWAYS."""
        config = DynamoModelConfig(model_name="test-model")
        assert config.nvext_cache_control_mode == CacheControlMode.ALWAYS

    def test_cache_control_mode_accepts_enum(self):
        """Test that cache_control_mode accepts CacheControlMode enum values."""
        config = DynamoModelConfig(model_name="test-model", nvext_cache_control_mode=CacheControlMode.FIRST_ONLY)
        assert config.nvext_cache_control_mode == CacheControlMode.FIRST_ONLY

    def test_cache_control_mode_accepts_string(self):
        """Test that cache_control_mode accepts string values matching enum."""
        config = DynamoModelConfig(model_name="test-model", nvext_cache_control_mode="first_only")
        assert config.nvext_cache_control_mode == CacheControlMode.FIRST_ONLY

    def test_cache_control_mode_rejects_invalid_string(self):
        """Test that cache_control_mode rejects invalid string values."""
        with pytest.raises(ValueError):
            DynamoModelConfig(model_name="test-model", nvext_cache_control_mode="invalid")

    def test_max_sensitivity_validation(self):
        """Test that nvext_max_sensitivity validates bounds."""
        config = DynamoModelConfig(model_name="test-model", nvext_max_sensitivity=1)
        assert config.nvext_max_sensitivity == 1

        config = DynamoModelConfig(model_name="test-model", nvext_max_sensitivity=10000)
        assert config.nvext_max_sensitivity == 10000

        with pytest.raises(ValueError):
            DynamoModelConfig(model_name="test-model", nvext_max_sensitivity=0)

        with pytest.raises(ValueError):
            DynamoModelConfig(model_name="test-model", nvext_max_sensitivity=-1)

    def test_get_dynamo_field_names(self):
        """Test that get_dynamo_field_names returns the correct field set."""
        field_names = DynamoModelConfig.get_dynamo_field_names()

        expected = frozenset({
            "enable_nvext_hints",
            "nvext_prefix_id_template",
            "nvext_prefix_total_requests",
            "nvext_prefix_osl",
            "nvext_prefix_iat",
            "request_timeout",
            "nvext_prediction_trie_path",
            "nvext_cache_pin_type",
            "nvext_cache_control_mode",
            "nvext_max_sensitivity",
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

    def test_prefix_id_stable_across_multiple_calls(self):
        """Test that the same prefix ID is returned for multiple calls within the same context."""
        DynamoPrefixContext.clear()

        first = DynamoPrefixContext.get()
        second = DynamoPrefixContext.get()
        third = DynamoPrefixContext.get()

        assert first == second == third
        assert "-d0" in first

    def test_override_prefix_id_stable_across_multiple_calls(self):
        """Test that an override prefix ID is stable across multiple get() calls."""
        DynamoPrefixContext.clear()
        DynamoPrefixContext.set("workflow-abc-123")

        assert DynamoPrefixContext.get() == "workflow-abc-123"
        assert DynamoPrefixContext.get() == "workflow-abc-123"
        assert DynamoPrefixContext.get() == "workflow-abc-123"

        DynamoPrefixContext.clear()

    async def test_prefix_id_consistent_across_transport_requests(self):
        """Test that the same prefix_id appears in agent_hints across multiple LLM requests.

        This verifies the critical property that all requests within the same
        workflow/conversation share the same prefix_id for KV cache affinity.
        """
        import json

        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport

        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=10,
            osl=512,
            iat=250,
            prediction_lookup=None,
        )

        DynamoPrefixContext.set("stable-prefix-test")

        prefix_ids = []
        for _ in range(5):
            request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
            await transport.handle_async_request(request)

            body = json.loads(mock_transport.handle_async_request.call_args[0][0].content.decode("utf-8"))
            prefix_ids.append(body["nvext"]["agent_hints"]["prefix_id"])

        assert all(pid == "stable-prefix-test" for pid in prefix_ids)
        assert len(prefix_ids) == 5

        DynamoPrefixContext.clear()


# ---------------------------------------------------------------------------
# HTTPX Client Creation Tests
# ---------------------------------------------------------------------------


class TestCreateHttpxClient:
    """Tests for _create_httpx_client_with_dynamo_hooks."""

    async def test_uses_custom_timeout(self):
        """Test that the function uses the provided timeout from config."""
        config = DynamoModelConfig(model_name="test", request_timeout=120.0)
        async with _create_httpx_client_with_dynamo_hooks(config) as client:
            assert client.timeout.connect == 120.0
            assert client.timeout.read == 120.0
            assert client.timeout.write == 120.0

    async def test_uses_default_timeout(self):
        """Test that the function uses default timeout when not specified."""
        config = DynamoModelConfig(model_name="test")
        async with _create_httpx_client_with_dynamo_hooks(config) as client:
            assert client.timeout.connect == 600.0

    async def test_creates_client_with_custom_transport(self):
        """Test that _create_httpx_client_with_dynamo_hooks uses _DynamoTransport when enable_nvext_hints=True."""
        from nat.llm.dynamo_llm import _DynamoTransport

        config = DynamoModelConfig(
            model_name="test",
            enable_nvext_hints=True,
            nvext_prefix_total_requests=7,
            nvext_prefix_osl=2048,
            nvext_prefix_iat=50,
            request_timeout=120.0,
        )
        async with _create_httpx_client_with_dynamo_hooks(config) as client:
            # Verify client uses custom transport
            assert isinstance(client._transport, _DynamoTransport)

            # Verify transport has correct values
            assert client._transport._total_requests == 7
            assert client._transport._osl == 2048
            assert client._transport._iat == 50
            assert client._transport._cache_pin_type == CachePinType.EPHEMERAL

            # Verify timeout
            assert client.timeout.read == 120.0

    async def test_creates_client_with_cache_pin_type_none(self):
        """Test that _create_httpx_client_with_dynamo_hooks passes cache_pin_type=None through."""
        from nat.llm.dynamo_llm import _DynamoTransport

        config = DynamoModelConfig(
            model_name="test",
            enable_nvext_hints=True,
            nvext_cache_pin_type=None,
        )
        async with _create_httpx_client_with_dynamo_hooks(config) as client:
            assert isinstance(client._transport, _DynamoTransport)
            assert client._transport._cache_pin_type is None

    async def test_creates_client_with_cache_control_mode_first_only(self):
        """Test that _create_httpx_client_with_dynamo_hooks passes cache_control_mode through."""
        from nat.llm.dynamo_llm import _DynamoTransport

        config = DynamoModelConfig(
            model_name="test",
            enable_nvext_hints=True,
            nvext_cache_control_mode=CacheControlMode.FIRST_ONLY,
        )
        async with _create_httpx_client_with_dynamo_hooks(config) as client:
            assert isinstance(client._transport, _DynamoTransport)
            assert client._transport._cache_control_mode == CacheControlMode.FIRST_ONLY

    @pytest.mark.parametrize(
        "config,expected_verify",
        [
            (DynamoModelConfig(model_name="test"), True),
            (DynamoModelConfig(model_name="test", verify_ssl=True), True),
            (DynamoModelConfig(model_name="test", verify_ssl=False), False),
        ],
        ids=["default_verify_ssl", "verify_ssl_true", "verify_ssl_false"],
    )
    async def test_verify_ssl_passed_to_client(self,
                                               config: DynamoModelConfig,
                                               expected_verify: bool,
                                               mock_httpx_async_client):
        """Verify that verify_ssl from config is passed to the underlying httpx.AsyncClient as verify."""
        async with _create_httpx_client_with_dynamo_hooks(config):
            pass
        mock_httpx_async_client.assert_called_once()
        call_kwargs = mock_httpx_async_client.call_args.kwargs
        assert call_kwargs["verify"] is expected_verify


# ---------------------------------------------------------------------------
# _DynamoTransport Tests
# ---------------------------------------------------------------------------


class TestDynamoTransport:
    """Tests for _DynamoTransport custom transport wrapper."""

    async def test_transport_injects_raw_agent_hints_by_default(self):
        """Test that _DynamoTransport injects raw integer values in nvext.agent_hints by default."""
        import json

        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport

        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=15,
            osl=2048,
            iat=50,
            prediction_lookup=None,
        )

        DynamoPrefixContext.set("test-prefix-123")

        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
        await transport.handle_async_request(request)

        modified_request = mock_transport.handle_async_request.call_args[0][0]
        body = json.loads(modified_request.content.decode("utf-8"))
        agent_hints = body["nvext"]["agent_hints"]

        assert agent_hints["prefix_id"] == "test-prefix-123"
        assert agent_hints["total_requests"] == 15
        assert agent_hints["osl"] == 2048
        assert agent_hints["iat"] == 50

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
        # Default latency_sensitivity=2, max_sensitivity=1000 -> priority=998
        assert agent_hints["priority"] == 998

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
        )

        DynamoPrefixContext.set("non-json-test")

        # Create request with non-JSON content
        request = httpx.Request("POST", "https://api.example.com/chat", content=b"plain text")

        # Should not raise; body is not JSON so nvext injection is skipped
        await transport.handle_async_request(request)

        # The request is forwarded unchanged (no nvext injected into non-JSON body)
        modified_request = mock_transport.handle_async_request.call_args[0][0]
        assert modified_request.content == b"plain text"

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
        )

        # Set prefix ID
        DynamoPrefixContext.set("prediction-test")

        # Create a POST request
        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})

        # Handle request
        await transport.handle_async_request(request)

        # Get the modified request
        modified_request = mock_transport.handle_async_request.call_args[0][0]

        # Verify raw prediction values in nvext.agent_hints
        import json
        body = json.loads(modified_request.content.decode("utf-8"))
        agent_hints = body["nvext"]["agent_hints"]

        assert agent_hints["total_requests"] == 25
        assert agent_hints["osl"] == 2500  # raw output_tokens.p90
        assert agent_hints["iat"] == 50  # raw interarrival_ms.mean

        # Verify lookup was called
        assert mock_lookup.find.called

        DynamoPrefixContext.clear()

    async def test_transport_injects_all_agent_hints_fields(self):
        """Test that nvext.agent_hints contains all expected fields with correct values."""
        import json

        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport

        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=10,
            osl=512,
            iat=250,
            prediction_lookup=None,
        )

        DynamoPrefixContext.set("test-all-fields")

        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test", "messages": []})
        await transport.handle_async_request(request)

        modified_request = mock_transport.handle_async_request.call_args[0][0]
        body = json.loads(modified_request.content.decode("utf-8"))
        assert "nvext" in body
        agent_hints = body["nvext"]["agent_hints"]

        # Custom processor.py fields
        assert agent_hints["prefix_id"] == "test-all-fields"
        assert agent_hints["total_requests"] == 10
        assert agent_hints["iat"] == 250
        # Standard Dynamo AgentHints fields
        assert agent_hints["osl"] == 512
        assert agent_hints["priority"] == 998  # 1000 - 2
        assert agent_hints["latency_sensitivity"] == 2.0

        DynamoPrefixContext.clear()

    async def test_transport_injects_latency_sensitivity_in_agent_hints(self):
        """Test that _DynamoTransport injects latency_sensitivity and priority in nvext.agent_hints."""
        import json

        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport

        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=10,
            osl=512,
            iat=750,
            prediction_lookup=None,
        )

        DynamoPrefixContext.set("test-latency-ann")

        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test", "messages": []})
        await transport.handle_async_request(request)

        modified_request = mock_transport.handle_async_request.call_args[0][0]
        body = json.loads(modified_request.content.decode("utf-8"))

        assert "nvext" in body
        assert "agent_hints" in body["nvext"]
        agent_hints = body["nvext"]["agent_hints"]
        assert agent_hints["latency_sensitivity"] == 2.0
        # priority = max_sensitivity(1000) - latency_sensitivity(2) = 998
        assert agent_hints["priority"] == 998

        DynamoPrefixContext.clear()

    async def test_transport_injects_cache_control_by_default(self):
        """Test that _DynamoTransport injects nvext.cache_control with ephemeral type and computed TTL."""
        import json

        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport

        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        # total_requests=10, iat=250 -> TTL = 10 * 250 = 2500ms
        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=10,
            osl=512,
            iat=250,
            prediction_lookup=None,
        )

        DynamoPrefixContext.set("cache-control-test")

        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test", "messages": []})
        await transport.handle_async_request(request)

        modified_request = mock_transport.handle_async_request.call_args[0][0]
        body = json.loads(modified_request.content.decode("utf-8"))

        assert "nvext" in body
        assert "cache_control" in body["nvext"]
        cache_control = body["nvext"]["cache_control"]
        assert cache_control["type"] == "ephemeral"
        assert cache_control["ttl"] == "3s"  # 10 * 250 = 2500ms -> ceil = 3s

        DynamoPrefixContext.clear()

    async def test_transport_cache_control_ttl_formatted_as_minutes(self):
        """Test that TTL is formatted as '<N>m' when evenly divisible by 60 seconds."""
        import json

        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport

        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        # total_requests=20, iat=3000 -> TTL = 60000ms = 60s = 1m
        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=20,
            osl=512,
            iat=3000,
            prediction_lookup=None,
        )

        DynamoPrefixContext.set("ttl-minutes-test")

        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
        await transport.handle_async_request(request)

        modified_request = mock_transport.handle_async_request.call_args[0][0]
        body = json.loads(modified_request.content.decode("utf-8"))

        assert body["nvext"]["cache_control"]["ttl"] == "1m"  # 20 * 3000 = 60000ms = 1m

        DynamoPrefixContext.clear()

    async def test_transport_cache_control_ttl_formatted_as_seconds(self):
        """Test that TTL is formatted as '<N>s' when not evenly divisible by 60."""
        import json

        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport

        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        # total_requests=20, iat=500 -> TTL = 10000ms = 10s
        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=20,
            osl=512,
            iat=500,
            prediction_lookup=None,
        )

        DynamoPrefixContext.set("ttl-seconds-test")

        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
        await transport.handle_async_request(request)

        modified_request = mock_transport.handle_async_request.call_args[0][0]
        body = json.loads(modified_request.content.decode("utf-8"))

        assert body["nvext"]["cache_control"]["ttl"] == "10s"  # 20 * 500 = 10000ms = 10s

        DynamoPrefixContext.clear()

    async def test_transport_no_cache_control_when_disabled(self):
        """Test that cache_control is NOT injected when cache_pin_type is None."""
        import json

        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport

        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=10,
            osl=512,
            iat=250,
            prediction_lookup=None,
            cache_pin_type=None,
        )

        DynamoPrefixContext.set("no-cache-control")

        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test", "messages": []})
        await transport.handle_async_request(request)

        modified_request = mock_transport.handle_async_request.call_args[0][0]
        body = json.loads(modified_request.content.decode("utf-8"))

        # agent_hints should still be present
        assert "agent_hints" in body["nvext"]
        # cache_control should NOT be present
        assert "cache_control" not in body["nvext"]

        DynamoPrefixContext.clear()

    async def test_transport_cache_control_uses_prediction_override(self):
        """Test that cache_control TTL uses prediction-overridden total_requests and iat."""
        import json

        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport
        from nat.profiler.prediction_trie.data_models import LLMCallPrediction
        from nat.profiler.prediction_trie.data_models import PredictionMetrics

        # Prediction: remaining_calls.mean=25, interarrival_ms.mean=50
        # Expected TTL = 25 * 50 = 1250ms (not 10 * 250 = 2500 from static config)
        mock_prediction = LLMCallPrediction(
            remaining_calls=PredictionMetrics(mean=25.0, p50=25.0, p90=30.0),
            output_tokens=PredictionMetrics(mean=2000.0, p50=2000.0, p90=2500.0),
            interarrival_ms=PredictionMetrics(mean=50.0, p50=50.0, p90=70.0),
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
        )

        DynamoPrefixContext.set("prediction-cache-test")

        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
        await transport.handle_async_request(request)

        modified_request = mock_transport.handle_async_request.call_args[0][0]
        body = json.loads(modified_request.content.decode("utf-8"))

        cache_control = body["nvext"]["cache_control"]
        assert cache_control["type"] == "ephemeral"
        assert cache_control["ttl"] == "2s"  # 25 * 50 = 1250ms -> ceil = 2s

        DynamoPrefixContext.clear()

    async def test_transport_uses_auto_latency_sensitivity(self):
        """When prediction has latency_sensitivity and no manual decorator, use it."""
        import json

        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport
        from nat.profiler.prediction_trie.data_models import LLMCallPrediction
        from nat.profiler.prediction_trie.data_models import PredictionMetrics

        mock_prediction = LLMCallPrediction(
            remaining_calls=PredictionMetrics(mean=5.0, p50=5.0, p90=7.0),
            output_tokens=PredictionMetrics(mean=200.0, p50=200.0, p90=300.0),
            interarrival_ms=PredictionMetrics(mean=100.0, p50=100.0, p90=150.0),
            latency_sensitivity=4,
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
            max_sensitivity=1000,
        )

        DynamoPrefixContext.set("auto-sensitivity-test")

        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
        await transport.handle_async_request(request)

        modified_request = mock_transport.handle_async_request.call_args[0][0]
        body = json.loads(modified_request.content.decode("utf-8"))
        agent_hints = body["nvext"]["agent_hints"]

        # Auto sensitivity=4 should be used (no manual decorator active)
        assert agent_hints["latency_sensitivity"] == 4.0
        assert agent_hints["priority"] == 1000 - 4

        DynamoPrefixContext.clear()

    async def test_transport_manual_sensitivity_overrides_auto(self):
        """When @latency_sensitive decorator is active, ignore prediction's auto sensitivity."""
        import json

        import httpx

        from nat.builder.context import Context
        from nat.llm.dynamo_llm import _DynamoTransport
        from nat.profiler.prediction_trie.data_models import LLMCallPrediction
        from nat.profiler.prediction_trie.data_models import PredictionMetrics

        mock_prediction = LLMCallPrediction(
            remaining_calls=PredictionMetrics(mean=5.0, p50=5.0, p90=7.0),
            output_tokens=PredictionMetrics(mean=200.0, p50=200.0, p90=300.0),
            interarrival_ms=PredictionMetrics(mean=100.0, p50=100.0, p90=150.0),
            latency_sensitivity=4,
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
            max_sensitivity=1000,
        )

        DynamoPrefixContext.set("manual-override-test")

        # Simulate @latency_sensitive(7) being active
        ctx = Context.get()
        with ctx.push_latency_sensitivity(7):
            request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
            await transport.handle_async_request(request)

        modified_request = mock_transport.handle_async_request.call_args[0][0]
        body = json.loads(modified_request.content.decode("utf-8"))
        agent_hints = body["nvext"]["agent_hints"]

        # Manual sensitivity=7 should win over auto sensitivity=4
        assert agent_hints["latency_sensitivity"] == 7.0
        assert agent_hints["priority"] == 1000 - 7

        DynamoPrefixContext.clear()

    async def test_transport_no_auto_sensitivity_when_prediction_is_none(self):
        """When prediction has no latency_sensitivity, use context default."""
        import json

        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport
        from nat.profiler.prediction_trie.data_models import LLMCallPrediction
        from nat.profiler.prediction_trie.data_models import PredictionMetrics

        mock_prediction = LLMCallPrediction(
            remaining_calls=PredictionMetrics(mean=5.0, p50=5.0, p90=7.0),
            output_tokens=PredictionMetrics(mean=200.0, p50=200.0, p90=300.0),
            interarrival_ms=PredictionMetrics(mean=100.0, p50=100.0, p90=150.0),  # latency_sensitivity=None (default)
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
            max_sensitivity=1000,
        )

        DynamoPrefixContext.set("no-auto-test")

        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
        await transport.handle_async_request(request)

        modified_request = mock_transport.handle_async_request.call_args[0][0]
        body = json.loads(modified_request.content.decode("utf-8"))
        agent_hints = body["nvext"]["agent_hints"]

        # Should use context default (2)
        assert agent_hints["latency_sensitivity"] == 2.0
        assert agent_hints["priority"] == 1000 - 2

        DynamoPrefixContext.clear()

    async def test_transport_raises_when_latency_exceeds_max(self):
        """Test that ValueError is raised when latency_sensitivity exceeds max_sensitivity."""
        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport

        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        # Default latency_sensitivity fallback is 2, max_sensitivity=1 -> 2 > 1 -> ValueError
        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=10,
            osl=512,
            iat=250,
            prediction_lookup=None,
            max_sensitivity=1,
        )

        DynamoPrefixContext.set("overflow-test")

        request = httpx.Request(
            "POST",
            "https://api.example.com/chat",
            json={
                "model": "test", "messages": []
            },
        )

        with pytest.raises(ValueError, match="latency_sensitivity.*exceeds.*max_sensitivity"):
            await transport.handle_async_request(request)

        DynamoPrefixContext.clear()

    async def test_transport_raises_when_latency_sensitivity_negative(self):
        """Test that ValueError is raised when latency_sensitivity is negative.

        Context.latency_sensitivity returns max(stack) so it cannot go negative through
        normal usage. We patch the context read directly to simulate a negative value
        arriving via a custom subclass or mock.
        """
        from unittest.mock import patch

        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport

        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=10,
            osl=512,
            iat=250,
            prediction_lookup=None,
            max_sensitivity=1000,
        )

        DynamoPrefixContext.set("negative-sensitivity-test")

        # Patch Context.latency_sensitivity to return -1 directly
        with patch("nat.llm.dynamo_llm.Context") as mock_ctx_cls:
            mock_ctx = MagicMock()
            mock_ctx.latency_sensitivity = -1
            mock_ctx.function_path = []
            mock_ctx_cls.get.return_value = mock_ctx

            request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
            with pytest.raises(ValueError, match="latency_sensitivity.*must be >= 0"):
                await transport.handle_async_request(request)

        DynamoPrefixContext.clear()

    async def test_transport_raises_when_total_requests_zero(self):
        """Test that ValueError is raised when prediction trie yields total_requests < 1."""
        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport
        from nat.profiler.prediction_trie.data_models import LLMCallPrediction
        from nat.profiler.prediction_trie.data_models import PredictionMetrics

        # Prediction with remaining_calls.mean=0 -> total_requests=0
        mock_prediction = LLMCallPrediction(
            remaining_calls=PredictionMetrics(mean=0.0, p50=0.0, p90=0.0),
            output_tokens=PredictionMetrics(mean=512.0, p50=512.0, p90=512.0),
            interarrival_ms=PredictionMetrics(mean=250.0, p50=250.0, p90=250.0),
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
        )

        DynamoPrefixContext.set("zero-total-requests-test")

        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
        with pytest.raises(ValueError, match="total_requests must be >= 1"):
            await transport.handle_async_request(request)

        DynamoPrefixContext.clear()

    async def test_transport_raises_when_osl_zero(self):
        """Test that ValueError is raised when prediction trie yields osl < 1."""
        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport
        from nat.profiler.prediction_trie.data_models import LLMCallPrediction
        from nat.profiler.prediction_trie.data_models import PredictionMetrics

        # Prediction with output_tokens.p90=0 -> osl_raw=0
        mock_prediction = LLMCallPrediction(
            remaining_calls=PredictionMetrics(mean=5.0, p50=5.0, p90=5.0),
            output_tokens=PredictionMetrics(mean=0.0, p50=0.0, p90=0.0),
            interarrival_ms=PredictionMetrics(mean=250.0, p50=250.0, p90=250.0),
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
        )

        DynamoPrefixContext.set("zero-osl-test")

        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
        with pytest.raises(ValueError, match="osl must be >= 1"):
            await transport.handle_async_request(request)

        DynamoPrefixContext.clear()

    async def test_transport_raises_when_iat_zero(self):
        """Test that ValueError is raised when prediction trie yields iat < 1."""
        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport
        from nat.profiler.prediction_trie.data_models import LLMCallPrediction
        from nat.profiler.prediction_trie.data_models import PredictionMetrics

        # Prediction with interarrival_ms.mean=0 -> iat_raw=0
        mock_prediction = LLMCallPrediction(
            remaining_calls=PredictionMetrics(mean=5.0, p50=5.0, p90=5.0),
            output_tokens=PredictionMetrics(mean=512.0, p50=512.0, p90=512.0),
            interarrival_ms=PredictionMetrics(mean=0.0, p50=0.0, p90=0.0),
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
        )

        DynamoPrefixContext.set("zero-iat-test")

        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
        with pytest.raises(ValueError, match="iat must be >= 1"):
            await transport.handle_async_request(request)

        DynamoPrefixContext.clear()

    async def test_transport_first_only_injects_cache_control_on_first_request(self):
        """Test that FIRST_ONLY mode injects cache_control on the first request for a prefix."""
        import json

        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport

        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=10,
            osl=512,
            iat=250,
            prediction_lookup=None,
            cache_control_mode=CacheControlMode.FIRST_ONLY,
        )

        DynamoPrefixContext.set("first-only-test")

        # First request: cache_control should be injected
        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test", "messages": []})
        await transport.handle_async_request(request)

        modified_request = mock_transport.handle_async_request.call_args[0][0]
        body = json.loads(modified_request.content.decode("utf-8"))

        assert "cache_control" in body["nvext"]
        assert body["nvext"]["cache_control"]["type"] == "ephemeral"
        assert body["nvext"]["cache_control"]["ttl"] == "3s"  # 10 * 250 = 2500ms -> ceil = 3s

        DynamoPrefixContext.clear()

    async def test_transport_first_only_skips_cache_control_on_subsequent_requests(self):
        """Test that FIRST_ONLY mode does NOT inject cache_control on the second+ request."""
        import json

        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport

        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=10,
            osl=512,
            iat=250,
            prediction_lookup=None,
            cache_control_mode=CacheControlMode.FIRST_ONLY,
        )

        DynamoPrefixContext.set("first-only-skip-test")

        # First request (call_index=1): should have cache_control
        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
        await transport.handle_async_request(request)

        first_body = json.loads(mock_transport.handle_async_request.call_args[0][0].content.decode("utf-8"))
        assert "cache_control" in first_body["nvext"]

        # Second request (call_index=2): should NOT have cache_control
        request2 = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
        await transport.handle_async_request(request2)

        second_body = json.loads(mock_transport.handle_async_request.call_args[0][0].content.decode("utf-8"))
        assert "cache_control" not in second_body["nvext"]

        # Third request (call_index=3): should still NOT have cache_control
        request3 = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
        await transport.handle_async_request(request3)

        third_body = json.loads(mock_transport.handle_async_request.call_args[0][0].content.decode("utf-8"))
        assert "cache_control" not in third_body["nvext"]

        # agent_hints should still be present on all requests
        assert "agent_hints" in third_body["nvext"]

        DynamoPrefixContext.clear()

    async def test_transport_first_only_tracks_prefixes_independently(self):
        """Test that FIRST_ONLY mode tracks each prefix_id independently."""
        import json

        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport

        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=10,
            osl=512,
            iat=250,
            prediction_lookup=None,
            cache_control_mode=CacheControlMode.FIRST_ONLY,
        )

        # First prefix, first request: should have cache_control
        DynamoPrefixContext.set("prefix-a")
        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
        await transport.handle_async_request(request)

        body_a1 = json.loads(mock_transport.handle_async_request.call_args[0][0].content.decode("utf-8"))
        assert "cache_control" in body_a1["nvext"]

        # First prefix, second request: should NOT have cache_control
        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
        await transport.handle_async_request(request)

        body_a2 = json.loads(mock_transport.handle_async_request.call_args[0][0].content.decode("utf-8"))
        assert "cache_control" not in body_a2["nvext"]

        # Second prefix, first request: SHOULD have cache_control (new prefix)
        DynamoPrefixContext.set("prefix-b")
        request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
        await transport.handle_async_request(request)

        body_b1 = json.loads(mock_transport.handle_async_request.call_args[0][0].content.decode("utf-8"))
        assert "cache_control" in body_b1["nvext"]

        DynamoPrefixContext.clear()

    async def test_transport_always_mode_injects_cache_control_every_request(self):
        """Test that ALWAYS mode (default) injects cache_control on every request."""
        import json

        import httpx

        from nat.llm.dynamo_llm import _DynamoTransport

        mock_response = httpx.Response(200, json={"result": "ok"})
        mock_transport = MagicMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        transport = _DynamoTransport(
            transport=mock_transport,
            total_requests=10,
            osl=512,
            iat=250,
            prediction_lookup=None,
            cache_control_mode=CacheControlMode.ALWAYS,
        )

        DynamoPrefixContext.set("always-mode-test")

        for i in range(3):
            request = httpx.Request("POST", "https://api.example.com/chat", json={"model": "test"})
            await transport.handle_async_request(request)

            body = json.loads(mock_transport.handle_async_request.call_args[0][0].content.decode("utf-8"))
            assert "cache_control" in body["nvext"], f"cache_control missing on request {i + 1}"

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
