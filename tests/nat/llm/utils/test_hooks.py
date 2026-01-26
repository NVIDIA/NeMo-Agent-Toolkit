# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit and integration tests for LLM HTTP event hooks."""

from unittest.mock import MagicMock

import pytest
from pytest_httpserver import HTTPServer

from nat.builder.context import ContextState
from nat.llm.utils.hooks import create_metadata_injection_client


class TestMetadataInjectionHook:
    """Unit tests for the metadata injection hook function."""

    @pytest.fixture(name="mock_httpx_request")
    def fixture_mock_httpx_request(self):
        """Create a mock httpx.Request."""
        mock_request = MagicMock()
        mock_request.headers = {}
        return mock_request

    @pytest.fixture(name="mock_input_message")
    def fixture_mock_input_message(self):
        """Create a mock input message with model_extra fields."""
        mock_msg = MagicMock()
        mock_msg.model_extra = {
            "scan_id": "scan-12345",
            "customer_id": "cust-789",
            "environment": "production",
        }
        return mock_msg

    async def test_hook_injects_metadata_fields(self, mock_httpx_request, mock_input_message):
        """Test that the hook injects custom metadata fields as headers."""
        client = create_metadata_injection_client()
        hook = client.event_hooks["request"][0]

        context_state = ContextState.get()
        context_state.input_message.set(mock_input_message)

        await hook(mock_httpx_request)

        assert mock_httpx_request.headers["X-Payload-scan-id"] == "scan-12345"
        assert mock_httpx_request.headers["X-Payload-customer-id"] == "cust-789"
        assert mock_httpx_request.headers["X-Payload-environment"] == "production"

        await client.aclose()

    async def test_hook_skips_none_values(self, mock_httpx_request, mock_input_message):
        """Test that None values are not injected as headers."""
        mock_input_message.model_extra = {
            "scan_id": "scan-123",
            "optional_field": None,
        }

        client = create_metadata_injection_client()
        hook = client.event_hooks["request"][0]

        context_state = ContextState.get()
        context_state.input_message.set(mock_input_message)

        await hook(mock_httpx_request)

        assert "X-Payload-scan-id" in mock_httpx_request.headers
        assert "X-Payload-optional-field" not in mock_httpx_request.headers

        await client.aclose()

    async def test_hook_handles_missing_context(self, mock_httpx_request):
        """Test that hook handles missing context gracefully."""
        client = create_metadata_injection_client()
        hook = client.event_hooks["request"][0]

        await hook(mock_httpx_request)

        payload_headers = [k for k in mock_httpx_request.headers if k.startswith("X-Payload-")]
        assert len(payload_headers) == 0

        await client.aclose()


class TestCreateMetadataInjectionClient:
    """Unit tests for create_metadata_injection_client function."""

    async def test_creates_client_with_event_hooks(self):
        """Test that client is created with event hooks."""
        client = create_metadata_injection_client()

        assert "request" in client.event_hooks
        assert len(client.event_hooks["request"]) == 1

        await client.aclose()


class TestMetadataInjectionIntegration:
    """Integration tests with mock HTTP server."""

    @pytest.fixture(name="mock_input_message")
    def fixture_mock_input_message(self):
        """Create a mock input message with model_extra fields."""
        mock_msg = MagicMock()
        mock_msg.model_extra = {
            "scan_id": "integration-test-123",
            "customer_id": "customer-456",
        }
        return mock_msg

    async def test_headers_sent_in_http_request(self, httpserver: HTTPServer, mock_input_message):
        """Test that custom metadata headers are sent in actual HTTP requests."""
        httpserver.expect_request(
            "/v1/chat/completions",
            method="POST",
        ).respond_with_json({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0, "message": {
                    "role": "assistant", "content": "Test response"
                }, "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15
            }
        })

        client = create_metadata_injection_client()
        context_state = ContextState.get()
        context_state.input_message.set(mock_input_message)

        response = await client.post(httpserver.url_for("/v1/chat/completions"),
                                     json={
                                         "model": "test-model", "messages": [{
                                             "role": "user", "content": "test"
                                         }]
                                     })

        assert response.status_code == 200

        requests = httpserver.log
        assert len(requests) == 1
        request_headers = requests[0][0].headers

        assert request_headers["X-Payload-scan-id"] == "integration-test-123"
        assert request_headers["X-Payload-customer-id"] == "customer-456"

        await client.aclose()

    async def test_request_succeeds_without_context(self, httpserver: HTTPServer):
        """Test that requests succeed even when ContextState is not available."""
        httpserver.expect_request(
            "/v1/chat/completions",
            method="POST",
        ).respond_with_json({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0, "message": {
                    "role": "assistant", "content": "Response"
                }, "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15
            }
        })

        client = create_metadata_injection_client()

        response = await client.post(httpserver.url_for("/v1/chat/completions"),
                                     json={
                                         "model": "test", "messages": [{
                                             "role": "user", "content": "test"
                                         }]
                                     })

        assert response.status_code == 200

        requests = httpserver.log
        request_headers = requests[0][0].headers

        payload_headers = [k for k in request_headers.keys() if k.startswith("X-Payload-")]
        assert len(payload_headers) == 0

        await client.aclose()
