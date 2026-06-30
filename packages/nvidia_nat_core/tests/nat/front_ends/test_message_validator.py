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
"""Tests for MessageValidator handling of auth_message type."""

import datetime
import uuid

import pytest

from nat.data_models.api_server import ObservabilityTraceContent
from nat.data_models.api_server import WebSocketAuthMessage
from nat.data_models.api_server import WebSocketAuthResponseMessage
from nat.data_models.api_server import WebSocketMessageType
from nat.data_models.api_server import WebSocketSystemResponseTokenMessage
from nat.data_models.interactive import HumanPromptText
from nat.front_ends.fastapi import message_validator as message_validator_module
from nat.front_ends.fastapi.message_validator import MessageValidator


@pytest.fixture(name="validator")
def fixture_validator() -> MessageValidator:
    return MessageValidator()


class TestAuthMessageSchemaMapping:

    async def test_schema_lookup_returns_auth_message(self, validator: MessageValidator):
        schema = await validator.get_message_schema_by_type(WebSocketMessageType.AUTH_MESSAGE)
        assert schema is WebSocketAuthMessage

    async def test_auth_message_in_mapping(self, validator: MessageValidator):
        assert WebSocketMessageType.AUTH_MESSAGE in validator._message_type_schema_mapping

    async def test_schema_lookup_returns_auth_response(self, validator: MessageValidator):
        schema = await validator.get_message_schema_by_type(WebSocketMessageType.AUTH_RESPONSE)
        assert schema is WebSocketAuthResponseMessage

    async def test_auth_response_in_mapping(self, validator: MessageValidator):
        assert WebSocketMessageType.AUTH_RESPONSE in validator._message_type_schema_mapping


class TestValidateAuthMessage:

    async def test_validate_jwt_auth_message(self, validator: MessageValidator):
        raw: dict = {
            "type": "auth_message",
            "payload": {
                "method": "jwt", "token": "eyJhbGciOiJub25lIn0.eyJzdWIiOiJ1c2VyMSJ9."
            },
        }
        result = await validator.validate_message(raw)
        assert isinstance(result, WebSocketAuthMessage)
        assert result.payload.method == "jwt"

    async def test_validate_api_key_auth_message(self, validator: MessageValidator):
        raw: dict = {
            "type": "auth_message",
            "payload": {
                "method": "api_key", "token": "nvapi-abc123"
            },
        }
        result = await validator.validate_message(raw)
        assert isinstance(result, WebSocketAuthMessage)
        assert result.payload.method == "api_key"

    async def test_validate_basic_auth_message(self, validator: MessageValidator):
        raw: dict = {
            "type": "auth_message",
            "payload": {
                "method": "basic", "username": "alice", "password": "s3cret"
            },
        }
        result = await validator.validate_message(raw)
        assert isinstance(result, WebSocketAuthMessage)
        assert result.payload.method == "basic"

    async def test_malformed_payload_returns_error(self, validator: MessageValidator):
        raw: dict = {
            "type": "auth_message",
            "payload": {
                "method": "jwt"
            },
        }
        result = await validator.validate_message(raw)
        assert isinstance(result, WebSocketSystemResponseTokenMessage)
        assert result.type == WebSocketMessageType.ERROR_MESSAGE


class TestMessageFactoryDefaults:
    async def test_factories_generate_fresh_ids_and_timestamps_if_defaults_are_omitted(
        self, validator: MessageValidator, monkeypatch: pytest.MonkeyPatch
    ):
        ids = iter(
            [
                uuid.UUID("00000000-0000-0000-0000-000000000001"),
                uuid.UUID("00000000-0000-0000-0000-000000000002"),
                uuid.UUID("00000000-0000-0000-0000-000000000003"),
                uuid.UUID("00000000-0000-0000-0000-000000000004"),
                uuid.UUID("00000000-0000-0000-0000-000000000005"),
                uuid.UUID("00000000-0000-0000-0000-000000000006"),
                uuid.UUID("00000000-0000-0000-0000-000000000007"),
                uuid.UUID("00000000-0000-0000-0000-000000000008"),
            ]
        )
        timestamps = iter([datetime.datetime(2026, 1, 1, 0, 0, index, tzinfo=datetime.UTC) for index in range(8)])

        class FreshDateTime(datetime.datetime):
            @classmethod
            def now(cls, tz=None):
                assert tz is datetime.UTC
                return next(timestamps)

        monkeypatch.setattr(message_validator_module.uuid, "uuid4", lambda: next(ids))
        monkeypatch.setattr(message_validator_module.datetime, "datetime", FreshDateTime)

        messages = [
            await validator.create_system_response_token_message(),
            await validator.create_system_response_token_message(),
            await validator.create_system_intermediate_step_message(),
            await validator.create_system_intermediate_step_message(),
            await validator.create_system_interaction_message(content=HumanPromptText(text="Continue?")),
            await validator.create_system_interaction_message(content=HumanPromptText(text="Continue?")),
            await validator.create_observability_trace_message(
                content=ObservabilityTraceContent(observability_trace_id="trace-1")
            ),
            await validator.create_observability_trace_message(
                content=ObservabilityTraceContent(observability_trace_id="trace-1")
            ),
        ]

        assert [message.id for message in messages] == [
            "00000000-0000-0000-0000-000000000001",
            "00000000-0000-0000-0000-000000000002",
            "00000000-0000-0000-0000-000000000003",
            "00000000-0000-0000-0000-000000000004",
            "00000000-0000-0000-0000-000000000005",
            "00000000-0000-0000-0000-000000000006",
            "00000000-0000-0000-0000-000000000007",
            "00000000-0000-0000-0000-000000000008",
        ]
        assert [message.timestamp for message in messages] == [
            "2026-01-01 00:00:00+00:00",
            "2026-01-01 00:00:01+00:00",
            "2026-01-01 00:00:02+00:00",
            "2026-01-01 00:00:03+00:00",
            "2026-01-01 00:00:04+00:00",
            "2026-01-01 00:00:05+00:00",
            "2026-01-01 00:00:06+00:00",
            "2026-01-01 00:00:07+00:00",
        ]

    async def test_missing_payload_returns_error(self, validator: MessageValidator):
        raw: dict = {"type": "auth_message"}
        result = await validator.validate_message(raw)
        assert isinstance(result, WebSocketSystemResponseTokenMessage)
        assert result.type == WebSocketMessageType.ERROR_MESSAGE

    async def test_unknown_method_returns_error(self, validator: MessageValidator):
        raw: dict = {
            "type": "auth_message",
            "payload": {
                "method": "oauth2", "token": "tok"
            },
        }
        result = await validator.validate_message(raw)
        assert isinstance(result, WebSocketSystemResponseTokenMessage)
        assert result.type == WebSocketMessageType.ERROR_MESSAGE

    async def test_extra_fields_on_auth_message_returns_error(self, validator: MessageValidator):
        raw: dict = {
            "type": "auth_message",
            "payload": {
                "method": "jwt", "token": "tok"
            },
            "extra_field": "bad",
        }
        result = await validator.validate_message(raw)
        assert isinstance(result, WebSocketSystemResponseTokenMessage)
        assert result.type == WebSocketMessageType.ERROR_MESSAGE
