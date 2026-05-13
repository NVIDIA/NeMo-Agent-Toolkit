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
"""Tests for WebSocket auth payload data models and the discriminated union."""

import pytest
from pydantic import SecretStr
from pydantic import TypeAdapter
from pydantic import ValidationError

from nat.data_models.api_server import ApiKeyAuthPayload
from nat.data_models.api_server import AuthMethod
from nat.data_models.api_server import AuthPayload
from nat.data_models.api_server import BasicAuthPayload
from nat.data_models.api_server import JwtAuthPayload
from nat.data_models.api_server import WebSocketAuthMessage
from nat.data_models.api_server import WebSocketAuthResponseMessage
from nat.data_models.api_server import WebSocketMessageType


class TestAuthMethodEnum:

    def test_values(self):
        assert AuthMethod.JWT == "jwt"
        assert AuthMethod.API_KEY == "api_key"
        assert AuthMethod.BASIC == "basic"

    def test_membership(self):
        assert set(AuthMethod) == {"jwt", "api_key", "basic"}


class TestJwtAuthPayload:

    def test_valid_construction(self):
        payload = JwtAuthPayload(method="jwt", token=SecretStr("eyJ..."))
        assert payload.method == AuthMethod.JWT
        assert payload.token.get_secret_value() == "eyJ..."

    def test_wrong_method_rejected(self):
        with pytest.raises(ValidationError):
            JwtAuthPayload(method="api_key", token=SecretStr("tok"))

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            JwtAuthPayload(method="jwt", token=SecretStr("tok"), refresh="r")

    def test_missing_token_rejected(self):
        with pytest.raises(ValidationError):
            JwtAuthPayload(method="jwt")


class TestApiKeyAuthPayload:

    def test_valid_construction(self):
        payload = ApiKeyAuthPayload(method="api_key", token=SecretStr("nvapi-abc"))
        assert payload.method == AuthMethod.API_KEY
        assert payload.token.get_secret_value() == "nvapi-abc"

    def test_wrong_method_rejected(self):
        with pytest.raises(ValidationError):
            ApiKeyAuthPayload(method="jwt", token=SecretStr("tok"))

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ApiKeyAuthPayload(method="api_key", token=SecretStr("tok"), extra="bad")


class TestBasicAuthPayload:

    def test_valid_construction(self):
        payload = BasicAuthPayload(method="basic", username="alice", password=SecretStr("s3cret"))
        assert payload.method == AuthMethod.BASIC
        assert payload.username == "alice"
        assert payload.password.get_secret_value() == "s3cret"

    def test_wrong_method_rejected(self):
        with pytest.raises(ValidationError):
            BasicAuthPayload(method="jwt", username="alice", password=SecretStr("s3cret"))

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            BasicAuthPayload(method="basic", username="alice", password=SecretStr("s3cret"), extra="bad")

    def test_missing_username_rejected(self):
        with pytest.raises(ValidationError):
            BasicAuthPayload(method="basic", password=SecretStr("s3cret"))

    def test_missing_password_rejected(self):
        with pytest.raises(ValidationError):
            BasicAuthPayload(method="basic", username="alice")


class TestAuthPayloadDiscriminator:
    """Validates the discriminated union resolves to the correct type based on ``method``."""

    _adapter: TypeAdapter = TypeAdapter(AuthPayload)

    def _parse(self, data: dict) -> AuthPayload:
        return self._adapter.validate_python(data)

    def test_routes_to_jwt(self):
        result = self._parse({"method": "jwt", "token": "eyJ..."})
        assert isinstance(result, JwtAuthPayload)

    def test_routes_to_api_key(self):
        result = self._parse({"method": "api_key", "token": "nvapi-xyz"})
        assert isinstance(result, ApiKeyAuthPayload)

    def test_routes_to_basic(self):
        result = self._parse({"method": "basic", "username": "alice", "password": "pass"})
        assert isinstance(result, BasicAuthPayload)

    def test_unknown_method_rejected(self):
        with pytest.raises(ValidationError):
            self._parse({"method": "oauth2", "token": "tok"})

    def test_missing_method_rejected(self):
        with pytest.raises(ValidationError):
            self._parse({"token": "tok"})


class TestWebSocketAuthMessage:

    def test_valid_jwt_message(self):
        msg = WebSocketAuthMessage(
            type="auth_message",
            payload=JwtAuthPayload(method="jwt", token=SecretStr("eyJ...")),
        )
        assert msg.type == WebSocketMessageType.AUTH_MESSAGE
        assert isinstance(msg.payload, JwtAuthPayload)

    def test_valid_api_key_message(self):
        msg = WebSocketAuthMessage(
            type="auth_message",
            payload=ApiKeyAuthPayload(method="api_key", token=SecretStr("nvapi-abc")),
        )
        assert isinstance(msg.payload, ApiKeyAuthPayload)

    def test_valid_basic_message(self):
        msg = WebSocketAuthMessage(
            type="auth_message",
            payload=BasicAuthPayload(method="basic", username="u", password=SecretStr("p")),
        )
        assert isinstance(msg.payload, BasicAuthPayload)

    def test_wrong_type_rejected(self):
        with pytest.raises(ValidationError):
            WebSocketAuthMessage(
                type="user_message",
                payload=JwtAuthPayload(method="jwt", token=SecretStr("tok")),
            )

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            WebSocketAuthMessage(
                type="auth_message",
                payload=JwtAuthPayload(method="jwt", token=SecretStr("tok")),
                extra_field="bad",
            )

    def test_missing_payload_rejected(self):
        with pytest.raises(ValidationError):
            WebSocketAuthMessage(type="auth_message")

    def test_from_raw_dict_with_discriminator(self):
        raw: dict = {
            "type": "auth_message",
            "payload": {
                "method": "basic", "username": "bob", "password": "pw"
            },
        }
        msg = WebSocketAuthMessage(**raw)
        assert isinstance(msg.payload, BasicAuthPayload)
        assert msg.payload.username == "bob"


class TestWebSocketAuthResponseMessage:

    def test_success_response(self):
        resp = WebSocketAuthResponseMessage(status="success", user_id="abc-123")
        assert resp.type == WebSocketMessageType.AUTH_RESPONSE
        assert resp.status == "success"
        assert resp.user_id == "abc-123"
        assert resp.payload is None

    def test_failure_response(self):
        from nat.data_models.api_server import Error
        from nat.data_models.api_server import ErrorTypes
        err = Error(code=ErrorTypes.INVALID_MESSAGE, message="fail", details="bad creds")
        resp = WebSocketAuthResponseMessage(status="error", payload=err)
        assert resp.status == "error"
        assert resp.user_id is None
        assert resp.payload.code == ErrorTypes.INVALID_MESSAGE

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            WebSocketAuthResponseMessage(status="success", extra="bad")

    def test_type_defaults_to_auth_response(self):
        resp = WebSocketAuthResponseMessage(status="success", user_id="x")
        assert resp.type == "auth_response_message"

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            WebSocketAuthResponseMessage(status="pending")

    def test_serialization_roundtrip(self):
        resp = WebSocketAuthResponseMessage(status="success", user_id="u-1")
        data: dict = resp.model_dump()
        assert data["type"] == "auth_response_message"
        assert data["status"] == "success"
        assert data["user_id"] == "u-1"
        assert data["payload"] is None
        restored = WebSocketAuthResponseMessage(**data)
        assert restored.user_id == "u-1"

    def test_error_serialization_roundtrip(self):
        from nat.data_models.api_server import Error
        from nat.data_models.api_server import ErrorTypes
        err = Error(code=ErrorTypes.INVALID_MESSAGE, message="fail", details="d")
        resp = WebSocketAuthResponseMessage(status="error", payload=err)
        data: dict = resp.model_dump()
        assert data["status"] == "error"
        assert data["payload"]["code"] == "invalid_message"
        restored = WebSocketAuthResponseMessage(**data)
        assert restored.payload.code == ErrorTypes.INVALID_MESSAGE
