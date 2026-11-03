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

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from nat.authentication.oauth2.oauth2_resource_server_config import OAuth2ResourceServerConfig
from nat.data_models.authentication import TokenValidationResult
from nat.plugins.mcp.server.front_end_plugin_worker import MCPFrontEndPluginWorker
from nat.plugins.mcp.server.introspection_token_verifier import IntrospectionTokenVerifier


@pytest.mark.parametrize("configured_audience,audiences,expected_resource",
                         [
                             ("service-id", ["other", "service-id"], "service-id"),
                             (None, ["other", "http://127.0.0.1:9901/"], "http://127.0.0.1:9901/"),
                             (None, ["other", "HTTP://127.0.0.1:9901/"], "HTTP://127.0.0.1:9901/"),
                             (None, ["foreign-resource"], "foreign-resource"),
                             (None, ["one", "two"], None),
                         ])
async def test_verifier_exposes_resource_and_verified_identity(
    configured_audience: str | None,
    audiences: list[str],
    expected_resource: str | None,
) -> None:
    """Expose validated resource and identity without rewriting raw audience claims."""
    expected_raw_audiences = audiences.copy()
    config = OAuth2ResourceServerConfig(issuer_url="https://issuer.example",
                                        audience=configured_audience,
                                        introspection_endpoint="https://issuer.example/introspect",
                                        client_id="resource-server",
                                        client_secret=SecretStr("synthetic-secret"))
    verifier = IntrospectionTokenVerifier(config, resource_server_url="http://127.0.0.1:9901/")
    verifier._bearer_token_validator.verify = AsyncMock(
        return_value=TokenValidationResult(active=True,
                                           client_id="shared-client",
                                           audience=audiences,
                                           subject="alice",
                                           issuer="https://issuer.example",
                                           scopes=["read"],
                                           token_type="Bearer"))
    token = await verifier.verify_token("synthetic-token")
    assert token is not None
    assert token.resource == expected_resource
    assert token.subject == "alice"
    assert token.claims is not None
    assert token.claims["iss"] == "https://issuer.example"
    assert token.claims["sub"] == "alice"
    assert token.claims["aud"] == expected_raw_audiences


@pytest.mark.parametrize("audience,expected_sdk_check", [(None, True), ("", True), ("service-id", False)])
async def test_server_resource_check_is_explicit(audience: str | None, expected_sdk_check: bool) -> None:
    """Delegate resource matching to the SDK only when no audience policy is configured."""
    config = OAuth2ResourceServerConfig(issuer_url="https://issuer.example",
                                        audience=audience,
                                        introspection_endpoint="https://issuer.example/introspect",
                                        client_id="resource-server",
                                        client_secret=SecretStr("synthetic-secret"))
    worker = MCPFrontEndPluginWorker.__new__(MCPFrontEndPluginWorker)
    worker.front_end_config = SimpleNamespace(server_auth=config,
                                              host="127.0.0.1",
                                              port=9901,
                                              name="resource-check",
                                              debug=False)
    server = await worker.create_mcp_server()
    assert server.settings.auth is not None
    assert server.settings.auth.validate_token_resource is expected_sdk_check
    assert str(server.settings.auth.resource_server_url) == "http://127.0.0.1:9901/"


@pytest.mark.parametrize("configured_audience", [None, ""])
@pytest.mark.parametrize("token_audiences,accepted",
                         [
                             (["http://127.0.0.1:9901/"], True),
                             (["other", "http://127.0.0.1:9901"], True),
                             (["other", "HTTP://127.0.0.1:9901"], True),
                             (["other", "http://127.0.0.1:9901/foreign"], False),
                             (["https://foreign.example/"], False),
                             ([], False),
                         ])
async def test_server_backend_enforces_resource_when_no_audience_policy_is_set(
    configured_audience: str | None,
    token_audiences: list[str],
    accepted: bool,
) -> None:
    """Enforce the advertised resource through SDK authentication when the audience is unset."""
    from starlette.authentication import AuthenticationBackend
    from starlette.middleware.authentication import AuthenticationMiddleware
    from starlette.requests import HTTPConnection

    config = OAuth2ResourceServerConfig(
        issuer_url="https://issuer.example",
        audience=configured_audience,
        introspection_endpoint="https://issuer.example/introspect",
        client_id="resource-server",
        client_secret=SecretStr("synthetic-secret"),
    )
    worker = MCPFrontEndPluginWorker.__new__(MCPFrontEndPluginWorker)
    worker.front_end_config = SimpleNamespace(server_auth=config,
                                              host="127.0.0.1",
                                              port=9901,
                                              name="resource-check",
                                              debug=False)
    server = await worker.create_mcp_server()
    server._token_verifier._bearer_token_validator.verify = AsyncMock(return_value=TokenValidationResult(
        active=True, client_id="client-a", audience=token_audiences, scopes=["read"], token_type="Bearer"))
    app = server.streamable_http_app()
    backend: AuthenticationBackend = next(middleware.kwargs["backend"] for middleware in app.user_middleware
                                          if middleware.cls is AuthenticationMiddleware)
    connection = HTTPConnection({"type": "http", "headers": [(b"authorization", b"Bearer synthetic")]})
    result = await backend.authenticate(connection)
    assert (result is not None) is accepted


@pytest.mark.parametrize("token_audience,accepted", [("service-id", True), ("other-service", False)])
async def test_configured_audience_is_enforced_by_worker_verifier(
    token_audience: str,
    accepted: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply the configured audience policy to introspection claims through the real validator."""
    config = OAuth2ResourceServerConfig(issuer_url="https://issuer.example",
                                        audience="service-id",
                                        introspection_endpoint="https://issuer.example/introspect",
                                        client_id="resource-server",
                                        client_secret=SecretStr("synthetic-secret"))
    worker = MCPFrontEndPluginWorker.__new__(MCPFrontEndPluginWorker)
    worker.front_end_config = SimpleNamespace(server_auth=config,
                                              host="127.0.0.1",
                                              port=9901,
                                              name="resource-check",
                                              debug=False)
    server = await worker.create_mcp_server()
    assert server.settings.auth is not None
    assert server.settings.auth.validate_token_resource is False
    introspect = AsyncMock(
        return_value={
            "active": True,
            "client_id": "client-a",
            "sub": "alice",
            "iss": "https://issuer.example",
            "aud": token_audience,
            "scope": "read",
        })
    monkeypatch.setattr(
        "nat.authentication.credential_validator.bearer_token_validator.AsyncOAuth2Client.introspect_token",
        introspect,
    )

    token = await server._token_verifier.verify_token("synthetic-opaque-token")

    introspect.assert_awaited_once_with("https://issuer.example/introspect",
                                        "synthetic-opaque-token",
                                        token_type_hint="access_token")
    assert (token is not None) is accepted
    if token is not None:
        assert token.resource == "service-id"
        assert token.subject == "alice"
        assert token.claims is not None
        assert token.claims["aud"] == [token_audience]
