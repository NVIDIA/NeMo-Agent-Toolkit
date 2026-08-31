# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from nat.authentication.jwt.jwt_auth_provider import JwtAuthProvider
from nat.authentication.jwt.jwt_auth_provider_config import JwtAuthProviderConfig
from nat.data_models.authentication import TokenValidationResult


def _provider() -> JwtAuthProvider:
    return JwtAuthProvider(
        JwtAuthProviderConfig(
            issuer_url="https://identity.example.com",
            jwks_uri="https://identity.example.com/jwks.json",
            audience="nat-api",
        ))


async def test_authenticate_returns_verified_token_result():
    provider = _provider()
    validation_result = TokenValidationResult(
        client_id="nat-client",
        subject="verified-user",
        issuer="https://identity.example.com",
        token_type="bearer",
        active=True,
    )
    with patch.object(provider, "verify", AsyncMock(return_value=validation_result)) as verify:
        result = await provider.authenticate(token="signed-token")

    verify.assert_awaited_once_with("signed-token")
    assert result.raw == validation_result.model_dump()


@pytest.mark.parametrize("token", [None, "", 123], ids=["missing", "empty", "non-string"])
async def test_authenticate_rejects_invalid_token_input(token):
    provider = _provider()

    with pytest.raises(ValueError, match="requires a token"):
        await provider.authenticate(token=token)


async def test_authenticate_rejects_inactive_token():
    provider = _provider()
    validation_result = TokenValidationResult(client_id=None, token_type="bearer", active=False)

    with patch.object(provider, "verify", AsyncMock(return_value=validation_result)) as verify:
        with pytest.raises(ValueError, match="JWT verification failed"):
            await provider.authenticate(token="rejected-token")

    verify.assert_awaited_once_with("rejected-token")
