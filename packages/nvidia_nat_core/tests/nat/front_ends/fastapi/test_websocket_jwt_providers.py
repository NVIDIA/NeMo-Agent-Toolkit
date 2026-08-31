# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from nat.authentication.jwt.jwt_auth_provider import JwtAuthProvider
from nat.authentication.jwt.jwt_auth_provider_config import JwtAuthProviderConfig
from nat.front_ends.fastapi.routes.websocket import _get_jwt_validators


def _provider(issuer: str) -> JwtAuthProvider:
    return JwtAuthProvider(
        JwtAuthProviderConfig(
            issuer_url=issuer,
            jwks_uri=f"{issuer}/jwks.json",
            audience="nat-api",
        ))


async def test_get_jwt_validators_resolves_named_providers_by_issuer():
    corporate = _provider("https://corporate.example.com")
    partner = _provider("https://partner.example.com")
    worker = MagicMock()
    worker.front_end_config.identity_authentication = ["corporate_jwt", "partner_jwt"]
    session_manager = MagicMock()
    session_manager.shared_builder.get_auth_provider = AsyncMock(side_effect=[corporate, partner])

    validators = await _get_jwt_validators(worker, session_manager)

    assert validators == {
        "https://corporate.example.com": corporate.validator,
        "https://partner.example.com": partner.validator,
    }


async def test_get_jwt_validators_rejects_non_jwt_provider():
    worker = MagicMock()
    worker.front_end_config.identity_authentication = ["not_jwt"]
    session_manager = MagicMock()
    session_manager.shared_builder.get_auth_provider = AsyncMock(return_value=MagicMock())

    with pytest.raises(ValueError, match="must have _type: jwt"):
        await _get_jwt_validators(worker, session_manager)


async def test_get_jwt_validators_rejects_duplicate_issuers():
    first = _provider("https://identity.example.com")
    second = _provider("https://identity.example.com")
    worker = MagicMock()
    worker.front_end_config.identity_authentication = ["first", "second"]
    session_manager = MagicMock()
    session_manager.shared_builder.get_auth_provider = AsyncMock(side_effect=[first, second])

    with pytest.raises(ValueError, match="Duplicate JWT issuer"):
        await _get_jwt_validators(worker, session_manager)
