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

import pytest
from pydantic import ValidationError

from nat.authentication.jwt.jwt_auth_provider_config import JwtAuthProviderConfig


def test_jwt_auth_provider_requires_complete_verification_policy():
    with pytest.raises(ValidationError):
        JwtAuthProviderConfig(issuer_url="https://identity.example.com")


def test_jwt_auth_provider_accepts_complete_verification_policy():
    config = JwtAuthProviderConfig(
        issuer_url="https://identity.example.com",
        jwks_uri="https://identity.example.com/jwks.json",
        audience="nat-api",
        scopes=["workflow:resume"],
    )

    assert config.issuer_url == "https://identity.example.com"
    assert config.jwks_uri == "https://identity.example.com/jwks.json"
    assert config.audience == "nat-api"
    assert config.scopes == ["workflow:resume"]


@pytest.mark.parametrize("field", ["issuer_url", "jwks_uri"])
def test_jwt_auth_provider_rejects_insecure_remote_urls(field: str):
    values = {
        "issuer_url": "https://identity.example.com",
        "jwks_uri": "https://identity.example.com/jwks.json",
        "audience": "nat-api",
    }
    values[field] = "http://identity.example.com"

    with pytest.raises(ValidationError, match="must use HTTPS"):
        JwtAuthProviderConfig(**values)
