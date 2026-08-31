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
"""Configuration for validating inbound JWT identity credentials."""

from urllib.parse import urlparse

from pydantic import Field
from pydantic import field_validator

from nat.data_models.authentication import AuthProviderBaseConfig


class JwtAuthProviderConfig(AuthProviderBaseConfig, name="jwt"):
    """Named JWT verification policy for inbound identity credentials."""

    issuer_url: str = Field(description="Expected JWT issuer claim.")
    jwks_uri: str = Field(description="Endpoint containing trusted public keys for signature verification.")
    audience: str = Field(description="Expected JWT audience claim.")
    scopes: list[str] = Field(default_factory=list, description="Scopes required in a verified JWT.")
    timeout: float = Field(default=10.0, gt=0, description="HTTP timeout for JWKS requests.")
    leeway: int = Field(default=60, ge=0, description="Clock-skew allowance for JWT time claims, in seconds.")

    @field_validator("issuer_url", "jwks_uri")
    @classmethod
    def require_secure_url(cls, value: str, info) -> str:
        parsed = urlparse(value)
        is_local_http = parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if not parsed.netloc or (parsed.scheme != "https" and not is_local_http):
            raise ValueError(f"{info.field_name} must use HTTPS (HTTP is allowed only for localhost)")
        return value
