# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from urllib.parse import urlparse

from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.common import OptionalSecretStr


class OAuth2ResourceServerConfig(AuthProviderBaseConfig, name="oauth2_resource_server"):
    """OAuth 2.0 Resource Server authentication configuration.

    Supports:
      • JWT access tokens via JWKS / OIDC Discovery / issuer fallback
      • Opaque access tokens via RFC 7662 introspection
    """

    issuer_url: str = Field(
        description=("The unique issuer identifier for an authorization server. "
                     "Required for validation and used to derive the default JWKS URI "
                     "(<issuer_url>/.well-known/jwks.json) if `jwks_uri` and `discovery_url` are not provided."), )
    scopes: list[str] = Field(
        default_factory=list,
        description="Scopes required by this API. Validation ensures the token grants all listed scopes.",
    )
    audience: str | None = Field(
        default=None,
        description=(
            "Expected audience (`aud`) claim for this API. If set, validation will reject tokens without this audience."
        ),
    )

    # JWT verification params
    jwks_uri: str | None = Field(
        default=None,
        description=("Direct JWKS endpoint URI for JWT signature verification. "
                     "Optional if discovery or issuer is provided."),
    )
    discovery_url: str | None = Field(
        default=None,
        description=("OIDC discovery metadata URL. Used to automatically resolve JWKS and introspection endpoints."),
    )

    # Opaque token (introspection) params
    introspection_endpoint: str | None = Field(
        default=None,
        description=("RFC 7662 token introspection endpoint. "
                     "Required for opaque token validation and must be used with `client_id` and `client_secret`."),
    )
    client_id: str | None = Field(
        default=None,
        description="OAuth2 client ID for authenticating to the introspection endpoint (opaque token validation).",
    )
    client_secret: OptionalSecretStr = Field(
        default=None,
        description="OAuth2 client secret for authenticating to the introspection endpoint (opaque token validation).",
    )

    @staticmethod
    def _is_https_or_localhost(url: str) -> bool:
        try:
            value = urlparse(url)
            if not value.scheme or not value.netloc:
                return False
            if value.scheme == "https":
                return True
            return value.scheme == "http" and (value.hostname in {"localhost", "127.0.0.1", "::1"})
        except Exception:
            return False

    @field_validator("issuer_url", "jwks_uri", "discovery_url", "introspection_endpoint")
    @classmethod
    def _require_valid_url(cls, value: str | None, info):
        if value is None:
            return value
        if not cls._is_https_or_localhost(value):
            raise ValueError(f"{info.field_name} must be HTTPS (http allowed only for localhost). Got: {value}")
        return value

    # ---------- Cross-field validation: ensure at least one viable path ----------

    @model_validator(mode="after")
    def _ensure_verification_path(self):
        """
        JWT path viable if any of: jwks_uri OR discovery_url OR issuer_url (fallback JWKS).
        Opaque path viable if: introspection_endpoint AND client_id AND client_secret.
        """
        has_jwt_path = bool(self.jwks_uri or self.discovery_url or self.issuer_url)
        has_opaque_path = bool(self.introspection_endpoint and self.client_id and self.client_secret)

        # If introspection endpoint is set, enforce creds are present
        if self.introspection_endpoint:
            missing = []
            if not self.client_id:
                missing.append("client_id")
            if not self.client_secret:
                missing.append("client_secret")
            if missing:
                raise ValueError(
                    f"introspection_endpoint configured but missing required credentials: {', '.join(missing)}")

        # Require at least one path
        if not (has_jwt_path or has_opaque_path):
            raise ValueError("Invalid configuration: no verification method available. "
                             "Configure one of the following:\n"
                             "  • JWT path: set jwks_uri OR discovery_url OR issuer_url (for JWKS fallback)\n"
                             "  • Opaque path: set introspection_endpoint + client_id + client_secret")

        return self
