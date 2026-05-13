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
"""Structured user identity model supporting multiple credential sources."""

import base64
import typing
import uuid

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PrivateAttr
from pydantic import SecretStr
from pydantic import model_validator

from nat.data_models.common import OptionalSecretStr
from nat.data_models.common import SerializableSecretStr

_USER_ID_NAMESPACE: uuid.UUID = uuid.uuid5(uuid.NAMESPACE_DNS, "nemo-agent-toolkit")


class JwtUserInfo(BaseModel):
    """JWT-derived identity fields extracted from decoded token claims.

    Registered claims (``sub``, ``iss``, ``aud``, ``exp``, ``iat``) per RFC 7519.
    Identity claims (``email``, ``preferred_username``, ``name``) per OpenID Connect Core 1.0.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    given_name: str | None = Field(default=None, description="Given name (``given_name`` claim).")
    family_name: str | None = Field(default=None, description="Family name (``family_name`` claim).")
    email: str | None = Field(default=None, description="Email address (``email`` claim).")
    preferred_username: str | None = Field(default=None, description="Login or username.")
    roles: list[str] = Field(default_factory=list, description="Role claims.")
    groups: list[str] = Field(default_factory=list, description="Group memberships.")
    scopes: list[str] = Field(default_factory=list, description="OAuth2 scopes granted.")
    issuer: str | None = Field(default=None, description="``iss`` claim; identifies the IdP.")
    subject: str | None = Field(default=None, description="``sub`` claim; canonical IdP user identifier.")
    audience: list[str] | None = Field(default=None, description="``aud`` claim.")
    expires_at: int | None = Field(default=None, description="``exp`` (unix timestamp).")
    issued_at: int | None = Field(default=None, description="``iat`` (unix timestamp).")
    client_id: str | None = Field(default=None, description="OAuth2 client identifier (``azp`` or ``client_id``).")
    claims: dict[str, typing.Any] = Field(default_factory=dict, description="Raw JWT claims dict.")

    @property
    def identity_claim(self) -> str | None:
        """Return the first non-empty value using ``sub > email > preferred_username`` precedence.
        ``sub`` is the stable, locally-unique identifier per RFC 7519 Section 4.1.2.
        ``email`` and ``preferred_username`` are OIDC fallbacks (OpenID Connect Core 1.0 Section 5.1).
        """
        for key in ("sub", "email", "preferred_username"):
            val: typing.Any = self.claims.get(key)
            if val and isinstance(val, str) and val.strip():
                return val.strip()
        for attr_val in (self.subject, self.email, self.preferred_username):
            if attr_val and isinstance(attr_val, str) and attr_val.strip():
                return attr_val.strip()
        return None


class BasicUserInfo(BaseModel):
    """Username/password identity.

    The user provides ``username`` and ``password``.  A base64-encoded
    ``credential`` (``base64(username:password)``) is derived automatically
    and used as the identity key for UUID v5 generation.

    Because the password is part of the identity key, changing a password
    produces a new ``user_id`` and the user's prior per-user workflow state becomes inaccessible.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    username: str = Field(min_length=1, description="Unique username identifying this user.")
    password: SerializableSecretStr = Field(description="Password for this user.")

    _credential: str = PrivateAttr()

    def model_post_init(self, __context: typing.Any) -> None:
        object.__setattr__(
            self,
            "_credential",
            base64.b64encode(f"{self.username}:{self.password.get_secret_value()}".encode()).decode(),
        )

    @property
    def credential(self) -> str:
        """Base64-encoded ``username:password`` used to differentiate users."""
        return self._credential


class UserInfo(BaseModel):
    """Resolved user identity, independent of how it was identified.

    Construct with exactly one identity source::

        UserInfo(basic_user=BasicUserInfo(username="alice", password="s3cret"))
        UserInfo(api_key=SecretStr("sk-service-abc123"))

    For runtime credentials (session cookie / JWT), use ``UserManager``
    or the ``_from_*`` factory classmethods.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    basic_user: BasicUserInfo | None = Field(default=None, description="Username/password identity.")
    api_key: OptionalSecretStr = Field(default=None, description="Static API key identity.")
    _user_id: str = PrivateAttr(default="")
    _session_cookie: str | None = PrivateAttr(default=None)
    _jwt: JwtUserInfo | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_single_identity_source(self) -> "UserInfo":
        sources: int = sum(1 for s in (self.basic_user, self.api_key) if s is not None)
        if sources > 1:
            raise ValueError(f"At most one identity source (basic_user, api_key) may be set, got {sources}")
        return self

    def model_post_init(self, __context: typing.Any) -> None:
        if self.basic_user is not None:
            self._set_user_id(self.basic_user.credential)
        elif self.api_key is not None:
            self._set_user_id(self.api_key.get_secret_value())

    def get_user_id(self) -> str:
        """Return the user ID."""
        return self._user_id

    def _set_user_id(self, identity_key: str) -> None:
        """Derive and set the deterministic UUID from an identity source value."""
        computed: str = str(uuid.uuid5(_USER_ID_NAMESPACE, identity_key))
        object.__setattr__(self, "_user_id", computed)

    def get_user_details(self) -> JwtUserInfo | BasicUserInfo | str | None:
        """Return the identity-source data used to create this user.

        Returns:
            ``JwtUserInfo`` for JWT users, ``BasicUserInfo`` for
            username/password users, the raw API key or cookie string for
            those users, or ``None`` if no source was set.
        """
        if self._jwt is not None:
            return self._jwt
        if self.basic_user is not None:
            return self.basic_user
        if self.api_key is not None:
            return self.api_key.get_secret_value()
        if self._session_cookie is not None:
            return self._session_cookie
        return None

    @classmethod
    def _from_session_cookie(cls, cookie: str) -> "UserInfo":
        instance: UserInfo = cls()
        object.__setattr__(instance, "_session_cookie", cookie)
        instance._set_user_id(cookie)
        return instance

    @classmethod
    def _from_api_key(cls, api_key: str) -> "UserInfo":
        return cls(api_key=SecretStr(api_key))

    @classmethod
    def _from_jwt(cls, jwt_info: JwtUserInfo) -> "UserInfo":
        identity: str | None = jwt_info.identity_claim
        if identity is None:
            raise ValueError("JWT contains no usable identity claim (sub, email, preferred_username)")
        instance: UserInfo = cls()
        object.__setattr__(instance, "_jwt", jwt_info)
        instance._set_user_id(identity)
        return instance
