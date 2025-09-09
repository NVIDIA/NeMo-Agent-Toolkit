# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import pytest
from pydantic import TypeAdapter
from pydantic import ValidationError

from nat.data_models.authentication import AuthenticatedContext  # enums; models
from nat.data_models.authentication import AuthFlowType
from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import BasicAuthCred
from nat.data_models.authentication import BearerTokenCred
from nat.data_models.authentication import CookieCred
from nat.data_models.authentication import Credential
from nat.data_models.authentication import CredentialKind
from nat.data_models.authentication import CredentialLocation
from nat.data_models.authentication import HeaderAuthScheme
from nat.data_models.authentication import HeaderCred
from nat.data_models.authentication import HTTPMethod
from nat.data_models.authentication import QueryCred


# --------------------------------------------------------------------------- #
# ENUM COVERAGE
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "enum_member, expected_value",
    [
        (CredentialLocation.HEADER, "header"),
        (CredentialLocation.QUERY, "query"),
        (CredentialLocation.COOKIE, "cookie"),
        (CredentialLocation.BODY, "body"),
        (AuthFlowType.API_KEY, "api_key"),
        (AuthFlowType.OAUTH2_CLIENT_CREDENTIALS, "oauth2_client_credentials"),
        (AuthFlowType.OAUTH2_AUTHORIZATION_CODE, "oauth2_auth_code_flow"),
        (AuthFlowType.OAUTH2_PASSWORD, "oauth2_password"),
        (AuthFlowType.OAUTH2_DEVICE_CODE, "oauth2_device_code"),
        (AuthFlowType.HTTP_BASIC, "http_basic"),
        (AuthFlowType.NONE, "none"),
        (HeaderAuthScheme.BEARER, "Bearer"),
        (HeaderAuthScheme.X_API_KEY, "X-API-Key"),
        (HeaderAuthScheme.BASIC, "Basic"),
        (HeaderAuthScheme.CUSTOM, "Custom"),
        (HTTPMethod.GET, "GET"),
        (HTTPMethod.POST, "POST"),
        (HTTPMethod.PUT, "PUT"),
        (HTTPMethod.DELETE, "DELETE"),
        (HTTPMethod.PATCH, "PATCH"),
        (HTTPMethod.HEAD, "HEAD"),
        (HTTPMethod.OPTIONS, "OPTIONS"),
        (CredentialKind.HEADER, "header"),
        (CredentialKind.QUERY, "query"),
        (CredentialKind.COOKIE, "cookie"),
        (CredentialKind.BASIC, "basic_auth"),
        (CredentialKind.BEARER, "bearer_token"),
    ],
)
def test_enum_values(enum_member, expected_value):
    """Verify all Enum members keep their canonical .value strings."""
    assert enum_member.value == expected_value


# --------------------------------------------------------------------------- #
# AUTHENTICATED CONTEXT
# --------------------------------------------------------------------------- #
def test_authenticated_context_all_fields():
    ctx = AuthenticatedContext(
        headers={"X-Test": "1"},
        query_params={"q": "v"},
        cookies={"sid": "abc"},
        body={"foo": "bar"},
        metadata={"trace_id": "123"},
    )
    assert ctx.headers["X-Test"] == "1"
    assert ctx.query_params["q"] == "v"
    assert ctx.cookies["sid"] == "abc"
    assert ctx.body["foo"] == "bar"
    assert ctx.metadata["trace_id"] == "123"


def test_authenticated_context_extra_forbidden():
    """Extra attributes should raise a ValidationError because extra='forbid'."""
    with pytest.raises(ValidationError):
        AuthenticatedContext(headers={}, bogus="nope")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# CREDENTIAL MODEL VALIDATION & DISCRIMINATED UNION
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "payload, expected_cls",
    [
        ({
            "kind": "header", "name": "X-API-Key", "value": "secret"
        }, HeaderCred),
        ({
            "kind": "query", "name": "token", "value": "abc"
        }, QueryCred),
        ({
            "kind": "cookie", "name": "session", "value": "xyz"
        }, CookieCred),
        (
            {
                "kind": "basic_auth", "username": "u", "password": "p"
            },
            BasicAuthCred,
        ),
        (
            {
                "kind": "bearer_token", "token": "tok"
            },
            BearerTokenCred,
        ),
    ],
)
def test_credential_discriminator_parsing(payload, expected_cls):
    cred = TypeAdapter(Credential).validate_python(payload)
    assert isinstance(cred, expected_cls)
    # discriminator preserved
    assert cred.kind.value == payload["kind"]


def test_credential_invalid_kind():
    with pytest.raises(ValidationError):
        TypeAdapter(Credential).validate_python({"kind": "unknown", "name": "X", "value": "oops"})


# --------------------------------------------------------------------------- #
# AUTHRESULT HELPERS
# --------------------------------------------------------------------------- #
def _make_all_creds():
    """Helper to build a representative credential set."""
    return [
        HeaderCred(name="X-Trace", value="trc123"),
        QueryCred(name="limit", value="100"),
        CookieCred(name="sid", value="cookie123"),
        BearerTokenCred(token="bearer-tok"),
        BasicAuthCred(username="alice", password="wonderland"),
    ]


def test_as_requests_kwargs():
    creds = _make_all_creds()
    res = AuthResult(credentials=creds)
    kw = res.as_requests_kwargs()

    # Headers
    assert kw["headers"]["X-Trace"] == "trc123"
    # Bearer token adds Authorization header
    assert kw["headers"]["Authorization"] == "Bearer bearer-tok"
    # Query params
    assert kw["params"]["limit"] == "100"
    # Cookies
    assert kw["cookies"]["sid"] == "cookie123"
    # Basic-auth
    assert kw["auth"] == ("alice", "wonderland")


def test_attach_merges_in_place():
    creds = _make_all_creds()
    res = AuthResult(credentials=creds)

    target = {
        "headers": {
            "User-Agent": "pytest"
        },
        "params": {
            "existing": "param"
        },
    }
    res.attach(target)

    # Existing keys are preserved
    assert target["headers"]["User-Agent"] == "pytest"
    assert target["params"]["existing"] == "param"
    # New credential-derived entries are merged
    assert target["headers"]["X-Trace"] == "trc123"
    assert target["headers"]["Authorization"].startswith("Bearer")
    assert target["cookies"]["sid"] == "cookie123"


@pytest.mark.parametrize(
    "delta, expected",
    [
        (-1, True),  # expired
        (+10, False),  # not expired
        (None, False),  # no expiry supplied
    ],
)
def test_is_expired(delta, expected):
    if delta is None:
        res = AuthResult(credentials=[])
    else:
        expires = datetime.now(timezone.utc) + timedelta(seconds=delta)
        res = AuthResult(credentials=[], token_expires_at=expires)
    assert res.is_expired() is expected


def test_bearer_token_custom_header_and_scheme():
    cred = BearerTokenCred(
        token="tok",
        scheme="Token",
        header_name="X-Token",
    )
    res = AuthResult(credentials=[cred])
    kw = res.as_requests_kwargs()
    assert kw["headers"]["X-Token"] == "Token tok"
