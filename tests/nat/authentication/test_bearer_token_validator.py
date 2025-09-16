import time
from typing import Any

import pytest
from authlib.jose import JsonWebKey
from authlib.jose import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from nat.authentication.credential_validator.bearer_token_validator import BearerTokenValidator
from nat.data_models.authentication import TokenValidationResult


# ========= Dynamic key generation (no hardcoded PEM) =========
@pytest.fixture(scope="session")
def rsa_private_pem() -> str:
    """Generate a fresh RSA private key (PKCS8 PEM) for signing JWTs."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pem.decode("utf-8")


@pytest.fixture(scope="session")
def jwks_from_private(rsa_private_pem: str) -> dict[str, Any]:
    """Create a JWKS dict (public only) from the generated private key."""
    # Import the private key and generate public key PEM
    from cryptography.hazmat.primitives.serialization import Encoding
    from cryptography.hazmat.primitives.serialization import PublicFormat
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    private_key = load_pem_private_key(rsa_private_pem.encode(), password=None)
    public_key = private_key.public_key()

    # Convert public key to PEM format
    public_key_pem = public_key.public_bytes(encoding=Encoding.PEM, format=PublicFormat.SubjectPublicKeyInfo)

    # Create JWK from public key PEM
    jwk = JsonWebKey.import_key(public_key_pem)
    jwk_dict = jwk.as_dict()

    # Add a key ID for easier matching
    jwk_dict['kid'] = 'test-key-id'
    jwk_dict['use'] = 'sig'
    jwk_dict['alg'] = 'RS256'

    return {"keys": [jwk_dict]}


# ========= Simple test constants =========
ISSUER = "https://issuer.test"
JWKS_URI = f"{ISSUER}/.well-known/jwks.json"
DISCOVERY_URL = f"{ISSUER}/.well-known/openid-configuration"
AUDIENCE = "api://resource"
SCOPES = ["read", "write"]


# ========= Helpers =========
def _make_jwt(
    rsa_private_pem: str,
    exp_offset_secs: int = 300,
    nbf_offset_secs: int = 0,
    scopes: list[str] | None = None,
    audience: str | list[str] | None = AUDIENCE,
    issuer: str = ISSUER,
) -> str:
    now = int(time.time())
    payload: dict[str, Any] = {
        "iss": issuer,
        "sub": "user-123",
        "aud": audience,
        "iat": now,
        "nbf": now + nbf_offset_secs if nbf_offset_secs else now,
        "exp": now + exp_offset_secs,
        "scope": " ".join(scopes) if scopes else None,
        "azp": "client-abc",
        "jti": "jwt-id-xyz",
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    # Create JWT header with key ID
    header = {"alg": "RS256", "typ": "JWT", "kid": "test-key-id"}

    token = jwt.encode(header, payload, rsa_private_pem)
    return token.decode() if isinstance(token, bytes) else token


class _FakeHTTPResponse:

    def __init__(self, json_data: dict[str, Any], status: int = 200):
        self._json = json_data
        self.status_code = status

    def json(self) -> dict[str, Any]:
        return self._json

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeAsyncHTTPClient:
    """Fake httpx.AsyncClient that returns discovery and JWKS payloads (success paths)."""

    def __init__(self, *args, **kwargs):
        self._closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._closed = True

    async def get(self, url: str, *args, **kwargs):
        # Filled by patching fixture to use test's dynamic JWKS
        jwks = kwargs.pop("_jwks_payload", None)

        if url == DISCOVERY_URL:
            return _FakeHTTPResponse({"jwks_uri": JWKS_URI})
        if url == JWKS_URI:
            return _FakeHTTPResponse(jwks)
        return _FakeHTTPResponse({"error": "not found"}, status=404)


class _FakeAsyncOAuth2Client:
    """Fake AsyncOAuth2Client that returns a configurable introspection response."""
    call_count = 0
    response: dict[str, Any] = {}

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def introspect_token(self, endpoint: str, token: str, token_type_hint: str = "access_token"):
        _FakeAsyncOAuth2Client.call_count += 1
        return _FakeAsyncOAuth2Client.response


@pytest.fixture(autouse=True)
def patch_httpx_and_oauth(monkeypatch, jwks_from_private):
    # Patch httpx.AsyncClient **inside the module under test**
    monkeypatch.setattr(
        "nat.authentication.credential_validator.bearer_token_validator.httpx.AsyncClient",
        _FakeAsyncHTTPClient,
        raising=True,
    )

    # Inject dynamic JWKS into our fake httpx client
    orig_get = _FakeAsyncHTTPClient.get

    async def get_with_jwks(self, url: str, *args, **kwargs):
        kwargs["_jwks_payload"] = jwks_from_private
        return await orig_get(self, url, *args, **kwargs)

    monkeypatch.setattr(_FakeAsyncHTTPClient, "get", get_with_jwks, raising=True)

    # Patch AsyncOAuth2Client **inside the module under test**
    monkeypatch.setattr(
        "nat.authentication.credential_validator.bearer_token_validator.AsyncOAuth2Client",
        _FakeAsyncOAuth2Client,
        raising=True,
    )

    # Reset counters per test
    _FakeAsyncOAuth2Client.call_count = 0
    _FakeAsyncOAuth2Client.response = {}
    yield


# ========= Validators (instances) =========
@pytest.fixture
def validator_with_discovery():
    return BearerTokenValidator(
        issuer=ISSUER,
        audience=AUDIENCE,
        scopes=SCOPES,
        discovery_url=DISCOVERY_URL,
        timeout=3.0,
        leeway=30,
    )


@pytest.fixture
def validator_with_jwks():
    return BearerTokenValidator(
        issuer=ISSUER,
        audience=AUDIENCE,
        scopes=SCOPES,
        jwks_uri=JWKS_URI,
        timeout=3.0,
        leeway=30,
    )


@pytest.fixture
def validator_opaque():
    return BearerTokenValidator(
        issuer=ISSUER,
        audience=AUDIENCE,
        scopes=SCOPES,
        introspection_endpoint=f"{ISSUER}/introspect",
        client_id="client-abc",
        client_secret="secret-xyz",
        timeout=3.0,
        leeway=30,
    )


@pytest.fixture
def validator_both():
    return BearerTokenValidator(
        issuer=ISSUER,
        audience=AUDIENCE,
        scopes=SCOPES,
        jwks_uri=JWKS_URI,
        introspection_endpoint=f"{ISSUER}/introspect",
        client_id="client-abc",
        client_secret="secret-xyz",
        timeout=3.0,
        leeway=30,
    )


# ========= JWT path =========
@pytest.mark.asyncio
async def test_jwt_happy_path_via_discovery(rsa_private_pem):
    # Create a minimal validator with no audience or scope requirements
    validator = BearerTokenValidator(
        issuer=ISSUER,
        discovery_url=DISCOVERY_URL,
        timeout=3.0,
        leeway=30,
    )

    # Create a simple JWT with matching issuer
    token = _make_jwt(rsa_private_pem, exp_offset_secs=300, scopes=SCOPES, issuer=ISSUER)
    res = await validator.verify(token)

    assert isinstance(res, TokenValidationResult)
    assert res.active is True
    assert res.issuer == ISSUER


@pytest.mark.asyncio
async def test_jwt_wrong_audience_rejected(validator_with_jwks, rsa_private_pem):
    token = _make_jwt(rsa_private_pem, exp_offset_secs=300, audience="other-aud", scopes=SCOPES)
    res = await validator_with_jwks.verify(token)
    assert res.active is False


# ========= Opaque path =========
@pytest.mark.asyncio
async def test_opaque_happy_path(validator_opaque):
    now = int(time.time())
    _FakeAsyncOAuth2Client.response = {
        "active": True,
        "client_id": "client-abc",
        "username": "alice",
        "token_type": "access_token",
        "exp": now + 600,
        "nbf": now - 10,
        "iat": now - 20,
        "sub": "user-123",
        "aud": [AUDIENCE],
        "iss": ISSUER,
        "jti": "opaque-id-1",
        "scope": "read write",
    }
    token = "opaque-secret-token-1234567890"
    res = await validator_opaque.verify(token)
    assert isinstance(res, TokenValidationResult)
    assert res.active is True
    assert res.audience == [AUDIENCE]
    assert set(res.scopes or []) == set(SCOPES)


@pytest.mark.asyncio
async def test_opaque_missing_scope_rejected(validator_opaque):
    now = int(time.time())
    _FakeAsyncOAuth2Client.response = {
        "active": True,
        "client_id": "client-abc",
        "token_type": "access_token",
        "exp": now + 600,
        "aud": [AUDIENCE],
        "iss": ISSUER,
        "scope": "read",  # missing "write"
    }
    token = "opaque-missing-scope"
    res = await validator_opaque.verify(token)
    assert res.active is False


# ========= Routing tests =========
@pytest.mark.asyncio
async def test_routing_uses_jwt_when_three_segments(validator_both, rsa_private_pem):
    jwt_token = _make_jwt(rsa_private_pem, exp_offset_secs=300, scopes=SCOPES)
    res = await validator_both.verify(jwt_token)
    assert res.active is True  # verified via JWKS/JWT path


@pytest.mark.asyncio
async def test_routing_uses_opaque_when_non_jwt(validator_both):
    now = int(time.time())
    _FakeAsyncOAuth2Client.response = {
        "active": True,
        "client_id": "client-abc",
        "token_type": "access_token",
        "exp": now + 600,
        "aud": [AUDIENCE],
        "iss": ISSUER,
        "scope": "read write",
    }
    non_jwt = "opaque-not-jwt-123456"
    res = await validator_both.verify(non_jwt)
    assert res.active is True  # verified via introspection
