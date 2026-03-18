# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from nat.data_models.authentication import AuthResult

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_telemetry_exporter
from nat.data_models.component_ref import AuthenticationRef
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.observability.mixin.batch_config_mixin import BatchConfigMixin
from nat.plugins.a365.exceptions import A365AuthenticationError, A365ConfigurationError

logger = logging.getLogger(__name__)

# --- Pluggable token extractor (interface + dependency injection) ---

_TOKEN_EXTRACTOR_SUPPORTED = (
    "BearerTokenCred or HeaderCred(Authorization)"
)


class TokenExtractor(Protocol):
    """Callable that extracts a bearer token from NAT's AuthResult.

    Used when the default (BearerTokenCred or HeaderCred(Authorization)) does not
    match your auth provider's credential shape. Register a custom extractor with
    register_token_extractor(name, callable) and set token_extractor=name in config.
    """

    def __call__(self, auth_result: "AuthResult") -> str | None: ...


def _default_token_extractor(auth_result: "AuthResult") -> str | None:
    """Default extractor: BearerTokenCred or HeaderCred(Authorization).

    Returns the bearer token string, or None if neither credential type is present.
    Caller should raise A365AuthenticationError with a clear message when None.
    """
    from nat.data_models.authentication import BearerTokenCred, HeaderCred
    from nat.authentication.interfaces import AUTHORIZATION_HEADER

    for cred in auth_result.credentials:
        if isinstance(cred, BearerTokenCred):
            return cred.token.get_secret_value()
        if isinstance(cred, HeaderCred) and cred.name == AUTHORIZATION_HEADER:
            raw = cred.value.get_secret_value()
            return raw[7:] if raw.startswith("Bearer ") else raw
    return None


_TOKEN_EXTRACTOR_REGISTRY: dict[str, Callable[["AuthResult"], str | None]] = {
    "default": _default_token_extractor,
}


def register_token_extractor(name: str, extractor: Callable[["AuthResult"], str | None]) -> None:
    """Register a custom token extractor for A365 telemetry.

    Use when your auth provider returns credentials in a shape the default extractor
    does not understand (e.g. a new NAT credential type). Then set
    token_extractor=\"name\" in your a365 telemetry exporter config.

    Args:
        name: Name to use in config (e.g. \"my_provider\").
        extractor: Callable (AuthResult) -> str | None. Return the bearer token or None.
    """
    _TOKEN_EXTRACTOR_REGISTRY[name] = extractor


def _get_token_extractor(name: str | None) -> Callable[["AuthResult"], str | None]:
    if name is None or name == "default":
        return _default_token_extractor
    if name not in _TOKEN_EXTRACTOR_REGISTRY:
        raise A365ConfigurationError(
            f"Unknown token_extractor '{name}'. "
            f"Registered: {sorted(_TOKEN_EXTRACTOR_REGISTRY.keys())}. "
            f"Use register_token_extractor(name, callable) to add custom extractors."
        )
    return _TOKEN_EXTRACTOR_REGISTRY[name]


def _raise_no_bearer_token(auth_result: "AuthResult") -> None:
    """Raise A365AuthenticationError with a clear message when no token could be extracted."""
    found = [type(c).__name__ for c in auth_result.credentials]
    raise A365AuthenticationError(
        f"No bearer token from auth provider. "
        f"Supported (default): {_TOKEN_EXTRACTOR_SUPPORTED}. "
        f"Found credential types: {found}"
    )


class _TokenCache:
    """Thread-safe token cache for AuthenticationRef-based token resolvers."""

    def __init__(self, token: str, expires_at: datetime | None):
        """Initialize token cache.

        Args:
            token: Initial bearer token
            expires_at: Token expiration time (UTC) or None if no expiration
        """
        import threading
        self._lock = threading.Lock()
        self._token = token
        self._expires_at = expires_at

    def _expires_at_utc(self) -> datetime | None:
        """Return expiration as timezone-aware UTC for comparison.

        If naive, assume local time (e.g. from provider using datetime.now() + delta).
        """
        if self._expires_at is None:
            return None
        if self._expires_at.tzinfo is not None:
            return self._expires_at
        local_tz = datetime.now().astimezone().tzinfo
        return self._expires_at.replace(tzinfo=local_tz).astimezone(timezone.utc)

    def get_token(self) -> str | None:
        """Get cached token if still valid (with 5 minute buffer).

        Returns:
            Token string if valid, None if expired or expiring soon
        """
        with self._lock:
            expires_utc = self._expires_at_utc()
            if expires_utc is not None:
                buffer_time = datetime.now(timezone.utc) + timedelta(minutes=5)
                if expires_utc > buffer_time:
                    return self._token
                return None
            return self._token

    def update_token(self, token: str, expires_at: datetime | None) -> None:
        """Update cached token.

        Args:
            token: New bearer token
            expires_at: New expiration time (UTC) or None
        """
        with self._lock:
            self._token = token
            self._expires_at = expires_at

    def is_expiring_soon(self, buffer_minutes: int = 5) -> bool:
        """Check if token is expiring soon.

        Args:
            buffer_minutes: Minutes before expiration to consider "soon" (default: 5)

        Returns:
            True if token expires within buffer time, False otherwise
        """
        with self._lock:
            expires_utc = self._expires_at_utc()
            if expires_utc is None:
                return False
            buffer_time = datetime.now(timezone.utc) + timedelta(minutes=buffer_minutes)
            return expires_utc <= buffer_time


async def _create_token_resolver_from_auth_ref(
    auth_ref: AuthenticationRef,
    builder: Builder,
) -> tuple[Callable[[str, str], str | None], object, _TokenCache]:
    """Create a token resolver callable from an AuthenticationRef.

    Creates a synchronous callable that returns cached tokens, with token refresh
    handled proactively before export operations.

    Args:
        auth_ref: AuthenticationRef to resolve
        builder: Builder instance for accessing auth providers

    Returns:
        Tuple of (token_resolver_callable, auth_provider_instance, token_cache)
        - token_resolver_callable: Sync callable (agent_id, tenant_id) -> token | None
        - auth_provider_instance: Auth provider for proactive token refresh
        - token_cache: Token cache instance for updating tokens

    Raises:
        A365AuthenticationError: If authentication fails or no token available
    """
    auth_provider = await builder.get_auth_provider(auth_ref)

    # Get user_id from context if available (needed for OAuth flows)
    from nat.builder.context import Context
    user_id = Context.get().user_id

    auth_result = await auth_provider.authenticate(user_id=user_id)
    if not auth_result.credentials:
        raise A365AuthenticationError("No credentials available from auth provider")

    extractor = _get_token_extractor(None)
    token = extractor(auth_result)
    if token is None:
        _raise_no_bearer_token(auth_result)
    expires_at = auth_result.token_expires_at

    token_cache = _TokenCache(token, expires_at)

    def token_resolver(agent_id: str, tenant_id: str) -> str | None:
        """Synchronous token resolver callable for SDK.

        Returns cached token if valid (with 5 minute buffer), None if expired.
        Token refresh is handled proactively before export operations.
        """
        return token_cache.get_token()

    return token_resolver, auth_provider, token_cache


class A365TelemetryExporter(BatchConfigMixin, TelemetryExporterBaseConfig, name="a365"):
    """A telemetry exporter to transmit traces to Microsoft Agent 365 backend.

    Auth: the referenced auth provider should return a bearer token via
    BearerTokenCred or HeaderCred(Authorization). For other credential shapes,
    register a custom token extractor with register_token_extractor(name, callable)
    and set token_extractor=name.
    """

    agent_id: str = Field(description="The Agent 365 agent ID")
    tenant_id: str = Field(description="The Azure tenant ID")
    token_resolver: AuthenticationRef = Field(
        description="Reference to NAT auth provider for token resolution (e.g., 'a365_auth')"
    )
    token_extractor: str | None = Field(
        default=None,
        description="Optional name of a registered token extractor. Default uses BearerTokenCred or HeaderCred(Authorization)."
    )
    cluster_category: str = Field(
        default="prod",
        description="Cluster category/environment (e.g., 'prod', 'dev')"
    )
    use_s2s_endpoint: bool = Field(
        default=False,
        description="Use service-to-service endpoint instead of standard endpoint"
    )
    suppress_invoke_agent_input: bool = Field(
        default=False,
        description="Suppress input messages for InvokeAgent spans"
    )


@register_telemetry_exporter(config_type=A365TelemetryExporter)
async def a365_telemetry_exporter(config: A365TelemetryExporter, builder: Builder):
    """Create an Agent 365 telemetry exporter.

    Integrates A365's Agent365Exporter with NAT's telemetry system to send
    OpenTelemetry spans to Microsoft Agent 365 backend endpoints.

    Auth is resolved lazily on first export (not at build time) because
    telemetry exporters are built in WorkflowBuilder.__aenter__ before
    auth providers exist. This keeps core unchanged; the plugin handles
    the timing constraint locally.
    """
    from nat.plugins.a365.telemetry.a365_exporter import A365OtelExporter

    token_extractor_fn = _get_token_extractor(config.token_extractor)

    # Defer auth: do not call get_auth_provider here (not available yet in __aenter__).
    token_cache = _TokenCache(None, None)

    def token_resolver(agent_id: str, tenant_id: str) -> str | None:
        """Sync callable for SDK; returns cached token (filled on first export)."""
        return token_cache.get_token()

    logger.info(
        f"A365 telemetry exporter initialized for agent_id={config.agent_id}, "
        f"tenant_id={config.tenant_id}, cluster={config.cluster_category}, "
        f"token_resolver=deferred (auth resolved on first export)"
    )

    exporter = A365OtelExporter(
        agent_id=config.agent_id,
        tenant_id=config.tenant_id,
        token_resolver=token_resolver,
        auth_provider=None,
        token_cache=token_cache,
        auth_ref=config.token_resolver,
        builder=builder,
        token_extractor=token_extractor_fn,
        cluster_category=config.cluster_category,
        use_s2s_endpoint=config.use_s2s_endpoint,
        suppress_invoke_agent_input=config.suppress_invoke_agent_input,
        batch_size=config.batch_size,
        flush_interval=config.flush_interval,
        max_queue_size=config.max_queue_size,
        drop_on_overflow=config.drop_on_overflow,
        shutdown_timeout=config.shutdown_timeout,
    )

    yield exporter
