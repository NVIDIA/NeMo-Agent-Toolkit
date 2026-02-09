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

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_telemetry_exporter
from nat.data_models.component_ref import AuthenticationRef
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.observability.mixin.batch_config_mixin import BatchConfigMixin
from nat.plugins.a365.exceptions import A365AuthenticationError

logger = logging.getLogger(__name__)


def _extract_token_from_auth_result(auth_result) -> str:
    """Extract bearer token from AuthResult credentials.

    Args:
        auth_result: AuthResult from auth provider

    Returns:
        Bearer token string

    Raises:
        A365AuthenticationError: If no bearer token found in credentials
    """
    from nat.data_models.authentication import BearerTokenCred, HeaderCred
    from nat.authentication.interfaces import AUTHORIZATION_HEADER

    for cred in auth_result.credentials:
        if isinstance(cred, BearerTokenCred):
            return cred.token.get_secret_value()
        elif isinstance(cred, HeaderCred) and cred.name == AUTHORIZATION_HEADER:
            header_value = cred.value.get_secret_value()
            # Strip "Bearer " prefix if present
            if header_value.startswith("Bearer "):
                return header_value[7:]  # Remove "Bearer " prefix
            return header_value

    raise A365AuthenticationError(
        f"No bearer token found in auth provider credentials. "
        f"Found credential types: {[type(c).__name__ for c in auth_result.credentials]}"
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

    def get_token(self) -> str | None:
        """Get cached token if still valid (with 5 minute buffer).

        Returns:
            Token string if valid, None if expired or expiring soon
        """
        with self._lock:
            if self._expires_at:
                buffer_time = datetime.now(timezone.utc) + timedelta(minutes=5)
                if self._expires_at > buffer_time:
                    return self._token
                return None
            else:
                # No expiration info, assume token is valid
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
            if self._expires_at is None:
                return False
            buffer_time = datetime.now(timezone.utc) + timedelta(minutes=buffer_minutes)
            return self._expires_at <= buffer_time


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

    token = _extract_token_from_auth_result(auth_result)
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
    """A telemetry exporter to transmit traces to Microsoft Agent 365 backend."""

    agent_id: str = Field(description="The Agent 365 agent ID")
    tenant_id: str = Field(description="The Azure tenant ID")
    token_resolver: AuthenticationRef = Field(
        description="Reference to NAT auth provider for token resolution (e.g., 'a365_auth')"
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
    """
    from nat.plugins.a365.telemetry.a365_exporter import A365OtelExporter

    token_resolver_callable, auth_provider, token_cache = await _create_token_resolver_from_auth_ref(
        config.token_resolver, builder
    )

    logger.info(
        f"A365 telemetry exporter initialized for agent_id={config.agent_id}, "
        f"tenant_id={config.tenant_id}, cluster={config.cluster_category}, "
        f"token_resolver=configured (auth_provider='{config.token_resolver}')"
    )

    exporter = A365OtelExporter(
        agent_id=config.agent_id,
        tenant_id=config.tenant_id,
        token_resolver=token_resolver_callable,
        auth_provider=auth_provider,  # Pass auth provider for proactive refresh
        token_cache=token_cache,  # Pass token cache for updating tokens
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
