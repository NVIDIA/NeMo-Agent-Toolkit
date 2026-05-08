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
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import NoReturn
from typing import Protocol

if TYPE_CHECKING:
    from nat.data_models.authentication import AuthResult

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_telemetry_exporter
from nat.data_models.component_ref import AuthenticationRef
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.observability.mixin.batch_config_mixin import BatchConfigMixin
from nat.plugins.a365.exceptions import A365AuthenticationError
from nat.plugins.a365.exceptions import A365ConfigurationError

logger = logging.getLogger(__name__)

# --- Pluggable token extractor (interface + dependency injection) ---

_TOKEN_EXTRACTOR_SUPPORTED = ("BearerTokenCred or HeaderCred(Authorization)")


class TokenExtractor(Protocol):
    """Callable that extracts a bearer token from NAT's AuthResult.

    Used when the default (BearerTokenCred or HeaderCred(Authorization)) does not
    match your auth provider's credential shape. Register a custom extractor with
    register_token_extractor(name, callable) and set token_extractor=name in config.
    """

    def __call__(self, auth_result: "AuthResult") -> str | None:
        ...


def _default_token_extractor(auth_result: "AuthResult") -> str | None:
    """Default extractor: BearerTokenCred or HeaderCred(Authorization).

    Returns the bearer token string, or None if neither credential type is present.
    Caller should raise A365AuthenticationError with a clear message when None.
    """
    from nat.authentication.interfaces import AUTHORIZATION_HEADER
    from nat.data_models.authentication import BearerTokenCred
    from nat.data_models.authentication import HeaderCred

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
        raise A365ConfigurationError(f"Unknown token_extractor '{name}'. "
                                     f"Registered: {sorted(_TOKEN_EXTRACTOR_REGISTRY.keys())}. "
                                     f"Use register_token_extractor(name, callable) to add custom extractors.")
    return _TOKEN_EXTRACTOR_REGISTRY[name]


def _raise_no_bearer_token(auth_result: "AuthResult") -> NoReturn:
    """Raise A365AuthenticationError with a clear message when no token could be extracted."""
    found = [type(c).__name__ for c in auth_result.credentials]
    raise A365AuthenticationError(f"No bearer token from auth provider. "
                                  f"Supported (default): {_TOKEN_EXTRACTOR_SUPPORTED}. "
                                  f"Found credential types: {found}")


class _AgentTokenCache:
    """Thread-safe token cache keyed by ``(agent_id, tenant_id)``.

    A process can host multiple A365 agents in the same workflow; each agent's
    bearer token must be tracked separately because the A365 backend partitions
    telemetry by ``gen_ai.agent.id`` and validates the token's ``appid`` claim
    against that route segment.
    """

    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        # key: (agent_id | None, tenant_id | None) -> (token, expires_at)
        self._entries: dict[tuple[str | None, str | None], tuple[str, datetime | None]] = {}

    @staticmethod
    def _expires_at_utc(expires_at: datetime | None) -> datetime | None:
        if expires_at is None:
            return None
        if expires_at.tzinfo is not None:
            return expires_at
        local_tz = datetime.now().astimezone().tzinfo
        return expires_at.replace(tzinfo=local_tz).astimezone(UTC)

    def get_token(self, agent_id: str | None, tenant_id: str | None) -> str | None:
        """Return cached token for the key if still valid (5 min buffer), else None."""
        with self._lock:
            entry = self._entries.get((agent_id, tenant_id))
            if entry is None:
                return None
            token, expires_at = entry
            expires_utc = self._expires_at_utc(expires_at)
            if expires_utc is None:
                return token
            buffer_time = datetime.now(UTC) + timedelta(minutes=5)
            return token if expires_utc > buffer_time else None

    def update_token(
        self,
        agent_id: str | None,
        tenant_id: str | None,
        *,
        token: str,
        expires_at: datetime | None,
    ) -> None:
        with self._lock:
            self._entries[(agent_id, tenant_id)] = (token, expires_at)

    def is_expiring_soon(
        self,
        agent_id: str | None,
        tenant_id: str | None,
        buffer_minutes: int = 5,
    ) -> bool:
        """True if the cached token is unset, expired, or expiring within ``buffer_minutes``."""
        with self._lock:
            entry = self._entries.get((agent_id, tenant_id))
            if entry is None:
                return True
            _, expires_at = entry
            expires_utc = self._expires_at_utc(expires_at)
            if expires_utc is None:
                return False
            buffer_time = datetime.now(UTC) + timedelta(minutes=buffer_minutes)
            return expires_utc <= buffer_time


class A365TelemetryExporter(BatchConfigMixin, TelemetryExporterBaseConfig, name="a365"):
    """A telemetry exporter to transmit traces to Microsoft Agent 365 backend.

    Auth: the referenced auth provider should return a bearer token via
    BearerTokenCred or HeaderCred(Authorization). For other credential shapes,
    register a custom token extractor with register_token_extractor(name, callable)
    and set token_extractor=name.

    Identity: when wired through the A365 front-end, each turn's agent and tenant
    ids are sourced from ``context.activity.get_agentic_instance_id()`` and
    ``context.activity.get_agentic_tenant_id()``. The ``agent_id`` and ``tenant_id``
    fields below are fallbacks used only when no per-turn identity is available
    (e.g., CLI workflows or non-agentic activities).

    Limitations to be aware of:

    - **Single auth provider**: ``token_resolver`` is a single ``AuthenticationRef``
      whose MSAL ``client_id`` becomes the ``appid`` claim of every emitted token.
      Hosting multiple A365 agents with distinct app registrations in the same
      process is not currently supported — only the agent matching the auth
      provider's ``client_id`` will pass A365's token-validation check.
    - **Cross-turn batching**: identity is read at export time from a contextvar.
      ``BatchingProcessor`` schedules a flush task on first enqueue; that task
      inherits a copy of the contextvar at creation time. Spans queued by later
      turns into the same batch are exported under the first turn's identity.
      For single-bot deployments this is invisible; for multi-bot deployments
      it can intermittently misattribute telemetry.
    """

    agent_id: str | None = Field(
        default=None,
        description=("Fallback Agent 365 agent ID, used when the front-end cannot supply a per-turn "
                     "identity via ``set_turn_identity``. Required for non-agentic / CLI workflows."),
    )
    tenant_id: str | None = Field(
        default=None,
        description=("Fallback Azure tenant ID, used when the front-end cannot supply a per-turn "
                     "identity via ``set_turn_identity``. Required for non-agentic / CLI workflows."),
    )
    token_resolver: AuthenticationRef = Field(
        description="Reference to NAT auth provider for token resolution (e.g., 'a365_auth')")
    token_extractor: str | None = Field(
        default=None,
        description=
        "Optional name of a registered token extractor. Default uses BearerTokenCred or HeaderCred(Authorization).")
    cluster_category: str = Field(default="prod", description="Cluster category/environment (e.g., 'prod', 'dev')")
    use_s2s_endpoint: bool = Field(default=False,
                                   description="Use service-to-service endpoint instead of standard endpoint")
    suppress_invoke_agent_input: bool = Field(default=False,
                                              description="Suppress input messages for InvokeAgent spans")


@register_telemetry_exporter(config_type=A365TelemetryExporter)
async def a365_telemetry_exporter(config: A365TelemetryExporter, builder: Builder):
    """Create an Agent 365 telemetry exporter.

    Auth resolution is deferred to first use of a given (agent_id, tenant_id)
    pair: the synchronous SDK callback returns whatever is in the cache, and
    the async helper on A365OtelExporter populates / refreshes entries before
    each export. This supports multi-agent processes (one cache entry per
    serving agent) and keeps WorkflowBuilder.__aenter__ free of auth I/O.
    """
    from nat.plugins.a365.telemetry.a365_exporter import A365OtelExporter

    token_extractor_fn = _get_token_extractor(config.token_extractor)
    token_cache = _AgentTokenCache()

    def token_resolver(agent_id: str, tenant_id: str) -> str | None:
        """Sync callable for the A365 SDK; returns cached bearer for the key.

        The async refresh path on A365OtelExporter ensures the cache is filled
        before the SDK calls this on each export.
        """
        return token_cache.get_token(agent_id, tenant_id)

    logger.info(
        "A365 telemetry exporter initialized for agent_id=%s, tenant_id=%s, "
        "cluster=%s, token_resolver=deferred (auth resolved per-agent on export)",
        config.agent_id,
        config.tenant_id,
        config.cluster_category,
    )

    exporter = A365OtelExporter(
        agent_id=config.agent_id,
        tenant_id=config.tenant_id,
        token_resolver=token_resolver,
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
