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

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from microsoft_agents_a365.observability.core.exporters.agent365_exporter import (
    Agent365Exporter,
)

from nat.builder.context import ContextState
from nat.plugins.a365.exceptions import A365AuthenticationError, A365SDKError
from nat.plugins.opentelemetry.otel_span import OtelSpan
from nat.plugins.opentelemetry.otel_span_exporter import OtelSpanExporter
from opentelemetry.sdk.trace import Event as OtelEvent
from opentelemetry.trace import Link as OtelLink

logger = logging.getLogger(__name__)


class _ReadableSpanAdapter:
    """Adapter that makes OtelSpan compatible with A365's ReadableSpan interface.

    A365's Agent365Exporter expects ReadableSpan objects with specific attributes.
    This adapter wraps OtelSpan and provides the expected interface.

    """

    def __init__(self, otel_span: OtelSpan, tenant_id: str, agent_id: str):
        """Initialize the adapter.

        Args:
            otel_span: The OtelSpan to adapt
            tenant_id: The tenant ID to add as an attribute
            agent_id: The agent ID to add as an attribute
        """
        self.context = otel_span.get_span_context()

        # Convert parent Span to SpanContext if it exists (A365 expects SpanContext, not Span)
        if otel_span.parent is not None:
            self.parent = otel_span.parent.get_span_context()
        else:
            self.parent = None

        # Add tenant_id and agent_id to attributes (required for A365 partitioning)
        self.attributes = dict(otel_span.attributes)
        self.attributes["tenant.id"] = tenant_id
        self.attributes["gen_ai.agent.id"] = agent_id

        self.events = []
        for event in otel_span.events:
            if isinstance(event, dict):
                # Event stored as dict (from span_converter)
                event_name = event.get("name", "")
                event_attrs = event.get("attributes", {})
                event_timestamp = event.get("timestamp", otel_span.start_time)
                otel_event = OtelEvent(
                    name=event_name,
                    timestamp=event_timestamp,
                    attributes=event_attrs,
                )
            else:
                otel_event = event
            self.events.append(otel_event)

        self.links = []
        for link in otel_span.links:
            if isinstance(link, dict):
                # Link stored as dict
                link_context = link.get("context")
                link_attrs = link.get("attributes", {})
                if link_context:
                    otel_link = OtelLink(context=link_context, attributes=link_attrs)
                    self.links.append(otel_link)
            elif isinstance(link, OtelLink):
                self.links.append(link)

        self.name = otel_span.name
        self.kind = otel_span.kind
        self.start_time = otel_span.start_time
        self.end_time = otel_span.end_time or otel_span.start_time  # Ensure end_time is set
        self.status = otel_span.status
        self.instrumentation_scope = otel_span.instrumentation_scope
        self.resource = otel_span.resource


def _convert_otel_span_to_readable(otel_span: OtelSpan, tenant_id: str, agent_id: str) -> _ReadableSpanAdapter:
    """Convert an OtelSpan to a ReadableSpan-compatible adapter for A365 exporter.

    A365's Agent365Exporter expects ReadableSpan objects with specific attributes.
    This function creates a compatible adapter object.

    Args:
        otel_span: The OtelSpan to convert
        tenant_id: The tenant ID to add as an attribute
        agent_id: The agent ID to add as an attribute

    Returns:
        _ReadableSpanAdapter object that mimics ReadableSpan interface
    """
    return _ReadableSpanAdapter(otel_span, tenant_id, agent_id)


class A365OtelExporter(OtelSpanExporter):
    """Agent 365 exporter for AI workflow observability.

    Integrates A365's Agent365Exporter with NAT's telemetry system to send
    OpenTelemetry spans to Microsoft Agent 365 backend endpoints.

    Args:
        agent_id: The Agent 365 agent ID
        tenant_id: The Azure tenant ID
        token_resolver: Callable that resolves auth token (agent_id, tenant_id) -> token
        cluster_category: Cluster category/environment (e.g., 'prod', 'dev')
        use_s2s_endpoint: Use service-to-service endpoint instead of standard endpoint
        suppress_invoke_agent_input: Suppress input messages for InvokeAgent spans
        context_state: Execution context for isolation
        batch_size: Batch size for exporting
        flush_interval: Flush interval for exporting
        max_queue_size: Maximum queue size for exporting
        drop_on_overflow: Drop on overflow for exporting
        shutdown_timeout: Shutdown timeout for exporting
        resource_attributes: Additional resource attributes for spans
    """

    def __init__(
        self,
        agent_id: str,
        tenant_id: str,
        token_resolver: Callable[[str, str], str | None] | None,
        cluster_category: str = "prod",
        use_s2s_endpoint: bool = False,
        suppress_invoke_agent_input: bool = False,
        context_state: ContextState | None = None,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        max_queue_size: int = 1000,
        drop_on_overflow: bool = False,
        shutdown_timeout: float = 10.0,
        resource_attributes: dict[str, str] | None = None,
        auth_provider=None,
        token_cache=None,
        auth_ref=None,
        builder=None,
    ):
        """Initialize the A365 exporter."""
        super().__init__(
            context_state=context_state,
            batch_size=batch_size,
            flush_interval=flush_interval,
            max_queue_size=max_queue_size,
            drop_on_overflow=drop_on_overflow,
            shutdown_timeout=shutdown_timeout,
            resource_attributes=resource_attributes,
        )

        self._agent_id = agent_id
        self._tenant_id = tenant_id
        self._token_resolver = token_resolver
        self._cluster_category = cluster_category
        self._use_s2s_endpoint = use_s2s_endpoint
        self._suppress_invoke_agent_input = suppress_invoke_agent_input
        self._auth_provider = auth_provider
        self._token_cache = token_cache
        self._auth_ref = auth_ref
        self._builder = builder
        self._auth_resolve_lock = asyncio.Lock()

        # SDK requires token_resolver to be non-None, so if None is passed, SDK will raise ValueError
        self._a365_exporter = Agent365Exporter(
            token_resolver=token_resolver,
            cluster_category=cluster_category,
            use_s2s_endpoint=use_s2s_endpoint,
        )

        logger.info(
            f"A365 telemetry exporter initialized for agent_id={agent_id}, "
            f"tenant_id={tenant_id}, cluster={cluster_category}"
        )

    async def _resolve_auth_once(self) -> None:
        """Resolve auth provider and fill token cache on first export (lazy).

        Telemetry is built in __aenter__ before auth exists; by first export,
        populate_builder has run so we can resolve here. Keeps core unchanged.
        """
        if self._auth_provider is not None or self._auth_ref is None or self._builder is None:
            return
        if self._token_cache is None:
            return
        async with self._auth_resolve_lock:
            if self._auth_provider is not None:
                return
            try:
                auth_provider = await self._builder.get_auth_provider(self._auth_ref)
                from nat.builder.context import Context
                user_id = Context.get().user_id
                auth_result = await auth_provider.authenticate(user_id=user_id)
                if not auth_result.credentials:
                    raise A365AuthenticationError("No credentials available from auth provider")
                from nat.data_models.authentication import BearerTokenCred, HeaderCred
                from nat.authentication.interfaces import AUTHORIZATION_HEADER
                token = None
                for cred in auth_result.credentials:
                    if isinstance(cred, BearerTokenCred):
                        token = cred.token.get_secret_value()
                        break
                    if isinstance(cred, HeaderCred) and cred.name == AUTHORIZATION_HEADER:
                        hv = cred.value.get_secret_value()
                        token = hv[7:] if hv.startswith("Bearer ") else hv
                        break
                if token is None:
                    raise A365AuthenticationError(
                        f"No bearer token in credentials. "
                        f"Types: {[type(c).__name__ for c in auth_result.credentials]}"
                    )
                self._token_cache.update_token(token, auth_result.token_expires_at)
                self._auth_provider = auth_provider
            except Exception as e:
                logger.error(
                    f"Failed to resolve auth on first export (agent_id={self._agent_id}, "
                    f"tenant_id={self._tenant_id}): {e}",
                    exc_info=True,
                )
                raise

    async def _refresh_token_if_needed(self) -> None:
        """Refresh token proactively if it's expiring soon.

        Only refreshes if using AuthenticationRef-based token resolver.
        """
        if self._auth_provider is None or self._token_cache is None:
            return

        if not self._token_cache.is_expiring_soon(buffer_minutes=5):
            return

        try:
            from nat.builder.context import Context
            user_id = Context.get().user_id

            logger.debug(
                f"Refreshing token proactively (agent_id={self._agent_id}, tenant_id={self._tenant_id})"
            )
            auth_result = await self._auth_provider.authenticate(user_id=user_id)
            if not auth_result.credentials:
                logger.warning("Token refresh failed: no credentials available")
                return

            from nat.data_models.authentication import BearerTokenCred, HeaderCred
            from nat.authentication.interfaces import AUTHORIZATION_HEADER

            token: str | None = None
            for cred in auth_result.credentials:
                if isinstance(cred, BearerTokenCred):
                    token = cred.token.get_secret_value()
                    break
                elif isinstance(cred, HeaderCred) and cred.name == AUTHORIZATION_HEADER:
                    header_value = cred.value.get_secret_value()
                    if header_value.startswith("Bearer "):
                        token = header_value[7:]  # Remove "Bearer " prefix
                    else:
                        token = header_value
                    break

            if token is None:
                logger.warning(
                    f"No bearer token found in refreshed credentials. "
                    f"Found credential types: {[type(c).__name__ for c in auth_result.credentials]}"
                )
                return
            expires_at = auth_result.token_expires_at
            self._token_cache.update_token(token, expires_at)

            logger.debug(
                f"Token refreshed successfully (expires_at={expires_at}, "
                f"agent_id={self._agent_id}, tenant_id={self._tenant_id})"
            )
        except Exception as e:
            logger.warning(
                f"Failed to refresh token proactively (agent_id={self._agent_id}, "
                f"tenant_id={self._tenant_id}): {e}. Export may fail if token is expired."
            )

    async def export_otel_spans(self, spans: list[OtelSpan]) -> None:
        """Export a list of OtelSpans using the A365 exporter.

        Converts OtelSpans to ReadableSpan format and exports via A365's Agent365Exporter.
        Uses asyncio.run_in_executor to bridge NAT's async interface with A365's sync exporter.

        Args:
            spans (list[OtelSpan]): The list of spans to export.

        Raises:
            Exception: If there's an error during span export (logged but not re-raised).
        """
        if not spans:
            return

        await self._resolve_auth_once()
        await self._refresh_token_if_needed()

        try:
            readable_spans = []
            for otel_span in spans:
                readable_span = _convert_otel_span_to_readable(
                    otel_span=otel_span,
                    tenant_id=self._tenant_id,
                    agent_id=self._agent_id,
                )
                readable_spans.append(readable_span)

            logger.debug(
                f"A365 exporter: converted {len(spans)} OtelSpans to ReadableSpan format "
                f"(tenant={self._tenant_id}, agent={self._agent_id})"
            )

            # Bridge async/sync: A365's Agent365Exporter.export() is synchronous
            # Run it in a thread pool executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._a365_exporter.export, readable_spans)

            logger.debug(
                f"A365 exporter: successfully exported {len(readable_spans)} spans "
                f"(tenant={self._tenant_id}, agent={self._agent_id})"
            )
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(
                f"Error exporting spans to A365 (tenant={self._tenant_id}, agent={self._agent_id}): {e}",
                exc_info=True,
            )
            # Check if it's an authentication error (token resolver failure)
            if "authentication" in error_msg or "unauthorized" in error_msg or "token" in error_msg:
                raise A365AuthenticationError(
                    f"Authentication failed while exporting telemetry: {str(e)}",
                    original_error=e
                ) from e
            else:
                raise A365SDKError(
                    f"Failed to export spans to A365: {str(e)}",
                    sdk_component="Agent365Exporter",
                    original_error=e
                ) from e
