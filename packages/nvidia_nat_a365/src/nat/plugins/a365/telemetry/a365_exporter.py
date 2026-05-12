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
import contextvars
import json
import logging
import os
import socket
import threading
import time
from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from microsoft_agents_a365.observability.core.constants import GEN_AI_AGENT_ID_KEY
from microsoft_agents_a365.observability.core.constants import GEN_AI_OPERATION_NAME_KEY
from microsoft_agents_a365.observability.core.constants import INVOKE_AGENT_OPERATION_NAME
from microsoft_agents_a365.observability.core.constants import TENANT_ID_KEY

try:
    from microsoft_agents_a365.observability.core.exporters.agent365_exporter import Agent365Exporter
except ImportError:
    from microsoft_agents_a365.observability.core.exporters import agent365_exporter as _agent365_exporter

    Agent365Exporter = _agent365_exporter._Agent365Exporter

from opentelemetry.sdk.trace import Event as OtelEvent
from opentelemetry.trace import Link as OtelLink

from nat.builder.context import ContextState
from nat.plugins.a365.exceptions import A365AuthenticationError
from nat.plugins.a365.exceptions import A365SDKError
from nat.plugins.a365.telemetry.register import _get_token_extractor
from nat.plugins.a365.telemetry.register import _raise_no_bearer_token
from nat.plugins.a365.turn_context import get_turn_identity
from nat.plugins.opentelemetry.otel_span import OtelSpan
from nat.plugins.opentelemetry.otel_span_exporter import OtelSpanExporter

logger = logging.getLogger(__name__)
_span_dump_lock = threading.Lock()

_DEFAULT_AGENT_NAME = "NeMo Agent Toolkit Workflow"
# A365_BLUEPRINT_ID intentionally has no default: it identifies a specific
# A365-registered application. Shipping a literal here would silently
# mis-attribute every operator's telemetry to a single blueprint. Operators
# MUST set the env var; otherwise the attribute is omitted.


@lru_cache(maxsize=1)
def _warn_missing_blueprint_id_once() -> None:
    logger.warning("A365_BLUEPRINT_ID is not set; spans will be exported without "
                   "'microsoft.a365.agent.blueprint.id'. A365 may reject or fail to "
                   "attribute these spans. Set A365_BLUEPRINT_ID to your registered "
                   "blueprint GUID to silence this warning.")


_REQUIRED_A365_ATTRIBUTES = (
    GEN_AI_OPERATION_NAME_KEY,
    GEN_AI_AGENT_ID_KEY,
    "gen_ai.agent.name",
    "microsoft.a365.agent.blueprint.id",
    "gen_ai.conversation.id",
    "microsoft.channel.name",
    "user.id",
    "client.address",
    "server.address",
    "server.port",
)


def _json_safe(value):
    """Convert SDK/OpenTelemetry values into a stable JSON-safe shape."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return value.value
    if hasattr(value, "name") and isinstance(value.name, str):
        return value.name
    return repr(value)


def _span_context_to_dict(context):
    if context is None:
        return None
    trace_id = getattr(context, "trace_id", None)
    span_id = getattr(context, "span_id", None)
    return {
        "trace_id": f"{trace_id:032x}" if isinstance(trace_id, int) else _json_safe(trace_id),
        "span_id": f"{span_id:016x}" if isinstance(span_id, int) else _json_safe(span_id),
        "is_remote": _json_safe(getattr(context, "is_remote", None)),
        "trace_flags": _json_safe(getattr(context, "trace_flags", None)),
        "trace_state": _json_safe(getattr(context, "trace_state", None)),
    }


def _resource_to_dict(resource):
    if resource is None:
        return None
    return {
        "attributes": _json_safe(getattr(resource, "attributes", {})),
        "schema_url": _json_safe(getattr(resource, "schema_url", None)),
    }


def _instrumentation_scope_to_dict(scope):
    if scope is None:
        return None
    return {
        "name": _json_safe(getattr(scope, "name", None)),
        "version": _json_safe(getattr(scope, "version", None)),
        "schema_url": _json_safe(getattr(scope, "schema_url", None)),
        "attributes": _json_safe(getattr(scope, "attributes", None)),
    }


def _event_to_dict(event):
    return {
        "name": _json_safe(getattr(event, "name", None)),
        "timestamp_unix_nano": _json_safe(getattr(event, "timestamp", None)),
        "attributes": _json_safe(getattr(event, "attributes", {})),
    }


def _link_to_dict(link):
    return {
        "context": _span_context_to_dict(getattr(link, "context", None)),
        "attributes": _json_safe(getattr(link, "attributes", {})),
    }


def _readable_span_to_dict(span):
    return {
        "name": _json_safe(getattr(span, "name", None)),
        "kind": _json_safe(getattr(span, "kind", None)),
        "context": _span_context_to_dict(getattr(span, "context", None)),
        "parent": _span_context_to_dict(getattr(span, "parent", None)),
        "start_time_unix_nano": _json_safe(getattr(span, "start_time", None)),
        "end_time_unix_nano": _json_safe(getattr(span, "end_time", None)),
        "status": _json_safe(getattr(span, "status", None)),
        "attributes": _json_safe(getattr(span, "attributes", {})),
        "events": [_event_to_dict(event) for event in getattr(span, "events", [])],
        "links": [_link_to_dict(link) for link in getattr(span, "links", [])],
        "resource": _resource_to_dict(getattr(span, "resource", None)),
        "instrumentation_scope": _instrumentation_scope_to_dict(getattr(span, "instrumentation_scope", None)),
    }


def _dump_readable_spans_if_configured(
    *,
    readable_spans,
    tenant_id: str,
    agent_id: str,
    use_s2s_endpoint: bool,
) -> None:
    dump_path = os.getenv("A365_SPAN_DUMP_PATH")
    dump_to_log = os.getenv("A365_SPAN_DUMP_TO_LOG", "").lower() in {"1", "true", "yes", "on"}
    if not dump_path and not dump_to_log:
        return

    payload = {
        "dumped_at": datetime.now(UTC).isoformat(),
        "tenant_id": tenant_id,
        "agent_id": agent_id,
        "use_s2s_endpoint": use_s2s_endpoint,
        "span_count": len(readable_spans),
        "spans": [_readable_span_to_dict(span) for span in readable_spans],
    }
    serialized_payload = json.dumps(payload, sort_keys=True)

    if dump_to_log:
        logger.info("A365_SPAN_DUMP_JSON %s", serialized_payload)

    if dump_path:
        try:
            path = Path(dump_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with _span_dump_lock:
                with path.open("a", encoding="utf-8") as dump_file:
                    dump_file.write(serialized_payload)
                    dump_file.write("\n")
            logger.info(
                "A365 exporter dumped %s spans to %s (agent_id=%s, tenant_id=%s)",
                len(readable_spans),
                dump_path,
                agent_id,
                tenant_id,
            )
        except Exception:
            logger.warning("Failed to dump A365 spans to %s", dump_path, exc_info=True)


def _set_string_attr(
    attributes: dict[str, Any],
    key: str,
    value: Any,
    *,
    overwrite: bool = False,
) -> None:
    if value is None:
        return
    text = str(value)
    if not text:
        return
    if overwrite or key not in attributes or attributes[key] in (None, ""):
        attributes[key] = text


def _messages_attr(role: str, content: Any) -> str | None:
    if content is None:
        return None
    text = str(content)
    if not text:
        return None
    return json.dumps([{"role": role, "content": text}])


def _add_invoke_message_fallbacks(attributes: dict[str, Any]) -> None:
    if attributes.get(GEN_AI_OPERATION_NAME_KEY) != INVOKE_AGENT_OPERATION_NAME:
        return
    if "gen_ai.input.messages" not in attributes:
        _set_string_attr(
            attributes,
            "gen_ai.input.messages",
            _messages_attr("user", attributes.get("input.value")),
        )
    if "gen_ai.output.messages" not in attributes:
        _set_string_attr(
            attributes,
            "gen_ai.output.messages",
            _messages_attr("assistant", attributes.get("output.value")),
        )


def _missing_required_attrs(readable_spans) -> list[dict[str, Any]]:
    missing_by_span = []
    for index, span in enumerate(readable_spans):
        attributes = getattr(span, "attributes", {})
        missing = [key for key in _REQUIRED_A365_ATTRIBUTES if attributes.get(key) in (None, "")]
        if attributes.get(GEN_AI_OPERATION_NAME_KEY) == INVOKE_AGENT_OPERATION_NAME:
            for key in ("gen_ai.input.messages", "gen_ai.output.messages"):
                if attributes.get(key) in (None, ""):
                    missing.append(key)
        if missing:
            missing_by_span.append({
                "index": index,
                "name": getattr(span, "name", None),
                "missing": missing,
            })
    return missing_by_span


class _ReadableSpanAdapter:
    """Adapter that makes OtelSpan compatible with A365's ReadableSpan interface.

    A365's Agent365Exporter expects ReadableSpan objects with specific attributes.
    This adapter wraps OtelSpan and provides the expected interface.

    """

    def __init__(self, otel_span: OtelSpan, tenant_id: str | None, agent_id: str | None):
        """Initialize the adapter.

        Args:
            otel_span: The OtelSpan to adapt
            tenant_id: Fallback tenant ID (used when no per-turn identity is set)
            agent_id: Fallback agent ID (used when no per-turn identity is set)
        """
        self.context = otel_span.get_span_context()

        # Convert parent Span to SpanContext if it exists (A365 expects SpanContext, not Span)
        if otel_span.parent is not None:
            self.parent = otel_span.parent.get_span_context()
        else:
            self.parent = None

        # Per-turn identity wins over static config; falls back when not in a turn.
        turn = get_turn_identity()
        effective_agent_id = turn.agent_app_id if turn is not None else agent_id
        effective_tenant_id = (turn.tenant_id if turn is not None and turn.tenant_id is not None else tenant_id)

        self.attributes = dict(otel_span.attributes)
        _set_string_attr(self.attributes, TENANT_ID_KEY, effective_tenant_id, overwrite=True)
        _set_string_attr(self.attributes, "tenant.id", effective_tenant_id, overwrite=True)
        _set_string_attr(self.attributes, "microsoft.tenant.id", effective_tenant_id, overwrite=True)
        _set_string_attr(self.attributes, GEN_AI_AGENT_ID_KEY, effective_agent_id, overwrite=True)
        self.attributes.setdefault(GEN_AI_OPERATION_NAME_KEY, INVOKE_AGENT_OPERATION_NAME)
        _set_string_attr(
            self.attributes,
            "gen_ai.agent.name",
            os.getenv("A365_AGENT_NAME", _DEFAULT_AGENT_NAME),
        )
        blueprint_id = os.getenv("A365_BLUEPRINT_ID")
        if blueprint_id:
            _set_string_attr(
                self.attributes,
                "microsoft.a365.agent.blueprint.id",
                blueprint_id,
            )
        else:
            _warn_missing_blueprint_id_once()
        _set_string_attr(
            self.attributes,
            "gen_ai.agent.description",
            os.getenv("A365_AGENT_DESCRIPTION"),
        )
        _set_string_attr(self.attributes, "gen_ai.execution.type", "HumanToAgent")

        if turn is not None:
            _set_string_attr(
                self.attributes,
                "gen_ai.conversation.id",
                turn.conversation_id,
            )
            _set_string_attr(
                self.attributes,
                "microsoft.session.id",
                turn.conversation_id,
            )
            _set_string_attr(
                self.attributes,
                "microsoft.channel.name",
                turn.channel_name,
            )
            _set_string_attr(
                self.attributes,
                "user.id",
                turn.user_id or turn.on_behalf_user_id,
            )
            _set_string_attr(self.attributes, "user.name", turn.user_name)
            _set_string_attr(self.attributes, "user.email", turn.user_email)
            _set_string_attr(self.attributes, "client.address", turn.client_address)

        _set_string_attr(
            self.attributes,
            "client.address",
            os.getenv("A365_CLIENT_ADDRESS", "0.0.0.0"),
        )
        _set_string_attr(
            self.attributes,
            "server.address",
            os.getenv("A365_SERVER_ADDRESS") or os.getenv("WEBSITE_HOSTNAME") or socket.getfqdn(),
        )
        _set_string_attr(
            self.attributes,
            "server.port",
            os.getenv("A365_SERVER_PORT", "443"),
        )
        _add_invoke_message_fallbacks(self.attributes)

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


def _convert_otel_span_to_readable(otel_span: OtelSpan, tenant_id: str | None,
                                   agent_id: str | None) -> _ReadableSpanAdapter:
    """Convert an OtelSpan to a ReadableSpan-compatible adapter for A365 exporter.

    A365's Agent365Exporter expects ReadableSpan objects with specific attributes.
    This function creates a compatible adapter object.

    Args:
        otel_span: The OtelSpan to convert
        tenant_id: Fallback tenant ID (used when no per-turn identity is set)
        agent_id: Fallback agent ID (used when no per-turn identity is set)

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
        agent_id: str | None,
        tenant_id: str | None,
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
        token_cache=None,
        auth_ref=None,
        builder=None,
        token_extractor=None,
    ):
        """Initialize the A365 exporter."""
        self._token_extractor = (token_extractor if token_extractor is not None else _get_token_extractor(None))
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
        self._token_cache = token_cache
        self._auth_ref = auth_ref
        self._builder = builder
        # One auth provider per (agent_id, tenant_id) key, lazily resolved.
        self._auth_providers: dict[tuple[str | None, str | None], Any] = {}
        self._auth_locks: dict[tuple[str | None, str | None], asyncio.Lock] = {}
        self._auth_locks_guard = asyncio.Lock()

        # SDK requires token_resolver to be non-None.
        self._a365_exporter = Agent365Exporter(
            token_resolver=token_resolver,
            cluster_category=cluster_category,
            use_s2s_endpoint=use_s2s_endpoint,
        )

        logger.info(f"A365 telemetry exporter initialized for agent_id={agent_id}, "
                    f"tenant_id={tenant_id}, cluster={cluster_category}")

    async def _ensure_token_for(self, agent_id: str, tenant_id: str) -> None:
        """Populate or refresh the cached bearer for ``(agent_id, tenant_id)``.

        Called from ``export_otel_spans`` for the identity stamped on the
        spans being exported. Skips the call when the cached token is still
        valid with a 5-minute buffer.
        """
        if (self._token_cache is None or self._auth_ref is None or self._builder is None):
            return

        key = (agent_id, tenant_id)
        if not self._token_cache.is_expiring_soon(agent_id, tenant_id):
            return

        lock = self._auth_locks.get(key)
        if lock is None:
            async with self._auth_locks_guard:
                lock = self._auth_locks.setdefault(key, asyncio.Lock())

        async with lock:
            if not self._token_cache.is_expiring_soon(agent_id, tenant_id):
                return

            try:
                from nat.builder.context import Context

                auth_provider = self._auth_providers.get(key)
                if auth_provider is None:
                    auth_provider = await self._builder.get_auth_provider(self._auth_ref)
                    self._auth_providers[key] = auth_provider

                user_id = Context.get().user_id
                auth_result = await auth_provider.authenticate(user_id=user_id)
                if not auth_result.credentials:
                    raise A365AuthenticationError("No credentials available from auth provider")

                token = self._token_extractor(auth_result)
                if token is None:
                    _raise_no_bearer_token(auth_result)

                self._token_cache.update_token(
                    agent_id,
                    tenant_id,
                    token=token,
                    expires_at=auth_result.token_expires_at,
                )
                logger.debug(
                    "A365 token resolved for agent=%s tenant=%s (expires_at=%s)",
                    agent_id,
                    tenant_id,
                    auth_result.token_expires_at,
                )
            except Exception:
                logger.error(
                    "Failed to resolve A365 token for agent=%s tenant=%s",
                    agent_id,
                    tenant_id,
                    exc_info=True,
                )
                raise

    async def export_otel_spans(self, spans: list[OtelSpan]) -> None:
        """Export a list of OtelSpans using the A365 exporter."""
        if not spans:
            return

        turn = get_turn_identity()
        effective_agent_id = turn.agent_app_id if turn is not None else self._agent_id
        effective_tenant_id = (turn.tenant_id if turn is not None and turn.tenant_id is not None else self._tenant_id)

        # Without identity, the SDK's partition_by_identity drops every span silently.
        # Surface that explicitly so misconfiguration / missing turn-identity is observable.
        if not effective_agent_id or not effective_tenant_id:
            logger.warning(
                "A365 export skipped for %d span(s): missing agent identity "
                "(turn=%s, fallback agent_id=%r, fallback tenant_id=%r). "
                "Configure A365TelemetryExporter.agent_id / tenant_id, or ensure "
                "the front-end publishes turn identity via set_turn_identity().",
                len(spans),
                turn,
                self._agent_id,
                self._tenant_id,
            )
            return

        await self._ensure_token_for(effective_agent_id, effective_tenant_id)

        try:
            readable_spans = [
                _convert_otel_span_to_readable(
                    otel_span=otel_span,
                    tenant_id=effective_tenant_id,
                    agent_id=effective_agent_id,
                ) for otel_span in spans
            ]

            logger.debug(f"A365 exporter: converted {len(spans)} OtelSpans to ReadableSpan format "
                         f"(tenant={effective_tenant_id}, agent={effective_agent_id})")

            missing_attrs = _missing_required_attrs(readable_spans)
            if missing_attrs:
                logger.warning(
                    "A365 exporter required attribute gaps before SDK export: %s",
                    json.dumps(missing_attrs, sort_keys=True),
                )

            _dump_readable_spans_if_configured(
                readable_spans=readable_spans,
                tenant_id=effective_tenant_id,
                agent_id=effective_agent_id,
                use_s2s_endpoint=self._use_s2s_endpoint,
            )

            loop = asyncio.get_running_loop()
            execution_context = contextvars.copy_context()
            started_at = time.perf_counter()
            logger.info(
                "A365 exporter submitting spans to SDK "
                "(span_count=%s, agent_id=%s, tenant_id=%s, s2s_endpoint=%s)",
                len(readable_spans),
                effective_agent_id,
                effective_tenant_id,
                self._use_s2s_endpoint,
            )
            result = await loop.run_in_executor(
                None,
                execution_context.run,
                self._a365_exporter.export,
                readable_spans,
            )

            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            logger.info(
                "A365 exporter SDK export call completed "
                "(span_count=%s, agent_id=%s, tenant_id=%s, s2s_endpoint=%s, "
                "elapsed_ms=%s, result=%r)",
                len(readable_spans),
                effective_agent_id,
                effective_tenant_id,
                self._use_s2s_endpoint,
                elapsed_ms,
                result,
            )
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(
                f"Error exporting spans to A365 (tenant={effective_tenant_id}, "
                f"agent={effective_agent_id}): {e}",
                exc_info=True,
            )
            if ("authentication" in error_msg or "unauthorized" in error_msg or "token" in error_msg):
                raise A365AuthenticationError(
                    f"Authentication failed while exporting telemetry: {str(e)}",
                    original_error=e,
                ) from e
            raise A365SDKError(
                f"Failed to export spans to A365: {str(e)}",
                sdk_component="Agent365Exporter",
                original_error=e,
            ) from e
