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

from nat.builder.context import ContextState
from nat.plugins.opentelemetry.otel_span import OtelSpan
from nat.plugins.opentelemetry.otel_span_exporter import OtelSpanExporter
from opentelemetry.sdk.trace import Event as OtelEvent
from opentelemetry.trace import Link as OtelLink

logger = logging.getLogger(__name__)


class _ReadableSpanAdapter:
    """Adapter that makes OtelSpan compatible with A365's ReadableSpan interface.

    A365's _Agent365Exporter expects ReadableSpan objects with specific attributes.
    This adapter wraps OtelSpan and provides the expected interface.

    """

    def __init__(self, otel_span: OtelSpan, tenant_id: str, agent_id: str):
        """Initialize the adapter.

        Args:
            otel_span: The OtelSpan to adapt
            tenant_id: The tenant ID to add as an attribute
            agent_id: The agent ID to add as an attribute
        """
        # Get span context
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

        # Convert events to OpenTelemetry SDK Event objects
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
                # Event is already an Event object
                otel_event = event
            self.events.append(otel_event)

        # Convert links to OpenTelemetry SDK Link objects
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
                # Link is already a Link object
                self.links.append(link)

        # Copy other required attributes
        self.name = otel_span.name
        self.kind = otel_span.kind
        self.start_time = otel_span.start_time
        self.end_time = otel_span.end_time or otel_span.start_time  # Ensure end_time is set
        self.status = otel_span.status
        self.instrumentation_scope = otel_span.instrumentation_scope
        self.resource = otel_span.resource


def _convert_otel_span_to_readable(otel_span: OtelSpan, tenant_id: str, agent_id: str) -> _ReadableSpanAdapter:
    """Convert an OtelSpan to a ReadableSpan-compatible adapter for A365 exporter.

    A365's _Agent365Exporter expects ReadableSpan objects with specific attributes.
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

    Stub implementation for plugin registration testing.
    Full implementation will integrate A365's _Agent365Exporter with NAT's telemetry system.

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
        token_resolver: Callable[[str, str], str | None],
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
    ):
        """Initialize the A365 exporter stub."""
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

        logger.info(
            f"A365 telemetry exporter stub initialized for agent_id={agent_id}, "
            f"tenant_id={tenant_id}, cluster={cluster_category}"
        )

    async def export_otel_spans(self, spans: list[OtelSpan]) -> None:
        """Export a list of OtelSpans using the A365 exporter.

        Converts OtelSpans to ReadableSpan format and exports via A365's _Agent365Exporter.

        Args:
            spans (list[OtelSpan]): The list of spans to export.
        """
        if not spans:
            return

        # Convert OtelSpans to ReadableSpan-like objects
        readable_spans = []
        for otel_span in spans:
            readable_span = _convert_otel_span_to_readable(
                otel_span=otel_span,
                tenant_id=self._tenant_id,
                agent_id=self._agent_id,
            )
            readable_spans.append(readable_span)

        # TODO: Integrate with A365's _Agent365Exporter
        # This requires:
        # 1. Import _Agent365Exporter from microsoft-agents-a365-observability-core
        # 2. Create instance with token_resolver
        # 3. Call exporter.export(readable_spans)
        logger.debug(
            f"A365 exporter: converted {len(spans)} OtelSpans to ReadableSpan format "
            f"(tenant={self._tenant_id}, agent={self._agent_id})"
        )

        # Stub: log what would be exported
        for readable_span in readable_spans:
            logger.debug(
                f"Would export span: name={readable_span.name}, "
                f"trace_id={readable_span.context.trace_id}, "
                f"span_id={readable_span.context.span_id}"
            )
