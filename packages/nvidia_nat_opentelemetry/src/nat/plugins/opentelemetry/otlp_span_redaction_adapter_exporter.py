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

import logging
from collections.abc import Callable
from collections.abc import Mapping
from enum import Enum
from typing import Any

from nat.builder.context import ContextState
from nat.observability.processor.redaction import SpanHeaderRedactionProcessor
from nat.observability.processor.span_tagging_processor import SpanTaggingProcessor
from nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin import OTLPProtocol
from nat.plugins.opentelemetry.otlp_span_adapter_exporter import OTLPSpanAdapterExporter

logger = logging.getLogger(__name__)


class OTLPSpanHeaderRedactionAdapterExporter(OTLPSpanAdapterExporter):
    """An OpenTelemetry OTLP span exporter with built-in redaction and privacy tagging.

    This class extends OTLPSpanAdapterExporter to provide automatic span redaction
    and privacy tagging capabilities. It automatically adds header-based redaction
    and span tagging processors to the processing pipeline.

    Key Features:
    - Header-based span redaction with configurable callback logic
    - Privacy level tagging for compliance and governance
    - Complete span processing pipeline (IntermediateStep → Span → Redaction → Tagging → OtelSpan → Batching → Export)
    - Batching support for efficient transmission
    - OTLP HTTP and gRPC protocol for maximum compatibility
    - Configurable authentication via headers
    - Resource attribute management
    - Error handling and retry logic

    The redaction processor allows conditional redaction based on authentication headers,
    while the tagging processor adds privacy-level metadata to spans for downstream
    processing and compliance tracking.

    This exporter is commonly used with services like:
    - OpenTelemetry Collector
    - DataDog (OTLP endpoint)
    - Jaeger (OTLP endpoint)
    - Grafana Tempo
    - Custom OTLP-compatible backends

    Example::

        def should_redact(auth_key: str) -> bool:
            return auth_key in ["sensitive_user", "test_user"]

        exporter = OTLPSpanRedactionAdapterExporter(
            endpoint="https://api.service.com/v1/traces",
            headers={"Authorization": "Bearer your-token"},
            protocol='http',
            redaction_attributes=["user.email", "request.body"],
            redaction_headers=["x-user-id"],
            redaction_callback=should_redact,
            redaction_value="REDACTED",
            tags={"privacy.level": PrivacyLevel.HIGH, "service.type": "sensitive"},
            batch_size=50,
            flush_interval=10.0
        )
    """

    def __init__(
            self,
            *,
            # OtelSpanExporter args
            context_state: ContextState | None = None,
            batch_size: int = 100,
            flush_interval: float = 5.0,
            max_queue_size: int = 1000,
            drop_on_overflow: bool = False,
            shutdown_timeout: float = 10.0,
            resource_attributes: dict[str, str] | None = None,
            # Redaction args
            redaction_attributes: list[str] | None = None,
            redaction_headers: list[str] | None = None,
            redaction_callback: Callable[..., Any] | None = None,
            redaction_enabled: bool = False,
            force_redaction: bool = False,
            redaction_value: str = "[REDACTED]",
            redaction_tag: str | None = None,
            tags: Mapping[str, Enum | str] | None = None,
            # OTLPSpanExporterMixin args
            endpoint: str,
            headers: dict[str, str] | None = None,
            protocol: OTLPProtocol = 'http',
            **otlp_kwargs):
        """Initialize the OTLP span exporter with redaction and tagging capabilities.

        Args:
            context_state: The context state for the exporter.
            batch_size: Number of spans to batch before exporting, default is 100.
            flush_interval: Time in seconds between automatic batch flushes, default is 5.0.
            max_queue_size: Maximum number of spans to queue, default is 1000.
            drop_on_overflow: Whether to drop spans when queue is full, default is False.
            shutdown_timeout: Maximum time to wait for export completion during shutdown, default is 10.0.
            resource_attributes: Additional resource attributes for spans.
            redaction_attributes: List of span attribute keys to redact when conditions are met.
            redaction_headers: List of header keys to check for authentication/user identification.
            redaction_callback: Function that returns true to redact spans based on header value, false otherwise.
            redaction_enabled: Whether the redaction processor is enabled, default is False.
            force_redaction: If True, always redact regardless of header checks, default is False.
            redaction_value: Value to replace redacted attributes with, default is "[REDACTED]".
            tags: Mapping of tag keys to their values (enums or strings) to add to spans.
            redaction_tag: Tag to add to spans when redaction occurs.
            endpoint: The endpoint for the OTLP service.
            headers: The headers for the OTLP service.
            protocol: The protocol to use for the OTLP service, default is 'http'.
            otlp_kwargs: Additional keyword arguments for the OTLP service.
        """
        super().__init__(context_state=context_state,
                         batch_size=batch_size,
                         flush_interval=flush_interval,
                         max_queue_size=max_queue_size,
                         drop_on_overflow=drop_on_overflow,
                         shutdown_timeout=shutdown_timeout,
                         resource_attributes=resource_attributes,
                         endpoint=endpoint,
                         headers=headers,
                         protocol=protocol,
                         **otlp_kwargs)

        # Insert redaction and tagging processors to the front of the processing pipeline
        self.add_processor(SpanHeaderRedactionProcessor(attributes=redaction_attributes or [],
                                                        headers=redaction_headers or [],
                                                        callback=redaction_callback or (lambda _: False),
                                                        enabled=redaction_enabled,
                                                        force_redact=force_redaction,
                                                        redaction_value=redaction_value,
                                                        redaction_tag=redaction_tag),
                           name="header_redaction",
                           position=0)

        self.add_processor(SpanTaggingProcessor(tags=tags), name="span_sensitivity_tagging", position=1)
