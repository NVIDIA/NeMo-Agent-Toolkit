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

from nat.builder.context import ContextState
from nat.observability.mixin.tagging_config_mixin import PrivacyLevel
from nat.observability.processor.header_redaction_processor import HeaderRedactionProcessor
from nat.observability.processor.span_tagging_processor import SpanTaggingProcessor
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
    - OTLP HTTP protocol for maximum compatibility
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

    Example:
        def should_redact(auth_key: str) -> bool:
            return auth_key in ["sensitive_user", "test_user"]

        exporter = OTLPSpanRedactionAdapterExporter(
            endpoint="https://api.service.com/v1/traces",
            headers={"Authorization": "Bearer your-token"},
            redaction_attributes=["user.email", "request.body"],
            redaction_header="x-user-id",
            redaction_callback=should_redact,
            redaction_value="REDACTED",
            privacy_tag_key="privacy.level",
            privacy_tag_value=PrivacyLevel.HIGH,
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
            redaction_header: str | None = None,
            redaction_callback: Callable[[str], bool] | None = None,
            redaction_enabled: bool = False,
            force_redaction: bool = False,
            redaction_value: str = "[REDACTED]",
            privacy_tag_key: str | None = None,
            privacy_tag_value: PrivacyLevel | None = None,
            # OTLPSpanExporterMixin args
            endpoint: str,
            headers: dict[str, str] | None = None,
            **otlp_kwargs):
        """Initialize the OTLP span exporter with redaction and tagging capabilities.

        Args:
            context_state: The context state for the exporter.
            batch_size: Number of spans to batch before exporting.
            flush_interval: Time in seconds between automatic batch flushes.
            max_queue_size: Maximum number of spans to queue.
            drop_on_overflow: Whether to drop spans when queue is full.
            shutdown_timeout: Maximum time to wait for export completion during shutdown.
            resource_attributes: Additional resource attributes for spans.
            redaction_attributes: List of span attribute keys to redact when conditions are met.
            redaction_header: Header key to check for authentication/user identification.
            redaction_callback: Function to determine if spans should be redacted based on header value.
            redaction_enabled: Whether the redaction processor is enabled.
            force_redaction: If True, always redact regardless of header checks.
            redaction_value: Value to replace redacted attributes with.
            privacy_tag_key: Key name for the privacy level tag to add to spans.
            privacy_tag_value: Privacy level value to assign to spans.
            endpoint: The endpoint for the OTLP service.
            headers: The headers for the OTLP service.
            **otlp_kwargs: Additional keyword arguments for the OTLP service.
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
                         **otlp_kwargs)

        # Insert redaction and tagging processors to the front of the processing pipeline
        self.add_processor(HeaderRedactionProcessor(attributes=redaction_attributes,
                                                    header=redaction_header,
                                                    callback=redaction_callback,
                                                    enabled=redaction_enabled,
                                                    force_redact=force_redaction,
                                                    redaction_value=redaction_value),
                           name="header_redaction",
                           position=0)

        self.add_processor(SpanTaggingProcessor(tag_key=privacy_tag_key,
                                                tag_value=privacy_tag_value.value if privacy_tag_value else None),
                           name="span_privacy_tagging",
                           position=1)
