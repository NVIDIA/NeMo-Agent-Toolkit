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

from nat.builder.context import ContextState
from nat.plugins.opentelemetry.otel_span_exporter import OtelSpanExporter
from nat.plugins.phoenix.mixin.phoenix_mixin import PhoenixMixin

logger = logging.getLogger(__name__)


class PhoenixOtelExporter(PhoenixMixin, OtelSpanExporter):  # pylint: disable=R0901
    """Phoenix exporter for AI workflow observability.

    Exports OpenTelemetry-compatible traces to Phoenix for visualization
    and analysis of AI agent behavior and performance.

    Features:
    - Automatic span conversion from NAT events
    - Phoenix-specific resource tagging
    - Project-based trace organization

    Args:
        context_state: Execution context for isolation
        endpoint: Phoenix server endpoint
        project: Project name for trace grouping
        batch_size: Batch size for exporting
        flush_interval: Flush interval for exporting
        max_queue_size: Maximum queue size for exporting
        drop_on_overflow: Drop on overflow for exporting
        shutdown_timeout: Shutdown timeout for exporting
    """

    def __init__(self,
                 context_state: ContextState | None = None,
                 batch_size: int = 100,
                 flush_interval: float = 5.0,
                 max_queue_size: int = 1000,
                 drop_on_overflow: bool = False,
                 shutdown_timeout: float = 10.0,
                 **phoenix_kwargs):
        super().__init__(context_state=context_state,
                         batch_size=batch_size,
                         flush_interval=flush_interval,
                         max_queue_size=max_queue_size,
                         drop_on_overflow=drop_on_overflow,
                         shutdown_timeout=shutdown_timeout,
                         **phoenix_kwargs)
