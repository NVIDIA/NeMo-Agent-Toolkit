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

from aiq.builder.context import AIQContextState
from aiq.plugins.opentelemetry.mixin.otlp_span_exporter_mixin import OTLPSpanExporterMixin
from aiq.plugins.opentelemetry.otel_span_exporter import OtelSpanExporter

logger = logging.getLogger(__name__)


class OTLPSpanAdapterExporter(OTLPSpanExporterMixin, OtelSpanExporter):
    """A opentelemetry OTLP span exporter that exports telemetry traces to externally hosted OTLP service."""

    def __init__(self, context_state: AIQContextState | None = None, **otlp_kwargs):
        super().__init__(context_state=context_state, **otlp_kwargs)
