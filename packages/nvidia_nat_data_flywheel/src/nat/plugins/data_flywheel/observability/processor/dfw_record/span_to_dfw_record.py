# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nat.data_models.span import Span
from nat.plugins.data_flywheel.observability.processor.dfw_record.trace_adapter_registry import TraceAdapterRegistry
from nat.plugins.data_flywheel.observability.schema.dfw_record import DFWRecord
from nat.plugins.data_flywheel.observability.schema.trace_source import TraceSource

logger = logging.getLogger(__name__)


def get_trace_source(span: Span) -> TraceSource:
    """Get the source of a span."""
    source_dict = {
        "source": {
            "framework": span.attributes.get("nat.framework", "unknown"),
            "input_value": span.attributes.get("input.value", None),
            "metadata": span.attributes.get("nat.metadata", None),
        },
        "span": span
    }
    return TraceSource(**source_dict)


def span_to_dfw_record(span: Span, client_id: str = "nat_client") -> DFWRecord | None:
    """Convert a span to DFW record using registered adapters."""
    trace_source = get_trace_source(span)

    adapter = TraceAdapterRegistry.get_adapter(trace_source)
    if adapter is None:
        framework_provider = f"{trace_source.source.framework}_{trace_source.source.provider}"
        logger.warning("No adapter found for framework: %s. Supported frameworks: %s",
                       framework_provider,
                       TraceAdapterRegistry.list_supported_frameworks())
        return None

    try:
        return adapter.convert(trace_source, client_id)
    except (ValueError, TypeError) as e:
        logger.error("Invalid input for adapter `%s`: %s", adapter.framework_identifier, str(e))
        return None
    except Exception as e:
        logger.error("Unexpected error in adapter `%s`: %s", adapter.framework_identifier, str(e))
        return None
