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

from pydantic import BaseModel

from nat.data_models.span import Span
from nat.plugins.data_flywheel.observability.processor.dfw_record.trace_adapter_registry import TraceAdapterRegistry
from nat.plugins.data_flywheel.observability.schema.trace_source import TraceSource

logger = logging.getLogger(__name__)


def get_trace_source(span: Span) -> TraceSource:
    """Get the source of a span.

    Args:
        span (Span): The span to get the source of

    Returns:
        TraceSource: The source of the span
    """
    source_dict = {
        "source": {
            "framework": span.attributes.get("nat.framework", "unknown"),
            "input_value": span.attributes.get("input.value", None),
            "metadata": span.attributes.get("nat.metadata", None),
        },
        "span": span
    }
    return TraceSource(**source_dict)


def span_to_dfw_record(span: Span, to_type: type[BaseModel], client_id: str = "nat_client") -> BaseModel | None:
    """Convert a span to DFW record using registered adapters.

    Args:
        span (Span): The span to convert
        to_type (type[BaseModel]): The type of the DFW record to convert to
        client_id (str): The client ID to use for the DFW record

    Returns:
        BaseModel | None: The converted DFW record
    """
    trace_source = get_trace_source(span)
    adapter = TraceAdapterRegistry.get_adapter(trace_source, to_type)
    if adapter is None:
        framework_provider = f"{trace_source.source.framework.value}_{trace_source.source.provider}"
        logger.warning("No adapter found for framework: '%s'. Supported frameworks: '%s'",
                       framework_provider,
                       TraceAdapterRegistry.list_supported_frameworks())
        return None

    try:
        return adapter.convert(trace_source, client_id)
    except (ValueError, TypeError) as e:
        logger.error("Invalid input for adapter '%s': '%s'", adapter.framework_identifier, str(e))
        return None
    except Exception as e:
        logger.error("Unexpected error in adapter '%s': '%s'", adapter.framework_identifier, str(e))
        return None
