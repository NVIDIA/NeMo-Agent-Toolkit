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
from enum import Enum
from typing import Any

from pydantic import BaseModel

from nat.data_models.span import Span
from nat.plugins.data_flywheel.observability.processor.trace_conversion.trace_adapter_registry import (
    TraceAdapterRegistry,  # noqa: F401
)
from nat.plugins.data_flywheel.observability.schema.trace_container import TraceContainer

logger = logging.getLogger(__name__)


def _get_string_value(value: Any) -> str:
    """Extract string value from enum or literal type safely.

    Args:
        value (Any): Could be an Enum, string, or other type

    Returns:
        str: String representation of the value
    """
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def get_trace_container(span: Span, client_id: str) -> TraceContainer:
    """Create a TraceContainer from a span for schema detection and conversion.

    Extracts trace data from span attributes and creates a TraceContainer where Pydantic's
    discriminated union will automatically detect the correct trace source schema type.

    Args:
        span (Span): The span containing trace attributes to extract
        client_id (str): The client ID to include in the trace source data

    Returns:
        TraceContainer: Container with automatically detected source type and original span

    Raises:
        ValueError: If span data doesn't match any registered trace source schemas
    """
    # Extract framework name from span attributes
    framework = _get_string_value(span.attributes.get("nat.framework", "langchain"))

    # Create trace source data - Pydantic union will detect correct schema type automatically
    source_dict = {
        "source": {
            "framework": framework,
            "input_value": span.attributes.get("input.value", None),
            "metadata": span.attributes.get("nat.metadata", None),
            "client_id": client_id,
        },
        "span": span
    }

    try:
        # Create TraceContainer - Pydantic discriminated union automatically detects source type
        trace_container = TraceContainer(**source_dict)
        logger.debug("Pydantic union detected source type: %s for framework: %s",
                     type(trace_container.source).__name__,
                     framework)
        return trace_container

    except Exception as e:
        # Schema detection failed - indicates missing adapter registration or malformed span data
        registry_data = TraceAdapterRegistry.list_registered_types()
        adapter_metadata = []
        for source_type, target_converters in registry_data.items():
            for target_type in target_converters.keys():
                target_name = getattr(target_type, '__name__', str(target_type))
                adapter_metadata.append(f"{source_type.__name__} -> {target_name}")

        raise ValueError(f"Trace source schema detection failed for framework '{framework}'. "
                         f"Span data structure doesn't match any registered trace source schemas. "
                         f"Available registered adapters: {adapter_metadata}. "
                         f"Ensure a schema is registered with @register_adapter() for this trace format. "
                         f"Original error: {e}") from e


def span_to_dfw_record(span: Span, to_type: type[BaseModel], client_id: str) -> BaseModel:
    """Convert a span to Data Flywheel record using registered trace adapters.

    Creates a TraceContainer from the span, automatically detects the trace source type
    via Pydantic schema matching, then uses the registered converter to transform it
    to the specified target type.

    Args:
        span (Span): The span containing trace data to convert.
        to_type (type[BaseModel]): Target Pydantic model type for the conversion.
        client_id (str): Client identifier to include in the trace data.

    Returns:
        BaseModel: Converted record of the specified type.

    Raises:
        ValueError: If no converter is registered for the detected source type -> target type,
            or if the conversion fails.
    """
    trace_container = get_trace_container(span, client_id)
    return TraceAdapterRegistry.convert(trace_container, to_type=to_type)
