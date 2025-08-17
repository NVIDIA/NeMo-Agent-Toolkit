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
from nat.plugins.data_flywheel.observability.processor.trace_conversion.trace_adapter_registry import \
    TraceAdapterRegistry
from nat.plugins.data_flywheel.observability.schema.trace_container import TraceContainer
from nat.utils.type_converter import GlobalTypeConverter

logger = logging.getLogger(__name__)


def _get_string_value(value: Any) -> str:
    """Extract string value from enum or literal type safely.

    Args:
        value (Any): Could be an Enum, string, or other type

    Returns:
        str: String representation of the value
    """
    if isinstance(value, Enum):
        return value.value
    return str(value)


def get_trace_source(span: Span, client_id: str) -> TraceContainer:
    """Get the source of a span.

    Args:
        span (Span): The span to get the source of
        client_id (str): The client ID to use for the DFW record

    Returns:
        TraceContainer: The source of the span
    """
    # Extract framework (provider will be detected by union schema matching)
    framework = _get_string_value(span.attributes.get("nat.framework", "langchain"))

    # Create source dictionary WITHOUT provider - let union detect it via schema matching
    source_dict = {
        "source": {
            "framework": framework,  # Don't include provider - let union schemas set their default values
            "input_value": span.attributes.get("input.value", None),
            "metadata": span.attributes.get("nat.metadata", None),
            "client_id": client_id,
        },
        "span": span
    }

    try:
        # Create the TraceContainer - union will pick the right type based on schema
        trace_container = TraceContainer(**source_dict)
        logger.debug("Union selected schema: %s with provider: %s",
                     type(trace_container.source),
                     getattr(trace_container.source, 'provider', 'unknown'))

        # Extract the provider that was detected by the union
        detected_provider = getattr(trace_container.source, 'provider', 'unknown')

        # Convert enum to string if needed
        detected_provider = _get_string_value(detected_provider)

        # Now convert to dynamic type - must have registered adapter
        return TraceAdapterRegistry.create_dynamic_instance(trace_container, framework, detected_provider)

    except Exception as e:
        # Schema detection failed - this indicates missing schema registration or malformed data
        available_schemas = list(TraceAdapterRegistry._registered_models.keys())
        schema_names = [schema.__name__ for schema in available_schemas]

        raise ValueError(f"Schema-based detection failed for framework '{framework}'. "
                         f"Data structure doesn't match any registered trace source schemas. "
                         f"Available schemas: {schema_names}. "
                         f"Ensure proper schema is registered with @register_adapter() for this data format. "
                         f"Original error: {e}") from e


def span_to_dfw_record(span: Span, to_type: type[BaseModel], client_id: str) -> BaseModel | None:
    """Convert a span to DFW record using registered adapters.

    Args:
        span (Span): The span to convert
        to_type (type[BaseModel]): The type of the DFW record to convert to
        client_id (str): The client ID to use for the DFW record

    Returns:
        BaseModel | None: The converted DFW record
    """
    trace_source = get_trace_source(span, client_id)
    return GlobalTypeConverter.convert(trace_source, to_type=to_type)
