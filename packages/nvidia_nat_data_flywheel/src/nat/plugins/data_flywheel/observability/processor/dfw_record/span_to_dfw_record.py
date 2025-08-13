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
from enum import Enum

from nat.data_models.span import Span
from nat.plugins.data_flywheel.observability.processor.dfw_record.langchain import convert_langchain_nim
from nat.plugins.data_flywheel.observability.processor.dfw_record.langchain import convert_langchain_openai
from nat.plugins.data_flywheel.observability.schema.dfw_record import DFWRecord
from nat.plugins.data_flywheel.observability.schema.trace_source import TraceSource

logger = logging.getLogger(__name__)


class TraceProviderFramework(Enum):
    LANGCHAIN_NIM = "langchain_nim"
    LANGCHAIN_OPENAI = "langchain_openai"


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

    trace_source = get_trace_source(span)
    try:
        trace_provider_framework = TraceProviderFramework(trace_source.source.framework + "_" +
                                                          trace_source.source.provider)
    except ValueError:
        logger.warning("Unsupported trace provider framework: `%s`",
                       trace_source.source.framework + "_" + trace_source.source.provider)
        return None

    match trace_provider_framework:
        case TraceProviderFramework.LANGCHAIN_NIM:
            return convert_langchain_nim(trace_source, client_id)
        case TraceProviderFramework.LANGCHAIN_OPENAI:
            return convert_langchain_openai(trace_source, client_id)
        case _:
            logger.warning("Unsupported trace provider framework: `%s`", trace_provider_framework)
            return None
