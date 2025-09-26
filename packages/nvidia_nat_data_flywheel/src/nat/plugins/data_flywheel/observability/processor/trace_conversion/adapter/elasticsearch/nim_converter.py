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

# pylint: disable=line-too-long
# flake8: noqa
from nat.plugins.data_flywheel.observability.processor.trace_conversion.adapter.elasticsearch.openai_converter import \
    convert_langchain_openai
from nat.plugins.data_flywheel.observability.processor.trace_conversion.trace_adapter_registry import \
    register_adapter
from nat.plugins.data_flywheel.observability.schema.provider.nim_trace_source import \
    NIMTraceSource
from nat.plugins.data_flywheel.observability.schema.sink.elasticsearch.dfw_es_record import \
    DFWESRecord
from nat.plugins.data_flywheel.observability.schema.trace_container import \
    TraceContainer

logger = logging.getLogger(__name__)


@register_adapter(trace_source_model=NIMTraceSource)
def convert_langchain_nim(trace_source: TraceContainer) -> DFWESRecord:
    """Convert a LangChain/LangGraph Nim trace source to a DFWESRecord.

    Args:
        trace_source (TraceContainer): The trace source to convert

    Returns:
        DFWESRecord: The converted DFW record
    """
    return convert_langchain_openai(trace_source)
