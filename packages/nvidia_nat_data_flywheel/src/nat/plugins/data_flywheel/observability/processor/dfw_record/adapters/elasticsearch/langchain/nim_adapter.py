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

from nat.plugins.data_flywheel.observability.processor.dfw_record.adapters import TraceSourceAdapter
from nat.plugins.data_flywheel.observability.processor.dfw_record.adapters.elasticsearch.langchain import \
    convert_langchain_nim
from nat.plugins.data_flywheel.observability.processor.dfw_record.trace_adapter_registry import register_adapter
from nat.plugins.data_flywheel.observability.schema.dfw_es_record import DFWESRecord
from nat.plugins.data_flywheel.observability.schema.trace_source import TraceSource

logger = logging.getLogger(__name__)


@register_adapter
class ESLangChainNimAdapter(TraceSourceAdapter[DFWESRecord]):
    """Adapter for LangChain NIM trace sources."""

    def can_handle(self, trace_source: TraceSource) -> bool:
        framework_provider = f"{trace_source.source.framework}_{trace_source.source.provider}"
        return framework_provider == "langchain_nim"

    def convert(self, trace_source: TraceSource, client_id: str) -> DFWESRecord | None:
        return convert_langchain_nim(trace_source, client_id)

    @property
    def framework_identifier(self) -> str:
        return "langchain_nim"

    @property
    def name(self) -> str:
        return f"elasticsearch_{self.framework_identifier}_adapter"
