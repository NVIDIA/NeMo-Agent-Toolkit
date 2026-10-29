# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Attach NAT's callback handler when a workflow uses CrewAI.

Registered from this package so nvidia-nat-core needs no CrewAI import.
"""

import logging

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.framework_instrumentation import _library_instrumented
from nat.builder.framework_instrumentation import register_framework_instrumentation

logger = logging.getLogger(__name__)


@register_framework_instrumentation(LLMFrameworkEnum.CREWAI)
def _instrument_crewai(workflow_llms: dict | None) -> None:
    if _library_instrumented["crewai"]:
        return
    from nat.plugins.crewai.crewai_callback_handler import CrewAIProfilerHandler
    CrewAIProfilerHandler().instrument()
    _library_instrumented["crewai"] = True
    logger.debug("CrewAI callback handler registered")
