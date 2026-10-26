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
"""Attach NAT's callback handler when a workflow uses LlamaIndex. Registered here so nvidia-nat-core needs no LlamaIndex import."""

import logging

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.framework_instrumentation import register_framework_instrumentation

logger = logging.getLogger(__name__)


def _instrument_llama_index(workflow_llms: dict | None) -> None:
    from llama_index.core import Settings
    from llama_index.core.callbacks import CallbackManager

    from nat.plugins.llama_index.callback_handler import LlamaIndexProfilerHandler

    Settings.callback_manager = CallbackManager([LlamaIndexProfilerHandler()])
    logger.debug("LlamaIndex callback handler registered")


register_framework_instrumentation(LLMFrameworkEnum.LLAMA_INDEX, _instrument_llama_index)
