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
from enum import StrEnum

logger = logging.getLogger(__name__)


class AIQComponentEnum(StrEnum):
    FRONT_END = "front_end"
    FUNCTION = "function"
    TOOL_WRAPPER = "tool_wrapper"
    LLM_PROVIDER = "llm_provider"
    LLM_CLIENT = "llm_client"
    EMBEDDER_PROVIDER = "embedder_provider"
    EMBEDDER_CLIENT = "embedder_client"
    EVALUATOR = "evaluator"
    MEMORY = "memory"
    RETRIEVER_PROVIDER = "retriever_provider"
    RETRIEVER_CLIENT = "retriever_client"
    REGISTRY_HANDLER = "registry_handler"
    LOGGING = "logging"
    TRACING = "tracing"
    PACKAGE = "package"
    OBJECT_STORE = "object_store"
    ITS_STRATEGY = "its_strategy"
    UNDEFINED = "undefined"


class ComponentGroup(StrEnum):
    EMBEDDERS = "embedders"
    FUNCTIONS = "functions"
    LLMS = "llms"
    MEMORY = "memory"
    OBJECT_STORES = "object_stores"
    RETRIEVERS = "retrievers"
    ITS_STRATEGIES = "its_strategies"
