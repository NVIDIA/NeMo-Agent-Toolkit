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


class ComponentEnum(StrEnum):
    # Keep sorted!!!
    AUTHENTICATION_PROVIDER = "auth_provider"
    EMBEDDER_CLIENT = "embedder_client"
    EMBEDDER_PROVIDER = "embedder_provider"
    EVALUATOR = "evaluator"
    FRONT_END = "front_end"
    FUNCTION = "function"
    FUNCTION_GROUP = "function_group"
    MIDDLEWARE = "middleware"
    TTC_STRATEGY = "ttc_strategy"
    LLM_CLIENT = "llm_client"
    LLM_PROVIDER = "llm_provider"
    LOGGING = "logging"
    MEMORY = "memory"
    OBJECT_STORE = "object_store"
    PACKAGE = "package"
    REGISTRY_HANDLER = "registry_handler"
    RETRIEVER_CLIENT = "retriever_client"
    RETRIEVER_PROVIDER = "retriever_provider"
    TOOL_WRAPPER = "tool_wrapper"
    TRACING = "tracing"
    UNDEFINED = "undefined"


class ComponentGroup(StrEnum):
    # Keep sorted!!!
    AUTHENTICATION = "authentication"
    EMBEDDERS = "embedders"
    FUNCTIONS = "functions"
    FUNCTION_GROUPS = "function_groups"
    MIDDLEWARE = "middleware"
    TTC_STRATEGIES = "ttc_strategies"
    LLMS = "llms"
    MEMORY = "memory"
    OBJECT_STORES = "object_stores"
    RETRIEVERS = "retrievers"


# Compatibility aliases with previous releases
AIQComponentEnum = ComponentEnum
