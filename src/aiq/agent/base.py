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
from abc import ABC
from abc import abstractmethod
from enum import Enum

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.graph import CompiledGraph

log = logging.getLogger(__name__)


class AgentDecision(Enum):
    TOOL = "tool"
    END = "finished"


class BaseAgent(ABC):

    def __init__(self,
                 llm: BaseChatModel,
                 tools: list[BaseTool],
                 callbacks: list[AsyncCallbackHandler] = None,
                 detailed_logs: bool = False):
        log.debug("Initializing Agent Graph")
        self.llm = llm
        self.tools = tools
        self.callbacks = callbacks or []
        self.detailed_logs = detailed_logs
        self.graph = None

    @abstractmethod
    async def _build_graph(self, state_schema) -> CompiledGraph:
        pass
