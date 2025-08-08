# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

from nat.builder.context import ContextState
from nat.builder.embedder import EmbedderProviderInfo
from nat.builder.function import Function
from nat.builder.function_base import FunctionBase
from nat.builder.function_base import InputT
from nat.builder.function_base import SingleOutputT
from nat.builder.function_base import StreamingOutputT
from nat.builder.llm import LLMProviderInfo
from nat.builder.retriever import RetrieverProviderInfo
from nat.data_models.config import Config
from nat.experimental.test_time_compute.models.strategy_base import StrategyBase
from nat.memory.interfaces import MemoryEditor
from nat.object_store.interfaces import ObjectStore
from nat.observability.exporter.base_exporter import BaseExporter
from nat.observability.exporter_manager import ExporterManager
from nat.runtime.runner import Runner

callback_handler_var: ContextVar[Any | None] = ContextVar("callback_handler_var", default=None)


class Workflow(FunctionBase[InputT, StreamingOutputT, SingleOutputT]):

    def __init__(self,
                 *,
                 config: Config,
                 entry_fn: Function[InputT, StreamingOutputT, SingleOutputT],
                 functions: dict[str, Function] | None = None,
                 llms: dict[str, LLMProviderInfo] | None = None,
                 embeddings: dict[str, EmbedderProviderInfo] | None = None,
                 memory: dict[str, MemoryEditor] | None = None,
                 object_stores: dict[str, ObjectStore] | None = None,
                 telemetry_exporters: dict[str, BaseExporter] | None = None,
                 retrievers: dict[str | None, RetrieverProviderInfo] | None = None,
                 ttc_strategies: dict[str, StrategyBase] | None = None,
                 context_state: ContextState):

        super().__init__(input_schema=entry_fn.input_schema,
                         streaming_output_schema=entry_fn.streaming_output_schema,
                         single_output_schema=entry_fn.single_output_schema)

        self.config = config
        self.functions = functions or {}
        self.llms = llms or {}
        self.embeddings = embeddings or {}
        self.memory = memory or {}
        self.telemetry_exporters = telemetry_exporters or {}
        self.object_stores = object_stores or {}
        self.retrievers = retrievers or {}

        self._exporter_manager = ExporterManager.from_exporters(self.telemetry_exporters)
        self.ttc_strategies = ttc_strategies or {}

        self._entry_fn = entry_fn

        self._context_state = context_state

    @property
    def has_streaming_output(self) -> bool:

        return self._entry_fn.has_streaming_output

    @property
    def has_single_output(self) -> bool:

        return self._entry_fn.has_single_output

    @asynccontextmanager
    async def run(self, message: InputT):
        """
        Called each time we start a new workflow run. We'll create
        a new top-level workflow span here.
        """

        async with Runner(input_message=message,
                          entry_fn=self._entry_fn,
                          context_state=self._context_state,
                          exporter_manager=self._exporter_manager.get()) as runner:

            # The caller can `yield runner` so they can do `runner.result()` or `runner.result_stream()`
            yield runner

    async def result_with_steps(self, message: InputT, to_type: type | None = None):

        async with self.run(message) as runner:

            from nat.eval.runtime_event_subscriber import pull_intermediate

            # Start the intermediate stream
            pull_done, intermediate_steps = pull_intermediate()

            # Wait on the result
            result = await runner.result(to_type=to_type)

            await pull_done.wait()

            return result, intermediate_steps

    @staticmethod
    def from_entry_fn(*,
                      config: Config,
                      entry_fn: Function[InputT, StreamingOutputT, SingleOutputT],
                      functions: dict[str, Function] | None = None,
                      llms: dict[str, LLMProviderInfo] | None = None,
                      embeddings: dict[str, EmbedderProviderInfo] | None = None,
                      memory: dict[str, MemoryEditor] | None = None,
                      object_stores: dict[str, ObjectStore] | None = None,
                      telemetry_exporters: dict[str, BaseExporter] | None = None,
                      retrievers: dict[str | None, RetrieverProviderInfo] | None = None,
                      ttc_strategies: dict[str, StrategyBase] | None = None,
                      context_state: ContextState) -> 'Workflow[InputT, StreamingOutputT, SingleOutputT]':

        input_type: type = entry_fn.input_type
        streaming_output_type = entry_fn.streaming_output_type
        single_output_type = entry_fn.single_output_type

        class WorkflowImpl(Workflow[input_type, streaming_output_type, single_output_type]):
            pass

        return WorkflowImpl(config=config,
                            entry_fn=entry_fn,
                            functions=functions,
                            llms=llms,
                            embeddings=embeddings,
                            memory=memory,
                            object_stores=object_stores,
                            telemetry_exporters=telemetry_exporters,
                            retrievers=retrievers,
                            ttc_strategies=ttc_strategies,
                            context_state=context_state)
