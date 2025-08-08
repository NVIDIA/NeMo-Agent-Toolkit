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

import asyncio
import typing
from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
from pathlib import Path

from aiq.authentication.interfaces import AuthProviderBase
from aiq.builder.context import AIQContext
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function import Function
from aiq.data_models.authentication import AuthProviderBaseConfig
from aiq.data_models.component_ref import AuthenticationRef
from aiq.data_models.component_ref import EmbedderRef
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.component_ref import MemoryRef
from aiq.data_models.component_ref import ObjectStoreRef
from aiq.data_models.component_ref import RetrieverRef
from aiq.data_models.component_ref import TTCStrategyRef
from aiq.data_models.embedder import EmbedderBaseConfig
from aiq.data_models.evaluator import EvaluatorBaseConfig
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.function_dependencies import FunctionDependencies
from aiq.data_models.llm import LLMBaseConfig
from aiq.data_models.memory import MemoryBaseConfig
from aiq.data_models.object_store import ObjectStoreBaseConfig
from aiq.data_models.retriever import RetrieverBaseConfig
from aiq.data_models.ttc_strategy import TTCStrategyBaseConfig
from aiq.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from aiq.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from aiq.memory.interfaces import MemoryEditor
from aiq.object_store.interfaces import ObjectStore
from aiq.retriever.interface import AIQRetriever


class UserManagerHolder():

    def __init__(self, context: AIQContext) -> None:
        self._context = context

    def get_id(self):
        return self._context.user_manager.get_id()


class Builder(ABC):  # pylint: disable=too-many-public-methods

    @abstractmethod
    async def add_function(self, name: str | FunctionRef, config: FunctionBaseConfig) -> Function:
        pass

    @abstractmethod
    def get_function(self, name: str | FunctionRef) -> Function:
        pass

    def get_functions(self, function_names: Sequence[str | FunctionRef]) -> list[Function]:

        return [self.get_function(name) for name in function_names]

    @abstractmethod
    def get_function_config(self, name: str | FunctionRef) -> FunctionBaseConfig:
        pass

    @abstractmethod
    async def set_workflow(self, config: FunctionBaseConfig) -> Function:
        pass

    @abstractmethod
    def get_workflow(self) -> Function:
        pass

    @abstractmethod
    def get_workflow_config(self) -> FunctionBaseConfig:
        pass

    def get_tools(self, tool_names: Sequence[str | FunctionRef],
                  wrapper_type: LLMFrameworkEnum | str) -> list[typing.Any]:

        return [self.get_tool(fn_name=n, wrapper_type=wrapper_type) for n in tool_names]

    @abstractmethod
    def get_tool(self, fn_name: str | FunctionRef, wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        pass

    @abstractmethod
    async def add_llm(self, name: str | LLMRef, config: LLMBaseConfig):
        pass

    @abstractmethod
    async def get_llm(self, llm_name: str | LLMRef, wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        pass

    async def get_llms(self, llm_names: Sequence[str | LLMRef],
                       wrapper_type: LLMFrameworkEnum | str) -> list[typing.Any]:

        coros = [self.get_llm(llm_name=n, wrapper_type=wrapper_type) for n in llm_names]

        llms = await asyncio.gather(*coros, return_exceptions=False)

        return list(llms)

    @abstractmethod
    def get_llm_config(self, llm_name: str | LLMRef) -> LLMBaseConfig:
        pass

    @abstractmethod
    async def add_auth_provider(self, name: str | AuthenticationRef, config: AuthProviderBaseConfig):
        pass

    @abstractmethod
    async def get_auth_provider(self, auth_provider_name: str | AuthenticationRef) -> AuthProviderBase:
        pass

    async def get_auth_providers(self, auth_provider_names: list[str | AuthenticationRef]):

        coros = [self.get_auth_provider(auth_provider_name=n) for n in auth_provider_names]

        auth_providers = await asyncio.gather(*coros, return_exceptions=False)

        return list(auth_providers)

    @abstractmethod
    async def add_object_store(self, name: str | ObjectStoreRef, config: ObjectStoreBaseConfig):
        pass

    async def get_object_store_clients(self, object_store_names: Sequence[str | ObjectStoreRef]) -> list[ObjectStore]:
        """
        Return a list of all object store clients.
        """
        return list(await asyncio.gather(*[self.get_object_store_client(name) for name in object_store_names]))

    @abstractmethod
    async def get_object_store_client(self, object_store_name: str | ObjectStoreRef) -> ObjectStore:
        pass

    @abstractmethod
    def get_object_store_config(self, object_store_name: str | ObjectStoreRef) -> ObjectStoreBaseConfig:
        pass

    @abstractmethod
    async def add_embedder(self, name: str | EmbedderRef, config: EmbedderBaseConfig):
        pass

    async def get_embedders(self, embedder_names: Sequence[str | EmbedderRef],
                            wrapper_type: LLMFrameworkEnum | str) -> list[typing.Any]:

        coros = [self.get_embedder(embedder_name=n, wrapper_type=wrapper_type) for n in embedder_names]

        embedders = await asyncio.gather(*coros, return_exceptions=False)

        return list(embedders)

    @abstractmethod
    async def get_embedder(self, embedder_name: str | EmbedderRef, wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        pass

    @abstractmethod
    def get_embedder_config(self, embedder_name: str | EmbedderRef) -> EmbedderBaseConfig:
        pass

    @abstractmethod
    async def add_memory_client(self, name: str | MemoryRef, config: MemoryBaseConfig):
        pass

    def get_memory_clients(self, memory_names: Sequence[str | MemoryRef]) -> list[MemoryEditor]:
        """
        Return a list of memory clients for the specified names.
        """
        return [self.get_memory_client(n) for n in memory_names]

    @abstractmethod
    def get_memory_client(self, memory_name: str | MemoryRef) -> MemoryEditor:
        """
        Return the instantiated memory client for the given name.
        """
        pass

    @abstractmethod
    def get_memory_client_config(self, memory_name: str | MemoryRef) -> MemoryBaseConfig:
        pass

    @abstractmethod
    async def add_retriever(self, name: str | RetrieverRef, config: RetrieverBaseConfig):
        pass

    async def get_retrievers(self,
                             retriever_names: Sequence[str | RetrieverRef],
                             wrapper_type: LLMFrameworkEnum | str | None = None):

        tasks = [self.get_retriever(n, wrapper_type=wrapper_type) for n in retriever_names]

        retrievers = await asyncio.gather(*tasks, return_exceptions=False)

        return list(retrievers)

    @typing.overload
    async def get_retriever(self, retriever_name: str | RetrieverRef,
                            wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        ...

    @typing.overload
    async def get_retriever(self, retriever_name: str | RetrieverRef, wrapper_type: None) -> AIQRetriever:
        ...

    @typing.overload
    async def get_retriever(self, retriever_name: str | RetrieverRef) -> AIQRetriever:
        ...

    @abstractmethod
    async def get_retriever(self,
                            retriever_name: str | RetrieverRef,
                            wrapper_type: LLMFrameworkEnum | str | None = None) -> typing.Any:
        pass

    @abstractmethod
    async def get_retriever_config(self, retriever_name: str | RetrieverRef) -> RetrieverBaseConfig:
        pass

    @abstractmethod
    async def add_ttc_strategy(self, name: str | str, config: TTCStrategyBaseConfig):
        pass

    @abstractmethod
    async def get_ttc_strategy(self,
                               strategy_name: str | TTCStrategyRef,
                               pipeline_type: PipelineTypeEnum,
                               stage_type: StageTypeEnum):
        pass

    @abstractmethod
    async def get_ttc_strategy_config(self,
                                      strategy_name: str | TTCStrategyRef,
                                      pipeline_type: PipelineTypeEnum,
                                      stage_type: StageTypeEnum) -> TTCStrategyBaseConfig:
        pass

    @abstractmethod
    def get_user_manager(self) -> UserManagerHolder:
        pass

    @abstractmethod
    def get_function_dependencies(self, fn_name: str) -> FunctionDependencies:
        pass


class EvalBuilder(Builder):

    @abstractmethod
    async def add_evaluator(self, name: str, config: EvaluatorBaseConfig):
        pass

    @abstractmethod
    def get_evaluator(self, evaluator_name: str) -> typing.Any:
        pass

    @abstractmethod
    def get_evaluator_config(self, evaluator_name: str) -> EvaluatorBaseConfig:
        pass

    @abstractmethod
    def get_max_concurrency(self) -> int:
        pass

    @abstractmethod
    def get_output_dir(self) -> Path:
        pass

    @abstractmethod
    def get_all_tools(self, wrapper_type: LLMFrameworkEnum | str) -> list[typing.Any]:
        pass
