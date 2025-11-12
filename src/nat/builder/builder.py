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

from nat.authentication.interfaces import AuthProviderBase
from nat.builder.context import Context
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.builder.function import FunctionGroup
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.component_ref import AuthenticationRef
from nat.data_models.component_ref import EmbedderRef
from nat.data_models.component_ref import FunctionGroupRef
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.component_ref import MemoryRef
from nat.data_models.component_ref import MiddlewareRef
from nat.data_models.component_ref import ObjectStoreRef
from nat.data_models.component_ref import RetrieverRef
from nat.data_models.component_ref import TTCStrategyRef
from nat.data_models.embedder import EmbedderBaseConfig
from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.function import FunctionGroupBaseConfig
from nat.data_models.function_dependencies import FunctionDependencies
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.memory import MemoryBaseConfig
from nat.data_models.middleware import MiddlewareBaseConfig
from nat.data_models.object_store import ObjectStoreBaseConfig
from nat.data_models.retriever import RetrieverBaseConfig
from nat.data_models.ttc_strategy import TTCStrategyBaseConfig
from nat.experimental.decorators.experimental_warning_decorator import experimental
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.memory.interfaces import MemoryEditor
from nat.middleware.middleware import Middleware
from nat.object_store.interfaces import ObjectStore
from nat.retriever.interface import Retriever

if typing.TYPE_CHECKING:
    from nat.experimental.test_time_compute.models.strategy_base import StrategyBase


class UserManagerHolder:

    def __init__(self, context: Context) -> None:
        self._context = context

    def get_id(self):
        return self._context.user_manager.get_id()


class Builder(ABC):

    @abstractmethod
    async def add_function(self, name: str | FunctionRef, config: FunctionBaseConfig) -> Function:
        pass

    @abstractmethod
    async def add_function_group(self, name: str | FunctionGroupRef, config: FunctionGroupBaseConfig) -> FunctionGroup:
        pass

    @abstractmethod
    async def get_function(self, name: str | FunctionRef) -> Function:
        pass

    @abstractmethod
    async def get_function_group(self, name: str | FunctionGroupRef) -> FunctionGroup:
        pass

    async def get_functions(self, function_names: Sequence[str | FunctionRef]) -> list[Function]:
        tasks = [self.get_function(name) for name in function_names]
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    async def get_function_groups(self, function_group_names: Sequence[str | FunctionGroupRef]) -> list[FunctionGroup]:
        tasks = [self.get_function_group(name) for name in function_group_names]
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    @abstractmethod
    def get_function_config(self, name: str | FunctionRef) -> FunctionBaseConfig:
        pass

    @abstractmethod
    def get_function_group_config(self, name: str | FunctionGroupRef) -> FunctionGroupBaseConfig:
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

    @abstractmethod
    async def get_tools(self,
                        tool_names: Sequence[str | FunctionRef | FunctionGroupRef],
                        wrapper_type: LLMFrameworkEnum | str) -> list[typing.Any]:
        pass

    @abstractmethod
    async def get_tool(self, fn_name: str | FunctionRef, wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        pass

    @abstractmethod
    async def add_llm(self, name: str | LLMRef, config: LLMBaseConfig) -> typing.Any:
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
    @experimental(feature_name="Authentication")
    async def add_auth_provider(self, name: str | AuthenticationRef,
                                config: AuthProviderBaseConfig) -> AuthProviderBase:
        pass

    @abstractmethod
    async def get_auth_provider(self, auth_provider_name: str | AuthenticationRef) -> AuthProviderBase:
        pass

    async def get_auth_providers(self, auth_provider_names: list[str | AuthenticationRef]):

        coros = [self.get_auth_provider(auth_provider_name=n) for n in auth_provider_names]

        auth_providers = await asyncio.gather(*coros, return_exceptions=False)

        return list(auth_providers)

    @abstractmethod
    async def add_object_store(self, name: str | ObjectStoreRef, config: ObjectStoreBaseConfig) -> ObjectStore:
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
    async def add_embedder(self, name: str | EmbedderRef, config: EmbedderBaseConfig) -> None:
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
    async def add_memory_client(self, name: str | MemoryRef, config: MemoryBaseConfig) -> MemoryEditor:
        pass

    async def get_memory_clients(self, memory_names: Sequence[str | MemoryRef]) -> list[MemoryEditor]:
        """
        Return a list of memory clients for the specified names.
        """
        tasks = [self.get_memory_client(n) for n in memory_names]
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    @abstractmethod
    async def get_memory_client(self, memory_name: str | MemoryRef) -> MemoryEditor:
        """
        Return the instantiated memory client for the given name.
        """
        pass

    @abstractmethod
    def get_memory_client_config(self, memory_name: str | MemoryRef) -> MemoryBaseConfig:
        pass

    @abstractmethod
    async def add_retriever(self, name: str | RetrieverRef, config: RetrieverBaseConfig) -> None:
        pass

    async def get_retrievers(self,
                             retriever_names: Sequence[str | RetrieverRef],
                             wrapper_type: LLMFrameworkEnum | str | None = None) -> list[Retriever]:

        tasks = [self.get_retriever(n, wrapper_type=wrapper_type) for n in retriever_names]

        retrievers = await asyncio.gather(*tasks, return_exceptions=False)

        return list(retrievers)

    @typing.overload
    async def get_retriever(self, retriever_name: str | RetrieverRef,
                            wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        ...

    @typing.overload
    async def get_retriever(self, retriever_name: str | RetrieverRef, wrapper_type: None) -> Retriever:
        ...

    @typing.overload
    async def get_retriever(self, retriever_name: str | RetrieverRef) -> Retriever:
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
    @experimental(feature_name="TTC")
    async def add_ttc_strategy(self, name: str | TTCStrategyRef, config: TTCStrategyBaseConfig):
        pass

    @abstractmethod
    async def get_ttc_strategy(self,
                               strategy_name: str | TTCStrategyRef,
                               pipeline_type: PipelineTypeEnum,
                               stage_type: StageTypeEnum) -> "StrategyBase":
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

    @abstractmethod
    def get_function_group_dependencies(self, fn_name: str) -> FunctionDependencies:
        pass

    @abstractmethod
    async def add_middleware(self, name: str | MiddlewareRef, config: MiddlewareBaseConfig) -> Middleware:
        """Add middleware to the builder.

        Args:
            name: The name or reference for the middleware
            config: The configuration for the middleware

        Returns:
            The built middleware instance
        """
        pass

    @abstractmethod
    async def get_middleware(self, middleware_name: str | MiddlewareRef) -> Middleware:
        """Get built middleware by name.

        Args:
            middleware_name: The name or reference of the middleware

        Returns:
            The built middleware instance
        """
        pass

    @abstractmethod
    def get_middleware_config(self, middleware_name: str | MiddlewareRef) -> MiddlewareBaseConfig:
        """Get the configuration for middleware.

        Args:
            middleware_name: The name or reference of the middleware

        Returns:
            The configuration for the middleware
        """
        pass

    async def get_middleware_list(self, middleware_names: Sequence[str | MiddlewareRef]) -> list[Middleware]:
        """Get multiple middleware by name.

        Args:
            middleware_names: The names or references of the middleware

        Returns:
            List of built middleware instances
        """
        tasks = [self.get_middleware(name) for name in middleware_names]
        return list(await asyncio.gather(*tasks, return_exceptions=False))


class EvalBuilder(ABC):

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
    async def get_all_tools(self, wrapper_type: LLMFrameworkEnum | str) -> list[typing.Any]:
        pass
