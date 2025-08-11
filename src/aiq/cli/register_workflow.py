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

from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.type_registry import AuthProviderBuildCallableT
from aiq.cli.type_registry import AuthProviderRegisteredCallableT
from aiq.cli.type_registry import EmbedderClientBuildCallableT
from aiq.cli.type_registry import EmbedderClientRegisteredCallableT
from aiq.cli.type_registry import EmbedderProviderBuildCallableT
from aiq.cli.type_registry import EmbedderProviderRegisteredCallableT
from aiq.cli.type_registry import EvaluatorBuildCallableT
from aiq.cli.type_registry import EvaluatorRegisteredCallableT
from aiq.cli.type_registry import FrontEndBuildCallableT
from aiq.cli.type_registry import FrontEndRegisteredCallableT
from aiq.cli.type_registry import FunctionBuildCallableT
from aiq.cli.type_registry import FunctionRegisteredCallableT
from aiq.cli.type_registry import LLMClientBuildCallableT
from aiq.cli.type_registry import LLMClientRegisteredCallableT
from aiq.cli.type_registry import LLMProviderBuildCallableT
from aiq.cli.type_registry import LoggingMethodBuildCallableT
from aiq.cli.type_registry import LoggingMethodConfigT
from aiq.cli.type_registry import LoggingMethodRegisteredCallableT
from aiq.cli.type_registry import MemoryBuildCallableT
from aiq.cli.type_registry import MemoryRegisteredCallableT
from aiq.cli.type_registry import ObjectStoreBuildCallableT
from aiq.cli.type_registry import ObjectStoreRegisteredCallableT
from aiq.cli.type_registry import RegisteredLoggingMethod
from aiq.cli.type_registry import RegisteredTelemetryExporter
from aiq.cli.type_registry import RegisteredToolWrapper
from aiq.cli.type_registry import RegistryHandlerBuildCallableT
from aiq.cli.type_registry import RegistryHandlerRegisteredCallableT
from aiq.cli.type_registry import RetrieverClientBuildCallableT
from aiq.cli.type_registry import RetrieverClientRegisteredCallableT
from aiq.cli.type_registry import RetrieverProviderBuildCallableT
from aiq.cli.type_registry import RetrieverProviderRegisteredCallableT
from aiq.cli.type_registry import TeleExporterRegisteredCallableT
from aiq.cli.type_registry import TelemetryExporterBuildCallableT
from aiq.cli.type_registry import TelemetryExporterConfigT
from aiq.cli.type_registry import ToolWrapperBuildCallableT
from aiq.cli.type_registry import TTCStrategyBuildCallableT
from aiq.cli.type_registry import TTCStrategyRegisterCallableT
from aiq.data_models.authentication import AuthProviderBaseConfigT
from aiq.data_models.component import ComponentEnum
from aiq.data_models.discovery_metadata import DiscoveryMetadata
from aiq.data_models.embedder import EmbedderBaseConfigT
from aiq.data_models.evaluator import EvaluatorBaseConfigT
from aiq.data_models.front_end import FrontEndConfigT
from aiq.data_models.function import FunctionConfigT
from aiq.data_models.llm import LLMBaseConfigT
from aiq.data_models.memory import MemoryBaseConfigT
from aiq.data_models.object_store import ObjectStoreBaseConfigT
from aiq.data_models.registry_handler import RegistryHandlerBaseConfigT
from aiq.data_models.retriever import RetrieverBaseConfigT


def register_telemetry_exporter(config_type: type[TelemetryExporterConfigT]):
    """
    Register a workflow with optional framework_wrappers for automatic profiler hooking.
    """

    def register_inner(
        fn: TelemetryExporterBuildCallableT[TelemetryExporterConfigT]
    ) -> TeleExporterRegisteredCallableT[TelemetryExporterConfigT]:
        from .type_registry import GlobalTypeRegistry

        context_manager_fn = asynccontextmanager(fn)

        discovery_metadata = DiscoveryMetadata.from_config_type(config_type=config_type,
                                                                component_type=ComponentEnum.TRACING)

        GlobalTypeRegistry.get().register_telemetry_exporter(
            RegisteredTelemetryExporter(full_type=config_type.full_type,
                                        config_type=config_type,
                                        build_fn=context_manager_fn,
                                        discovery_metadata=discovery_metadata))

        return context_manager_fn

    return register_inner


def register_logging_method(config_type: type[LoggingMethodConfigT]):

    def register_inner(
        fn: LoggingMethodBuildCallableT[LoggingMethodConfigT]
    ) -> LoggingMethodRegisteredCallableT[LoggingMethodConfigT]:
        from .type_registry import GlobalTypeRegistry

        context_manager_fn = asynccontextmanager(fn)

        discovery_metadata = DiscoveryMetadata.from_config_type(config_type=config_type,
                                                                component_type=ComponentEnum.LOGGING)

        GlobalTypeRegistry.get().register_logging_method(
            RegisteredLoggingMethod(full_type=config_type.full_type,
                                    config_type=config_type,
                                    build_fn=context_manager_fn,
                                    discovery_metadata=discovery_metadata))

        return context_manager_fn

    return register_inner


def register_front_end(config_type: type[FrontEndConfigT]):
    """
    Register a front end which is responsible for hosting a workflow.
    """

    def register_front_end_inner(
            fn: FrontEndBuildCallableT[FrontEndConfigT]) -> FrontEndRegisteredCallableT[FrontEndConfigT]:
        from .type_registry import GlobalTypeRegistry
        from .type_registry import RegisteredFrontEndInfo

        context_manager_fn = asynccontextmanager(fn)

        discovery_metadata = DiscoveryMetadata.from_config_type(config_type=config_type,
                                                                component_type=ComponentEnum.FRONT_END)

        GlobalTypeRegistry.get().register_front_end(
            RegisteredFrontEndInfo(full_type=config_type.full_type,
                                   config_type=config_type,
                                   build_fn=context_manager_fn,
                                   discovery_metadata=discovery_metadata))

        return context_manager_fn

    return register_front_end_inner


def register_function(config_type: type[FunctionConfigT],
                      framework_wrappers: list[LLMFrameworkEnum | str] | None = None):
    """
    Register a workflow with optional framework_wrappers for automatic profiler hooking.
    """

    def register_function_inner(
            fn: FunctionBuildCallableT[FunctionConfigT]) -> FunctionRegisteredCallableT[FunctionConfigT]:
        from .type_registry import GlobalTypeRegistry
        from .type_registry import RegisteredFunctionInfo

        context_manager_fn = asynccontextmanager(fn)

        if framework_wrappers is None:
            framework_wrappers_list: list[str] = []
        else:
            framework_wrappers_list = list(framework_wrappers)

        discovery_metadata = DiscoveryMetadata.from_config_type(config_type=config_type,
                                                                component_type=ComponentEnum.FUNCTION)

        GlobalTypeRegistry.get().register_function(
            RegisteredFunctionInfo(
                full_type=config_type.full_type,
                config_type=config_type,
                build_fn=context_manager_fn,
                framework_wrappers=framework_wrappers_list,
                discovery_metadata=discovery_metadata,
            ))

        return context_manager_fn

    return register_function_inner


def register_llm_provider(config_type: type[LLMBaseConfigT]):

    def register_llm_provider_inner(
            fn: LLMProviderBuildCallableT[LLMBaseConfigT]) -> LLMClientRegisteredCallableT[LLMBaseConfigT]:
        from .type_registry import GlobalTypeRegistry
        from .type_registry import RegisteredLLMProviderInfo

        context_manager_fn = asynccontextmanager(fn)

        discovery_metadata = DiscoveryMetadata.from_config_type(config_type=config_type,
                                                                component_type=ComponentEnum.LLM_PROVIDER)

        GlobalTypeRegistry.get().register_llm_provider(
            RegisteredLLMProviderInfo(full_type=config_type.full_type,
                                      config_type=config_type,
                                      build_fn=context_manager_fn,
                                      discovery_metadata=discovery_metadata))

        return context_manager_fn

    return register_llm_provider_inner


def register_auth_provider(config_type: type[AuthProviderBaseConfigT]):

    def register_auth_provider_inner(
        fn: AuthProviderBuildCallableT[AuthProviderBaseConfigT]
    ) -> AuthProviderRegisteredCallableT[AuthProviderBaseConfigT]:
        from .type_registry import GlobalTypeRegistry
        from .type_registry import RegisteredAuthProviderInfo

        context_manager_fn = asynccontextmanager(fn)

        discovery_metadata = DiscoveryMetadata.from_config_type(config_type=config_type,
                                                                component_type=ComponentEnum.AUTHENTICATION_PROVIDER)

        GlobalTypeRegistry.get().register_auth_provider(
            RegisteredAuthProviderInfo(full_type=config_type.full_type,
                                       config_type=config_type,
                                       build_fn=context_manager_fn,
                                       discovery_metadata=discovery_metadata))

        return context_manager_fn

    return register_auth_provider_inner


def register_llm_client(config_type: type[LLMBaseConfigT], wrapper_type: LLMFrameworkEnum | str):

    def register_llm_client_inner(
            fn: LLMClientBuildCallableT[LLMBaseConfigT]) -> LLMClientRegisteredCallableT[LLMBaseConfigT]:
        from .type_registry import GlobalTypeRegistry
        from .type_registry import RegisteredLLMClientInfo

        context_manager_fn = asynccontextmanager(fn)

        discovery_metadata = DiscoveryMetadata.from_provider_framework_map(config_type=config_type,
                                                                           wrapper_type=wrapper_type,
                                                                           provider_type=ComponentEnum.LLM_PROVIDER,
                                                                           component_type=ComponentEnum.LLM_CLIENT)
        GlobalTypeRegistry.get().register_llm_client(
            RegisteredLLMClientInfo(full_type=config_type.full_type,
                                    config_type=config_type,
                                    build_fn=context_manager_fn,
                                    llm_framework=wrapper_type,
                                    discovery_metadata=discovery_metadata))

        return context_manager_fn

    return register_llm_client_inner


def register_embedder_provider(config_type: type[EmbedderBaseConfigT]):

    def register_embedder_provider_inner(
        fn: EmbedderProviderBuildCallableT[EmbedderBaseConfigT]
    ) -> EmbedderProviderRegisteredCallableT[EmbedderBaseConfigT]:
        from .type_registry import GlobalTypeRegistry
        from .type_registry import RegisteredEmbedderProviderInfo

        context_manager_fn = asynccontextmanager(fn)

        discovery_metadata = DiscoveryMetadata.from_config_type(config_type=config_type,
                                                                component_type=ComponentEnum.EMBEDDER_PROVIDER)

        GlobalTypeRegistry.get().register_embedder_provider(
            RegisteredEmbedderProviderInfo(full_type=config_type.full_type,
                                           config_type=config_type,
                                           build_fn=context_manager_fn,
                                           discovery_metadata=discovery_metadata))

        return context_manager_fn

    return register_embedder_provider_inner


def register_embedder_client(config_type: type[EmbedderBaseConfigT], wrapper_type: LLMFrameworkEnum | str):

    def register_embedder_client_inner(
        fn: EmbedderClientBuildCallableT[EmbedderBaseConfigT]
    ) -> EmbedderClientRegisteredCallableT[EmbedderBaseConfigT]:
        from .type_registry import GlobalTypeRegistry
        from .type_registry import RegisteredEmbedderClientInfo

        context_manager_fn = asynccontextmanager(fn)

        discovery_metadata = DiscoveryMetadata.from_provider_framework_map(
            config_type=config_type,
            wrapper_type=wrapper_type,
            provider_type=ComponentEnum.EMBEDDER_PROVIDER,
            component_type=ComponentEnum.EMBEDDER_CLIENT)

        GlobalTypeRegistry.get().register_embedder_client(
            RegisteredEmbedderClientInfo(full_type=config_type.full_type,
                                         config_type=config_type,
                                         build_fn=context_manager_fn,
                                         llm_framework=wrapper_type,
                                         discovery_metadata=discovery_metadata))

        return context_manager_fn

    return register_embedder_client_inner


def register_evaluator(config_type: type[EvaluatorBaseConfigT]):

    def register_evaluator_inner(
            fn: EvaluatorBuildCallableT[EvaluatorBaseConfigT]) -> EvaluatorRegisteredCallableT[EvaluatorBaseConfigT]:
        from .type_registry import GlobalTypeRegistry
        from .type_registry import RegisteredEvaluatorInfo

        context_manager_fn = asynccontextmanager(fn)

        discovery_metadata = DiscoveryMetadata.from_config_type(config_type=config_type,
                                                                component_type=ComponentEnum.EVALUATOR)

        GlobalTypeRegistry.get().register_evaluator(
            RegisteredEvaluatorInfo(full_type=config_type.full_type,
                                    config_type=config_type,
                                    build_fn=context_manager_fn,
                                    discovery_metadata=discovery_metadata))

        return context_manager_fn

    return register_evaluator_inner


def register_memory(config_type: type[MemoryBaseConfigT]):

    def register_memory_inner(
            fn: MemoryBuildCallableT[MemoryBaseConfigT]) -> MemoryRegisteredCallableT[MemoryBaseConfigT]:
        from .type_registry import GlobalTypeRegistry
        from .type_registry import RegisteredMemoryInfo

        context_manager_fn = asynccontextmanager(fn)

        discovery_metadata = DiscoveryMetadata.from_config_type(config_type=config_type,
                                                                component_type=ComponentEnum.MEMORY)

        GlobalTypeRegistry.get().register_memory(
            RegisteredMemoryInfo(full_type=config_type.full_type,
                                 config_type=config_type,
                                 build_fn=context_manager_fn,
                                 discovery_metadata=discovery_metadata))

        return context_manager_fn

    return register_memory_inner


def register_object_store(config_type: type[ObjectStoreBaseConfigT]):

    def register_kv_store_inner(
        fn: ObjectStoreBuildCallableT[ObjectStoreBaseConfigT]
    ) -> ObjectStoreRegisteredCallableT[ObjectStoreBaseConfigT]:
        from .type_registry import GlobalTypeRegistry
        from .type_registry import RegisteredObjectStoreInfo

        context_manager_fn = asynccontextmanager(fn)

        discovery_metadata = DiscoveryMetadata.from_config_type(config_type=config_type,
                                                                component_type=ComponentEnum.OBJECT_STORE)

        GlobalTypeRegistry.get().register_object_store(
            RegisteredObjectStoreInfo(full_type=config_type.full_type,
                                      config_type=config_type,
                                      build_fn=context_manager_fn,
                                      discovery_metadata=discovery_metadata))

        return context_manager_fn

    return register_kv_store_inner


def register_ttc_strategy(config_type: type[TTCStrategyRegisterCallableT]):

    def register_ttc_strategy_inner(
        fn: TTCStrategyBuildCallableT[TTCStrategyRegisterCallableT]
    ) -> TTCStrategyRegisterCallableT[TTCStrategyRegisterCallableT]:
        from .type_registry import GlobalTypeRegistry
        from .type_registry import RegisteredTTCStrategyInfo

        context_manager_fn = asynccontextmanager(fn)

        discovery_metadata = DiscoveryMetadata.from_config_type(config_type=config_type,
                                                                component_type=ComponentEnum.TTC_STRATEGY)

        GlobalTypeRegistry.get().register_ttc_strategy(
            RegisteredTTCStrategyInfo(full_type=config_type.full_type,
                                      config_type=config_type,
                                      build_fn=context_manager_fn,
                                      discovery_metadata=discovery_metadata))

        return context_manager_fn

    return register_ttc_strategy_inner


def register_retriever_provider(config_type: type[RetrieverBaseConfigT]):

    def register_retriever_provider_inner(
        fn: RetrieverProviderBuildCallableT[RetrieverBaseConfigT]
    ) -> RetrieverProviderRegisteredCallableT[RetrieverBaseConfigT]:
        from .type_registry import GlobalTypeRegistry
        from .type_registry import RegisteredRetrieverProviderInfo

        context_manager_fn = asynccontextmanager(fn)

        discovery_metadata = DiscoveryMetadata.from_config_type(config_type=config_type,
                                                                component_type=ComponentEnum.RETRIEVER_PROVIDER)

        GlobalTypeRegistry.get().register_retriever_provider(
            RegisteredRetrieverProviderInfo(full_type=config_type.full_type,
                                            config_type=config_type,
                                            build_fn=context_manager_fn,
                                            discovery_metadata=discovery_metadata))

        return context_manager_fn

    return register_retriever_provider_inner


def register_retriever_client(config_type: type[RetrieverBaseConfigT], wrapper_type: LLMFrameworkEnum | str | None):

    def register_retriever_client_inner(
        fn: RetrieverClientBuildCallableT[RetrieverBaseConfigT]
    ) -> RetrieverClientRegisteredCallableT[RetrieverBaseConfigT]:
        from .type_registry import GlobalTypeRegistry
        from .type_registry import RegisteredRetrieverClientInfo

        context_manager_fn = asynccontextmanager(fn)

        discovery_metadata = DiscoveryMetadata.from_provider_framework_map(
            config_type=config_type,
            wrapper_type=wrapper_type,
            provider_type=ComponentEnum.RETRIEVER_PROVIDER,
            component_type=ComponentEnum.RETRIEVER_CLIENT,
        )

        GlobalTypeRegistry.get().register_retriever_client(
            RegisteredRetrieverClientInfo(full_type=config_type.full_type,
                                          config_type=config_type,
                                          build_fn=context_manager_fn,
                                          llm_framework=wrapper_type,
                                          discovery_metadata=discovery_metadata))

        return context_manager_fn

    return register_retriever_client_inner


def register_tool_wrapper(wrapper_type: LLMFrameworkEnum | str):

    def _inner(fn: ToolWrapperBuildCallableT) -> ToolWrapperBuildCallableT:
        from .type_registry import GlobalTypeRegistry

        discovery_metadata = DiscoveryMetadata.from_fn_wrapper(fn=fn,
                                                               wrapper_type=wrapper_type,
                                                               component_type=ComponentEnum.TOOL_WRAPPER)
        GlobalTypeRegistry.get().register_tool_wrapper(
            RegisteredToolWrapper(llm_framework=wrapper_type, build_fn=fn, discovery_metadata=discovery_metadata))

        return fn

    return _inner


def register_registry_handler(config_type: type[RegistryHandlerBaseConfigT]):

    def register_registry_handler_inner(
        fn: RegistryHandlerBuildCallableT[RegistryHandlerBaseConfigT]
    ) -> RegistryHandlerRegisteredCallableT[RegistryHandlerBaseConfigT]:
        from .type_registry import GlobalTypeRegistry
        from .type_registry import RegisteredRegistryHandlerInfo

        context_manager_fn = asynccontextmanager(fn)

        discovery_metadata = DiscoveryMetadata.from_config_type(config_type=config_type,
                                                                component_type=ComponentEnum.REGISTRY_HANDLER)

        GlobalTypeRegistry.get().register_registry_handler(
            RegisteredRegistryHandlerInfo(full_type=config_type.full_type,
                                          config_type=config_type,
                                          build_fn=context_manager_fn,
                                          discovery_metadata=discovery_metadata))

        return context_manager_fn

    return register_registry_handler_inner
