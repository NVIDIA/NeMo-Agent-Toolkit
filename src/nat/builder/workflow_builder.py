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
import dataclasses
import inspect
import logging
import typing
import warnings
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from contextlib import AsyncExitStack
from contextlib import asynccontextmanager
from typing import cast

from nat.authentication.interfaces import AuthProviderBase
from nat.builder.builder import Builder
from nat.builder.builder import UserManagerHolder
from nat.builder.component_utils import ComponentInstanceData
from nat.builder.component_utils import build_dependency_sequence
from nat.builder.context import Context
from nat.builder.context import ContextState
from nat.builder.embedder import EmbedderProviderInfo
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.builder.function import FunctionGroup
from nat.builder.function import LambdaFunction
from nat.builder.function_info import FunctionInfo
from nat.builder.llm import LLMProviderInfo
from nat.builder.retriever import RetrieverProviderInfo
from nat.builder.workflow import Workflow
from nat.cli.type_registry import GlobalTypeRegistry
from nat.cli.type_registry import TypeRegistry
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.component import ComponentGroup
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
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.data_models.embedder import EmbedderBaseConfig
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.function import FunctionGroupBaseConfig
from nat.data_models.function_dependencies import FunctionDependencies
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.memory import MemoryBaseConfig
from nat.data_models.middleware import MiddlewareBaseConfig
from nat.data_models.object_store import ObjectStoreBaseConfig
from nat.data_models.retriever import RetrieverBaseConfig
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.data_models.ttc_strategy import TTCStrategyBaseConfig
from nat.experimental.decorators.experimental_warning_decorator import experimental
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.experimental.test_time_compute.models.strategy_base import StrategyBase
from nat.memory.interfaces import MemoryEditor
from nat.middleware.function_middleware import FunctionMiddleware
from nat.middleware.middleware import Middleware
from nat.object_store.interfaces import ObjectStore
from nat.observability.exporter.base_exporter import BaseExporter
from nat.profiler.decorators.framework_wrapper import chain_wrapped_build_fn
from nat.profiler.utils import detect_llm_frameworks_in_build_fn
from nat.retriever.interface import Retriever
from nat.utils.type_utils import override

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ConfiguredTelemetryExporter:
    config: TelemetryExporterBaseConfig
    instance: BaseExporter


@dataclasses.dataclass
class ConfiguredFunction:
    config: FunctionBaseConfig
    instance: Function


@dataclasses.dataclass
class ConfiguredFunctionGroup:
    config: FunctionGroupBaseConfig
    instance: FunctionGroup


@dataclasses.dataclass
class ConfiguredLLM:
    config: LLMBaseConfig
    instance: LLMProviderInfo


@dataclasses.dataclass
class ConfiguredEmbedder:
    config: EmbedderBaseConfig
    instance: EmbedderProviderInfo


@dataclasses.dataclass
class ConfiguredMemory:
    config: MemoryBaseConfig
    instance: MemoryEditor


@dataclasses.dataclass
class ConfiguredObjectStore:
    config: ObjectStoreBaseConfig
    instance: ObjectStore


@dataclasses.dataclass
class ConfiguredRetriever:
    config: RetrieverBaseConfig
    instance: RetrieverProviderInfo


@dataclasses.dataclass
class ConfiguredAuthProvider:
    config: AuthProviderBaseConfig
    instance: AuthProviderBase


@dataclasses.dataclass
class ConfiguredTTCStrategy:
    config: TTCStrategyBaseConfig
    instance: StrategyBase


@dataclasses.dataclass
class ConfiguredMiddleware:
    config: MiddlewareBaseConfig
    instance: Middleware


class WorkflowBuilder(Builder, AbstractAsyncContextManager):

    def __init__(self, *, general_config: GeneralConfig | None = None, registry: TypeRegistry | None = None):

        if general_config is None:
            general_config = GeneralConfig()

        if registry is None:
            registry = GlobalTypeRegistry.get()

        self.general_config = general_config

        self._registry = registry

        self._logging_handlers: dict[str, logging.Handler] = {}
        self._removed_root_handlers: list[tuple[logging.Handler, int]] = []
        self._telemetry_exporters: dict[str, ConfiguredTelemetryExporter] = {}

        self._functions: dict[str, ConfiguredFunction] = {}
        self._function_groups: dict[str, ConfiguredFunctionGroup] = {}
        self._workflow: ConfiguredFunction | None = None

        self._llms: dict[str, ConfiguredLLM] = {}
        self._auth_providers: dict[str, ConfiguredAuthProvider] = {}
        self._embedders: dict[str, ConfiguredEmbedder] = {}
        self._memory_clients: dict[str, ConfiguredMemory] = {}
        self._object_stores: dict[str, ConfiguredObjectStore] = {}
        self._retrievers: dict[str, ConfiguredRetriever] = {}
        self._ttc_strategies: dict[str, ConfiguredTTCStrategy] = {}
        self._middleware: dict[str, ConfiguredMiddleware] = {}

        self._context_state = ContextState.get()

        self._exit_stack: AsyncExitStack | None = None

        # Create a mapping to track function name -> other function names it depends on
        self.function_dependencies: dict[str, FunctionDependencies] = {}
        self.function_group_dependencies: dict[str, FunctionDependencies] = {}
        self.current_function_building: str | None = None
        self.current_function_group_building: str | None = None

    async def __aenter__(self):

        self._exit_stack = AsyncExitStack()

        # Get the telemetry info from the config
        telemetry_config = self.general_config.telemetry

        # If we have logging configuration, we need to manage the root logger properly
        root_logger = logging.getLogger()

        # Collect configured handler types to determine if we need to adjust existing handlers
        # This is somewhat of a hack by inspecting the class name of the config object
        has_console_handler = any(
            hasattr(config, "__class__") and "console" in config.__class__.__name__.lower()
            for config in telemetry_config.logging.values())

        for key, logging_config in telemetry_config.logging.items():
            # Use the same pattern as tracing, but for logging
            logging_info = self._registry.get_logging_method(type(logging_config))
            handler = await self._exit_stack.enter_async_context(logging_info.build_fn(logging_config, self))

            # Type check
            if not isinstance(handler, logging.Handler):
                raise TypeError(f"Expected a logging.Handler from {key}, got {type(handler)}")

            # Store them in a dict so we can un-register them if needed
            self._logging_handlers[key] = handler

            # Now attach to NAT's root logger
            root_logger.addHandler(handler)

        # If we added logging handlers, manage existing handlers appropriately
        if self._logging_handlers:
            min_handler_level = min((handler.level for handler in root_logger.handlers), default=logging.CRITICAL)

            # Ensure the root logger level allows messages through
            root_logger.level = max(root_logger.level, min_handler_level)

            # If a console handler is configured, adjust or remove default CLI handlers
            # to avoid duplicate output while preserving workflow visibility
            if has_console_handler:
                # Remove existing StreamHandlers that are not the newly configured ones
                for handler in root_logger.handlers[:]:
                    if type(handler) is logging.StreamHandler and handler not in self._logging_handlers.values():
                        self._removed_root_handlers.append((handler, handler.level))
                        root_logger.removeHandler(handler)
            else:
                # No console handler configured, but adjust existing handler levels
                # to respect the minimum configured level for file/other handlers
                for handler in root_logger.handlers[:]:
                    if type(handler) is logging.StreamHandler:
                        old_level = handler.level
                        handler.setLevel(min_handler_level)
                        self._removed_root_handlers.append((handler, old_level))

        # Add the telemetry exporters
        for key, telemetry_exporter_config in telemetry_config.tracing.items():
            await self.add_telemetry_exporter(key, telemetry_exporter_config)

        return self

    async def __aexit__(self, *exc_details):

        assert self._exit_stack is not None, "Exit stack not initialized"

        root_logger = logging.getLogger()

        # Remove custom logging handlers
        for handler in self._logging_handlers.values():
            root_logger.removeHandler(handler)

        # Restore original handlers and their levels
        for handler, old_level in self._removed_root_handlers:
            if handler not in root_logger.handlers:
                root_logger.addHandler(handler)
            handler.setLevel(old_level)

        await self._exit_stack.__aexit__(*exc_details)

    async def build(self, entry_function: str | None = None) -> Workflow:
        """
        Creates an instance of a workflow object using the added components and the desired entry function.

        Parameters
        ----------
        entry_function : str | None, optional
            The function name to use as the entry point for the created workflow. If None, the entry point will be the
            specified workflow function. By default None

        Returns
        -------
        Workflow
            A created workflow.

        Raises
        ------
        ValueError
            If the workflow has not been set before building.
        """

        if (self._workflow is None):
            raise ValueError("Must set a workflow before building")

        # Set of all functions which are "included" by function groups
        included_functions = set()
        # Dictionary of function configs
        function_configs = dict()
        # Dictionary of function group configs
        function_group_configs = dict()
        # Dictionary of function instances
        function_instances = dict()
        # Dictionary of function group instances
        function_group_instances = dict()

        for k, v in self._function_groups.items():
            included_functions.update((await v.instance.get_included_functions()).keys())
            function_group_configs[k] = v.config
            function_group_instances[k] = v.instance

        # Function configs need to be restricted to only the functions that are not in a function group
        for k, v in self._functions.items():
            if k not in included_functions:
                function_configs[k] = v.config
                function_instances[k] = v.instance

        # Build the config from the added objects
        config = Config(general=self.general_config,
                        functions=function_configs,
                        function_groups=function_group_configs,
                        workflow=self._workflow.config,
                        llms={
                            k: v.config
                            for k, v in self._llms.items()
                        },
                        embedders={
                            k: v.config
                            for k, v in self._embedders.items()
                        },
                        memory={
                            k: v.config
                            for k, v in self._memory_clients.items()
                        },
                        object_stores={
                            k: v.config
                            for k, v in self._object_stores.items()
                        },
                        retrievers={
                            k: v.config
                            for k, v in self._retrievers.items()
                        },
                        ttc_strategies={
                            k: v.config
                            for k, v in self._ttc_strategies.items()
                        })

        if (entry_function is None):
            entry_fn_obj = self.get_workflow()
        else:
            entry_fn_obj = await self.get_function(entry_function)

        workflow = Workflow.from_entry_fn(config=config,
                                          entry_fn=entry_fn_obj,
                                          functions=function_instances,
                                          function_groups=function_group_instances,
                                          llms={
                                              k: v.instance
                                              for k, v in self._llms.items()
                                          },
                                          embeddings={
                                              k: v.instance
                                              for k, v in self._embedders.items()
                                          },
                                          memory={
                                              k: v.instance
                                              for k, v in self._memory_clients.items()
                                          },
                                          object_stores={
                                              k: v.instance
                                              for k, v in self._object_stores.items()
                                          },
                                          telemetry_exporters={
                                              k: v.instance
                                              for k, v in self._telemetry_exporters.items()
                                          },
                                          retrievers={
                                              k: v.instance
                                              for k, v in self._retrievers.items()
                                          },
                                          ttc_strategies={
                                              k: v.instance
                                              for k, v in self._ttc_strategies.items()
                                          },
                                          context_state=self._context_state)

        return workflow

    def _get_exit_stack(self) -> AsyncExitStack:

        if self._exit_stack is None:
            raise ValueError(
                "Exit stack not initialized. Did you forget to call `async with WorkflowBuilder() as builder`?")

        return self._exit_stack

    async def _build_function(self, name: str, config: FunctionBaseConfig) -> ConfiguredFunction:
        registration = self._registry.get_function(type(config))

        inner_builder = ChildBuilder(self)

        # We need to do this for every function because we don't know
        # Where LLama Index Agents are Instantiated and Settings need to
        # be set before the function is built
        # It's only slower the first time because of the import
        # So we can afford to do this for every function

        llms = {k: v.instance for k, v in self._llms.items()}
        function_frameworks = detect_llm_frameworks_in_build_fn(registration)

        build_fn = chain_wrapped_build_fn(registration.build_fn, llms, function_frameworks)

        # Set the currently building function so the ChildBuilder can track dependencies
        self.current_function_building = config.type
        # Empty set of dependencies for the current function
        self.function_dependencies[config.type] = FunctionDependencies()

        build_result = await self._get_exit_stack().enter_async_context(build_fn(config, inner_builder))

        self.function_dependencies[name] = inner_builder.dependencies

        # If the build result is a function, wrap it in a FunctionInfo
        if inspect.isfunction(build_result):

            build_result = FunctionInfo.from_fn(build_result)

        if (isinstance(build_result, FunctionInfo)):
            # Create the function object
            build_result = LambdaFunction.from_info(config=config, info=build_result, instance_name=name)

        if (not isinstance(build_result, Function)):
            raise ValueError("Expected a function, FunctionInfo object, or FunctionBase object to be "
                             f"returned from the function builder. Got {type(build_result)}")

        # Resolve middleware names from config to middleware instances
        # Only FunctionMiddleware types can be used with functions
        middleware_instances = []
        for middleware_name in config.middleware:
            if middleware_name not in self._middleware:
                raise ValueError(f"Middleware `{middleware_name}` not found for function `{name}`. "
                                 f"It must be configured in the `middleware` section of the YAML configuration.")
            middleware_obj = self._middleware[middleware_name].instance
            if not isinstance(middleware_obj, FunctionMiddleware):
                raise TypeError(
                    f"Middleware `{middleware_name}` is not a FunctionMiddleware and cannot be used with functions. "
                    f"Only FunctionMiddleware types support function-specific wrapping.")
            middleware_instances.append(middleware_obj)

        build_result.configure_middleware(middleware_instances)

        return ConfiguredFunction(config=config, instance=build_result)

    async def _build_function_group(self, name: str, config: FunctionGroupBaseConfig) -> ConfiguredFunctionGroup:
        """Build a function group from the provided configuration.

        Args:
            name: The name of the function group
            config: The function group configuration

        Returns:
            ConfiguredFunctionGroup: The built function group

        Raises:
            ValueError: If the function group builder returns invalid results
        """
        registration = self._registry.get_function_group(type(config))

        inner_builder = ChildBuilder(self)

        # Build the function group - use the same wrapping pattern as _build_function
        llms = {k: v.instance for k, v in self._llms.items()}
        function_frameworks = detect_llm_frameworks_in_build_fn(registration)

        build_fn = chain_wrapped_build_fn(registration.build_fn, llms, function_frameworks)

        # Set the currently building function group so the ChildBuilder can track dependencies
        self.current_function_group_building = config.type
        # Empty set of dependencies for the current function group
        self.function_group_dependencies[config.type] = FunctionDependencies()

        build_result = await self._get_exit_stack().enter_async_context(build_fn(config, inner_builder))

        self.function_group_dependencies[name] = inner_builder.dependencies

        if not isinstance(build_result, FunctionGroup):
            raise ValueError("Expected a FunctionGroup object to be returned from the function group builder. "
                             f"Got {type(build_result)}")

        # Resolve middleware names from config to middleware instances
        # Only FunctionMiddleware types can be used with function groups
        middleware_instances = []
        for middleware_name in config.middleware:
            if middleware_name not in self._middleware:
                raise ValueError(f"Middleware `{middleware_name}` not found for function group `{name}`. "
                                 f"It must be configured in the `middleware` section of the YAML configuration.")
            middleware_obj = self._middleware[middleware_name].instance
            if not isinstance(middleware_obj, FunctionMiddleware):
                raise TypeError(f"Middleware `{middleware_name}` is not a FunctionMiddleware and "
                                f"cannot be used with function groups. "
                                f"Only FunctionMiddleware types support function-specific wrapping.")
            middleware_instances.append(middleware_obj)

        # Configure middleware for the function group
        build_result.configure_middleware(middleware_instances)

        # set the instance name for the function group based on the workflow-provided name
        build_result.set_instance_name(name)
        return ConfiguredFunctionGroup(config=config, instance=build_result)

    @override
    async def add_function(self, name: str | FunctionRef, config: FunctionBaseConfig) -> Function:
        if isinstance(name, FunctionRef):
            name = str(name)

        if (name in self._functions or name in self._function_groups):
            raise ValueError(f"Function `{name}` already exists in the list of functions or function groups")

        build_result = await self._build_function(name=name, config=config)

        self._functions[name] = build_result

        return build_result.instance

    @override
    async def add_function_group(self, name: str | FunctionGroupRef, config: FunctionGroupBaseConfig) -> FunctionGroup:
        if isinstance(name, FunctionGroupRef):
            name = str(name)

        if (name in self._function_groups or name in self._functions):
            raise ValueError(f"Function group `{name}` already exists in the list of function groups or functions")

        # Build the function group
        build_result = await self._build_function_group(name=name, config=config)

        self._function_groups[name] = build_result

        # If the function group exposes functions, add them to the global function registry
        # If the function group exposes functions, record and add them to the registry
        included_functions = await build_result.instance.get_included_functions()
        for k in included_functions:
            if k in self._functions:
                raise ValueError(f"Exposed function `{k}` from group `{name}` conflicts with an existing function")
        self._functions.update({
            k: ConfiguredFunction(config=v.config, instance=v)
            for k, v in included_functions.items()
        })

        return build_result.instance

    @override
    async def get_function(self, name: str | FunctionRef) -> Function:
        if isinstance(name, FunctionRef):
            name = str(name)
        if name not in self._functions:
            raise ValueError(f"Function `{name}` not found")

        return self._functions[name].instance

    @override
    async def get_function_group(self, name: str | FunctionGroupRef) -> FunctionGroup:
        if isinstance(name, FunctionGroupRef):
            name = str(name)
        if name not in self._function_groups:
            raise ValueError(f"Function group `{name}` not found")

        return self._function_groups[name].instance

    @override
    def get_function_config(self, name: str | FunctionRef) -> FunctionBaseConfig:
        if isinstance(name, FunctionRef):
            name = str(name)
        if name not in self._functions:
            raise ValueError(f"Function `{name}` not found")

        return self._functions[name].config

    @override
    def get_function_group_config(self, name: str | FunctionGroupRef) -> FunctionGroupBaseConfig:
        if isinstance(name, FunctionGroupRef):
            name = str(name)
        if name not in self._function_groups:
            raise ValueError(f"Function group `{name}` not found")

        return self._function_groups[name].config

    @override
    async def set_workflow(self, config: FunctionBaseConfig) -> Function:

        if self._workflow is not None:
            warnings.warn("Overwriting existing workflow")

        build_result = await self._build_function(name="<workflow>", config=config)

        self._workflow = build_result

        return build_result.instance

    @override
    def get_workflow(self) -> Function:

        if self._workflow is None:
            raise ValueError("No workflow set")

        return self._workflow.instance

    @override
    def get_workflow_config(self) -> FunctionBaseConfig:
        if self._workflow is None:
            raise ValueError("No workflow set")

        return self._workflow.config

    @override
    def get_function_dependencies(self, fn_name: str | FunctionRef) -> FunctionDependencies:
        if isinstance(fn_name, FunctionRef):
            fn_name = str(fn_name)
        return self.function_dependencies[fn_name]

    @override
    def get_function_group_dependencies(self, fn_name: str | FunctionGroupRef) -> FunctionDependencies:
        if isinstance(fn_name, FunctionGroupRef):
            fn_name = str(fn_name)
        return self.function_group_dependencies[fn_name]

    @override
    async def get_tools(self,
                        tool_names: Sequence[str | FunctionRef | FunctionGroupRef],
                        wrapper_type: LLMFrameworkEnum | str) -> list[typing.Any]:

        unique = set(tool_names)
        if len(unique) != len(tool_names):
            raise ValueError("Tool names must be unique")

        async def _get_tools(n: str | FunctionRef | FunctionGroupRef):
            tools = []
            is_function_group_ref = isinstance(n, FunctionGroupRef)
            if isinstance(n, FunctionRef) or is_function_group_ref:
                n = str(n)
            if n not in self._function_groups:
                # the passed tool name is probably a function, but first check if it's a function group
                if is_function_group_ref:
                    raise ValueError(f"Function group `{n}` not found in the list of function groups")
                tools.append(await self.get_tool(n, wrapper_type))
            else:
                tool_wrapper_reg = self._registry.get_tool_wrapper(llm_framework=wrapper_type)
                current_function_group = self._function_groups[n]
                for fn_name, fn_instance in (await current_function_group.instance.get_accessible_functions()).items():
                    try:
                        tools.append(tool_wrapper_reg.build_fn(fn_name, fn_instance, self))
                    except Exception:
                        logger.error("Error fetching tool `%s`", fn_name, exc_info=True)
                        raise
            return tools

        tool_lists = await asyncio.gather(*[_get_tools(n) for n in tool_names])
        # Flatten the list of lists into a single list
        return [tool for tools in tool_lists for tool in tools]

    @override
    async def get_tool(self, fn_name: str | FunctionRef, wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        if isinstance(fn_name, FunctionRef):
            fn_name = str(fn_name)
        if fn_name not in self._functions:
            raise ValueError(f"Function `{fn_name}` not found in list of functions")
        fn = self._functions[fn_name]
        try:
            # Using the registry, get the tool wrapper for the requested framework
            tool_wrapper_reg = self._registry.get_tool_wrapper(llm_framework=wrapper_type)

            # Wrap in the correct wrapper
            return tool_wrapper_reg.build_fn(fn_name, fn.instance, self)
        except Exception as e:
            logger.error("Error fetching tool `%s`: %s", fn_name, e)
            raise

    @override
    async def add_llm(self, name: str | LLMRef, config: LLMBaseConfig) -> None:

        if (name in self._llms):
            raise ValueError(f"LLM `{name}` already exists in the list of LLMs")

        try:
            llm_info = self._registry.get_llm_provider(type(config))

            info_obj = await self._get_exit_stack().enter_async_context(llm_info.build_fn(config, self))

            self._llms[name] = ConfiguredLLM(config=config, instance=info_obj)
        except Exception as e:
            logger.error("Error adding llm `%s` with config `%s`: %s", name, config, e)
            raise

    @override
    async def get_llm(self, llm_name: str | LLMRef, wrapper_type: LLMFrameworkEnum | str) -> typing.Any:

        if (llm_name not in self._llms):
            raise ValueError(f"LLM `{llm_name}` not found")

        try:
            # Get llm info
            llm_info = self._llms[llm_name]

            # Generate wrapped client from registered client info
            client_info = self._registry.get_llm_client(config_type=type(llm_info.config), wrapper_type=wrapper_type)

            client = await self._get_exit_stack().enter_async_context(client_info.build_fn(llm_info.config, self))

            # Return a frameworks specific client
            return client
        except Exception as e:
            logger.error("Error getting llm `%s` with wrapper `%s`: %s", llm_name, wrapper_type, e)
            raise

    @override
    def get_llm_config(self, llm_name: str | LLMRef) -> LLMBaseConfig:

        if llm_name not in self._llms:
            raise ValueError(f"LLM `{llm_name}` not found")

        # Return the tool configuration object
        return self._llms[llm_name].config

    @experimental(feature_name="Authentication")
    @override
    async def add_auth_provider(self, name: str | AuthenticationRef,
                                config: AuthProviderBaseConfig) -> AuthProviderBase:
        """
        Add an authentication provider to the workflow by constructing it from a configuration object.

        Note: The Authentication Provider API is experimental and the API may change in future releases.

        Parameters
        ----------
        name : str | AuthenticationRef
            The name of the authentication provider to add.
        config : AuthProviderBaseConfig
            The configuration for the authentication provider.

        Returns
        -------
        AuthProviderBase
            The authentication provider instance.

        Raises
        ------
        ValueError
            If the authentication provider is already in the list of authentication providers.
        """

        if (name in self._auth_providers):
            raise ValueError(f"Authentication `{name}` already exists in the list of Authentication Providers")

        try:
            authentication_info = self._registry.get_auth_provider(type(config))

            info_obj = await self._get_exit_stack().enter_async_context(authentication_info.build_fn(config, self))

            self._auth_providers[name] = ConfiguredAuthProvider(config=config, instance=info_obj)

            return info_obj
        except Exception as e:
            logger.error("Error adding authentication `%s` with config `%s`: %s", name, config, e)
            raise

    @override
    async def get_auth_provider(self, auth_provider_name: str) -> AuthProviderBase:
        """
        Get the authentication provider instance for the given name.

        Note: The Authentication Provider API is experimental and the API may change in future releases.

        Parameters
        ----------
        auth_provider_name : str
            The name of the authentication provider to get.

        Returns
        -------
        AuthProviderBase
            The authentication provider instance.

        Raises
        ------
        ValueError
            If the authentication provider is not found.
        """

        if auth_provider_name not in self._auth_providers:
            raise ValueError(f"Authentication `{auth_provider_name}` not found")

        return self._auth_providers[auth_provider_name].instance

    @override
    async def add_embedder(self, name: str | EmbedderRef, config: EmbedderBaseConfig) -> None:

        if (name in self._embedders):
            raise ValueError(f"Embedder `{name}` already exists in the list of embedders")

        try:
            embedder_info = self._registry.get_embedder_provider(type(config))

            info_obj = await self._get_exit_stack().enter_async_context(embedder_info.build_fn(config, self))

            self._embedders[name] = ConfiguredEmbedder(config=config, instance=info_obj)
        except Exception as e:
            logger.error("Error adding embedder `%s` with config `%s`: %s", name, config, e)
            raise

    @override
    async def get_embedder(self, embedder_name: str | EmbedderRef, wrapper_type: LLMFrameworkEnum | str):

        if (embedder_name not in self._embedders):
            raise ValueError(f"Embedder `{embedder_name}` not found")

        try:
            # Get embedder info
            embedder_info = self._embedders[embedder_name]

            # Generate wrapped client from registered client info
            client_info = self._registry.get_embedder_client(config_type=type(embedder_info.config),
                                                             wrapper_type=wrapper_type)
            client = await self._get_exit_stack().enter_async_context(client_info.build_fn(embedder_info.config, self))

            # Return a frameworks specific client
            return client
        except Exception as e:
            logger.error("Error getting embedder `%s` with wrapper `%s`: %s", embedder_name, wrapper_type, e)
            raise

    @override
    def get_embedder_config(self, embedder_name: str | EmbedderRef) -> EmbedderBaseConfig:

        if embedder_name not in self._embedders:
            raise ValueError(f"Tool `{embedder_name}` not found")

        # Return the tool configuration object
        return self._embedders[embedder_name].config

    @override
    async def add_memory_client(self, name: str | MemoryRef, config: MemoryBaseConfig) -> MemoryEditor:

        if (name in self._memory_clients):
            raise ValueError(f"Memory `{name}` already exists in the list of memories")

        memory_info = self._registry.get_memory(type(config))

        info_obj = await self._get_exit_stack().enter_async_context(memory_info.build_fn(config, self))

        self._memory_clients[name] = ConfiguredMemory(config=config, instance=info_obj)

        return info_obj

    @override
    async def get_memory_client(self, memory_name: str | MemoryRef) -> MemoryEditor:
        """
        Return the instantiated memory client for the given name.
        """
        if memory_name not in self._memory_clients:
            raise ValueError(f"Memory `{memory_name}` not found")

        return self._memory_clients[memory_name].instance

    @override
    def get_memory_client_config(self, memory_name: str | MemoryRef) -> MemoryBaseConfig:

        if memory_name not in self._memory_clients:
            raise ValueError(f"Memory `{memory_name}` not found")

        # Return the tool configuration object
        return self._memory_clients[memory_name].config

    @override
    async def add_object_store(self, name: str | ObjectStoreRef, config: ObjectStoreBaseConfig) -> ObjectStore:
        if name in self._object_stores:
            raise ValueError(f"Object store `{name}` already exists in the list of object stores")

        object_store_info = self._registry.get_object_store(type(config))

        info_obj = await self._get_exit_stack().enter_async_context(object_store_info.build_fn(config, self))

        self._object_stores[name] = ConfiguredObjectStore(config=config, instance=info_obj)

        return info_obj

    @override
    async def get_object_store_client(self, object_store_name: str | ObjectStoreRef) -> ObjectStore:
        if object_store_name not in self._object_stores:
            raise ValueError(f"Object store `{object_store_name}` not found")

        return self._object_stores[object_store_name].instance

    @override
    def get_object_store_config(self, object_store_name: str | ObjectStoreRef) -> ObjectStoreBaseConfig:
        if object_store_name not in self._object_stores:
            raise ValueError(f"Object store `{object_store_name}` not found")

        return self._object_stores[object_store_name].config

    @override
    async def add_retriever(self, name: str | RetrieverRef, config: RetrieverBaseConfig) -> None:

        if (name in self._retrievers):
            raise ValueError(f"Retriever '{name}' already exists in the list of retrievers")

        try:
            retriever_info = self._registry.get_retriever_provider(type(config))

            info_obj = await self._get_exit_stack().enter_async_context(retriever_info.build_fn(config, self))

            self._retrievers[name] = ConfiguredRetriever(config=config, instance=info_obj)

        except Exception as e:
            logger.error("Error adding retriever `%s` with config `%s`: %s", name, config, e)
            raise

    @override
    async def get_retriever(self,
                            retriever_name: str | RetrieverRef,
                            wrapper_type: LLMFrameworkEnum | str | None = None):

        if retriever_name not in self._retrievers:
            raise ValueError(f"Retriever '{retriever_name}' not found")

        try:
            # Get retriever info
            retriever_info = self._retrievers[retriever_name]

            # Generate wrapped client from registered client info
            client_info = self._registry.get_retriever_client(config_type=type(retriever_info.config),
                                                              wrapper_type=wrapper_type)

            client = await self._get_exit_stack().enter_async_context(client_info.build_fn(retriever_info.config, self))

            # Return a frameworks specific client
            return client
        except Exception as e:
            logger.error("Error getting retriever `%s` with wrapper `%s`: %s", retriever_name, wrapper_type, e)
            raise

    @override
    async def get_retriever_config(self, retriever_name: str | RetrieverRef) -> RetrieverBaseConfig:

        if retriever_name not in self._retrievers:
            raise ValueError(f"Retriever `{retriever_name}` not found")

        return self._retrievers[retriever_name].config

    @override
    @experimental(feature_name="TTC")
    async def add_ttc_strategy(self, name: str | TTCStrategyRef, config: TTCStrategyBaseConfig) -> None:
        if (name in self._ttc_strategies):
            raise ValueError(f"TTC strategy '{name}' already exists in the list of TTC strategies")

        try:
            ttc_strategy_info = self._registry.get_ttc_strategy(type(config))

            info_obj = await self._get_exit_stack().enter_async_context(ttc_strategy_info.build_fn(config, self))

            self._ttc_strategies[name] = ConfiguredTTCStrategy(config=config, instance=info_obj)

        except Exception as e:
            logger.error("Error adding TTC strategy `%s` with config `%s`: %s", name, config, e)
            raise

    @override
    async def get_ttc_strategy(self,
                               strategy_name: str | TTCStrategyRef,
                               pipeline_type: PipelineTypeEnum,
                               stage_type: StageTypeEnum) -> StrategyBase:

        if strategy_name not in self._ttc_strategies:
            raise ValueError(f"TTC strategy '{strategy_name}' not found")

        try:
            # Get strategy info
            ttc_strategy_info = self._ttc_strategies[strategy_name]

            instance = ttc_strategy_info.instance

            if not stage_type == instance.stage_type():
                raise ValueError(f"TTC strategy '{strategy_name}' is not compatible with stage type '{stage_type}'")

            if pipeline_type not in instance.supported_pipeline_types():
                raise ValueError(
                    f"TTC strategy '{strategy_name}' is not compatible with pipeline type '{pipeline_type}'")

            instance.set_pipeline_type(pipeline_type)

            return instance
        except Exception as e:
            logger.error("Error getting TTC strategy `%s`: %s", strategy_name, e)
            raise

    @override
    async def get_ttc_strategy_config(self,
                                      strategy_name: str | TTCStrategyRef,
                                      pipeline_type: PipelineTypeEnum,
                                      stage_type: StageTypeEnum) -> TTCStrategyBaseConfig:
        if strategy_name not in self._ttc_strategies:
            raise ValueError(f"TTC strategy '{strategy_name}' not found")

        strategy_info = self._ttc_strategies[strategy_name]
        instance = strategy_info.instance
        config = strategy_info.config

        if not stage_type == instance.stage_type():
            raise ValueError(f"TTC strategy '{strategy_name}' is not compatible with stage type '{stage_type}'")

        if pipeline_type not in instance.supported_pipeline_types():
            raise ValueError(f"TTC strategy '{strategy_name}' is not compatible with pipeline type '{pipeline_type}'")

        return config

    @override
    async def add_middleware(self, name: str | MiddlewareRef, config: MiddlewareBaseConfig) -> Middleware:
        """Add middleware to the builder.

        Args:
            name: The name or reference for the middleware
            config: The configuration for the middleware

        Returns:
            The built middleware instance

        Raises:
            ValueError: If the middleware already exists
        """
        if name in self._middleware:
            raise ValueError(f"Middleware `{name}` already exists in the list of middleware")

        try:
            middleware_info = self._registry.get_middleware(type(config))

            middleware_instance = await self._get_exit_stack().enter_async_context(
                middleware_info.build_fn(config, self))

            self._middleware[name] = ConfiguredMiddleware(config=config, instance=middleware_instance)

            return middleware_instance
        except Exception as e:
            logger.error("Error adding function middleware `%s` with config `%s`: %s", name, config, e)
            raise

    @override
    async def get_middleware(self, middleware_name: str | MiddlewareRef) -> Middleware:
        """Get built middleware by name.

        Args:
            middleware_name: The name or reference of the middleware

        Returns:
            The built middleware instance

        Raises:
            ValueError: If the middleware is not found
        """
        if middleware_name not in self._middleware:
            raise ValueError(f"Middleware `{middleware_name}` not found")

        return self._middleware[middleware_name].instance

    @override
    def get_middleware_config(self, middleware_name: str | MiddlewareRef) -> MiddlewareBaseConfig:
        """Get the configuration for middleware.

        Args:
            middleware_name: The name or reference of the middleware

        Returns:
            The configuration for the middleware

        Raises:
            ValueError: If the middleware is not found
        """
        if middleware_name not in self._middleware:
            raise ValueError(f"Middleware `{middleware_name}` not found")

        return self._middleware[middleware_name].config

    @override
    def get_user_manager(self):
        return UserManagerHolder(context=Context(self._context_state))

    async def add_telemetry_exporter(self, name: str, config: TelemetryExporterBaseConfig) -> None:
        """Add an configured telemetry exporter to the builder.

        Args:
            name (str): The name of the telemetry exporter
            config (TelemetryExporterBaseConfig): The configuration for the exporter
        """
        if (name in self._telemetry_exporters):
            raise ValueError(f"Telemetry exporter '{name}' already exists in the list of telemetry exporters")

        exporter_info = self._registry.get_telemetry_exporter(type(config))

        # Build the exporter outside the lock (parallel)
        exporter_context_manager = exporter_info.build_fn(config, self)

        # Only protect the shared state modifications (serialized)
        exporter = await self._get_exit_stack().enter_async_context(exporter_context_manager)
        self._telemetry_exporters[name] = ConfiguredTelemetryExporter(config=config, instance=exporter)

    def _log_build_failure(self,
                           component_name: str,
                           component_type: str,
                           completed_components: list[tuple[str, str]],
                           remaining_components: list[tuple[str, str]],
                           original_error: Exception) -> None:
        """
        Common method to log comprehensive build failure information.

        Args:
            component_name (str): The name of the component that failed to build
            component_type (str): The type of the component that failed to build
            completed_components (list[tuple[str, str]]): List of (name, type) tuples for successfully built components
            remaining_components (list[tuple[str, str]]): List of (name, type) tuples for components still to be built
            original_error (Exception): The original exception that caused the failure
        """
        logger.error("Failed to initialize component %s (%s)", component_name, component_type)

        if completed_components:
            logger.error("Successfully built components:")
            for name, comp_type in completed_components:
                logger.error("- %s (%s)", name, comp_type)
        else:
            logger.error("No components were successfully built before this failure")

        if remaining_components:
            logger.error("Remaining components to build:")
            for name, comp_type in remaining_components:
                logger.error("- %s (%s)", name, comp_type)
        else:
            logger.error("No remaining components to build")

        logger.error("Original error: %s", original_error, exc_info=True)

    def _log_build_failure_component(self,
                                     failing_component: ComponentInstanceData,
                                     completed_components: list[tuple[str, str]],
                                     remaining_components: list[tuple[str, str]],
                                     original_error: Exception) -> None:
        """
        Log comprehensive component build failure information.

        Args:
            failing_component (ComponentInstanceData): The ComponentInstanceData that failed to build
            completed_components (list[tuple[str, str]]): List of (name, type) tuples for successfully built components
            remaining_components (list[tuple[str, str]]): List of (name, type) tuples for components still to be built
            original_error (Exception): The original exception that caused the failure
        """
        component_name = failing_component.name
        component_type = failing_component.component_group.value

        self._log_build_failure(component_name,
                                component_type,
                                completed_components,
                                remaining_components,
                                original_error)

    def _log_build_failure_workflow(self,
                                    completed_components: list[tuple[str, str]],
                                    remaining_components: list[tuple[str, str]],
                                    original_error: Exception) -> None:
        """
        Log comprehensive workflow build failure information.

        Args:
            completed_components (list[tuple[str, str]]): List of (name, type) tuples for successfully built components
            remaining_components (list[tuple[str, str]]): List of (name, type) tuples for components still to be built
            original_error (Exception): The original exception that caused the failure
        """
        self._log_build_failure("<workflow>", "workflow", completed_components, remaining_components, original_error)

    async def populate_builder(self, config: Config, skip_workflow: bool = False):
        """
        Populate the builder with components and optionally set up the workflow.

        Args:
            config (Config): The configuration object containing component definitions.
            skip_workflow (bool): If True, skips the workflow instantiation step. Defaults to False.

        """
        # Generate the build sequence
        build_sequence = build_dependency_sequence(config)

        # Initialize progress tracking
        completed_components = []
        remaining_components = [(str(comp.name), comp.component_group.value) for comp in build_sequence
                                if not comp.is_root]
        if not skip_workflow:
            remaining_components.append(("<workflow>", "workflow"))

        # Loop over all objects and add to the workflow builder
        for component_instance in build_sequence:
            try:
                # Remove from remaining as we start building (if not root)
                if not component_instance.is_root:
                    remaining_components.remove(
                        (str(component_instance.name), component_instance.component_group.value))

                # Instantiate a the llm
                if component_instance.component_group == ComponentGroup.LLMS:
                    await self.add_llm(component_instance.name, cast(LLMBaseConfig, component_instance.config))
                # Instantiate a the embedder
                elif component_instance.component_group == ComponentGroup.EMBEDDERS:
                    await self.add_embedder(component_instance.name,
                                            cast(EmbedderBaseConfig, component_instance.config))
                # Instantiate a memory client
                elif component_instance.component_group == ComponentGroup.MEMORY:
                    await self.add_memory_client(component_instance.name,
                                                 cast(MemoryBaseConfig, component_instance.config))
                # Instantiate a object store client
                elif component_instance.component_group == ComponentGroup.OBJECT_STORES:
                    await self.add_object_store(component_instance.name,
                                                cast(ObjectStoreBaseConfig, component_instance.config))
                # Instantiate a retriever client
                elif component_instance.component_group == ComponentGroup.RETRIEVERS:
                    await self.add_retriever(component_instance.name,
                                             cast(RetrieverBaseConfig, component_instance.config))
                # Instantiate middleware
                elif component_instance.component_group == ComponentGroup.MIDDLEWARE:
                    await self.add_middleware(component_instance.name,
                                              cast(MiddlewareBaseConfig, component_instance.config))
                # Instantiate a function group
                elif component_instance.component_group == ComponentGroup.FUNCTION_GROUPS:
                    await self.add_function_group(component_instance.name,
                                                  cast(FunctionGroupBaseConfig, component_instance.config))
                # Instantiate a function
                elif component_instance.component_group == ComponentGroup.FUNCTIONS:
                    # If the function is the root, set it as the workflow later
                    if (not component_instance.is_root):
                        await self.add_function(component_instance.name,
                                                cast(FunctionBaseConfig, component_instance.config))
                elif component_instance.component_group == ComponentGroup.TTC_STRATEGIES:
                    await self.add_ttc_strategy(component_instance.name,
                                                cast(TTCStrategyBaseConfig, component_instance.config))

                elif component_instance.component_group == ComponentGroup.AUTHENTICATION:
                    await self.add_auth_provider(component_instance.name,
                                                 cast(AuthProviderBaseConfig, component_instance.config))
                else:
                    raise ValueError(f"Unknown component group {component_instance.component_group}")

                # Add to completed after successful build (if not root)
                if not component_instance.is_root:
                    completed_components.append(
                        (str(component_instance.name), component_instance.component_group.value))

            except Exception as e:
                self._log_build_failure_component(component_instance, completed_components, remaining_components, e)
                raise

        # Instantiate the workflow
        if not skip_workflow:
            try:
                # Remove workflow from remaining as we start building
                remaining_components.remove(("<workflow>", "workflow"))
                await self.set_workflow(config.workflow)
                completed_components.append(("<workflow>", "workflow"))
            except Exception as e:
                self._log_build_failure_workflow(completed_components, remaining_components, e)
                raise

    @classmethod
    @asynccontextmanager
    async def from_config(cls, config: Config):

        async with cls(general_config=config.general) as builder:
            await builder.populate_builder(config)
            yield builder


class ChildBuilder(Builder):

    def __init__(self, workflow_builder: WorkflowBuilder) -> None:

        self._workflow_builder = workflow_builder

        self._dependencies = FunctionDependencies()

    @property
    def dependencies(self) -> FunctionDependencies:
        return self._dependencies

    @override
    async def add_function(self, name: str, config: FunctionBaseConfig) -> Function:
        return await self._workflow_builder.add_function(name, config)

    @override
    async def add_function_group(self, name: str, config: FunctionGroupBaseConfig) -> FunctionGroup:
        return await self._workflow_builder.add_function_group(name, config)

    @override
    async def get_function(self, name: str) -> Function:
        # If a function tries to get another function, we assume it uses it
        fn = await self._workflow_builder.get_function(name)

        self._dependencies.add_function(name)

        return fn

    @override
    async def get_function_group(self, name: str) -> FunctionGroup:
        # If a function tries to get a function group, we assume it uses it
        function_group = await self._workflow_builder.get_function_group(name)

        self._dependencies.add_function_group(name)

        return function_group

    @override
    def get_function_config(self, name: str) -> FunctionBaseConfig:
        return self._workflow_builder.get_function_config(name)

    @override
    def get_function_group_config(self, name: str) -> FunctionGroupBaseConfig:
        return self._workflow_builder.get_function_group_config(name)

    @override
    async def set_workflow(self, config: FunctionBaseConfig) -> Function:
        return await self._workflow_builder.set_workflow(config)

    @override
    def get_workflow(self) -> Function:
        return self._workflow_builder.get_workflow()

    @override
    def get_workflow_config(self) -> FunctionBaseConfig:
        return self._workflow_builder.get_workflow_config()

    @override
    async def get_tools(self,
                        tool_names: Sequence[str | FunctionRef | FunctionGroupRef],
                        wrapper_type: LLMFrameworkEnum | str) -> list[typing.Any]:
        tools = await self._workflow_builder.get_tools(tool_names, wrapper_type)
        for tool_name in tool_names:
            if tool_name in self._workflow_builder._function_groups:
                self._dependencies.add_function_group(tool_name)
            else:
                self._dependencies.add_function(tool_name)
        return tools

    @override
    async def get_tool(self, fn_name: str | FunctionRef, wrapper_type: LLMFrameworkEnum | str):
        # If a function tries to get another function as a tool, we assume it uses it
        fn = await self._workflow_builder.get_tool(fn_name, wrapper_type)

        self._dependencies.add_function(fn_name)

        return fn

    @override
    async def add_llm(self, name: str, config: LLMBaseConfig) -> None:
        return await self._workflow_builder.add_llm(name, config)

    @experimental(feature_name="Authentication")
    @override
    async def add_auth_provider(self, name: str, config: AuthProviderBaseConfig) -> AuthProviderBase:
        return await self._workflow_builder.add_auth_provider(name, config)

    @override
    async def get_auth_provider(self, auth_provider_name: str):
        return await self._workflow_builder.get_auth_provider(auth_provider_name)

    @override
    async def get_llm(self, llm_name: str, wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        llm = await self._workflow_builder.get_llm(llm_name, wrapper_type)

        self._dependencies.add_llm(llm_name)

        return llm

    @override
    def get_llm_config(self, llm_name: str) -> LLMBaseConfig:
        return self._workflow_builder.get_llm_config(llm_name)

    @override
    async def add_embedder(self, name: str, config: EmbedderBaseConfig) -> None:
        await self._workflow_builder.add_embedder(name, config)

    @override
    async def get_embedder(self, embedder_name: str, wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        embedder = await self._workflow_builder.get_embedder(embedder_name, wrapper_type)

        self._dependencies.add_embedder(embedder_name)

        return embedder

    @override
    def get_embedder_config(self, embedder_name: str) -> EmbedderBaseConfig:
        return self._workflow_builder.get_embedder_config(embedder_name)

    @override
    async def add_memory_client(self, name: str, config: MemoryBaseConfig) -> MemoryEditor:
        return await self._workflow_builder.add_memory_client(name, config)

    @override
    async def get_memory_client(self, memory_name: str) -> MemoryEditor:
        """
        Return the instantiated memory client for the given name.
        """
        memory_client = await self._workflow_builder.get_memory_client(memory_name)

        self._dependencies.add_memory_client(memory_name)

        return memory_client

    @override
    def get_memory_client_config(self, memory_name: str) -> MemoryBaseConfig:
        return self._workflow_builder.get_memory_client_config(memory_name=memory_name)

    @override
    async def add_object_store(self, name: str, config: ObjectStoreBaseConfig):
        return await self._workflow_builder.add_object_store(name, config)

    @override
    async def get_object_store_client(self, object_store_name: str) -> ObjectStore:
        """
        Return the instantiated object store client for the given name.
        """
        object_store_client = await self._workflow_builder.get_object_store_client(object_store_name)

        self._dependencies.add_object_store(object_store_name)

        return object_store_client

    @override
    def get_object_store_config(self, object_store_name: str) -> ObjectStoreBaseConfig:
        return self._workflow_builder.get_object_store_config(object_store_name)

    @override
    @experimental(feature_name="TTC")
    async def add_ttc_strategy(self, name: str, config: TTCStrategyBaseConfig) -> None:
        await self._workflow_builder.add_ttc_strategy(name, config)

    @override
    async def get_ttc_strategy(self,
                               strategy_name: str | TTCStrategyRef,
                               pipeline_type: PipelineTypeEnum,
                               stage_type: StageTypeEnum) -> StrategyBase:
        return await self._workflow_builder.get_ttc_strategy(strategy_name=strategy_name,
                                                             pipeline_type=pipeline_type,
                                                             stage_type=stage_type)

    @override
    async def get_ttc_strategy_config(self,
                                      strategy_name: str | TTCStrategyRef,
                                      pipeline_type: PipelineTypeEnum,
                                      stage_type: StageTypeEnum) -> TTCStrategyBaseConfig:
        return await self._workflow_builder.get_ttc_strategy_config(strategy_name=strategy_name,
                                                                    pipeline_type=pipeline_type,
                                                                    stage_type=stage_type)

    @override
    async def add_retriever(self, name: str, config: RetrieverBaseConfig) -> None:
        await self._workflow_builder.add_retriever(name, config)

    @override
    async def get_retriever(self, retriever_name: str, wrapper_type: LLMFrameworkEnum | str | None = None) -> Retriever:
        if not wrapper_type:
            return await self._workflow_builder.get_retriever(retriever_name=retriever_name)
        return await self._workflow_builder.get_retriever(retriever_name=retriever_name, wrapper_type=wrapper_type)

    @override
    async def get_retriever_config(self, retriever_name: str) -> RetrieverBaseConfig:
        return await self._workflow_builder.get_retriever_config(retriever_name=retriever_name)

    @override
    def get_user_manager(self) -> UserManagerHolder:
        return self._workflow_builder.get_user_manager()

    @override
    def get_function_dependencies(self, fn_name: str) -> FunctionDependencies:
        return self._workflow_builder.get_function_dependencies(fn_name)

    @override
    def get_function_group_dependencies(self, fn_name: str) -> FunctionDependencies:
        return self._workflow_builder.get_function_group_dependencies(fn_name)

    @override
    async def add_middleware(self, name: str | MiddlewareRef, config: MiddlewareBaseConfig) -> Middleware:
        """Add middleware to the builder."""
        return await self._workflow_builder.add_middleware(name, config)

    @override
    async def get_middleware(self, middleware_name: str | MiddlewareRef) -> Middleware:
        """Get built middleware by name."""
        return await self._workflow_builder.get_middleware(middleware_name)

    @override
    def get_middleware_config(self, middleware_name: str | MiddlewareRef) -> MiddlewareBaseConfig:
        """Get the configuration for middleware."""
        return self._workflow_builder.get_middleware_config(middleware_name)
