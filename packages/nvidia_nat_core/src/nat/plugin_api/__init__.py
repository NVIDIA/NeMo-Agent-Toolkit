# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Stable public API for NeMo Agent Toolkit plugin authors.

External plugin packages should prefer importing from this module instead of depending on implementation-oriented
modules such as ``nat.cli.register_workflow`` or ``nat.builder.function`` directly.

The ``Builder`` class is re-exported because plugin callables are typed around it, but only a subset of its methods
is part of the stable contract. The following ``Builder`` methods belong to subsystems that are intentionally
deferred from the public plugin API (see ``DEFERRED_PLUGIN_API_CANDIDATES`` in the plugin API tests) and may change
without notice; plugin authors must not depend on them:

* Auth providers: ``add_auth_provider``, ``get_auth_provider``, ``get_auth_providers``.
* Finetuning: ``add_trainer``, ``add_trainer_adapter``, ``add_trajectory_builder``, ``get_trainer``,
  ``get_trainer_adapter``, ``get_trajectory_builder``, ``get_trainer_config``, ``get_trainer_adapter_config``,
  ``get_trajectory_builder_config``.
* Test-time compute: ``add_ttc_strategy``, ``get_ttc_strategy``, ``get_ttc_strategy_config``.

``test_builder_stable_surface_is_explicit`` pins the full method partition, so any new method added to ``Builder``
must be explicitly categorized as stable or deferred.
"""

from nat.builder.builder import Builder
from nat.builder.builder import EvalBuilder
from nat.builder.dataset_loader import DatasetLoaderInfo
from nat.builder.embedder import EmbedderProviderInfo
from nat.builder.evaluator import EvaluatorInfo
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.builder.function import FunctionGroup
from nat.builder.function_info import FunctionInfo
from nat.builder.llm import LLMProviderInfo
from nat.builder.retriever import RetrieverProviderInfo
from nat.cli.register_workflow import register_dataset_loader
from nat.cli.register_workflow import register_embedder_client
from nat.cli.register_workflow import register_embedder_provider
from nat.cli.register_workflow import register_eval_callback
from nat.cli.register_workflow import register_evaluator
from nat.cli.register_workflow import register_function
from nat.cli.register_workflow import register_function_group
from nat.cli.register_workflow import register_llm_client
from nat.cli.register_workflow import register_llm_provider
from nat.cli.register_workflow import register_memory
from nat.cli.register_workflow import register_middleware
from nat.cli.register_workflow import register_object_store
from nat.cli.register_workflow import register_per_user_function
from nat.cli.register_workflow import register_per_user_function_group
from nat.cli.register_workflow import register_retriever_client
from nat.cli.register_workflow import register_retriever_provider
from nat.cli.register_workflow import register_telemetry_exporter
from nat.cli.register_workflow import register_tool_wrapper
from nat.data_models.common import OptionalSecretStr
from nat.data_models.common import SerializableSecretStr
from nat.data_models.common import get_secret_value
from nat.data_models.common import set_secret_from_env
from nat.data_models.component_ref import ComponentRef
from nat.data_models.component_ref import EmbedderRef
from nat.data_models.component_ref import FunctionGroupRef
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.component_ref import MemoryRef
from nat.data_models.component_ref import MiddlewareRef
from nat.data_models.component_ref import ObjectStoreRef
from nat.data_models.component_ref import RetrieverRef
from nat.data_models.dataset_handler import EvalDatasetBaseConfig
from nat.data_models.embedder import EmbedderBaseConfig
from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.function import FunctionGroupBaseConfig
from nat.data_models.interactive import HumanPrompt
from nat.data_models.interactive import InteractionResponse
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.memory import MemoryBaseConfig
from nat.data_models.middleware import FunctionMiddlewareBaseConfig
from nat.data_models.middleware import MiddlewareBaseConfig
from nat.data_models.object_store import KeyAlreadyExistsError
from nat.data_models.object_store import NoSuchKeyError
from nat.data_models.object_store import ObjectStoreBaseConfig
from nat.data_models.retriever import RetrieverBaseConfig
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.memory.interfaces import MemoryEditor
from nat.memory.interfaces import MemoryManager
from nat.memory.interfaces import MemoryReader
from nat.memory.interfaces import MemoryWriter
from nat.memory.models import MemoryItem
from nat.middleware.dynamic.dynamic_function_middleware import DynamicFunctionMiddleware
from nat.middleware.dynamic.dynamic_middleware_config import DynamicMiddlewareConfig
from nat.middleware.function_middleware import FunctionMiddleware
from nat.middleware.hitl.hitl_middleware import HITLMiddleware
from nat.middleware.hitl.hitl_middleware_config import HITLMiddlewareConfig
from nat.middleware.middleware import FunctionMiddlewareContext
from nat.middleware.middleware import InvocationAction
from nat.middleware.middleware import InvocationContext
from nat.object_store.interfaces import ObjectStore
from nat.object_store.models import ObjectStoreItem
from nat.retriever.interface import Retriever
from nat.retriever.models import Document
from nat.retriever.models import RetrieverOutput

# Public contract: keep this list exact and update docs/source/extend/plugin-api.md plus
# packages/nvidia_nat_core/tests/nat/test_plugin_api.py whenever symbols are added or removed.
__all__ = [
    "Builder",
    "ComponentRef",
    "DatasetLoaderInfo",
    "Document",
    "DynamicFunctionMiddleware",
    "DynamicMiddlewareConfig",
    "EmbedderBaseConfig",
    "EmbedderProviderInfo",
    "EmbedderRef",
    "EvalBuilder",
    "EvalDatasetBaseConfig",
    "EvaluatorBaseConfig",
    "EvaluatorInfo",
    "Function",
    "FunctionBaseConfig",
    "FunctionGroup",
    "FunctionGroupBaseConfig",
    "FunctionGroupRef",
    "FunctionInfo",
    "FunctionRef",
    "FunctionMiddleware",
    "FunctionMiddlewareBaseConfig",
    "FunctionMiddlewareContext",
    "HITLMiddleware",
    "HITLMiddlewareConfig",
    "HumanPrompt",
    "InteractionResponse",
    "InvocationAction",
    "InvocationContext",
    "KeyAlreadyExistsError",
    "LLMBaseConfig",
    "LLMFrameworkEnum",
    "LLMProviderInfo",
    "LLMRef",
    "MemoryBaseConfig",
    "MemoryEditor",
    "MemoryItem",
    "MemoryManager",
    "MemoryReader",
    "MemoryRef",
    "MemoryWriter",
    "MiddlewareBaseConfig",
    "MiddlewareRef",
    "NoSuchKeyError",
    "ObjectStore",
    "ObjectStoreRef",
    "ObjectStoreItem",
    "ObjectStoreBaseConfig",
    "OptionalSecretStr",
    "Retriever",
    "RetrieverBaseConfig",
    "RetrieverOutput",
    "RetrieverProviderInfo",
    "RetrieverRef",
    "SerializableSecretStr",
    "TelemetryExporterBaseConfig",
    "get_secret_value",
    "register_dataset_loader",
    "register_embedder_client",
    "register_embedder_provider",
    "register_eval_callback",
    "register_evaluator",
    "register_function",
    "register_function_group",
    "register_llm_client",
    "register_llm_provider",
    "register_memory",
    "register_middleware",
    "register_object_store",
    "register_per_user_function",
    "register_per_user_function_group",
    "register_retriever_client",
    "register_retriever_provider",
    "register_telemetry_exporter",
    "register_tool_wrapper",
    "set_secret_from_env",
]
