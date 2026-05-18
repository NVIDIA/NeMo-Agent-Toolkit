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
"""

from nat.builder.builder import Builder
from nat.builder.builder import EvalBuilder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.builder.function import FunctionGroup
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_auth_provider
from nat.cli.register_workflow import register_dataset_loader
from nat.cli.register_workflow import register_embedder_client
from nat.cli.register_workflow import register_embedder_provider
from nat.cli.register_workflow import register_eval_callback
from nat.cli.register_workflow import register_evaluator
from nat.cli.register_workflow import register_front_end
from nat.cli.register_workflow import register_function
from nat.cli.register_workflow import register_function_group
from nat.cli.register_workflow import register_llm_client
from nat.cli.register_workflow import register_llm_provider
from nat.cli.register_workflow import register_logging_method
from nat.cli.register_workflow import register_memory
from nat.cli.register_workflow import register_middleware
from nat.cli.register_workflow import register_object_store
from nat.cli.register_workflow import register_optimizer
from nat.cli.register_workflow import register_optimizer_callback
from nat.cli.register_workflow import register_per_user_function
from nat.cli.register_workflow import register_per_user_function_group
from nat.cli.register_workflow import register_registry_handler
from nat.cli.register_workflow import register_retriever_client
from nat.cli.register_workflow import register_retriever_provider
from nat.cli.register_workflow import register_telemetry_exporter
from nat.cli.register_workflow import register_tool_wrapper
from nat.cli.register_workflow import register_trainer
from nat.cli.register_workflow import register_trainer_adapter
from nat.cli.register_workflow import register_trajectory_builder
from nat.cli.register_workflow import register_ttc_strategy
from nat.data_models.common import OptionalSecretStr
from nat.data_models.common import SerializableSecretStr
from nat.data_models.common import get_secret_value
from nat.data_models.common import set_secret_from_env
from nat.data_models.component_ref import AuthenticationRef
from nat.data_models.component_ref import ComponentRef
from nat.data_models.component_ref import EmbedderRef
from nat.data_models.component_ref import FunctionGroupRef
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.component_ref import MemoryRef
from nat.data_models.component_ref import MiddlewareRef
from nat.data_models.component_ref import ObjectStoreRef
from nat.data_models.component_ref import RetrieverRef
from nat.data_models.component_ref import TrainerAdapterRef
from nat.data_models.component_ref import TrainerRef
from nat.data_models.component_ref import TrajectoryBuilderRef
from nat.data_models.component_ref import TTCStrategyRef
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.function import FunctionGroupBaseConfig

__all__ = [
    "AuthenticationRef",
    "Builder",
    "ComponentRef",
    "EmbedderRef",
    "EvalBuilder",
    "Function",
    "FunctionBaseConfig",
    "FunctionGroup",
    "FunctionGroupBaseConfig",
    "FunctionGroupRef",
    "FunctionInfo",
    "FunctionRef",
    "LLMFrameworkEnum",
    "LLMRef",
    "MemoryRef",
    "MiddlewareRef",
    "ObjectStoreRef",
    "OptionalSecretStr",
    "RetrieverRef",
    "SerializableSecretStr",
    "TTCStrategyRef",
    "TrainerAdapterRef",
    "TrainerRef",
    "TrajectoryBuilderRef",
    "get_secret_value",
    "register_auth_provider",
    "register_dataset_loader",
    "register_embedder_client",
    "register_embedder_provider",
    "register_eval_callback",
    "register_evaluator",
    "register_front_end",
    "register_function",
    "register_function_group",
    "register_llm_client",
    "register_llm_provider",
    "register_logging_method",
    "register_memory",
    "register_middleware",
    "register_object_store",
    "register_optimizer",
    "register_optimizer_callback",
    "register_per_user_function",
    "register_per_user_function_group",
    "register_registry_handler",
    "register_retriever_client",
    "register_retriever_provider",
    "register_telemetry_exporter",
    "register_tool_wrapper",
    "register_trainer",
    "register_trainer_adapter",
    "register_trajectory_builder",
    "register_ttc_strategy",
    "set_secret_from_env",
]
