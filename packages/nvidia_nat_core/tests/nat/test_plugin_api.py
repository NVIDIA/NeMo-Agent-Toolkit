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

import importlib
from pathlib import Path

from nat import plugin_api

EXPECTED_PLUGIN_API_EXPORTS = {
    "AuthProviderBaseConfig": ("nat.data_models.authentication", "AuthProviderBaseConfig"),
    "AuthenticationRef": ("nat.data_models.component_ref", "AuthenticationRef"),
    "Builder": ("nat.builder.builder", "Builder"),
    "ComponentRef": ("nat.data_models.component_ref", "ComponentRef"),
    "DatasetLoaderInfo": ("nat.builder.dataset_loader", "DatasetLoaderInfo"),
    "DynamicFunctionMiddleware": ("nat.middleware.dynamic.dynamic_function_middleware", "DynamicFunctionMiddleware"),
    "DynamicMiddlewareConfig": ("nat.middleware.dynamic.dynamic_middleware_config", "DynamicMiddlewareConfig"),
    "EmbedderBaseConfig": ("nat.data_models.embedder", "EmbedderBaseConfig"),
    "EmbedderProviderInfo": ("nat.builder.embedder", "EmbedderProviderInfo"),
    "EmbedderRef": ("nat.data_models.component_ref", "EmbedderRef"),
    "EvalBuilder": ("nat.builder.builder", "EvalBuilder"),
    "EvalDatasetBaseConfig": ("nat.data_models.dataset_handler", "EvalDatasetBaseConfig"),
    "EvaluatorBaseConfig": ("nat.data_models.evaluator", "EvaluatorBaseConfig"),
    "EvaluatorInfo": ("nat.builder.evaluator", "EvaluatorInfo"),
    "FrontEndBaseConfig": ("nat.data_models.front_end", "FrontEndBaseConfig"),
    "Function": ("nat.builder.function", "Function"),
    "FunctionBaseConfig": ("nat.data_models.function", "FunctionBaseConfig"),
    "FunctionGroup": ("nat.builder.function", "FunctionGroup"),
    "FunctionGroupBaseConfig": ("nat.data_models.function", "FunctionGroupBaseConfig"),
    "FunctionGroupRef": ("nat.data_models.component_ref", "FunctionGroupRef"),
    "FunctionInfo": ("nat.builder.function_info", "FunctionInfo"),
    "FunctionRef": ("nat.data_models.component_ref", "FunctionRef"),
    "FunctionMiddleware": ("nat.middleware.function_middleware", "FunctionMiddleware"),
    "FunctionMiddlewareBaseConfig": ("nat.data_models.middleware", "FunctionMiddlewareBaseConfig"),
    "FunctionMiddlewareContext": ("nat.middleware.middleware", "FunctionMiddlewareContext"),
    "InvocationContext": ("nat.middleware.middleware", "InvocationContext"),
    "KeyAlreadyExistsError": ("nat.data_models.object_store", "KeyAlreadyExistsError"),
    "LLMBaseConfig": ("nat.data_models.llm", "LLMBaseConfig"),
    "LLMFrameworkEnum": ("nat.builder.framework_enum", "LLMFrameworkEnum"),
    "LLMProviderInfo": ("nat.builder.llm", "LLMProviderInfo"),
    "LLMRef": ("nat.data_models.component_ref", "LLMRef"),
    "LoggingBaseConfig": ("nat.data_models.logging", "LoggingBaseConfig"),
    "MemoryBaseConfig": ("nat.data_models.memory", "MemoryBaseConfig"),
    "MemoryEditor": ("nat.memory.interfaces", "MemoryEditor"),
    "MemoryItem": ("nat.memory.models", "MemoryItem"),
    "MemoryManager": ("nat.memory.interfaces", "MemoryManager"),
    "MemoryReader": ("nat.memory.interfaces", "MemoryReader"),
    "MemoryRef": ("nat.data_models.component_ref", "MemoryRef"),
    "MemoryWriter": ("nat.memory.interfaces", "MemoryWriter"),
    "MiddlewareBaseConfig": ("nat.data_models.middleware", "MiddlewareBaseConfig"),
    "MiddlewareRef": ("nat.data_models.component_ref", "MiddlewareRef"),
    "NoSuchKeyError": ("nat.data_models.object_store", "NoSuchKeyError"),
    "ObjectStore": ("nat.object_store.interfaces", "ObjectStore"),
    "ObjectStoreRef": ("nat.data_models.component_ref", "ObjectStoreRef"),
    "ObjectStoreItem": ("nat.object_store.models", "ObjectStoreItem"),
    "ObjectStoreBaseConfig": ("nat.data_models.object_store", "ObjectStoreBaseConfig"),
    "OptimizerStrategyBaseConfig": ("nat.data_models.optimizer", "OptimizerStrategyBaseConfig"),
    "OptionalSecretStr": ("nat.data_models.common", "OptionalSecretStr"),
    "PromptOptimizationConfig": ("nat.data_models.optimizer", "PromptOptimizationConfig"),
    "RegistryHandlerBaseConfig": ("nat.data_models.registry_handler", "RegistryHandlerBaseConfig"),
    "RetrieverBaseConfig": ("nat.data_models.retriever", "RetrieverBaseConfig"),
    "RetrieverProviderInfo": ("nat.builder.retriever", "RetrieverProviderInfo"),
    "RetrieverRef": ("nat.data_models.component_ref", "RetrieverRef"),
    "SerializableSecretStr": ("nat.data_models.common", "SerializableSecretStr"),
    "TelemetryExporterBaseConfig": ("nat.data_models.telemetry_exporter", "TelemetryExporterBaseConfig"),
    "TTCStrategyRef": ("nat.data_models.component_ref", "TTCStrategyRef"),
    "TTCStrategyBaseConfig": ("nat.data_models.ttc_strategy", "TTCStrategyBaseConfig"),
    "TrainerAdapterConfig": ("nat.data_models.finetuning", "TrainerAdapterConfig"),
    "TrainerAdapterRef": ("nat.data_models.component_ref", "TrainerAdapterRef"),
    "TrainerConfig": ("nat.data_models.finetuning", "TrainerConfig"),
    "TrainerRef": ("nat.data_models.component_ref", "TrainerRef"),
    "TrajectoryBuilderConfig": ("nat.data_models.finetuning", "TrajectoryBuilderConfig"),
    "TrajectoryBuilderRef": ("nat.data_models.component_ref", "TrajectoryBuilderRef"),
    "get_secret_value": ("nat.data_models.common", "get_secret_value"),
    "register_auth_provider": ("nat.cli.register_workflow", "register_auth_provider"),
    "register_dataset_loader": ("nat.cli.register_workflow", "register_dataset_loader"),
    "register_embedder_client": ("nat.cli.register_workflow", "register_embedder_client"),
    "register_embedder_provider": ("nat.cli.register_workflow", "register_embedder_provider"),
    "register_eval_callback": ("nat.cli.register_workflow", "register_eval_callback"),
    "register_evaluator": ("nat.cli.register_workflow", "register_evaluator"),
    "register_front_end": ("nat.cli.register_workflow", "register_front_end"),
    "register_function": ("nat.cli.register_workflow", "register_function"),
    "register_function_group": ("nat.cli.register_workflow", "register_function_group"),
    "register_llm_client": ("nat.cli.register_workflow", "register_llm_client"),
    "register_llm_provider": ("nat.cli.register_workflow", "register_llm_provider"),
    "register_logging_method": ("nat.cli.register_workflow", "register_logging_method"),
    "register_memory": ("nat.cli.register_workflow", "register_memory"),
    "register_middleware": ("nat.cli.register_workflow", "register_middleware"),
    "register_object_store": ("nat.cli.register_workflow", "register_object_store"),
    "register_optimizer": ("nat.cli.register_workflow", "register_optimizer"),
    "register_optimizer_callback": ("nat.cli.register_workflow", "register_optimizer_callback"),
    "register_per_user_function": ("nat.cli.register_workflow", "register_per_user_function"),
    "register_per_user_function_group": ("nat.cli.register_workflow", "register_per_user_function_group"),
    "register_registry_handler": ("nat.cli.register_workflow", "register_registry_handler"),
    "register_retriever_client": ("nat.cli.register_workflow", "register_retriever_client"),
    "register_retriever_provider": ("nat.cli.register_workflow", "register_retriever_provider"),
    "register_telemetry_exporter": ("nat.cli.register_workflow", "register_telemetry_exporter"),
    "register_tool_wrapper": ("nat.cli.register_workflow", "register_tool_wrapper"),
    "register_trainer": ("nat.cli.register_workflow", "register_trainer"),
    "register_trainer_adapter": ("nat.cli.register_workflow", "register_trainer_adapter"),
    "register_trajectory_builder": ("nat.cli.register_workflow", "register_trajectory_builder"),
    "register_ttc_strategy": ("nat.cli.register_workflow", "register_ttc_strategy"),
    "set_secret_from_env": ("nat.data_models.common", "set_secret_from_env"),
}


def test_plugin_api_exports_public_contract():
    assert len(plugin_api.__all__) == len(set(plugin_api.__all__))
    assert set(plugin_api.__all__) == set(EXPECTED_PLUGIN_API_EXPORTS)

    for public_name, (module_name, source_name) in EXPECTED_PLUGIN_API_EXPORTS.items():
        source_module = importlib.import_module(module_name)
        assert getattr(plugin_api, public_name) is getattr(source_module, source_name)


def test_plugin_authoring_docs_prefer_public_api_imports():
    repo_root = Path(__file__).parents[4]
    paths = [
        repo_root / "docs/source/components/sharing-components.md",
        repo_root / "docs/source/build-workflows/advanced/middleware.md",
        repo_root / "docs/source/build-workflows/functions-and-function-groups/function-groups.md",
        repo_root / "docs/source/extend/custom-components",
        repo_root / "docs/source/extend/plugins.md",
        repo_root / "docs/source/improve-workflows/evaluate.md",
        repo_root / "docs/source/improve-workflows/optimizer.md",
        repo_root / "docs/source/improve-workflows/test-time-compute.md",
        repo_root / "docs/source/run-workflows/existing-agents/langgraph.md",
        repo_root / "packages/nvidia_nat_core/src/nat/cli/commands/workflow/templates/workflow.py.j2",
    ]
    denied_patterns = [
        "nat.cli.register_workflow",
        "nat.cli.register_llm_client",
        "from nat.builder.dataset_loader import DatasetLoaderInfo",
        "from nat.builder.embedder import EmbedderProviderInfo",
        "from nat.builder.evaluator import EvaluatorInfo",
        "from nat.builder.framework_enum import LLMFrameworkEnum",
        "from nat.builder.builder import Builder",
        "from nat.builder.builder import EvalBuilder",
        "from nat.builder.function import FunctionGroup",
        "from nat.builder.function_info import FunctionInfo",
        "from nat.builder.llm import LLMProviderInfo",
        "from nat.builder.retriever import RetrieverProviderInfo",
        "from nat.data_models.authentication import AuthProviderBaseConfig",
        "from nat.data_models.component_ref import",
        "from nat.data_models.dataset_handler import EvalDatasetBaseConfig",
        "from nat.data_models.embedder import EmbedderBaseConfig",
        "from nat.data_models.evaluator import EvaluatorBaseConfig",
        "from nat.data_models.finetuning import TrainerAdapterConfig",
        "from nat.data_models.finetuning import TrainerConfig",
        "from nat.data_models.finetuning import TrajectoryBuilderConfig",
        "from nat.data_models.front_end import FrontEndBaseConfig",
        "from nat.data_models.function import FunctionBaseConfig",
        "from nat.data_models.function import FunctionGroupBaseConfig",
        "from nat.data_models.llm import LLMBaseConfig",
        "from nat.data_models.logging import LoggingBaseConfig",
        "from nat.data_models.memory import MemoryBaseConfig",
        "from nat.data_models.middleware import FunctionMiddlewareBaseConfig",
        "from nat.data_models.middleware import MiddlewareBaseConfig",
        "from nat.data_models.object_store import KeyAlreadyExistsError",
        "from nat.data_models.object_store import NoSuchKeyError",
        "from nat.data_models.object_store import ObjectStoreBaseConfig",
        "from nat.data_models.optimizer import OptimizerStrategyBaseConfig",
        "from nat.data_models.optimizer import PromptOptimizationConfig",
        "from nat.data_models.registry_handler import RegistryHandlerBaseConfig",
        "from nat.data_models.retriever import RetrieverBaseConfig",
        "from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig",
        "from nat.data_models.ttc_strategy import TTCStrategyBaseConfig",
        "from nat.memory.interfaces import MemoryEditor",
        "from nat.memory.interfaces import MemoryManager",
        "from nat.memory.interfaces import MemoryReader",
        "from nat.memory.interfaces import MemoryWriter",
        "from nat.memory.models import MemoryItem",
        "from nat.middleware.dynamic.dynamic_function_middleware import DynamicFunctionMiddleware",
        "from nat.middleware.dynamic.dynamic_middleware_config import DynamicMiddlewareConfig",
        "from nat.middleware.function_middleware import FunctionMiddleware",
        "from nat.middleware.middleware import FunctionMiddlewareContext",
        "from nat.middleware.middleware import InvocationContext",
        "from nat.object_store.interfaces import ObjectStore",
        "from nat.object_store.models import ObjectStoreItem",
    ]

    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(path.rglob("*.md"))
        else:
            files.append(path)

    violations = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        for pattern in denied_patterns:
            if pattern in text:
                violations.append(f"{file_path.relative_to(repo_root)} contains {pattern!r}")

    assert not violations, "Plugin authoring docs should use nat.plugin_api:\n" + "\n".join(violations)
