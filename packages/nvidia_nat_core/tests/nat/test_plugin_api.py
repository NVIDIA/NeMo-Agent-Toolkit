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
import re
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
    "OptionalSecretStr": ("nat.data_models.common", "OptionalSecretStr"),
    "RetrieverBaseConfig": ("nat.data_models.retriever", "RetrieverBaseConfig"),
    "RetrieverProviderInfo": ("nat.builder.retriever", "RetrieverProviderInfo"),
    "RetrieverRef": ("nat.data_models.component_ref", "RetrieverRef"),
    "SerializableSecretStr": ("nat.data_models.common", "SerializableSecretStr"),
    "TelemetryExporterBaseConfig": ("nat.data_models.telemetry_exporter", "TelemetryExporterBaseConfig"),
    "get_secret_value": ("nat.data_models.common", "get_secret_value"),
    "register_auth_provider": ("nat.cli.register_workflow", "register_auth_provider"),
    "register_dataset_loader": ("nat.cli.register_workflow", "register_dataset_loader"),
    "register_embedder_client": ("nat.cli.register_workflow", "register_embedder_client"),
    "register_embedder_provider": ("nat.cli.register_workflow", "register_embedder_provider"),
    "register_eval_callback": ("nat.cli.register_workflow", "register_eval_callback"),
    "register_evaluator": ("nat.cli.register_workflow", "register_evaluator"),
    "register_function": ("nat.cli.register_workflow", "register_function"),
    "register_function_group": ("nat.cli.register_workflow", "register_function_group"),
    "register_llm_client": ("nat.cli.register_workflow", "register_llm_client"),
    "register_llm_provider": ("nat.cli.register_workflow", "register_llm_provider"),
    "register_memory": ("nat.cli.register_workflow", "register_memory"),
    "register_middleware": ("nat.cli.register_workflow", "register_middleware"),
    "register_object_store": ("nat.cli.register_workflow", "register_object_store"),
    "register_per_user_function": ("nat.cli.register_workflow", "register_per_user_function"),
    "register_per_user_function_group": ("nat.cli.register_workflow", "register_per_user_function_group"),
    "register_retriever_client": ("nat.cli.register_workflow", "register_retriever_client"),
    "register_retriever_provider": ("nat.cli.register_workflow", "register_retriever_provider"),
    "register_telemetry_exporter": ("nat.cli.register_workflow", "register_telemetry_exporter"),
    "register_tool_wrapper": ("nat.cli.register_workflow", "register_tool_wrapper"),
    "set_secret_from_env": ("nat.data_models.common", "set_secret_from_env"),
}

DEFERRED_PLUGIN_API_CANDIDATES = {
    "FrontEndBaseConfig": {
        "source": ("nat.data_models.front_end", "FrontEndBaseConfig"),
        "reason": "runtime hosting surface; needs explicit compatibility and security contract",
    },
    "LoggingBaseConfig": {
        "source": ("nat.data_models.logging", "LoggingBaseConfig"),
        "reason": "log sink surface; needs clearer trust guidance for sensitive logs",
    },
    "OptimizerStrategyBaseConfig": {
        "source": ("nat.data_models.optimizer", "OptimizerStrategyBaseConfig"),
        "reason": "specialized optimizer subsystem API",
    },
    "PromptOptimizationConfig": {
        "source": ("nat.data_models.optimizer", "PromptOptimizationConfig"),
        "reason": "specialized optimizer subsystem API",
    },
    "RegistryHandlerBaseConfig": {
        "source": ("nat.data_models.registry_handler", "RegistryHandlerBaseConfig"),
        "reason": "registry resolution surface; needs extension-contract review",
    },
    "TTCStrategyBaseConfig": {
        "source": ("nat.data_models.ttc_strategy", "TTCStrategyBaseConfig"),
        "reason": "advanced test-time compute subsystem API",
    },
    "TTCStrategyRef": {
        "source": ("nat.data_models.component_ref", "TTCStrategyRef"),
        "reason": "advanced test-time compute subsystem API",
    },
    "TrainerAdapterConfig": {
        "source": ("nat.data_models.finetuning", "TrainerAdapterConfig"),
        "reason": "specialized finetuning subsystem API",
    },
    "TrainerAdapterRef": {
        "source": ("nat.data_models.component_ref", "TrainerAdapterRef"),
        "reason": "specialized finetuning subsystem API",
    },
    "TrainerConfig": {
        "source": ("nat.data_models.finetuning", "TrainerConfig"),
        "reason": "specialized finetuning subsystem API",
    },
    "TrainerRef": {
        "source": ("nat.data_models.component_ref", "TrainerRef"),
        "reason": "specialized finetuning subsystem API",
    },
    "TrajectoryBuilderConfig": {
        "source": ("nat.data_models.finetuning", "TrajectoryBuilderConfig"),
        "reason": "specialized finetuning subsystem API",
    },
    "TrajectoryBuilderRef": {
        "source": ("nat.data_models.component_ref", "TrajectoryBuilderRef"),
        "reason": "specialized finetuning subsystem API",
    },
    "register_front_end": {
        "source": ("nat.cli.register_workflow", "register_front_end"),
        "reason": "runtime hosting surface; needs explicit compatibility and security contract",
    },
    "register_logging_method": {
        "source": ("nat.cli.register_workflow", "register_logging_method"),
        "reason": "log sink surface; needs clearer trust guidance for sensitive logs",
    },
    "register_optimizer": {
        "source": ("nat.cli.register_workflow", "register_optimizer"),
        "reason": "specialized optimizer subsystem API",
    },
    "register_optimizer_callback": {
        "source": ("nat.cli.register_workflow", "register_optimizer_callback"),
        "reason": "specialized optimizer subsystem API",
    },
    "register_registry_handler": {
        "source": ("nat.cli.register_workflow", "register_registry_handler"),
        "reason": "registry resolution surface; needs extension-contract review",
    },
    "register_trainer": {
        "source": ("nat.cli.register_workflow", "register_trainer"),
        "reason": "specialized finetuning subsystem API",
    },
    "register_trainer_adapter": {
        "source": ("nat.cli.register_workflow", "register_trainer_adapter"),
        "reason": "specialized finetuning subsystem API",
    },
    "register_trajectory_builder": {
        "source": ("nat.cli.register_workflow", "register_trajectory_builder"),
        "reason": "specialized finetuning subsystem API",
    },
    "register_ttc_strategy": {
        "source": ("nat.cli.register_workflow", "register_ttc_strategy"),
        "reason": "advanced test-time compute subsystem API",
    },
}


def test_plugin_api_exports_public_contract():
    assert len(plugin_api.__all__) == len(set(plugin_api.__all__))
    assert set(plugin_api.__all__) == set(EXPECTED_PLUGIN_API_EXPORTS)

    for public_name, (module_name, source_name) in EXPECTED_PLUGIN_API_EXPORTS.items():
        source_module = importlib.import_module(module_name)
        assert getattr(plugin_api, public_name) is getattr(source_module, source_name)


def test_deferred_plugin_api_candidates_remain_unpromoted():
    assert not (set(DEFERRED_PLUGIN_API_CANDIDATES) & set(EXPECTED_PLUGIN_API_EXPORTS))
    assert not (set(DEFERRED_PLUGIN_API_CANDIDATES) & set(plugin_api.__all__))

    for candidate, metadata in DEFERRED_PLUGIN_API_CANDIDATES.items():
        module_name, source_name = metadata["source"]
        source_module = importlib.import_module(module_name)
        assert getattr(source_module, source_name) is not None, f"Deferred plugin API candidate {candidate} moved"
        assert metadata["reason"]


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
        "from nat.cli.register_workflow import register_auth_provider",
        "from nat.cli.register_workflow import register_dataset_loader",
        "from nat.cli.register_workflow import register_embedder_client",
        "from nat.cli.register_workflow import register_embedder_provider",
        "from nat.cli.register_workflow import register_eval_callback",
        "from nat.cli.register_workflow import register_evaluator",
        "from nat.cli.register_workflow import register_function",
        "from nat.cli.register_workflow import register_function_group",
        "from nat.cli.register_workflow import register_llm_client",
        "from nat.cli.register_workflow import register_llm_provider",
        "from nat.cli.register_workflow import register_memory",
        "from nat.cli.register_workflow import register_middleware",
        "from nat.cli.register_workflow import register_object_store",
        "from nat.cli.register_workflow import register_per_user_function",
        "from nat.cli.register_workflow import register_per_user_function_group",
        "from nat.cli.register_workflow import register_retriever_client",
        "from nat.cli.register_workflow import register_retriever_provider",
        "from nat.cli.register_workflow import register_telemetry_exporter",
        "from nat.cli.register_workflow import register_tool_wrapper",
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
        "from nat.data_models.function import FunctionBaseConfig",
        "from nat.data_models.function import FunctionGroupBaseConfig",
        "from nat.data_models.llm import LLMBaseConfig",
        "from nat.data_models.memory import MemoryBaseConfig",
        "from nat.data_models.middleware import FunctionMiddlewareBaseConfig",
        "from nat.data_models.middleware import MiddlewareBaseConfig",
        "from nat.data_models.object_store import KeyAlreadyExistsError",
        "from nat.data_models.object_store import NoSuchKeyError",
        "from nat.data_models.object_store import ObjectStoreBaseConfig",
        "from nat.data_models.retriever import RetrieverBaseConfig",
        "from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig",
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
    public_imports = re.compile(r"from nat\.plugin_api import ([^\n]+)")
    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        for pattern in denied_patterns:
            if pattern in text:
                violations.append(f"{file_path.relative_to(repo_root)} contains {pattern!r}")
        for match in public_imports.finditer(text):
            imported_names = [name.strip().split(" as ")[0] for name in match.group(1).split(",")]
            for imported_name in imported_names:
                if imported_name and imported_name not in EXPECTED_PLUGIN_API_EXPORTS:
                    violations.append(
                        f"{file_path.relative_to(repo_root)} imports non-public nat.plugin_api symbol "
                        f"{imported_name!r}")

    assert not violations, "Plugin authoring docs should use nat.plugin_api:\n" + "\n".join(violations)
