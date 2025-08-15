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

import importlib
import subprocess

import pytest

# Prevent isort from removing the pylint disable comments
# isort:skip_file


def test_aiq_subclass_is_nat_subclass():
    with pytest.deprecated_call():
        from aiq.data_models import function as aiq_function  # pylint: disable=no-name-in-module

        class MyAIQFunctionConfig(aiq_function.FunctionBaseConfig):
            pass

        from nat.data_models import function as nat_function
        assert issubclass(MyAIQFunctionConfig, nat_function.FunctionBaseConfig)


def test_cli_compat():
    expected_deprecation_warning = ("The 'aiq' command is deprecated and will be removed in a future release. "
                                    "Please use the 'nat' command instead.")

    result = subprocess.run("aiq", capture_output=True, check=True)
    assert expected_deprecation_warning in result.stderr.decode(encoding="utf-8")


@pytest.mark.parametrize(
    "module_name, alias_name, target_name",
    [
        ("aiq.builder.context", "AIQContextState", "ContextState"),
        ("aiq.builder.context", "AIQContext", "Context"),
        ("aiq.builder.user_interaction_manager", "AIQUserInteractionManager", "UserInteractionManager"),
        ("aiq.cli.commands.workflow.workflow_commands", "AIQPackageError", "PackageError"),
        ("aiq.data_models.api_server", "AIQChatRequest", "ChatRequest"),
        ("aiq.data_models.api_server", "AIQChoiceMessage", "ChoiceMessage"),
        ("aiq.data_models.api_server", "AIQChoiceDelta", "ChoiceDelta"),
        ("aiq.data_models.api_server", "AIQChoice", "Choice"),
        ("aiq.data_models.api_server", "AIQUsage", "Usage"),
        ("aiq.data_models.api_server", "AIQResponseSerializable", "ResponseSerializable"),
        ("aiq.data_models.api_server", "AIQResponseBaseModelOutput", "ResponseBaseModelOutput"),
        ("aiq.data_models.api_server", "AIQResponseBaseModelIntermediate", "ResponseBaseModelIntermediate"),
        ("aiq.data_models.api_server", "AIQChatResponse", "ChatResponse"),
        ("aiq.data_models.api_server", "AIQChatResponseChunk", "ChatResponseChunk"),
        ("aiq.data_models.api_server", "AIQResponseIntermediateStep", "ResponseIntermediateStep"),
        ("aiq.data_models.api_server", "AIQResponsePayloadOutput", "ResponsePayloadOutput"),
        ("aiq.data_models.api_server", "AIQGenerateResponse", "GenerateResponse"),
        ("aiq.data_models.component", "AIQComponentEnum", "ComponentEnum"),
        ("aiq.data_models.config", "AIQConfig", "Config"),
        ("aiq.front_ends.fastapi.fastapi_front_end_config", "AIQEvaluateRequest", "EvaluateRequest"),
        ("aiq.front_ends.fastapi.fastapi_front_end_config", "AIQEvaluateResponse", "EvaluateResponse"),
        ("aiq.front_ends.fastapi.fastapi_front_end_config", "AIQAsyncGenerateResponse", "AsyncGenerateResponse"),
        ("aiq.front_ends.fastapi.fastapi_front_end_config", "AIQEvaluateStatusResponse", "EvaluateStatusResponse"),
        ("aiq.front_ends.fastapi.fastapi_front_end_config",
         "AIQAsyncGenerationStatusResponse",
         "AsyncGenerationStatusResponse"),
        ("aiq.registry_handlers.package_utils", "build_aiq_artifact", "build_artifact"),
        ("aiq.registry_handlers.schemas.publish", "BuiltAIQArtifact", "BuiltArtifact"),
        ("aiq.registry_handlers.schemas.publish", "AIQArtifact", "Artifact"),
        ("aiq.retriever.interface", "AIQRetriever", "Retriever"),
        ("aiq.retriever.models", "AIQDocument", "Document"),
        ("aiq.runtime.loader", "get_all_aiq_entrypoints_distro_mapping", "get_all_entrypoints_distro_mapping"),
        ("aiq.runtime.runner", "AIQRunnerState", "RunnerState"),
        ("aiq.runtime.runner", "AIQRunner", "Runner"),
        ("aiq.runtime.session", "AIQSessionManager", "SessionManager"),
        ("aiq.tool.retriever", "AIQRetrieverConfig", "RetrieverConfig"),
        ("aiq.tool.retriever", "aiq_retriever_tool", "retriever_tool"),
        ("aiq.experimental.decorators.experimental_warning_decorator", "aiq_experimental", "experimental"),
    ])
@pytest.mark.parametrize("use_nat_namespace", [False, True])
def test_compatibility_aliases(module_name: str, alias_name: str, target_name: str, use_nat_namespace: bool):
    """
    Tests the compatibility aliases for classes and functions which contain "aiq" in the name.
    This test verifies that the alias points to the correct target, and that it is available under both the 'aiq'
    namespace and the 'nat' namespace.
    """
    if use_nat_namespace:
        module_name = module_name.replace("aiq.", "nat.", 1)
        assert module_name.startswith("nat.")

    module = importlib.import_module(module_name)
    alias = getattr(module, alias_name)
    target = getattr(module, target_name)
    assert alias is target
