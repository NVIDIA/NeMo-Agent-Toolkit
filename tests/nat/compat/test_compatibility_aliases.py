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

import pytest


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
        ("aiq.registry_handlers.schemas.publish", "BuiltAIQArtifact", "BuiltArtifact"),
        ("aiq.registry_handlers.schemas.publish", "AIQArtifact", "Artifact"),
        ("aiq.retriever.interface", "AIQRetriever", "Retriever"),
        ("aiq.retriever.models", "AIQDocument", "Document"),
        ("aiq.runtime.runner", "AIQRunnerState", "RunnerState"),
        ("aiq.runtime.runner", "AIQRunner", "Runner"),
        ("aiq.runtime.session", "AIQSessionManager", "SessionManager"),
        ("aiq.tool.retriever", "AIQRetrieverConfig", "RetrieverConfig"),
        ("aiq.tool.retriever", "aiq_retriever_tool", "retriever_tool"),
        ("aiq.experimental.decorators.experimental_warning_decorator", "aiq_experimental", "experimental"),
    ])
def test_compatibility_aliases(module_name: str, alias_name: str, target_name: str):
    module = importlib.import_module(module_name)
    alias = getattr(module, alias_name)
    target = getattr(module, target_name)
    assert alias is target
