<!--
    SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    SPDX-License-Identifier: Apache-2.0

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
-->

# Migration Guide

NeMo Agent toolkit is designed to be backwards compatible with the previous version of the toolkit except for changes documented on this page.

Additionally, all new contributions should rely on the most recent version of the toolkit and not rely on any deprecated functionality.

## Migrating to a new version of NeMo Agent toolkit

It is strongly encouraged to migrate any existing code to the latest conventions and remove any deprecated functionality.

## Version Specific Changes

### v1.2.0

#### Package Changes
* The `aiqtoolkit` package has been renamed to `nvidia-nat`.

:::{warning}
`aiqtoolkit` will be removed in a future release and is published as a transitional package.
:::

#### Module Changes
* The {py:mod}`aiq` module has been deprecated. Use {py:mod}`nat` instead.

:::{warning}
{py:mod}`aiq` will be removed in a future release.
:::

#### CLI Changes
* The `aiq` command has been deprecated. Use `nat` instead.

:::{warning}
The `aiq` command will be removed in a future release.
:::

#### API Changes

:::{note}
Compatibility aliases are in place to ensure backwards compatibility, however it is strongly encouraged to migrate to the new names.
:::

* Types which previously contained `AIQ` have had their `AIQ` prefix removed.
  * {py:class}`aiq.data_models.config.AIQConfig` -> {py:class}`nat.data_models.config.Config`
  * {py:class}`aiq.builder.context.AIQContext` -> {py:class}`nat.builder.context.Context`
  * {py:class}`aiq.builder.context.AIQContextState` -> {py:class}`nat.builder.context.ContextState`
  * {py:class}`aiq.builder.user_interaction_manager.AIQUserInteractionManager` -> {py:class}`nat.builder.user_interaction_manager.UserInteractionManager`
  * {py:class}`aiq.cli.commands.workflow.workflow_commands.AIQPackageError` -> {py:class}`nat.cli.commands.workflow.workflow_commands.PackageError`
  * {py:class}`aiq.data_models.api_server.AIQChatRequest` -> {py:class}`nat.data_models.api_server.ChatRequest`
  * {py:class}`aiq.data_models.api_server.AIQChoiceMessage` -> {py:class}`nat.data_models.api_server.ChoiceMessage`
  * {py:class}`aiq.data_models.api_server.AIQChoiceDelta` -> {py:class}`nat.data_models.api_server.ChoiceDelta`
  * {py:class}`aiq.data_models.api_server.AIQChoice` -> {py:class}`nat.data_models.api_server.Choice`
  * {py:class}`aiq.data_models.api_server.AIQUsage` -> {py:class}`nat.data_models.api_server.Usage`
  * {py:class}`aiq.data_models.api_server.AIQResponseSerializable` -> {py:class}`nat.data_models.api_server.ResponseSerializable`
  * {py:class}`aiq.data_models.api_server.AIQResponseBaseModelOutput` -> {py:class}`nat.data_models.api_server.ResponseBaseModelOutput`
  * {py:class}`aiq.data_models.api_server.AIQResponseBaseModelIntermediate` -> {py:class}`nat.data_models.api_server.ResponseBaseModelIntermediate`
  * {py:class}`aiq.data_models.api_server.AIQChatResponse` -> {py:class}`nat.data_models.api_server.ChatResponse`
  * {py:class}`aiq.data_models.api_server.AIQChatResponseChunk` -> {py:class}`nat.data_models.api_server.ChatResponseChunk`
  * {py:class}`aiq.data_models.api_server.AIQResponseIntermediateStep` -> {py:class}`nat.data_models.api_server.ResponseIntermediateStep`
  * {py:class}`aiq.data_models.api_server.AIQResponsePayloadOutput` -> {py:class}`nat.data_models.api_server.ResponsePayloadOutput`
  * {py:class}`aiq.data_models.api_server.AIQGenerateResponse` -> {py:class}`nat.data_models.api_server.GenerateResponse`
  * {py:class}`aiq.data_models.component.AIQComponentEnum` -> {py:class}`nat.data_models.component.ComponentEnum`
  * {py:class}`aiq.front_ends.fastapi.fastapi_front_end_config.AIQEvaluateRequest` -> {py:class}`nat.front_ends.fastapi.fastapi_front_end_config.EvaluateRequest`
  * {py:class}`aiq.front_ends.fastapi.fastapi_front_end_config.AIQEvaluateResponse` -> {py:class}`nat.front_ends.fastapi.fastapi_front_end_config.EvaluateResponse`
  * {py:class}`aiq.front_ends.fastapi.fastapi_front_end_config.AIQAsyncGenerateResponse` -> {py:class}`nat.front_ends.fastapi.fastapi_front_end_config.AsyncGenerateResponse`
  * {py:class}`aiq.front_ends.fastapi.fastapi_front_end_config.AIQEvaluateStatusResponse` -> {py:class}`nat.front_ends.fastapi.fastapi_front_end_config.EvaluateStatusResponse`
  * {py:class}`aiq.front_ends.fastapi.fastapi_front_end_config.AIQAsyncGenerationStatusResponse` -> {py:class}`nat.front_ends.fastapi.fastapi_front_end_config.AsyncGenerationStatusResponse`
  * {py:class}`aiq.registry_handlers.schemas.publish.BuiltAIQArtifact` -> {py:class}`nat.registry_handlers.schemas.publish.BuiltArtifact`
  * {py:class}`aiq.registry_handlers.schemas.publish.AIQArtifact` -> {py:class}`nat.registry_handlers.schemas.publish.Artifact`
  * {py:class}`aiq.retriever.interface.AIQRetriever` -> {py:class}`nat.retriever.interface.Retriever`
  * {py:class}`aiq.retriever.models.AIQDocument` -> {py:class}`nat.retriever.models.Document`
  * {py:class}`aiq.runtime.runner.AIQRunnerState` -> {py:class}`nat.runtime.runner.RunnerState`
  * {py:class}`aiq.runtime.runner.AIQRunner` -> {py:class}`nat.runtime.runner.Runner`
  * {py:class}`aiq.runtime.session.AIQSessionManager` -> {py:class}`nat.runtime.session.SessionManager`
  * {py:class}`aiq.tool.retriever.AIQRetrieverConfig` -> {py:class}`nat.tool.retriever.RetrieverConfig`
* Functions and decorators which previously contained `aiq_` have had `aiq` removed. **Compatibility aliases are in place to ensure backwards compatibility.**
  * {py:func}`aiq.experimental.decorators.experimental_warning_decorator.aiq_experimental` -> {py:func}`nat.experimental.decorators.experimental_warning_decorator.experimental`
  * {py:func}`aiq.registry_handlers.package_utils.build_aiq_artifact` -> {py:func}`nat.registry_handlers.package_utils.build_artifact`
  * {py:func}`aiq.runtime.loader.get_all_aiq_entrypoints_distro_mapping` -> {py:func}`nat.runtime.loader.get_all_entrypoints_distro_mapping`
  * {py:func}`aiq.tool.retriever.aiq_retriever_tool` -> {py:func}`nat.tool.retriever.retriever_tool`

### v1.1.0

#### Package Changes
* The `agentiq` package has been renamed to `aiqtoolkit`.

:::{warning}
`agentiq` will be removed in a future release and is published as a transitional package.
:::
