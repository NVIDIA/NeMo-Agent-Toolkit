<!--
    SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

NeMo Agent Toolkit is designed to be backwards compatible with the previous version of the toolkit except for changes documented on this page.

Additionally, all new contributions should rely on the most recent version of the toolkit and not rely on any deprecated functionality.

## Migrating to a new version of NeMo Agent Toolkit

It is strongly encouraged to migrate any existing code to the latest conventions and remove any deprecated functionality.

## Version Specific Changes

### v1.4.0

#### Weave Trace Identifier Namespace

Weave trace identifiers now use the `nat` namespace.

If you depend on Weave trace names, update your dashboards and alert filters:
- Replace any old namespace prefixes (`aiq`) with `nat`.
- Re-run any saved queries that filter on trace or span names.

#### Calculator Function Group Migration

The calculator tools moved to a single function group with new names and input schemas.

Update your configurations and tool calls:
- Use the `calculator` function group with names such as `calculator__add` and `calculator__compare`.
- Pass numeric arrays for arithmetic inputs instead of parsing strings.

#### Sensitive Configuration Fields Use Secret Types

Sensitive configuration fields now use Pydantic `SecretStr` types for redaction and serialization.

If you read or set secret fields in code:
- Use `.get_secret_value()` when you need the raw value.
- Update any tests that compare secrets to expect `SecretStr` behavior.

#### Zep Cloud v3 Migration

The Zep Cloud integration now targets the v3 API and thread-based memory.

To migrate existing Zep configurations:
- Upgrade your dependency to `zep-cloud~=3.0`.
- Update any session-based references to thread-based APIs and ensure `conversation_id` is set for per-thread storage.

#### LLM and Embedder Model Dump Changes

All LLM and embedder providers now use `exclude_unset=True` for `model_dump`.

If you rely on implicit defaults being forwarded:
- Set explicit values in your configuration for fields you need to send.
- Update any custom providers that serialize configuration files to use the same behavior.

#### Per-User Function Instantiation

Functions and function groups can now be registered as per-user components.

If you enabled per-user workflows:
- Register per-user functions with `@register_per_user_function()` and ensure schemas are explicit.
- Verify your `nat serve` usage sets a `nat-session` cookie so per-user workflows can resolve a user ID.

#### Removal of `default_user_id` in General config

The `default_user_id` field was removed to prevent unsafe per-user workflow sharing.

To migrate existing configurations:
- Remove `default_user_id` from the `general` configuration section.
- For `nat run` and `nat eval`, set the new `user_id` fields in `ConsoleFrontEndConfig` and `EvaluationRunConfig`.

#### Function Group Separator Change

Function group names now use `__` instead of `.`.

To migrate:
- Update function names from `group.function` to `group__function` in configuration files and tool calls.
- Watch for deprecation warnings if you still use the legacy separator.

#### MCP Frontend Refactor

MCP server and frontend code moved into the `nvidia-nat-mcp` package.

Update your MCP usage:
- Import `MCPFrontEndPluginWorker` from `nat.plugins.mcp.server.front_end_plugin_worker`.
- Recommended: migrate any `mcp_tool_wrapper` usage to `mcp_client`.

#### `nvidia-nat-all` Packaging Changes

The `nvidia-nat-all` meta-package removed conflicting optional dependencies.

If you rely on extras:
- Reinstall `nvidia-nat-all` and review the updated optional dependency list.
- Install additional frameworks explicitly when needed to avoid conflicts.

### v1.3.0

#### CLI Changes

The MCP server CLI commands have been restructured.

* `nat mcp` is now a command group and can no longer be used to start the MCP server.
* `nat mcp serve` is now the main command to start the MCP server.
* `nat info mcp` has been removed. Use the new `nat mcp client` command instead.

**Listing MCP Tools:**
```bash
# Old (v1.2)
nat info mcp
nat info mcp --tool tool_name

# New (v1.3)
nat mcp client tool list
nat mcp client tool list --tool tool_name
```

**Pinging MCP Server:**
```bash
# Old (v1.2)
nat info mcp ping --url http://localhost:9901/sse

# New (v1.3)
nat mcp client ping --url http://localhost:9901/mcp
```

#### API Changes

##### API Server Data Models

The {py:mod}`nat.data_models.api_server` module has been updated to improve type safety and OpenAI API compatibility.

* {py:class}`nat.data_models.api_server.Choice` has been split into two specialized models:
  * {py:class}`nat.data_models.api_server.ChatResponseChoice` - for non-streaming responses (contains `message` field)
  * {py:class}`nat.data_models.api_server.ChatResponseChunkChoice` - for streaming responses (contains `delta` field)
  * {py:class}`nat.data_models.api_server.Choice` remains as a backward compatibility alias for `ChatResponseChoice`

* {py:class}`nat.data_models.api_server.ChatResponse` now requires `usage` field (no longer optional).

##### Builder `get_*` methods switched to asynchronous

The following builder methods have been switched to asynchronous to be aligned with other builder methods.

* {py:meth}`nat.builder.Builder.get_function` is now marked as async
* {py:meth}`nat.builder.Builder.get_functions` is now marked as async
* {py:meth}`nat.builder.Builder.get_memory_client` is now marked as async
* {py:meth}`nat.builder.Builder.get_memory_clients` is now marked as async
* {py:meth}`nat.builder.Builder.get_tool` is now marked as async
* {py:meth}`nat.builder.Builder.get_tools` is now marked as async

**Migration example:**

```python
# Old (v1.2)
function = builder.get_function("my_function")

# New (v1.3)
function = await builder.get_function("my_function")
```

##### MCP Default Transport Changed

- v1.2: Used SSE transport at `http://localhost:9901/sse`
- v1.3: Uses streamable-http transport at `http://localhost:9901/mcp`

To use SSE transport for backward compatibility:
```bash
nat mcp serve --config_file config.yml --transport sse
```

:::{warning}
SSE transport does not support authentication. For production deployments, use `streamable-http` transport with authentication configured.
:::

#### Package Changes

Core MCP functionality has been moved to the `nvidia-nat-mcp` package.

If you are using MCP functionality, you will need to install the `nvidia-nat[mcp]` extra.

#### Package Dependency Updates

The following dependency updates may affect your workflows:

* `mcp` updated from `~1.10` to `~1.13` - Update your MCP server configurations if needed
* `uvicorn` limited to `<0.36` for `nest_asyncio` compatibility
* `langchain-core` updated to `~0.3.75` - Review any custom LangChain workflows for compatibility
* `langgraph` updated to `~0.6.7` - Review any custom LangGraph workflows for compatibility
* `crewai` updated to `~0.193.2` - Review any custom CrewAI workflows for compatibility
* `semantic-kernel` updated to `~1.35` - Review any custom Semantic Kernel workflows for compatibility

#### Deprecations

:::{warning}
The following features are deprecated and will be removed in a future release.
:::

* {py:attr}`nat.telemetry_exporters.weave.WeaveTelemetryExporter.entity` - The `entity` field is deprecated. Remove this field from your Weave exporter configuration.
* `use_uvloop` configuration option - This setting in the general section of the config is deprecated. Remove this option from your workflow configurations.

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
