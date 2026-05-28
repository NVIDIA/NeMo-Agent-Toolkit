<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# NVIDIA NeMo Agent Toolkit — Public Plugin API

NVIDIA NeMo Agent Toolkit external plugin packages should import plugin-authoring APIs from `nat.plugin_api`.
This module is the stable public import surface for registering common plugin components and authoring functions or
function groups.

```python
from nat.plugin_api import Builder
from nat.plugin_api import FunctionGroup
from nat.plugin_api import FunctionGroupBaseConfig
from nat.plugin_api import SerializableSecretStr
from nat.plugin_api import register_function_group
```

## Public Surface

The following `nat.plugin_api` exports are intended for plugin authors:

- Registration decorators for common external plugin components, such as `register_function`,
  `register_function_group`, `register_llm_provider`, `register_embedder_provider`, `register_retriever_provider`,
  `register_memory`, `register_object_store`, `register_middleware`, `register_telemetry_exporter`, and
  `register_tool_wrapper`.
- Function authoring types, including `Builder`, `EvalBuilder`, `Function`, `FunctionInfo`, and `FunctionGroup`.
- Component configuration bases, including `FunctionBaseConfig`, `FunctionGroupBaseConfig`, `LLMBaseConfig`,
  `EmbedderBaseConfig`, `RetrieverBaseConfig`, `MemoryBaseConfig`, `ObjectStoreBaseConfig`,
  `MiddlewareBaseConfig`, `AuthProviderBaseConfig`, `EvaluatorBaseConfig`, `EvalDatasetBaseConfig`, and
  `TelemetryExporterBaseConfig`.
- Registration return helpers, including `LLMProviderInfo`, `EmbedderProviderInfo`, `RetrieverProviderInfo`,
  `EvaluatorInfo`, and `DatasetLoaderInfo`.
- Small implementation contracts needed by registered components, including `FunctionMiddleware`,
  `DynamicFunctionMiddleware`, `MemoryEditor`, `ObjectStore`, and their associated context or value models.
- Component reference types, such as `FunctionRef`, `FunctionGroupRef`, `LLMRef`, `EmbedderRef`, `RetrieverRef`,
  `MemoryRef`, `ObjectStoreRef`, `MiddlewareRef`, and `AuthenticationRef`.
- Framework wrapper identifiers, including `LLMFrameworkEnum`.
- Secret helpers, including `SerializableSecretStr`, `OptionalSecretStr`, `get_secret_value`, and `set_secret_from_env`.

When a symbol is exported from `nat.plugin_api`, external packages can depend on that symbol's documented behavior across
minor and patch releases. Breaking changes to this public surface require a major release.

Installed plugins execute as trusted Python code in the application environment. This public facade defines stable import
paths and authoring contracts; it does not make untrusted plugin packages safe to install or execute.

The contract is intentionally explicit: adding a new public symbol requires adding it to `nat.plugin_api.__all__`, the
public API export test, and this documentation. Symbols that are not exported from `nat.plugin_api` should be treated as
implementation details unless a subsystem guide explicitly documents them as a specialized extension interface. Larger
subsystem-specific pipelines, such as telemetry processors or finetuning runtime interfaces, remain in their owning
modules until those contracts are promoted deliberately.

## Surface Review

The public facade is intentionally narrower than every component type supported by the runtime. The table below records
the current promotion decision for the major plugin-authoring surfaces.

| Area | Public API status | Motivation |
| --- | --- | --- |
| Functions | Stable public | Core external plugin unit. Third-party tool and workflow packages need `register_function`, `FunctionBaseConfig`, `FunctionInfo`, and `Builder`. |
| Function groups | Stable public | Best fit for providers exposing multiple related tools. Supports external packages that share clients/resources and expose `group__function` names. |
| Builders | Stable public | Registered build functions receive a builder. Authors need a stable builder type without depending on `WorkflowBuilder`. |
| Config bases | Stable public | Public decorators require corresponding config base classes for typed YAML/discovery contracts. |
| Provider info objects | Stable public | LLM, embedder, retriever, dataset, and evaluator registrations yield these helper objects. |
| Component refs | Stable public | External configs need stable references to configured functions, LLMs, embedders, retrievers, memory, object stores, middleware, and auth providers. |
| Secrets | Stable public | External providers commonly need API keys and environment-backed secrets. Public helpers reduce raw-string credential patterns. |
| Registration decorators | Stable public | Decorators are the core plugin discovery and registration API. |
| LLM | Stable public | External LLM providers and framework clients are primary integration points. |
| Embedder | Stable public | External embedding providers and framework clients are expected provider plugins. |
| Retriever | Stable public | External retrieval providers and framework clients are expected provider plugins. |
| Evaluator and dataset loader | Stable public | Evaluation integrations and dataset loaders are documented plugin types with direct external authoring use cases. |
| Memory | Stable public, trusted plugin | External memory backends are documented integration points. They may handle user data, so plugins must be trusted. |
| Object store | Stable public, trusted plugin | External storage backends need the config base, object-store interface, item model, and standard errors. Plugins must be trusted. |
| Middleware | Stable public, trusted plugin | Middleware supports caching, policy, auth injection, redaction, and tracing. It can observe or alter calls, so plugins must be trusted. |
| Telemetry | Stable public, trusted plugin | External observability exporters are common integrations. They may receive traces or user data, so plugins must be trusted. |
| Auth provider | Stable public, trusted plugin | API integrations need auth providers. They handle credentials or tokens, so plugins must be trusted. |
| Front end | Deferred | Runtime hosting surfaces need a more explicit compatibility and security contract before being promoted through `nat.plugin_api`. |
| Logging | Deferred | External log sinks may exfiltrate sensitive logs. Keep the existing implementation API until the stable contract and trust guidance are clearer. |
| Registry handler | Deferred | Registry handlers influence component discovery/resolution. Keep out of the stable facade until that extension contract is reviewed. |
| Optimizer and optimizer callback | Deferred | Optimizer extension points are specialized subsystem APIs and are not required by common integration packages. |
| Trainer, trainer adapter, and trajectory builder | Deferred | Finetuning extensions are broad subsystem APIs. Keep them in owning modules until the finetuning compatibility contract is promoted deliberately. |
| TTC strategy | Deferred | Test-time compute is an advanced/experimental subsystem. Do not imply stable public facade support until that API matures. |

Deferred surfaces remain available through their existing modules where those subsystem guides document them, but they are
not part of the stable `nat.plugin_api` facade. The deferred candidate list is also captured in
`packages/nvidia_nat_core/tests/nat/test_plugin_api.py` so CI can catch accidental promotion or stale candidate paths.

## Private Implementation Modules

Implementation modules remain importable for backward compatibility, but external plugin packages should not treat them
as stable API contracts. Prefer `nat.plugin_api` over direct imports from modules such as:

- `nat.cli.register_workflow`
- `nat.cli.type_registry`
- `nat.builder.function`
- `nat.builder.function_base`
- `nat.builder.function_info`
- `nat.builder.workflow_builder`

Concrete builders such as `WorkflowBuilder` are runtime implementation details. Plugin builders should type against
`Builder`, and plugin tests should use the utilities in `nvidia-nat-test` where possible.

## Function Group Contract

Function groups are the preferred pattern when one external service exposes multiple related tools. A function group:

- Shares one configuration object across all functions in the group.
- Can share clients, connections, caches, and other resources.
- Exposes functions using `instance_name__function_name`.
- Supports `include` and `exclude` fields from `FunctionGroupBaseConfig` to control which functions are exposed through
  workflow tool references.

```python
class SearchConfig(FunctionGroupBaseConfig, name="search_provider"):
    api_key: SerializableSecretStr


@register_function_group(config_type=SearchConfig)
async def build_search(config: SearchConfig, _builder: Builder):
    group = FunctionGroup(config=config, instance_name="search")

    async def search(query: str) -> dict:
        ...

    group.add_function("search", search, description=search.__doc__)
    yield group
```

With this group configured as `search`, the function name is `search__search`.
