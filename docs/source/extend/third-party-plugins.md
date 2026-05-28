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

# Third-Party Plugin Packages

NVIDIA NeMo Agent Toolkit supports plugin packages that are developed, released, and maintained outside of the main
NeMo Agent Toolkit repository. Third-party plugin packages use the same runtime discovery, configuration, and
observability paths as first-party packages. After a package is installed in the same Python environment as
`nvidia-nat-core`, the toolkit discovers it through Python entry points.

This guide describes the recommended model for partner-owned plugin packages. For the stable Python import surface, see
the [Public Plugin API](./plugin-api.md). For general plugin discovery and supported plugin types, see the
[Plugin System](./plugins.md).

## Ownership Model

Third-party plugin packages are owned by the provider or partner that maintains the integration. The partner owns:

- The GitHub repository, branch protection, contribution policy, and release process.
- Package publishing to PyPI or another package repository.
- Compatibility testing against supported NeMo Agent Toolkit versions.
- Issues and pull requests that affect only the partner integration.
- Documentation for installation, configuration, credentials, and bug routing.

The NeMo Agent Toolkit project owns the stable public plugin API, first-party package behavior, and issues in
`nvidia-nat-core` that surface through third-party plugins.

## When to Use a Third-Party Package

Use a third-party-owned package for new partner integrations when the provider is best positioned to track its own API
roadmap, release cadence, and service behavior. This model is especially useful for:

- Agentic tools and function groups.
- LLM, embedder, and retriever providers or framework clients.
- Telemetry exporters, memory backends, object stores, and authentication providers.
- Custom `nat` CLI subcommands.
- Specialized front-end integrations.

New partner integrations should prefer this model over adding provider-specific code to the NeMo Agent Toolkit
monorepo.

## Naming

Use names that make the package discoverable and clearly associated with the toolkit:

- GitHub repository owner: `<provider>`
- GitHub repository name: `nemo-agent-toolkit-<product>`
- PyPI package: `nemo-agent-toolkit-<provider>`
- Python import package: `nat.plugins.<provider>`
- Entry point name: `nat_<provider>`

For example, a Tavily integration could use:

- Repository owner: `tavily-ai`
- Repository name: `nemo-agent-toolkit-tavily`
- PyPI package: `nemo-agent-toolkit-tavily`
- Import package: `nat.plugins.tavily`
- Entry point: `nat_tavily`

## Package Layout

Third-party packages should use a PEP 420 namespace package layout compatible with other NeMo Agent Toolkit
distributions:

```text
nemo-agent-toolkit-provider/
|-- pyproject.toml
|-- README.md
|-- src/
|   `-- nat/
|       `-- plugins/
|           `-- provider/
|               |-- __init__.py
|               |-- register.py
|               `-- tools.py
`-- tests/
    `-- test_provider.py
```

Do not add `__init__.py` files in the shared `nat` or `nat.plugins` namespace directories. These directories are shared
namespace packages across NeMo Agent Toolkit distributions. The provider package directory itself should contain an
`__init__.py`.

The `register.py` module should import the modules that define plugin registration decorators so those decorators run
when the entry point is loaded.

```python
from . import tools

__all__ = ["tools"]
```

## Entry Points

New external plugin packages should register component plugins under `nat.plugins`:

```toml
[project.entry-points."nat.plugins"]
nat_provider = "nat.plugins.provider.register"
```

The runtime also continues to load `nat.components` entry points for backward compatibility with existing packages.
Do not use `nat.components` for new third-party plugin packages.

Use these entry point groups for other extension points:

| Entry point group | Use |
| --- | --- |
| `nat.plugins` | Functions, function groups, LLM clients, retrievers, embedders, telemetry exporters, memory backends, object stores, authentication providers, and other component plugins. |
| `nat.cli` | Custom `nat` CLI subcommands. |
| `nat.front_ends` | Specialized front-end implementations. Front-end registration is not part of the stable `nat.plugin_api` facade. |

## Public API Imports

Third-party plugin code should import stable plugin-authoring APIs from `nat.plugin_api`:

```python
from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import register_function
```

Avoid depending on implementation modules such as `nat.cli.register_workflow`, `nat.builder.workflow_builder`, or
`nat.builder.function_info` unless a subsystem guide explicitly documents that module as the extension surface. Symbols
exported from `nat.plugin_api` are the stable public contract for external plugin packages.

## Framework-Agnostic Tools

Register tools with `register_function` or `register_function_group` unless the integration intentionally depends on a
specific framework. Framework-agnostic tools can be consumed through the toolkit's registered tool wrappers for
LangChain/LangGraph, CrewAI, LlamaIndex, AutoGen, Semantic Kernel, Google ADK, Agno, AWS Strands, and other supported
frameworks.

```python
from pydantic import Field

from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import register_function


class ProviderSearchConfig(FunctionBaseConfig, name="provider_search"):
    """Search using the provider API."""

    api_key: str = Field(description="Provider API key.")


@register_function(config_type=ProviderSearchConfig)
async def provider_search(config: ProviderSearchConfig, _builder: Builder):
    async def search(query: str) -> str:
        """Search for information using the provider API."""
        ...

    yield FunctionInfo.from_fn(search)
```

When possible, implement tools against the provider's own SDK or raw HTTP API rather than a framework-specific wrapper
package. Use framework-specific registration only when the integration cannot be expressed as a framework-agnostic
tool.

## Dependencies

Third-party packages should depend on `nvidia-nat-core` with a version range that allows compatible minor releases but
blocks unreviewed major releases:

```toml
dependencies = [
  "nvidia-nat-core>=1.2,<2.0",
]
```

Choose the lower bound based on the first NeMo Agent Toolkit version that includes the public API symbols your package
uses. Update the upper bound when you have tested compatibility with a new major version.

Use optional dependencies for provider SDK extras, development tools, and integration test dependencies when possible.

## Installation and Discovery

End users install third-party plugin packages directly into the same environment as `nvidia-nat-core`:

```bash
uv add nemo-agent-toolkit-provider
```

or:

```bash
pip install nemo-agent-toolkit-provider
```

After installation, the toolkit discovers the package through `importlib.metadata.entry_points()`. Users do not need to
edit a NeMo Agent Toolkit configuration file to load the package itself. They only need to reference the registered
component `_type` values in workflow configuration.

Use `nat info components` to confirm that a package is installed and discoverable:

```bash
uv run nat info components
```

## Development Workflow

Use `uv` for local development when possible. This matches the primary NeMo Agent Toolkit development toolchain and
keeps lock files compatible with the toolkit's CI patterns.

Common commands are:

```bash
uv sync
uv lock
uv run pytest
```

## Required Documentation

Each third-party plugin repository should include a README with:

- Installation commands for `uv` and `pip`.
- A minimal workflow YAML example.
- Configuration schema and credential setup.
- Supported NeMo Agent Toolkit versions.
- Testing instructions.
- Bug routing guidance for provider-owned issues versus `nvidia-nat-core` issues.
- License information.

## Testing

At minimum, third-party plugin packages should include:

- Unit tests for provider-specific logic.
- A loader smoke test that installs or imports the package, loads the entry point, and verifies that the registered
  component can be discovered.
- At least one representative end-to-end test that invokes the plugin with a mock, stub, or local test service.
- Compatibility CI against supported NeMo Agent Toolkit versions.

Provider integrations that require credentials should mark those tests as integration tests and skip them when the
required environment variables are not set.

## Lifecycle

Third-party plugin packages generally follow this lifecycle:

1. Apply: Open an issue or pull request with the proposed package name, integration scope, license, and repository.
2. Develop: Build the package against the public plugin API and the package layout in this guide.
3. Review: Validate layout, license, naming, entry points, README, and smoke tests.
4. List: Add the approved package to NeMo Agent Toolkit documentation after the partner publishes it.
5. Verify or feature: Promote the package when it meets higher compatibility or announcement requirements.
6. Maintain: Keep compatibility CI passing and update the package as provider APIs or toolkit versions change.
7. Deprecate or archive: Remove inactive packages from toolkit documentation while leaving published package versions
   in the partner's package repository.

## Listing and Promotion

The NeMo Agent Toolkit documentation may list third-party plugin packages that follow these guidelines. Plugin listings
can use the following lifecycle:

| Tier | Requirements | Benefits |
| --- | --- | --- |
| Listed | Package layout, naming, license review, entry point registration, and one-time loader smoke check. | Listed in NeMo Agent Toolkit plugin documentation. |
| Verified | Listed requirements, partner-run compatibility CI for supported toolkit versions, and maintained README. | Eligible for coding assistant instructions and stronger discoverability. |
| Featured | Verified requirements plus a coordinated release or feature announcement. | Eligible for release-note placement and other promotion. |

Inactive packages may be removed from NeMo Agent Toolkit documentation. Previously published package versions remain in
the partner's package repository.

## Support Boundaries

NVIDIA may provide design review, compatibility guidance, public API stability commitments, and documentation links for
approved third-party plugins.

NVIDIA does not run partner CI, publish partner packages, accept liability for partner code, or guarantee feature parity
between third-party providers. Bugs in provider-owned integration code should be filed in the provider's repository.
Bugs in `nvidia-nat-core` should be filed in the NeMo Agent Toolkit repository.

## Submission Checklist

Before requesting inclusion in NeMo Agent Toolkit documentation, verify that the package has:

- A repository and package name that follow the naming guidance.
- Source under `src/nat/plugins/<provider>`.
- No `__init__.py` files in shared namespace package directories.
- A `nat.plugins` entry point for component plugins.
- Imports from `nat.plugin_api` for public plugin-authoring APIs.
- A compatible `nvidia-nat-core` dependency range.
- An Apache-2.0 or approved permissive license.
- A README with install, configuration, workflow, testing, and bug routing instructions.
- Tests, including a loader smoke test.
- Compatibility CI for supported toolkit versions.
