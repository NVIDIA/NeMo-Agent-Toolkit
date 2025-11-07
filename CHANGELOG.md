<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Changelog
All notable changes to this project will be documented in this file.

## [1.3.1] - 2025-11-07

### 📦 Overview
This is a minor release with documentation updates, bug fixes, and non-breaking improvements.

### ✨ New Features
* feat: Add claude-sonnet-4.5 support by model-gating `top_p` by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1134
* Add support for arbitrary JSON body types in custom routes by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1163
### 🐛 Bug Fixes
* bug: fix non json serializable objects in config by @bbednarski9 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1112
* fix ADK demo multi-user session by @antoniomtz in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1120
* Fixes to detect optional parameters in tool conversion used by "nat mcp serve" by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1133
* Async Chat fixes by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1131
* Fix code concatenation issue with `code_execution_tool` when using a Piston server by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1154
* Fix documentation version switcher by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1167
### 📝 Documentation Updates
* Misc Documentation Fixes by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1136
* Document the need to install `nvidia-nat-test` prior to using `ToolTestRunner` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1150
* Update reasoning diagrams by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1153
* Update Quick Start UI documentation by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1156
* Add `security-considerations.md` document by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1168
* docs: 1.3.1 changelog by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1166

### 🙌 New Contributors
* @antoniomtz made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1120


## [1.3.0] - 2025-10-24

### 🚀 Notable Features and Improvements
* [ADK Support](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.3/docs/source/reference/frameworks-overview.md): Supports Google Agent Development Kit (ADK). Adds tool calling, core observability, and LLM integration in this release.
* [Control-Flow Agents](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.3/docs/source/workflows/about/index.md): [Sequential Executor](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.3/docs/source/workflows/about/sequential-executor.md) (Linear Agent) and [Router Agent](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.3/docs/source/workflows/about/router-agent.md) now control flow patterns of tool calls and sub-agents.
* [Function Groups](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.3/docs/source/workflows/function-groups.md): Packages multiple related functions together so they share configuration, context, and resources.
* [Hyperparameter Agent Optimizer](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.3/docs/source/reference/optimizer.md): Automates hyperparameter tuning and prompt engineering for workflows.
* [Introductory Notebook Improvements](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.3/examples/notebooks/README.md): Reorganizes getting started notebooks and adds Open in Colab links.
* [LLM Improvements](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.3/docs/source/workflows/llms/index.md)
  - Adds LiteLLM Provider
  - Supports GPT-5 (`/chat/completions` endpoint only)
  - Adds Nemotron thinking configuration
* [MCP Improvements](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.3/docs/source/workflows/mcp/index.md)
  - Supports `streamable-http` - `sse` is no longer the default transport type.
  - Supports initial authorization - Enables connecting to MCP servers that require authentication.
  - Supports multiple MCP tools from a single configuration - Pulls in entire tool sets published by MCP servers or filters them based on user configuration.
  - Enhances CLI utilities for MCP servers and clients - Improves the `nat mcp` sub command for querying, calling, and listing tools.
* Python 3.13 support

### 🚨 Breaking Changes
* Redis Configuration Changes in @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/649
* MCP enhancements: improves server config and adds support for all transport types (stdio, streamable-http) by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/718
* Move MCP client to a separate sub-package by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/768
* Signature change for `BaseAgent` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/757
* Builtin GitHub tools switched to Function Groups by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/684
* Fix chat history support in tool_calling_agent by @gfreeman-nvidia in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/837
* Change `nat mcp` to a command group with `serve` and `client` subcommands by @zhongxuanwang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/811
* Builder `get_*` functions should be marked `async` by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/834
* MCP Client Auth Support (part-2) by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/854
* ReWOO Agent Workflow Refactoring (Dependency DAG for async Executor). by @billxbf in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/861
* Reduce phoenix dependencies by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/985
* Remove example with poor performance by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1011
* Deprecate the `WeaveTelemetryExporter.entity` field by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1016
* Syncing UI submodule to bring secure proxy server updates by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1044

### ✨ New Features
* Add features `nat workflow create` a versioned dependency and `data` and symlinks folder by @zhongxuanwang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/639
* Feature: Azure OpenAI LLM provider and client by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/643
* Timezone Support for `datetime` Tool and Normalize Response Time Handling by @zhongxuanwang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/660
* Feature: GPT-5 Support by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/664
* Customize Log Truncation in Config by @RohanAdwankar in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/723
* feat: Support for Nemotron thinking configuration by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/717
* Track agent system prompt in config and add config to skip maintenance check by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/724
* Add `nvidia-nat-data-flywheel` subpackage with NeMo Data Flywheel integrations by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/716
* Enhance `ProcessingExporter` system to support redaction of content in telemetry traces by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/751
* feat: Python 3.13 support by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/761
* Add test LLM provider to support testing by @zhongxuanwang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/764
* Support additional provider parameters in LLM and Embedder config by @YosiElias in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/749
* Add return_direct option to tool_calling_agent for direct tool responses by @thepatrickchin in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/775
* Enable MCP auth for NAT MCP clients by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/752
* Add function group filtering by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/807
* Implement `Sequential Executor` tool by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/806
* Add a /debug route to NAT MCP frontend to expose MCP tools by @zhongxuanwang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/813
* MCP OAuth2 Token Introspection Validator by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/809
* [Synopsys] Feature: Google ADK Integration by @saglave in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/726
* Add a blueprint for Haystack Deep Research Agent by @oryx1729 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/461
* fix: re-add litellm after accidental removal by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/852
* Add `mcp/client/tool/list` endpoint by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/853
* feat: LiteLLM support for LangChain/LangGraph, Agno, CrewAI, LlamaIndex by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/881
* Add configurable token storage to MCP auth by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/883
* feat: Improve the developer journey for example notebooks by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/912
* feat: Add .env loading support to NAT cli by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/952
* feat: make built-in agents input adaptable by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/959
* UI submodule update 1.3 by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1002
* feat: switch to nemotron reasoning models by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1036

### 🔧 Improvements
* Collapse the `docs` dependency group into the `dev` dependencies group by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/591
* Forward-merge release/1.2 into develop by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/679
* Forward-merge release/1.2 into develop by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/690
* Weave: Group workflow traces under the parent evaluation call by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/663
* Misc release script improvements by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/692
* Fix `pytest` fixture deprecation warning by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/698
* Adopt ruff in CI by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/694
* Upload test results to codecov by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/699
* Add Coderabbit config by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/710
* Allow custom post-processing of EvalInput after the workflow is run by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/701
* Adding a Needs Triage label to issues which are created externally by @mdemoret-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/700
* Add fixtures allowing e2e tests to be optionally skipped upon missing environment variables by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/720
* Enable running e2e tests for nightly CI by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/727
* Enable the forward merger plugin of the rapids ops bot by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/728
* Ensure error reporting and propagating in a consistent pattern by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/722
* Improve input normalization of `ReAct` agent by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/730
* Update version of numpy to be more recent by @zhongxuanwang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/746
* chore: update LangChain and LangGraph versions by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/780
* Refactor OTLPSpanHeaderRedactionAdapterExporter to support multiple headers and Span tags by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/767
* Require approval from the `nat-dep-approvers` group for dependency changes by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/782
* Add `tool_call_max_retries` option to ReWOO agent by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/785
* Add NAT Agent Hyperparameter Optimizer by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/650
* Deprecating `use_uvloop` from general section of the config. by @zhongxuanwang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/794
* Report nightly test results by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/797
* Expanding nightly E2E tests by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/812
* Move MCP Client functionality to function groups by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/814
* Add `raise_tool_call_error` option to `ReWOO` agent by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/819
* Improved Dask shutdown by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/825
* fix: improve Google ADK structure and fix callback handlers for tools and LLMs by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/848
* Skip some tests in `test_mcp_client_base.py` to avoid blocking CI by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/850
* Implement OAuth2 security test coverage by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/862
* Console Auth Flow Exception Improvement by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/867
* Simplify simple-calculator MCP example by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/874
* ReWOO typing enhancements; more ruff checks; prefer `langchain-tavily` by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/872
* Mandate user id for MCP oauth2 authentication by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/873
* chore: bump nat-ui submodule by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/876
* Add additional E2E tests for examples by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/849
* Include branch name in nightly test report by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/879
* Improve new workflow template by @zhongxuanwang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/851
* Enhance OpenAI Chat API Compatibility by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/889
* chore: additional workflow template cleanup by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/899
* Add hierarchical IDs for consistent telemetry and reporting by @zhongxuanwang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/863
* Perform vale spelling checks on notebooks by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/896
* Implement deprecated decorator for deprecation notices by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/904
* Implement session aware MCP client routing by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/898
* feat: Improve the developer journey for example notebooks (part 2) by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/913
* Set the title warning to an error by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/918
* Improve multi-user MCP client handling by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/922
* Avoid Pydantic serialization warning triggered by tests by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/925
* fix: haystack deep research agent must be part of examples by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/927
* Silence several warnings being emitted by the tests by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/928
* UI submodule update by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/932
* Add ADK to TestLLM by @zhongxuanwang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/937
* Add opensearch service to CI by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/944
* Add an integration test for the custom route front-end example by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/945
* fix: move Google ADK agent example back to ADK example by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/943
* Add a docker compose yaml for running integration tests by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/946
* Clean up MCP logs by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/957
* Limit when we upload to artifactory by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/965
* Add security related warnings to MCP auth documentation by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/979
* Add E2E test for the simple calculator HITL example by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/984
* Add additional E2E tests for examples  by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/986
* Add an E2E test for the simple calculator MCP example by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/990
* Add E2E test for Redis Memory example by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/994
* Enable Chat History for WebSocket Messages by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/999
* fix: explicitly add `tool.uv.managed = true` to pyproject.toml files by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1003
* fix: address coderabbit feedback given from forward merge PR by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1007
* Fix string concatenation by @zhongxuanwang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/995
* Update backend corresponding to the MCP UI changes by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/988
* Add E2E tests for Simple Calculator Observability example by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1019
* fix: update authlib by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1017
* feat: unify wording for agent docs; clarify local LLMs; update telemetry package by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1042
* Always perform wheel builds in nightly CI by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1046
* docs: getting started notebook no. 1-5 cleanup by @bbednarski9 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1039
* fix: reintroduce `--all-files` to pre-commit CI by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1048
* Add E2E test for Langfuse observability example by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1047
* Refactor Optimizer Documentation for Clarity by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1026
* Move pareto visualzation section to docs from example by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1064
* Add location for prompt optimization functions by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1077
* Fix WebSocket HITL Response Schema and Update UI Submodule by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1079
* Add E2E test for Simple Calculator LangSmith observability example by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1083
* feat(mcp): allow MCP Server `--tool_name` filter to reference function groups by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1093
* Add gRPC Protocol Support to OTLP Span Exporters by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1102

### 🐛 Bug Fixes
* Toolcalling prompt by @gfreeman-nvidia in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/617
* Fix missing f-string prefixes in error messages by @YosiElias in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/686
* Configure `setuptools_scm` to use the `--first-parent` flag by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/693
* Persist User Message ID For HTTP Connections by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/696
* fix(pytest): suppress upstream pydantic warning from mem0 by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/709
* fix(`ModelGatedFieldMixin`): support multiple and indirect inheritance; rename to `GatedFieldMixin` by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/707
* Configure coderabbit not to apply conflicting labels by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/714
* Add missing implementation of abstract methods of `ToolTestRunner` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/729
* fix: Improved model detection/rules for `ThinkingMixin` by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/733
* Set `StreamHandler` to use `sys.stdout` in `console` registered logging method by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/742
* Add observability support when using MCP front end by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/741
* Fix issues in GPU Sizing Launchable Notebook by @nv-edwli in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/747
* fix(llm): resolve patch order to apply retry before thinking by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/750
* Remove conflicting/redundant `langchain-milvus` deps by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/754
* Update weave to 0.52 to handle incompat with gql 4.0.0 by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/755
* fix: include `thinking` in model_dump for `serve` config serialization by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/771
* Update MCP client readme to use the streamable-http example by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/777
* Fix async endpoints when using multiple workers by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/725
* fix: ensure `model_dump` excludes `None` fields when appropriate by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/772
* Fix workflow create documentation and command by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/790
* Fix `run_ci_local.sh` to not prompt for username/password by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/792
* Fix NAT FastAPI front end with Stdio-MCP server fails to initialize by @zhongxuanwang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/788
* Enable agent optimizer and refine LangChain callback handling. by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/796
* fix(cli): nat workflow create should validate workflow name by @Akshat8510 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/787
* Fix issue where optimizable params are in model dump by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/801
* Prevent retry storms in nested method calls by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/803
* Fix the `test_unified_api_server` integration tests by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/804
* Fixes `chat_completion` returning wrong type and substitute `.content` with `.text()` by @zhongxuanwang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/802
* Move visualization import into method for Optimizer by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/810
* fix: ensure workflows set a `FunctionGroup`s `instance_name` by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/815
* Make workflow name and description configurable as MCP tools by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/816
* fix: ensure `ContextVars` are all properly initialized by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/822
* Update RedisEditor to retrieve full document data from Redis when using get_memory tool by @thepatrickchin in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/823
* fix: correct logic for `test_unified_api_server.py` by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/826
* Fix `TraceAdapterRegistry` lookup errors in `SpanToDFWRecordProcessor` by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/831
* Fix `test_azure_openai_minimal_agent` test by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/832
* Avoid calling 'git lfs install' as CI already performs this by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/833
* Add missing dependencies by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/835
* Enable running tests for examples by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/838
* fix: Ensure console front-end validation is called by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/844
* Fix tests under `examples/`, remove all pytest `skip` markers by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/846
* Fix `chat_history` processing logic in ReAct agent by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/847
* Improve robustness of MCP client remote tool calling  by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/840
* Fix swallowing client errors bug by @zhongxuanwang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/841
* Declare `pip` as a direct dependency  by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/871
* Improve the re-connect handling mechanism for MCP with auth by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/875
* fix: ensure registration of adk demo functions; reduce warnings by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/885
* Fix problem with displaying MCP tools via the client CLI commands by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/888
* Revert "mcp-client-cli: Note that client and server transports must match." by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/892
* Fix Google ADK Demo registration by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/901
* fix: haystack deep research example test failure by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/910
* fix: Improve version detection for prerelease workflow creation by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/911
* fix: TTC must await get_function from builder by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/914
* fix: Docker must redeclare args in multi-stage builds; fix path in docs by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/916
* fix: pin uvicorn to prevent nest_asyncio patch error by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/929
* Resolve cancel scope error in MCP session cleanup with lifetime task by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/931
* fix: import error for weave sanitize by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/934
* fix: add missing awaits for get_memory_client by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/935
* Update package metadata by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/938
* fix: ensure console logging is configurable by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/947
* Enable the upload step for the release branches in nightly builds by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/953
* fix: custom plot charts function should error on invalid chart types by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/949
* Attempt to fix wheel metadata by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/958
* fix: ensure mcp client can load exported function group functions by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/960
* fix: Dockerfiles must not use any arg expansion on `COPY --from` lines by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/962
* fix: tracing in configs, clarify directions for simple web query by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/968
* Fix profiler agent tests by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/969
* Fix the simple calc hitl example by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/970
* Fix the profiler agent E2E test by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/975
* fix: small changes to improve reliability of getting started notebooks by @bbednarski9 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/989
* Fix reasoning models ending with v1 to use detailed thinking format by @jiayin-nvidia in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/980
* fix: Update system message (if exists) for thinking injection by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/987
* fix: improve agent-to-agent calling by simplifying pydantic model by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/993
* fix: langchain web ingest script must not always add CUDA documents by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1018
* Suppress error log generated when terminating NAT MCP server with `ctrl + C` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1022
* docs: prevent coderabbit from applying common labels by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1029
* Fix MCP auth redirect handling for remote-ssh and update docs by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1023
* Fix broken E2E tests by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1033
* Fixing the repeated step id bug by @zhongxuanwang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1032
* fix: update mcp test to not patch multiple times by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1045
* Keep original `NaN` or `null` scores from LLM judge in eval output  by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1058
* fix: eval integration test should inspect Ragas evaluators by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1067
* fix: pin langchain to prevent upgrade by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1074
* fix: strip rc package from notebook by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1081
* fix: apply coderabbit suggestions from forward merge by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1089
* fix: generalize eval test by reducing assumptions by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1090
* fix(mcp-client): support anyOf and oneOf when constructing schemas by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1094
* fix(rewoo): replace placeholder IFF type is `str` by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1105
* fix(examples-hitl): `RetryReactAgent` must work with function groups by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1106

### 📝 Documentation Updates
* fix(docs): Update Phoenix URL by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/721
* Add Dynatrace as otel export destination by @robertjahn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/736
* Add GPU Sizing Launchable Notebook to ``notebooks`` directory by @nv-edwli in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/738
* Adjust GPU Sizing Launchable Notebook by @nv-edwli in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/739
* docs: use https GitHub URL for easier installation and contribution by @mengdig-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/743
* docs: add notes on `nat eval` requiring `[profiling]` sub-package by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/769
* docs: add supported platforms to README and Installing Guide by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/766
* Add cursor rules for `test_llm` from `nvidia-nat-test` package by @zhongxuanwang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/774
* Remove issue #72 from list of known issues by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/779
* Add 'LangGraph' in locations where 'LangChain' appears by @zhongxuanwang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/778
* Document async endpoint functionality by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/786
* Document the NeMo-Agent-Toolkit-Examples repo by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/793
* Add GitHub Release Notes Template by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/808
* Move router agent to `control_flow` category by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/817
* docs: update example README to match current examples; move `haystack_deep_research_agent` by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/866
* Add Tracing Exporters configuration guide for Dynatrace by @robertjahn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/856
* MCP authentication Overview Doc by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/877
* Optimizer doc fix by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/884
* PyPi package install README updates by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/886
* mcp-client-cli: Note that client and server transports must match. by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/887
* Fix notebook link to install instructions by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/890
* Add a note that the transport on the MCP client andMCP  server need to match by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/893
* fix: documentation CLI tree update by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/900
* Update MCP related CLI commands in `cli.md` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/903
* Update CLI Docs for Optimizer by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/902
* docs: add note about increasing file descriptor limit by @nouraellm in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/897
* Add sizing calc summary in the main CLI docs by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/908
* Add missing plugins to list in `installing.md` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/906
* Include plugins in the staged API tree for documentation builds by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/909
* docs: Add nat object-store documentation to CLI docs by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/907
* fix: ReWOO example must properly escape quotes in string by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/915
* docs: Clarify MinIO directions for simple_web_query_eval by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/917
* fix: Prefer OpenAI schema for ReAct and Tool Calling Agents by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/923
* feat: Enable GFM-style Mermaid code blocks in Sphinx by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/924
* docs: update top-level README with libraries; remove outdated uvloop by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/926
* Update UI Submodule and Reference Docs by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/939
* Update sizing calc with pre-requisites by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/942
* docs: update Using Local LLMs (model name and directions) by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/941
* docs: clarify the need for a virtual environment in setup by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/948
* docs: clarify the need for a separate venv for local vLLM by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/950
* docs: improve automated description example; hoist Milvus to top-level by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/963
* Document running integration tests locally by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/966
* Update MCP documentation for consistency and clarity  by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/977
* docs: update create workflow guide by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/983
* docs: remove duplicate line in MCP authentication documentation by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/982
* ReWOO doc and test example fix by @billxbf in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/997
* fix: update notebook cells to remove unnecessary comments by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1008
* Update Migration Guide with Guidance on API data model changes by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1012
* Resolve Doc Build Issues by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1015
* Fix misplaced sample output in the MCP client doc by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1021
* fix: correct file paths in evaluate documentation by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1024
* docs: update migration guide for 1.3 by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1027
* docs: update the automated function description example by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1028
* docs: Fix missing await in memory documentation examples by @jackaldenryan in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1035
* docs: update ADK demo example; add framework documentation by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1040
* Added documentation for the data flywheel observability plugin by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1031
* Document clearly that auth is not supported on the MCP server side by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1043
* docs: update function groups documentation by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1025
* docs: add google colab links by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1051
* Added a note in the MCP doc for directly referencing a MCP tool within a client by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1052
* Update `evaluate.md` to add options to avoid `[429] Too Many Requests` errors by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1056
* docs: document observability provider support by providers by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1060
* Document writing E2E integration tests by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1062
* docs: document observability provider requirements by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1059
* docs: improve documentation for `nat eval` output files by @bbednarski9 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1061
* docs: update function group documentation and object store example by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1065
* docs: update Ragas docs; remove RAG references from Ragas by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1066
* docs: fix dynatrace OTLP link by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1069
* docs: clarify evaluators output files by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1068
* docs: add documentation for ThinkingMixin by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1071
* Minor documentation for LangSmith tracing by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1075
* docs: use github and sphinx flavored admonitions where appropriate by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1078
* Added summary and made doc changes to align with standards by @lvojtku in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1085
* docs: clarify that function groups can be used as part of `tool_name` list by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1092
* Improve Readme for 1.3 Release by @mdemoret-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1100
* docs: 1.3 changelog by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1070

### 🙌 New Contributors
* @zhongxuanwang-nv made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/639
* @YosiElias made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/686
* @RohanAdwankar made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/723
* @robertjahn made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/736
* @nv-edwli made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/738
* @mengdig-nv made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/743
* @Akshat8510 made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/787
* @saglave made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/726
* @oryx1729 made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/461
* @billxbf made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/861
* @nouraellm made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/897
* @jiayin-nvidia made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/980
* @bbednarski9 made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/989
* @jackaldenryan made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1035

## [1.2.1] - 2025-08-20
### 📦 Overview
This is a documentation only release, there are no code changes in this release.

### 📜 Full Change Log
* Add a version switcher to the documentation builds https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/681

## [1.2.0] - 2025-08-20
### 📦 Overview
The NeMo Agent toolkit, formerly known as Agent Intelligence (AIQ) toolkit, has been renamed to align with the NVIDIA NeMo family of products. This release brings significant new capabilities and improvements across authentication, resource management, observability, and developer experience. The toolkit continues to offer backwards compatibility, making the transition seamless for existing users.

While NeMo Agent Toolkit is designed to be compatible with the previous version, users are encouraged to update their code to follow the latest conventions and best practices. Migration instructions are provided in the [migration guide](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/docs/source/resources/migration-guide.md).

### 🚨 Breaking Changes
* Remove outdated/unsupported devcontainer by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/626
* Rename `aiq` namespace to `nat` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/618
* Update `AIQ` to `NAT` in documentation and comments by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/614
* Remove `AIQ` prefix from class and function names by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/606
* Rename aiqtoolkit packages to nvidia-nat by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/598
* Observability redesign to reduce dependencies and improve flexibility by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/379

### 🚀 Notable Features and Improvements
* [Authentication for Tool Calling](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/docs/source/reference/api-authentication.md): Implement robust authentication mechanisms that enable secure and configurable access management for tool invocation within agent workflows.
* [Test Time Compute](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/docs/source/reference/test-time-compute.md): Dynamically reallocate compute resources after model training, allowing agents to optimize reasoning, factual accuracy, and system robustness without retraining the base model.
* [Sizing Calculator](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/docs/source/workflows/sizing-calc.md): Estimate GPU cluster requirements to support your target number of users and desired response times, simplifying deployment planning and scaling.
* [Object Store Integration](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/docs/source/extend/object-store.md): Connect and manage data through supported object stores, improving agent extensibility and enabling advanced data workflows.
* [Enhanced Cursor Rules](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/docs/source/tutorials/build-a-demo-agent-workflow-using-cursor-rules.md): Build new workflows or extend existing ones by leveraging cursor rules, making agent development faster and more flexible.
* [Interactive Notebooks](https://github.com/NVIDIA/NeMo-Agent-Toolkit/tree/release/1.2/examples/notebooks): Access a suite of onboarding and example notebooks to accelerate agent workflow development, testing, and experimentation.
* [Observability Refactor](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/docs/source/workflows/observe/index.md): Onboard new observability and monitoring platforms more easily, and take advantage of improved plug-in architecture for workflow inspection and analysis.
* [Examples Reorganization](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/examples/README.md): Organize examples by functionality, making it easier to find and use the examples.

### 📜 Full Change Log
* Use consistent casing for ReAct agent by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/293
* Update alert triage agent's prompt by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/297
* Move Wikipedia search to separate file by @jkornblum-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/237
* Release documentation fixes by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/300
* Add a `pyproject.toml` to `simple_rag` example allowing for declared dependencies by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/284
* Update version in develop, in prep for the next release by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/294
* Add field validation for the evaluate API by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/311
* Intermediate steps: evaluation fix by @titericz in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/312
* Fix or silence warnings emitted by tests by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/305
* Add documentation for `load_workflow()` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/306
* Adding pytest-pretty for nice test outputs by @benomahony in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/194
* feat(telemetry): add langfuse and langsmith telemetry exporters #233 by @briancaffey in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/235
* Check links in markdown files by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/323
* Eval doc updates by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/322
* Add unit tests for the alert triage agent example by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/252
* Add support for AWS Bedrock LLM Provider by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/238
* Add missing import in `load_workflow` documentation by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/329
* propose another solution to problem[copy] by @LunaticMaestro in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/298
* Support additional_instructions by @gfreeman-nvidia in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/302
* Update installing.md by @manny-pi in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/316
* Add an async version of the /generate endpoint by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/315
* Update trajectory eval documentation by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/338
* Rename test mode to offline mode by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/343
* Simplify offline mode with `aiq eval` by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/344
* fix mcp client schema creation in flat lists by @slopp in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/346
* Refactor for better prompt and tool description organization by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/350
* Extend `IntermediateStep` to support tool schemas in tool calling LLM requests by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/357
* Fix AttributeError bug for otel_telemetry_exporter by @ZhongxuanWang in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/335
* Update `OpenAIModelConfig` to support `stream_usage` option by @mdemoret-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/328
* Rename to NeMo Agent toolkit by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/359
* fix(phoenix): set project name when using phoenix telemetry exporter (#337) by @briancaffey in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/347
* Account for the "required fields" list in the mcp_input_schema by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/360
* Provide a config to pass the complete dataset entry as an EvalInputItem field to evaluators by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/355
* Simplify custom evaluator definition by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/358
* Add Patronus OTEL Exporter by @hersheybar in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/341
* Expand Alert Triage Agent Offline Dataset by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/369
* Add Custom Classification Accuracy Evaluator for the Alert Triage Agent by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/373
* Add `Cursor rules` to improve Cursor support for development by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/319
* Added LLM retry logic to handle rate limiting LLM without frequent Exception by @liamy-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/368
* Fixes Function and LambdaFunction classes to push active function instance names by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/374
* TunableRagEvaluator: Re-enable inheriting from the base abc by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/375
* Add Job ID Appending to Output Directories and Maximum Folders Threshold by @ZhongxuanWang in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/331
* Add support for custom functions in bottleneck analysis by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/380
* Persist chat conversation ID for workflow tool usage by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/326
* Add support for Weave evaluation by @ayulockin in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/264
* Update the information displayed in the Weave Eval dashboard by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/390
* Allow non-json string outputs for workflows that use unstructured datasets by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/396
* Add aws region config for s3 eval uploads by @munjalp6 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/397
* add support for union types in mcp client by @cheese-head in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/372
* Add percentile computation (p90, p95, p99) to profiling by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/398
* Ragas custom evaluation field in evaluator by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/400
* Reorganize the examples into categories and improve re-use of example components by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/411
* Improve descriptions in top level examples README.md by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/415
* Add ragaai catalyst exporters by @vishalk-06 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/395
* Update MCP version by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/417
* feature request: Add galileo tracing workflow by @franz101 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/404
* Update index.md by @sugsharma in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/420
* Windows compatibility for temp file handling by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/423
* Sizing calculator to estimate the number of GPU for a target number of users by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/399
* Update and move W&B Weave Redact PII example by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/424
* Refactor IntermediateStep `parent_id` for clarification by @mdemoret-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/330
* Add Cursor rules for latinisms by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/426
* Resolve examples organization drift by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/429
* NeMo Agent rename by @lvojtku in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/422
* Removes redundant config variable from Retry Agent Function. by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/447
* Add otelcollector doc and example by @slopp in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/451
* Improve error logging during workflow initialization failure by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/464
* Added an AIQToolkit function that can be invoked to perform a simple completions task, given a natural language prompt. by @sayalinvidia in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/460
* Improve MCP error logging with connection failures by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/470
* Enable testing tools in isolation by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/391
* Refactor examples to improve discoverability and improve uniformity by @mdemoret-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/476
* Add Inference Time Scaling Module by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/381
* Refactor Agno Personal Finance Function and Update Configuration by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/477
* Add object store by @balvisio in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/299
* Observability redesign to reduce dependencies and improve flexibility by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/379
* Adding OpenAI Chat Completions API compatibility by @dfagnou in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/421
* Enhance code execution sandbox with improved error handling and debugging by @vikalluru in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/409
* Fixing inheritance on the OTel collector exporter and adding project name by @mdemoret-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/481
* Refactor retry mechanism and update retry mixin field config by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/480
* Integrate latest `RetryMixin` fixes with `aiqtoolkit_agno` subpackage by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/483
* Fix incorrect file paths in simple calculator example by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/482
* Documentation edits for sizing calculator by @lvojtku in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/436
* feat(redis): add redis memory backend and redis memory example #376 by @briancaffey in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/377
* Improve error handling and recovery mechanisms in agents by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/418
* Update `git clone` under `/doc` folder to point to `main` branch by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/484
* Pin `datasets` version in toplevel `pyproject.toml` by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/487
* Fix `otelcollector` to ensure project name is added to `OtelSpan` resource + added weave cleanup logic by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/488
* Fix shared field reference bug in `TypedBaseModel` inheritance by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/489
* Streamlining API Authentication by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/251
* Add experimental decorator to auth and ITS strategy methods by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/493
* Unify examples README structure by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/485
* Cleanup authorization settings to remove unnecessary options by @mdemoret-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/495
* Move `WeaveMixin._weave_calls` to `IsolatedAttribute` to avoid cleanup race conditions by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/499
* Set SCM versioning for `text_file_ingest` allowing it to be built in CI by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/501
* Update PII example to improve user experience and `WeaveExporter` robustness by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/502
* Fix `pyproject.toml` for `text_file_ingest` example by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/505
* Update `ci/checks.sh` to run all of the same checks performed by CI by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/506
* Suppress stack trace in error message in `ReActOutputParserException` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/507
* Clarify intermediate output formatting in agent tool_calling example by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/504
* Fix fastapi endpoint for plot_charts by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/508
* Fix UI docs to launch the simple calculator by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/511
* Update prerequisite and system prompt of `redis` memory example by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/510
* Fix HITL `por_to_jiratickets` example by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/515
* Relax overly restrictive constraints on AIQChatRequest model by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/512
* Fix file paths for simple_calculator_eval by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/517
* Fix: getting_started docker containers build with added compiler dependency by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/518
* Documentation: Specify minimum uv version by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/520
* Fix Simple Calculator HITL Example. by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/519
* Fix outdated path in `pyproject.toml` in `text_file_ingest` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/524
* Fix issue where aiq fails for certain log levels by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/523
* Update catalyst readme document by @vishalk-06 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/492
* Fix outdated file references under `/examples` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/526
* Misc Documentation cleanups by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/527
* Misc cleanups/fixes for `installing.md` document by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/528
* Fix: file path references in examples and docs by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/536
* Resolve batch flushing failure during `SpanExporter` cleanup by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/532
* Fix typos in the observability info commands by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/529
* Publish the linear fit data in the CalcRunner Output by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/498
* Restructure example README to fully align with reorganization by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/539
* Add example for Vulnerability Analysis for Container Security Blueprint by @ashsong-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/530
* Misc cleanups for `docs/source/quick-start/launching-ui.md` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/537
* Fix grammar error in uninstall help string by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/540
* Fix: custom routing example typos and output clarification by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/541
* Update the UI submodule to adopt fixes by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/543
* Fix: Examples README output clarifications; installation command by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/542
* Ensure type system and functional behavior are consistent for `to_type` specifications by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/513
* Documentation: update memory section to include redis; fix code references by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/544
* Update the dataset in the swe-bench README by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/546
* Fix alert triage agent documentation on system output by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/545
* Fix several dependency and documentation issues under `/examples` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/547
* Fixes to the `add-tools-to-a-workflow.md` tutorial by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/548
* Example: update `swe_bench` README to reflect output changes by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/550
* Update Cursor rules and documentations to remove unnecessary installation checks by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/549
* Update `object_store` example to use NVIDIA key instead of missing OPENAI key by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/552
* Remove deprecated code usage in the `por_to_jiratickets` example by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/557
* Fix `simple_auth` link to UI repository by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/553
* Update the LangSmith environment variable names in `simple_calculator_observability` example by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/558
* Improvements to extending telemetry exporters docs by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/554
* Misc cleanups for `create-a-new-workflow.md` document by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/556
* Reduce the number of warnings logged while running the `getting_started` examples by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/563
* Update observability system documentation to reflect modern architecture and remove snippets by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/562
* Add docker container for oauth server to fix configuration issues by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/561
* Ensure imports are lazily loaded in plugins improving startup time by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/564
* General improvements to `observe-workflow-with-catalyst.md` to improve experience by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/555
* Update Simple Auth Example Config File Path by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/566
* Convert `cursor_rules_demo` GIF files to Git LFS by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/567
* UI Submodule Update by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/568
* Restructure agents documentation by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/569
* Increase package/distro resolution in `DiscoveryMetadata` to improve utility of `aiq info components` by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/574
* Minor cleanups for the `run-workflows.md` document by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/572
* Add CI check for path validation within repository by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/573
* Improvements to observability plugin documentation by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/578
* Fixes for `adding-an-authentication-provider.md` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/579
* Minor cleanups for `sizing-calc.md` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/577
* minor doc update to pass lint by @gfreeman-nvidia in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/582
* Fixing missing space preventing proper render of snippet in markdown by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/581
* Fixing wrong override usage to make it compatible with py 3.11 by @vikalluru in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/585
* Updating UI submodule by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/588
* Fixes for `api-authentication.md` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/583
* Object Store code, documentation, and example improvements by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/587
* Fix module discovery errors when publishing with registry handlers by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/592
* Update logging levels in `ProcessingExporter`and `BatchingProcessor` to reduce shutdown noise by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/589
* Update template to use `logger` instead of `print` by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/590
* Fix fence indenting, remove ignore pattern by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/594
* Remove Unused Authentication Components from Refactor by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/596
* Minor cleanup to `using-local-llms.md` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/595
* Merge Post VDR changes by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/597
* Rename aiqtoolkit packages to nvidia-nat by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/598
* Rename Inference Time Scaling to Test Time Compute by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/600
* CI: upload script updated to set the artifactory path's top level dir by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/602
* Rename ITS tool functions to TTC tool functions by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/605
* Fix the artifactory component name to aiqtoolkit for all packages by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/603
* Fix Pylint in CI by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/609
* Remove `AIQ` prefix from class and function names by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/606
* Add support for synchronous LangChain tool calling by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/612
* Send Conversation ID with WebSocket Messages by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/613
* Adds support for MCP server /health endpoint, custom routes and a client `mcp ping` command  by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/576
* Updating UI submodule by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/622
* Rename `aiq` namespace to `nat` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/618
* Remove outdated/unsupported devcontainer by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/626
* Use issue types instead of title prefixes by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/619
* Fixes to `weave` telemetry exporter to ensure traces are properly sent to Weave by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/627
* Apply work-around for #621 to the gitlab ci scripts by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/630
* Revert unintended change to `artifactory_upload.sh` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/631
* Bugfix (Object Store): remove unnecessary S3 refs in config; fix mysql upload script by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/628
* Refactor embedder client structure for LangChain and Llama Index. by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/634
* Documentation: Update Using Local LLMs by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/623
* Align WebSocket Workflow Output with HTTP Output by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/635
* Update `AIQ` to `NAT` in documentation and comments by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/614
* Documentation(Providers): Surface LLM; add Embedders and Retrievers by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/625
* CI: improve path-check utility; fix broken links; add more path check rules by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/601
* Fix broken `additional_instructions` options for `ReWOO` agent by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/640
* Updating ui submodule by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/641
* Fix symlink structure to be consistent across all examples by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/642
* Fix: add missing uv.source for `simple_calculator_hitl` by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/638
* Enable datasets with custom formats by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/615
* Update uv.lock by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/644
* Run CI for commits to the release branch by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/645
* Add a note that the dataset needs to be uploaded to the S3 bucket by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/646
* Add UI documentation links and installation instructions by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/647
* Consolidate CI pipelines by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/632
* Install git-lfs in docs CI stage by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/648
* Fix `aiq` compatibility by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/651
* Bugfix: Align Python Version Ranges by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/655
* Docs: Add Migration Guide by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/653
* Update third-party-license files for v1.2 by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/657
* Add notebooks to show users how to get started with the toolkit and build agents by @cdgamarose-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/656
* Remove redundant prefix from directory names by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/665
* Docs: Add Upgrade Fix to Troubleshooting by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/659
* Enhance README with badges, installation, instructions, and a roadmap by @mdemoret-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/654
* Adding `.nspect-allowlist.toml` to remediate false positives found by scanner by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/668
* Fix: Remove `pickle` from MySQL-based Object Store by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/669
* Enable `BUILD_NAT_COMPAT` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/670
* Fix paths for compatibility packages by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/672
* Add missing compatibility package for `aiqtoolkit-weave` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/674
* Add chat_history to the context of ReAct and ReWOO agent by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/673

### 🙌 New Contributors
* @titericz made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/312
* @benomahony made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/194
* @briancaffey made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/235
* @LunaticMaestro made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/298
* @gfreeman-nvidia made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/302
* @manny-pi made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/316
* @slopp made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/346
* @ZhongxuanWang made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/335
* @hersheybar made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/341
* @munjalp6 made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/397
* @cheese-head made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/372
* @vishalk-06 made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/395
* @franz101 made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/404
* @sugsharma made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/420
* @lvojtku made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/422
* @sayalinvidia made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/460
* @dfagnou made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/421
* @vikalluru made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/409
* @ashsong-nv made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/530

## [1.1.0] - 2025-05-16
### Key Features
- Full MCP (Model Context Protocol) support
- Weave tracing
- Agno integration
- ReWOO Agent
- Alert Triage Agent Example

### What's Changed
* Have the examples README point to the absolute path by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/4
* Set initial version will be 1.0.0 by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/7
* Update `examples/simple_rag/README.md` to verify the installation of `lxml` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/9
* Use a separate README for pypi by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/10
* Document the need to install from source to run examples by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/8
* Fixing broken links by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/14
* Cleanup readmes by @mdemoret-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/15
* Pypi readme updates by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/16
* Final 1.0.0 cleanup by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/18
* Add subpackage readmes redirecting to the main package by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/20
* Update README.md by @gzitzlsb-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/25
* Fix #27 Documentation fix by @atalhens in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/28
* Fix #29 - Simple_calculator example throws error - list index out of range when given subtraction by @atalhens in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/31
* Fix: #32 Recursion Issue by @atalhens in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/33
* "Sharing NVIDIA AgentIQ Components" docs typo fix by @avoroshilov in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/42
* First pass at setting up issue templates by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/6
* Provide a cleaner progress bar when running evaluators in parallel by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/38
* Setup GHA CI by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/46
* Switch UI submodule to https by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/53
* gitlab ci pipeline cleanup by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/54
* Allow str or None for retriever description by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/55
* Fix case where res['categories'] = None by @balvisio in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/22
* Misc CI improvements by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/56
* CI Documentation improvements by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/24
* Add missing `platformdirs` dependency by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/62
* Fix `aiq` command error when the parent directory of `AIQ_CONFIG_DIR` does not exist by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/63
* Fix broken image link in multi_frameworks documentation by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/61
* Updating doc string for AIQSessionManager class. by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/64
* Fix ragas evaluate unit tests by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/68
* Normalize Gannt Chart Timestamps in Profiler Nested Stack Analysis by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/70
* Scripts for running CI locally by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/59
* Update types for `topic` and `description` attributes in  `AIQRetrieverConfig` to allow `None` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/76
* Add support for customizing output and uploading it to remote storage (S3 bucket) by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/71
* Support ARM in CI by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/73
* Allow overriding configuration values not set in the YAML by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/85
* Fix bug where `--workers` flag was being ignored by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/88
* Adding Cors config for api server by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/89
* Update changelog for 1.1.0a1 alpha release by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/90
* Updated changelog with another bug fix by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/93
* Adjust how the base_sha is passed into the workflow by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/81
* Changes for evaluating remote workflows by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/57
* Fix a bug in our pytest plugin causing test coverage to be under-reported by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/105
* Docker container for AgentIQ by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/87
* Modify JSON serialization to handle non-serializable objects by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/106
* Upload nightly builds and release builds to pypi by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/112
* Ensure the nightly builds have a unique alpha version number by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/115
* Ensure tags are fetched prior to determining the version by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/116
* Fix CI variable value by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/117
* Use setuptools_scm environment variables to set the version by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/118
* Only set the setuptools_scm variable when performing a nightly build by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/119
* Add a release PR template by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/123
* Add an async /evaluate endpoint to trigger evaluation jobs on a remote cluster by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/109
* Update /evaluate endpoint doc by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/126
* Add function tracking decorator and update IntermediateStep by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/98
* Fix typo in aiq.profiler.decorators by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/132
* Update the start command to use `validate_schema` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/82
* Document using local/self-hosted models by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/101
* added Agno integration by @wenqiglantz in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/36
* MCP Front-End Implementation by @VictorYudin in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/133
* Make kwargs optional to the eval output customizer scripts by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/139
* Add an example that shows simple_calculator running with a MCP service. by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/131
* add `gitdiagram` to README by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/141
* Updating HITL reference guide to instruct users to toggle ws mode and… by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/142
* Add override option to the eval CLI command by @Hritik003 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/129
* Implement ReWOO Agent by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/75
* Fix type hints and docstrings for `ModelTrainer` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/107
* Delete workflow confirmation check in CLI - #114 by @atalhens in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/137
* Improve Agent logging by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/136
* Add nicer error message for agents without tools by @jkornblum-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/146
* Add `colorama` to core dependency by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/149
* Rename packages  agentiq -> aiqtoolkit by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/152
* Rename AIQ_COMPONENT_NAME, remove unused COMPONENT_NAME by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/153
* Group wheels under a common `aiqtoolkit` directory by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/154
* Fix wheel upload wildcards by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/155
* Support Python `3.11` for AgentIQ by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/148
* fix pydantic version incompatibility, closes #74 by @zac-wang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/159
* Rename AgentIQ to Agent Intelligence Toolkit by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/160
* Create config file symlink with `aiq workflow create` command by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/166
* Rename generate/stream/full to generate/full and add filter_steps parameter by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/164
* Add support for environment variable interpolation in config files by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/157
* UI submodule rename by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/168
* Consistent Trace Nesting in Parallel Function Calling  by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/162
* Fix broken links in examples documentation by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/177
* Remove support for Python `3.13` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/178
* Add transitional packages by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/181
* Add a tunable RAG evaluator by @liamy-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/110
* CLI Documentation fixes in remote registry configuration section by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/184
* Fix uploading of transitional packages by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/187
* Update `AIQChatRequest` to support image and audio input by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/182
* Fix hyperlink ins the simple_calculator README by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/188
* Add support for fine-grained tracing using W&B Weave by @ayulockin in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/170
* Fix typo in CPR detected by co-pilot by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/190
* Note the name change in the top-level documentation and README.md by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/163
* fix typo in evaluate documentation for max_concurrency by @soumilinandi in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/191
* Fix a typo in the weave README by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/195
* Update simple example `eval` dataset by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/200
* Config option to specify the intermediate step types in workflow_output.json by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/198
* Update the Judge LLM settings in the examples to avoid retries by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/204
* Make `opentelemetry` and `phoenix` as optional dependencies by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/167
* Support user-defined HTTP request metadata in workflow tools. by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/130
* Check if request is present before setting attributes by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/209
* Add the alert triage agent example by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/193
* Updating ui submodule by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/211
* Fix plugin dependencies by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/208
* [FEA]add profiler agent to the examples folder by @zac-wang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/120
* Regenerate `uv.lock`, cleaned up `pyproject.toml` for profiler agent example and fixed broken link in `README` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/210
* Removed `disable=unused-argument` from pylint checks by @Hritik003 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/186
* Exception handling for discovery_metadata.py by @VictorYudin in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/215
* Fix incorrect eval output config access by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/219
* Treat a tagged commit the same as a nightly build by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/217
* Feature/add aiqtoolkit UI submodule by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/214
* Add a CLI command to list all tools available via the MCP server by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/221
* For remote evaluation, workflow config is not needed by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/225
* Move configurable parameters from env vars to config file by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/222
* Fix vulnerabilities in the alert triage agent example by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/227
* Add e2e test for the alert triage agent by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/226
* Fix remaining nSpect vulnerabilities for `1.1.0` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/229
* Remove redundant span stack handling and error logging by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/231
* Feature/add aiqtoolkit UI submodule by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/234
* Fix `Dockerfile` build failure for `v1.1.0-rc3` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/240
* Bugfix for alert triage agent to run in python 3.11 by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/244
* Misc example readme fixes by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/242
* Fix multiple documentation and logging bugs for `v1.1.0-rc3` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/245
* Consolidate MCP client and server docs, examples by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/246
* Update version of llama-index to 0.12.21 by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/257
* Fix environment variable interpolation with console frontend by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/255
* [AIQ][25.05][RC3] Example to showcase Metadata support by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/256
* mem: If conversation is not provided build it from memory by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/253
* Documentation restructure by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/189
* Prompt engineering to force `ReAct` agent to use memory for `simple_rag` example by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/260
* simple-calculator: Additional input validation by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/259
* Removed simple_mcp example by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/266
* Adding reference links to examples in README. by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/265
* mcp-client.md: Add a note to check that the MCP time-service is running by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/267
* Remove username from `README` log by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/271
* Enhance error handling in MCP tool invocation by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/263
* Resolves a linting error in MCP tool by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/274
* Fix long-term memory issues of `semantic_kernel` example by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/270
* Update to reflect new naming guidelines by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/258
* Updating submodule that fixes UI broken links by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/273
* Change the example input for `Multi Frameworks` example by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/277
* Fix intermediate steps parents when the parent is a Tool by @mdemoret-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/269
* Set mcp-proxy version in the sample Dockerfile to 0.5 by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/278
* Add an FAQ document by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/275
* Fix missing tool issue with `profiler_agent` example by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/279
* Add missing `telemetry` dependency to `profiler_agent` example by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/281
* eval-readme: Add instruction to copy the workflow output before re-runs by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/280
* Add additional notes for intermittent long-term memory issues in examples by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/282
* Run tests on all supported versions of Python by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/283
* Fix the intermediate steps span logic to work better with nested coroutines and tasks by @mdemoret-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/285

### New Contributors
* @dagardner-nv made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/7
* @yczhang-nv made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/9
* @gzitzlsb-nv made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/25
* @atalhens made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/28
* @avoroshilov made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/42
* @balvisio made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/22
* @ericevans-nv made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/64
* @dnandakumar-nv made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/70
* @wenqiglantz made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/36
* @VictorYudin made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/133
* @Hritik003 made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/129
* @jkornblum-nv made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/146
* @zac-wang-nv made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/159
* @mpenn made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/166
* @liamy-nv made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/110
* @ayulockin made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/170
* @soumilinandi made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/191
* @hsin-c made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/193

## [1.1.0a1] - 2025-04-05

### Added
- Added CORS configuration for the FastAPI server
- Added support for customizing evaluation outputs and uploading results to remote storage

### Fixed
- Fixed `aiq serve` when running the `simple_rag` workflow example
- Added missing `platformdirs` dependency to `aiqtoolkit` package

## [1.0.0] - 2024-12-04

### Added

- First release.
