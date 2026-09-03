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

# Framework Examples

NeMo Agent Toolkit is framework-agnostic: it can wrap and orchestrate agents built with
LangChain, LlamaIndex, CrewAI, Semantic Kernel, Google ADK, AutoGen, Agno, Strands
Agents, and Haystack, among others. Each subdirectory here is a maintained, runnable
reference example for one of those frameworks.

The table below is generated nightly by the [framework-parity
harness](../../.github/workflows/framework_parity.yml)
([source](../../scripts/framework_parity)), which installs each example's own package
into a fresh, isolated environment and validates it there — the same thing a new user
does when they follow one of the READMEs below. For frameworks with a single-credential
canonical config, it goes a step further and actually runs the workflow, then checks
the profiler's own event stream for a real LLM span with non-zero token usage, rather
than only checking that the YAML parses.

<!-- FRAMEWORK_PARITY_TABLE:START -->
_Not yet generated. Run `python scripts/framework_parity/orchestrate.py` and then
`scripts/framework_parity/render_table.py` to populate this table._

<!-- FRAMEWORK_PARITY_TABLE:END -->

## Examples

- [`crewai_demo`](crewai_demo/README.md) — CrewAI
- [`semantic_kernel_demo`](semantic_kernel_demo/README.md) — Semantic Kernel
- [`adk_demo`](adk_demo/README.md) — Google ADK
- [`multi_frameworks`](multi_frameworks/README.md) — LangChain + LlamaIndex in one workflow
- [`agno_personal_finance`](agno_personal_finance/README.md) — Agno
- [`nat_autogen_demo`](nat_autogen_demo/README.md) — AutoGen
- [`strands_demo`](strands_demo/README.md) — Strands Agents
- [`haystack_deep_research_agent`](haystack_deep_research_agent/README.md) — Haystack
- [`auto_wrapper/langchain_deep_research`](auto_wrapper/langchain_deep_research/README.md) — wrapping an existing LangChain workflow without rewriting it
