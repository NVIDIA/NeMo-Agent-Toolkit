# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
The framework-parity matrix.

NAT claims to be framework-agnostic: it ships a maintained reference example for each
of its supported agent frameworks. This module is the single source of truth for what
"maintained" is actually checked against for each one:

  - ``structural``: install the example's own package (with its own declared extras,
    exactly as a user would) into an isolated environment and run the real ``nat
    validate`` CLI against its shipped config file. This requires no external
    credentials and runs on every scheduled harness invocation.

  - ``live``: in addition to the structural check, actually execute the example's
    canonical question through the real framework and a real LLM, then inspect the
    emitted IntermediateStep event stream to confirm the profiler attached at least
    one LLM span with non-zero token usage. This is the one that actually backs the
    "framework-agnostic" claim -- a config that merely parses is not proof a
    framework integration works end to end.

Only entries whose `required_live_env` is fully satisfiable by a single, cheap
credential are wired for the live tier today. Several real examples in this repo
(agno_personal_finance, nat_autogen_demo, strands_demo, multi_frameworks,
haystack_deep_research_agent) reach across multiple external services (NIM, SerpAPI,
Tavily, AWS Bedrock, MCP servers) as part of their normal, realistic configuration --
wiring live checks for those means either provisioning that whole service list as CI
secrets, or forking a second "minimal" config that no longer matches what a user
actually installs. Neither is worth doing to turn a yellow cell green; they run the
structural tier only, honestly labeled, until that changes.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class FrameworkEntry:
    # Short key, matches the GitHub Actions matrix value and the result filename.
    key: str
    # Human-readable name for the badge table.
    display_name: str
    # Path to the example's directory, relative to the repo root.
    example_dir: str
    # Path to the config file to validate/run, relative to the repo root.
    config_file: str
    # PyPI distribution name of the underlying framework, for `importlib.metadata`
    # version reporting in the badge table.
    framework_package: str
    # Env vars that must ALL be set for the live tier to run. If any are missing, the
    # entry falls back to the structural-only tier and is reported as "skipped (live)"
    # rather than failed.
    required_live_env: tuple[str, ...]
    # The question to run through the workflow during the live tier.
    question: str = "Plan a 1-day trip to Tokyo, covering one hotel and one activity."


FRAMEWORK_MATRIX: tuple[FrameworkEntry, ...] = (
    FrameworkEntry(
        key="crewai",
        display_name="CrewAI",
        example_dir="examples/frameworks/crewai_demo",
        config_file="examples/frameworks/crewai_demo/src/nat_crewai_demo/configs/config.yml",
        framework_package="crewai",
        required_live_env=("OPENAI_API_KEY", ),
    ),
    FrameworkEntry(
        key="semantic_kernel",
        display_name="Semantic Kernel",
        example_dir="examples/frameworks/semantic_kernel_demo",
        config_file="examples/frameworks/semantic_kernel_demo/src/nat_semantic_kernel_demo/configs/config.yml",
        framework_package="semantic-kernel",
        required_live_env=("OPENAI_API_KEY", "MEM0_API_KEY"),
    ),
    FrameworkEntry(
        key="adk",
        display_name="Google ADK",
        example_dir="examples/frameworks/adk_demo",
        config_file="examples/frameworks/adk_demo/src/nat_adk_demo/configs/config_oai.yml",
        framework_package="google-adk",
        required_live_env=("OPENAI_API_KEY", ),
        question="What is the weather in Tokyo right now?",
    ),
    FrameworkEntry(
        key="langchain_llama_index",
        display_name="LangChain + LlamaIndex",
        example_dir="examples/frameworks/multi_frameworks",
        config_file="examples/frameworks/multi_frameworks/src/nat_multi_frameworks/configs/config.yml",
        framework_package="langchain",
        required_live_env=(),  # needs NVIDIA_API_KEY + TAVILY_API_KEY; structural only for now.
    ),
    FrameworkEntry(
        key="agno",
        display_name="Agno",
        example_dir="examples/frameworks/agno_personal_finance",
        config_file="examples/frameworks/agno_personal_finance/src/nat_agno_personal_finance/configs/config.yml",
        framework_package="agno",
        required_live_env=(),  # needs NVIDIA_API_KEY + SERP_API_KEY; structural only for now.
    ),
    FrameworkEntry(
        key="autogen",
        display_name="AutoGen",
        example_dir="examples/frameworks/nat_autogen_demo",
        config_file="examples/frameworks/nat_autogen_demo/src/nat_autogen_demo/configs/config.yml",
        framework_package="autogen-agentchat",
        required_live_env=(),  # mixes NIM/OpenAI/Azure + an MCP client; structural only for now.
    ),
    FrameworkEntry(
        key="strands",
        display_name="Strands Agents",
        example_dir="examples/frameworks/strands_demo",
        config_file="examples/frameworks/strands_demo/src/nat_strands_demo/configs/config.yml",
        framework_package="strands-agents",
        required_live_env=(),  # mixes NIM/AWS Bedrock/OpenAI; structural only for now.
    ),
    FrameworkEntry(
        key="haystack",
        display_name="Haystack",
        example_dir="examples/frameworks/haystack_deep_research_agent",
        config_file=("examples/frameworks/haystack_deep_research_agent/src/"
                    "nat_haystack_deep_research_agent/configs/config.yml"),
        framework_package="haystack-ai",
        required_live_env=(),  # needs NVIDIA_API_KEY; structural only for now.
    ),
)


def get_entry(key: str) -> FrameworkEntry:
    for entry in FRAMEWORK_MATRIX:
        if entry.key == key:
            return entry
    raise KeyError(f"No framework-parity entry named {key!r}. "
                   f"Known keys: {[e.key for e in FRAMEWORK_MATRIX]}")
