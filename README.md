<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

![NVIDIA NeMo Agent Toolkit](./docs/source/_static/banner.png "NeMo Agent Toolkit banner image")

# NVIDIA NeMo Agent Toolkit

<!-- vale off (due to hyperlinks) -->
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Agent-Toolkit)](https://github.com/NVIDIA/NeMo-Agent-Toolkit/releases)
[![PyPI version](https://img.shields.io/pypi/v/nvidia-nat)](https://pypi.org/project/nvidia-nat/)
[![GitHub issues](https://img.shields.io/github/issues/NVIDIA/NeMo-Agent-Toolkit)](https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/NVIDIA/NeMo-Agent-Toolkit)](https://github.com/NVIDIA/NeMo-Agent-Toolkit/pulls)
[![GitHub Repo stars](https://img.shields.io/github/stars/NVIDIA/NeMo-Agent-Toolkit)](https://github.com/NVIDIA/NeMo-Agent-Toolkit)
[![GitHub forks](https://img.shields.io/github/forks/NVIDIA/NeMo-Agent-Toolkit)](https://github.com/NVIDIA/NeMo-Agent-Toolkit/network/members)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/NeMo-Agent-Toolkit)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/NeMo-Agent-Toolkit/)
<!-- vale on -->

<div align="center">

*NVIDIA NeMo Agent Toolkit adds intelligence to AI agents across any framework—enhancing speed, accuracy, and decision-making through enterprise-grade instrumentation, observability, and continuous learning.*

</div>

## 🔥 New Features

- [**LangGraph Agent Automatic Wrapper:**](./examples/frameworks/auto_wrapper/langchain_deep_research/README.md) Easily onboard existing LangGraph agents to NeMo Agent Toolkit. Use the automatic wrapper to access NeMo Agent Toolkit advanced features with very little modification of LangGraph agents.

- [**Automatic Reinforcement Learning (RL):**](./docs/source/improve-workflows/finetuning/index.md) Improve your agent quality by fine-tuning open LLMs to better understand your agent's workflows, tools, and prompts. Perform GRPO with [OpenPipe ART](./docs/source/improve-workflows/finetuning/rl_with_openpipe.md) or DPO with [NeMo Customizer](./docs/source/improve-workflows/finetuning/dpo_with_nemo_customizer.md) using NeMo Agent Toolkit built-in evaluation system as a verifier.

- [**Initial NVIDIA Dynamo Integration:**](./examples/dynamo_integration/README.md) Accelerate end-to-end deployment of agentic workflows with initial Dynamo support. Utilize the new agent-aware router to improve worker latency by predicting future agent behavior.

- [**A2A Support:**](./docs/source/components/integrations/a2a.md) Build teams of distributed agents using the A2A protocol.

- [**Safety and Security Engine:**](./examples/safety_and_security/retail_agent/README.md) Strengthen safety and security workflows by simulating scenario-based attacks, profiling risk, running guardrail-ready evaluations, and applying defenses with red teaming. Validate defenses, profile risk, monitor behavior, and harden agents across any framework.

- [**Amazon Bedrock AgentCore and Strands Agents Support:**](./docs/source/components/integrations/frameworks.md#strands) Build agents using Strands Agents framework and deploy them securely on Amazon Bedrock AgentCore runtime.

- [**Microsoft AutoGen Support:**](./docs/source/components/integrations/frameworks.md#autogen) Build agents using the Microsoft AutoGen framework.

- [**Per-User Functions:**](./docs/source/extend/custom-components/custom-functions/per-user-functions.md) Use per-user functions for deferred instantiation, enabling per-user stateful functions, per-user resources, and other features.

## ✨ Key Features

- 🛠️ **Building Agents**: Accelerate your agent development with tools that make it easier to get your agent into production.
  - 🧩 [**Framework Agnostic:**](./docs/source/components/integrations/frameworks.md) Work side-by-side with agentic frameworks to add the instrumentation necessary for observing, profiling, and optimizing your agents. Use the toolkit with popular frameworks such as [LangChain](https://www.langchain.com/), [LlamaIndex](https://www.llamaindex.ai/), [CrewAI](https://www.crewai.com/), [Microsoft Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/), and [Google ADK](https://google.github.io/adk-docs/), as well as custom enterprise agentic frameworks and simple Python agents.
  - 🔁 [**Reusability:**](./docs/source/components/sharing-components.md) Build components once and use them multiple times to maximize the value from development effort.
  - ⚡ [**Customization:**](docs/source/get-started/tutorials/customize-a-workflow.md) Start with a pre-built agent, tool, or workflow, and customize it to your needs.
  - 💬 [**Built-In User Interface:**](./docs/source/run-workflows/launching-ui.md) Use the NeMo Agent Toolkit UI chat interface to interact with your agents, visualize output, and debug workflows.
- 📈 **Agent Insights:** Utilize NeMo Agent Toolkit instrumentation to better understand how your agents function at runtime.
  - 📊 [**Profiling:**](./docs/source/improve-workflows/profiler.md) Profile entire workflows from the agent level all the way down to individual tokens to identify bottlenecks, analyze token efficiency, and guide developers in optimizing their agents.
  - 🔎 [**Observability:**](./docs/source/run-workflows/observe/observe.md) Track performance, trace execution flows, and gain insights into your agent behaviors in production.
- 🚀 **Agent Optimization:** Improve your agent's quality, accuracy, and performance with a suite of tools for all phases of the agent lifecycle.
  - 🧪 [**Evaluation System:**](./docs/source/improve-workflows/evaluate.md) Validate and maintain accuracy of agentic workflows with a suite of tools for offline evaluation.
  - 🎯 [**Hyper-Parameter and Prompt Optimizer:**](./docs/source/improve-workflows/optimizer.md) Automatically identify the best configuration and prompts to ensure you are getting the most out of your agent.
  - 🧠 [**Fine-tuning with Reinforcement Learning:**](./docs/source/improve-workflows/finetuning/index.md) Fine-tune LLMs specifically for your agent and train intrinsic information about your workflow directly into the model.
  - ⚡ [**NVIDIA Dynamo Integration:**](./examples/dynamo_integration/README.md) Use Dynamo and NeMo Agent Toolkit together to improve agent performance at scale.
- 🔌 **Protocol Support:** Integrate with common protocols used to build agents.
  - 🔗 [**Model Context Protocol (MCP):**](./docs/source/build-workflows/mcp-client.md) Integrate [MCP tools](./docs/source/build-workflows/mcp-client.md) into your agents or serve your tools and agents as an [MCP server](./docs/source/run-workflows/mcp-server.md) for others to consume.
  - 🤝 [**Agent-to-Agent (A2A) Protocol:**](./docs/source/components/integrations/a2a.md) Build teams of distributed agents with full support for authentication.

With NeMo Agent Toolkit, you can move quickly, experiment freely, and ensure reliability across all your agent-driven projects.

## 🚀 Installation

Before you begin using NeMo Agent Toolkit, ensure that you have Python 3.11, 3.12, or 3.13 installed on your system.

> [!NOTE]
> For users who want to run the examples, it's required to clone the repository and install from source to get the necessary files required to run the examples. Please refer to the [Examples](./examples/README.md) documentation for more information.

To install the latest stable version of NeMo Agent Toolkit from PyPI, run the following command:

```bash
pip install nvidia-nat
```

NeMo Agent Toolkit has many optional dependencies that can be installed with the core package. Optional dependencies are grouped by framework. For example, to install the LangChain/LangGraph plugin, run the following:

```bash
pip install "nvidia-nat[langchain]"
```

Detailed installation instructions, including the full list of optional dependencies and their conflicts, can be found in the [Installation Guide](./docs/source/get-started/installation.md).

## 🌟 Hello World Example

Before getting started, it's possible to run this simple workflow and many other examples in Google Colab with no setup. Click here to open the introduction notebook: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/NeMo-Agent-Toolkit/).

1. Ensure you have set the `NVIDIA_API_KEY` environment variable to allow the example to use NVIDIA NIMs. An API key can be obtained by visiting [`build.nvidia.com`](https://build.nvidia.com/) and creating an account.

   ```bash
   export NVIDIA_API_KEY=<your_api_key>
   ```

2. Create the NeMo Agent Toolkit workflow configuration file. This file will define the agents, tools, and workflows that will be used in the example. Save the following as `workflow.yml`:

   ```yaml
   functions:
      # Add a tool to search wikipedia
      wikipedia_search:
         _type: wiki_search
         max_results: 2

   llms:
      # Tell NeMo Agent Toolkit which LLM to use for the agent
      nim_llm:
         _type: nim
         model_name: nvidia/nemotron-3-nano-30b-a3b
         temperature: 0.0
         chat_template_kwargs:
            enable_thinking: false

   workflow:
      # Use an agent that 'reasons' and 'acts'
      _type: react_agent
      # Give it access to our wikipedia search tool
      tool_names: [wikipedia_search]
      # Tell it which LLM to use
      llm_name: nim_llm
      # Make it verbose
      verbose: true
      # Retry up to 3 times
      parse_agent_response_max_retries: 3
   ```

3. Run the Hello World example using the `nat` CLI and the `workflow.yml` file.

   ```bash
   nat run --config_file workflow.yml --input "List five subspecies of Aardvarks"
   ```

   This will run the workflow and output the results to the console.

   ```console
   Workflow Result:
   ['Here are five subspecies of Aardvarks:\n\n1. Orycteropus afer afer (Southern aardvark)\n2. O. a. adametzi  Grote, 1921 (Western aardvark)\n3. O. a. aethiopicus  Sundevall, 1843\n4. O. a. angolensis  Zukowsky & Haltenorth, 1957\n5. O. a. erikssoni  Lönnberg, 1906']
   ```

## 📚 Additional Resources

* 📖 [Documentation](https://docs.nvidia.com/nemo/agent-toolkit/latest): Explore the full documentation for NeMo Agent Toolkit.
* 🧭 [Get Started Guide](./docs/source/get-started/installation.md): Set up your environment and start building with NeMo Agent Toolkit.
* 🤝 [Contributing](./docs/source/resources/contributing/index.md): Learn how to contribute to NeMo Agent Toolkit and set up your development environment.
* 🧪 [Examples](./examples/README.md): Explore examples of NeMo Agent Toolkit workflows located in the [`examples`](./examples) directory of the source repository.
* 🛠️ [Create and Customize NeMo Agent Toolkit Workflows](docs/source/get-started/tutorials/customize-a-workflow.md): Learn how to create and customize NeMo Agent Toolkit workflows.
* 🎯 [Evaluate with NeMo Agent Toolkit](./docs/source/improve-workflows/evaluate.md): Learn how to evaluate your NeMo Agent Toolkit workflows.
* 🆘 [Troubleshooting](./docs/source/resources/troubleshooting.md): Get help with common issues.


## 🛣️ Roadmap

- [x] Automatic Reinforcement Learning (RL) to fine-tune LLMs for a specific agent.
- [x] Integration with [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) to reduce LLM latency at scale.
- [ ] Improve agent throughput with KV-Cache optimization.
- [ ] Integration with [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) to improve agent safety and security.
- [ ] Improved memory interface to support self-improving agents.

## 💬 Feedback

We would love to hear from you! Please file an issue on [GitHub](https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues) if you have any feedback or feature requests.

## 🤝 Acknowledgements

We would like to thank the following groups for their contribution to the toolkit:

- [Synopsys](https://www.synopsys.com/)
  - Google ADK framework support.
  - Microsoft AutoGen framework support.
- [W&B Weave Team](https://wandb.ai/site/weave/)
  - Contributions to the evaluation and telemetry system.

In addition, we would like to thank the following open source projects that made NeMo Agent Toolkit possible:

- [Agent2Agent (A2A) Protocol](https://github.com/a2aproject/A2A)
- [CrewAI](https://github.com/crewAIInc/crewAI)
- [Dynamo](https://github.com/ai-dynamo/dynamo)
- [FastAPI](https://github.com/tiangolo/fastapi)
- [Google Agent Development Kit (ADK)](https://github.com/google/adk-python)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Llama-Index](https://github.com/run-llama/llama_index)
- [Mem0ai](https://github.com/mem0ai/mem0)
- [Microsoft AutoGen](https://github.com/microsoft/autogen)
- [MinIO](https://github.com/minio/minio)
- [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol/modelcontextprotocol)
- [OpenTelemetry](https://github.com/open-telemetry/opentelemetry-python)
- [Phoenix](https://github.com/arize-ai/phoenix)
- [Ragas](https://github.com/explodinggradients/ragas)
- [Redis](https://github.com/redis/redis-py)
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel)
- [Strands](https://github.com/strands-agents/sdk-python)
- [uv](https://github.com/astral-sh/uv)
- [Weave](https://github.com/wandb/weave)
