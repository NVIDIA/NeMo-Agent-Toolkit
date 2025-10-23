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

# ReWOO Agent Example

This example demonstrates how to use A configurable [ReWOO](https://arxiv.org/abs/2305.18323) (Reasoning WithOut Observation) Agent with the NeMo Agent toolkit. For this purpose NeMo Agent toolkit provides a [`rewoo_agent`](../../../docs/source/workflows/about/rewoo-agent.md) workflow type.

## Table of Contents

- [Key Features](#key-features)
- [Graph Structure](#graph-structure)
- [Installation and Setup](#installation-and-setup)
  - [Install this Workflow](#install-this-workflow)
  - [Set Up API Keys](#set-up-api-keys)
- [Run the Workflow](#run-the-workflow)
  - [Starting the NeMo Agent Toolkit Server](#starting-the-nemo-agent-toolkit-server)
  - [Making Requests to the NeMo Agent Toolkit Server](#making-requests-to-the-nemo-agent-toolkit-server)
  - [Evaluating the ReWOO Agent Workflow](#evaluating-the-rewoo-agent-workflow)

## Key Features

- **ReWOO Agent Architecture:** Demonstrates the unique `rewoo_agent` workflow type that implements Reasoning Without Observation, separating planning, execution, and solving into distinct phases.
- **Three-Node Graph Structure:** Uses a distinctive architecture with Planner Node (creates complete execution plan), Executor Node (executes tools systematically), and Solver Node (synthesizes final results).
- **Systematic Tool Execution:** Shows how ReWOO first plans all necessary steps upfront, then executes them systematically without dynamic re-planning, leading to more predictable tool usage patterns.
- **Calculator and Internet Search Integration:** Includes `calculator_inequality` and `internet_search` tools to demonstrate multi-step reasoning that requires both mathematical computation and web research.
- **Plan-Execute-Solve Pattern:** Demonstrates the ReWOO approach of complete upfront planning followed by systematic execution and final result synthesis.

## Graph Structure

The ReWOO agent uses a unique three-node graph architecture that separates planning, execution, and solving into distinct phases. The following diagram illustrates the agent's workflow:

<div align="center">
<img src="../../../docs/source/_static/rewoo_agent.png" alt="ReWOO Agent Graph Structure" width="400" style="max-width: 100%; height: auto;">
</div>

**Workflow Overview:**
- **Start**: The agent begins processing with user input
- **Planner Node**: Creates a complete execution plan with all necessary steps upfront. Plans are parsed into a Dependency Graph for parallel execution. 
- **Executor Node**: Executes tools according to the plan. Non-dependent tool calls are executed in parallel at each level.
- **Solver Node**: Takes all execution results and generates the final answer
- **End**: Process completes with the final response

This architecture differs from other agents by separating reasoning (planning) from execution, allowing for more systematic and predictable tool usage patterns. The ReWOO approach first plans all steps, then executes them systematically, and finally synthesizes the results.

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent toolkit.

### Install this Workflow

From the root directory of the NeMo Agent toolkit library, run the following commands:

```bash
uv sync --all-groups --all-extras
uv pip install -e .
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

Prior to using the `tavily_internet_search` tool, create an account at [`tavily.com``](https://tavily.com/) and obtain an API key. Once obtained, set the `TAVILY_API_KEY` environment variable to the API key:
```bash
export TAVILY_API_KEY=<YOUR_TAVILY_API_KEY>
```

## Configuration

The ReWOO agent is configured through the `config.yml` file. The following configuration options are available:

### Configurable Options

* `tool_names`: A list of tools that the agent can call. The tools must be functions or function groups configured in the YAML file

* `llm_name`: The LLM the agent should use. The LLM must be configured in the YAML file

* `verbose`: Defaults to False (useful to prevent logging of sensitive data). If set to True, the agent will log input, output, and intermediate steps.

* `include_tool_input_schema_in_tool_description`: Defaults to True. If set to True, the agent will include tool input schemas in tool descriptions.

* `description`: Defaults to "ReWOO Agent Workflow". When the ReWOO agent is configured as a function, this config option allows us to control the tool description (for example, when used as a tool within another agent).

* `planner_prompt`: Optional. Allows us to override the planner prompt for the ReWOO agent. The prompt must have variables for tools and must instruct the LLM to output in the ReWOO planner format.

* `solver_prompt`: Optional. Allows us to override the solver prompt for the ReWOO agent. The prompt must have variables for plan and task.

* `tool_call_max_retries`: Defaults to 3. The number of retries before raising a tool call error.

* `max_history`:  Defaults to 15. Maximum number of messages to keep in the conversation history.

* `log_response_max_chars`: Defaults to 1000. Maximum number of characters to display in logs when logging tool responses.

* `additional_planner_instructions`: Optional. Defaults to `None`. Additional instructions to provide to the agent in addition to the base planner prompt.

* `additional_solver_instructions`: Optional. Defaults to `None`. Additional instructions to provide to the agent in addition to the base solver prompt.

* `raise_tool_call_error`: Defaults to True. Whether to raise a exception immediately if a tool call fails. If set to False, the tool call error message will be included in the tool response and passed to the next tool.

## Run the Workflow

Run the following command from the root of the NeMo Agent toolkit repo to execute this workflow with the specified input:

```bash
nat run --config_file=examples/agents/rewoo/configs/config.yml --input "Who would be older today, Einstein or Bohr?"
```

**Expected Workflow Output**
```console
<snipped for brevity>

- ReWOO agent output:
------------------------------
[AGENT]
Agent input: Who would be older today, Einstein or Bohr?
Agent's thoughts: 
[
  {
    "plan": "Find Einstein's birthdate",
    "evidence": {
      "placeholder": "#E1",
      "tool": "internet_search",
      "tool_input": {"question": "Einstein birthdate"}
    }
  },
  {
    "plan": "Find Bohr's birthdate",
    "evidence": {
      "placeholder": "#E2",
      "tool": "internet_search",
      "tool_input": {"question": "Bohr birthdate"}
    }
  },
  {
    "plan": "Compare Einstein's and Bohr's birthdates to determine who would be older today",
    "evidence": {
      "placeholder": "#E3",
      "tool": "haystack_chitchat_agent",
      "tool_input": {"inputs": "Who would be older today, Einstein born #E1 or Bohr born #E2?"}
    }
  }
]
------------------------------
2025-10-14 19:14:02 - INFO     - nat.agent.rewoo_agent.agent:289 - ReWOO agent execution levels: [['#E1', '#E2'], ['#E3']]
2025-10-14 19:14:02 - INFO     - nat.agent.base:221 - 
------------------------------
[AGENT]
Calling tools: internet_search
Tool's input: {'question': 'Bohr birthdate'}
Tool's response: 
content='<Document href="https://www.facebook.com/TheWorldsofDavidDarling/posts/born-on-this-date-oct-7-in-1885-the-danish-physicist-niels-bohr-who-played-a-cru/1278740440721508/"/>\nNiels Bohr, in full Niels Henrik David Bohr, (born October 7, 1885, Copenhagen, Denmark—died November 18, 1962, Copenhagen), Danish\n</Document>\n\n---\n\n<Document href="https://en.wikipedia.org/wiki/Niels_Bohr"/>\n**Niels Henrik David Bohr** (Danish: ; 7 October 1885 – 18 November 1962) was a Danish theoretical physicist who made foundational contributions to understanding atomic structure and quantum theory, for which he received the Nobel Prize in Physics in 1922. J. Thomson (1914) * Ivan Pavlov (1915) * James Dewar (1916) * Pierre Paul Émile Roux (1917) * Hendrik Lorentz (1918) * William Bayliss (1919) * Horace Tabberer Brown (1920) * Joseph Larmor (1921) * Ernest Rutherford (1922) * Horace Lamb (1923) * Edward Albert Sharpey-Schafer (1924) * Albert Einstein (1925) * Frederick Gowland Hopkins (1926) *...(rest of response truncated)
------------------------------
2025-10-14 19:14:02 - INFO     - nat.agent.base:221 - 
------------------------------
[AGENT]
Calling tools: internet_search
Tool's input: {'question': 'Einstein birthdate'}
Tool's response: 
content='<Document href="https://www.facebook.com/albert.einstein.fans/posts/albert-einstein-was-born-on-march-14-1879-happy-birthday-/1204655314357103/"/>\nAlbert Einstein - Albert Einstein was born on March 14,... Albert Einstein\'s post ### **Albert Einstein** Albert Einstein was born on March 14, 1879. Happy birthday!! Image 1: 🎂Image 2: 🎉Image 3: 🎈 Image 4: No photo description available. Image 5 Image 6 67K 5.3K comments 9.1K shares A not well known fact that number Pi which is 3.14 is assigned after Einstein\'s birthday! Image 7Image 8Image 9 Happy heavenly birthday Mr. Einstein! Image 10: 🎂Image 11: 🎈 Image 12: GIFmedia1.tenor.co Image 13Image 14 happy birthday to me too! Image 15Image 16 Image 17Image 18 My birthday too though a bit later than 1879 Image 19: 😂 Image 20Image 21 Image 22 Image 23 Image 24 Image 25Image 26 Image 27\n</Document>\n\n---\n\n<Document href="https://en.wikipedia.org/wiki/Albert_Einstein"/>\nAlbert Einstein (14 March 1879 – 18 April 1955) was a German-...(rest of response truncated)
------------------------------
2025-10-14 19:14:02 - INFO     - nat.agent.rewoo_agent.agent:373 - [AGENT] Completed level 0 with 2 tools
2025-10-14 19:14:05 - INFO     - nat_multi_frameworks.haystack_agent:57 - output from langchain_research_tool: Based on the information provided, Albert Einstein was born on March 14, 1879, and Niels Bohr was born on October 7, 1885. Therefore, Einstein would be older than Bohr by approximately 6 years.
2025-10-14 19:14:05 - INFO     - nat.agent.base:221 - 
------------------------------
[AGENT]
Calling tools: haystack_chitchat_agent
Tool's input: {'inputs': 'Who would be older today, Einstein born <Document href="https://www.facebook.com/albert.einstein.fans/posts/albert-einstein-was-born-on-march-14-1879-happy-birthday-/1204655314357103/"/>\nAlbert Einstein - Albert Einstein was born on March 14,... Albert Einstein\'s post ### **Albert Einstein** Albert Einstein was born on March 14, 1879. Happy birthday!! Image 1: 🎂Image 2: 🎉Image 3: 🎈 Image 4: No photo description available. Image 5 Image 6 67K 5.3K comments 9.1K shares A not well known fact that number Pi which is 3.14 is assigned after Einstein\'s birthday! Image 7Image 8Image 9 Happy heavenly birthday Mr. Einstein! Image 10: 🎂Image 11: 🎈 Image 12: GIFmedia1.tenor.co Image 13Image 14 happy birthday to me too! Image 15Image 16 Image 17Image 18 My birthday too though a bit later than 1879 Image 19: 😂 Image 20Image 21 Image 22 Image 23 Image 24 Image 25Image 26 Image 27\n</Document>\n\n---\n\n<Document href="https://en.wikipedia.org/wiki/Albert_Einstein"/>\nAlbert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist ; Born in the German Empire ; In 1905, sometimes described as his annus\n</Document>\n\n---\n\n<Document href="https://www.facebook.com/WorldJewishCong/posts/today-is-the-birthday-of-albert-einstein-born-on-march-14-1879-one-of-the-greate/1051939843629078/"/>\nHe was born on March 14, 1879, in Ulm, in the Kingdom of Württemberg in the German Empire. Einstein is best known for his theory of relativity,\n</Document> or Bohr born <Document href="https://www.facebook.com/TheWorldsofDavidDarling/posts/born-on-this-date-oct-7-in-1885-the-danish-physicist-niels-bohr-who-played-a-cru/1278740440721508/"/>\nNiels Bohr, in full Niels Henrik David Bohr, (born October 7, 1885, Copenhagen, Denmark—died November 18, 1962, Copenhagen), Danish\n</Document>\n\n---\n\n<Document href="https://en.wikipedia.org/wiki/Niels_Bohr"/>\n**Niels Henrik David Bohr** (Danish: ; 7 October 1885 – 18 November 1962) was a Danish theoretical physicist who made foundational contributions to understanding atomic structure and quantum theory, for which he received the Nobel Prize in Physics in 1922. J. Thomson (1914) * Ivan Pavlov (1915) * James Dewar (1916) * Pierre Paul Émile Roux (1917) * Hendrik Lorentz (1918) * William Bayliss (1919) * Horace Tabberer Brown (1920) * Joseph Larmor (1921) * Ernest Rutherford (1922) * Horace Lamb (1923) * Edward Albert Sharpey-Schafer (1924) * Albert Einstein (1925) * Frederick Gowland Hopkins (1926) * Charles Scott Sherrington (1927) * Charles Algernon Parsons (1928) * Max Planck (1929) * William Henry Bragg (1930) * Arthur Schuster (1931) * George Ellery Hale (1932) * Theobald Smith (1933) * John Scott Haldane (1934) * Charles Thomson Rees Wilson (1935) * Arthur Evans (1936) * Henry Hallett Dale (1937) * Niels Bohr (1938) * Thomas Hunt Morgan (1939) * Paul Langevin (1940) * Thomas Lewis "Thomas Lewis (cardiologist)") (1941) * Robert Robinson "Robert Robinson (chemist)") (1942) * Joseph Barcroft (1943) * Geoffrey Ingram Taylor (1944) * Oswald Avery (1945) * Edgar Douglas Adrian (1946) * G.\n</Document>\n\n---\n\n<Document href="https://www.facebook.com/ictp.page/posts/happy-belated-birthday-to-niels-bohr-the-distinguished-danish-physicist-born-7-o/3631629133523362/"/>\n- ICTP: International Centre for Theoretical Physics | Facebook ICTP: International Centre for Theoretical Physics\'s post ### **ICTP: International Centre for Theoretical Physics** Happy (belated) Birthday to Niels Bohr! The distinguished Danish physicist, born 7 October 1885, made fundamental contributions to #atomic structure and #quantummechanics, was a #philosopher of #science, won the Physics #Nobel Prize in 1922, helped Jews escape the Nazis and helped #refugee scientists during WWII, and called for #international cooperation on #nuclearenergyImage 1: 🏆 I remember this one from university : An expert is someone who learns more and more about less and less, until eventually he knows everything about nothing. Happy Birthday. Or An expert is someone who knows more and more about less and less untill he knows every thing about nothing ! Happy Birthday!\n</Document>?'}
Tool's response: 
content='Based on the information provided, Albert Einstein was born on March 14, 1879, and Niels Bohr was born on October 7, 1885. Therefore, Einstein would be older than Bohr by approximately 6 years.' name='haystack_chitchat_agent' tool_call_id='haystack_chitchat_agent'
------------------------------
2025-10-14 19:14:05 - INFO     - nat.agent.rewoo_agent.agent:373 - [AGENT] Completed level 1 with 1 tools
2025-10-14 19:14:05 - INFO     - nat.agent.rewoo_agent.agent:493 - ReWOO agent solver output: 
------------------------------
[AGENT]
Agent input: Who would be older today, Einstein or Bohr?
Agent's thoughts: 
Einstein
------------------------------
2025-10-14 19:14:05 - WARNING  - nat.builder.intermediate_step_manager:94 - Step id 8660f3ce-1732-4951-9dbc-beea6f9a43ef not found in outstanding start steps
2025-10-14 19:14:05 - INFO     - nat.front_ends.console.console_front_end_plugin:102 - --------------------------------------------------
Workflow Result:
['Einstein']
```

### Starting the NeMo Agent Toolkit Server

You can start the NeMo Agent toolkit server using the `nat serve` command with the appropriate configuration file.

**Starting the ReWOO Agent Example Workflow**

```bash
nat serve --config_file=examples/agents/rewoo/configs/config.yml
```

### Making Requests to the NeMo Agent Toolkit Server

Once the server is running, you can make HTTP requests to interact with the workflow.

#### Non-Streaming Requests

**Non-Streaming Request to the ReWOO Agent Example Workflow**

```bash
curl --request POST \
  --url http://localhost:8000/generate \
  --header 'Content-Type: application/json' \
  --data "{\"input_message\": \"Who would be older today, Einstein or Bohr?\"}"
```

#### Streaming Requests

**Streaming Request to the ReWOO Agent Example Workflow**

```bash
curl --request POST \
  --url http://localhost:8000/generate/stream \
  --header 'Content-Type: application/json' \
  --data "{\"input_message\": \"Who would be older today, Einstein or Bohr?\"}"
```
---

### Evaluating the ReWOO Agent Workflow
**Run and evaluate the `rewoo_agent` example Workflow**

```bash
nat eval --config_file=examples/agents/rewoo/configs/config.yml
```
