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

* `tool_names`: A list of tools that the agent can call. The tools must be functions configured in the YAML file

* `llm_name`: The LLM the agent should use. The LLM must be configured in the YAML file

* `verbose`: Defaults to False (useful to prevent logging of sensitive data). If set to True, the agent will log input, output, and intermediate steps.

* `include_tool_input_schema_in_tool_description`: Defaults to True. If set to True, the agent will include tool input schemas in tool descriptions.

* `description`: Defaults to "ReWOO Agent Workflow". When the ReWOO agent is configured as a function, this config option allows us to control the tool description (for example, when used as a tool within another agent).

* `planner_prompt`: Optional. Allows us to override the planner prompt for the ReWOO agent. The prompt must have variables for tools and must instruct the LLM to output in the ReWOO planner format.

* `solver_prompt`: Optional. Allows us to override the solver prompt for the ReWOO agent. The prompt must have variables for plan and task.

* `tool_call_max_retries`: Defaults to 3. The number of retries before raising a tool call error.

* `max_history`:  Defaults to 15. Maximum number of messages to keep in the conversation history.

* `log_response_max_chars`: Defaults to 1000. Maximum number of characters to display in logs when logging tool responses.

* `use_openai_api`: Defaults to False. If set to True, the ReWOO agent will output in OpenAI API spec. If set to False, strings will be used.

* `additional_planner_instructions`: Optional. Defaults to `None`. Additional instructions to provide to the agent in addition to the base planner prompt.

* `additional_solver_instructions`: Optional. Defaults to `None`. Additional instructions to provide to the agent in addition to the base solver prompt.

* `raise_tool_call_error`: Defaults to True. Whether to raise a exception immediately if a tool call fails. If set to False, the tool call error message will be included in the tool response and passed to the next tool.

## Run the Workflow

Run the following command from the root of the NeMo Agent toolkit repo to execute this workflow with the specified input:

```bash
nat run --config_file=examples/agents/rewoo/configs/config.yml --input "Make a joke comparing Elon and Mark Zuckerberg's birthdays?"
```

**Expected Workflow Output**
```console
<snipped for brevity>

- ReWOO agent output:
------------------------------
[AGENT]
Agent input: Make a joke comparing Elon and Mark Zuckerberg's birthdays?
Agent's thoughts: 
[
  {
    "plan": "Find Elon Musk's birthday",
    "evidence": {
      "placeholder": "#E1",
      "tool": "internet_search",
      "tool_input": {"question": "Elon Musk birthday"}
    }
  },
  {
    "plan": "Find Mark Zuckerberg's birthday",
    "evidence": {
      "placeholder": "#E2",
      "tool": "internet_search",
      "tool_input": {"question": "Mark Zuckerberg birthday"}
    }
  },
  {
    "plan": "Compare the birthdays and create a joke",
    "evidence": {
      "placeholder": "#E3",
      "tool": "haystack_chitchat_agent",
      "tool_input": {"inputs": "Compare the birthdays of Elon Musk (#E1) and Mark Zuckerberg (#E2) and create a joke"}
    }
  }
]
------------------------------
2025-09-27 20:12:14,522 - nat.agent.rewoo_agent.agent - INFO - ReWOO agent execution levels: [['#E1', '#E2'], ['#E3']]
/raid/binfeng/workspace/NeMo-Agent-Toolkit/packages/nvidia_nat_langchain/src/nat/plugins/langchain/tools/tavily_internet_search.py:45: LangChainDeprecationWarning: The class `TavilySearchResults` was deprecated in LangChain 0.3.25 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-tavily package and should be used instead. To use it run `pip install -U :class:`~langchain-tavily` and import as `from :class:`~langchain_tavily import TavilySearch``.
  tavily_search = TavilySearchResults(max_results=tool_config.max_results)
2025-09-27 20:12:15,939 - nat.agent.base - INFO - 
------------------------------
[AGENT]
Calling tools: internet_search
Tool's input: {'question': 'Elon Musk birthday'}
Tool's response: 
content='<Document href="https://www.ebsco.com/research-starters/biography/elon-musk"/>\nElon Musk was born on June 28, 1971, in Pretoria, Transvaal (now Gauteng), South Africa, one of three children born to Canadian model and dietitian Maye Musk (née Haldeman) and South African engineer Errol Musk, now divorced. He left high school and emigrated from South Africa to Canada in 1988 at the age of seventeen, primarily because he objected philosophically to mandatory conscription into the South African military, which at the time was the primary enforcement vehicle for apartheid.\n</Document>\n\n---\n\n<Document href="https://www.jagranjosh.com/general-knowledge/elon-reeve-musk-1588776062-1"/>\nElon Musk, born on June 28, 1971, in Pretoria, South Africa, is a prominent entrepreneur known for his roles in companies like Tesla, SpaceX, and Neuralink. Recently, he welcomed his 14th child, Seldon Lycurgus, with Shivon Zilis, an executive at Neuralink.\n\nThe name "Seldon" is believed to be in...(rest of response truncated)
------------------------------
2025-09-27 20:12:17,757 - nat.agent.base - INFO - 
------------------------------
[AGENT]
Calling tools: internet_search
Tool's input: {'question': 'Mark Zuckerberg birthday'}
Tool's response: 
content='<Document href="https://simple.wikipedia.org/wiki/Mark_Zuckerberg"/>\n| Mark Zuckerberg | |\n --- |\n| Zuckerberg in 2020 | |\n| Born | Mark Elliot Zuckerberg   (1984-05-14) May 14, 1984 (age 41)  White Plains, New York, USA |\n| Education | Harvard University (no degree) |\n| Occupations |  Internet entrepreneur")  philanthropist  media mogul |\n| Years active | 2004–present |\n| Known for | Co-founding and leading Meta, Inc. |\n| Height | 171 cm (5 ft 7 in) |\n| Title |  Founder and CEO of Meta, Inc.  Co-founder and co-CEO of Chan Zuckerberg Initiative | [...] Mark Elliot Zuckerberg (born White Plains, New York, 1984) is an American who created Facebook when he was still studying computer science. The founding of Facebook made Zuckerberg a billionaire, one of the youngest and richest billionaires of all time according to Forbes.\n\nBesides computer programming, Zuckerberg is also interested in foreign languages, especially Mandarin Chinese. Mark Zuckerberg was born at White ...(rest of response truncated)
------------------------------
2025-09-27 20:12:17,757 - nat.agent.rewoo_agent.agent - INFO - [AGENT] Completed level 0 with 2 tools
2025-09-27 20:12:23,318 - nat_multi_frameworks.haystack_agent - INFO - output from langchain_research_tool: After comparing the birthdays of Elon Musk and Mark Zuckerberg, I found that:

Elon Musk was born on June 28, 1971
Mark Zuckerberg was born on May 14, 1984

Here's a joke:

Why did Elon Musk and Mark Zuckerberg go to therapy together?

Because Elon was feeling a little "spacey" (get it? SpaceX?) and Mark was having a "facebook" identity crisis... but in the end, they just realized they were born to be different - 13 years and 44 days apart, to be exact!
2025-09-27 20:12:23,318 - nat.agent.base - INFO - 
------------------------------
[AGENT]
Calling tools: haystack_chitchat_agent
Tool's input: {'inputs': 'Compare the birthdays of Elon Musk (<Document href="https://www.ebsco.com/research-starters/biography/elon-musk"/>\nElon Musk was born on June 28, 1971, in Pretoria, Transvaal (now Gauteng), South Africa, one of three children born to Canadian model and dietitian Maye Musk (née Haldeman) and South African engineer Errol Musk, now divorced. He left high school and emigrated from South Africa to Canada in 1988 at the age of seventeen, primarily because he objected philosophically to mandatory conscription into the South African military, which at the time was the primary enforcement vehicle for apartheid.\n</Document>\n\n---\n\n<Document href="https://www.jagranjosh.com/general-knowledge/elon-reeve-musk-1588776062-1"/>\nElon Musk, born on June 28, 1971, in Pretoria, South Africa, is a prominent entrepreneur known for his roles in companies like Tesla, SpaceX, and Neuralink. Recently, he welcomed his 14th child, Seldon Lycurgus, with Shivon Zilis, an executive at Neuralink.\n\nThe name "Seldon" is believed to be inspired by Hari Seldon, a character from Isaac Asimov\'s "Foundation" series, while "Lycurgus" refers to the ancient Spartan lawgiver. [...] Elon Reeve Musk was born on June 28, 1971, in Pretoria, South Africa. He is the eldest of three siblings in a family with diverse talents and interests.\n\nHis early life was marked by intellectual curiosity but also challenges, including bullying at school and a difficult relationship with his father. Musk showed an early aptitude for technology and entrepreneurship, creating and selling a video game called Blastar at the age of 12.\n\n### Parents [...] +\n\n  Elon Reeve Musk was born on June 28, 1971, in Pretoria, South Africa to Maye Musk and Errol Musk.\n\nGet here current GK and GK quiz questions in English and Hindi for India, World, Sports and Competitive exam preparation. Download the Jagran Josh Current Affairs App.\n\n## Trending\n</Document>\n\n---\n\n<Document href="https://en.wikipedia.org/wiki/Elon_Musk"/>\nElon Reeve Musk was born on June 28, 1971, in Pretoria, South Africa\'s administrative capital.( He is of British and Pennsylvania Dutch ancestry.( His mother, Maye (néeHaldeman), is a model and dietitian born in Saskatchewan, Canada, and raised in South Africa.( Musk therefore holds both South African and Canadian citizenship from birth.( His father, Errol Musk, is a South African electromechanical engineer, pilot, sailor, consultant, emerald dealer, and property developer, who partly owned a [...] Elon Reeve Musk (/ˈ iː l ɒ n/_EE-lon_; born June 28, 1971) is an international businessman and entrepreneur known for his leadership of Tesla, SpaceX, X (formerly Twitter) "X (formerly Twitter)"), and the Department of Government Efficiency (DOGE). Musk has been the wealthiest person in the world since 2021; as of May 2025,( estimates his net worth to be US$424.7 billion. [...] | Image 5_(cropped).jpg) Musk in 2022 |\n|  |\n| Senior Advisor to the President |\n| In office January 20, 2025– May 30, 2025 Serving with Massad Boulos |\n| President | Donald Trump |\n| Preceded by | Tom Perez |\n|  |\n| Personal details |\n| Born | Elon Reeve Musk (1971-06-28) June 28, 1971 (age 54) Pretoria, South Africa |\n| Citizenship |  South Africa  Canada  United States |\n| Political party | Independent |\n</Document>) and Mark Zuckerberg (<Document href="https://simple.wikipedia.org/wiki/Mark_Zuckerberg"/>\n| Mark Zuckerberg | |\n --- |\n| Zuckerberg in 2020 | |\n| Born | Mark Elliot Zuckerberg   (1984-05-14) May 14, 1984 (age 41)  White Plains, New York, USA |\n| Education | Harvard University (no degree) |\n| Occupations |  Internet entrepreneur")  philanthropist  media mogul |\n| Years active | 2004–present |\n| Known for | Co-founding and leading Meta, Inc. |\n| Height | 171 cm (5 ft 7 in) |\n| Title |  Founder and CEO of Meta, Inc.  Co-founder and co-CEO of Chan Zuckerberg Initiative | [...] Mark Elliot Zuckerberg (born White Plains, New York, 1984) is an American who created Facebook when he was still studying computer science. The founding of Facebook made Zuckerberg a billionaire, one of the youngest and richest billionaires of all time according to Forbes.\n\nBesides computer programming, Zuckerberg is also interested in foreign languages, especially Mandarin Chinese. Mark Zuckerberg was born at White Plains Hospital in White Plains, New York but now he lives in California.\n</Document>\n\n---\n\n<Document href="https://en.wikipedia.org/wiki/Mark_Zuckerberg"/>\nMark Elliot Zuckerberg (/ˈzʌkərbɜːrɡ/; born May 14, 1984) is an American businessman who co-founded the social media service Facebook and its parent company Meta Platforms, of which he is the chairman, chief executive officer, and controlling shareholder. [...] ## Early life\n\nMark Elliot Zuckerberg was born on May 14, 1984, in White Plains, New York, to psychiatrist Karen (née Kempner) and dentist Edward Zuckerberg. He and his three sisters (Arielle, Randi, and Donna) were raised in a Reform Jewish household in Dobbs Ferry, New York. Their great-grandparents were emigrants from Austria, Germany, and Poland. Zuckerberg initially attended Ardsley High School before transferring to Phillips Exeter Academy. He was captain of the fencing team.\n</Document>\n\n---\n\n<Document href="https://www.biography.com/business-leaders/mark-zuckerberg"/>\nMark Elliot Zuckerberg was born on May 14, 1984, in White Plains, New York, into a comfortable, well-educated family. His father, Edward, ran a dental practice attached to the family’s home, and his mother, Karen, worked as a psychiatrist before becoming a stay-at-home mom. He was raised in the Westchester village of Dobbs Ferry with his three siblings Randi, Donna, and Arielle. [...] ## Quick Facts\n\nFULL NAME: Mark Elliot Zuckerberg BORN: May 14, 1984BIRTHPLACE: White Plains, New YorkSPOUSE: Priscilla Chan (2012-present)CHILDREN: Maxima, August, and AureliaASTROLOGICAL SIGN: Taurus\n\n## Early Life [...] In December 2015, the couple welcomed their first child, a daughter named Maxima, Max for short. Zuckerberg and Chan had two more daughters together: August (named after her birth month), born in August 2017, and Aurelia, born in March 2023.\n\n## Net Worth\n</Document>) and create a joke'}
Tool's response: 
content='After comparing the birthdays of Elon Musk and Mark Zuckerberg, I found that:\n\nElon Musk was born on June 28, 1971\nMark Zuckerberg was born on May 14, 1984\n\nHere\'s a joke:\n\nWhy did Elon Musk and Mark Zuckerberg go to therapy together?\n\nBecause Elon was feeling a little "spacey" (get it? SpaceX?) and Mark was having a "facebook" identity crisis... but in the end, they just realized they were born to be different - 13 years and 44 days apart, to be exact!' name='haystack_chitchat_agent' tool_call_id='haystack_chitchat_agent'
------------------------------
2025-09-27 20:12:23,319 - nat.agent.rewoo_agent.agent - INFO - [AGENT] Completed level 1 with 1 tools
2025-09-27 20:12:24,472 - nat.agent.rewoo_agent.agent - INFO - ReWOO agent solver output: 
------------------------------
[AGENT]
Agent input: Make a joke comparing Elon and Mark Zuckerberg's birthdays?
Agent's thoughts: 
Why did Elon Musk and Mark Zuckerberg go to therapy together? Because Elon was feeling a little "spacey" and Mark was having a "facebook" identity crisis... but in the end, they just realized they were born to be different - 13 years and 44 days apart, to be exact!
------------------------------
2025-09-27 20:12:24,473 - nat.front_ends.console.console_front_end_plugin - INFO - 
--------------------------------------------------
Workflow Result:
['Why did Elon Musk and Mark Zuckerberg go to therapy together? Because Elon was feeling a little "spacey" and Mark was having a "facebook" identity crisis... but in the end, they just realized they were born to be different - 13 years and 44 days apart, to be exact!']
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
  --data "{\"input_message\": \"Make a joke comparing Elon and Mark Zuckerberg's birthdays?\"}"
```

#### Streaming Requests

**Streaming Request to the ReWOO Agent Example Workflow**

```bash
curl --request POST \
  --url http://localhost:8000/generate/stream \
  --header 'Content-Type: application/json' \
  --data "{\"input_message\": \"Make a joke comparing Elon and Mark Zuckerberg's birthdays?\"}"
```
---

### Evaluating the ReWOO Agent Workflow
**Run and evaluate the `rewoo_agent` example Workflow**

```bash
nat eval --config_file=examples/agents/rewoo/configs/config.yml
```
