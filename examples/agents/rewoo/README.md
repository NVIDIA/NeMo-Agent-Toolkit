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

<!--
  SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# ReWOO Agent

A configurable ReWOO Agent. This agent leverages the AgentIQ plugin system and `WorkflowBuilder` to integrate pre-built and custom tools into the workflow. Key elements are summarized below:

## Key Features

- **Pre-built Tools:** Leverages core AgentIQ library agent and tools.
- **ReWOO Agent:** The ReWOO pattern eliminates the need to include the system prompt and all previous steps for every reasoning iteration, thereby reducing token usage and boosting performance.
- **Custom Plugin System:** Developers can bring in new tools using plugins.
- **High-level API:** Enables defining functions that transform into asynchronous LangChain tools.
- **Agentic Workflows:** Fully configurable via YAML for flexibility and productivity.
- **Ease of Use:** Simplifies developer experience and deployment.

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/intro/install.md) to create the development environment and install AgentIQ.

### Install this Workflow:

From the root directory of the AgentIQ library, run the following commands:

```bash
uv pip install -e .
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/intro/get-started.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

Prior to using the `tavily_internet_search` tool, create an account at [`tavily.com``](https://tavily.com/) and obtain an API key. Once obtained, set the `TAVILY_API_KEY` environment variable to the API key:
```bash
export TAVILY_API_KEY=<YOUR_TAVILY_API_KEY>
```
---

Run the following command from the root of the AgentIQ repo to execute this workflow with the specified input:

```bash
aiq run  --config_file=examples/agents/rewoo/configs/config.yml --input "Which city held the Olympic game in the year represented by the bigger number of 1996 and 2004?"
```

**Expected Output**

```console
$ aiq run  --config_file=examples/agents/rewoo/configs/config.yml --input "Which city held the Olympic game in the year represented by the bigger number of 1996 and 2004?"
2025-04-21 10:37:04,178 - aiq.runtime.loader - WARNING - Loading module 'aiq_automated_description_generation.register' from entry point 'aiq_automated_description_generation' took a long time (321.641922 ms). Ensure all imports are inside your registered functions.
2025-04-21 10:37:04,370 - aiq.cli.commands.start - INFO - Starting AgentIQ from config file: 'examples/agents/rewoo/configs/config.yml'
2025-04-21 10:37:04,375 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
2025-04-21 10:37:04,422 - haystack.tracing.tracer - INFO - Auto-enabled tracing for 'OpenTelemetryTracer'
2025-04-21 10:37:04,427 - aiq.profiler.decorators.framework_wrapper - INFO - Langchain callback handler registered
2025-04-21 10:37:04,446 - aiq.agent.rewoo_agent.agent - INFO - Filling the prompt variables "tools" and "tool_names", using the tools provided in the config.
2025-04-21 10:37:04,446 - aiq.agent.rewoo_agent.agent - INFO - Adding the tools' input schema to the tools' description
2025-04-21 10:37:04,446 - aiq.agent.rewoo_agent.agent - INFO - Initialized ReWOO Agent Graph
2025-04-21 10:37:04,451 - aiq.agent.rewoo_agent.agent - INFO - ReWOO Graph built and compiled successfully
2025-04-21 10:37:04,451 - aiq.agent.rewoo_agent.agent - INFO - ReWOO Graph built and compiled successfully

Configuration Summary:
--------------------
Workflow Type: rewoo_agent
Number of Functions: 6
Number of LLMs: 1
Number of Embedders: 0
Number of Memory: 0
Number of Retrievers: 0

2025-04-21 10:37:04,452 - aiq.front_ends.console.console_front_end_plugin - INFO - Processing input: ('Which city held the Olympic game in the year represented by the bigger number of 1996 and 2004?',)
2025-04-21 10:37:06,449 - aiq.agent.rewoo_agent.agent - INFO - The task was: Which city held the Olympic game in the year represented by the bigger number of 1996 and 2004?
2025-04-21 10:37:06,450 - aiq.agent.rewoo_agent.agent - INFO - The planner's thoughts are:
[
  {
    "plan": "Compare the numbers 1996 and 2004 to determine which one is bigger.",
    "evidence": {
      "placeholder": "#E1",
      "tool": "calculator_inequality",
      "tool_input": {"text": "2004 > 1996"}
    }
  },
  {
    "plan": "Since 2004 is bigger, search for the city that held the Olympic Games in 2004.",
    "evidence": {
      "placeholder": "#E2",
      "tool": "internet_search",
      "tool_input": {"question": "Which city held the Olympic Games in 2004?"}
    }
  }
]
2025-04-21 10:37:06,451 - aiq.agent.rewoo_agent.agent - INFO - Calling tool calculator_inequality with input: {'text': '2004 > 1996'}
2025-04-21 10:37:06,451 - aiq.agent.rewoo_agent.agent - INFO - Tool input is already a dictionary. Use the tool input as is.
2025-04-21 10:37:06,458 - aiq.agent.rewoo_agent.agent - INFO - Calling tool internet_search with input: {'question': 'Which city held the Olympic Games in 2004?'}
2025-04-21 10:37:06,458 - aiq.agent.rewoo_agent.agent - INFO - Tool input is already a dictionary. Use the tool input as is.
2025-04-21 10:37:09,978 - aiq.observability.async_otel_listener - INFO - Intermediate step stream completed. No more events will arrive.
2025-04-21 10:37:09,978 - aiq.front_ends.console.console_front_end_plugin - INFO - --------------------------------------------------
Workflow Result:
['Athens']
--------------------------------------------------
```
---

### Starting the AgentIQ Server

You can start the AgentIQ server using the `aiq serve` command with the appropriate configuration file.

**Starting the ReWOO Agent Example Workflow**

```bash
aiq serve --config_file=examples/agents/rewoo/configs/config.yml
```

### Making Requests to the AgentIQ Server

Once the server is running, you can make HTTP requests to interact with the workflow.

#### Non-Streaming Requests

**Non-Streaming Request to the ReWOO Agent Example Workflow**

```bash
curl --request POST \
  --url http://localhost:8000/generate \
  --header 'Content-Type: application/json' \
  --data '{"input_message": "Which city held the Olympic game in the year represented by the bigger number of 1996 and 2004?"}'
```

#### Streaming Requests

**Streaming Request to the ReWOO Agent Example Workflow**

```bash
curl --request POST \
  --url http://localhost:8000/generate/stream \
  --header 'Content-Type: application/json' \
  --data '{"input_message": "Which city held the Olympic game in the year represented by the bigger number of 1996 and 2004?"}'
```
---

### Evaluating the ReWOO Agent Workflow
**Run and evaluate the `rewoo_agent` example Workflow**

```bash
aiq eval --config_file=examples/agents/rewoo/configs/config.yml
```
