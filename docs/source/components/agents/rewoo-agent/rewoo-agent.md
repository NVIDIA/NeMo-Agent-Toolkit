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

# Configure the ReWOO Agent
Configure the NVIDIA NeMo Agent toolkit ReWOO agent as a [workflow](../../../build-workflows/about-building-workflows.md) or a [function](../../../build-workflows/functions-and-function-groups/functions.md). The ReWOO agent workflow is broken down into three phases, planning, execution, and solution.

## Requirements
The ReWOO agent requires the `nvidia-nat[langchain]` plugin to be installed, which can be installed with one of the following commands.

- If you have performed a source code checkout, you can install this with the following command:

```bash
uv pip install -e '.[langchain]'
```

- If you have installed the NeMo Agent toolkit from a package, you can install this with the following command:

```bash
uv pip install "nvidia-nat[langchain]"
```
## Configuration

You can use the ReWOO agent as a workflow or a function.

### Example 1: ReWOO Agent as a Workflow to Configure `config.yml`
To use the ReWOO agent as a workflow, configure the YAML file as follows:
```yaml
workflow:
  _type: rewoo_agent
  tool_names: [wikipedia_search, current_datetime, code_generation, math_agent]
  llm_name: nim_llm
  verbose: true
  use_tool_schema: true
```
### Example 2: ReWOO Agent as a Function to Configure `config.yml`
In your YAML file, to use the ReWOO agent as a function:
```yaml
function_groups:
  calculator:
    _type: calculator
functions:
  math_agent:
    _type: rewoo_agent
    tool_names: [calculator]
    description: 'Useful for performing simple mathematical calculations.'
```

### Configurable Options
The following are more ways you can configure your config file when using the ReWOO agent:
* `workflow_alias`: Defaults to `None`. The alias of the workflow. Useful when the ReWOO agent is configured as a workflow and need to expose a customized name as a tool.

* `tool_names`: A list of [tools](../../../build-workflows/functions-and-function-groups/functions.md#agents-and-tools) that the agent can call. The tools must be [functions](../../../build-workflows/functions-and-function-groups/functions.md) or [function groups](../../../build-workflows/functions-and-function-groups/function-groups.md) configured in the YAML file.

* `llm_name`: The [LLM](../../../build-workflows/llms/index.md) the agent should use. The LLM must be configured in the YAML file

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

## Example ReWOO Agent Workflow

Imagine a ReWOO agent needs to answer:

> "What was the weather in New York last year on this date?"

#### Planning Phase
The agent creates a plan like:
```json
[
  {
    "plan": "Get today's date",
    "evidence": {
      "placeholder": "#E1",
      "tool": "current_datetime",
      "tool_input": {}
    }
  },
  {
    "plan": "Search for historical weather data",
    "evidence": {
      "placeholder": "#E2",
      "tool": "weather_search",
      "tool_input": "New York weather on #E1 last year"
    }
  }
]
```

#### Execution Phase
1. Execute the first step to get today's date.
2. Use that date to search for historical weather data.
3. Replace placeholders with actual results.

#### Solution Phase
Generates the final answer using all gathered evidence.

### ReWOO Prompting and Output Format

The ReWOO agent uses two distinct prompts:

* **Planner Prompt**: Generates a JSON array of planning steps, each containing:
   - A plan description
   - Evidence object with placeholder, tool name, and tool input

* **Solver Prompt**: Uses the plan and gathered evidence to generate the final answer.


## Limitations

ReWOO agents, while efficient, come with several limitations:

**Planning Overhead**: The initial planning phase requires the agent to think through the entire task before starting execution. This can be inefficient for simple tasks that could be solved with fewer steps.

**Limited Adaptability**: Since the plan is created upfront, the agent cannot easily adapt to unexpected tool failures or new information that might require a different approach.

**Complex Planning Requirements**: The planning phase requires the agent to have a good understanding of all available tools and their capabilities. Poor tool descriptions or complex tool interactions can lead to suboptimal plans.

In summary, ReWOO agents are most effective for tasks that benefit from upfront planning (relatively stable workflow) and where token efficiency is important. They may not be the best choice for tasks requiring high adaptability and uncertainty of tool outputs.
