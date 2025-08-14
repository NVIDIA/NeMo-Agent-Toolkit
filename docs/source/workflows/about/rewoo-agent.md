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

# ReWOO Agent
The ReWOO (Reasoning WithOut Observation) agent is an advanced AI system that decouples reasoning from observations to improve efficiency in augmented language models. Based on the [ReWOO paper](https://arxiv.org/abs/2305.18323), this agent separates the planning and execution phases to reduce token consumption and improve performance.

The ReWOO agent's implementation follows the paper's methodology of decoupling reasoning from observations, which leads to more efficient tool usage and better performance in complex reasoning tasks.


## Features
- **Decoupled Architecture**: Separates planning and execution phases for improved efficiency
- **Pre-built Tools**: Leverages core library agent and tools
- **Efficient Token Usage**: Reduces token consumption by decoupling reasoning from observations
- **Custom Plugin System**: Developers can bring in new tools using plugins
- **Customizable Prompts**: Modify planner and solver prompts for specific needs
- **Agentic Workflows**: Fully configurable via YAML for flexibility and productivity
- **Ease of Use**: Simplifies developer experience and deployment

## Benefits

* **Token Efficiency**: By planning all steps upfront and using placeholders (e.g., "#E1", "#E2") for intermediate results, ReWOO significantly reduces token consumption. These placeholders are replaced with actual values during execution, eliminating the need to include full tool outputs in each reasoning step.

* **Cleaner Reasoning**: The separation of planning and execution allows the agent to focus purely on logical reasoning during the planning phase, without being distracted by intermediate results. The placeholder system makes data flow between steps explicit and manageable.

* **Reduced Hallucination**: By having a clear plan before execution, the agent is less likely to make incorrect assumptions or get sidetracked by intermediate results.

## Configuration

The ReWOO agent may be utilized as a workflow or a function.

### Example `config.yml`
In your YAML file, to use the ReWOO agent (`rewoo_agent`) as a workflow:
```yaml
workflow:
  _type: rewoo_agent
  tool_names: [wikipedia_search, current_datetime, code_generation, math_agent]
  llm_name: nim_llm
  verbose: true
  use_tool_schema: true
```

In your YAML file, to use the ReWOO agent as a function:
```yaml
functions:
  calculator_multiply:
    _type: calculator_multiply
  calculator_inequality:
    _type: calculator_inequality
  calculator_divide:
    _type: nat_simple_calculator/calculator_divide
  math_agent:
    _type: rewoo_agent
    tool_names:
      - calculator_multiply
      - calculator_inequality
      - calculator_divide
    description: 'Useful for performing simple mathematical calculations.'
```

### Configurable Options:

* `tool_names`: A list of tools that the agent can call. The tools must be functions configured in the YAML file

* `llm_name`: The LLM the agent should use. The LLM must be configured in the YAML file

* `verbose`: Defaults to False (useful to prevent logging of sensitive data). If set to True, the agent will log input, output, and intermediate steps.

* `include_tool_input_schema_in_tool_description`: Defaults to True. If set to True, the agent will include tool input schemas in tool descriptions.

* `description`: Defaults to "ReWOO Agent Workflow". When the ReWOO agent is configured as a function, this config option allows us to control the tool description (for example, when used as a tool within another agent).

* `planner_prompt`: Optional. Allows us to override the planner prompt for the ReWOO agent. The prompt must have variables for tools and must instruct the LLM to output in the ReWOO planner format.

* `solver_prompt`: Optional. Allows us to override the solver prompt for the ReWOO agent. The prompt must have variables for plan and task.

* `max_history`:  Defaults to 15. Maximum number of messages to keep in the conversation history.

* `use_openai_api`: Defaults to False. If set to True, the ReWOO agent will output in OpenAI API spec. If set to False, strings will be used.

* `additional_planner_instructions`: Optional. Defaults to `None`. Additional instructions to provide to the agent in addition to the base planner prompt.

* `additional_solver_instructions`: Optional. Defaults to `None`. Additional instructions to provide to the agent in addition to the base solver prompt.


## **Step-by-Step Breakdown of a ReWOO Agent**

1. **Planning Phase** – The agent receives a task and creates a complete plan with all necessary tool calls and evidence placeholders.
2. **Execution Phase** – The agent executes each step of the plan sequentially, replacing placeholders with actual tool outputs.
3. **Solution Phase** – The agent uses all gathered evidence to generate the final answer.

### Example Walkthrough

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
1. Executes the first step to get today's date
2. Uses that date to search for historical weather data
3. Replaces placeholders with actual results

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

* Sequential Execution: ReWOO agents execute steps sequentially, which means they cannot take advantage of parallel execution opportunities. This can lead to longer execution times for tasks that could be parallelized.

* Planning Overhead: The initial planning phase requires the agent to think through the entire task before starting execution. This can be inefficient for simple tasks that could be solved with fewer steps.

* Limited Adaptability: Since the plan is created upfront, the agent cannot easily adapt to unexpected tool failures or new information that might require a different approach.

* Complex Planning Requirements: The planning phase requires the agent to have a good understanding of all available tools and their capabilities. Poor tool descriptions or complex tool interactions can lead to suboptimal plans.

* Memory Constraints: The agent needs to maintain the entire plan and all intermediate results in memory, which could be challenging for very long or complex tasks.

In summary, ReWOO agents are most effective for tasks that benefit from upfront planning and where token efficiency is important. They may not be the best choice for tasks requiring high adaptability or parallel execution.
