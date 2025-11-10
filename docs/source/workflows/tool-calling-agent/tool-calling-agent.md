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

# Configure the Tool Calling Agent
Configure the NVIDIA NeMo Agent toolkit ReAct agent as a workflow or a function.

## Requirements
The tool calling agent requires the `nvidia-nat[langchain]` plugin, which can be installed with one of the following commands.

- If you have performed a source code checkout:

```bash
uv pip install -e '.[langchain]'
```

- If you have installed the NeMo Agent toolkit from a package:

```bash
uv pip install "nvidia-nat[langchain]"
```

## Configuration
The tool calling agent may be utilized as a workflow or a function, by entering the value `tool_calling_agent` for the `_type` option, as shown in the example configurations below.

### Example 1: Tool Calling Agent as a Workflow to Configure `config.yml`
To use the ReAct agent as a workflow, configure the YAML file as follows:
```yaml
workflow:
  _type: tool_calling_agent
  tool_names: [wikipedia_search, current_datetime, code_generation]
  llm_name: nim_llm
  verbose: true
  handle_tool_errors: true
```

### Example 2: Tool Calling Agent as a Function to Configure `config.yml`
In your YAML file, to use the tool calling agent as a function:
```yaml
function_groups:
  calculator:
    _type: calculator
functions:
  math_agent:
    _type: tool_calling_agent
    tool_names: [calculator]
    llm_name: agent_llm
    verbose: true
    handle_tool_errors: true
    description: 'Useful for performing simple mathematical calculations.'
```

### Configurable Options

* `workflow_alias`: Defaults to `None`. The alias of the workflow. Useful when the Tool Calling agent is configured as a workflow and need to expose a customized name as a tool.

* `tool_names`: A list of tools that the agent can call. The tools must be functions or function groups configured in the YAML file

* `llm_name`: The LLM the agent should use. The LLM must be configured in the YAML file

* `verbose`: Defaults to False (useful to prevent logging of sensitive data). If set to True, the agent will log input, output, and intermediate steps.

* `handle_tool_errors`: Defaults to True. All tool errors will be caught and a `ToolMessage` with an error message will be returned, allowing the agent to retry.

* `max_iterations`: Defaults to 15. The maximum number of tool calls the agent may perform.

* `return_direct`: Optional list of tool names that should return their output directly without additional agent processing. When a tool in this list is called, its response is returned immediately to the user, bypassing the agent's reasoning step.

* `description`:  Defaults to "Tool Calling Agent Workflow". When the agent is configured as a function, this config option allows us to control the tool description (for example, when used as a tool within another agent).

---

## Step-by-Step Breakdown of a Tool-Calling Agent

1. **User Query** – The agent receives an input or problem to solve.
2. **Function Matching** – The agent determines the best tool to call based on the input.
3. **Tool Execution** – The agent calls the tool with the necessary parameters.
4. **Response Handling** – The tool returns a structured response, which the agent passes to the user.

### **Example Walkthrough**

Imagine a tool-calling agent needs to answer:

> "What’s the current weather in New York?"

#### Single Step Execution
1. **User Query:** "What’s the current weather in New York?"
2. **Function Matching:** The agent identifies the `get_weather(location)` tool.
3. **Tool Execution:** Calls `get_weather("New York")`.
4. **Response Handling:** The tool returns `72°F, clear skies`, and the agent directly provides the answer.

Since tool calling agents execute function calls directly, they are more efficient for structured tasks that don’t require intermediate reasoning.

---

## Limitations
The following are the limitations of tool calling agents:

* Requires an LLM that supports tool calling or function calling.

* Does not perform complex reasoning and decision-making between tool calls.

* Since it uses the tool name, description, and input parameters, it requires well-named input parameters for each tool.
