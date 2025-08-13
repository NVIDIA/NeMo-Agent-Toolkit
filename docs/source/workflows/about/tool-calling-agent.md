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

# Tool Calling Agent

A tool calling agent is an AI system that directly invokes external tools based on structured function definitions. Unlike ReAct agents, it does not reason between steps but instead relies on predefined tool schemas to decide which tool to call. To decide which tools to use to answer the question, agent uses the name, description, and input parameter schema of each tool to decide which tools to use to answer the question. Not all LLMs support tool calling / function calling, and can be used with tool calling agents.

---

## Features
- **Pre-built Tools**: Leverages core library agent and tools.
- **Tool Calling / Function calling Agent:** Leverages tool / function input schema to appropriately route to the correct tool
- **Custom Plugin System**: Developers can bring in new tools using plugins.
- **Agentic Workflows**: Fully configurable via YAML for flexibility and productivity.
- **Ease of Use**: Simplifies developer experience and deployment.

---

## Requirements
The tool calling agent requires the `nvidia-nat[langchain]` plugin to be installed.

After you've performed a source code checkout, install this with the following command:

```bash
uv pip install -e '.[langchain]'
```

## Configuration
The tool calling agent may be utilized as a workflow or a function.

### Example `config.yml`
In your YAML file, to use the tool calling agent (`tool_calling_agent`) as a workflow:
```yaml
workflow:
  _type: tool_calling_agent
  tool_names: [wikipedia_search, current_datetime, code_generation]
  llm_name: nim_llm
  verbose: true
  handle_tool_errors: true
```

In your YAML file, to use the tool calling agent as a function:
```yaml
functions:
  calculator_multiply:
    _type: calculator_multiply
  calculator_inequality:
    _type: calculator_inequality
  calculator_divide:
    _type: nat_simple_calculator/calculator_divide
  math_agent:
    _type: tool_calling_agent
    tool_names:
      - calculator_multiply
      - calculator_inequality
      - calculator_divide
    llm_name: agent_llm
    verbose: true
    handle_tool_errors: true
    description: 'Useful for performing simple mathematical calculations.'
```

### Configurable Options

* `tool_names`: A list of tools that the agent can call. The tools must be functions configured in the YAML file

* `llm_name`: The LLM the agent should use. The LLM must be configured in the YAML file

* `verbose`: Defaults to False (useful to prevent logging of sensitive data). If set to True, the agent will log input, output, and intermediate steps.

* `handle_tool_errors`: Defaults to True. All tool errors will be caught and a `ToolMessage` with an error message will be returned, allowing the agent to retry.

* `max_iterations`: Defaults to 15. The maximum number of tool calls the agent may perform.

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
