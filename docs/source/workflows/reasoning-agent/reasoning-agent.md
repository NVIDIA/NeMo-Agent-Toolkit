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

# Configure a Reasoning Agent
Configure the NVIDIA NeMo Agent toolkit reasoning agent as a workflow or a function. We recommend using the reasoning wrapper with any NVIDIA NeMo Agent toolkit function that could improve performance from task-specific plan generation.
---

## Requirements
The reasoning agent requires the `nvidia-nat[langchain]` plugin, which can be installed with one of the following commands.

- If you have performed a source code checkout:

```bash
uv pip install -e '.[langchain]'
```

- If you have installed the NeMo Agent toolkit from a package:

```bash
uv pip install "nvidia-nat[langchain]"
```

## Configuration

The reasoning agent can be used as a workflow or a function. Follow the example below to configure your `config.yml` YAML file.

```yaml
workflow:
  _type: reasoning_agent
  llm_name: nemotron_model
  # The augmented_fn is the nat Function that the execution plan is passed to. Usually an agent entry point.
  augmented_fn: react_agent
  verbose: true
```

### Configurable Options
The following are more ways you can configure your config file when using the reasoning agent:
* `workflow_alias`: Defaults to `None`. The alias of the workflow. Useful when the Reasoning agent is configured as a workflow and need to expose a customized name as a tool.

* `llm_name`: The LLM the agent should use. The LLM must be configured in the YAML file. The LLM must support thinking tags.

* `verbose`: Defaults to False (useful to prevent logging of sensitive data). If set to True, the agent will log input, output, and intermediate steps.

* `augmented_fn`: The function to reason on. The function should be an agent and must be defined in the config YAML.

* `reasoning_prompt_template`: The prompt used in the first step of the reasoning agent. Defaults to:
  ```python
  """
  You are an expert reasoning model task with creating a detailed execution plan for a system that has the following description

  **Description:**
  {augmented_function_desc}
  
  Given the following input and a list of available tools, please provide a detailed step-by-step plan that an instruction following system can use to address the input. Ensure the plan includes:
  1. Identifying the key components of the input.
  2. Determining the most suitable tools for each task.
  3. Outlining the sequence of actions to be taken.
  
  **Input:**
  {input_text}
  
  **Tools and description of the tool:**
  {tools}
  
  An example plan could look like this:
  1. Call tool A with input X
  2. Call tool B with input Y
  3. Interpret the output of tool A and B
  4. Return the final result
  
  **PLAN:**
  {plan}
  """
  ```


* `instruction_prompt_template`: The prompt used in the final step of the reasoning agent.  Defaults to:
  ```python
  """
  Answer the following question based on message history: {input_text}
  
  Here is a plan for execution that you could use to guide you if you wanted to:
  {reasoning_output}
  
  NOTE: Remember to follow your guidance on how to format output, etc.
  
  You must respond with the answer to the original question directly to the user.
  """
  ```

---

## The Reasoning Agent Workflow
When you enter a prompt with the reasoning agent, it runs through the following workflow:
1. **User Query** – The agent receives an input or problem to solve.
2. **Reasoning on top of Function** – The agent reasons the best plan of action to take, based on the input and the augmented underlying function.
3. **Instruction / Plan Execution** – The agent invokes the underlying function, passing its plan of action along to it.

For an example of using reasoning agent with the ReAct agent, refer to [Run the Workflow](https://github.com/NVIDIA/NeMo-Agent-Toolkit/tree/develop/examples/agents/react#run-the-workflow).

### Comparing ReAct Agent With and Without the Reasoning Agent

#### ReAct Agent Without Reasoning Agent
![Running Workflows](../../_static/agent_without_reasoning_wrapper.png)

#### ReAct Agent With Reasoning Agent
![Running Workflows](../../_static/agent_with_reasoning_wrapper.png)

---

## Limitations
The following are the limitations of reasoning agents:
* Requires a thinking/reasoning LLM, such as DeepSeek R1. There should be thought tags within the LLM output:
  >&lt;think&gt;&lt;/think&gt;

* Performs reasoning up front and does not revisit the plan to revise strategy during execution like a ReAct agent does. Revising the strategy is beneficial if a tool returns a non-useful response (let's say our retriever tool did not have any relevant search results to the user's original question).
