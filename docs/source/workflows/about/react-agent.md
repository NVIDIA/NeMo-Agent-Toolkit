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

# ReAct Agent
Agents are a major use-case for language models. Agents are systems that use LLMs to reason and determine what actions
to take and what inputs to use for those actions. After executing those actions, the agent uses the LLM to determine
if more actions are required. This agent is a ReAct (Reasoning and Acting) Agent, based on the [ReAct paper](https://react-lm.github.io/).

The ReAct agent's prompt is directly inspired by the prompt examples in the appendix of the
paper.

---

## Features
- **Pre-built Tools**: Leverages core library agent and tools.
- **ReAct Reasoning**: Performs reasoning between tool calls; utilizes tool names and descriptions to appropriately route to the correct tool.
- **Custom Plugin System**: Developers can bring in new tools using plugins.
- **Customizable Prompt**: Modify, tweak, or change the prompt for your specific needs.
- **Agentic Workflows**: Fully configurable via YAML for flexibility and productivity.
- **Ease of Use**: Simplifies developer experience and deployment.

---

## Requirements
The ReAct agent requires the `aiqtoolkit[langchain]` plugin to be installed.

If you have performed a source code checkout, you can install this with the following command:

```bash
uv pip install -e '.[langchain]'
```

## Configuration

The ReAct agent may be utilized as a Workflow or a Function.

### Example `config.yml`
In your YAML file, to use the ReAct agent as a workflow:
```yaml
workflow:
  _type: react_agent
  tool_names: [wikipedia_search, current_datetime, code_generation, math_agent]
  llm_name: nim_llm
  verbose: true
  handle_parsing_errors: true
  max_retries: 2
```
In your YAML file, to use the ReAct agent as a function:
```yaml
functions:
  calculator_multiply:
    _type: calculator_multiply
  calculator_inequality:
    _type: calculator_inequality
  calculator_divide:
    _type: aiq_simple_calculator/calculator_divide
  math_agent:
    _type: react_agent
    tool_names:
      - calculator_multiply
      - calculator_inequality
      - calculator_divide
    description: 'Useful for performing simple mathematical calculations.'
```

### Configurable Options:
* `tool_names`: A list of tools that the agent can call. The tools must be functions configured in the YAML file.

* `llm_name`: The LLM the agent should use. The LLM must be configured in the YAML file.

* `verbose`: Defaults to `False` (useful to prevent logging of sensitive data).  If set to `True`, the Agent will log input, output, and intermediate steps.

* `retry_parsing_errors`: Defaults to `True`.  Sometimes, the Agent may hallucinate and might not output exactly in the ReAct output format (due to inherit LLM variability.  These hallucinations can be reduced by tweaking the prompt to be more specific for your use-case.); if set to `True`, the Agent will identify the issue with the LLM output (how exactly are we missing the ReAct output format?) and will retry the LLM call, including the output format error information.

* `max_retries`: Defaults to `1`.  Maximum amount of times the Agent may retry parsing errors.  Prevents the Agent from getting into infinite hallucination loops.

* `max_iterations`: Defaults to `15`.  The ReAct agent may reason between tool calls, and might use multiple tools to answer the question; the maximum amount of tool calls the Agent may take before answering the original question.

* `description`:  Defaults to `"ReAct Agent Workflow"`.  When the ReAct agent is configured as a function, this config option allows us to control the tool description (for example, when used as a tool within another agent).

* `system_prompt`:  Optional.  Allows us to override the system prompt for the ReAct agent.
If modifying the prompt, see the limitations section below. The prompt must have variables for tools, and must instruct the LLM to output in the ReAct output format.

* `max_history`:  Defaults to `15`. Maximum number of messages to keep in the conversation history.

* `use_openai_api`: Defaults to `False`.  If set to `True`, the ReAct agent will output in OpenAI API spec. If set to `False`, strings will be used.

* `include_tool_input_schema_in_tool_description`: Defaults to `True`.  If set to `True`, the ReAct agent will inspect its tools' input schemas, and append the following to each tool description:
  >. Arguments must be provided as a valid JSON object following this format: {tool_schema}

---

## How the ReAct Agent works

A **ReAct (Reasoning and Acting) agent** is an AI system that decides what actions to take by reasoning step-by-step. Instead of making a decision in one go, it follows an **iterative thought process**, inspired by the [ReAct paper](https://react-lm.github.io/).
The Agent uses an LLM to make the decisions, and to summarize the tool responses in natural human language.  To decide which tool(s) to use to answer the question, the ReAct agent uses the names and descriptions of its tools.

### **Step-by-Step Breakdown of a ReAct Agent**

1. **Observation** – The agent receives an input or problem to solve.
2. **Reasoning (Thought)** – The agent thinks about what to do next.
3. **Action** – The agent calls a tool (like a search API, calculator, or database query).
4. **Observation (Feedback)** – The agent examines the tool’s response.
5. **Repeat** – If more steps are needed, it repeats the process.

### Example Walkthrough

Imagine a ReAct agent needs to answer:

> "What’s the current weather in New York?"

#### Iteration 1
- **Observation:** The agent sees the question.
- **Thought:** "I don’t have the weather data, but I can use a weather API."
- **Action:** Calls the weather API.

#### **Iteration 2**
- **Observation:** The API returns `72°F, clear skies`.
- **Thought:** "Now I can answer the user’s question."
- **Action:** Returns: *"The weather in New York is 72°F with clear skies."*

### ReAct Prompting and Output Format

ReAct agents require the LLM to output in ReAct output format.  This is an example of the ReAct output format for calling a tool:
```
Thought: To answer this question, I need to find information about Djikstra.

Action: wikipedia_search
Action Input: Djikstra

Observation: (I will wait for the human to call the wikipedia tool and provide the response...)

```
This is an example of the ReAct output format when the agent has the final answer:
```
Thought: I now know the final answer

Final Answer: Djikstra was a Dutch computer scientist, programmer, software engineer, mathematician, and science essayist. He is best known for his work on the shortest path problem and the development of Dijkstra's algorithm, which is used to find the shortest path between nodes in a weighted graph.

```

We may tweak, modify, or completely change the ReAct agent prompt, but the LLM output must match the ReAct output format, and the prompt must have a prompt variable named `{tools}` and `{tool_names}`

A sample ReAct agent prompt is provided in prompt.py:
```
Answer the following questions as best you can. You may ask the human to use the following tools:

{tools}

You may respond in one of two formats.
Use the following format exactly to ask the human to use a tool:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: The input to the action (if there is no required input, include "Action Input: None")
Observation: wait for the human to respond with the result from the tool, do not assume the response

... (this Thought/Action/Action Input/Observation can repeat N times. If you do not need to use a tool, or after asking the human to use any tools and waiting for the human to respond, you might know the final answer.)
Use the following format once you have the final answer:

Thought: I now know the final answer
Final Answer: the final answer to the original input question
```

---

## Limitations
ReAct (Reasoning and Acting) agents are powerful but come with several limitations that make them less efficient in certain use cases compared to tool-calling agents or reasoning agents. The limitations are as follows:

* ReAct agents Require More LLM Calls

  ReAct agents perform reasoning step-by-step, which means they first generate thoughts, then take an action, then reason again based on the result. This iterative process can lead to multiple LLM calls per task, increasing latency and API costs.

* Prompt-Sensitivity & Tuning Overhead

  Since ReAct agents rely heavily on prompting, they require careful tuning. The quality of their decisions depends on the structure of the prompt and the examples given. A poorly tuned prompt can lead to inefficient reasoning or incorrect tool usage.

* Possible Risk of Hallucination

  ReAct agents reason between steps, which sometimes results in hallucinations where the model makes incorrect assumptions or misinterprets tool responses. Unlike structured tool-calling agents, they lack built-in constraints to prevent invalid reasoning paths. Sometimes, the LLM does not output in the ReAct output format.

* Increased Complexity in Long Chains

  For workflows that involve multiple steps and dependencies, ReAct agents may struggle with consistency. If an early reasoning step is flawed, it can propagate errors throughout the execution, making debugging difficult.

* Lack of Parallelism

  ReAct agents execute sequentially:

  > Think → Act → Observe → Repeat.

  This prevents them from efficiently handling tasks that could be executed in parallel, such as making multiple API calls simultaneously.

In summary, ReAct agents frequently require a bit of tuning to optimize performance and ensure the best results. Proper prompt engineering and configuration adjustments may be necessary depending on the complexity of the tasks required.
