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

# Simple Calculator - Human in the Loop

This example demonstrates **human in the loop capabilities** of the AIQ toolkit using the Simple Calculator workflow. Learn how to reuse a registered function that leverages the human in the loop capabilities of the toolkit to gate agent behavior. In this case, user approval will be requested to allow the agent to make additional tool calls to reach a final answer.

## Key Features

- **Pre-built Tools:** Leverages core AIQ toolkit library tools.
- **Custom Plugin System:** Developers can bring in new tools using plugins.
- **High-level API:** Enables defining functions that transform into asynchronous LangChain tools.
- **Agentic Workflows:** Fully configurable via YAML for flexibility and productivity.
- **Ease of Use:** Simplifies developer experience and deployment.
- **Human in the Loop:** Solicits approval from the user before allowing the agent to make additional tool calls.


## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install AIQ toolkit.

### Install this Workflow:

From the root directory of the AIQ toolkit library, run the following commands:

```bash
uv pip install -e examples/intermediate/HITL/simple_calculator_hitl
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

### Human in the Loop (HITL) Configuration
It is often helpful, or even required, to have human input during the execution of an agent workflow. For example, to ask about preferences, confirmations, or to provide additional information.
The AIQ toolkit library provides a way to add HITL interaction to any tool or function, allowing for the dynamic collection of information during the workflow execution, without the need for coding it
into the agent itself. For instance, this example asks for user approval to increase the maximum iterations of the ReAct agent to allow additional tool calling. This is enabled by leveraging a reusable plugin developed in the `examples/intermediate/HITL/por_to_jiratickets` example. We can view the implementation in the
`aiq_por_to_jiratickets.hitl_approval_tool.py` file. The implementation is shown below:

```python
@register_function(config_type=HITLApprovalFnConfig)
async def hitl_approval_function(config: HITLApprovalFnConfig, builder: Builder):

    import re

    prompt = f"{config.prompt} Please confirm if you would like to proceed. Respond with 'yes' or 'no'."

    async def _arun(unused: str = "") -> bool:

        aiq_context = AIQContext.get()
        user_input_manager = aiq_context.user_interaction_manager

        human_prompt_text = HumanPromptText(text=prompt, required=True, placeholder="<your response here>")
        response: InteractionResponse = await user_input_manager.prompt_user_input(human_prompt_text)
        response_str = response.content.text.lower()  # type: ignore
        selected_option = re.search(r'\b(yes)\b', response_str)

        if selected_option:
            return True
        return False
        # Rest of the function
```

As we see above, requesting user input using AIQ toolkit is straightforward. We can use the `user_input_manager` to prompt the user for input. The user's response is then processed to determine the next steps in the workflow.
This can occur in any tool or function in the workflow, allowing for dynamic interaction with the user as needed.

## Example Usage

### Run the Workflow

Run the following command from the root of the AIQ toolkit repo to execute this workflow with the specified input:

```bash
aiq run --config_file examples/intermediate/HITL/simple_calculator_hitl/configs/config-hitl.yml  --input "Is 2 * 4 greater than 5?"
```

**Expected Output When Giving Permission**

```console
$ aiq run --config_file examples/intermediate/HITL/simple_calculator_hitl/configs/config-hitl.yml --input "Is 2 * 4 greater than 5?"
2025-07-03 17:04:50,605 - aiq.runtime.loader - WARNING - Loading module 'aiq_profiler_agent.register' from entry point 'aiq_profiler_agent' took a long time (336.575270 ms). Ensure all imports are inside your registered functions.
2025-07-03 17:04:50,714 - aiq.runtime.loader - WARNING - Loading module 'aiq_multi_frameworks.register' from entry point 'aiq_multi_frameworks' took a long time (104.676008 ms). Ensure all imports are inside your registered functions.
2025-07-03 17:04:51,306 - aiq.runtime.loader - WARNING - Loading module 'aiq_automated_description_generation.register' from entry point 'aiq_automated_description_generation' took a long time (573.463202 ms). Ensure all imports are inside your registered functions.
2025-07-03 17:04:51,585 - aiq.runtime.loader - WARNING - Loading module 'aiq_por_to_jiratickets.register' from entry point 'aiq_por_to_jiratickets' took a long time (259.028673 ms). Ensure all imports are inside your registered functions.
2025-07-03 17:04:51,780 - aiq.runtime.loader - WARNING - Loading module 'aiq_alert_triage_agent.register' from entry point 'aiq_alert_triage_agent' took a long time (107.508183 ms). Ensure all imports are inside your registered functions.
2025-07-03 17:04:51,996 - aiq.cli.commands.start - INFO - Starting AIQ Toolkit from config file: 'examples/intermediate/HITL/simple_calculator_hitl/configs/config-hitl.yml'
2025-07-03 17:04:52,000 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.

Configuration Summary:
--------------------
Workflow Type: retry_react_agent
Number of Functions: 7
Number of LLMs: 2
Number of Embedders: 0
Number of Memory: 0
Number of Retrievers: 0

2025-07-03 17:04:53,694 - aiq.agent.react_agent.agent - INFO - 
------------------------------
[AGENT]
Agent input: Is 2 * 4 greater than 5?
Agent's thoughts: 
Thought: To answer this question, I need to first calculate the product of 2 and 4, and then compare it to 5.
Action: calculator_multiply
Action Input: {'text': '2 * 4'}

------------------------------
2025-07-03 17:04:53,706 - aiq.agent.react_agent.agent - INFO - 
------------------------------
[AGENT]
Calling tools: calculator_multiply
Tool's input: {"text": "2 * 4"}
Tool's response: 
The product of 2 * 4 is 8
------------------------------
2025-07-03 17:04:54,683 - aiq.agent.react_agent.agent - INFO - 
------------------------------
[AGENT]
Agent input: Is 2 * 4 greater than 5?
Agent's thoughts: 
Thought: Now that I have the product of 2 and 4, I can compare it to 5 to determine if it's greater.
Action: calculator_inequality
Action Input: {'text': '8 > 5'}
------------------------------
2025-07-03 17:04:54,691 - aiq.agent.react_agent.agent - INFO - 
------------------------------
[AGENT]
Calling tools: calculator_inequality
Tool's input: {"text": "8 > 5"}
Tool's response: 
First number 8 is greater than the second number 5
------------------------------
2025-07-03 17:04:54,693 - aiq.agent.react_agent.register - ERROR - [AGENT] ReAct Agent failed with exception: Recursion limit of 4 reached without hitting a stop condition. You can increase the limit by setting the `recursion_limit` config key.
For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/GRAPH_RECURSION_LIMIT
Traceback (most recent call last):
  File "/home/mpenn/containers/nemo-agent-toolkit-github/aiqtoolkit/src/aiq/agent/react_agent/register.py", line 104, in _response_fn
    state = await graph.ainvoke(state, config={'recursion_limit': (config.max_iterations + 1) * 2})
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mpenn/containers/nemo-agent-toolkit-github/aiqtoolkit/.venv/lib/python3.12/site-packages/langgraph/pregel/__init__.py", line 2389, in ainvoke
    async for chunk in self.astream(
  File "/home/mpenn/containers/nemo-agent-toolkit-github/aiqtoolkit/.venv/lib/python3.12/site-packages/langgraph/pregel/__init__.py", line 2296, in astream
    raise GraphRecursionError(msg)
langgraph.errors.GraphRecursionError: Recursion limit of 4 reached without hitting a stop condition. You can increase the limit by setting the `recursion_limit` config key.
For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/GRAPH_RECURSION_LIMIT
2025-07-03 17:04:54,696 - aiq_simple_calculator_hitl.register - INFO - Recursion error detected, prompting user to increase recursion limit
You have reached the maximum number of iterations.
Please confirm if you would like to proceed with more iterations.
 Please confirm if you would like to proceed. Respond with 'yes' or 'no'.: yes
2025-07-03 17:04:56,267 - aiq_simple_calculator_hitl.register - INFO - Attempt 2: Increasing max_iterations to 2
2025-07-03 17:04:57,239 - aiq.agent.react_agent.agent - INFO - 
------------------------------
[AGENT]
Agent input: Is 2 * 4 greater than 5?
Agent's thoughts: 
Thought: To answer this question, I need to first calculate the product of 2 and 4, and then compare it to 5.
Action: calculator_multiply
Action Input: {'text': '2 * 4'}

------------------------------
2025-07-03 17:04:57,247 - aiq.agent.react_agent.agent - INFO - 
------------------------------
[AGENT]
Calling tools: calculator_multiply
Tool's input: {"text": "2 * 4"}
Tool's response: 
The product of 2 * 4 is 8
------------------------------
2025-07-03 17:04:58,358 - aiq.agent.react_agent.agent - INFO - 
------------------------------
[AGENT]
Agent input: Is 2 * 4 greater than 5?
Agent's thoughts: 
Thought: Now that I have the product of 2 and 4, I can compare it to 5 to determine if it's greater.
Action: calculator_inequality
Action Input: {'text': '8 > 5'}
------------------------------
2025-07-03 17:04:58,368 - aiq.agent.react_agent.agent - INFO - 
------------------------------
[AGENT]
Calling tools: calculator_inequality
Tool's input: {"text": "8 > 5"}
Tool's response: 
First number 8 is greater than the second number 5
------------------------------
2025-07-03 17:04:59,018 - aiq.agent.react_agent.agent - INFO - 
------------------------------
[AGENT]
Agent input: Is 2 * 4 greater than 5?
Agent's thoughts: 
Thought: I now know the final answer
Final Answer: Yes, 2 * 4 is greater than 5.
------------------------------
2025-07-03 17:04:59,021 - aiq.utils.type_converter - WARNING - Indirect type conversion used to convert <class 'str'> to <class 'str'>, which may lead to unintended conversions. Consider adding a direct converter from <class 'str'> to <class 'str'> to ensure correctness.
2025-07-03 17:04:59,021 - aiq.front_ends.console.console_front_end_plugin - INFO - 
--------------------------------------------------
Workflow Result:
['Yes, 2 * 4 is greater than 5.']
--------------------------------------------------
```

**Expected Output When Not Giving Permission**

```console
$ aiq run --config_file examples/intermediate/HITL/simple_calculator_hitl/configs/config-hitl.yml --input "Is 2 * 4 greater than 5?"
2025-07-03 17:07:00,827 - aiq.runtime.loader - WARNING - Loading module 'aiq_automated_description_generation.register' from entry point 'aiq_automated_description_generation' took a long time (581.382036 ms). Ensure all imports are inside your registered functions.
2025-07-03 17:07:01,093 - aiq.runtime.loader - WARNING - Loading module 'aiq_por_to_jiratickets.register' from entry point 'aiq_por_to_jiratickets' took a long time (245.827675 ms). Ensure all imports are inside your registered functions.
2025-07-03 17:07:01,482 - aiq.cli.commands.start - INFO - Starting AIQ Toolkit from config file: 'examples/intermediate/HITL/simple_calculator_hitl/configs/config-hitl.yml'
2025-07-03 17:07:01,486 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.

Configuration Summary:
--------------------
Workflow Type: retry_react_agent
Number of Functions: 7
Number of LLMs: 2
Number of Embedders: 0
Number of Memory: 0
Number of Retrievers: 0

2025-07-03 17:07:03,057 - aiq.agent.react_agent.agent - INFO - 
------------------------------
[AGENT]
Agent input: Is 2 * 4 greater than 5?
Agent's thoughts: 
Thought: To answer this question, I need to first calculate the product of 2 and 4, and then compare it to 5.
Action: calculator_multiply
Action Input: {'text': '2 * 4'}

------------------------------
2025-07-03 17:07:03,076 - aiq.agent.react_agent.agent - INFO - 
------------------------------
[AGENT]
Calling tools: calculator_multiply
Tool's input: {"text": "2 * 4"}
Tool's response: 
The product of 2 * 4 is 8
------------------------------
2025-07-03 17:07:04,091 - aiq.agent.react_agent.agent - INFO - 
------------------------------
[AGENT]
Agent input: Is 2 * 4 greater than 5?
Agent's thoughts: 
Thought: Now that I have the product of 2 and 4, I can compare it to 5 to determine if it's greater.
Action: calculator_inequality
Action Input: {'text': '8 > 5'}
------------------------------
2025-07-03 17:07:04,099 - aiq.agent.react_agent.agent - INFO - 
------------------------------
[AGENT]
Calling tools: calculator_inequality
Tool's input: {"text": "8 > 5"}
Tool's response: 
First number 8 is greater than the second number 5
------------------------------
2025-07-03 17:07:04,101 - aiq.agent.react_agent.register - ERROR - [AGENT] ReAct Agent failed with exception: Recursion limit of 4 reached without hitting a stop condition. You can increase the limit by setting the `recursion_limit` config key.
For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/GRAPH_RECURSION_LIMIT
Traceback (most recent call last):
  File "/home/mpenn/containers/nemo-agent-toolkit-github/aiqtoolkit/src/aiq/agent/react_agent/register.py", line 104, in _response_fn
    state = await graph.ainvoke(state, config={'recursion_limit': (config.max_iterations + 1) * 2})
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mpenn/containers/nemo-agent-toolkit-github/aiqtoolkit/.venv/lib/python3.12/site-packages/langgraph/pregel/__init__.py", line 2389, in ainvoke
    async for chunk in self.astream(
  File "/home/mpenn/containers/nemo-agent-toolkit-github/aiqtoolkit/.venv/lib/python3.12/site-packages/langgraph/pregel/__init__.py", line 2296, in astream
    raise GraphRecursionError(msg)
langgraph.errors.GraphRecursionError: Recursion limit of 4 reached without hitting a stop condition. You can increase the limit by setting the `recursion_limit` config key.
For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/GRAPH_RECURSION_LIMIT
2025-07-03 17:07:04,105 - aiq_simple_calculator_hitl.register - INFO - Recursion error detected, prompting user to increase recursion limit
You have reached the maximum number of iterations.
Please confirm if you would like to proceed with more iterations.
 Please confirm if you would like to proceed. Respond with 'yes' or 'no'.: no
2025-07-03 17:07:08,193 - aiq.front_ends.console.console_front_end_plugin - INFO - 
--------------------------------------------------
Workflow Result:
['I seem to be having a problem.']
--------------------------------------------------
```