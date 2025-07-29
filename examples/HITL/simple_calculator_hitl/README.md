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

This example demonstrates **human in the loop capabilities** of the NeMo Agent toolkit using the Simple Calculator workflow. Learn how to reuse a registered function that leverages the human in the loop capabilities of the toolkit to gate agent behavior. In this case, user approval will be requested to allow the agent to make additional tool calls to reach a final answer.

## Table of Contents

- [Key Features](#key-features)
- [Installation and Setup](#installation-and-setup)
  - [Install this Workflow](#install-this-workflow)
  - [Set Up API Keys](#set-up-api-keys)
  - [Human in the Loop (HITL) Configuration](#human-in-the-loop-hitl-configuration)
- [Example Usage](#example-usage)
  - [Run the Workflow](#run-the-workflow)

## Key Features

- **Human-in-the-Loop Integration:** Demonstrates the `hitl_approval_function` that requests user approval before allowing the agent to increase iteration limits and make additional tool calls.
- **Dynamic Recursion Limit Management:** Shows how to handle agent recursion limits by prompting users for permission to extend maximum iterations when the agent needs more steps to complete a task.
- **User Interaction Manager:** Demonstrates the NeMo Agent toolkit `user_input_manager` for prompting user input and processing responses during workflow execution.
- **Conditional Workflow Continuation:** Shows how agent behavior can be gated based on user responses, allowing workflows to stop or continue based on human approval.
- **Retry ReAct Agent:** Uses a custom `retry_react_agent` workflow that can recover from recursion limits with user permission and increased iteration capacity.


## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent toolkit.

### Install this Workflow

Install this example:

```bash
uv pip install -e examples/HITL/simple_calculator_hitl
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

### Human in the Loop (HITL) Configuration
It is often helpful, or even required, to have human input during the execution of an agent workflow. For example, to ask about preferences, confirmations, or to provide additional information.
The NeMo Agent toolkit library provides a way to add HITL interaction to any tool or function, allowing for the dynamic collection of information during the workflow execution, without the need for coding it
into the agent itself. For instance, this example asks for user approval to increase the maximum iterations of the ReAct agent to allow additional tool calling. This is enabled by leveraging a reusable plugin developed in the `examples/custom_functions/por_to_jiratickets` example. We can view the implementation in the
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

As we see above, requesting user input using NeMo Agent toolkit is straightforward. We can use the `user_input_manager` to prompt the user for input. The user's response is then processed to determine the next steps in the workflow.
This can occur in any tool or function in the workflow, allowing for dynamic interaction with the user as needed.

## Example Usage

### Run the Workflow

Run the following command from the root of the NeMo Agent toolkit repo to execute this workflow with the specified input:

```bash
aiq run --config_file examples/HITL/simple_calculator_hitl/configs/config-hitl.yml  --input "Is 2 * 4 greater than 5?"
```

**Expected Workflow Result When Giving Permission**

```console
<snipped for brevity>

langgraph.errors.GraphRecursionError: Recursion limit of 4 reached without hitting a stop condition. You can increase the limit by setting the `recursion_limit` config key.
For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/GRAPH_RECURSION_LIMIT
2025-07-03 17:04:54,696 - aiq_simple_calculator_hitl.register - INFO - Recursion error detected, prompting user to increase recursion limit
You have reached the maximum number of iterations.
Please confirm if you would like to proceed with more iterations.
 Please confirm if you would like to proceed. Respond with 'yes' or 'no'.: yes
2025-07-03 17:04:56,267 - aiq_simple_calculator_hitl.register - INFO - Attempt 2: Increasing max_iterations to 2

<snipped for brevity>


Workflow Result:
['Yes, 2 * 4 is greater than 5.']
```

**Expected Workflow Result When Not Giving Permission**

```console
<snipped for brevity>

langgraph.errors.GraphRecursionError: Recursion limit of 4 reached without hitting a stop condition. You can increase the limit by setting the `recursion_limit` config key.
For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/GRAPH_RECURSION_LIMIT
2025-07-03 17:07:04,105 - aiq_simple_calculator_hitl.register - INFO - Recursion error detected, prompting user to increase recursion limit
You have reached the maximum number of iterations.
Please confirm if you would like to proceed with more iterations.
 Please confirm if you would like to proceed. Respond with 'yes' or 'no'.: no

<snipped for brevity>
Workflow Result:
['I seem to be having a problem.']
```
