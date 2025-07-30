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

# A Simple Jira Agent that Extracts POR and creates tickets

A minimal example demonstrating an end-to-end Jira ticket creating agentic workflow. This workflow leverages the NeMo Agent toolkit plugin system to integrate pre-built and custom tools into the workflow. Key elements are summarized below:

## Table of Contents

- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
  - [Install this Workflow](#install-this-workflow)
  - [Set Up API Keys](#set-up-api-keys)
  - [Update `config.yml` with Jira domain and PROJECT KEY](#update-configyml-with-jira-domain-and-project-key)
  - [Human in the Loop (HITL) Configuration](#human-in-the-loop-hitl-configuration)
- [Example Usage](#example-usage)
  - [Run the Workflow](#run-the-workflow)

## Key Features

- **Document-to-Jira Workflow:** Demonstrates extraction of epics, tasks, features, and bugs from PRD/POR documents using LLM processing and automatic conversion to structured Jira tickets.
- **Jira REST API Integration:** Shows comprehensive Jira integration with `create_jira_tickets_tool`, `extract_from_por_tool`, and `get_jira_tickets_tool` for complete ticket lifecycle management.
- **Human-in-the-Loop Approval:** Implements `hitl_approval_tool` that requires explicit user confirmation before creating Jira tickets, demonstrating secure workflow gates and user control.
- **Intelligent Story Point Assignment:** Automatically assigns story points based on complexity and effort estimation using LLM analysis of extracted requirements.
- **Structured Requirement Extraction:** Processes requirement documents to identify and categorize different work items with appropriate descriptions, priorities, and ticket types.

## Prerequisites

Access to a Jira system is required. You will need enough permissions to obtain a Jira token.

Steps to create a Jira token:
1. Go to `User Profile`
2. Navigate to `API token authentication`
3. Click `Create a new API token`

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent toolkit.

### Install this Workflow:

From the root directory of the NeMo Agent toolkit library, run the following commands:

```bash
uv pip install -e examples/HITL/por_to_jiratickets
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
export JIRA_USERID=<YOUR_JIRA_USERNAME>
export JIRA_TOKEN=<YOUR_JIRA_TOKEN>
```

### Update `config.yml` with Jira domain and PROJECT KEY
```
    jira_domain: "https://<YOUR_COMPANY_DOMAIN>.com"
    jira_project_key: "<YOUR_JIRA_PROJECTKEY>"
```

### Human in the Loop (HITL) Configuration
It is often helpful, or even required, to have human input during the execution of an agent workflow. For example, to ask about preferences, confirmations, or to provide additional information.
The NeMo Agent toolkit library provides a way to add HITL interaction to any tool or function, allowing for the dynamic collection of information during the workflow execution, without the need for coding it
into the agent itself. For instance, this example asks for user permission to create Jira issues and tickets before creating them. We can view the implementation in the
`aiq_por_to_jiratickets.jira_tickets_tool.py` file. The implementation is below:

```python
### The reusable HITL function
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

    yield FunctionInfo.from_fn(_arun,
                               description=("This function will be used to get the user's response to the prompt"))


### The JIRA function that uses the HITL function
@register_function(config_type=CreateJiraToolConfig)
async def create_jira_tickets_tool(config: CreateJiraToolConfig, builder: Builder):

    hitl_approval_fn = builder.get_function(config.hitl_approval_fn)

    async def _arun(input_text: str) -> str:

        # Get user confirmation first
        try:
            selected_option = await hitl_approval_fn.acall_invoke()

            if not selected_option:
                return "Did not receive user confirmation to upload to Jira. You can exit with a final answer."

        except Exception as e:
            logger.error("An error occurred when getting interaction content: %s", e)
            logger.info("Defaulting to not uploading to Jira")
            return ("Did not upload to Jira because human confirmation was not received. "
                    "You can exit with a final answer")

        logger.debug("Creating %s in Jira", input_text)
        # Rest of the function
```
As we see above, requesting user input using NeMo Agent toolkit is straightforward. We can use the `user_input_manager` to prompt the user for input. The user's response is then processed to determine the next steps in the workflow.
This can occur in any tool or function in the workflow, allowing for dynamic interaction with the user as needed.

## Example Usage

### Run the Workflow

Run the following command from the root of the NeMo Agent toolkit repo to execute this workflow with the specified input:

```bash
aiq run --config_file examples/HITL/por_to_jiratickets/configs/config.yml  --input "Can you extract por file por_requirements.txt, assign story points and create jira tickets for epics first and then followed by tasks?"
```

**Expected Workflow Result When Giving Permission**
```console
<snipped for brevity>

------------------------------
[AGENT]
Calling tools: extract_por_tool
Tool's input: {"input_text": "por_requirements.txt"}
Tool's response:
Extraction complete. You can now ask me to show epics or tasks.
------------------------------

<snipped for brevity>

------------------------------
[AGENT]
Agent input: Can you extract por file por_requirements.txt, assign story points and create jira tickets for epics first and then followed by tasks?
Agent's thoughts:
Thought: I now know the final answer

<snipped for brevity>

Workflow Result:
['Jira tickets for epics and tasks have been created. Epics: AIQ-1158, AIQ-1163, AIQ-1159, AIQ-1162, AIQ-1161, AIQ-1160. Tasks: AIQ-1166, AIQ-1169, AIQ-1170, AIQ-1164, AIQ-1171, AIQ-1168, AIQ-1172, AIQ-1174, AIQ-1165, AIQ-1175, AIQ-1173, AIQ-1167.']
```

**Expected Workflow Result When Not Giving Permission**

```console
<snipped for brevity>

Action: create_jira_tickets_tool
Action Input: {'input_text': 'epics'}
2025-03-12 16:49:54,916 - aiq.agent.react_agent.agent - INFO - Calling tool create_jira_tickets_tool with input: {'input_text': 'epics'}
2025-03-12 16:49:54,916 - aiq.agent.react_agent.agent - INFO - Successfully parsed structured tool input from Action Input
I would like to create Jira tickets for the extracted data. Please confirm if you would like to proceed. Respond with 'yes' or 'no'.: no

<snipped for brevity>

Workflow Result:
['Jira tickets for epics were not created due to lack of user confirmation.']

```
