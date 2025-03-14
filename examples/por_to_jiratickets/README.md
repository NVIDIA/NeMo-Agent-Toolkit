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

A minimal example demonstrating an end-to-end Jira ticket creating agentic workflow. This workflow leverages the AgentIQ plugin system to integrate pre-built and custom tools into the workflow. Key elements are summarized below:

## Key Features

- **Pre-built Tools:** Leverages core AgentIQ library tools.
- **Custom Plugin System:** Developers can bring in new tools using plugins.
- **High-level API:** Enables defining functions that transform into asynchronous LangChain tools.
- **Agentic Workflows:** Fully configurable via YAML for flexibility and productivity.
- **Ease of Use:** Simplifies developer experience and deployment.
- **Jira Agent Tool Call:** Following tools are available for the agent to extract POR, create and get Jira tickets.
   - `create_jira_ticket`()`: This function creates Jira ticket using the REST API. It requires specifying the project key, Jira token, Jira username, domain, and also ticket type (e.g., Bug, Task, Story), description and priority. Upon successful creation, it returns the ticket ID and URL.
   -  `extract_from_por_tool`: Extract epics, tasks, features and bugs from the given PRO/PRD file using the LLM chain and store the result. Assigns story points for each type based on complexity/effort and also fills in description for each.
   -  `get_jira_tickets_tool`: This function retrieves existing Jira tickets based on a JQL (Jira Query Language) filter. It fetches relevant information like ticket summary, status, and assignee. The returned data can be used for tracking or reporting.


## Installation and Setup

### Setup Virtual Environment and Install AgentIQ

From the root directory of the AgentIQ library, run the following commands. Skip this if you've already installed AgentIQ.

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install uv
uv sync --all-extras
```

### Install this Workflow:

From the root directory of the AgentIQ library, run the following commands:

```bash
uv pip install -e examples/por_to_jiratickets
```

### Set Up API Keys

You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
export JIRA_USERID=<YOUR_JIRA_USERNAME>
export JIRA_TOKEN=<YOUR_JIRA_TOKEN>
```

Steps to create a Jira token: Go to `User Profile` -> `API token authentication`-> `Creat a new API token`

### Update `Config.yml` with Jira domain and PROJECT KEY
```
    jira_domain: "https://<YOUR_COMPANY_DOMAIN>.com"
    jira_project_key: "<YOUR_JIRA_PROJECTKEY>"
```

### Human in the Loop (HITL) Configuration
It is often helpful, or even required, to have human input during the execution of an agent workflow. For example, to ask about preferences, confirmations, or to provide additional information.
The AgentIQ library provides a way to add HITL interaction to any tool or function, allowing for the dynamic collection of information during the workflow execution, without the need for coding it
into the agent itself. For instance, this example asks for user permission to create Jira issues and tickets before creating them. We can view the implementation in the
`aiq_por_to_jiratickets.jira_tickets_tool.py` file. The implementation is below:

```python
@register_function(config_type=CreateJiraToolConfig)
async def create_jira_tickets_tool(config: CreateJiraToolConfig, builder: Builder):

    async def _arun(input_text: str) -> str:

        # Get user confirmation first
        try:
            aiq_context = AIQContext.get()
            user_input_manager = aiq_context.user_interaction_manager

            prompt = ("I would like to create Jira tickets for the extracted data. "
                      "Please confirm if you would like to proceed. Respond with 'yes' or 'no'.")

            human_prompt_text = HumanPromptText(text=prompt, required=True, placeholder="<your response here>")

            response = await user_input_manager.prompt_user_input(human_prompt_text)

            response_text = response.content.text.lower()

            # Regex to see if the response has yes in it
            # Set value to True if the response is yes
            import re
            selected_option = re.search(r'\b(yes)\b', response_text)
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
As we see above, requesting user input using AgentIQ is straightforward. We can use the `user_input_manager` to prompt the user for input. The user's response is then processed to determine the next steps in the workflow.
This can occur in any tool or function in the workflow, allowing for dynamic interaction with the user as needed.

## Example Usage

### Run the Workflow

Run the following command from the root of the AgentIQ repo to execute this workflow with the specified input:

```bash
aiq run --config_file examples/por_to_jiratickets/configs/config.yml  --input "Can you extract por file por_requirements.txt, assign story points and create jira tickets for epics first and then followed by tasks?"
```

**Expected Output When Giving Permission**

```console
2025-03-12 15:28:34,484 - aiq.agent.react_agent.agent - INFO - Filling the prompt variables "tools" and "tool_names", using the tools provided in the config.
2025-03-12 15:28:34,484 - aiq.agent.react_agent.agent - INFO - Adding the tools' input schema to the tools' description
2025-03-12 15:28:34,485 - aiq.agent.react_agent.agent - INFO - Initialized ReAct Agent Graph
2025-03-12 15:28:34,553 - aiq.agent.react_agent.agent - INFO - ReAct Graph built and compiled successfully
2025-03-12 15:28:34,555 - aiq.front_ends.console.console_front_end_plugin - INFO - Processing input: ('Can you extract por file por_requirements.txt, assign story points and create jira tickets for epics first and then followed by tasks?',)
2025-03-12 15:28:34,558 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 4
Number of LLMs: 2
Number of Embedders: 0
Number of Memory: 0
Number of Retrievers: 0

2025-03-12 15:28:35,769 - aiq.agent.react_agent.agent - INFO - The user's question was: Can you extract por file por_requirements.txt, assign story points and create jira tickets for epics first and then followed by tasks?
2025-03-12 15:28:35,770 - aiq.agent.react_agent.agent - INFO - The agent's thoughts are:
Thought: To accomplish this task, I need to first extract the epics and tasks from the POR file, assign story points, and then create Jira tickets for epics and tasks separately.

Action: extract_por_tool
Action Input: {'input_text': 'por_requirements.txt'}

2025-03-12 15:28:35,775 - aiq.agent.react_agent.agent - INFO - Calling tool extract_por_tool with input: {'input_text': 'por_requirements.txt'}
2025-03-12 15:28:35,775 - aiq.agent.react_agent.agent - INFO - Successfully parsed structured tool input from Action Input
2025-03-12 15:29:00,696 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1
2025-03-12 15:29:01,880 - aiq.agent.react_agent.agent - INFO -

The agent's thoughts are:
Thought: Now that the extraction is complete, I can ask to show the epics and tasks to verify the extraction results. After verification, I can proceed with creating Jira tickets for epics and tasks.

Action: show_jira_tickets
Action Input: {'input_text': 'epics'}
2025-03-12 15:29:01,882 - aiq.agent.react_agent.agent - INFO - Calling tool show_jira_tickets with input: {'input_text': 'epics'}
2025-03-12 15:29:01,882 - aiq.agent.react_agent.agent - INFO - Successfully parsed structured tool input from Action Input
2025-03-12 15:29:01,888 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1
I would like to create Jira tickets for the extracted data. Please confirm if you would like to proceed. Respond with 'yes' or 'no'.: 2025-03-12 15:29:02,868 - aiq.agent.react_agent.agent - INFO -

The agent's thoughts are:
Thought: The epics have been successfully extracted and displayed. Now, I need to create Jira tickets for these epics.

Action: create_jira_tickets_tool
Action Input: {'input_text': 'epics'}
2025-03-12 15:29:02,869 - aiq.agent.react_agent.agent - INFO - Calling tool create_jira_tickets_tool with input: {'input_text': 'epics'}
2025-03-12 15:29:02,869 - aiq.agent.react_agent.agent - INFO - Successfully parsed structured tool input from Action Input
yes
2025-03-12 15:31:06,092 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1
2025-03-12 15:31:07,157 - aiq.agent.react_agent.agent - INFO -

The agent's thoughts are:
Thought: The Jira tickets for epics have been successfully created. Next, I need to show the tasks to verify their extraction results.

Action: show_jira_tickets
Action Input: {'input_text': 'tasks'}
2025-03-12 15:31:07,160 - aiq.agent.react_agent.agent - INFO - Calling tool show_jira_tickets with input: {'input_text': 'tasks'}
2025-03-12 15:31:07,160 - aiq.agent.react_agent.agent - INFO - Successfully parsed structured tool input from Action Input
2025-03-12 15:31:07,164 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1
2025-03-12 15:31:08,135 - aiq.agent.react_agent.agent - INFO -

The agent's thoughts are:
Thought: The tasks have been successfully extracted and displayed. Now, I need to create Jira tickets for these tasks.

Action: create_jira_tickets_tool
Action Input: {'input_text': 'tasks'}
2025-03-12 15:31:08,138 - aiq.agent.react_agent.agent - INFO - Calling tool create_jira_tickets_tool with input: {'input_text': 'tasks'}
2025-03-12 15:31:08,138 - aiq.agent.react_agent.agent - INFO - Successfully parsed structured tool input from Action Input
I would like to create Jira tickets for the extracted data. Please confirm if you would like to proceed. Respond with 'yes' or 'no'.: yes
2025-03-12 15:31:15,897 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1
2025-03-12 15:31:21,529 - aiq.agent.react_agent.agent - INFO -

The agent's thoughts are:
Thought: I now know the final answer

Final Answer: Jira tickets for epics and tasks have been successfully created.
2025-03-12 15:31:21,532 - aiq.observability.async_otel_listener - INFO - Intermediate step stream completed. No more events will arrive.
2025-03-12 15:31:21,532 - aiq.front_ends.console.console_front_end_plugin - INFO - --------------------------------------------------
Workflow Result:
['Jira tickets for epics and tasks have been successfully created.']
--------------------------------------------------
```
**Expected Output When Not Giving Permission**

```console
2025-03-12 16:49:27,564 - aiq.front_ends.console.console_front_end_plugin - INFO - Processing input: ('Can you extract por file por_requirements.txt, assign story points and create jira tickets for epics first and then followed by tasks?',)
2025-03-12 16:49:27,567 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 4
Number of LLMs: 2
Number of Embedders: 0
Number of Memory: 0
Number of Retrievers: 0

2025-03-12 16:49:28,994 - aiq.agent.react_agent.agent - INFO - The user's question was: Can you extract por file por_requirements.txt, assign story points and create jira tickets for epics first and then followed by tasks?
2025-03-12 16:49:28,994 - aiq.agent.react_agent.agent - INFO - The agent's thoughts are:
Thought: To accomplish this task, I need to first extract the epics and tasks from the POR file, assign story points, and then create Jira tickets for epics and tasks separately.

Action: extract_por_tool
Action Input: {'input_text': 'por_requirements.txt'}

2025-03-12 16:49:28,999 - aiq.agent.react_agent.agent - INFO - Calling tool extract_por_tool with input: {'input_text': 'por_requirements.txt'}
2025-03-12 16:49:28,999 - aiq.agent.react_agent.agent - INFO - Successfully parsed structured tool input from Action Input
2025-03-12 16:49:53,727 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1
2025-03-12 16:49:54,912 - aiq.agent.react_agent.agent - INFO -

The agent's thoughts are:
Thought: Now that the extraction is complete, I can ask to show the epics and tasks that were extracted, but my main goal is to create Jira tickets for epics first and then tasks.

Action: create_jira_tickets_tool
Action Input: {'input_text': 'epics'}
2025-03-12 16:49:54,916 - aiq.agent.react_agent.agent - INFO - Calling tool create_jira_tickets_tool with input: {'input_text': 'epics'}
2025-03-12 16:49:54,916 - aiq.agent.react_agent.agent - INFO - Successfully parsed structured tool input from Action Input
I would like to create Jira tickets for the extracted data. Please confirm if you would like to proceed. Respond with 'yes' or 'no'.: no
2025-03-12 16:49:59,963 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1
2025-03-12 16:50:07,570 - aiq.agent.react_agent.agent - INFO -

The agent's thoughts are:
Thought: I now know the final answer

Final Answer: Jira tickets for epics were not created due to lack of user confirmation.
2025-03-12 16:50:07,574 - aiq.observability.async_otel_listener - INFO - Intermediate step stream completed. No more events will arrive.
2025-03-12 16:50:07,574 - aiq.front_ends.console.console_front_end_plugin - INFO - --------------------------------------------------
Workflow Result:
['Jira tickets for epics were not created due to lack of user confirmation.']
--------------------------------------------------
```
