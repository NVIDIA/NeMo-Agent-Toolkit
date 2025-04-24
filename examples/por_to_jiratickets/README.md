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

If you have not already done so, follow the instructions in the [Install Guide](../../docs/source/intro/install.md) to create the development environment and install AgentIQ.

### Install this Workflow:

From the root directory of the AgentIQ library, run the following commands:

```bash
uv pip install -e examples/por_to_jiratickets
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../docs/source/intro/get-started.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

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
$ aiq run --config_file examples/por_to_jiratickets/configs/config.yml  --input "Can you extract por file por_requirements.txt, assign story points and create jira tickets for epics first and then followed by tasks?"
2025-04-23 15:46:33,770 - aiq.runtime.loader - WARNING - Loading module 'aiq_automated_description_generation.register' from entry point 'aiq_automated_description_generation' took a long time (501.032114 ms). Ensure all imports are inside your registered functions.
2025-04-23 15:46:34,105 - aiq.cli.commands.start - INFO - Starting AgentIQ from config file: 'examples/por_to_jiratickets/configs/config.yml'
2025-04-23 15:46:34,112 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
2025-04-23 15:46:34,147 - aiq.profiler.utils - WARNING - Discovered frameworks: {<LLMFrameworkEnum.LANGCHAIN: 'langchain'>} in function extract_from_por_tool by inspecting source. It is recommended and more reliable to instead add the used LLMFrameworkEnum types in the framework_wrappers argument when calling @register_function.
/nvme/1/yuchenz/projects/AgentIQ/examples/por_to_jiratickets/src/aiq_por_to_jiratickets/extract_por_tool.py:141: LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 1.0. Use :meth:`~RunnableSequence, e.g., `prompt | llm`` instead.
  chain = LLMChain(llm=llm, prompt=prompt)

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 4
Number of LLMs: 2
Number of Embedders: 0
Number of Memory: 0
Number of Retrievers: 0

2025-04-23 15:46:35,415 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: Can you extract por file por_requirements.txt, assign story points and create jira tickets for epics first and then followed by tasks?
Agent's thoughts:
Thought: The user wants to extract epics and tasks from a POR file, assign story points, and create Jira tickets for epics and tasks. The first step is to extract the epics and tasks from the POR file.

Action: extract_por_tool
Action Input: {'input_text': 'por_requirements.txt'}

------------------------------
/nvme/1/yuchenz/projects/AgentIQ/examples/por_to_jiratickets/src/aiq_por_to_jiratickets/extract_por_tool.py:152: LangChainDeprecationWarning: The method `Chain.arun` was deprecated in langchain 0.1.0 and will be removed in 1.0. Use :meth:`~ainvoke` instead.
  response = await chain.arun(por_content=input_text)
2025-04-23 15:51:28,095 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: extract_por_tool
Tool's input: {"input_text": "por_requirements.txt"}
Tool's response:
Extraction complete. You can now ask me to show epics or tasks.
------------------------------
2025-04-23 15:51:33,159 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: Can you extract por file por_requirements.txt, assign story points and create jira tickets for epics first and then followed by tasks?
Agent's thoughts:
Thought: Now that the extraction is complete, I can ask the human to show the extracted epics and tasks. However, the user's original request was to create Jira tickets for epics first and then tasks. So, I will ask the human to create Jira tickets for epics.

Action: create_jira_tickets_tool
Action Input: {'input_text': 'epics'}
------------------------------
I would like to create Jira tickets for the extracted data. Please confirm if you would like to proceed. Respond with 'yes' or 'no'.: yes
2025-04-23 15:51:52,134 - httpx - INFO - HTTP Request: POST https://jirasw.nvidia.com/rest/api/2/issue "HTTP/1.1 201 Created"
2025-04-23 15:51:52,197 - httpx - INFO - HTTP Request: POST https://jirasw.nvidia.com/rest/api/2/issue "HTTP/1.1 201 Created"
2025-04-23 15:51:52,211 - httpx - INFO - HTTP Request: POST https://jirasw.nvidia.com/rest/api/2/issue "HTTP/1.1 201 Created"
2025-04-23 15:51:52,334 - httpx - INFO - HTTP Request: POST https://jirasw.nvidia.com/rest/api/2/issue "HTTP/1.1 201 Created"
2025-04-23 15:51:52,356 - httpx - INFO - HTTP Request: POST https://jirasw.nvidia.com/rest/api/2/issue "HTTP/1.1 201 Created"
2025-04-23 15:51:52,370 - httpx - INFO - HTTP Request: POST https://jirasw.nvidia.com/rest/api/2/issue "HTTP/1.1 201 Created"
2025-04-23 15:51:52,373 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: create_jira_tickets_tool
Tool's input: {"input_text": "epics"}
Tool's response:
### Created epics:
- **AIQ-1158**: https://jirasw.nvidia.com/browse/AIQ-1158
- **AIQ-1163**: https://jirasw.nvidia.com/browse/AIQ-1163
- **AIQ-1159**: https://jirasw.nvidia.com/browse/AIQ-1159
- **AIQ-1162**: https://jirasw.nvidia.com/browse/AIQ-1162
- **AIQ-1161**: https://jirasw.nvidia.com/browse/AIQ-1161
- **AIQ-1160**: https://jirasw.nvidia.com/browse/AIQ-1160
------------------------------
2025-04-23 15:51:53,217 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: Can you extract por file por_requirements.txt, assign story points and create jira tickets for epics first and then followed by tasks?
Agent's thoughts:
Thought: The Jira tickets for epics have been created. The next step is to create Jira tickets for tasks, as per the user's original request.

Action: create_jira_tickets_tool
Action Input: {'input_text': 'tasks'}
------------------------------
I would like to create Jira tickets for the extracted data. Please confirm if you would like to proceed. Respond with 'yes' or 'no'.: yes
2025-04-23 15:51:57,269 - httpx - INFO - HTTP Request: POST https://jirasw.nvidia.com/rest/api/2/issue "HTTP/1.1 201 Created"
2025-04-23 15:51:57,301 - httpx - INFO - HTTP Request: POST https://jirasw.nvidia.com/rest/api/2/issue "HTTP/1.1 201 Created"
2025-04-23 15:51:57,389 - httpx - INFO - HTTP Request: POST https://jirasw.nvidia.com/rest/api/2/issue "HTTP/1.1 201 Created"
2025-04-23 15:51:57,424 - httpx - INFO - HTTP Request: POST https://jirasw.nvidia.com/rest/api/2/issue "HTTP/1.1 201 Created"
2025-04-23 15:51:57,647 - httpx - INFO - HTTP Request: POST https://jirasw.nvidia.com/rest/api/2/issue "HTTP/1.1 201 Created"
2025-04-23 15:51:57,682 - httpx - INFO - HTTP Request: POST https://jirasw.nvidia.com/rest/api/2/issue "HTTP/1.1 201 Created"
2025-04-23 15:51:57,694 - httpx - INFO - HTTP Request: POST https://jirasw.nvidia.com/rest/api/2/issue "HTTP/1.1 201 Created"
2025-04-23 15:51:57,777 - httpx - INFO - HTTP Request: POST https://jirasw.nvidia.com/rest/api/2/issue "HTTP/1.1 201 Created"
2025-04-23 15:51:57,801 - httpx - INFO - HTTP Request: POST https://jirasw.nvidia.com/rest/api/2/issue "HTTP/1.1 201 Created"
2025-04-23 15:51:57,841 - httpx - INFO - HTTP Request: POST https://jirasw.nvidia.com/rest/api/2/issue "HTTP/1.1 201 Created"
2025-04-23 15:51:58,042 - httpx - INFO - HTTP Request: POST https://jirasw.nvidia.com/rest/api/2/issue "HTTP/1.1 201 Created"
2025-04-23 15:51:58,117 - httpx - INFO - HTTP Request: POST https://jirasw.nvidia.com/rest/api/2/issue "HTTP/1.1 201 Created"
2025-04-23 15:51:58,120 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: create_jira_tickets_tool
Tool's input: {"input_text": "tasks"}
Tool's response:
### Created tasks:
- **AIQ-1166**: https://jirasw.nvidia.com/browse/AIQ-1166
- **AIQ-1169**: https://jirasw.nvidia.com/browse/AIQ-1169
- **AIQ-1170**: https://jirasw.nvidia.com/browse/AIQ-1170
- **AIQ-1164**: https://jirasw.nvidia.com/browse/AIQ-1164
- **AIQ-1171**: https://jirasw.nvidia.com/browse/AIQ-1171
- **AIQ-1168**: https://jirasw.nvidia.com/browse/AIQ-1168
- **AIQ-1172**: https://jirasw.nvidia.com/browse/AIQ-1172
- **AIQ-1174**: https://jirasw.nvidia.com/browse/AIQ-1174
- **AIQ-1165**: https://jirasw.nvidia.com/browse/AIQ-1165
- **AIQ-1175**: https://jirasw.nvidia.com/browse/AIQ-1175
- **AIQ-1173**: https://jirasw.nvidia.com/browse/AIQ-1173
- **AIQ-1167**: https://jirasw.nvidia.com/browse/AIQ-1167
------------------------------
2025-04-23 15:56:27,177 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: Can you extract por file por_requirements.txt, assign story points and create jira tickets for epics first and then followed by tasks?
Agent's thoughts:
Thought: I now know the final answer

Final Answer: Jira tickets for epics and tasks have been created. Epics: AIQ-1158, AIQ-1163, AIQ-1159, AIQ-1162, AIQ-1161, AIQ-1160. Tasks: AIQ-1166, AIQ-1169, AIQ-1170, AIQ-1164, AIQ-1171, AIQ-1168, AIQ-1172, AIQ-1174, AIQ-1165, AIQ-1175, AIQ-1173, AIQ-1167.
------------------------------
2025-04-23 15:56:27,180 - aiq.front_ends.console.console_front_end_plugin - INFO -
--------------------------------------------------
Workflow Result:
['Jira tickets for epics and tasks have been created. Epics: AIQ-1158, AIQ-1163, AIQ-1159, AIQ-1162, AIQ-1161, AIQ-1160. Tasks: AIQ-1166, AIQ-1169, AIQ-1170, AIQ-1164, AIQ-1171, AIQ-1168, AIQ-1172, AIQ-1174, AIQ-1165, AIQ-1175, AIQ-1173, AIQ-1167.']
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
