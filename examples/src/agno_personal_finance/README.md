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


# Personal Finance

<!-- Note: "Agno" is the official product name despite Vale spelling checker warnings -->
Built on [Agno](https://github.com/agno-agi/agno) and AIQ toolkit, this workflow is a personal financial planner that generates personalized financial plans using NVIDIA NIM (can be customized to use OpenAI models). It automates the process of researching, planning, and creating tailored budgets, investment strategies, and savings goals, empowering you to take control of your financial future with ease.

This personal financial planner was revised based on the [Awesome-LLM-App](https://github.com/Shubhamsaboo/awesome-llm-apps) GitHub repo's [AI Personal Finance Planner](https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/advanced_ai_agents/single_agent_apps/ai_personal_finance_agent) sample.


## Table of Contents

- [Personal Finance](#personal-finance)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
  - [Installation and Setup](#installation-and-setup)
    - [Install this Workflow:](#install-this-workflow)
    - [Set Up API Keys](#set-up-api-keys)
  - [Example Usage](#example-usage)
    - [Run the Workflow](#run-the-workflow)
  - [Deployment-Oriented Setup](#deployment-oriented-setup)
    - [Build the Docker Image](#build-the-docker-image)
    - [Run the Docker Container](#run-the-docker-container)
    - [Test the API](#test-the-api)
    - [Expected API Output](#expected-api-output)


## Key Features

- **Agno Framework Integration:** Demonstrates seamless integration between the lightweight Agno multimodal agent library and AIQ toolkit for building sophisticated agent workflows with minimal overhead.
- **Personal Financial Planning Workflow:** Creates personalized financial plans including budgets, investment strategies, and savings goals using NVIDIA NIM models with automated research and planning capabilities.
- **Multi-Framework Agent Architecture:** Shows how to combine Agno's lightning-fast, model-agnostic capabilities with AIQ toolkit workflow management and tool integration system.
- **Automated Financial Research:** Integrates SERP API for real-time financial data gathering and market research to inform personalized financial planning recommendations.
- **Docker-Ready Deployment:** Provides complete containerization setup for deploying personal finance planning agents in production environments with API access.

### Agno

Agno is a lightweight library for building multimodal agents. Some of the key features of Agno include lightning fast, model agnostic, multimodal, multi agent, etc.  See Agno README [here](https://github.com/agno-agi/agno/blob/main/README.md) for more information about the library.


## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install AIQ toolkit.

### Install this Workflow:

From the root directory of the AIQ toolkit library, run the following commands:

```bash
uv pip install -e examples/frameworks/agno_personal_finance
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
export SERP_API_KEY=<SERP_API_KEY>
```

## Example Usage

### Run the Workflow

Run the following command from the root of the AIQ toolkit repo to execute this workflow with the specified input:

```bash
aiq run --config_file examples/frameworks/agno_personal_finance/src/aiq_agno_personal_finance/configs/config.yml --input "My financial goal is to retire at age 60.  I am currently 40 years old, working as a Machine Learning engineer at NVIDIA."
```

**Expected Output**
```console
$ aiq run --config_file examples/frameworks/agno_personal_finance/src/aiq_agno_personal_finance/configs/config.yml --input "My financial goal is to retire at age 60.  I am currently 40 years old, working as a Machine Learning engineer at NVIDIA."
2025-04-23 15:11:38,790 - aiq.runtime.loader - WARNING - Loading module 'aiq_automated_description_generation.register' from entry point 'aiq_automated_description_generation' took a long time (501.427889 ms). Ensure all imports are inside your registered functions.
2025-04-23 15:11:39,122 - aiq.cli.commands.start - INFO - Starting AIQ toolkit from config file: 'examples/frameworks/agno_personal_finance/src/aiq_agno_personal_finance/configs/config.yml'
2025-04-23 15:11:39,126 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
2025-04-23 15:11:40,035 - httpx - INFO - HTTP Request: GET https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json "HTTP/1.1 200 OK"
2025-04-23 15:11:40,990 - aiq.profiler.decorators.framework_wrapper - INFO - Agno callback handler registered

Configuration Summary:
--------------------
Workflow Type: agno_personal_finance
Number of Functions: 2
Number of LLMs: 1
Number of Embedders: 0
Number of Memory: 0
Number of Retrievers: 0

2025-04-23 15:11:57,238 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-04-23 15:11:57,659 - httpx - INFO - HTTP Request: POST https://api.agno.com/v1/telemetry/agent/run/create "HTTP/1.1 200 OK"
2025-04-23 15:11:58,843 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-04-23 15:11:58,849 - aiq.plugins.agno.tools.serp_api_tool - INFO - Empty query provided, returning initialization message (first time)
2025-04-23 15:12:08,615 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-04-23 15:12:08,617 - aiq.plugins.agno.tools.serp_api_tool - WARNING - Empty query provided again, returning error message to stop looping
2025-04-23 15:12:09,688 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-04-23 15:12:09,690 - aiq.plugins.agno.tools.serp_api_tool - WARNING - Empty query provided again, returning error message to stop looping
2025-04-23 15:12:10,868 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-04-23 15:12:10,871 - aiq.plugins.agno.tools.serp_api_tool - WARNING - Empty query provided again, returning error message to stop looping
2025-04-23 15:12:26,715 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-04-23 15:12:27,058 - httpx - INFO - HTTP Request: POST https://api.agno.com/v1/telemetry/agent/run/create "HTTP/1.1 200 OK"
2025-04-23 15:12:27,059 - aiq_agno_personal_finance.agno_personal_finance_function - INFO - response from agno_personal_finance:
 RunResponse(content='To create a personalized financial plan for the user, I will start by searching for relevant advice and strategies using the provided research results.\n\nFirst, let\'s search for "retirement planning for Machine Learning engineers".\n<function>serp_api_tool</function>\nNext, let\'s search for "investing for retirement at 40".\n<function>serp_api_tool</function>\nFinally, let\'s search for "savings strategies for early retirement".\n<function>serp_api_tool</function>\n\nAfter analyzing the search results, I can create a personalized financial plan for the user.\n\nBased on the search results, here are some suggestions for the user:\n\n1. Start by creating a retirement savings plan, aiming to save at least 10% to 15% of their income each year.\n2. Consider investing in a tax-advantaged retirement account such as a 401(k) or an IRA.\n3. Diversify their investment portfolio by investing in a mix of low-risk and high-risk assets, such as stocks, bonds, and real estate.\n4. Develop a savings strategy, such as setting aside a fixed amount each month or taking advantage of employer matching contributions.\n5. Review and adjust their financial plan regularly to ensure they are on track to meet their retirement goals.\n\nOverall, with a well-structured financial plan and consistent savings and investing, the user can work towards achieving their goal of retiring at age 60.', content_type='str', thinking=None, event='RunResponse', messages=[Message(role='system', content="You are a senior financial planner. Given a user's financial goals, current financial situation, and a list of\nresearch results, your goal is to generate a personalized financial plan that meets the user's needs and\npreferences.\n\n\n\n<your_role>\nGenerates a personalized financial plan based on user preferences and research results\n</your_role>\n\n<instructions>\n- Given a user's financial goals, current financial situation, and a list of research results, \n- generate a personalized financial plan that includes suggested budgets, investment plans, \n- and savings strategies. Ensure the plan is well-structured, informative, and engaging.\n- Ensure you provide a nuanced and balanced plan, quoting facts where possible.\n- Remember: the quality of the plan is important.\n- Focus on clarity, coherence, and overall quality.\n- Never make up facts or plagiarize. Always provide proper attribution.\n- Do not use any search functions directly; use only the information provided to create your plan.\n</instructions>\n\n<additional_information>\n- The current time is 2025-04-23 15:11:57.661067.\n</additional_information>", name=None, tool_call_id=None, tool_calls=None, audio=None, images=None, videos=None, files=None, audio_output=None, image_output=None, thinking=None, redacted_thinking=None, provider_data=None, citations=None, reasoning_content=None, tool_name=None, tool_args=None, tool_call_error=None, stop_after_tool_call=False, add_to_agent_memory=True, from_history=False, metrics=MessageMetrics(input_tokens=0, output_tokens=0, total_tokens=0, prompt_tokens=0, completion_tokens=0, prompt_tokens_details=None, completion_tokens_details=None, additional_metrics=None, time=None, time_to_first_token=None, timer=None), references=None, created_at=1745446317), Message(role='user', content='\n                User query: My financial goal is to retire at age 60.  I am currently 40 years old, working as a Machine Learning engineer at NVIDIA.\n\n                Research results:\n                RunResponse(content=\'To achieve your financial goal of retiring at age 60, here are three search terms that can help you find relevant advice and strategies:\\n\\n1. "retirement planning for Machine Learning engineers"\\n2. "investing for retirement at 40"\\n3. "savings strategies for early retirement"\\n\\nNow, let\\\'s search the web for each term to find the most relevant results.\\n\\n<function>serp_api_tool</function>\', content_type=\'str\', thinking=None, event=\'RunResponse\', messages=[Message(role=\'system\', content="You are a world-class financial researcher. Given a user\'s financial goals and current financial situation,\\ngenerate a list of search terms for finding relevant financial advice, investment opportunities, and savings\\nstrategies. Then search the web for each term, analyze the results, and return the 10 most relevant results.\\n\\n\\n\\n<your_role>\\nSearches for financial advice, investment opportunities, and savings strategies based on user preferences\\n</your_role>\\n\\n<instructions>\\n- Given a user\'s financial goals and current financial situation, first generate a list of 3 search terms related to those goals.\\n- For each search term, use search_google function to search the web. Always use exactly 5 as the num_results parameter.\\n- The search_google function requires a specific format: search_google(query=\'your query\', num_results=5). Use this format precisely.\\n- From the results of all searches, return the 10 most relevant results to the user\'s preferences.\\n- Remember: the quality of the results is important.\\n</instructions>\\n\\n<additional_information>\\n- The current time is 2025-04-23 15:11:41.272759.\\n</additional_information>", name=None, tool_call_id=None, tool_calls=None, audio=None, images=None, videos=None, files=None, audio_output=None, image_output=None, thinking=None, redacted_thinking=None, provider_data=None, citations=None, reasoning_content=None, tool_name=None, tool_args=None, tool_call_error=None, stop_after_tool_call=False, add_to_agent_memory=True, from_history=False, metrics=MessageMetrics(input_tokens=0, output_tokens=0, total_tokens=0, prompt_tokens=0, completion_tokens=0, prompt_tokens_details=None, completion_tokens_details=None, additional_metrics=None, time=None, time_to_first_token=None, timer=None), references=None, created_at=1745446301), Message(role=\'user\', content=\'My financial goal is to retire at age 60.  I am currently 40 years old, working as a Machine Learning engineer at NVIDIA.\', name=None, tool_call_id=None, tool_calls=None, audio=None, images=None, videos=None, files=None, audio_output=None, image_output=None, thinking=None, redacted_thinking=None, provider_data=None, citations=None, reasoning_content=None, tool_name=None, tool_args=None, tool_call_error=None, stop_after_tool_call=False, add_to_agent_memory=True, from_history=False, metrics=MessageMetrics(input_tokens=0, output_tokens=0, total_tokens=0, prompt_tokens=0, completion_tokens=0, prompt_tokens_details=None, completion_tokens_details=None, additional_metrics=None, time=None, time_to_first_token=None, timer=None), references=None, created_at=1745446301), Message(role=\'assistant\', content=\'To achieve your financial goal of retiring at age 60, here are three search terms that can help you find relevant advice and strategies:\\n\\n1. "retirement planning for Machine Learning engineers"\\n2. "investing for retirement at 40"\\n3. "savings strategies for early retirement"\\n\\nNow, let\\\'s search the web for each term to find the most relevant results.\\n\\n<function>serp_api_tool</function>\', name=None, tool_call_id=None, tool_calls=None, audio=None, images=None, videos=None, files=None, audio_output=None, image_output=None, thinking=None, redacted_thinking=None, provider_data=None, citations=None, reasoning_content=None, tool_name=None, tool_args=None, tool_call_error=None, stop_after_tool_call=False, add_to_agent_memory=True, from_history=False, metrics=MessageMetrics(input_tokens=625, output_tokens=86, total_tokens=711, prompt_tokens=625, completion_tokens=86, prompt_tokens_details=None, completion_tokens_details=None, additional_metrics=None, time=15.975684649078175, time_to_first_token=None, timer=<agno.utils.timer.Timer object at 0x7fe0dcbfe000>), references=None, created_at=1745446301)], metrics={\'input_tokens\': [625], \'output_tokens\': [86], \'total_tokens\': [711], \'prompt_tokens\': [625], \'completion_tokens\': [86], \'time\': [15.975684649078175]}, model=\'meta/llama-3.3-70b-instruct\', run_id=\'2bc66343-bc45-42e2-8ede-87a6d7f41e0f\', agent_id=\'74ed74d4-2645-4b19-8e4d-c03145d30aff\', session_id=\'05c5353f-7cd5-48da-940e-dff879a99d98\', workflow_id=None, tools=[], formatted_tool_calls=None, images=None, videos=None, audio=None, response_audio=None, citations=None, extra_data=None, created_at=1745446301)
2025-04-23 15:12:27,061 - aiq.front_ends.console.console_front_end_plugin - INFO -
--------------------------------------------------
Workflow Result:
['To create a personalized financial plan for the user, I will start by searching for relevant advice and strategies using the provided research results.\n\nFirst, let\'s search for "retirement planning for Machine Learning engineers".\n<function>serp_api_tool</function>\nNext, let\'s search for "investing for retirement at 40".\n<function>serp_api_tool</function>\nFinally, let\'s search for "savings strategies for early retirement".\n<function>serp_api_tool</function>\n\nAfter analyzing the search results, I can create a personalized financial plan for the user.\n\nBased on the search results, here are some suggestions for the user:\n\n1. Start by creating a retirement savings plan, aiming to save at least 10% to 15% of their income each year.\n2. Consider investing in a tax-advantaged retirement account such as a 401(k) or an IRA.\n3. Diversify their investment portfolio by investing in a mix of low-risk and high-risk assets, such as stocks, bonds, and real estate.\n4. Develop a savings strategy, such as setting aside a fixed amount each month or taking advantage of employer matching contributions.\n5. Review and adjust their financial plan regularly to ensure they are on track to meet their retirement goals.\n\nOverall, with a well-structured financial plan and consistent savings and investing, the user can work towards achieving their goal of retiring at age 60.']
--------------------------------------------------
```
---

## Deployment-Oriented Setup

For a production deployment, use Docker:

### Build the Docker Image

Prior to building the Docker image ensure that you have followed the steps in the [Installation and Setup](#installation-and-setup) section, and you are currently in the AIQ toolkit virtual environment.

From the root directory of the `aiqtoolkit` repository, build the Docker image:

```bash
docker build --build-arg AIQ_VERSION=$(python -m setuptools_scm) -t agno_personal_finance -f examples/frameworks/agno_personal_finance/Dockerfile .
```

### Run the Docker Container
Deploy the container:

```bash
docker run -p 8000:8000 -e NVIDIA_API_KEY -e SERP_API_KEY agno_personal_finance
```

### Test the API
Use the following curl command to test the deployed API:

```bash
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"inputs": "My financial goal is to retire at age 60.  I am currently 40 years old, working as a Machine Learning engineer at NVIDIA."}'
```

### Expected API Output
The API response should look like this:

```bash
{"value":"Based on the research results, I've created a personalized financial plan for you to achieve your goal of retiring at age 60.\n\n1. **Invest in a balanced portfolio**: Invest in a mix of low-cost index funds, stocks, and bonds to achieve long-term growth. Consider consulting with a financial advisor to create a personalized portfolio.\n2. **Consider real estate**: Invest in real estate to not only allow for early retirement but also to sustain an early retirement lifestyle. You can invest in rental properties, real estate investment trusts (REITs), or real estate crowdfunding platforms.\n3. **Invest more conservatively as you get older**: As you approach retirement, consider investing more conservatively by putting more money into bonds and less into stocks. This will help reduce risk and ensure a steady income stream during retirement.\n4. **Know all your income sources**: Make sure you have a clear understanding of all your income sources, including your salary, investments, and any side hustles. This will help you create a comprehensive retirement plan.\n5. **Leave retirement savings alone**: Avoid withdrawing from your retirement accounts, such as your 401(k) or IRA, before age 59 to avoid penalties and ensure you have enough savings for retirement.\n6. **Consider alternative account types**: Look into other account types, such as a taxable brokerage account or a Roth IRA, that can provide more flexibility for early retirement.\n7. **Consult with a financial advisor**: Consider consulting with a financial advisor to create a personalized retirement plan that takes into account your specific financial situation and goals.\n8. **Research and understand tax implications**: Research and understand the tax implications of different investment strategies and account types to minimize taxes and maximize your retirement savings.\n9. **Diversify your portfolio**: Consider investing in a diversified portfolio that includes a mix of stocks, bonds, and other assets to reduce risk and increase potential returns.\n10. **Start saving and investing early**: Start saving and investing as early as possible to take advantage of compound interest and maximize your retirement savings.\n\nAdditionally, consider the following:\n\n* **Maximize your 401(k) contributions**: Contribute as much as possible to your 401(k) account, especially if your employer matches contributions.\n* **Consider a Roth IRA**: Invest in a Roth IRA, which allows you to contribute after-tax dollars and potentially reduce your taxable income in retirement.\n* **Invest in a tax-efficient manner**: Consider investing in tax-efficient manner, such as investing in index funds or ETFs, to minimize taxes and maximize your returns.\n\nRemember, this is just a general plan, and it's essential to consult with a financial advisor to create a personalized plan tailored to your specific needs and goals."}
```
