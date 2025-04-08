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

# Using the NVIDIA AgentIQ User Interface and API Server
AgentIQ provides a user interface for interacting with your running workflow.

There are currently four workflow transactions that can be initiated using HTTP or WebSocket when the AgentIQ server is
running: `generate non-streaming`,`generate streaming`, `chat non-streaming`, and `chat streaming`. The following are types of interfaces you can use to interact with your running workflows.
  - **Generate Interface:** Uses the transaction schema defined by your workflow. The interface documentation is accessible
    using Swagger while the server is running [`http://localhost:8000/docs`](http://localhost:8000/docs).
  - **Chat Interface:** [OpenAI API Documentation](https://platform.openai.com/docs/guides/text?api-mode=chat) provides
    details on chat formats compatible with the AgentIQ server.

## User Interface Features
- Chat history
- Interact with Workflow via HTTP API
- Interact with Workflow via WebSocket
- Enable or disable Workflow intermediate steps
- Expand all Workflow intermediate steps by default
- Override intermediate steps with the same ID

### Generate Non-Streaming Transaction
- **Route:** `/generate`
- **Description:** A non-streaming transaction that waits until all workflow data is available before sending the
result back to the client. The transaction schema is defined by the workflow.
- HTTP Request Example:
  ```bash
  curl --request POST \
    --url http://localhost:8000/generate \
    --header 'Content-Type: application/json' \
    --data '{
      "input_message": "Is 4 + 4 greater than the current hour of the day"
    }'
  ```
- **HTTP Response Example:**
  ```json
  {
    "value":"No, 4 + 4 is not greater than the current hour of the day."
  }
  ```
### Generate Streaming Transaction
  - **Route:** `/generate/stream`
  - **Description:** A streaming transaction that allows data to be sent in chunks as it becomes available from the
    workflow, rather than waiting for the complete response to be available.
- HTTP Request Example:
  ```bash
  curl --request POST \
    --url http://localhost:8000/generate/stream \
    --header 'Content-Type: application/json' \
    --data '{
      "input_message": "Is 4 + 4 greater than the current hour of the day"
    }'
  ```
- HTTP Intermediate Step Stream Example:
  ```json
  "intermediate_data": {
    "id": "ba5191e6-b818-4206-ac14-863112e597fe",
    "parent_id": "5db32854-d9b2-4e75-9001-543da6a55dd0",
    "type": "markdown",
    "name": "meta/llama-3.1-70b-instruct",
    "payload": "**Input:**\n```python\n[SystemMessage(content='\\nAnswer the following questions as best you can. You
                may ask the human to use the following tools:\\n\\ncalculator_multiply: This is a mathematical tool used to multiply
                two numbers together. It takes 2 numbers as an input and computes their numeric product as the output.. . Arguments
                must be provided as a valid JSON object following this format: {\\'text\\': FieldInfo(annotation=str,
                required=True)}\\ncalculator_inequality: This is a mathematical tool used to perform an inequality comparison
                between two numbers. It takes two numbers as an input and determines if one is greater or are equal.. . Arguments
                must be provided as a valid JSON object following this format: {\\'text\\': FieldInfo(annotation=str,
                required=True)}\\ncurrent_datetime: Returns the current date and time in human readable format.. . Arguments must
                be provided as a valid JSON object following this format: {\\'unused\\': FieldInfo(annotation=str, required=True)}
                \\ncalculator_divide: This is a mathematical tool used to divide one number by another. It takes 2 numbers as an
                input and computes their numeric quotient as the output.. . Arguments must be provided as a valid JSON object
                following this format: {\\'text\\': FieldInfo(annotation=str, required=True)}\\n\\nYou may respond in one of two
                formats.\\nUse the following format exactly to ask the human to use a tool:\\n\\nQuestion: the input question you
                must answer\\nThought: you should always think about what to do\\nAction: the action to take, should be one of
                [calculator_multiply,calculator_inequality,current_datetime,calculator_divide]\\nAction Input: the input to the
                action (if there is no required input, include \"Action Input: None\")  \\nObservation: wait for the human to
                respond with the result from the tool, do not assume the response\\n\\n... (this Thought/Action/Action
                Input/Observation can repeat N times. If you do not need to use a tool, or after asking the human to use any tools
                and waiting for the human to respond, you might know the final answer.)\\nUse the following format once you have
                the final answer:\\n\\nThought: I now know the final answer\\nFinal Answer: the final answer to the original input
                question\\n', additional_kwargs={}, response_metadata={}), HumanMessage(content='\\nQuestion: Is 4 + 4 greater
                than the current hour of the day\\n', additional_kwargs={}, response_metadata={}), AIMessage(content='Thought:
                To answer this question, I need to know the current hour of the day and compare it to 4 + 4.\\n\\nAction:
                current_datetime\\nAction Input: None\\n\\n', additional_kwargs={}, response_metadata={}), HumanMessage(content='The
                current time of day is 2025-03-11 16:05:11', additional_kwargs={}, response_metadata={}),
                AIMessage(content=\"Thought: Now that I have the current time, I can extract the hour and compare it to 4 + 4.
                \\n\\nAction: calculator_multiply\\nAction Input: {'text': '4 + 4'}\", additional_kwargs={}, response_metadata={}),
                HumanMessage(content='The product of 4 * 4 is 16', additional_kwargs={}, response_metadata={}),
                AIMessage(content=\"Thought: Now that I have the result of 4 + 4, which is 8, I can compare it to the current
                hour.\\n\\nAction: calculator_inequality\\nAction Input: {'text': '8 &gt; 16'}\", additional_kwargs={},
                response_metadata={}), HumanMessage(content='First number 8 is less than the second number 16',
                additional_kwargs={}, response_metadata={})]\n```\n\n**Output:**\nThought: I now know the final answer\n\nFinal
                Answer: No, 4 + 4 (which is 8) is not greater than the current hour of the day (which is 16)."
  }
  ```
- **HTTP Response Example:**
  ```json
  "data": { "value": "No, 4 + 4 (which is 8) is not greater than the current hour of the day (which is 15)." }
  ```
### Generate Streaming Raw Transaction
  - **Route:** `/generate/stream/full`
  - **Description:** Same as `/generate/stream` but provides raw `IntermediateStep` objects
    without any step adaptor translations. This endpoint is used for remote evaluation of workflows.
  - **HTTP Request Example:**
    ```bash
    curl --request POST \
    --url http://localhost:8000/generate/stream/full \
    --header 'Content-Type: application/json' \
    --data '{
      "input_message": "Is 4 + 4 greater than the current hour of the day"
    }'
    ```
- **HTTP Intermediate Step Stream Example:**
  ```json
  "intermediate_data": {"id":"dda55b33-edd1-4dde-b938-182676a42a19","parent_id":"8282eb42-01dd-4db6-9fd5-915ed4a2a032","type":"LLM_END","name":"meta/llama-3.1-70b-instruct","payload":"{\"event_type\":\"LLM_END\",\"event_timestamp\":1744051441.449566,\"span_event_timestamp\":1744051440.5072863,\"framework\":\"langchain\",\"name\":\"meta/llama-3.1-70b-instruct\",\"tags\":null,\"metadata\":{\"chat_responses\":[{\"text\":\"Thought: I now know the final answer\\n\\nFinal Answer: No, 4 + 4 (which is 8) is not greater than the current hour of the day (which is 11).\",\"generation_info\":null,\"type\":\"ChatGenerationChunk\",\"message\":{\"content\":\"Thought: I now know the final answer\\n\\nFinal Answer: No, 4 + 4 (which is 8) is not greater than the current hour of the day (which is 11).\",\"additional_kwargs\":{},\"response_metadata\":{\"finish_reason\":\"stop\",\"model_name\":\"meta/llama-3.1-70b-instruct\"},\"type\":\"AIMessageChunk\",\"name\":null,\"id\":\"run-dda55b33-edd1-4dde-b938-182676a42a19\"}}],\"chat_inputs\":null,\"tool_inputs\":null,\"tool_outputs\":null,\"tool_info\":null},\"data\":{\"input\":\"First number 8 is less than the second number 11\",\"output\":\"Thought: I now know the final answer\\n\\nFinal Answer: No, 4 + 4 (which is 8) is not greater than the current hour of the day (which is 11).\",\"chunk\":null},\"usage_info\":{\"token_usage\":{\"prompt_tokens\":37109,\"completion_tokens\":902,\"total_tokens\":38011},\"num_llm_calls\":0,\"seconds_between_calls\":0},\"UUID\":\"dda55b33-edd1-4dde-b938-182676a42a19\"}"}
  ```
- **HTTP Response Example:**
  ```json
  "data": {"value":"No, 4 + 4 (which is 8) is not greater than the current hour of the day (which is 11)."}
  ```

### Chat Non-Streaming Transaction
  - **Route:** `/chat`
  - **Description:** An OpenAI compatible non-streaming chat transaction.
  - **HTTP Request Example:**
    ```bash
    curl --request POST \
    --url http://localhost:8000/chat \
    --header 'Content-Type: application/json' \
    --data '{
      "messages": [
        {
          "role": "user",
          "content":  "Is 4 + 4 greater than the current hour of the day"
        }
      ],
      "use_knowledge_base": true
    }'
    ```
- **HTTP Response Example:**
  ```json
  {
    "id": "b92d1f05-200a-4540-a9f1-c1487bfb3685",
    "object": "chat.completion",
    "model": "",
    "created": "2025-03-11T21:12:43.671665Z",
    "choices": [
        {
            "message": {
                "content": "No, 4 + 4 (which is 8) is not greater than the current hour of the day (which is 16).",
                "role": null
            },
            "finish_reason": "stop",
            "index": 0
        }
    ],
    "usage": {
        "prompt_tokens": 0,
        "completion_tokens": 20,
        "total_tokens": 20
    }
  }
  ```
### Chat Streaming Transaction
  - **Route:** `/chat/stream`
  - **Description:** An OpenAI compatible streaming chat transaction.
  - **HTTP Request Example:**
    ```bash
    curl --request POST \
    --url http://localhost:8000/chat/stream \
    --header 'Content-Type: application/json' \
    --data '{
      "messages": [
        {
          "role": "user",
          "content":  "Is 4 + 4 greater than the current hour of the day"
        }
      ],
      "use_knowledge_base": true
    }'
    ```
- **HTTP Intermediate Step Example:**
  ```json
  "intermediate_data": {
    "id": "9ed4bce7-191c-41cb-be08-7a72d30166cc",
    "parent_id": "136edafb-797b-42cd-bd11-29153359b193",
    "type": "markdown",
    "name": "meta/llama-3.1-70b-instruct",
    "payload": "**Input:**\n```python\n[SystemMessage(content='\\nAnswer the following questions as best you can. You
                may ask the human to use the following tools:\\n\\ncalculator_multiply: This is a mathematical tool used to multiply
                two numbers together. It takes 2 numbers as an input and computes their numeric product as the output.. . Arguments
                must be provided as a valid JSON object following this format: {\\'text\\': FieldInfo(annotation=str,
                required=True)}\\ncalculator_inequality: This is a mathematical tool used to perform an inequality comparison
                between two numbers. It takes two numbers as an input and determines if one is greater or are equal.. .
                Arguments must be provided as a valid JSON object following this format: {\\'text\\': FieldInfo(annotation=str,
                required=True)}\\ncurrent_datetime: Returns the current date and time in human readable format.. . Arguments
                must be provided as a valid JSON object following this format: {\\'unused\\': FieldInfo(annotation=str,
                required=True)}\\ncalculator_divide: This is a mathematical tool used to divide one number by another. It takes
                2 numbers as an input and computes their numeric quotient as the output.. . Arguments must be provided as a
                valid JSON object following this format: {\\'text\\': FieldInfo(annotation=str, required=True)}\\n\\nYou may
                respond in one of two formats.\\nUse the following format exactly to ask the human to use a tool:\\n\\nQuestion:
                the input question you must answer\\nThought: you should always think about what to do\\nAction: the action to
                take, should be one of [calculator_multiply,calculator_inequality,current_datetime,calculator_divide]\\nAction
                Input: the input to the action (if there is no required input, include \"Action Input: None\")  \\nObservation:
                wait for the human to respond with the result from the tool, do not assume the response\\n\\n...
                (this Thought/Action/Action Input/Observation can repeat N times. If you do not need to use a tool, or after
                asking the human to use any tools and waiting for the human to respond, you might know the final answer.)\\nUse
                the following format once you have the final answer:\\n\\nThought: I now know the final answer\\nFinal Answer:
                the final answer to the original input question\\n', additional_kwargs={}, response_metadata={}),
                HumanMessage(content='\\nQuestion: Is 4 + 4 greater than the current hour of the day\\n', additional_kwargs={},
                response_metadata={}), AIMessage(content='Thought: To answer this question, I need to know the current hour of
                the day and compare it to 4 + 4.\\n\\nAction: current_datetime\\nAction Input: None\\n\\n', additional_kwargs={},
                response_metadata={}), HumanMessage(content='The current time of day is 2025-03-11 16:24:52',
                additional_kwargs={}, response_metadata={}), AIMessage(content=\"Thought: Now that I have the current time, I can
                extract the hour and compare it to 4 + 4.\\n\\nAction: calculator_multiply\\nAction Input: {'text': '4 + 4'}\",
                additional_kwargs={}, response_metadata={}), HumanMessage(content='The product of 4 * 4 is 16',
                additional_kwargs={}, response_metadata={}), AIMessage(content=\"Thought: Now that I have the result of 4 + 4,
                which is 8, I can compare it to the current hour.\\n\\nAction: calculator_inequality\\nAction Input:
                {'text': '8 &gt; 16'}\", additional_kwargs={}, response_metadata={}), HumanMessage(content='First number 8 is
                less than the second number 16', additional_kwargs={}, response_metadata={})]\n```\n\n**Output:**\nThought: I now
                know the final answer\n\nFinal Answer: No, 4 + 4 (which is 8) is not greater than the current hour of the day
                (which is 16)."
  }
  ```
- **HTTP Response Example:**
  ```json
  "data": {
    "id": "194d22dc-6c1b-44ee-a8d7-bf2b59c1cb6b",
    "choices": [
        {
            "message": {
                "content": "No, 4 + 4 (which is 8) is not greater than the current hour of the day (which is 16).",
                "role": null
            },
            "finish_reason": "stop",
            "index": 0
        }
    ],
    "created": "2025-03-11T21:24:56.961939Z",
    "model": "",
    "object": "chat.completion.chunk"
  }
  ```

### Choosing between Streaming and Non-Streaming
Use streaming if you need real-time updates or live communication where users expect immediate feedback. Use non-streaming if your workflow responds with simple updates and less feedback is needed.

## Walk-through
In this walk-through, we will guide you through the steps to set up and configure the AgentIQ user interface. Refer to `examples/simple_calculator/README.md` to set up the simple calculator workflow demonstrated in the following walk-through properly.


The AgentIQ UI is located in a git submodule at `external/agentiq-opensource-ui`. Ensure you have checked out all of the
git submodules by running the following:
```bash
git submodule update --init --recursive
```

### Start the AgentIQ Server
You can start the AgentIQ server using the `aiq serve` command with the appropriate configuration file.

```bash
aiq serve --config_file=examples/simple_calculator/configs/config.yml
```
Running this command will produce the expected output as shown below:
```bash
2025-03-07 12:54:20,394 - aiq.cli.commands.start - INFO - Starting AgentIQ from config file: 'examples/simple_calculator/configs/config.yml'
WARNING:  Current configuration will not reload as not all conditions are met, please refer to documentation.
INFO:     Started server process [47250]
INFO:     Waiting for application startup.
2025-03-07 12:54:20,730 - aiq.profiler.decorators - INFO - Langchain callback handler registered
2025-03-07 12:54:21,313 - aiq.agent.react_agent.agent - INFO - Filling the prompt variables "tools" and "tool_names", using the tools provided in the config.
2025-03-07 12:54:21,313 - aiq.agent.react_agent.agent - INFO - Initialized ReAct Agent Graph
2025-03-07 12:54:21,316 - aiq.agent.react_agent.agent - INFO - ReAct Graph built and compiled successfully
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)
  Current configuration will not reload as not all conditions are met, please refer to documentation.
INFO:     Started server process [47250]
INFO:     Waiting for application startup.
2025-03-07 12:54:20,730 - aiq.profiler.decorators - INFO - Langchain callback handler registered
2025-03-07 12:54:21,313 - aiq.agent.react_agent.agent - INFO - Filling the prompt variables "tools" and "tool_names", using the tools provided in the config.
2025-03-07 12:54:21,313 - aiq.agent.react_agent.agent - INFO - Initialized ReAct Agent Graph
2025-03-07 12:54:21,316 - aiq.agent.react_agent.agent - INFO - ReAct Graph built and compiled successfully
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)
```

### Verify the AgentIQ Server is Running
After the server is running, you can make HTTP requests to interact with the workflow.

```bash
curl --request POST \
  --url http://localhost:8000/generate \
  --header 'Content-Type: application/json' \
  --data '{
    "input_message": "Is 4 + 4 greater than the current hour of the day?",
    "use_knowledge_base": true
}'
```

Running this command will produce the following expected output:
**Note:** The response depends on the current time of day the command executes.
```bash
{
  "value": "No, 8 is less than the current hour of the day (4)."
}
```

### Launch the AgentIQ User Interface
After the AgentIQ server starts, launch the web user interface. Launching the UI requires that Node.js v18+ is installed. Instructions for downloading and installing Node.js can be found in the official documentation [here](https://nodejs.org/en/download).

```bash
cd external/agentiq-opensource-ui
npm install
npm run dev
```
After the web development server starts, open a web browser and navigate to [`http://localhost:3000/`](http://localhost:3000/).
Port `3001` is an alternative port if port `3000` (default) is in use.

![AgentIQ Web User Interface](../_static/ui_home_page.png)

### Connect the User Interface to the AgentIQ Server Using HTTP API
Configure the settings by selecting the `Settings` icon located on the bottom left corner of the home page.

![AgentIQ Web UI Settings](../_static/ui_generate_example_settings.png)

#### Settings Options
**Note:** It is recommended to select /chat/stream for intermediate results streaming.
- `Theme`: Light or Dark Theme.
- `HTTP URL for Chat Completion`: REST API enpoint.
  - /generate
  - /generate/stream
  - /chat
  - /chat/stream
- `WebSocket URL for Completion`: WebSocket URL to connect to running AgentIQ server.
- `WebSocket Schema` - Workflow schema type over WebSocket connection.

### Simple Calculator Example Conversation
Interact with the chat interface by prompting the Agent with the
message: `Is 4 + 4 greater than the current hour of the day?`

![AgentIQ Web UI Workflow Result](../_static/ui_generate_example.png)

## AgentIQ API Server Interaction Guide
A custom user interface can communicate with the API server using both HTTP requests and WebSocket connections.
For details on proper WebSocket messaging integration, refer to the [WebSocket Messaging Interface](../references/websockets.md) documentation.
