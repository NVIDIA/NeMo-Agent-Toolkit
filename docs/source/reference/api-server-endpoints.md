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

# NVIDIA NeMo Agent Toolkit API Server Endpoints

There are currently five workflow transactions that can be initiated using HTTP or WebSocket when the NeMo Agent toolkit server is running: `generate non-streaming`, `generate async`, `generate streaming`, `chat non-streaming`, and `chat streaming`. The following are types of interfaces you can use to interact with your running workflows.
  - **Generate Interface:** Uses the transaction schema defined by your workflow. The interface documentation is accessible
    using Swagger while the server is running [`http://localhost:8000/docs`](http://localhost:8000/docs).
  - **Chat Interface:** [OpenAI API Documentation](https://platform.openai.com/docs/guides/text?api-mode=chat) provides
    details on chat formats compatible with the NeMo Agent toolkit server.


## Start the NeMo Agent Toolkit Server
This section describes how to start the NeMo Agent toolkit server.
### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key.
```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

Before you use the following examples, ensure that the simple calculator workflow has been installed and is running on http://localhost:8000 by running the following commands:
```bash
uv pip install -e examples/getting_started/simple_calculator
nat serve --config_file examples/getting_started/simple_calculator/configs/config.yml
```

## Generate Non-Streaming Transaction
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
- **HTTP Response Example:** (actual response will vary based on the time of day)
  ```json
  {
    "value":"No, 4 + 4 is not greater than the current hour of the day."
  }
  ```

## Asynchronous Generate
The asynchronous generate endpoint allows clients to submit a workflow to run in the background and return a response immediately with a unique identifier for the workflow. This can be used to query the status and results of the workflow at a later time. This is useful for long-running workflows, which would otherwise cause the client to time out.

This endpoint is only available when the `async_endpoints` optional dependency extra is installed. For users installing from source, this can be done by running `uv pip install -e '.[async_endpoints]'` from the root directory of the NeMo Agent toolkit library. Similarly, for users installing from PyPI, this can be done by running `pip install "nvidia-nat[async_endpoints]"`.

Asynchronous jobs are managed using [Dask](https://docs.dask.org/en/stable/). By default, a local Dask cluster is created at start time, however you can also configure the server to connect to an existing Dask scheduler by setting the `scheduler_address` configuration parameter. The Dask scheduler is used to manage the execution of asynchronous jobs, and can be configured to run on a single machine or across a cluster of machines. Job history and metadata is stored in a SQL database using [SQLAlchemy](https://www.sqlalchemy.org/). By default, a temporary SQLite database is created at start time, however you can also configure the server to use a persistent database by setting the `db_url` configuration parameter. Refer to the [SQLAlchemy documentation](https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls) for the format of the `db_url` parameter. Any database supported by [SQLAlchemy's Asynchronous I/O extension](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html) can be used. Refer to [SQLAlchemy's Dialects](https://docs.sqlalchemy.org/en/20/dialects/index.html) for a complete list (many but not all of these support Asynchronous I/O).

### Asynchronous Specific CLI flags
The following CLI flags are available to configure the asynchronous generate endpoint when using `nat serve`:
* --scheduler_address: The address of an existing Dask scheduler to connect to. If not set, a local Dask cluster will be created.
* --db_url: The [SQLAlchemy database](https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls) URL to use for storing job history and metadata. If not set, a temporary SQLite database will be created.
* --max_concurrent_jobs: The maximum number of asynchronous jobs to run concurrently. Default is 10. This is only used when `scheduler_address` is not set.
* --dask_workers: The type of Dask workers to use. Options are `threads` for Threaded Dask workers or `processes` for Process based Dask workers. Default is `processes`. This is only used when `scheduler_address` is not set.
* --dask_log_level: The logging level for Dask. Default is `WARNING`.

:::{note}
When processes are used Dask workers, standard output and standard error from the workflow will not be visible in the server logs, however threaded Dask workers will allow workflow output to be visible in the server logs. When multiple concurrent jobs are running using threaded Dask workers, workflow output from different jobs may be interleaved in the server logs.
:::


- **Route:** `/generate/async`
- **Description:** A non-streaming transaction that submits a workflow to run in the background.
- **Optional Fields:**
  - `job_id`: A unique identifier for the job. If not provided, a UUID will be generated. It can be any string value. However, it is the caller's responsibility to ensure uniqueness. If `job_id` already exists, the server will return the latest status for that job.
  - `sync_timeout`: The maximum time in seconds to wait for the job to complete before returning a response. If the job completes in less than `sync_timeout` seconds then the response will include the job result, otherwise the `job_id` and `status` is returned. The default is `0`, which causes the request to return immediately, and maximum value for this field is `300`.
  - `expiry_seconds`: The amount of time in seconds after the job completes (either successfully or unsuccessfully), which any output files will be preserved before being deleted. Default is `3600` (1 hours), minimum value is `600` (10 minutes) and maximum value for this field is `86400` (24 hours). The text output in the response is not affected by this field.

### Example Request and Response
- HTTP Request Example:
  ```bash
  curl --request POST \
    --url http://localhost:8000/generate/async \
    --header 'Content-Type: application/json' \
    --data '{
      "input_message": "Is 4 + 4 greater than the current hour of the day"
    }'
  ```
- **HTTP Response Example:**
  ```json
  {
    "job_id": "8548a0e6-ecdc-44b0-a253-695cd746594c",
    "status": "submitted"
  }
  ```

### Example Request and Response with `sync_timeout`

- HTTP Request Example:
  ```bash
  curl --request POST \
    --url http://localhost:8000/generate/async \
    --header 'Content-Type: application/json' \
    --data '{
      "input_message": "Is 4 + 4 greater than the current hour of the day",
      "job_id": "example-job-123",
      "sync_timeout": 10
    }'
  ```
- **HTTP Response Example:**
  ```json
  {
    "created_at": "2025-09-10T20:52:24.768066",
    "error": null,
    "expires_at": "2025-09-10T21:52:30.734659Z",
    "job_id": "example-job-123",
    "output": {
      "value": "No, 4 + 4 is not greater than the current hour of the day."
    },
    "status": "success",
    "updated_at": "2025-09-10T20:52:30.734659"
  }
  ```

## Generate Streaming Transaction
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
## Generate Streaming Full Transaction
  - **Route:** `/generate/full`
  - **Description:** Same as `/generate/stream` but provides raw `IntermediateStep` objects
    without any step adaptor translations. Use the `filter_steps` query parameter to filter
    steps by type (comma-separated list) or set to 'none' to suppress all intermediate steps.
  - **HTTP Request Example:**
    ```bash
    curl --request POST \
    --url http://localhost:8000/generate/full \
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
- **HTTP Request Example with Filter:**
  By default, all intermediate steps are streamed. Use the `filter_steps` query parameter to filter steps by type (comma-separated list) or set to `none` to suppress all intermediate steps.

  Suppress all intermediate steps (only get final output):
  ```bash
  curl --request POST \
    --url 'http://localhost:8000/generate/full?filter_steps=none' \
    --header 'Content-Type: application/json' \
    --data '{"input_message": "Is 4 + 4 greater than the current hour of the day"}'
  ```
  Get only specific step types:
  ```bash
  curl --request POST \
    --url 'http://localhost:8000/generate/full?filter_steps=LLM_END,TOOL_END' \
    --header 'Content-Type: application/json' \
    --data '{"input_message": "Is 4 + 4 greater than the current hour of the day"}'
  ```

## Chat Non-Streaming Transaction
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
      ]
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
## Chat Streaming Transaction
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
      ]
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

## OpenAI Chat Completions API Compatible Endpoint

The NeMo Agent Toolkit provides full OpenAI Chat Completions API compatibility through a dedicated endpoint that enables seamless integration with existing OpenAI-compatible client libraries and workflows.

### Overview

When the OpenAI v1 compatible endpoint is configured, the toolkit creates a single endpoint that fully implements the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat) specification. This endpoint handles both streaming and non-streaming requests based on the `stream` parameter, exactly like the official OpenAI API.

#### Key Benefits

- **Drop-in Replacement**: Works with existing OpenAI client libraries without code changes
- **Full API Compatibility**: Supports all OpenAI Chat Completions API parameters
- **Industry Standard**: Familiar interface for developers already using OpenAI
- **Future-Proof**: Aligned with established API patterns and ecosystem tools

### Configuration

To enable the OpenAI v1 compatible endpoint, set `openai_api_v1_path` in your FastAPI front-end configuration:

```yaml
general:
  front_end:
    _type: fastapi
    workflow:
      method: POST
      openai_api_v1_path: /v1/chat/completions
```

#### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `openai_api_v1_path` | string | `null` | Path for the OpenAI v1 compatible endpoint |
| `openai_api_path` | string | `/chat` | Path for legacy OpenAI endpoints |
| `method` | string | `POST` | HTTP method for the endpoint |

### Endpoint Behavior

#### OpenAI v1 Compatible Mode (`openai_api_v1_path` configured)

Creates a single endpoint that handles both streaming and non-streaming requests:

- **Route**: `/v1/chat/completions` (configurable via `openai_api_v1_path`)
- **Method**: POST
- **Content-Type**: `application/json`
- **Behavior**: Routes to streaming or non-streaming based on `stream` parameter

#### Legacy Mode (`openai_api_v1_path` not configured)

Creates separate endpoints for different request types:

- **Non-streaming**: `/<openai_api_path>`
- **Streaming**: `<openai_api_path>/stream`

### Request Format

The endpoint accepts all standard OpenAI Chat Completions API parameters:

| Parameter | Type | Description | Validation |
|-----------|------|-------------|------------|
| `messages` | array | **Required.** List of messages in conversation format | min 1 item |
| `model` | string | Model identifier | - |
| `frequency_penalty` | number | Decreases likelihood of repeating tokens | -2.0 to 2.0 |
| `logit_bias` | object | Modify likelihood of specific tokens | token ID → bias |
| `logprobs` | boolean | Return log probabilities | - |
| `top_logprobs` | integer | Number of most likely tokens to return | 0 to 20 |
| `max_tokens` | integer | Maximum tokens to generate | ≥ 1 |
| `n` | integer | Number of completions to generate | 1 to 128 |
| `presence_penalty` | number | Increases likelihood of new topics | -2.0 to 2.0 |
| `response_format` | object | Specify response format | - |
| `seed` | integer | Random seed for deterministic outputs | - |
| `service_tier` | string | Service tier selection | "auto" or "default" |
| `stop` | string/array | Stop sequences | - |
| `stream` | boolean | Enable streaming responses | default: false |
| `stream_options` | object | Streaming configuration options | - |
| `temperature` | number | Sampling temperature | 0.0 to 2.0 |
| `top_p` | number | Nucleus sampling parameter | 0.0 to 1.0 |
| `tools` | array | Available function tools | - |
| `tool_choice` | string/object | Tool selection strategy | - |
| `parallel_tool_calls` | boolean | Enable parallel tool execution | default: true |
| `user` | string | End-user identifier | - |

### Usage Examples

#### cURL Examples

**Non-Streaming Request:**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/llama-3.1-8b-instruct",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "stream": false,
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

**Streaming Request:**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/llama-3.1-8b-instruct",
    "messages": [
      {"role": "user", "content": "Tell me a short story"}
    ],
    "stream": true,
    "temperature": 0.7
  }'
```

#### Client Library Examples

**OpenAI Python Client:**

```python
from openai import OpenAI

# Initialize client pointing to your NeMo Agent Toolkit server
client = OpenAI(
    api_key="not-needed",  # API key not required for local deployment
    base_url="http://localhost:8000/v1"
)

# Non-streaming chat completion
response = client.chat.completions.create(
    model="nvidia/llama-3.1-8b-instruct",
    messages=[
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    stream=False,
    temperature=0.7,
    max_tokens=150
)

print(response.choices[0].message.content)
```

**AI SDK (JavaScript/TypeScript):**

```typescript
import { openai } from '@ai-sdk/openai';
import { generateText } from 'ai';

// Configure custom OpenAI provider
const customOpenAI = openai({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'not-needed'
});

// Non-streaming generation
const { text } = await generateText({
  model: customOpenAI('nvidia/llama-3.1-8b-instruct'),
  prompt: 'Explain the benefits of renewable energy',
  temperature: 0.7,
  maxTokens: 200
});

console.log(text);
```

### Migration Guide

#### From Legacy Mode

If you're currently using legacy mode with separate endpoints:

1. **Update Configuration**: Set `openai_api_v1_path: /v1/chat/completions`
2. **Update Client Code**: Use single endpoint with `stream` parameter
3. **Test Thoroughly**: Verify both streaming and non-streaming functionality

#### From OpenAI API

If you're migrating from OpenAI's API:

1. **Update Base URL**: Point to your NeMo Agent Toolkit server
2. **Update Model Names**: Use your configured model identifiers
3. **Test Compatibility**: Verify all features work as expected

## Evaluation Endpoint
You can also evaluate workflows via the NeMo Agent toolkit `evaluate` endpoint. For more information, refer to the [NeMo Agent toolkit Evaluation Endpoint](../reference/evaluate-api.md) documentation.

## Choosing between Streaming and Non-Streaming
Use streaming if you need real-time updates or live communication where users expect immediate feedback. Use non-streaming if your workflow responds with simple updates and less feedback is needed.

## NeMo Agent Toolkit API Server Interaction Guide
A custom user interface can communicate with the API server using both HTTP requests and WebSocket connections.
For details on proper WebSocket messaging integration, refer to the [WebSocket Messaging Interface](../reference/websockets.md) documentation.
