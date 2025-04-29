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
# Simple A2A Client Example
## Run the sample currency exchange agent in the A2A repository
Clone the [A2A repository](https://github.com/google/A2A/) and run the langgraph sample exchange agent by following the instructions [here](https://github.com/google/A2A/tree/main/samples/python/agents/langgraph#setup--running).


## Install and run the simple A2A client example in this repository

Install `aiqtoolkit` from source by following the instructions [here](https://github.com/NVIDIA/AgentIQ/). Run the simple A2A client example:
```bash
aiq run --config_file=examples/simple_a2a/configs/config.yml --input "What was the exchange rate between usd and euro 100 days ago?"
```

### Sample output

```bash
5-04-28 17:07:22,360 - aiq.agent.react_agent.agent - INFO - The user's question was: What was the exchange rate between usd and euro 100 days ago?
2025-04-28 17:07:22,361 - aiq.agent.react_agent.agent - INFO - The agent's thoughts are:
Thought: To find the exchange rate between USD and EUR 100 days ago, I need to know the current date and then subtract 100 days from it. After that, I can use the get_a2a_exchange_rate tool to find the exchange rate for the given date.

Action: current_datetime
Action Input: None


2025-04-28 17:07:22,365 - aiq.agent.react_agent.agent - INFO - Calling tool current_datetime with input: None

2025-04-28 17:07:22,365 - aiq.agent.react_agent.agent - INFO - Successfully parsed structured tool input from Action Input
2025-04-28 17:07:22,371 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1
2025-04-28 17:07:23,626 - aiq.agent.react_agent.agent - INFO -

The agent's thoughts are:
Thought: Now that I have the current date, I can subtract 100 days from it to find the date 100 days ago.

Action: get_a2a_exchange_rate
Action Input: {"tool_input": "{\"date\": \"2025-01-19\", \"base\": \"USD\", \"target\": \"EUR\"}"}
2025-04-28 17:07:23,629 - aiq.agent.react_agent.agent - INFO - Calling tool get_a2a_exchange_rate with input: {"tool_input": "{\"date\": \"2025-01-19\", \"base\": \"USD\", \"target\": \"EUR\"}"}
2025-04-28 17:07:23,630 - aiq.agent.react_agent.agent - INFO - Successfully parsed structured tool input from Action Input
2025-04-28 17:07:23,632 - aiq.tool.a2a.a2a_function - INFO - A2A Tool input: {"date": "2025-01-19", "base": "USD", "target": "EUR"}
>>>>>>>>>>>>>>>>>>>>>>>>> SNIPPED >>>>>>>>>>>>>>>>>>>>>>>>>
2025-04-28 17:07:27,615 - httpx - INFO - HTTP Request: POST http://localhost:10000 "HTTP/1.1 200 OK"
2025-04-28 17:07:27,617 - aiq.tool.a2a.a2a_client - INFO - Stream event: {"jsonrpc":"2.0","id":"c91ee55f7a05443c9070020bc92674e0","result":{"id":"1bdde5253bc64f8798b424a9c1c3df47","status":{"state":"working","message":{"role":"agent","parts":[{"type":"text","text":"Looking up the exchange rates..."}]},"timestamp":"2025-04-28T17:07:24.414047"},"final":false}}
2025-04-28 17:07:27,617 - aiq.tool.a2a.a2a_client - INFO - Stream event: {"jsonrpc":"2.0","id":"c91ee55f7a05443c9070020bc92674e0","result":{"id":"1bdde5253bc64f8798b424a9c1c3df47","status":{"state":"working","message":{"role":"agent","parts":[{"type":"text","text":"Processing the exchange rates.."}]},"timestamp":"2025-04-28T17:07:25.240253"},"final":false}}
2025-04-28 17:07:27,618 - aiq.tool.a2a.a2a_client - INFO - Stream event: {"jsonrpc":"2.0","id":"c91ee55f7a05443c9070020bc92674e0","result":{"id":"1bdde5253bc64f8798b424a9c1c3df47","artifact":{"parts":[{"type":"text","text":"I retrieved the exchange rate between USD and EUR for January 17, 2025. The exchange rate is 1 USD to 0.97106 EUR. Note that the date provided was unavailable and the rate from the closest available date was provided."}],"index":0,"append":false}}}
2025-04-28 17:07:27,619 - aiq.tool.a2a.a2a_client - INFO - Stream event: {"jsonrpc":"2.0","id":"c91ee55f7a05443c9070020bc92674e0","result":{"id":"1bdde5253bc64f8798b424a9c1c3df47","status":{"state":"completed","timestamp":"2025-04-28T17:07:27.612059"},"final":true}}
2025-04-28 17:07:27,619 - aiq.tool.a2a.a2a_client - INFO - Task result: I retrieved the exchange rate between USD and EUR for January 17, 2025. The exchange rate is 1 USD to 0.97106 EUR. Note that the date provided was unavailable and the rate from the closest available date was provided.
2025-04-28 17:07:27,623 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1
2025-04-28 17:07:30,335 - aiq.agent.react_agent.agent - INFO -

The agent's thoughts are:
Thought: I now know the final answer

Final Answer: The exchange rate between USD and EUR 100 days ago (on January 19, 2025) was approximately 1 USD to 0.97106 EUR.
2025-04-28 17:07:30,337 - aiq.observability.async_otel_listener - INFO - Intermediate step stream completed. No more events will arrive.
2025-04-28 17:07:30,337 - aiq.front_ends.console.console_front_end_plugin - INFO - --------------------------------------------------
Workflow Result:
['The exchange rate between USD and EUR 100 days ago (on January 19, 2025) was approximately 1 USD to 0.97106 EUR.']
```

## Sample usage with tool access via MCP and agent access via A2A

The configuration file `examples/simple_a2a/configs/config-mcp-client.yml` uses MCP to get the current date and time and then uses A2A to get the exchange rate for the given date.

To run this sample configuration file, start the MCP time service by running the following the instructions in the [MCP Server Example README](../mcp_server/README.md).

Run the `simple_a2a` workflow with the configuration file:
```bash
aiq run --config_file=examples/simple_a2a/configs/config-mcp-client.yml --input "What was the exchange rate between usd and euro 100 days ago?"
```
