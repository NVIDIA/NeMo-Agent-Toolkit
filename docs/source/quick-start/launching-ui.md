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

# Launching the NVIDIA Agent Intelligence Toolkit API Server and User Interface
NVIDIA Agent Intelligence (AIQ) toolkit provides a user interface for interacting with your running workflow.

## User Interface Features
- Chat history
- Interact with Workflow via HTTP API
- Interact with Workflow via WebSocket
- Enable or disable Workflow intermediate steps
- Expand all Workflow intermediate steps by default
- Override intermediate steps with the same ID

## Walk-through
This walk-through guides you through the steps to set up and configure the AIQ toolkit user interface. Refer to `examples/basic/functions/simple_calculator/README.md` to set up the simple calculator workflow demonstrated in the following walk-through properly.


The AIQ toolkit UI is located in a git submodule at `external/aiqtoolkit-opensource-ui`. Ensure you have checked out all of the
git submodules by running the following:
```bash
git submodule update --init --recursive
```

### Start the AIQ Toolkit Server
You can start the AIQ toolkit server using the `aiq serve` command with the appropriate configuration file.

```bash
aiq serve --config_file=examples/basic/functions/simple_calculator/configs/config.yml
```
Running this command will produce the expected output as shown below:
```bash
2025-03-07 12:54:20,394 - aiq.cli.commands.start - INFO - Starting AIQ toolkit from config file: 'examples/basic/functions/simple_calculator/configs/config.yml'
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

### Verify the AIQ Toolkit Server is Running
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
> **Note:** The response depends on the current time of day the command executes.
```bash
{
  "value": "No, 8 is less than the current hour of the day (4)."
}
```

### Launch the AIQ Toolkit User Interface
After the AIQ toolkit server starts, launch the web user interface. Launching the UI requires that Node.js v18+ is installed. Instructions for downloading and installing Node.js can be found in the official [Node.js documentation](https://nodejs.org/en/download).

```bash
cd external/aiqtoolkit-opensource-ui
npm install
npm run dev
```
After the web development server starts, open a web browser and navigate to [`http://localhost:3000/`](http://localhost:3000/).
Port `3001` is an alternative port if port `3000` (default) is in use.

![AIQ toolkit Web User Interface](../_static/ui_home_page.png)

### Connect the User Interface to the AIQ Toolkit Server Using HTTP API
Configure the settings by selecting the `Settings` icon located on the bottom left corner of the home page.

![AIQ toolkit Web UI Settings](../_static/ui_generate_example_settings.png)

#### Settings Options
**Note:** It is recommended to select /chat/stream for intermediate results streaming.
- `Theme`: Light or Dark Theme.
- `HTTP URL for Chat Completion`: REST API enpoint.
  - /generate
  - /generate/stream
  - /chat
  - /chat/stream
- `WebSocket URL for Completion`: WebSocket URL to connect to running AIQ toolkit server.
- `WebSocket Schema` - Workflow schema type over WebSocket connection.

### Simple Calculator Example Conversation
Interact with the chat interface by prompting the Agent with the
message: `Is 4 + 4 greater than the current hour of the day?`

![AIQ toolkit Web UI Workflow Result](../_static/ui_generate_example.png)
