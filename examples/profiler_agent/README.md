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

# AIQ Profiler Agent

An agent-based system for analyzing and profiling LLM applications.


## Installation

0. Start the Phoenix server locally or use a remote Phoenix server
```
docker run -p 6006:6006 -p 4317:4317 -i -t arizephoenix/phoenix:latest
```

1. Clone the repository and submodules:
   ```
   uv pip install -e examples/profiler_agent
   ```
3. Configuration
    To use a remote phoenix server, modify the config/config.yml to point to the URL

## Usage

1. Start the Phoenix server if not already running.

2. Run the profiler agent:
   ```
   aiq serve --config_file=examples/profiler_agent/configs/config.yml  --host 0.0.0.0 --port 8088
   ```

3. Launch the AIQ Toolkit User Interface by using the instructions in the [Using AIQ Toolkit UI and Server](https://github.com/NVIDIA/aiq/blob/main/docs/source/guides/using-aiqtoolkit-ui-and-server.md) guide.

4. Query the agent with natural language via the UI:
   ```
   "Show me flowchart of last 3 runs"
   "Show me the token usage of last run"
   "Analyze the last 2 runs"

   ```

## Features

- Query Phoenix traces with natural language
- Analyze LLM application performance metrics
- Generate trace visualizations
- Extract user queries across trace spans
