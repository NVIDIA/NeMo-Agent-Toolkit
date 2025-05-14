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

<!--
  SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# A Simple Plot Chart Agent

A minimal example demonstrating an E2E chart plotting agentic workflow fully configured by a YAML file. This workflow leverages the AIQ Toolkit plugin system and `Builder` to integrate pre-built and custom tools into the workflow. Key elements are summarized below:

## Table of Contents

* [Key Features](#key-features)
* [Installation and Usage](#installation-and-setup)
* [Example Usage](#example-usage)

## Key Features

- **Pre-built Tools:** Leverages core AIQ Toolkit library tools.
- **Custom Plugin System:** Developers can bring in new tools using plugins.
- **High-level API:** Enables defining functions that transform into asynchronous LangChain tools.
- **Agentic Workflows:** Fully configurable via YAML for flexibility and productivity.
- **Ease of Use:** Simplifies developer experience and deployment.

## Installation and Setup

### Setup Virtual Environment and Install AIQ Toolkit

If you have not already done so, follow the instructions in the [Install Guide](../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install AIQ Toolkit.

### Install this Workflow:

From the root directory of the AIQ Toolkit library, run the following commands:

```bash
uv pip install -e examples/plot_charts
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

## Example Usage

### Run the Workflow

Run the following command from the root of the AIQ Toolkit repo to execute this workflow with the specified input:

```bash
aiq run --config_file examples/plot_charts/configs/config.yml  --input "make a line chart for me"
```

**Expected Output**

```console
/home/coder/dev/ai-query-engine
/home/coder/dev/ai-query-engine/examples/plot_charts/example_data.json
routed_output= line_chart
**line_chart** xValues=['2020', '2021', '2022', '2023', '2024'] yValues=[{'data': [2, 5, 2.2, 7.5, 3], 'label': 'USA'}, {'data': [2, 5.5, 2, 8.5, 1.5], 'label': 'EMEA'}] chart_name='USA vs EMEA Data by Year'
y=
 {'data': [2, 5, 2.2, 7.5, 3], 'label': 'USA'}
label=
 USA
y_data_points=
 [2, 5, 2.2, 7.5, 3]
2024-11-19 17:13:35,104 - matplotlib.category - INFO - Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.
2024-11-19 17:13:35,109 - matplotlib.category - INFO - Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.
2024-11-19 17:13:35,133 - matplotlib.category - INFO - Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.
2024-11-19 17:13:35,136 - matplotlib.category - INFO - Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.
y=
 {'data': [2, 5.5, 2, 8.5, 1.5], 'label': 'EMEA'}
label=
 EMEA
y_data_points=
 [2, 5.5, 2, 8.5, 1.5]
2024-11-19 17:13:35,148 - matplotlib.category - INFO - Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.
2024-11-19 17:13:35,151 - matplotlib.category - INFO - Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.
2024-11-19 17:13:35,164 - matplotlib.category - INFO - Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.
2024-11-19 17:13:35,167 - matplotlib.category - INFO - Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.
**bot_message** line chart is generated, the image path can be found here : ./USA vs EMEA Data by Year.png
------------------------------
plotting agent output {'input': 'make a line chart for me', 'invoked_chain': 'line_chart', 'chat_history': [], 'bot_message': 'line chart is generated, the image path can be found here : ./USA vs EMEA Data by Year.png', 'img_path': './USA vs EMEA Data by Year.png'}
2024-11-19 17:13:35,244 - aiq.cli.run - INFO - --------------------------------------------------
Workflow Result:
['Saved output to ./USA vs EMEA Data by Year.png']
--------------------------------------------------
2024-11-19 17:13:35,244 - aiq.cli.entrypoint - INFO - Total time: 114.49 sec
2024-11-19 17:13:35,244 - aiq.cli.entrypoint - INFO - Pipeline runtime: 114.41 sec
```

Note: in this run, the image is saved to **./USA vs EMEA Data by Year.png** in the root folder of the AIQ Toolkit repository. Depending on the input, your run might have a different image name, please check the **`bot_message`** output to find the image.



Note: this is a multi-agents system, you can also try out some other examples listed below :
```bash
aiq run --config_file examples/plot_charts/configs/config.yml  --input "no I change my mind, make a bar chart instead"
```
```bash
aiq run --config_file examples/plot_charts/configs/config.yml  --input "tell me a joke"
```


### Launch the Workflow Server

Run the following command from the root of the AIQ Toolkit repo to serve this workflow:

```bash
aiq serve --config_file examples/plot_charts/configs/config.yml
```

**Expected Output**

```console
INFO:     Started server process [162278]
INFO:     Waiting for application startup.
Starting up
/home/coder/dev/ai-query-engine/examples/plot_charts/src/plot_charts/create_plot.py:10: LangChainDeprecationWarning: As of langchain-core 0.3.0, LangChain uses pydantic v2 internally. The langchain_core.pydantic_v1 module was a compatibility shim for pydantic v1, and should no longer be used. Please update the code to import from Pydantic directly.

For example, replace imports like: `from langchain_core.pydantic_v1 import BaseModel`
with: `from pydantic import BaseModel`
or the v1 compatibility namespace if you are working in a code base that has not been fully upgraded to pydantic 2 yet.         from pydantic.v1 import BaseModel

  from .plot_chain_agent import PlotChartAgents
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

**Triggering the Workflow Server**

The workflow server can be triggered using the following curl command from another terminal:

```bash
curl --request POST   --url http://localhost:8000/generate   --header 'Content-Type: application/json'   --data '{"input_message": "make a trend chart for me"}'
```

**Expected Output**
```json
{"value":"Saved output to ./USA vs EMEA Performance Over Time.png"}
```

Find the image in the root folder of the AIQ Toolkit repository with the image name displayed above
