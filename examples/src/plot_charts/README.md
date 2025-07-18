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

# Plot Charts Agent

A simple and reusable example that demonstrates creating charts from data using the AIQ Toolkit. This workflow can generate line charts, bar charts, and scatter plots from JSON data files based on user requests. The implementation follows AIQ Toolkit best practices for configuration-driven, reusable workflows.

## Table of Contents

* [Key Features](#key-features)
* [Installation and Usage](#installation-and-setup)
* [Configuration](#configuration)
* [Example Usage](#example-usage)

## Key Features

- **Simple Configuration:** All parameters configurable through YAML files
- **Multiple Chart Types:** Supports line charts, bar charts, and scatter plots
- **Flexible Data Sources:** Configurable data file paths for different datasets
- **LLM Integration:** Uses configured LLM to generate intelligent chart descriptions
- **Reusable Design:** Follows standard AIQ Toolkit patterns for easy customization
- **Output Control:** Configurable output directory and chart sizing

## Installation and Setup

### Setup Virtual Environment and Install AIQ Toolkit

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install AIQ toolkit.

### Install this Workflow:

From the root directory of the AIQ toolkit library, run the following commands:

```bash
uv pip install -e examples/src/plot_charts
```

### Set Up API Keys

If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

## Configuration

The workflow is fully configurable through the `config.yml` file. Here are the available configuration options:

### Data Configuration
- **`data_file_path`**: Path to the JSON data file (default: `"example_data.json"`)
- **`output_directory`**: Directory where charts will be saved (default: `"outputs"`)

### Chart Configuration
- **`chart_types`**: List of supported chart types (default: `["line", "bar", "scatter"]`)
- **`max_data_points`**: Maximum number of data points to prevent excessive processing (default: `100`)
- **`figure_size`**: Chart dimensions as [width, height] (default: `[10, 6]`)

### Example Configuration

```yaml
workflow:
  _type: plot_charts
  llm_name: nim_llm
  data_file_path: "my_custom_data.json"
  output_directory: "my_charts"
  chart_types: ["line", "bar"]
  max_data_points: 50
  figure_size: [12, 8]
```

### Data Format

The data file should be in JSON format with the following structure:

```json
{
  "xValues": ["2020", "2021", "2022", "2023", "2024"],
  "yValues": [
    {
      "data": [2, 5, 2.2, 7.5, 3],
      "label": "USA"
    },
    {
      "data": [2, 5.5, 2, 8.5, 1.5],
      "label": "EMEA"
    }
  ]
}
```

## Example Usage

### Run the Workflow

Run the following command from the root of the AIQ toolkit repo to execute this workflow:

```bash
aiq run --config_file examples/src/plot_charts/src/aiq_plot_charts/configs/config.yml --input "create a line chart"
```

**Expected Output**

```console
Successfully created line chart saved to: outputs/line_chart_1703123456.png

Chart description: The line chart shows performance trends for USA and EMEA regions from 2020 to 2024, with both regions showing growth patterns over the time period.
```

### Different Chart Types

You can request different chart types:

```bash
# Bar chart
aiq run --config_file examples/src/plot_charts/src/aiq_plot_charts/configs/config.yml --input "create a bar chart comparing the data"

# Scatter plot
aiq run --config_file examples/src/plot_charts/src/aiq_plot_charts/configs/config.yml --input "show me a scatter plot"
```

### Launch the Workflow Server

Run the following command from the root of the AIQ toolkit repo to serve this workflow:

```bash
aiq serve --config_file examples/src/plot_charts/src/aiq_plot_charts/configs/config.yml
```

**Triggering the Workflow Server**

The workflow server can be triggered using the following curl command:

```bash
curl --request POST \
  --url http://localhost:8000/generate \
  --header 'Content-Type: application/json' \
  --data '{"input_message": "create a line chart showing trends over time"}'
```

**Expected Output**
```json
{
  "value": "Successfully created line chart saved to: outputs/line_chart_1703123456.png\n\nChart description: The line chart displays comparative performance data for USA and EMEA regions across a five-year period."
}
```

## Customization Examples

### Using Different Data Sources

1. Create your own data file following the JSON format above
2. Update the configuration:

```yaml
workflow:
  _type: plot_charts
  llm_name: nim_llm
  data_file_path: "path/to/your/data.json"
```

### Customizing Chart Types

To support only specific chart types:

```yaml
workflow:
  _type: plot_charts
  llm_name: nim_llm
  chart_types: ["bar"]  # Only bar charts
```

### Changing Output Location

To save charts to a specific directory:

```yaml
workflow:
  _type: plot_charts
  llm_name: nim_llm
  output_directory: "/path/to/your/charts"
```
