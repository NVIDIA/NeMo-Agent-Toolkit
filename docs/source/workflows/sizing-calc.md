<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Sizing your GPU Cluster with AIQ Toolkit
AIQ toolkit provides a sizing calculator to estimate the GPU cluster size required for a workflow. The sizing calculator uses the evaluation system in the AIQ toolkit. Refer to the [Evaluate](./evaluate.md) documentation for more details on the evaluation system.

## Quick Start
### Gather Metrics
```
aiq profile calc --config examples/simple_calculator/src/aiq_simple_calculator/configs/config-sizing-calc.yml --output_dir .tmp/aiq/examples/simple_calculator/calc/ --concurrencies 1,2,4,8,16,32 --num_passes 2
```
### Estimate GPU Cluster Size
```
aiq profile calc --offline_mode --output_dir .tmp/aiq/examples/simple_calculator/calc/ --test_gpu_count 8 --target_workflow_runtime 10 --target_users 100
```
You can optionally combine the two steps and get the GPU estimate in `online_mode` by providing the target and test parameters in the previous command.


## Detailed Guide
### Gather Metrics
To use the calculator you can first gather metrics from the workflow and separately size the cluster in `offline_mode` using the previously gathered metrics.
Sample command:
```
aiq profile calc --config examples/simple_calculator/src/aiq_simple_calculator/configs/config-sizing-calc.yml --output_dir .tmp/aiq/examples/simple_calculator/calc/ --concurrencies 1,2,4,8,16,32 --num_passes 2
```

#### Dataset
To use the calculator, you need a representative dataset of inputs. The size of the dataset can be as small as one input. However if your workflow takes widely different trajectories for different inputs it is recommended to have datapoints for each trajectory.

The dataset is provided in the evals section of the workflow configuration file.
`examples/simple_calculator/src/aiq_simple_calculator/configs/config-sizing-calc.yml`:
```yaml
eval:
  general:
    output_dir: .tmp/aiq/examples/simple_calculator/eval
    dataset:
      _type: json
      file_path: examples/simple_calculator/data/simple_calculator.json
```
Other parameters in the eval section are not used by the calculator. For further details refer to the [Evaluate](./evaluate.md) documentation.

The dataset used by the calculator does not need to include ground truth answers. Only the inputs are needed.
For example, the following dataset is valid:
```json
[
    {
        "id": 1,
        "question": "What is the product of 3 and 7, and is it greater than the current hour?",
    },
    {
        "id": 2,
        "question": "What is the product of 4 and 5, and is it greater than the current hour?",
    }
]
```

#### Specifying the Concurrency Range
A slope based mechanism is used to estimate the GPU count required for the workflow. To create an accurate linear fit it is recommended to use a wide range of concurrency values. A minimum of 10 concurrency values is recommended. The concurrency range is specified as a comma separated list in the `concurrencies` command line parameter.

In addition to the concurrency range you can specify the number of passes made with each concurrency. By default the number of passes is 1 or a multiple of the concurrency if the dataset is larger than the concurrency value.

If the size of the dataset is smaller than the concurrency range specified, the dataset is repeated to match the concurrency range.

#### Sample Output
The per-concurrency metrics are stored in the `output_dir` specified in the command line. It is recommended to use a separate output directory for the calculator than the one used for the evaluation. This is to avoid accidental deletion of the metrics when the evaluation jobs are cleaned up.

By default the metrics of the latest run overwrite the previous runs. You can use the `--append_jobs` command line parameter to store each run in a separate subdirectory.

The results of each run are available as a summary table, analysis plots and a JSON file.

##### Sample Summary output
![Summary output](../_static/sizing_calc_online.png).

##### Sample Plots
![Analysis plot output](../_static/concurrency_vs_p95_analysis.png).

##### JSON output for Additional Analysis
The JSON file contains the per-concurrency metrics that can be used for additional analysis.
Sample output:
`calc_runner_output.json`:
```json
{
  "gpu_estimates": {
    "gpu_estimate_by_wf_runtime": 76.61472307484419,
    "gpu_estimate_by_llm_latency": null
  },
  "per_concurrency_data": {
    "1": {
      "gpu_estimates": {
        "gpu_estimate_by_wf_runtime": 309.15830421447754,
        "gpu_estimate_by_llm_latency": null
      },
      "out_of_range_runs": {
        "num_items_greater_than_target_latency": 0,
        "num_items_greater_than_target_runtime": 0,
        "workflow_interrupted": false
      },
      >>>>>> SNIPPED <<<<<
    }
  }
}
```
Output has been truncated for brevity.

#### Using a Remote Workflow
By default the calculator runs the workflow locally to gather metrics. You can use the `--remote_endpoint` and `--remote_endpoint_timeout` command line parameters to use a remote workflow for gathering metrics.

Start the Remote Workflow:
```
aiq start fastapi --config_file=examples/simple_calculator/src/aiq_simple_calculator/configs/config-sizing-calc.yml
```

Run the Calculator using the remote endpoint:
```
aiq profile calc --config_file examples/simple_calculator/src/aiq_simple_calculator/configs/config-sizing-calc.yml --output_dir .tmp/aiq/examples/simple_calculator/calc/ --concurrencies 1,2,4,8,16,32 --num_passes 2 --endpoint http://localhost:8000
```

#### Failed Workflows
Based on the test setup you may encounter failures as the concurrency value increases. When a workflow fails for an input the pass is stopped for that particular concurrency value. The pass is tagged with a `workflow_interrupted` flag in the JSON output. Such concurrency fails are not included in the GPU estimate. This is information is indicated in the summary table via an `Alerts` column.

- TODO: Provide a sample output with alerts.

### Estimate GPU Cluster Size


#### Target and Test Parameters:
Target parameters:
Test parameters:

#### Slope-based Estimation
WIP: Talk about slope-based estimation.


## Using the Sizing Calculator Programatically
TODO: Add details
