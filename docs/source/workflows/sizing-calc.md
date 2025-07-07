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
AIQ toolkit provides a sizing calculator to estimate the GPU cluster size required to accommodate a target number of users with a target response time. The estimation is based on the performance of the workflow at different concurrency levels.

The sizing calculator uses the evaluation and profiling systems in the AIQ toolkit. Refer to the [Evaluate](./evaluate.md) documentation for more details on the evaluation system.

## Quick Start

### Step 1: Gather Metrics
Collect performance data at different concurrency levels:
```
aiq profile calc --config_file examples/simple_calculator/src/aiq_simple_calculator/configs/config-sizing-calc.yml --output_dir .tmp/aiq/examples/simple_calculator/calc/ --concurrencies 1,2,4,8,16,32 --num_passes 2
```
### Step 2: Estimate GPU Cluster Size
Use the previously collected metrics to estimate the GPU cluster size:
```
aiq profile calc --offline_mode --output_dir .tmp/aiq/examples/simple_calculator/calc/ --test_gpu_count 8 --target_workflow_runtime 10 --target_users 100
```

You can optionally combine both steps by adding the target and test parameters to the first command. For example:
```
aiq profile calc --config_file examples/simple_calculator/src/aiq_simple_calculator/configs/config-sizing-calc.yml --output_dir .tmp/aiq/examples/simple_calculator/calc/ --concurrencies 1,2,4,8,16,32 --num_passes 2 --test_gpu_count 8 --target_workflow_runtime 10 --target_users 100
```
This will run the workflow at the specified concurrency levels and estimate the GPU cluster size.

# Detailed Guide
## Gather Metrics
To use the calculator you can first gather metrics from the workflow and then separately size the cluster in `offline_mode` using the previously gathered metrics.

**Sample Command for Gathering Metrics**

```
aiq profile calc --config examples/simple_calculator/src/aiq_simple_calculator/configs/config-sizing-calc.yml --output_dir .tmp/aiq/examples/simple_calculator/calc/ --concurrencies 1,2,4,8,16,32 --num_passes 2
```

### Dataset Requirements
To use the calculator, you need a representative dataset of inputs. The size of the dataset can be as small as one input. However, if your workflow's behavior varies significantly depending on the input, it is recommended to include representative datapoints for each trajectory.

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
In addition to the dataset you need to specify the `eval.general.output_dir` parameter for storing the evaluation results. Other parameters in the eval section are not used by the calculator. For further details refer to the [Evaluate](./evaluate.md) documentation.

The dataset used by the sizing calculator does not need to include ground truth answers. Only the inputs are needed.
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

### Specifying the Concurrency Range
A slope based mechanism is used to estimate the GPU count required for the workflow. To create a robust linear fit it is recommended to use a wide range of concurrency values. A minimum of ten concurrency values is recommended, though the calculator can work with fewer values (accuracy may decrease). The concurrency range is specified as a comma separated list with the `--concurrencies` command line parameter.

In addition to the concurrency range you can specify the number of passes made with each concurrency with the `--num_passes` command line parameter. By default the number of passes is one or a multiple of the concurrency if the dataset is larger than the concurrency value.

If the size of the dataset is smaller than the concurrency range specified, the dataset is repeated to match the concurrency range.

### Sample Output
The per-concurrency metrics are stored in the `output_dir` specified in the command line. It is recommended to use a separate output directory for the calculator than the one used for the evaluation (specified via `eval.general.output_dir` in the workflow configuration file). This is to avoid accidental deletion of the calculator metrics when the evaluation jobs are cleaned up.

By default the metrics of the latest calculator run overwrite the previous runs. You can use the `--append_jobs` command line parameter to store each run in a separate subdirectory.

The results of each run are available as:
- a summary table,
- analysis plots and
- a JSON file.

**Summary Table**

The summary table provides an overview of the per-concurrency metrics.
- The P95 LLM latency is computed across all LLM invocations. If multiple models are used the value will trend towards the latency of the model with the highest latency.
- The P95 workflow runtime is the response time of the workflow and is computed across all runs at the specified concurrency.
- The total runtime is the total time taken to process the entire dataset at a specified concurrency level.

| Concurrency | p95 LLM Latency | p95 WF Runtime | Total Runtime |
|-------------|-----------------|----------------|---------------|
| 1           | 1.10438         | 3.86448        | 7.71942       |
| 2           | 1.29807         | 4.67094        | 9.03173       |
| 4           | 4.71100         | 8.73793        | 15.2026       |
| 8           | 1.99912         | 8.0153         | 16.2514       |
| 16          | 3.95316         | 14.394         | 27.5015       |
| 32          | 6.42463         | 23.3671        | 45.4901       |

**Analysis Plots**

The plots provide a visual representation of the concurrency vs. latency and concurrency vs. runtime. The trend line is a linear fit of the concurrency vs. time metrics. The slope of the trend line is used to estimate the GPU count required for the workflow. The R² value of the trend line indicates the accuracy of the linear fit and has a value between 0 and 1. A R² value close to 1 indicates a good fit.
![Analysis plot output](../_static/concurrency_vs_p95_analysis.png).


**JSON Output**

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

### Using a Remote Workflow
By default the calculator runs the workflow locally to gather metrics. You can use the `--endpoint` and `--endpoint_timeout` command line parameters to use a remote workflow for gathering metrics.

Start the Remote Workflow:
```
aiq start fastapi --config_file=examples/simple_calculator/src/aiq_simple_calculator/configs/config.yml
```

Run the Calculator using the remote endpoint:
```
aiq profile calc --config_file examples/simple_calculator/src/aiq_simple_calculator/configs/config-sizing-calc.yml --output_dir .tmp/aiq/examples/simple_calculator/calc/ --concurrencies 1,2,4,8,16,32 --num_passes 2 --endpoint http://localhost:8000
```
The configuration file used for running the calculator only needs to specify the `eval` section. The `workflow` section is not used by the calculator when running with a remote endpoint.

### Handling Failed Workflows
Based on the test setup you may encounter failures as the concurrency value increases. When a workflow fails for an input the pass is stopped for that particular concurrency value. The pass is tagged with a `workflow_interrupted` flag in the JSON output. Such concurrencies, with a `workflow_interrupted` flag set to `true`, are not included in the GPU estimate. This information is indicated in the summary table via an `Alerts` column.

**Sample output with alerts:**
```
Per concurrency results:
Alerts: !W = Workflow interrupted
| Alerts | Concurrency | p95 LLM Latency | p95 WF Runtime | Total Runtime |
|--------|-------------|-----------------|----------------|---------------|
|        | 1           | 1.10438         | 3.86448        | 7.71942       |
|        | 2           | 1.29807         | 4.67094        | 9.03173       |
| !W     | 4           | 4.71100         | 8.73793        | 15.2026       |
|        | 8           | 1.99912         | 8.0153         | 16.2514       |
|        | 16          | 3.95316         | 14.394         | 27.5015       |
|        | 32          | 6.42463         | 23.3671        | 45.4901       |
```

In this example, the workflow failed at concurrency level 4 (indicated by `!W` in the Alerts column). The time metrics for concurrency 4 are not included in the GPU estimate as they are not reliable and may skew the linear fit used to estimate the GPU count.

## Estimate GPU Cluster Size
Once the metrics are gathered, you can estimate the GPU cluster size using the `aiq profile calc` command in `offline_mode`.
Sample command:
```
aiq profile calc --offline_mode --output_dir .tmp/aiq/examples/simple_calculator/calc/ --test_gpu_count 8 --target_workflow_runtime 10 --target_users 100
```

### Target and Test Parameters
** Target Parameters **
To estimate the GPU cluster size, you need to specify the target number of users and the target workflow runtime i.e. the maximum acceptable response time for the workflow.

Optionally you can specify the target p95 LLM latency if the LLM latency is a defining factor for the workflow and if it is possible to measure the maximum acceptable LLM latency.
- `target_users`: Target number of users to support.
- `target_workflow_runtime`: Target p95 workflow runtime (seconds). Can be set to 0 to ignore.
- `target_llm_latency`: Target p95 LLM latency (seconds). Can be set to 0 to ignore.

** Test Parameters **
You need to specify the number of GPUs used for running the workflow via the `--test_gpu_count` command line parameter. This is the number of GPUs used during the profiling run, not the target cluster size. This information is used to extrapolate the GPU count required for the target users.

### Slope-based Estimation

The sizing calculator uses a **slope-based estimation** approach to determine how your workflow’s performance scales with increasing concurrency. This method helps estimate the number of GPUs required to meet your target user load and response time.

**How it works:**

1. **Linear Fit of Concurrency vs. Time Metrics**
   - The calculator runs your workflow at several different concurrency levels.
   - For each level, it measures key metrics such as p95 LLM latency and p95 workflow runtime.
   - It then fits a straight line (using least squares regression) to the data points, modeling how time metrics change as concurrency increases.

2. **Slope and Intercept**
   - The **slope** of the fitted line represents how much the time metric (latency or runtime) increases for each additional concurrent user. A slope of 1.0 means that the time metric increases perfectly linearly with the concurrency. A slope greater than 1.0 means that the time metric increases faster than linearly with the concurrency and optimization should be done to reduce the slope.
   - The **intercept** represents the baseline time metric when concurrency is zero (theoretical minimum). Note that this is a mathematical extrapolation and may not correspond to actual measurements at concurrency=0. It is indicative of the overhead of the workflow.

3. **R² Value**
   - The calculator computes the R² (coefficient of determination) to indicate how well the linear model fits your data. An R² value close to 1.0 means a good fit.

4. **Outlier Removal**
   - Outliers (data points that deviate significantly from the trend) are automatically detected and removed to ensure a robust fit. The R² value is used to determine which points are considered outliers and should be excluded from the fit.

5. **Estimating Required Concurrency**
   - Using your target time metric (e.g., target workflow runtime), the calculator determines the maximum concurrency that can be supported for the `test_gpu_count` while still meeting the target. This is the `calculated_concurrency` in the formula below.

6. **GPU Count Formula**
   - The required GPU count is estimated using the formula:
     ```
     calculated_concurrency = (target_time_metric - intercept) / slope
     gpu_estimate = (target_users / calculated_concurrency) * test_gpu_count
     ```
   - This formula scales your test results to your target user load, based on the observed scaling behavior.

**Example:**

Suppose your target workflow runtime is 10 seconds, the linear fit gives a slope of 0.6, and an intercept of 3.5. The calculator will:
- Compute the concurrency that achieves a 10s runtime:
  `(10 - 3.5) / 0.6 ≈ 10.83`
- If you tested with 8 GPUs and want to support 100 users:
  `(100 / 10.83) * 8 ≈ 73.9 GPUs`

**Key Points:**
- The more concurrency levels you test, the more accurate the estimation.
- Outliers and failed runs are excluded from the fit.
- The calculator provides both runtime-based and latency-based GPU estimates (if both targets are specified).

#### Interpreting the Results
The sizing calculator provides two GPU count estimates:
- `Estimated GPU count (Workflow Runtime)`: Estimated GPU count based on the target workflow runtime.
- `Estimated GPU count (LLM Latency)`: Estimated GPU count based on the target LLM latency.

The calculator will provide both estimates if both target parameters are specified. You can use a maximum of the two estimates as the final GPU count to accommodate the target users.

**Sample output:**
```
Targets: LLM Latency ≤ 0.0s, Workflow Runtime ≤ 10.0s, Users = 100
Test parameters: GPUs = 8
Per concurrency results:
| Concurrency | p95 LLM Latency | p95 WF Runtime | Total Runtime | Runtime OOR | GPUs (WF Runtime, Rough) |
|-------------|-----------------|----------------|---------------|-------------|--------------------------|
| 1           | 1.10438         | 3.86448        | 7.71942       | 0           | 309.158                  |
| 2           | 1.29807         | 4.67094        | 9.03173       | 0           | 186.837                  |
| 4           | 4.71100         | 8.73793        | 15.2026       | 0           | 174.759                  |
| 8           | 1.99912         | 8.0153         | 16.2514       | 0           | 80.153                   |
| 16          | 3.95316         | 14.394         | 27.5015       | 32          |                          |
| 32          | 6.42463         | 23.3671        | 45.4901       | 64          |                          |

=== GPU ESTIMATES ===
Estimated GPU count (Workflow Runtime): 76.6
```

**Note:** The GPU estimate is for the total cluster, not per GPU.
In addition to the slope based estimation, the calculator also provides a rough estimate of the GPU count required for the target user based on the data from each concurrency level. This is useful to get a quick estimate of the GPU count required for the workflow but is not as accurate as the slope based estimation and is not recommended for production use.

## Programmatic Usage
In addition to the command line interface, the sizing calculator can be used programmatically.

**Sample code:**
```python
import asyncio
from aiq.profiler.calc.data_models import CalcRunnerConfig
from aiq.cli.commands.profile.calc import CalcRunner
from aiq.profiler.calc.data_models import CalcRunnerOutput

async def run_calc():
    runner_config = CalcRunnerConfig(
        config_file="examples/simple_calculator/src/aiq_simple_calculator/configs/config-sizing-calc.yml",
        output_dir=".tmp/aiq/examples/simple_calculator/calc/",
        concurrencies=[1, 2, 4, 8, 16, 32],
        num_passes=2,
        test_gpu_count=8,
        target_workflow_runtime=10,
        target_users=100,
    )
    runner = CalcRunner(runner_config)
    result: CalcRunnerOutput = await runner.run()
    # Access GPU estimates and per-concurrency metrics from result
    print(result.gpu_estimates)
    print(result.per_concurrency_data)

# Run the async calc function
asyncio.run(run_calc())
```

`CalcRunnerOutput` is a Pydantic model that contains the per-concurrency metrics and the GPU count estimates.
See the [CalcRunnerOutput](../../../src/aiq/profiler/calc/data_models.py) for more details.

# Summary
This guide provides a step-by-step process to estimate the GPU cluster size required to accommodate a target number of users with a target response time. The estimation is based on the performance of the workflow at different concurrency levels.
