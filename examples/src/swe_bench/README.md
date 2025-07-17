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

# Solving problems in a SWE bench dataset using AIQ Toolkit
This example provides a skeleton workflow which can be used to implement predictors to solve problems in a SWE bench dataset.

# Pre-requisites
SWE bench evaluations run inside a Docker container. Ensure that Docker is installed and the Docker service is running before proceeding.

## Installation & Running Docker
- Install AIQ toolkit: If you have not already done so, follow the instructions in the [Install Guide](../../../../docs/source/quick-start/installing.md#install-from-source)
- Install Docker: Follow the official installation guide for your platform: [Docker Installation Guide](https://docs.docker.com/engine/install/)
- Start Docker Service:
  - Linux: Run`sudo systemctl start docker` (ensure your user has permission to run Docker).
  - Mac & Windows: Docker Desktop should be running in the background.

## Verify Docker Installation
Run the following command to verify that Docker is installed and running correctly:
```bash
docker info
```

# Quickstart
1. Install the `swe_bench` example:
```bash
uv pip install -e examples/intermediate/evaluation_and_profiling/swe_bench
```
2. Run the example via the `aiq eval` CLI command:
```bash
aiq eval --config_file examples/intermediate/evaluation_and_profiling/swe_bench/configs/config_gold.yml
```

## Datasets
This workflow requires the `swe_bench` dataset as a JSON or Parquet file. A few public datasets are provided in the data directory -
- data/dev_dataset_lite.json, downloaded from [SWE-bench_Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/viewer/default/dev)
- data/test_dataset_lite.json, downloaded from [SWE-bench_Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/viewer/default/test)
- data/test_dataset_verified.json, downloaded from [SWE-bench_Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)

And can be used to test the workflow by specifying the dataset in the configuration file:
```yaml
eval:
  general:
    datasets:
      test_verified:
        _type: json
        file_path: examples/intermediate/evaluation_and_profiling/swe_bench/data/test_dataset_verified.json
```

Alternately you can read any remote dataset by specifying the pandas URL in the configuration file:
```yaml
eval:
  datasets:
    test_verified:
        _type: parquet
        file_path: "hf://datasets/princeton-nlp/SWE-bench_Verified/data/test-00000-of-00001.parquet"
```


The input to the workflow is a [Pydantic](https://docs.pydantic.dev) model, `SWEBenchInput`. Refer to `src/aiq/data_models/swe_bench_model.py` for the model definition.

### Filtering dataset entries
You can limit the number of `swe_bench` instances in the dataset, that are solved and evaluated, via a filter in the configuration file. For example:
```yaml
eval:
  general:
    dataset:
      _type: json
      file_path: examples/intermediate/evaluation_and_profiling/swe_bench/data/test_dataset_lite.json
      id_key: instance_id
      structure: # For swe-bench the entire row is the input
        disable: true
      filter:
        allowlist:
          field:
            instance_id:
              - sympy__sympy-20590
              - sympy__sympy-21055
```

This configuration runs the workflow and evaluation only on the two specified instances.

You can alternately filter out instances that are not to be solved and evaluated, via `eval.swe_bench.filter.denylist_instance_ids`. For example:
```yaml
eval:
  general:
    dataset:
      _type: json
      file_path: examples/intermediate/evaluation_and_profiling/swe_bench/data/test_dataset_lite.json
      id_key: instance_id
      structure: # For swe-bench the entire row is the input
        disable: true
      filter:
        denylist:
          field:
            instance_id:
              - "astropy__astropy-6938"
              - "astropy__astropy-7746"
              - "psf__requests-2317"
              - "psf__requests-2674"
```
The configuration runs the workflow and evaluation on all instances in the dataset except the `denied` ones.

## Predictors
A predictor is a class that takes in a SWE bench input instance, solves the problem in the instance, and returns a patch.

The predictor uses the `repo`, `problem_statement` and `hints_text` in the `SWEBenchInput` instance to fix the bug in the code. It then returns the fix as a code patch.

The predictor should not use -
- the patch fields, `patch` and `test_patch` (or)
- the tests, `PASS_TO_PASS` and `FAIL_TO_PASS`
in the input instance.

That information is only used for evaluation. Using it can taint the predictor and lead to overfitting.

These predictors are provided in this AIQ toolkit example:
- `gold` - Uses the patch from the `SWEBenchInput` instance, bypassing problem-solving logic. See [predict_gold_stub.py](src/aiq_swe_bench/predictors/predict_gold/predict_gold_stub.py) and configuration file `examples/intermediate/evaluation_and_profiling/swe_bench/configs/config_gold.yml`.
- `skeleton` - Skeleton code for creating a problem-solving workflow. This code can be copied to create a net-new predictor. See [predict_skeleton.py](src/aiq_swe_bench/predictors/predict_skeleton/predict_skeleton.py) and configuration file `examples/intermediate/evaluation_and_profiling/swe_bench/configs/config_skeleton.yml`.

### Adding a net new predictor
To add a new predictor:
- Create a new directory in the predictors directory, copy over the contents of [predictors/predict_skeleton](src/aiq_swe_bench/predictors/predict_skeleton/). Rename the files and fill in the logic to solve the problem.
- Register the new predictor class with an unique name using the `@register_predictor` decorator.
- Import the new predictor class in [predictors/register.py](src/aiq_swe_bench/predictors/register.py) to make it discoverable by the AIQ toolkit `swe_bench` harness.

## Evaluation
The `model_patch` returned by the `swe_bench` workflow is run through the `swe_bench` evaluation harness. This harness -
- Launches a docker container with the `swe_bench` test image
- Installs the repo from the `SWEBenchInput` instance
- Applies the model patch in the `SWEBenchOutput`.
- Applies any test patch in the `SWEBenchInput` instance.
- Runs the `PASS_TO_PASS` and `FAIL_TO_PASS` tests in the `SWEBenchInput` instance
- Returns the evaluation results as a JSON report file with additional logs for troubleshooting.

The evaluation results, logs and reports, are stored in the output directory specified in the configuration file via `eval.general.output_dir`.



### Sample output
Run:
```bash
aiq eval --config_file examples/intermediate/evaluation_and_profiling/swe_bench/configs/config_gold.yml
```
Logs snippet:
```
2025-01-20 12:07:45,202 - aiq.eval.evaluate - INFO - Starting swe_bench run aiq_0
Running 1 unevaluated instances...
Base image sweb.base.py.x86_64:latest already exists, skipping build.
Base images built successfully.
No environment images need to be built.
Running 1 instances...
1 ran successfully, 0 failed: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [03:21<00:00, 201.41s/it]
All instances run.
Cleaning cached images...
Removed 0 images.
Total instances: 1
Instances submitted: 1
Instances completed: 1
Instances incomplete: 0
Instances resolved: 1
Instances unresolved: 0
Instances with empty patches: 0
Instances with errors: 0
Unstopped containers: 0
Unremoved images: 0
Report written to nim_llm.aiq_0.json
2025-01-20 12:11:07,202 - aiq.eval.evaluate - INFO - Completed swe_bench run aiq_0
2025-01-20 12:11:07,206 - aiq.eval.evaluate - INFO - Evaluation results written to .tmp/aiq/examples/intermediate/evaluation_and_profiling/swe_bench/eval_output.json
2025-01-20 12:11:07,206 - aiq.eval.evaluate - INFO - SWE_bench report and logs written to .tmp/aiq/examples/intermediate/evaluation_and_profiling/swe_bench/swe_bench_reports directory
2025-01-20 12:11:07,206 - aiq.eval.evaluate - INFO - Ending evaluation run with config file: examples/intermediate/evaluation_and_profiling/swe_bench/configs/config_gold.yml
2025-01-20 12:11:07,208 - aiq.cli.entrypoint - INFO - Total time: 210.71 sec
```
