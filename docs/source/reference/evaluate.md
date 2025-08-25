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

# Evaluating NVIDIA NeMo Agent Toolkit Workflows Details

:::{warning}
**Experimental Feature**: The Evaluation API is experimental and may change in future releases. Future versions may introduce breaking changes without notice.
:::

:::{note}
We recommend reading the [Evaluating NeMo Agent Toolkit Workflows](../workflows/evaluate.md) guide before proceeding with this detailed documentation.
:::

NeMo Agent toolkit provides a set of evaluators to run and evaluate the workflows. In addition to the built-in evaluators, the toolkit provides a plugin system to add custom evaluators.

Example:
```bash
nat eval --config_file=examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config.yml
```

## Using Datasets
Run and evaluate the workflow on a specified dataset. The dataset files types are `json`, `jsonl`, `csv`, `xls`, or `parquet`.

Download and use datasets provided by NeMo Agent toolkit examples by running the following.

```bash
git lfs fetch
git lfs pull
```
 The dataset used for evaluation is specified in the configuration file through the `eval.general.dataset`. For example, to use the `langsmith.json` dataset, the configuration is as follows:
```yaml
eval:
  general:
    dataset:
      _type: json
      file_path: examples/evaluation_and_profiling/simple_web_query_eval/data/langsmith.json
```

### Dataset Format
Each dataset file contains a list of records. Each record is a dictionary with keys as the column names and values as the data. For example, a sample record in a `json` dataset file is as follows:
```json
{
"id": "q_1",
"question": "What is langsmith",
"answer": "LangSmith is a platform for LLM application development, monitoring, and testing"
},
```

A dataset entry are either structured or unstructured.
For structured entries, the default names of the columns are `id`, `question`, and `answer`,
where the libraries know that `question` is an input and `answer` is the output. You can
change the column names and their configurations in the `config.yml` file
with `eval.general.dataset.structure`.
```yaml
eval:
  general:
    dataset:
      structure:
        id: "my_id"
        question: "my_question"
        answer: "my_answer"
```

For unstructured entries, the entire dictionary is the input to the workflow
but the libraries don't know the individual columns. The input and the workflow output
goes through evaluation, where evaluators, such as swe-bench evaluator, handle
unstructured entries. The following is an example configuration for
the swe-bench evaluator:
```yaml
eval:
  general:
    dataset:
      _type: json
      file_path: examples/evaluation_and_profiling/swe_bench/data/test_dataset_lite.json
      id_key: instance_id
      structure: # For swe-bench the entire row is the input
        disable: true
```

### Accessing Additional Dataset Fields in Evaluators
In some evaluation scenarios, you may have additional fields in your dataset that are not consumed by the workflow but are required by the evaluator. These fields are automatically available during evaluation via the `full_dataset_entry` field in the `EvalInputItem` object. The entire dataset entry is passed as a dictionary to the evaluator, making all dataset fields available for custom evaluators that require access to fields like `labels` or `metadata` which are not part of the workflow's inputs but are relevant for scoring or analysis.

### Filtering Datasets
While evaluating large datasets, you can filter the dataset to a
smaller subset by allowing or denying entries with the `eval.general.dataset.filter`
in the `config.yml` file. The filter is a dictionary with keys as the column names and
values as the filter values.

The following is an example configuration, where evaluation
runs on a subset of the swe-bench-verified dataset, which has 500 entries. The configuration runs the
evaluation on two entries with instance identifications (`instance_id`), `sympy__sympy-20590`
and `sympy__sympy-21055`. The evaluation iteratively develops and debugs the workflows.
```yaml
eval:
    dataset:
      _type: json
      file_path: examples/evaluation_and_profiling/swe_bench/data/test_dataset_verified.json
      id_key: instance_id
      structure:
        disable: true
      filter:
        allowlist:
          field:
            instance_id:
              - sympy__sympy-20590
              - sympy__sympy-21055
```
The swe-bench verified dataset has 500 entries but above configuration runs the workflow and evaluation on only two entries with `instance_id` `sympy__sympy-20590` and `sympy__sympy-21055`. This is useful for iterative development and troubleshooting of the workflow.

You can also skip entries from the dataset. Here is an example configuration to skip entries with `instance_id` `sympy__sympy-20590` and `sympy__sympy-21055`:
```yaml
eval:
    dataset:
      _type: json
      file_path: examples/evaluation_and_profiling/swe_bench/data/test_dataset_verified.json
      id_key: instance_id
      structure:
        disable: true
      filter:
        denylist:
          field:
            instance_id:
              - sympy__sympy-20590
              - sympy__sympy-21055
```

### Custom Dataset Format
You can use a dataset with a custom format by providing a custom dataset parser function.

**Example:**
`examples/evaluation_and_profiling/simple_calculator_eval/configs/config-custom-dataset-format.yml`:
```yaml
eval:
  general:
    dataset:
      _type: custom
      file_path: examples/evaluation_and_profiling/simple_calculator_eval/data/simple_calculator_nested.json
      function: nat_simple_calculator_eval.scripts.custom_dataset_parser.extract_nested_questions
      kwargs:
        difficulty: "medium"
        max_rows: 5
```
This example configuration uses a custom dataset parser function to:
- extract the nested questions from the example dataset
- filter them by difficulty
- return only the first five questions

The example dataset `simple_calculator_nested.json` is a nested JSON file with questions and answers. The custom dataset parser function is a Python function that takes the dataset `file_path`, optional `kwargs` and returns an `EvalInput` object. Signature of the sample custom dataset parser function is as follows:
```python
def extract_nested_questions(file_path: Path, difficulty: str = None, max_rows: int = None) -> EvalInput:
```

{py:class}`~nat.eval.evaluator.evaluator_model.EvalInput` is a Pydantic model that contains a list of `EvalInputItem` objects.
{py:class}`~nat.eval.evaluator.evaluator_model.EvalInputItem` is a Pydantic model that contains the fields for an item in the dataset.
The custom dataset parser function should fill the following fields in the `EvalInputItem` object:
- `id`: The id of the item. Every item in the dataset must have a unique id of type `str` or `int`.
- `input_obj`: This is the question.
- `expected_output_obj`: This is the ground truth answer.
- `full_dataset_entry`: This is the entire dataset entry and is passed as is to the evaluator.

To run the evaluation using the custom dataset parser, run the following command:
```bash
nat eval --config_file=examples/evaluation_and_profiling/simple_calculator_eval/configs/config-custom-dataset-format.yml
```
### Custom Pre-evaluation Process Function
You can provide a custom function to process the eval input after the workflow runs but before evaluation begins. This allows you to modify, filter, or enrich the evaluation data.

**Example:**
`examples/evaluation_and_profiling/simple_calculator_eval/configs/config-with-custom-post-process.yml`:
```yaml
eval:
  general:
    output:
      dir: .tmp/nat/examples/simple_calculator/eval-with-post-process
      custom_pre_eval_process_function: nat_simple_calculator_eval.scripts.custom_post_process.normalize_calculator_outputs
    dataset:
      _type: json
      file_path: examples/getting_started/simple_calculator/src/nat_simple_calculator/data/simple_calculator.json
```
This example configuration uses a custom pre-evaluation process function to normalize numerical outputs for consistent evaluation.

The custom pre-evaluation process function is a Python function that takes an `EvalInputItem` object and returns a modified `EvalInputItem` object.
**Helper Function**: You can use the `copy_with_updates()` method in the `EvalInputItem` object to easily update only specific fields while preserving all others:
```python
# Update only the output_obj field
return item.copy_with_updates(output_obj="new output")

# Update multiple fields
return item.copy_with_updates(
    output_obj="new output",
    expected_output_obj="new expected"
)
```

Signature of the sample custom pre-evaluation process function is as follows:
```python
def normalize_calculator_outputs(item: EvalInputItem) -> EvalInputItem:
```

Common use cases for custom pre-evaluation process functions include:
- **Data normalization**: Standardize formats for consistent evaluation
- **Quality filtering**: Remove incomplete or invalid workflow outputs
- **Metadata enhancement**: Add processing information to dataset entries
- **Output transformation**: Modify generated answers before evaluation

To run the evaluation using the custom pre-evaluation process function, run the following command:
```bash
nat eval --config_file=examples/evaluation_and_profiling/simple_calculator_eval/configs/config-with-custom-post-process.yml
```


## NeMo Agent Toolkit Built-in Evaluators
NeMo Agent toolkit provides the following built-in evaluator:
- `ragas` - An evaluator to run and evaluate RAG-like workflows using the public RAGAS API.
- `trajectory` - An evaluator to run and evaluate the LangChain agent trajectory.
- `swe_bench` - An evaluator to run and evaluate the workflow on the SWE-Bench dataset.

### RAGAS Evaluator
[RAGAS](https://docs.ragas.io/) is an OSS evaluation framework that enables end-to-end
evaluation of RAG workflows. NeMo Agent toolkit provides an interface to RAGAS to evaluate the performance of RAG-like workflows.

RAGAS provides a set of evaluation metrics to configure in the `config.yml` file
by adding an evaluator section with type`ragas`.

**Example:**
```yaml
eval:
  evaluators:
    rag_accuracy:
      _type: ragas
      metric: AnswerAccuracy
      llm_name: nim_rag_eval_llm
    rag_groundedness:
      _type: ragas
      metric: ResponseGroundedness
      llm_name: nim_rag_eval_llm
    rag_relevance:
      _type: ragas
      metric: ContextRelevance
      llm_name: nim_rag_eval_llm
    rag_factual_correctness:
      _type: ragas
      metric:
        FactualCorrectness:
          kwargs:
            mode: precision
      llm_name: nim_rag_eval_large_llm # requires more tokens
```

In the example four `ragas` evaluators are configured to evaluate various `ragas`metrics. The metric can be a string or a dictionary. If the metric is a dictionary, the `kwargs` provided are passed to the metric function.

The following `ragas` metrics are recommended for RAG like workflows -
`AnswerAccuracy`: Evaluates the accuracy of the answer generated by the workflow against the expected answer or ground truth.
`ContextRelevance`: Evaluates the relevance of the context retrieved by the workflow against the question.
`ResponseGroundedness`: Evaluates the `groundedness` of the response generated by the workflow based on the context retrieved by the workflow.

### Agent Trajectory Evaluator
The `trajectory` evaluator uses LangChain agent trajectory evaluation to evaluate the workflow. To use the `trajectory` evaluator, add the following configuration to the `config.yml` file.
```yaml
eval:
  evaluators:
    trajectory:
      _type: trajectory
      llm_name: nim_trajectory_eval_llm
```

### Swe-benchmark Evaluator
Workflows can use the swe-bench evaluator to solve swe-bench problems. To evaluate the patch, generated by the workflow, install the repository and run the `PASS_TO_PASS` and `FAIL_TO_PASS` tests.

**Example:**
```yaml
eval:
  general:
    dataset:
      _type: json
      file_path: examples/evaluation_and_profiling/swe_bench/data/test_dataset_lite.json
      id_key: instance_id
      structure: # For swe-bench the entire row is the input
        disable: true

  evaluators:
    swe_bench:
      _type: swe_bench
      run_id: nat_1
```
The swe-bench evaluator uses unstructured dataset entries. The entire row is provided as input to the workflow.

### Tunable RAG Evaluator
The tunable RAG evaluator is a customizable LLM evaluator that allows for flexible evaluation of RAG workflows.
It includes a default scoring mechanism based on an expected answer description rather than a ground truth answer.

The judge LLM prompt is tunable and can be provided in the `config.yml` file.

A default scoring method is provided as follows:
- Coverage: Evaluates if the answer covers all mandatory elements of the expected answer.
- Correctness: Evaluates if the answer is correct compared to the expected answer.
- Relevance: Evaluates if the answer is relevant to the question.

These weights can be optionally tuned by setting the `default_score_weights` parameter in the `config.yml` file. If not set, each score will be equally weighted.

The default scoring can be overridden by setting the config boolean `default_scoring` to false and providing your own scoring mechanism which you describe in your custom judge LLM prompt.
Note: if you do choose to use the default scoring method, you are still able to tune the judge LLM prompt.

**Example:**
`examples/evaluation_and_profiling/simple_calculator_eval/configs/config-tunable-rag-eval.yml`:
```yaml
eval:
  evaluators:
    tuneable_eval:
      _type: tunable_rag_evaluator
      llm_name: nim_rag_eval_llm
      # (optional) retry control params for handling rate limiting
      llm_retry_control_params:
        stop_after_attempt: 3
        # set initial backoff (seconds)
        initial_backoff_delay_seconds: 1
        # Add jitter to exponential backoff
        has_exponential_jitter: true
      default_scoring: false
      default_score_weights:
        coverage: 0.5
        correctness: 0.3
        relevance: 0.2
      judge_llm_prompt: >
        You are an intelligent evaluator that scores the generated answer based on the description of the expected answer.
        The score is a measure of how well the generated answer matches the description of the expected answer based on the question.
        Take into account the question, the relevance of the answer to the question and the quality compared to the description of the expected answer.

        Rules:
        - The score must be a float of any value between 0.0 and 1.0 on a sliding scale.
        - The reasoning string must be concise and to the point. It should be 1 sentence and 2 only if extra description is needed. It must explain why the score was given and what is different between the generated answer and the expected answer.
```

Note: In your evaluation dataset, make sure that the `answer` field is a description of the expected answer with details on what is expected from the generated answer.

**Example:**
`examples/evaluation_and_profiling/simple_calculator_eval/configs/config-tunable-rag-eval.yml`:
```json
{
  "id": 1,
  "question": "What is the product of 3 and 7, and is it greater than the current hour?",
  "answer": "Answer must have the answer of product of 3 and 7 and whether it is greater than the current hour"
}
```

**Sample Usage:**
```bash
nat eval --config_file=examples/evaluation_and_profiling/simple_calculator_eval/configs/config-tunable-rag-eval.yml
```

## Adding Custom Evaluators
You can add custom evaluators to evaluate the workflow output. To add a custom evaluator, you need to implement the evaluator and register it with the NeMo Agent toolkit evaluator system. See the [Custom Evaluator](../extend/custom-evaluator.md) documentation for more information.


## Running multiple repetitions
You can run multiple repetitions of the evaluation by running a command line option `--reps`. For example, to run the evaluation 5 times, run the following command:
```bash
nat eval --config_file=examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config.yml --reps=5
```
This will allow you to get an average score across multiple runs and analyze the variation in the generated outputs.

## Running evaluation on large datasets
Similar to how evaluators are run in parallel, entries in the dataset are also processed in parallel. Concurrency is configurable using the `eval.general.max_concurrency` parameter in the `config.yml` file. The default value is 8. Increase or decrease the value based on the available resources.
```yaml
eval:
  general:
    max_concurrency: 4
```

## Pickup where you left off
When running the evaluation on a large dataset, it is recommended to resume the evaluation from where it was left off. This is particularly useful while using overloaded services that may timeout while running the workflow. When that happens a workflow interrupted warning is issued and workflow output is saved to a file.

You can then re-run evaluation on that output file along with `--skip_completed_entries` options.

Pass-1:
```
nat eval --config_file=examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config.yml
```
This pass results in workflow interrupted warning. You can then do another pass.

Pass-2:
```bash
cp .tmp/nat/examples/getting_started/simple_web_query/workflow_output.json .tmp/simple_workflow_output.json
nat eval --config_file=examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config.yml --skip_completed_entries --dataset=.tmp/simple_workflow_output.json
```

## Running evaluation offline
You can evaluate a dataset with previously generated answers via the `--skip_workflow` option. In this case the dataset has both the expected `answer` and the `generated_answer`.
```bash
cp .tmp/nat/examples/getting_started/simple_web_query/workflow_output.json .tmp/simple_workflow_output.json
nat eval --config_file=examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config.yml --skip_workflow --dataset=.tmp/simple_workflow_output.json
```
This assumes that the workflow output was previously generated and stored in `.tmp/nat/examples/getting_started/simple_web_query/workflow_output.json`

## Running the workflow over a dataset without evaluation
You can do this by running `nat eval` with a workflow configuration file that includes an `eval` section with no `evaluators`.
```yaml
eval:
  general:
    output_dir: ./.tmp/nat/examples/getting_started/simple_web_query/
    dataset:
      _type: json
      file_path: examples/evaluation_and_profiling/simple_web_query_eval/data/langsmith.json
```

## Evaluation output
The output of the workflow is stored as `workflow_output.json` in the `output_dir` provided in the config.yml -
```yaml
eval:
  general:
    output_dir: ./.tmp/nat/examples/getting_started/simple_web_query/
```
Here is a sample workflow output snipped generated by running evaluation on the `simple` example workflow -
```
  {
    "id": "1",
    "question": "What is langsmith",
    "answer": "LangSmith is a platform for LLM application development, monitoring, and testing",
    "generated_answer": "LangSmith is a platform for LLM (Large Language Model) application development, monitoring, and testing. It provides features such as automations, threads, annotating traces, adding runs to a dataset, prototyping, and debugging to support the development lifecycle of LLM applications.",
    "intermediate_steps": [
      {
        >>>>>>>>>>>>>>> SNIPPED >>>>>>>>>>>>>>>>>>>>>>
      }
    ],
    "expected_intermediate_steps": []
  },
```

The output of the evaluators are stored in distinct files in the same `output_dir` as `<evaluator_name>_output.json`. An evaluator typically provides an average score and a score per-entry. Here is a sample `rag_accuracy` output -
```bash
{
  "average_score": 0.6666666666666666,
  "eval_output_items": [
    {
      "id": 1,
      "score": 0.5,
      "reasoning": {
        "user_input": "What is langsmith"
      }
    },
    {
      "id": 2,
      "score": 0.75,
      "reasoning": {
        "user_input": "How do I proptotype with langsmith"
      }
    },
    {
      "id": 3,
      "score": 0.75,
      "reasoning": {
        "user_input": "What are langsmith automations?"
      }
    }
  ]
}
```

## Workflow Output Intermediate Step Filtering
The workflow_output.json file contains the intermediate steps for each entry in the dataset. The intermediate steps are filtered using the `eval.general.output.workflow_output_step_filter` parameter in the `config.yml` file. The default value for the filter is `[LLM_END, TOOL_END]`. You can customize the filter by providing a list of intermediate step types to include in the output file.

**Example:**
`examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config.yml` can be modified to include the intermediate steps in the output by adding the following configuration:
```yaml
eval:
  general:
    output:
    workflow_output_step_filter: [LLM_END, TOOL_START, TOOL_END]
```

## Customizing the output
You can customize the output of the pipeline by providing custom scripts. One or more Python scripts can be provided in the `eval.general.output_scripts` section of the `config.yml` file.

The custom scripts are executed after the evaluation is complete. They are executed as Python scripts with the `kwargs` provided in the `eval.general.output.custom_scripts.<script_name>.kwargs` section.

The `kwargs` typically include the file or directory to operate on. To avoid overwriting contents it is recommended to provide a unique output file or directory name for the customization. It is also recommended that changes be limited to the contents of the output directory to avoid unintended side effects.

**Example:**
```yaml
eval:
  general:
    output:
      dir: ./.tmp/nat/examples/simple_output/
      custom_scripts:
        convert_workflow_to_csv:
          script: examples/evaluation_and_profiling/simple_web_query_eval/scripts/workflow_to_csv.py
          kwargs:
            # The input and output are relative to the output directory
            input: workflow_output.json
            output: workflow.csv
```

## Remote Storage
### Evaluating remote datasets
You can evaluate a remote dataset by provide the information needed to download the dataset in the `eval.general.dataset` section of the `config.yml` file. The following is an example configuration to evaluate a remote dataset.
```yaml
eval:
  general:
    dataset:
      _type: json
      # Download dataset from remote storage using S3 credentials
      remote_file_path: input/langsmith.json
      file_path: ./.tmp/nat/examples/simple_input/langsmith.json
      s3:
        endpoint_url: http://10.185.X.X:9000
        bucket: nat-simple-bucket
        access_key: fake_access_key
        secret_key: fake_secret_key
```
The `remote_file_path` is the path to the dataset in the remote storage. The `file_path` is the local path where the dataset will be downloaded. The `s3` section contains the information needed to access the remote storage.

### Preserving outputs across multiple runs
By default, evaluation outputs are written to the same directory specified in `eval.general.output.dir`. This means that running the evaluation multiple times will overwrite previous results. To keep the outputs from each run separate, enable the `append_job_id_to_output_dir` option in the `job_management` section:

```yaml
eval:
  general:
    output:
      dir: ./.tmp/nat/examples/simple_output/
      job_management:
        append_job_id_to_output_dir: true
      cleanup: false
```

When `append_job_id_to_output_dir` is set to `true`, a unique job ID (`job_{UUID}`) is automatically generated for each evaluation run and appended to the output directory path. This results in:
- Local output path: `./.tmp/nat/examples/getting_started/simple_web_query/jobs/job_{unique-job-id}/`
- Remote output path (if S3 is configured): `output/jobs/job_{unique-job-id}/`

The `cleanup` option is used to control the cleanup of the output directory. If `cleanup` is set to `true`, the entire output directory and all job `sub-directories` are deleted at the beginning of the evaluation. Therefore, `cleanup` must be set to `false` if you want to preserve the output directory and job `sub-directories`.

### Uploading output directory to remote storage
You can upload the contents of the entire output directory to remote storage by providing the information needed to upload the output directory in the `eval.general.output` section of the `config.yml` file. The following is an example configuration to upload the output directory to remote storage.

For connecting with S3 using endpoint URL:
```yaml
eval:
  general:
    output:
      # Upload contents of output directory to remote storage using custom endpoint url & S3 credentials
      remote_dir: output
      s3:
        endpoint_url: http://10.185.X.X:9000
        bucket: nat-simple-bucket
        access_key: fake-access-key
        secret_key: fake-secret-key
```

For connecting with default S3 you can use `region_name` instead of `endpoint_url`:
```yaml
eval:
  general:
    output:
      # Upload contents of output directory to remote storage using S3 credentials
      remote_dir: output
      s3:
        region_name: us-west-2
        bucket: nat-simple-bucket
        access_key: fake-access-key
        secret_key: fake-secret-key
```

### Cleanup output directory
The contents of the output directory can be deleted before running the evaluation pipeline by specifying the `eval.general.output.cleanup` section in the `config.yml` file. The following is an example configuration to clean up the output directory before running the evaluation pipeline.
```yaml
eval:
  general:
    output:
      dir: ./.tmp/nat/examples/simple_output/
      cleanup: true
```
Output directory cleanup is disabled by default for easy troubleshooting.

### Job eviction from output directory
When running multiple evaluations, especially with `append_job_id_to_output_dir` enabled, the output directory can accumulate a large number of job folders over time. You can control this growth using a job eviction policy.
Configure job eviction with the following options in the `config.yml` file:
```yaml
eval:
  general:
    output:
      dir: ./.tmp/nat/examples/simple_output/
      cleanup: false
      job_management:
        append_job_id_to_output_dir: true
        max_jobs: 5
        eviction_policy: TIME_CREATED
```
Configuration notes:
- `max_jobs` sets the maximum number of job directories to keep. The oldest ones will be evicted based on the selected policy. Default is 0, which means no limit.
- `eviction_policy` controls how "oldest" is determined—either by creation time (TIME_CREATED) or last modification time (TIME_MODIFIED). Default is TIME_CREATED.

## Profiling and Performance Monitoring of NeMo Agent Toolkit Workflows
You can profile workflows using the NeMo Agent toolkit evaluation system. For more information, see the [Profiler](../workflows/profiler.md) documentation.
