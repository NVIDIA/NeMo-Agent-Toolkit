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


# Email phishing analyzer

## Table of Contents

- [Key Features](#key-features)
- [Installation and Setup](#installation-and-setup)
  - [Install this Workflow:](#install-this-workflow)
  - [Set Up API Keys](#set-up-api-keys)
- [Example Usage](#example-usage)
  - [Run the Workflow](#run-the-workflow)
- [Optimization](#optimization)
  - [What Is Being Optimized](#what-is-being-optimized)
  - [Optimization Configuration](#optimization-configuration)
  - [Run the Optimizer](#run-the-optimizer)
  - [Outputs](#outputs)
- [Deployment-Oriented Setup](#deployment-oriented-setup)
  - [Build the Docker Image](#build-the-docker-image)
  - [Run the Docker Container](#run-the-docker-container)
  - [Test the API](#test-the-api)
  - [Expected API Output](#expected-api-output)


## Key Features

- **Email Security Analysis:** Demonstrates an `email_phishing_analyzer` tool that examines email content for suspicious patterns, social engineering tactics, and phishing indicators using LLM-based analysis.
- **ReAct Agent Integration:** Uses a `react_agent` that can reason about email content and determine when to invoke the phishing analysis tool based on suspicious characteristics.
- **Phishing Detection Workflow:** Shows how to analyze emails for common phishing techniques including requests for sensitive information, urgency tactics, and suspicious sender patterns.
- **Security-Focused LLM Application:** Demonstrates how to apply AI reasoning to cybersecurity use cases with specialized prompting and analysis workflows.
- **Threat Assessment Pipeline:** Provides a foundation for building automated email security screening systems that can classify potential threats.


## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent toolkit.

### Install this Workflow:

From the root directory of the NeMo Agent toolkit library, run the following commands:

```bash
uv pip install -e examples/evaluation_and_profiling/email_phishing_analyzer
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

## Example Usage

### Run the Workflow

Run the following command from the root of the NeMo Agent toolkit repo to execute this workflow with the specified input:

```bash
nat run --config_file examples/evaluation_and_profiling/email_phishing_analyzer/configs/config.yml --input "Dear [Customer], Thank you for your purchase on [Date]. We have processed a refund of $[Amount] to your account. Please provide your account and routing numbers so we can complete the transaction. Thank you, [Your Company]"
```

The configuration file specified above contains configurations for the NeMo Agent Toolkit `evaluation` and `profiler` capabilities. Additional documentation for evaluation configuration can be found in the [evaluation guide](../../../docs/source/workflows/evaluate.md). Furthermore, similar documentation for profiling configuration can be found in the [profiling guide](../../../docs/source/workflows/profiler.md).

**Expected Workflow Output**
```console
2025-04-23 15:24:54,183 - nat.runtime.loader - WARNING - Loading module 'nat_automated_description_generation.register' from entry point 'nat_automated_description_generation' took a long time (502.501011 ms). Ensure all imports are inside your registered functions.
2025-04-23 15:24:54,483 - nat.cli.commands.start - INFO - Starting NeMo Agent toolkit from config file: 'examples/evaluation_and_profiling/email_phishing_analyzer/configs/config.yml'
2025-04-23 15:24:54,495 - nat.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 1
Number of LLMs: 3
Number of Embedders: 0
Number of Memory: 0
Number of Retrievers: 0

2025-04-23 15:24:58,017 - nat.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: Dear [Customer], Thank you for your purchase on [Date]. We have processed a refund of 0 to your account. Please provide your account and routing numbers so we can complete the transaction. Thank you, [Your Company]
Agent's thoughts:
Thought: This email seems suspicious as it asks for sensitive information such as account and routing numbers. I should analyze it for signs of phishing.

Action: email_phishing_analyzer
Action Input: {'text': 'Dear [Customer], Thank you for your purchase on [Date]. We have processed a refund of 0 to your account. Please provide your account and routing numbers so we can complete the transaction. Thank you, [Your Company]'}
Observation
------------------------------
/nemo-agent-toolkit/examples/evaluation_and_profiling/email_phishing_analyzer/src/nat_email_phishing_analyzer/register.py:56: LangChainDeprecationWarning: The method `BaseChatModel.apredict` was deprecated in langchain-core 0.1.7 and will be removed in 1.0. Use :meth:`~ainvoke` instead.
  response = await llm.apredict(config.prompt.format(body=text))
2025-04-23 15:25:07,477 - nat.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: email_phishing_analyzer
Tool's input: {"text": "Dear [Customer], Thank you for your purchase on [Date]. We have processed a refund of 0 to your account. Please provide your account and routing numbers so we can complete the transaction. Thank you, [Your Company]"}
Tool's response:
{"is_likely_phishing": true, "explanation": "The email exhibits suspicious signals that may indicate phishing. Specifically, the email requests sensitive personal information (account and routing numbers) under the guise of completing a refund transaction. Legitimate companies typically do not request such information via email, as it is a security risk. Additionally, the refund amount of '0' is unusual and may be an attempt to create a sense of urgency or confusion. The tone of the email is also somewhat generic and lacks personalization, which is another common trait of phishing emails."}
------------------------------
2025-04-23 15:25:08,862 - nat.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: Dear [Customer], Thank you for your purchase on [Date]. We have processed a refund of 0 to your account. Please provide your account and routing numbers so we can complete the transaction. Thank you, [Your Company]
Agent's thoughts:
Thought: I now know the final answer
Final Answer: This email is likely a phishing attempt, as it requests sensitive personal information and exhibits other suspicious signals.
------------------------------
2025-04-23 15:25:08,866 - nat.front_ends.console.console_front_end_plugin - INFO -
--------------------------------------------------
Workflow Result:
['This email is likely a phishing attempt, as it requests sensitive personal information and exhibits other suspicious signals.']
```

## Optimization

This example includes an optimization configuration that uses the NeMo Agent toolkit Optimizer to tune the workflow. For detailed information 
about the NeMo Agent Toolkit Optimizer, refer to the [Optimizer Documentation](../../../docs/source/reference/optimizer.md).

### What Is Being Optimized
- **Tool parameters**: The `email_phishing_analyzer` exposes two optimizable fields on its config:
  - **`llm`**: Categorical choice between `llama_3_405` and `llama_3_70`.
  - **`prompt`**: The prompt template used to analyze the email body (prompt optimization is disabled by default in this config; see below to enable).
- **LLM hyperparameters**: For each LLM in `llms`, numeric hyperparameters are marked as optimizable:
  - **`temperature`**, **`top_p`**, **`max_tokens`**.

Evaluation during optimization uses the dataset at `examples/evaluation_and_profiling/email_phishing_analyzer/data/smaller_test.csv` with `body` as the question and `label` as the ground truth.

### Optimization Configuration
The optimization-ready configuration is located at:
`examples/evaluation_and_profiling/email_phishing_analyzer/configs/config_optimizer.yml`

Key parts of the config:

```yaml
functions:
  email_phishing_analyzer:
    _type: email_phishing_analyzer
    optimizable_params:
      - llm
      - prompt
  
  # Prompt optimization functions are defined here
  prompt_init:
    _type: prompt_init
    optimizer_llm: prompt_optimizer
    system_objective: Agent that triages an email to see if it is a phishing attempt or not.
  
  prompt_recombination:
    _type: prompt_recombiner
    optimizer_llm: prompt_optimizer
    system_objective: Agent that triages an email to see if it is a phishing attempt or not.

llms:
  llama_3_405:
    _type: nim
    model_name: meta/llama-3.1-405b-instruct
    temperature: 0.0
    max_tokens: 1024
    optimizable_params: [temperature, top_p, max_tokens]
  llama_3_70:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    max_tokens: 1024
    optimizable_params: [temperature, top_p, max_tokens]
    
eval:
  general:
    output_dir: ./.tmp/eval/examples/evaluation_and_profiling/email_phishing_analyzer/original
    verbose: true
    dataset:
        _type: csv
        file_path: examples/evaluation_and_profiling/email_phishing_analyzer/data/smaller_test.csv
        id_key: "subject"
        structure:
          question_key: body
          answer_key: label

  evaluators:
    accuracy:
      _type: ragas
      metric: AnswerAccuracy
      llm_name: prompt_optimizer
    llm_latency:
      _type: avg_llm_latency
    token_efficiency:
      _type: avg_tokens_per_llm_end

optimizer:
  output_path: ./.tmp/examples/evaluation_and_profiling/email_phishing_analyzer/optimizer/
  reps_per_param_set: 1
  eval_metrics:
    accuracy:
      evaluator_name: accuracy
      direction: maximize
    token_efficiency:
      evaluator_name: token_efficiency
      direction: minimize
    latency:
      evaluator_name: llm_latency
      direction: minimize


  numeric:
    enabled: true
    n_trials: 3

  prompt:
    enabled: true
    prompt_population_init_function: prompt_init
    prompt_recombination_function: prompt_recombination
    ga_generations: 3
    ga_population_size: 3
    ga_diversity_lambda: 0.3
    ga_parallel_evaluations: 1
```

Notes:
- The `prompt_init` and `prompt_recombination` functions are defined in the `functions` section of the same config file
- These functions use the `prompt_optimizer` LLM to generate prompt variations based on the `system_objective`
- Increase `optimizer.numeric.n_trials` for a deeper search (for example, 20–50)
- To optimize prompts, set `optimizer.prompt.enabled: true`

### Run the Optimizer
From the repository root:

```bash
nat optimize --config_file examples/evaluation_and_profiling/email_phishing_analyzer/configs/config_optimizer.yml
```

Ensure `NVIDIA_API_KEY` is set in your environment.

### Outputs
Results are written to the path specified by `optimizer.output_path`. Expect artifacts such as:
- `optimized_config.yml`: Tuned configuration derived from the selected trial.
- You will also see a configuration file for each iteration of numeric trials. For example, `config_numeric_trial_0.yml`
  will contain the configuration for the first numeric trial. This is helpful for selecting specific trials whose metrics
  you may prefer to the optimizer selected trial.
- `trials_dataframe_params.csv`: Full Optuna trials `dataframe` with columns:
  - `values_accuracy`, `values_token_efficiency`, `values_latency`: Metric scores (named after your `eval_metrics`)
  - `params_*`: Parameter values for each trial
  - `datetime_start`, `datetime_complete`, `duration`: Timing information
  - `rep_scores`: Raw scores for each repetition
- `plots`: This directory will contain Pareto visualizations of the optimization results.
- For prompt optimization (when enabled): `optimized_prompts.json` and per-generation prompt history. Per generation prompt
  history files are named `optimized_prompts_gen{N}.json` where `{N}` is the generation number starting from 1.

#### Understanding the Pareto Visualizations

For a detailed guide on interpreting the output of the optimization process, including the 
Pareto visualizations, refer to the [Optimizer Output Analysis](../../../docs/source/reference/optimizer.md#understanding-the-output) section in the 
NeMo Agent toolkit documentation.

---

## Deployment-Oriented Setup

For a production deployment, use Docker:

### Build the Docker Image

Prior to building the Docker image ensure that you have followed the steps in the [Installation and Setup](#installation-and-setup) section, and you are currently in the NeMo Agent toolkit virtual environment.

From the root directory of the NeMo Agent toolkit repository, build the Docker image:

```bash
docker build --build-arg NAT_VERSION=$(python -m setuptools_scm) -t email_phishing_analyzer -f examples/evaluation_and_profiling/email_phishing_analyzer/Dockerfile .
```

### Run the Docker Container
Deploy the container:

```bash
docker run -p 8000:8000 -e NVIDIA_API_KEY email_phishing_analyzer
```

### Test the API
Use the following curl command to test the deployed API:

```bash
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"input_message": "Dear [Customer], Thank you for your purchase on [Date]. We have processed a refund of $[Amount] to your account. Please provide your account and routing numbers so we can complete the transaction. Thank you, [Your Company]"}'
```

### Expected API Output
The API response should look like this:

```json
{"value":"This email is likely a phishing attempt. It requests sensitive information, such as account and routing numbers, which is a common tactic used by scammers. The email also lacks specific details about the purchase, which is unusual for a refund notification. Additionally, the greeting is impersonal, which suggests a lack of personalization. It is recommended to be cautious when responding to such emails and to verify the authenticity of the email before providing any sensitive information."}
```
