<!--
Copyright (c) 2025-2026, NVIDIA CORPORATION

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
<!-- path-check-skip-file -->

> [!NOTE]
> ⚠️ **EXPERIMENTAL**: This integration between NeMo Agent toolkit and Dynamo is experimental and under active development. APIs, configurations, and features may change without notice.

# Agent Leaderboard v2 - Evaluation Guide

This guide walks through the complete process of running decision-only evaluations using the `react_benchmark_agent`: downloading data, configuring evaluations, running experiments, and analyzing results.

Currently this agent supports evaluation exclusively for the [Galileo Agent Leaderboard v2](https://huggingface.co/datasets/galileo-ai/agent-leaderboard-v2). However, we plan to extend the set of evaluation tool sets and benchmarks and will update this document accordingly.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Configuration Files](#configuration-files)
5. [Running Evaluations](#running-evaluations)
6. [Self-Evaluation Loop](#self-evaluation-loop)
7. [Understanding Results](#understanding-results)
8. [Performance Analysis](#performance-analysis)
9. [Concurrency Benchmarking](#concurrency-benchmarking)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Software Requirements

1. **Python 3.11, 3.12, or 3.13** installed
2. **NeMo Agent toolkit** repository cloned
3. **Docker**
4. **Hugging Face account** with access to Llama-3.3-70B-Instruct model

### Hardware Requirements (Dynamo Backend)

Running these evaluations requires a Dynamo backend with adequate GPU resources. The following are the minimum and recommended specifications:

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU Architecture** | NVIDIA Hopper (H100) or Blackwell (B200) | B200 for optimal performance |
| **GPU Count** | 4 GPUs (TP=4 for 70B model) | 8 GPUs for optimal performance |
| **GPU Memory** | 80GB per GPU (H100) | 192GB per GPU (B200) |

> **Note**: The Llama-3.3-70B-Instruct model requires approximately 140GB of GPU memory when loaded with TP=4 (tensor parallelism across 4 GPUs). Ensure your GPU configuration has sufficient aggregate memory.

### Dynamo Backend

The Dynamo backend must be running on `localhost:8099` before executing evaluations. See the [Dynamo Setup Guide](../../../external/dynamo/README.md) for detailed instructions on:

- Starting Dynamo in unified or disaggregated mode
- Configuring GPU workers and tensor parallelism
- Setting up the Thompson Sampling router for KV cache optimization
- Troubleshooting common issues

> **Note:** For a more abbreviated way to kick off experimentation, see the [Quick Start](../README.md#quick-start) section in the parent README. This document provides a more detailed explanations of the different test patterns and configurations available.

---

## Environment Setup

### Create Virtual Environment

```bash
# Navigate to the repository root
cd /path/to/NeMo-Agent-Toolkit

# Create virtual environment with uv
uv venv "${HOME}/.venvs/nat_dynamo_eval" --python 3.13
source "${HOME}/.venvs/nat_dynamo_eval/bin/activate"

# Install nvidia-nat with LangChain support
uv pip install -e ".[langchain]"

# Install visualization dependencies
uv pip install matplotlib scipy

# Install the workflow package
cd examples/dynamo_integration/react_benchmark_agent
uv pip install -e .
```

### Environment Configuration

Key environment variables:

```bash
# Hugging Face configuration (required for gated datasets)
# Set HF_HOME to a local directory if your home is on NFS (avoids file locking issues)
export HF_HOME=/path/to/local/storage/.cache/huggingface
export HF_TOKEN=<your_huggingface_token>
```

> **Note:** Dynamo-specific environment variables (`DYNAMO_BACKEND`, `DYNAMO_MODEL`, `DYNAMO_PORT`) are used by the test scripts in `external/dynamo/` and are not required for running evaluations. See [Dynamo Setup Guide](../../../external/dynamo/README.md) for those options.

### Start Dynamo Backend

Before running evaluations, ensure Dynamo is running:

<!-- path-check-skip-begin -->
```bash
cd ../../../external/dynamo/ # NeMo-Agent-Toolkit/external/dynamo
bash start_dynamo_unified.sh
bash test_dynamo_integration.sh
```
<!-- path-check-skip-end -->

> **Note:** To customize GPU workers and tensor parallelism, edit the configuration variables at the top of `external/dynamo/start_dynamo_unified.sh`:
> - `WORKER_GPUS="4,5,6,7"` - GPU device IDs to use (for example, `"0,1"` for first 2 GPUs)
> - `TP_SIZE=4` - Tensor parallel size (must match number of GPUs)
> - `HTTP_PORT=8099` - API endpoint port
> - `LOCAL_MODEL_DIR="..."` - Path to your local model weights

See [Dynamo Setup Guide](../../../external/dynamo/README.md) for detailed configuration options.

---

## Dataset Preparation

### Download and Preprocess

```bash
cd ../../../examples/dynamo_integration
source "${HOME}/.venvs/nat_dynamo_eval/bin/activate"
export HF_TOKEN=<your_huggingface_token>

python scripts/download_agent_leaderboard_v2.py --domains banking
```

**Creates**:
- `data/agent_leaderboard_v2_banking.json` - 100 enriched scenarios
- `data/raw/banking/tools.json` - 20 banking tool schemas
- Each scenario includes `expected_tool_calls` derived from `user_goals`

### Create Test Subsets

The minimal test config (`eval_config_no_rethinking_minimal_test.yml`) requires a test subset. This configuration can be used for quick end-to-end tests, without running the entire dataset through `nat eval`. Create it with:

```bash
# cd /path/to/NeMo-Agent-Toolkit/examples/dynamo_integration

# 3-scenario subset for quick testing (required by eval_config_no_rethinking_minimal_test.yml)
python scripts/create_test_subset.py \
  --input-file ./data/agent_leaderboard_v2_banking.json \
  --output-file ./data/agent_leaderboard_v2_test_subset.json \
  --num-scenarios 3

# Single scenario for debugging
python scripts/create_test_subset.py \
  --input-file ./data/agent_leaderboard_v2_banking.json \
  --output-file ./data/agent_leaderboard_v2_single.json \
  --num-scenarios 1
```

### Dataset Format

Each scenario in the dataset contains:

```json
{
  "id": "banking_scenario_000",
  "question": "I need to check my balance and transfer $500...",
  "user_goals": ["Check account balance", "Transfer funds", "Verify transaction"],
  "available_tools": [...],
  "expected_tool_calls": ["get_account_balance", "transfer_funds", "get_transaction_history"],
  "metadata": {...}
}
```

---

## Configuration Files

### Available Configurations

| Configuration File | Description | Dataset | Use Case |
|-------------|-------------|---------|----------|
| `eval_config_no_rethinking_full_test.yml` | Full evaluation | 100 scenarios | Production benchmarks |
| `eval_config_no_rethinking_minimal_test.yml` | Quick test | 3 scenarios | Validation |
| `eval_config_rethinking_full_test.yml` | Self-evaluation loop | 100 scenarios | Quality optimization |
| `profile_rethinking_full_test.yml` | Profiler + self-eval | 100 scenarios | Performance analysis |
| `optimize_rethinking_full_test.yml` | Prefix header optimization | 100 scenarios | Dynamo Predictive KV-Aware Cache router tuning |
| `config_dynamo_e2e_test.yml` | LangChain + Dynamo integration | Single query | Framework integration test |
| `config_dynamo_prefix_e2e_test.yml` | LangChain + Dynamo with prefix headers | Single query | KV cache optimization test |
| `config_dynamo_adk_e2e_test.yml` | Google ADK + Dynamo integration | Single query | ADK framework integration test |

All config files are located in `react_benchmark_agent/configs/`.

### Framework Integration Tests

The Dynamo LLM provider supports multiple agent frameworks. Each framework has a dedicated e2e test configuration to verify the integration works correctly.

#### Why Test ADK + Dynamo Integration?

Google ADK (Agent Development Kit) is an increasingly popular framework for building AI agents. Testing the Dynamo + ADK integration is important because:

1. **Different header injection mechanism**: ADK uses LiteLLM under the hood, which requires passing headers via `extra_headers` at client initialization time, unlike LangChain which uses `httpx` event hooks for per-request injection.

2. **Conversation-level prefix ID consistency**: All requests from the same ADK client instance share the same prefix ID, which is ideal for KV cache optimization in multi-turn conversations.

3. **Provider prefix requirements**: LiteLLM requires model names to be prefixed with the provider (for example, `openai:llama-3.3-70b`) for custom endpoints, which differs from LangChain's direct model name usage.

#### Running ADK Integration Test

```bash
# Install ADK demo package (required for ADK workflow)
cd ../../ # /path/to/NeMo-Agent-Toolkit
pip install -e './examples/frameworks/adk_demo' # may need to use --no-deps depending on working branch version

# Run the ADK + Dynamo integration test (basic I/O)
nat run --config_file examples/dynamo_integration/react_benchmark_agent/configs/config_dynamo_adk_e2e_test.yml \
    --input "Hello! What is 2+2?"
```

**Expected output**: The Dynamo prefix headers should be logged, and all LLM calls within the conversation will share the same prefix ID (for example, `adk-dynamo-test-7a31631c0ec24857`).

> **Note**: The ADK e2e test is configured for **basic I/O testing only** (no tool calling). This is because ADK with LiteLLM requires OpenAI-style function calling support from the model endpoint, which vanilla llama models served via vLLM or Dynamo don't support out of the box. For tool-calling workflows with Dynamo, use the LangChain + ReAct agent `configs` (for example, `config_dynamo_prefix_e2e_test.yml`) which parse tool calls from text output.

### Key Configuration Sections

#### LLM Configuration

```yaml
llms:
  dynamo_llm:
    _type: dynamo
    model_name: llama-3.3-70b
    base_url: http://localhost:8099/v1
    api_key: dummy
    temperature: 0.0
    max_tokens: 8192
    stop: ["Observation:", "\nThought:"]  # CRITICAL: Prevents observation hallucination
    
    # Optional: Customize prefix headers (sent by default with "nat-dynamo-{uuid}")
    # prefix_template: "react-benchmark-{uuid}"  # Custom template
    prefix_total_requests: 10
    prefix_osl: MEDIUM  # Output Sequence Length: LOW | MEDIUM | HIGH
    prefix_iat: MEDIUM  # Inter-Arrival Time: LOW | MEDIUM | HIGH
```

> **Note**: The `dynamo` LLM type automatically sends prefix headers for KV cache optimization. Headers are enabled by default using the template `nat-dynamo-{uuid}`. You can customize the template with `prefix_template` or disable headers entirely by setting `prefix_template: null`. These headers help the Predictive KVCache-Aware Thompson Sampling router make optimal routing decisions (see [Dynamo Setup Guide](../../../external/dynamo/README.md)).

#### Decision-Only Tool Configuration

For TSQ evaluation, tools must be configured in decision-only mode:

```yaml
functions:
  react_benchmark_agent:
    _type: react_benchmark_agent
    prefix: "Agent:"
    decision_only: true
    canned_response_template: "Successfully executed {tool_name}. Operation completed."

function_groups:
  banking_tools:
    _type: banking_tools_group
    # tools.json available after running: /examples/dynamo_integration/scripts/download_agent_leaderboard_v2.py
    tools_json_path: ./examples/dynamo_integration/data/raw/banking/tools.json
    decision_only: true
    include: [
      get_account_balance,
      get_transaction_history,
      transfer_funds,
      # ... all 20 banking tools
    ]
```

> **Note**: The `decision_only: true` setting is required for TSQ evaluation. It makes tools return canned responses instead of executing real banking operations. The `canned_response_template` defines the response format (for example, "Successfully executed {tool_name}"). This allows evaluation of tool *selection* without needing actual backend services.

#### Workflow Configuration

```yaml
workflow:
  _type: react_agent
  llm_name: dynamo_llm
  tool_names: [
    banking_tools.get_account_balance,
    banking_tools.transfer_funds,
    # ... all tools with banking_tools. prefix
  ]
  verbose: true
  max_tool_calls: 25
  recursion_limit: 50
  pass_tool_call_errors_to_agent: true
```

#### Evaluation Settings

```yaml
eval:
  general:
    max_concurrency: 36  # Range: 1-64
    
    output:
      dir: ./examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/
      cleanup: false
      job_management:
        append_job_id_to_output_dir: true
    
    dataset:
      _type: json
      file_path: ./examples/dynamo_integration/data/agent_leaderboard_v2_banking.json
      structure:
        disable: true

  evaluators:
    tool_selection_quality:
      _type: tsq_evaluator
      llm_name: eval_llm
      strict_mode: false
      tool_weight: 1.0
      parameter_weight: 0.0  # Set > 0 to evaluate parameter accuracy
      verbose: true
```

---

## Running Evaluations

### Verify Dynamo is Running

```bash
curl http://localhost:8099/health
# Expected: HTTP 200 OK, else check dynamo runtime
```

If Dynamo isn't running, see [Dynamo Setup Guide](../../../external/dynamo/README.md).

### Run Quick Validation (3 scenarios)

> **Prerequisite**: Create the test subset file first (if not already created):
> ```bash
> cd /path/to/NeMo-Agent-Toolkit/examples/dynamo_integration
> python scripts/create_test_subset.py \
>   --input-file ./data/agent_leaderboard_v2_banking.json \
>   --output-file ./data/agent_leaderboard_v2_test_subset.json
> ```

```bash
# cd /path/to/NeMo-Agent-Toolkit
source "${HOME}/.venvs/nat_dynamo_eval/bin/activate"

nat eval --config_file examples/dynamo_integration/react_benchmark_agent/configs/eval_config_no_rethinking_minimal_test.yml
```

**Runtime**: <1 minute  
**Expected TSQ**: 0.3 - 0.6

### Run Full Evaluation (100 scenarios)

```bash
nat eval --config_file examples/dynamo_integration/react_benchmark_agent/configs/eval_config_no_rethinking_full_test.yml
```

**Runtime**: ~30-60 minutes (depends on concurrency)  
**Expected TSQ**: 0.4 - 0.7

### Expected Output

```text
✓ 20/20 banking tool stubs registered
✓ Tool stub executed: get_exchange_rates with 3 parameters
✓ Tool stub executed: setup_automatic_bill_pay with 8 parameters
Running workflow: 100%|██████████| 100/100 [00:45:12<00:00]
✓ TSQ Evaluation complete: average_score=0.571
```

---

## Self-Evaluation Loop

The self-evaluation mechanism allows the agent to evaluate its own tool selection and retry if insufficient. This can improve TSQ scores by 5-15%.

### How It Works

```text
User Question
    ↓
[Attempt 1] ReAct Agent executes
    ↓
Tool calls captured: [Tool A, Tool B, Tool C]
    ↓
Self-Evaluator LLM reviews:
  - Are these tools sufficient?
  - Is anything missing?
    ↓
Evaluation Result:
  - is_sufficient: false
  - confidence: 0.60
  - missing_steps: ["verify_transaction"]
    ↓
[Decision] Confidence < threshold → Retry
    ↓
[Attempt 2] ReAct Agent executes (with feedback)
    ↓
Tool calls captured: [Tool A, Tool B, Tool C, Tool D]
    ↓
Self-Evaluator: is_sufficient: true, confidence: 0.85
    ↓
✓ Accept result
```

### Configuration

Use `eval_config_rethinking_full_test.yml`:

```yaml
functions:
  # Define the ReAct workflow as a function
  react_workflow:
    _type: react_agent
    llm_name: dynamo_llm
    tool_names: [banking_tools.get_account_balance, ...]
    verbose: true
    max_tool_calls: 25

# Wrap with self-evaluating agent
workflow:
  _type: self_evaluating_agent_with_feedback
  wrapped_agent: react_workflow
  evaluator_llm: eval_llm
  max_retries: 5
  min_confidence_threshold: 0.85
  pass_feedback_to_agent: true  # KEY: Pass evaluation feedback on retry
  verbose: true
  feedback_template: |
    PREVIOUS ATTEMPT FEEDBACK:
    
    Your previous tool selection was evaluated and found to be insufficient.
    
    EVALUATION: {reasoning}
    MISSING STEPS: {missing_steps}
    SUGGESTIONS: {suggestions}
    
    Please try again, addressing the issues identified above.
```

### Self-Evaluation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wrapped_agent` | `FunctionRef` | *required* | Reference to underlying ReAct agent |
| `evaluator_llm` | `LLMRef` | *required* | LLM for self-evaluation |
| `max_retries` | `int` | 2 | Maximum retry attempts (0-5) |
| `min_confidence_threshold` | `float` | 0.7 | Minimum confidence to accept (0.0-1.0) |
| `pass_feedback_to_agent` | `bool` | false | Pass evaluation feedback on retry |
| `verbose` | `bool` | true | Enable detailed logging |

### Running with Self-Evaluation

```bash
nat eval --config_file examples/dynamo_integration/react_benchmark_agent/configs/eval_config_rethinking_full_test.yml
```

### Log Output Example

```text
================================================================================
Attempt 1/6
================================================================================
INFO: Captured 2 tool calls
INFO:   1. get_account_balance
INFO:   2. transfer_funds
--------------------------------------------------------------------------------
Self-Evaluation Result:
  Sufficient: False
  Confidence: 0.60
  Reasoning: Missing verification step after transfer
  Missing steps: verify_transaction_status
--------------------------------------------------------------------------------
✗ Tool sequence insufficient - retrying...
================================================================================
Attempt 2/6
================================================================================
INFO: Captured 3 tool calls
INFO:   1. get_account_balance
INFO:   2. transfer_funds
INFO:   3. get_transaction_history
--------------------------------------------------------------------------------
Self-Evaluation Result:
  Sufficient: True
  Confidence: 0.85
--------------------------------------------------------------------------------
✓ Tool sequence accepted
```

### Performance Impact

| Metric | Without Self-Eval | With Self-Eval |
|--------|-------------------|----------------|
| Average attempts per question | 1 | 1.3-1.8 |
| Token usage | Baseline | +15-20% |
| Latency | Baseline | +30-80% |
| TSQ score improvement | - | +5-15% |

### Tuning Recommendations

**For Speed:**
```yaml
max_retries: 1
min_confidence_threshold: 0.6
```

**For Quality:**
```yaml
max_retries: 3
min_confidence_threshold: 0.85
pass_feedback_to_agent: true
```

---

## Understanding Results

### Output Files

Results are saved to `react_benchmark_agent/outputs/dynamo_evals/<job_id>/`:

| File | Description |
|------|-------------|
| `tool_selection_quality_output.json` | TSQ scores per scenario |
| `standardized_data_all.csv` | Profiler data (tokens, timestamps) |
| `all_requests_profiler_traces.json` | Raw trace data |
| `workflow_profiling_report.txt` | Human-readable profiling summary |

### TSQ Output Structure

```json
{
  "average_score": 0.571,
  "eval_output_items": [{
    "id": "banking_scenario_000",
    "score": 0.571,
    "reasoning": {
      "tool_selection_accuracy": 0.571,
      "parameter_usage_accuracy": 0.0,
      "actual_tool_calls": 5,
      "expected_tool_calls": 8,
      "details": {
        "actual_tools": ["get_exchange_rates", "setup_automatic_bill_pay", ...],
        "expected_tools": ["get_credit_card_information", "report_lost_stolen_card", ...]
      }
    }
  }]
}
```

### TSQ Calculation

TSQ uses F1 score to balance precision and recall:

```text
Precision = Correct Tools / Actual Tools Called
Recall    = Correct Tools / Expected Tools
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```

**Example:**
```python
actual_tools = {tool1, tool2, tool3}      # 3 tools called
expected_tools = {tool2, tool3, tool4, tool5}  # 4 tools expected
intersection = {tool2, tool3}              # 2 correct

precision = 2/3 = 0.667   # Called 1 extra unnecessary tool
recall    = 2/4 = 0.500   # Missed 2 expected tools
f1_score  = 2 × (0.667 × 0.500) / (0.667 + 0.500) = 0.571
```

### Interpreting Scores

| Score Range | Quality | Interpretation |
|-------------|---------|----------------|
| 0.0 - 0.3 | Poor | Agent selecting wrong tools |
| 0.3 - 0.6 | Moderate | Right general idea, some confusion |
| 0.6 - 0.8 | Good | Mostly correct tool selection |
| 0.8 - 1.0 | Excellent | Near-perfect tool selection |

---

## Performance Analysis

### Throughput Analysis

After evaluation, analyze token generation performance:

```bash
cd examples/dynamo_integration # /path/to/NeMo-Agent-Toolkit/examples/dynamo_integration

python scripts/throughput_analysis.py \
  react_benchmark_agent/outputs/dynamo_evals/<workflow_output_dir>/<job_id>/standardized_data_all.csv
```

**Output metrics:**
- **TTFT (Time To First Token)**: Mean, median, P90, P95, P99
- **ITL (Inter-Token Latency)**: Time between consecutive tokens
- **Per-Request Throughput**: Tokens per second for individual calls
- **Aggregate Throughput**: Total tokens / wall-clock time

**Example output:**
```text
================================================================================
LLM Performance Analysis Summary
================================================================================

Dataset Overview:
  Total LLM Calls:        210
  Total Tokens Generated: 20,880
  Wall-Clock Time:        236.3s

--------------------------Time To First Token (TTFT)----------------------------
  Mean:     52.44 ms
  Median:   52.70 ms
  P95:      54.10 ms

------------Inter-Token Latency (ITL) / Time Per Output Token (TPOT)------------
  Mean:     10.74 ms
  Median:   10.88 ms
  P95:      11.21 ms

-----------------------Per-Request Throughput (Tokens Per Second)---------------
  Mean:     89.43 tok per second
  Median:   89.42 tok per second

-----------------Aggregate Throughput (All Concurrent Requests)-----------------
  Aggregate Throughput:   88.37 tokens per second
================================================================================
```

### Throughput vs TSQ Plots

Generate scatter plots comparing throughput metrics against TSQ scores:

```bash
# cd /path/to/NeMo-Agent-Toolkit/examples/dynamo_integration

python scripts/plot_throughput_histograms_per_request.py \
  react_benchmark_agent/outputs/dynamo_evals/<workflow-output-dir>/jobs
```

**Generated plots:**
- `ttft_histogram.png` - Time To First Token distribution
- `itl_histogram.png` - Inter-Token Latency distribution
- `tps_histogram.png` - Tokens Per Second distribution
- `total_tokens_histogram.png` - Total tokens per request distribution
- `llm_calls_histogram.png` - LLM calls per request distribution
- `total_duration_histogram.png` - Request duration distribution
- `summary_throughput_histograms.png` - Multi-panel summary
- `throughput_histogram_data.csv` - Aggregated histogram data
- `throughput_histogram_per_llm_call_data.csv` - Per-LLM-call data

---

## Concurrency Benchmarking

The `scripts/run_concurrency_benchmark.sh` script automates performance testing across different concurrency levels.

### What It Does

1. Runs evaluations with `max_concurrency` set to 2, 4 (configurable)
2. Tracks each job and its output directory
3. Analyzes performance using `scripts/throughput_analysis.py`
4. Aggregates results into CSV and markdown reports

### Running the Benchmark

```bash
# cd /path/to/NeMo-Agent-Toolkit/examples/dynamo_integration

./scripts/run_concurrency_benchmark.sh # could take ~30 minutes to run
# When prompted, enter a unique name (e.g., "baseline_v1")
```

### Output Structure

```
react_benchmark_agent/outputs/benchmarks/<name>_<timestamp>/
├── benchmark_results.csv          # Machine-readable CSV
├── benchmark_report.md            # Human-readable markdown
├── analysis_16.txt                 # Detailed analysis for concurrency=16
├── analysis_32.txt                 # Detailed analysis for concurrency=32
└── ...
```

### CSV Format

```text
concurrency,total_llm_calls,total_tokens,total_duration_sec,
ttft_mean_ms,ttft_median_ms,ttft_p90_ms,ttft_p95_ms,
itl_mean_ms,itl_median_ms,itl_p90_ms,itl_p95_ms,
throughput_mean_toks,throughput_median_toks,...
```

### Expected Runtime

- Each eval run: 15-30 minutes (depends on dataset size)
- Total benchmark (2 concurrency levels by default): **30-60 minutes**
- Runs sequentially to avoid interference

### Customization

Edit `scripts/run_concurrency_benchmark.sh` to change concurrency levels, for example:

```bash
# Change concurrency levels (around line 66)
CONCURRENCY_LEVELS=(1 2 4 8 16 32)
```

---

## Troubleshooting

### Permission Denied Downloading Dataset

**Symptom**: `PermissionError: [Errno 13] Permission denied: '.../.cache/huggingface/hub/.locks/...'`

**Cause**: Your home directory is on NFS and doesn't support file locking

**Fix**: Set `HF_HOME` to a local writable directory (not on NFS):
```bash
export HF_HOME=/path/to/local/storage/.cache/huggingface
```

### Tools Not Executing (Hallucinated Observations)

**Symptom**: Observations don't match mock JSON responses

**Fix**: Ensure stop sequence and system prompt are set:
```yaml
llms:
  dynamo_llm:
    stop: ["Observation:"]

workflow:
  system_prompt: |
    ... STOP HERE. DO NOT generate the Observation ...
```

### TSQ Score Always 0.0

**Symptom**: `actual_tool_calls: 0`

**Cause**: Tools aren't being executed or tool stubs aren't configured for decision-only mode.

**Fix**: 
1. Check logs for "Tool stub executed" - if missing, tools aren't running
2. Ensure your `function` and `function_groups` `config` files have `decision_only: true` and a `canned_response_template`:

```yaml
functions:
  react_benchmark_agent:
    _type: react_benchmark_agent
    prefix: "Agent:"
    decision_only: true
    canned_response_template: "Successfully executed {tool_name}. Operation completed."

function_groups:
  banking_tools:
    _type: banking_tools_group
    # tools.json available after running: /examples/dynamo_integration/scripts/download_agent_leaderboard_v2.py
    tools_json_path: ./examples/dynamo_integration/data/raw/banking/tools.json
    decision_only: true
```

Both `decision_only: true` settings are required. The `canned_response_template` defines the mock response format returned by tools. See [Decision-Only Tool Configuration](#decision-only-tool-configuration) for details.

### Module Not Found

**Symptom**: `ModuleNotFoundError: react_benchmark_agent`

**Fix**:
```bash
cd examples/dynamo_integration/react_benchmark_agent
pip install -e . --force-reinstall
```

### File Not Found Errors

**Symptom**: Configuration paths not resolving

**Fix**: Run `nat eval` from the repository root:
```bash
cd /path/to/NeMo-Agent-Toolkit  # repository root, not workflow directory
nat eval --config_file examples/dynamo_integration/react_benchmark_agent/configs/...
```

### Recursion Limit Reached

**Symptom**: `GraphRecursionError: Recursion limit of 42 reached`

**Fix**: Increase recursion limit in config:
```yaml
workflow:
  recursion_limit: 100
  max_tool_calls: 40
```

### Self-Evaluation Always Retries

**Symptom**: Agent never accepts tool sequence

**Fix**: Lower confidence threshold:
```yaml
workflow:
  _type: self_evaluating_agent_with_feedback
  min_confidence_threshold: 0.6  # More lenient
```

### Dynamo Connection Errors

**Check Dynamo health**:
```bash
curl http://localhost:8099/health
```

**Restart if needed**:
```bash
cd /path/to/NeMo-Agent-Toolkit/external/dynamo
bash stop_dynamo.sh
bash start_dynamo_unified.sh
```

See [Dynamo Setup Guide](../../../external/dynamo/README.md) for detailed troubleshooting.

---

## Quick Reference

All commands should be run from the NeMo-Agent-Toolkit repository root.

### End-to-End Tests (Workflow Runs)

```bash
# Basic Dynamo connectivity test
nat run --config_file examples/dynamo_integration/react_benchmark_agent/configs/config_dynamo_e2e_test.yml \
  --input "What time is it?"

# Dynamo with prefix headers (for KV cache optimization)
nat run --config_file examples/dynamo_integration/react_benchmark_agent/configs/config_dynamo_prefix_e2e_test.yml \
  --input "What time is it?"

# ADK + Dynamo integration test
nat run --config_file examples/dynamo_integration/react_benchmark_agent/configs/config_dynamo_adk_e2e_test.yml \
  --input "Hello! What is 2+2?"
```

### Evaluations

```bash
# Quick validation (3 scenarios, ~1 minute)
nat eval --config_file examples/dynamo_integration/react_benchmark_agent/configs/eval_config_no_rethinking_minimal_test.yml

# Full evaluation without self-evaluation (100 scenarios, ~5-10 min)
nat eval --config_file examples/dynamo_integration/react_benchmark_agent/configs/eval_config_no_rethinking_full_test.yml

# Full evaluation with self-evaluation loop (100 scenarios, ~45 min)
nat eval --config_file examples/dynamo_integration/react_benchmark_agent/configs/eval_config_rethinking_full_test.yml
```

### Optimization

```bash
# Optimize Dynamo prefix header parameters for the Predictive KV-Aware Thompson Sampling router
# 
# Parameters optimized:
#   - prefix_total_requests: Expected requests per prefix (search space: 1-20, step 5)
#   - prefix_osl: Output Sequence Length hint (LOW | MEDIUM | HIGH)
#   - prefix_iat: Inter-Arrival Time hint (LOW | MEDIUM | HIGH)
#
# Objectives (multi-objective optimization, all minimized):
#   - avg_llm_latency (70% weight) - Primary: reduce LLM response time
#   - avg_workflow_runtime (20% weight) - Secondary: reduce total task time
#   - avg_num_llm_calls (10% weight) - Tertiary: improve efficiency
#
# Uses grid search over the parameter space to find optimal routing hints.
# WARNING: this run could use MANY tokens - be mindful and run at your own risk.
nat optimize --config_file examples/dynamo_integration/react_benchmark_agent/configs/optimize_rethinking_full_test.yml
```

### Profiling

```bash
# Profile with comprehensive LLM and workflow metrics
#
# Metrics collected:
#   - TTFT (Time To First Token) - measures prompt processing latency
#   - ITL (Inter-Token Latency) - measures token generation speed
#   - Throughput (tokens/second) - measures generation efficiency
#   - Token usage patterns and forecasting
#   - Bottleneck analysis with nested call stacks
#   - Concurrency spike detection
#
# Output: standardized_data_all.csv for Pareto optimality analysis
# Use with: python scripts/throughput_analysis.py <output_dir>/standardized_data_all.csv
#
# The Pareto analysis identifies configurations that are optimal trade-offs
# between latency, throughput, and quality (TSQ). No single point dominates
# all others across all objectives - these form the Pareto frontier.
nat profile --config_file examples/dynamo_integration/react_benchmark_agent/configs/profile_rethinking_full_test.yml
```

---

## Next Steps

1. **Start small**: Test with `eval_config_no_rethinking_minimal_test.yml` first
2. **Enable self-evaluation**: Try `eval_config_rethinking_full_test.yml` for quality improvement
3. **Benchmark concurrency**: Run `scripts/run_concurrency_benchmark.sh` to find optimal settings on your machine
4. **Analyze results**: Use throughput analysis scripts to identify bottlenecks
5. **Iterate**: Tune prompts, temperature, and retry settings for your use case
