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

# NVIDIA NeMo Agent Toolkit and Dynamo Integration

**Complexity:** 🛑 Advanced

> [!NOTE]
> ⚠️ **EXPERIMENTAL**: This integration between NeMo Agent Toolkit and Dynamo is experimental and under active development. APIs, configurations, and features may change without notice.

> [!WARNING]
> **This example requires a Linux system with an NVIDIA GPU.** See the [Dynamo Support Matrix](https://docs.nvidia.com/dynamo/archive/0.7.0/reference/support-matrix.html) for full details.
>
> **Supported Platforms:**
> - Ubuntu 22.04 / 24.04 (x86_64)
> - Ubuntu 24.04 (ARM64)
> - CentOS Stream 9 (x86_64, experimental)
>
> **Not Supported:**
> - ❌ macOS (Intel or Apple Silicon)
> - ❌ Windows
>
> You do **not** need to install `ai-dynamo` or `ai-dynamo-runtime` packages locally. The Dynamo server runs inside pre-built Docker images from NGC (`nvcr.io/nvidia/ai-dynamo/sglang-runtime`), which include all necessary components. The NeMo Agent Toolkit Dynamo LLM client (`_type: dynamo`) is a pure HTTP client that works on any platform.

## Overview

> [!IMPORTANT]
> **Prerequisite**: Before running these examples, complete the [Dynamo Backend Setup Guide](../../external/dynamo/README.md) to set up and verify your Dynamo inference server is running and responding to `curl` requests.

**This set of example agents and evaluations demonstrate the capability to integrate NeMo Agent Toolkit agents with LLM inference accelerated by NVIDIA Dynamo-hosted LLM endpoints.**

This set of examples is intended to grow over time as the synergies between NVIDIA NeMo Agent Toolkit and [Dynamo](https://github.com/ai-dynamo/dynamo) evolve. In the first set of examples, we will analyze the performance (throughput and latency) of NeMo Agent Toolkit agents requests to Dynamo and seek out key optimizations. Agentic LLM requests have predictable patterns with respect to conversation length, system prompts, and tool-calling. We aim to co-design our inference servers to provide better performance in a repeatable, mock, decision-only evaluation harness. The harness uses the Banking data subset and mock tools from the [Galileo Agent Leaderboard v2](https://huggingface.co/datasets/galileo-ai/agent-leaderboard-v2) benchmark to simulate agentic tool selection quality (TSQ).

Most of these examples could be tested using a managed LLM service, like an NVIDIA NIM model endpoint, for inference. However, the intended analysis would require hosting the LLM endpoints on your own GPU cluster using Dynamo.


### Key Features

- **Decision-Only Tool Calling**: Tool stubs capture intent without executing banking operations
- **Dynamo Backend**: Fast LLM inference with KV cache optimization (default Dynamo method) and a predictive Thompson sampling router (new implementation)
- **Self-Evaluation Loop**: Agent can re-evaluate and retry tool selection for improved quality.
- **Comprehensive Metrics and Visualizations**: TSQ scores (accuracy of parameters has been excluded), token throughput, latency analysis. Visualized in A/B scatter plots and histograms for analysis.
- **NeMo Agent Toolkit**: Full integration with toolkit evaluators, optimizer, and profiler

## Prerequisites

### Software Requirements

1. **Python 3.11, 3.12, or 3.13** installed
2. **NeMo Agent Toolkit** repository cloned with LangChain integration (`uv pip install -e ".[langchain]"`)
3. **Docker** with NVIDIA Container Toolkit
4. **NVIDIA Driver** with CUDA 12.0+ support, `nvidia-fabricmanager` enabled, and matching your driver version. Verify with:

    ```bash
    docker run --rm --gpus all nvidia/cuda:12.4.0-runtime-ubuntu22.04 \
      bash -c "apt-get update && apt-get install -y python3-pip && pip3 install torch && python3 -c 'import torch; print(torch.cuda.is_available())'"
    ```

    The output should show `True`. If it shows `False` with error 802, ensure `nvidia-fabricmanager` is installed, running, and matches your driver version.

5. **Hugging Face account** with access to Llama-3.3-70B-Instruct model (requires approval from Meta)
6. **Model weights downloaded** - Follow the model download instructions in the [Dynamo Setup Guide](../../external/dynamo/README.md#download-model-weights-can-skip-if-already-done)

### Hardware Requirements (Dynamo Backend)

These experiments are designed to run against a Dynamo backend for LLM inference. The following GPU resources are required:

| Component | Minimum | Recommended |
| --------- | ------- | ----------- |
| **GPU Architecture** | NVIDIA Hopper (H100) | B200 for optimal performance |
| **GPU Count** | 4 GPUs (TP=4 for 70B model) | 8 GPUs for optimal performance |
| **GPU Memory** | 96GB per GPU (H100) | 192GB per GPU (B200) |

> **Note**: The Llama-3.3-70B-Instruct model requires approximately 140GB of GPU memory when loaded with TP=4 (tensor parallelism across 4 GPUs). While it is possible to run evaluations against a managed LLM service (such as NVIDIA NIM), the intended performance analysis requires hosting Dynamo on your own GPU cluster to measure latency, throughput, and KV cache optimization metrics.

See the [Dynamo Setup Guide](../../external/dynamo/README.md) for detailed hardware requirements and configuration options.

## Documentation

| Document | Description |
| -------- | ----------- |
| **[Complete Evaluation Guide](react_benchmark_agent/README.md)** | Complete walkthrough: downloading data, running evaluations, analyzing results, self-evaluation loop |
| **[Dynamo Setup](../../external/dynamo/README.md)** | Setting up Dynamo backend, startup scripts, Thompson Sampling router, dynamic prefix headers |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System architecture diagrams, component interactions, data flow |

## Quick Start

> [!NOTE]
> The instructions below are an **abbreviated quick start**. For detailed environment setup, thorough explanations of each step, configuration options, and troubleshooting guidance, refer to the [Complete Evaluation Guide](react_benchmark_agent/README.md#environment-setup).

<!-- path-check-skip-begin -->
```bash
# 1. Setup environment
cd /path/to/NeMo-Agent-Toolkit
uv venv "${HOME}/.venvs/nat_dynamo_eval" --python 3.13
source "${HOME}/.venvs/nat_dynamo_eval/bin/activate"
uv pip install -e ".[langchain]"
uv pip install matplotlib scipy

# 2. Install the workflow package
cd examples/dynamo_integration/react_benchmark_agent
uv pip install -e .

# 3. Source environment variables
cd ../ # NeMo-Agent-Toolkit/examples/dynamo_integration
cp .env.example .env
vi .env # update the environment variables then source
[ -f .env ] && source .env || { echo "Warning: .env not found" >&2; false; }

# 4. Download the dataset (requires HuggingFace account)
python scripts/download_agent_leaderboard_v2.py --domains banking

# 5. Download the model weights (requires HuggingFace account)
mkdir -p "$(dirname "$DYNAMO_MODEL_DIR")"
hf download meta-llama/Llama-3.3-70B-Instruct --local-dir "$DYNAMO_MODEL_DIR"

# 6. Start Dynamo backend (see Dynamo README for details)
cd "$DYNAMO_REPO_DIR" # cd /path/to/NeMo-Agent-Toolkit/external/dynamo
bash start_dynamo_unified.sh > startup_output.txt 2>&1 # wait ~5 minutes for the server to start

# Requirements for start_dynamo_unified.sh:
#   - Docker with NVIDIA Container Toolkit (nvidia-docker)
#   - 4x NVIDIA GPUs (set WORKER_GPUS to the available set of machines)
#   - Model weights: downloaded per previous instructions
#   - Check that default ports are available: 8099 (HTTP API), 2379 (ETCD), 4222 (NATS)

# 7. Run evaluation
cd ../../ # NeMo-Agent-Toolkit/
nat eval --config_file examples/dynamo_integration/react_benchmark_agent/configs/eval_config_no_rethinking_full_test.yml

# 8. Visualize results (after evaluation completes)
cd examples/dynamo_integration
python scripts/plot_throughput_vs_tsq_per_request.py \
  ./react_benchmark_agent/outputs/dynamo_evals/banking_data_eval_full_test/jobs/
# Generates: ttft_vs_tsq.png, itl_vs_tsq.png, throughput_vs_tsq.png in the jobs/ directory
```
<!-- path-check-skip-end -->

## Performance Comparison

To compare the performance of different configurations or runs, execute multiple evaluation jobs with different settings and then use the comparison script to analyze the results:

```bash
# Run multiple jobs with different configurations for comparison
nat eval --config_file examples/dynamo_integration/react_benchmark_agent/configs/eval_config_no_rethinking_full_test.yml
nat eval --config_file examples/dynamo_integration/react_benchmark_agent/configs/eval_config_rethinking_full_test.yml

# Compare performance across all jobs
python scripts/plot_throughput_vs_tsq_per_request.py \
  <path_to_eval_output_jobs_directory>
```

This script will generate comparative visualizations showing throughput vs. Tool Selection Quality (TSQ) metrics across all jobs in the specified directory, allowing you to analyze the performance differences between different agent configurations.

> [!NOTE]
> **Multi-Backend Comparisons**: Evaluation runs can be performed across multiple Dynamo backend configurations (e.g., different routing strategies, tensor parallelism settings, or hardware configurations) and compared using the same script. Simply run evaluations against different Dynamo deployments and place the results in the same jobs directory for side-by-side analysis.
> [!NOTE]
> To customize GPU workers and tensor parallelism, edit the configuration variables at the top of [start_dynamo_unified.sh](../../external/dynamo/start_dynamo_unified.sh).
> [!WARNING]
> The first load of model weights to `SGLang` workers can take significant time.

After running this end-to-end evaluation, you will have confirmed functional model services on Dynamo, dataset access, and agent execution.

## Quick Stop
<!-- path-check-skip-begin -->
```bash
# 1. When testing is complete don't forget to stop workers and free GPU memory
cd /path/to/NeMo-Agent-Toolkit/external/dynamo # NeMo-Agent-Toolkit/external/dynamo
bash stop_dynamo.sh
```
<!-- path-check-skip-end -->


### Understanding Evaluation Artifacts

The `nat eval` command generates the following artifacts in the job output directory (for example, `outputs/dynamo_evals/banking_data_eval_full_test/jobs/job_<uuid>/`):

| File | Description |
|------|-------------|
| `workflow_output.json` | Raw workflow execution results for each scenario, including generated answers and trajectories |
| `tool_selection_quality_output.json` | TSQ evaluation scores per scenario, with detailed tool-by-tool scoring breakdowns |
| `inference_optimization.json` | Summary statistics for inference performance optimization |
| `standardized_data_all.csv` | Profiler data in CSV format containing per-LLM-call timing metrics (TTFT, ITL, duration, token counts) |
| `all_requests_profiler_traces.json` | Comprehensive profiler traces with full event-level detail for debugging and deep analysis |

### Visualizing Baseline Performance

Use these scripts to analyze and visualize your evaluation results:

<!-- path-check-skip-begin -->
| Script | Example Usage | Optional Flags | Outcome |
|--------|---------------|----------------|---------|
| `throughput_analysis.py` | `python scripts/throughput_analysis.py ./react_benchmark_agent/outputs/dynamo_evals/banking_data_eval_full_test/jobs/job_<uuid>/standardized_data_all.csv` | None | Calculates TTFT, ITL, and tokens-per-second statistics from profiler CSV. Outputs: `tokens_per_second_analysis.csv` and `inter_token_latency_distribution.csv` |
| `plot_throughput_vs_tsq_per_request.py` | `python scripts/plot_throughput_vs_tsq_per_request.py ./react_benchmark_agent/outputs/dynamo_evals/banking_data_eval_full_test/jobs/` | `--output DIR`, `--color-by PARAM` | Generates scatter plots of TTFT, ITL, throughput vs TSQ scores. Pass the `jobs/` directory (not individual job directories). Defaults to multi-experiment comparison. For single experiment, move job to a nested directory. |
| `plot_throughput_histograms_per_request.py` | `python scripts/plot_throughput_histograms_per_request.py ./react_benchmark_agent/outputs/dynamo_evals/banking_data_eval_full_test/jobs/` | `--output DIR` | Generates histograms showing distribution of TTFT, ITL, throughput (100 bins each), plus Total Tokens (50 bins), LLM Calls (25 bins), Duration (25 bins). |
| `run_concurrency_benchmark.sh` | `bash scripts/run_concurrency_benchmark.sh` | Interactive prompts | Runs evaluations at multiple concurrency levels. Outputs `benchmark_results.csv`, `benchmark_report.md`, and `analysis_*.txt` |
| `create_test_subset.py` | `python scripts/create_test_subset.py --num-scenarios 3` | `--input-file PATH`, `--output-file PATH` | Creates smaller dataset subset for quick end-to-end validation testing |
<!-- path-check-skip-end -->

## Project Structure

<!-- path-check-skip-begin -->
```text
examples/dynamo_integration/
├── README.md                          # This file
├── ARCHITECTURE.md                    # Architecture diagrams
│
├── scripts/                           # Utility scripts
│   ├── download_agent_leaderboard_v2.py       # Dataset downloader
│   ├── create_test_subset.py                  # Test subset generator for quick E2E tests
│   ├── run_concurrency_benchmark.sh           # Throughput benchmarking
│   ├── throughput_analysis.py                 # Analyze profiler output
│   ├── plot_throughput_vs_tsq_per_request.py  # Generate throughput vs TSQ plots
│   └── plot_throughput_histograms_per_request.py  # Generate throughput histogram plots
│
├── data/                              # Datasets (generated by download script)
│   ├── agent_leaderboard_v2_all.json      # Full dataset (all domains)
│   ├── agent_leaderboard_v2_banking.json  # 100 banking scenarios
│   └── raw/banking/                       # Raw banking data
│       ├── tools.json                     # 20 banking tool schemas
│       ├── adaptive_tool_use.json         # Adaptive tool usage patterns
│       └── personas.json                  # User persona definitions
│
└── react_benchmark_agent/             # Workflow package
    ├── pyproject.toml                 # Package definition
    ├── README.md                      # Workflow-specific documentation
    ├── configs/                       # Configuration files (symlink)
    │   ├── eval_config_no_rethinking_full_test.yml    # Full dataset evaluation
    │   ├── eval_config_no_rethinking_minimal_test.yml # 3-scenario test
    │   ├── eval_config_rethinking_full_test.yml       # Self-evaluation with feedback
    │   ├── profile_rethinking_full_test.yml           # Profiler + self-evaluation
    │   ├── optimize_rethinking_full_test.yml          # Prefix header optimization
    │   ├── config_dynamo_e2e_test.yml                 # Basic Dynamo workflow
    │   ├── config_dynamo_prefix_e2e_test.yml          # Dynamo with prefix headers
    │   └── config_dynamo_adk_e2e_test.yml             # Dynamo with ADK integration
    │
    ├── src/react_benchmark_agent/     # Source code
    │   ├── __init__.py                # Package initialization
    │   ├── register.py                # Component registration
    │   ├── react_benchmark_agent.py   # Main benchmark agent implementation
    │   ├── banking_tools.py           # Tool stub registration
    │   ├── tool_intent_stubs.py       # Intent capture system
    │   ├── self_evaluating_agent_with_feedback.py  # Self-evaluation wrapper
    │   └── evaluators/
    │       ├── __init__.py            # Evaluators package
    │       ├── tsq_evaluator.py       # Tool Selection Quality evaluator
    │       └── action_completion_evaluator.py  # Action completion evaluator
    │
    ├── tests/                         # Unit tests
    │   ├── test_tsq_formula.py        # TSQ calculation tests
    │   ├── test_self_evaluation.py    # Self-evaluation tests
    │   └── test_tool_intent_buffer.py # Tool intent buffer tests
    │
    └── outputs/                       # Evaluation results (generated at runtime)
        ├── benchmarks/                # Concurrency benchmark results
        │   └── <benchmark_run>/
        │       ├── benchmark_report.md
        │       └── benchmark_results.csv
        └── dynamo_evals/
            └── <eval_name>/jobs/<job_id>/
                ├── tool_selection_quality_output.json
                ├── standardized_data_all.csv
                ├── all_requests_profiler_traces.json
                ├── inference_optimization.json
                ├── workflow_output.json
                ├── inter_token_latency_distribution.csv
                └── tokens_per_second_analysis.csv

external/dynamo/                       # Dynamo backend (separate location)
├── README.md                          # Dynamo setup guide
├── start_dynamo_unified.sh            # Start Dynamo (unified mode)
├── start_dynamo_unified_thompson_hints.sh # Start Dynamo with Thompson router
├── start_dynamo_disagg.sh             # Start Dynamo (disaggregated mode)
├── stop_dynamo.sh                     # Stop all Dynamo services
├── test_dynamo_integration.sh         # Integration tests
├── monitor_dynamo.sh                  # Monitor running services
└── generalized/                       # Custom router components
    ├── frontend.py                    # Frontend request handler
    ├── processor.py                   # Request processor
    └── router.py                      # Routing logic
```
<!-- path-check-skip-end -->

## Basic Configuration Options

### Basic Evaluation (No Self-Evaluation)
```yaml
workflow:
  _type: react_agent
  llm_name: dynamo_llm
  tool_names: [banking_tools.get_account_balance, ...]
```

### With Self-Evaluation Loop
```yaml
workflow:
  _type: self_evaluating_agent_with_feedback
  wrapped_agent: react_workflow
  evaluator_llm: eval_llm
  max_retries: 5
  min_confidence_threshold: 0.85
  pass_feedback_to_agent: true
```

See [Evaluation Guide](react_benchmark_agent/README.md) for complete configuration documentation.

## Metrics

| Metric | Description |
|--------|-------------|
| **TSQ (Tool Selection Quality)** | F1 score comparing actual vs expected tool calls |
| **TTFT (Time To First Token)** | Latency before first token arrives |
| **ITL (Inter-Token Latency)** | Time between consecutive tokens |
| **Throughput** | Tokens per second (aggregate and per-request) |

## Troubleshooting and Support

For troubleshooting common issues, refer to the [Complete Evaluation Guide - Troubleshooting](react_benchmark_agent/README.md#troubleshooting) section, which covers:

- Permission denied errors when downloading datasets
- Tools not executing (hallucinated observations)
- TSQ score always returning 0.0
- Module not found errors
- File path resolution issues
- Recursion limit errors
- Self-evaluation configuration issues
- Dynamo connection errors

For Dynamo-specific issues, see the [Dynamo Setup Guide - Troubleshooting](../../external/dynamo/README.md#troubleshooting) section.

---
