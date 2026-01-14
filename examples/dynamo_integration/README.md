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

> [!NOTE]
> ⚠️ **EXPERIMENTAL**: This integration between NeMo Agent toolkit and Dynamo is experimental and under active development. APIs, configurations, and features may change without notice.

## Overview

**This set of example agents and evaluations demonstrate the capability to integrate NeMo Agent toolkit agents with LLM inference accelerated by NVIDIA Dynamo-hosted LLM endpoints.**

This set of examples is intended to grow over time as the synergies between NeMo Agent toolkit and [Dynamo](https://github.com/ai-dynamo/dynamo) evolve. In the first set of examples, we will analyze the performance (throughput and latency) of NeMo Agent toolkit agents requests to Dynamo and seek out key optimizations. Agentic LLM requests have predictable patterns with respect to conversation length, system prompts, and tool-calling. We aim to co-design our inference servers to provide better performance in a repeatable, mock, decision-only evaluation harness. The harness uses the Banking data subset and mock tools from the [Galileo Agent Leaderboard v2](https://huggingface.co/datasets/galileo-ai/agent-leaderboard-v2) benchmark to simulate agentic tool selection quality (TSQ).

Most of these examples could be tested using a managed LLM service, like an NVIDIA NIM model endpoint, for inference. However, the intended analysis would require hosting the LLM endpoints on your own GPU cluster using Dynamo.


### Key Features

- **Decision-Only Tool Calling**: Tool stubs capture intent without executing banking operations
- **Dynamo Backend**: Fast LLM inference with KV cache optimization (default Dynamo method) and a predictive Thompson sampling router (new implementation)
- **Self-Evaluation Loop**: Agent can re-evaluate and retry tool selection for improved quality.
- **Comprehensive Metrics and Visualizations**: TSQ scores (accuracy of parameters has been excluded), token throughput, latency analysis. Visualized in A/B scatter plots and histograms for analysis.
- **NeMo Agent toolkit Framework**: Full integration with NeMo Agent toolkit evaluators, optimizer, and profiler

## Quick Start

```bash
# 1. Setup environment
cd /path/to/NeMo-Agent-Toolkit
uv venv "${HOME}/.venvs/nat_dynamo_eval" --python 3.13
source "${HOME}/.venvs/nat_dynamo_eval/bin/activate"
uv pip install -e ".[langchain]"
uv pip install matplotlib scipy


# 2. Install the workflow package
# <!-- path-check-skip-next-line -->
cd examples/dynamo_integration/react_benchmark_agent # NeMo-Agent-Toolkit/examples/dynamo_integration/react_benchmark_agent
uv pip install -e .

# 3. Download the dataset (requires HuggingFace account)
# <!-- path-check-skip-next-line -->
cd ../ # NeMo-Agent-Toolkit/examples/dynamo_integration
export HF_TOKEN=<your_huggingface_token>
export HF_HOME=<your-user-path/.cache/huggingface>
python scripts/download_agent_leaderboard_v2.py --domains banking

# 4. Start Dynamo backend (see Dynamo README for details)
# <!-- path-check-skip-next-line -->
cd ../../external/dynamo # NeMo-Agent-Toolkit/external/dynamo
bash start_dynamo_unified.sh # wait ~5 minutes for the server to start

# Requirements for start_dynamo_unified.sh:
#   - Docker with NVIDIA Container Toolkit (nvidia-docker)
#   - 4x NVIDIA GPUs (default: device IDs 4,5,6,7, configurable via WORKER_GPUS)
#   - Model weights: either local path or HF_TOKEN to download Llama-3.3-70B-Instruct
#   - Ports available: 8099 (HTTP API), 2389 (ETCD), 4232 (NATS)
#   - curl and jq for health checks

# Note: To customize GPU workers and tensor parallelism, edit the configuration
# variables at the top of external/dynamo/start_dynamo_unified.sh:
#   WORKER_GPUS="4,5,6,7"  # GPU device IDs to use (e.g., "0,1" for first 2 GPUs)
#   TP_SIZE=4              # Tensor parallel size (must match number of GPUs)
#   HTTP_PORT=8099         # API endpoint port
#   LOCAL_MODEL_DIR="..."  # Path to your local model weights

# 5. Run evaluation
cd ../../ # NeMo-Agent-Toolkit/
nat eval --config_file examples/dynamo_integration/react_benchmark_agent/configs/eval_config_no_rethinking_full_test.yml
```

After running this end-to-end evaluation, you will have confirmed functional model services on Dynamo, dataset access, and agent execution.

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

## Documentation

| Document | Description |
|----------|-------------|
| **[Complete Evaluation Guide](react_benchmark_agent/README.md)** | Complete walkthrough: downloading data, running evaluations, analyzing results, self-evaluation loop |
| **[Dynamo Setup](../../external/dynamo/README.md)** | Setting up Dynamo backend, startup scripts, Thompson Sampling router, dynamic prefix headers |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System architecture diagrams, component interactions, data flow |

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
    │   ├── prefix_utils.py            # Prefix header utilities
    │   └── evaluators/
    │       ├── __init__.py            # Evaluators package
    │       ├── tsq_evaluator.py       # Tool Selection Quality evaluator
    │       └── action_completion_evaluator.py  # Action completion evaluator
    │
    ├── tests/                         # Unit tests
    │   ├── test_tsq_formula.py        # TSQ calculation tests
    │   ├── test_self_evaluation.py    # Self-evaluation tests
    │   ├── test_prefix_utils.py       # Prefix utilities tests
    │   └── validate_prefix_config.py  # Prefix configuration validation
    │
    └── outputs/                       # Evaluation results
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

## Configuration Options

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

## Requirements

### Software Requirements

- **Python 3.11, 3.12, or 3.13**
- **Docker**
- **NeMo Agent toolkit** with LangChain integration (`uv pip install -e ".[langchain]"`)
- **Hugging Face account** with access to Llama-3.3-70B-Instruct model (for dataset download and model weights)

### Hardware Requirements (Dynamo Backend)

These experiments are designed to run against a Dynamo backend for LLM inference. The following GPU resources are required:

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU Architecture** | NVIDIA Hopper (H100) or Blackwell (B200) | B200 for optimal performance |
| **GPU Count** | 4 GPUs (TP=4 for 70B model) | 8 GPUs for optimal performance |
| **GPU Memory** | 80GB per GPU (H100) | 192GB per GPU (B200) |

> **Important**: The Llama-3.3-70B-Instruct model requires approximately 140GB of GPU memory when loaded with TP=4. While it is possible to run evaluations against a managed LLM service (such as NVIDIA NIM), the intended performance analysis requires hosting Dynamo on your own GPU cluster to measure latency, throughput, and KV cache optimization metrics.

See the [Dynamo Setup Guide](../../external/dynamo/README.md) for detailed hardware requirements and configuration options

## Troubleshooting

### Permission Denied Downloading Dataset

If you see `PermissionError: [Errno 13] Permission denied` when downloading the dataset, your home directory may be on NFS which doesn't support file locking. Set `HF_HOME` to a local writable directory:

```bash
export HF_HOME=/path/to/local/storage/.cache/huggingface
export HF_TOKEN=<my_huggingface_read_token>
```

## Support

For issues:
1. Check [Dynamo Setup Guide](../../external/dynamo/README.md) troubleshooting section
2. Review logs in `react_benchmark_agent/outputs/dynamo_evals/<job_id>/`
3. Verify Dynamo health: `curl http://localhost:8099/health`

---
