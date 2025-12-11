# NVIDIA NeMo Agent Toolkit and Dynamo Integration

## Overview

**This set of example agents and evaluations demonstrate the capability to integrate NVIDIA NeMo Agent Toolkit (NAT) agents with LLM inference accelerated by NVIDIA Dynamo-hosted LLM endpoints.**

This set of examples is intended to grow over time as the synergies between NAT and [Dynamo](https://github.com/ai-dynamo/dynamo) evolve. In the first set of examples, we will analyze the performance (throughput and latency) of NAT agents requests to Dynamo and seek out key optimizations. Agentic LLM requests have predictable patterns with respect to conversation length, system prompts, and tool-calling. We aim to codesign our inference servers to provide better performance in a repeatable, mock, decision-only evaluation harness. The harness uses the Banking data subset and mock tools from the [Galileo Agent Leaderboard v2](https://huggingface.co/datasets/galileo-ai/agent-leaderboard-v2) benchmark to simulate agentic tool selection quality (TSQ).

Most of these examples can be tested using a managed LLM service, like an NVIDIA NIM model endpoint, for inference. However, the core analysis wwould require hosting the LLM endpoints on your own GPU cluster.


### Key Features

- **Decision-Only Tool Calling**: Tool stubs capture intent without executing banking operations
- **Dynamo Backend**: Fast LLM inference with KV cache optimization (default method) Thompson Sampling router (new implementation)
- **Self-Evaluation Loop**: Agent can re-evaluate and retry tool selection for improved quality
- **Comprehensive Metrics and Visualizations**: TSQ scores, token throughput, latency analysis. Visualized in A/B scatterplots.
- **NAT Framework**: Full integration with NeMo Agent Toolkit evaluators and profiler

## Quick Start

```bash
# 1. Setup environment
cd /path/to/NeMo-Agent-Toolkit
uv venv "${HOME}/.venvs/nat_dynamo_eval" --python 3.13
source "${HOME}/.venvs/nat_dynamo_eval/bin/activate"
uv pip install -e ".[langchain]"
uv pip install matplotlib
uv pip install scipy

# 2. Install the workflow package
cd examples/dynamo_integration/react_benchmark_agent
uv pip install -e .

# 3. Download the dataset (requires HuggingFace account)
cd examples/dynamo_integration
export HF_TOKEN=<your_huggingface_token>
python scripts/download_agent_leaderboard_v2.py --domains banking

# 4. Start Dynamo backend (see Dynamo README for details)
cd /path/to/NeMo-Agent-Toolkit/external/dynamo
bash start_dynamo_unified.sh

# Note: To customize GPU workers and tensor parallelism, edit the configuration
# variables at the top of external/dynamo/start_dynamo_unified.sh:
#   WORKER_GPUS="4,5,6,7"  # GPU device IDs to use (e.g., "0,1" for first 2 GPUs)
#   TP_SIZE=4              # Tensor parallel size (must match number of GPUs)
#   HTTP_PORT=8099         # API endpoint port
#   LOCAL_MODEL_DIR="..."  # Path to your local model weights

# 5. Run evaluation
cd /path/to/NeMo-Agent-Toolkit
nat eval --config_file examples/dynamo_integration/react_benchmark_agent/configs/eval_config_no_rethinking_full_test.yml
```

After running this this end-to-end evaluation, you will have confirmed functional model services on Dynamo, dataset access, and agent execution. From here, we recommend that users visualize their baseline performance via the available scripts:

| Script | Example Usage | Optional Flags | Outcome |
|--------|---------------|----------------|---------|
| `throughput_analysis.py` | `python scripts/throughput_analysis.py ./react_benchmark_agent/outputs/dynamo_evals/jobs/job_*/standardized_data_all.csv` | None | Calculates TTFT, ITL, and tokens/sec statistics from profiler CSV. Outputs: `tokens_per_second_analysis.csv` (per-LLM-call metrics) and `inter_token_latency_distribution.csv` (raw ITL data) |
| `plot_throughput_vs_tsq_per_request.py` | `python scripts/plot_throughput_vs_tsq_per_request.py ./react_benchmark_agent/outputs/dynamo_evals/<experiment_output_dir_name>` | `--output DIR` (if user prefers custom output), `--color-by PARAM` (color by hyperparameter) | Generates scatter plots of TTFT, ITL, throughput vs TSQ scores. Reads from `standardized_data_all.csv` and `tool_selection_quality_output.json`. Supports multi-experiment comparison |
| `run_concurrency_benchmark.sh` | `bash scripts/run_concurrency_benchmark.sh` | Interactive prompts for benchmark name | Runs evaluations at multiple concurrency levels (16, 32) by default. Outputs `benchmark_results.csv`, `benchmark_report.md`, and `analysis_N.txt` for each concurrency experiment |
| `create_test_subset.py` | `python scripts/create_test_subset.py` | `--num-scenarios N` (default: 3), `--input-file PATH`, `--output-file PATH` | Creates smaller dataset subset for quick end-to-end validation testing |


## Documentation

| Document | Description |
|----------|-------------|
| **[Complete Evaluation Guide](react_benchmark_agent/README.md)** | Complete walkthrough: downloading data, running evaluations, analyzing results, self-evaluation loop |
| **[Dynamo Setup](../../external/dynamo/README.md)** | Setting up Dynamo backend, startup scripts, Thompson Sampling router, dynamic prefix headers |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System architecture diagrams, component interactions, data flow |

## Project Structure

```
examples/dynamo_integration/
├── README.md                          # This file
├── ARCHITECTURE.md                    # Architecture diagrams
│
├── scripts/                           # Utility scripts
│   ├── download_agent_leaderboard_v2.py   # Dataset downloader
│   ├── create_test_subset.py              # Test subset generator for quick E2E tests
│   ├── run_concurrency_benchmark.sh       # Throughput benchmarking
│   ├── throughput_analysis.py             # Analyze profiler output
│   └── plot_throughput_vs_tsq_per_request.py  # Generate throughput plots
│
├── data/                              # Datasets (generated by download script)
│   ├── agent_leaderboard_v2_banking.json  # 100 banking scenarios
│   └── raw/banking/tools.json             # 20 banking tool schemas
│
└── react_benchmark_agent/             # Workflow package
    ├── pyproject.toml                 # Package definition
    ├── configs/                       # Configuration files (symlink)
    │   ├── eval_config_no_rethinking_full_test.yml  # Full dataset evaluation
    │   ├── eval_config_no_rethinking_minimal_test.yml # 3-scenario test
    │   ├── eval_config_rethinking_full_test.yml      # Self-evaluation with feedback
    │   ├── profile_predictive_prefix_headers.yml   # Profiler + self-evaluation
    │   ├── optimize_predictive_prefix_headers.yml  # Prefix header optimization
    │   ├── config_dynamo_e2e_test.yml          # Basic Dynamo workflow
    │   └── config_dynamo_prefix_e2e_test.yml  # Dynamo with prefix headers
    │
    ├── src/react_benchmark_agent/     # Source code
    │   ├── register.py                # Component registration
    │   ├── banking_tools.py           # Tool stub registration
    │   ├── tool_intent_stubs.py       # Intent capture system
    │   ├── self_evaluating_agent.py   # Self-evaluation wrapper
    │   └── evaluators/
    │       └── tsq_evaluator.py       # Tool Selection Quality
    │
    ├── tests/                         # Unit tests
    │   ├── test_tsq_formula.py        # TSQ calculation tests
    │   └── test_self_evaluation.py    # Self-evaluation tests
    │
    └── outputs/                       # Evaluation results
        └── dynamo_evals/
            └── <job_id>/
                ├── tool_selection_quality_output.json
                └── standardized_data_all.csv

external/dynamo/                       # Dynamo backend (separate location)
├── README.md                          # Dynamo setup guide
├── start_dynamo_unified.sh            # Start Dynamo (unified mode)
├── start_dynamo_unified_thompson_hints.sh # Start Dynamo with Thompson router
├── start_dynamo_disagg.sh             # Start Dynamo (disaggregated mode)
├── stop_dynamo.sh                     # Stop all Dynamo services
├── test_dynamo_integration.sh         # Integration tests
├── monitor_dynamo.sh                  # Monitor running services
└── generalized/                       # Custom router components
```

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

- Python 3.11, 3.12, or 3.13
- NVIDIA GPU(s) with CUDA support
- Docker (for Dynamo backend)
- NeMo Agent Toolkit with LangChain integration

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
