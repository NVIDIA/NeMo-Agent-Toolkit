# nvidia-nat-config-optimizer

Workflow config and prompt optimization for [NVIDIA NeMo Agent Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit). Provides genetic-algorithm and numeric (Optuna) optimizers for workflow configs and prompts. Scoped to config-level optimization (hyperparameters, prompts); excludes runtime/inference optimizations.

Install with NAT: `pip install nvidia-nat` (config optimizer is included by default), or `pip install nvidia-nat-core nvidia-nat-config-optimizer`.

Config-optimizer-only (minimal deps): `pip install nvidia-nat-config-optimizer` (requires `nvidia-nat-core` for eval contracts).

## Development / testing

From **repo root** (install test deps, then run optimizer tests):

```bash
uv sync --extra test
uv run pytest packages/nvidia_nat_config_optimizer/tests/ -v
```

For Pareto/visualization tests (matplotlib), install the optimizer with the visualization extra first:

```bash
cd packages/nvidia_nat_config_optimizer && uv sync --extra test --extra visualization && uv run pytest tests/ -v
```

Or run the full repo test suite (all packages): `python ci/scripts/run_tests.py`
