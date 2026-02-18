# nvidia-nat-optimizer

Prompt and parameter optimization for [NVIDIA NeMo Agent Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit). Provides genetic-algorithm and numeric (Optuna) optimizers for workflow configs and prompts.

Install with NAT: `pip install nvidia-nat` (optimizer is included by default), or `pip install nvidia-nat-core nvidia-nat-optimizer`.

Optimizer-only (minimal deps): `pip install nvidia-nat-optimizer` (requires `nvidia-nat-core` for eval contracts).

## Development / testing

From **repo root** (install test deps, then run optimizer tests):

```bash
uv sync --extra test
uv run pytest packages/nvidia_nat_optimizer/tests/ -v
```

For Pareto/visualization tests (matplotlib), install the optimizer with the visualization extra first:

```bash
cd packages/nvidia_nat_optimizer && uv sync --extra test --extra visualization && uv run pytest tests/ -v
```

Or run the full repo test suite (all packages): `python ci/scripts/run_tests.py`
