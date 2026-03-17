<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# nvidia-nat-config-optimizer

Workflow config and prompt optimization for [NVIDIA NeMo Agent Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit). Provides genetic-algorithm and numeric (Optuna) optimizers for workflow configs and prompts. Scoped to config-level optimization (hyperparameters, prompts); excludes runtime/inference optimizations.

Install with NeMo Agent Toolkit: `pip install nvidia-nat[config-optimizer]` or `pip install nvidia-nat-core nvidia-nat-config-optimizer`.

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
