<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Adding a Custom Optimizer

:::{note}
We recommend reading the [Optimizer](../../improve-workflows/optimizer.md) guide before proceeding with this documentation.
:::

NeMo Agent Toolkit provides a pluggable optimizer system for tuning workflow parameters and prompts. The built-in strategies include Optuna-based numeric optimization and a genetic algorithm (GA) for prompt optimization. You can add custom optimization strategies by implementing one of the optimizer base classes and registering it with the `@register_optimizer` decorator.

## Key Interfaces

* **Configuration Base Classes**
   - {py:class}`~nat.data_models.optimizer.OptimizerStrategyBaseConfig`: Base class that all optimizer strategy configuration models must extend. Provides an `enabled` field and integrates with the NeMo Agent Toolkit type registry.
   - {py:class}`~nat.data_models.optimizer.PromptOptimizationConfig`: Base for prompt optimization strategy configuration models. Adds `prompt_population_init_function` and `prompt_recombination_function` fields.
   - {py:class}`~nat.data_models.optimizer.OptunaParameterOptimizationConfig`: Built-in config for Optuna-based numeric/enum parameter optimization.

* **Optimizer ABCs**
   - {py:class}`~nat.plugins.config_optimizer.prompts.base.BasePromptOptimizer`: Abstract base class for prompt optimization strategies. Requires implementing an async `run()` method that persists optimized prompts to disk; the in-memory Config is left unchanged.
   - {py:class}`~nat.plugins.config_optimizer.parameters.base.BaseParameterOptimizer`: Abstract base class for parameter optimization strategies. Requires implementing an async `run()` method that returns an optimized `Config`.

* **Registration**
   - {py:deco}`~nat.cli.register_workflow.register_optimizer`: Decorator that registers an optimizer strategy with the global type registry so the optimizer runtime can resolve the strategy from the type of `cfg.optimizer.numeric` or `cfg.optimizer.prompt`.

## Adding a Custom Prompt Optimizer

### 1. Define a config class

Create a config class extending {py:class}`~nat.data_models.optimizer.PromptOptimizationConfig` with a unique `name`:

```python
from pydantic import Field

from nat.data_models.optimizer import PromptOptimizationConfig


class IterativeRefinementPromptConfig(PromptOptimizationConfig, name="iterative"):
    max_iterations: int = Field(default=20, description="Maximum refinement iterations.")
    candidates_per_iteration: int = Field(default=5, description="Number of candidate prompts to generate per iteration.")
    improvement_threshold: float = Field(default=0.01, description="Minimum score improvement to continue iterating.")
```

### 2. Implement the Optimizer

Implement {py:class}`~nat.plugins.config_optimizer.prompts.base.BasePromptOptimizer`:

```python
from nat.plugins.config_optimizer.prompts.base import BasePromptOptimizer
from nat.data_models.config import Config
from nat.data_models.optimizable import SearchSpace
from nat.data_models.optimizer import OptimizerConfig, OptimizerRunConfig


class IterativeRefinementPromptOptimizer(BasePromptOptimizer):

    async def run(
        self,
        *,
        base_cfg: Config,
        full_space: dict[str, SearchSpace],
        optimizer_config: OptimizerConfig,
        opt_run_config: OptimizerRunConfig,
    ) -> None:
        ir_config = optimizer_config.prompt  # Your IterativeRefinementPromptConfig instance

        # Extract prompt parameters from full_space
        prompt_space = {k: v for k, v in full_space.items() if v.is_prompt}
        if not prompt_space:
            return

        # Implement your optimization loop here
        # Use ir_config.max_iterations, ir_config.candidates_per_iteration, etc.
        ...
```

The `run()` method receives:
- `base_cfg`: The workflow configuration to optimize.
- `full_space`: A dictionary of parameter names to {py:class}`~nat.data_models.optimizable.SearchSpace` definitions. Filter for `is_prompt=True` entries to find prompt parameters.
- `optimizer_config`: The full {py:class}`~nat.data_models.optimizer.OptimizerConfig`. Access your strategy config via `optimizer_config.prompt`.
- `opt_run_config`: Runtime parameters including dataset path, endpoint, and result JSON path.

### 3. Register the Optimizer

Use the {py:deco}`~nat.cli.register_workflow.register_optimizer` decorator to register your strategy:

```python
from nat.cli.register_workflow import register_optimizer


@register_optimizer(config_type=IterativeRefinementPromptConfig)
async def register_iterative_prompt_optimizer(config: IterativeRefinementPromptConfig):
    yield IterativeRefinementPromptOptimizer()
```

### 4. Import for Discovery

Import the registration function in your project's `register.py` to ensure it runs at startup:

<!-- path-check-skip-next-line -->
```python
from . import iterative_prompt_optimizer  # noqa: F401 — triggers @register_optimizer
```

### 5. Configure Programmatically

Custom strategy selection for `optimizer.prompt` is currently programmatic. After loading your workflow config, set `cfg.optimizer.prompt` to your custom config before calling `optimize_config`:

```python
from nat.plugins.config_optimizer.optimizer_runtime import optimize_config
from nat.data_models.optimizer import OptimizerRunConfig
from nat.runtime.loader import load_config

cfg = load_config("workflow.yml")
cfg.optimizer.prompt = IterativeRefinementPromptConfig(
    enabled=True,
    max_iterations=200,
    candidates_per_iteration=10,
    improvement_threshold=0.01,
    prompt_population_init_function="my_init_fn",
)

await optimize_config(
    OptimizerRunConfig(
        config_file=cfg,
        dataset="dataset.json",
        result_json_path="$",
    )
)
```

## Adding a Custom Parameter Optimizer

The pattern is the same, but parameter optimizers extend {py:class}`~nat.plugins.config_optimizer.parameters.base.BaseParameterOptimizer` and return an optimized {py:class}`~nat.data_models.config.Config`:

### 1. Define a config class

```python
from pydantic import Field

from nat.data_models.optimizer import OptimizerStrategyBaseConfig


class RandomSearchConfig(OptimizerStrategyBaseConfig, name="random_search"):
    n_samples: int = Field(default=50, description="Number of random samples to evaluate.")
```

### 2. Implement the Optimizer

```python
from nat.plugins.config_optimizer.parameters.base import BaseParameterOptimizer
from nat.data_models.config import Config
from nat.data_models.optimizable import SearchSpace
from nat.data_models.optimizer import OptimizerConfig, OptimizerRunConfig


class RandomSearchOptimizer(BaseParameterOptimizer):

    async def run(
        self,
        *,
        base_cfg: Config,
        full_space: dict[str, SearchSpace],
        optimizer_config: OptimizerConfig,
        opt_run_config: OptimizerRunConfig,
    ) -> Config:
        rs_config = optimizer_config.numeric  # Your RandomSearchConfig instance

        # Filter out prompt parameters
        param_space = {k: v for k, v in full_space.items() if not v.is_prompt}
        if not param_space:
            return base_cfg

        # Implement random search logic here
        # Return the best config found
        ...
        return best_cfg
```

### 3. Register and Configure

```python
from nat.cli.register_workflow import register_optimizer


@register_optimizer(config_type=RandomSearchConfig)
async def register_random_search(config: RandomSearchConfig):
    yield RandomSearchOptimizer()
```

Custom strategy selection for `optimizer.numeric` is also programmatic:

```python
from nat.plugins.config_optimizer.optimizer_runtime import optimize_config
from nat.data_models.optimizer import OptimizerRunConfig
from nat.runtime.loader import load_config

cfg = load_config("workflow.yml")
cfg.optimizer.numeric = RandomSearchConfig(enabled=True, n_samples=100)

await optimize_config(
    OptimizerRunConfig(
        config_file=cfg,
        dataset="dataset.json",
        result_json_path="$",
    )
)
```
