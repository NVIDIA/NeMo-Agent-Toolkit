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

<!-- path-check-skip-begin -->

# Extending the Finetuning Harness

This guide covers how to create custom components for the NeMo Agent toolkit finetuning harness. You'll learn about the three core interfaces, how to implement them, and best practices for creating robust, reusable components.

## Architecture Overview

The finetuning harness uses three abstract interfaces that you can implement to support any training backend or workflow:

```
┌────────────────────────────────────────────────────────────────────────┐
│                         Your Implementation                            │
│                                                                        │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────┐ │
│  │ TrajectoryBuilder   │  │   TrainerAdapter    │  │     Trainer     │ │
│  │                     │  │                     │  │                 │ │
│  │ Collects episodes   │  │ Bridges to backend  │  │ Orchestrates    │ │
│  │ from workflow runs  │  │ training systems    │  │ the loop        │ │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────┘ │
│           │                        │                       │           │
│           └────────────────────────┼───────────────────────┘           │
│                                    │                                   │
│                     Implements Abstract Interfaces                     │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Core Interfaces                                │
│                                                                         │
│  nat.finetuning.interfaces.trajectory_builder.TrajectoryBuilder         │
│  nat.finetuning.interfaces.trainer_adapter.TrainerAdapter               │
│  nat.finetuning.interfaces.finetuning_runner.Trainer                    │
└─────────────────────────────────────────────────────────────────────────┘
```

Each component has a specific responsibility:

| Component | Responsibility | Key Methods |
|-----------|---------------|-------------|
| **TrajectoryBuilder** | Generate training data from workflow executions | `start_run()`, `finalize()`, `compute_reward()` |
| **TrainerAdapter** | Bridge between NeMo Agent toolkit and external training backends | `submit()`, `status()`, `wait_until_complete()` |
| **Trainer** | Orchestrate the complete finetuning workflow | `run_epoch()`, `run()`, `get_metrics()` |

## The TrajectoryBuilder Interface

The `TrajectoryBuilder` is responsible for generating training data from workflow executions. It runs your workflow on a dataset, collects the conversation history with log probabilities, and computes rewards.

### Interface Definition

```python
from abc import ABC, abstractmethod
from typing import Any

from nat.data_models.finetuning import FinetuneConfig, TrajectoryBuilderConfig, TrajectoryCollection
from nat.eval.config import EvaluationRunOutput
from nat.eval.evaluator.evaluator_model import EvalOutputItem


class TrajectoryBuilder(ABC):
    """Abstract interface for building trajectories from episode items."""

    def __init__(self, trajectory_builder_config: TrajectoryBuilderConfig):
        self.trajectory_builder_config = trajectory_builder_config
        self.run_config: FinetuneConfig = None

    async def initialize(self, run_config: FinetuneConfig) -> None:
        """Initialize resources needed for trajectory building."""
        self.run_config = run_config
        self.trajectory_builder_config.reward = self.run_config.reward_function

    async def run_eval(self) -> EvaluationRunOutput:
        """Run NeMo Agent toolkit Evaluation to generate episode items."""
        # Default implementation uses the evaluation system
        from nat.eval.evaluate import EvaluationRun, EvaluationRunConfig
        # ... runs evaluation and returns output

    @abstractmethod
    async def start_run(self, run_id: str, meta: dict | None = None) -> None:
        """Start trajectory collection for this run."""
        raise NotImplementedError

    @abstractmethod
    async def finalize(self, run_id: str, meta: dict | None = None) -> TrajectoryCollection:
        """Finalize and return the collected trajectories."""
        raise NotImplementedError

    async def compute_reward(self, output_item: EvalOutputItem, meta: dict | None = None) -> float:
        """Compute reward from an evaluation output item."""
        return float(output_item.score) if output_item.score is not None else 0.0

    @abstractmethod
    def log_progress(self, run_id: str, metrics: dict[str, Any], output_dir: str | None = None) -> None:
        """Log trajectory building progress."""
        raise NotImplementedError
```

### Key Concepts

**Evaluation Runs**: The `run_eval()` method leverages the evaluation system to execute the workflow on your dataset. This handles:
- Loading the dataset
- Running the workflow with proper concurrency
- Capturing intermediate steps (including [LLM](../../build-workflows/llms/index.md) calls with logprobs)
- Computing evaluator scores

**Trajectory Parsing**: The `finalize()` method must convert raw intermediate steps into the `Trajectory` format. This involves:
- Extracting conversation messages
- Ensuring assistant messages have log probabilities
- Filtering to target functions/models
- Grouping by example ID

**Reward Computation**: The default `compute_reward()` uses the evaluator score directly. Override this for custom reward shaping.

### Implementing a Custom TrajectoryBuilder

#### Step 1: Define the Configuration

Create a configuration class that inherits from `TrajectoryBuilderConfig`:

```python
from pydantic import Field
from nat.data_models.finetuning import TrajectoryBuilderConfig


class MyTrajectoryBuilderConfig(TrajectoryBuilderConfig, name="my_traj_builder"):
    """Configuration for my custom trajectory builder."""

    num_generations: int = Field(
        default=2,
        ge=1,
        description="Number of trajectory generations per example"
    )

    include_tool_calls: bool = Field(
        default=True,
        description="Whether to include tool call messages in trajectories"
    )

    min_episode_length: int = Field(
        default=2,
        description="Minimum number of messages required for a valid trajectory"
    )
```

The `name="my_traj_builder"` parameter registers this config type so it can be referenced in YAML as `_type: my_traj_builder`.

#### Step 2: Implement the Builder

Implement the `TrajectoryBuilder` interface's methods. 

#### Step 3: Register the Component

Create a registration module:

```python
from nat.builder.builder import Builder
from nat.cli.register_workflow import register_trajectory_builder

from .my_trajectory_builder import MyTrajectoryBuilder, MyTrajectoryBuilderConfig


@register_trajectory_builder(config_type=MyTrajectoryBuilderConfig)
async def my_trajectory_builder(config: MyTrajectoryBuilderConfig, builder: Builder):
    """
    Register the custom trajectory builder.

    Args:
        config: The trajectory builder configuration
        builder: The workflow builder (for accessing other components)

    Yields:
        A configured trajectory builder instance
    """
    yield MyTrajectoryBuilder(trajectory_builder_config=config)
```

## The TrainerAdapter Interface

The `TrainerAdapter` bridges the gap between NeMo Agent toolkit and external training backends. It handles data format conversion, job submission, and status monitoring.

### Interface Definition

```python
from abc import ABC, abstractmethod
from typing import Any

from nat.data_models.finetuning import (
    FinetuneConfig,
    TrainerAdapterConfig,
    TrainingJobRef,
    TrainingJobStatus,
    TrajectoryCollection,
)


class TrainerAdapter(ABC):
    """Adapter to send Trajectories to remote training cluster for weight updates."""

    def __init__(self, adapter_config: TrainerAdapterConfig):
        self.adapter_config = adapter_config
        self.run_config: FinetuneConfig = None

    async def initialize(self, run_config: FinetuneConfig) -> None:
        """Initialize resources needed for the adapter."""
        self.run_config = run_config
        self.adapter_config.reward = self.run_config.reward_function

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check the health of the remote training backend."""
        raise NotImplementedError

    @abstractmethod
    async def submit(self, trajectories: TrajectoryCollection) -> TrainingJobRef:
        """Submit trajectories to the remote training backend."""
        raise NotImplementedError

    @abstractmethod
    async def status(self, ref: TrainingJobRef) -> TrainingJobStatus:
        """Get the status of a submitted training job."""
        raise NotImplementedError

    @abstractmethod
    async def wait_until_complete(self, ref: TrainingJobRef, poll_interval: float = 10.0) -> TrainingJobStatus:
        """Wait until the training job is complete."""
        raise NotImplementedError

    @abstractmethod
    def log_progress(self, ref: TrainingJobRef, metrics: dict[str, Any], output_dir: str | None = None) -> None:
        """Log training adapter progress."""
        raise NotImplementedError
```

### Key Concepts

**Health Checks**: The `is_healthy()` method verifies backend connectivity before attempting training. This catches configuration issues early.

**Data Format Conversion**: The `submit()` method must convert instances of `TrajectoryCollection` to whatever format your backend expects. This is often the most complex part.

**Async Job Management**: Training jobs run asynchronously. The adapter tracks job state and provides methods to query status and wait for completion.

### Implementing a Custom TrainerAdapter

#### Step 1: Define the Configuration

```python
from pydantic import BaseModel, Field
from nat.data_models.finetuning import TrainerAdapterConfig


class MyBackendConfig(BaseModel):
    """Configuration for the training backend."""

    endpoint: str = Field(description="Training API endpoint URL")
    api_key: str = Field(description="API key for authentication")
    timeout: int = Field(default=3600, description="Request timeout in seconds")

    # Training hyperparameters
    learning_rate: float = Field(default=1e-5, description="Learning rate")
    batch_size: int = Field(default=4, description="Training batch size")
    gradient_accumulation_steps: int = Field(default=4, description="Gradient accumulation")


class MyTrainerAdapterConfig(TrainerAdapterConfig, name="my_trainer_adapter"):
    """Configuration for my trainer adapter."""

    backend: MyBackendConfig = Field(description="Backend configuration")

    validate_trajectories: bool = Field(
        default=True,
        description="Whether to validate trajectories before submission"
    )
```

#### Step 2: Implement the Adapter

Implement the `TrainerAdapter` interface's methods.

#### Step 3: Register the Component

```python
from nat.builder.builder import Builder
from nat.cli.register_workflow import register_trainer_adapter

from .my_trainer_adapter import MyTrainerAdapter, MyTrainerAdapterConfig


@register_trainer_adapter(config_type=MyTrainerAdapterConfig)
async def my_trainer_adapter(config: MyTrainerAdapterConfig, builder: Builder):
    """Register the custom trainer adapter."""
    yield MyTrainerAdapter(adapter_config=config)
```

## The Trainer Interface

The `Trainer` orchestrates the complete finetuning workflow, coordinating the trajectory builder and trainer adapter across multiple epochs.

### Interface Definition

```python
from abc import ABC, abstractmethod
from typing import Any

from nat.data_models.finetuning import (
    FinetuneConfig,
    FinetuneRunConfig,
    TrainerConfig,
    TrainingJobRef,
    TrainingJobStatus,
    TrajectoryCollection,
)
from nat.eval.config import EvaluationRunOutput
from nat.finetuning.interfaces.trainer_adapter import TrainerAdapter
from nat.finetuning.interfaces.trajectory_builder import TrajectoryBuilder


class Trainer(ABC):
    """Abstract interface for running finetuning workflows."""

    def __init__(self, trainer_config: TrainerConfig, **kwargs) -> None:
        self.trainer_config = trainer_config
        self.run_config: FinetuneConfig = None
        self.curriculum_config = None
        self.trajectory_builder: TrajectoryBuilder = None
        self.trainer_adapter: TrainerAdapter = None
        self._curriculum_state = None

    async def bind_components(self, trajectory_builder: TrajectoryBuilder, trainer_adapter: TrainerAdapter) -> None:
        """Bind the trajectory builder and trainer adapter."""
        self.trajectory_builder = trajectory_builder
        self.trainer_adapter = trainer_adapter

    async def initialize(self, run_config: FinetuneConfig) -> None:
        """Initialize the trainer and all components."""
        self.run_config = run_config
        self.curriculum_config = run_config.curriculum_learning

        # Initialize curriculum state
        self._curriculum_state = {
            "current_percentile": self.curriculum_config.initial_percentile,
            "last_expansion_epoch": -1,
            "total_groups": 0,
            "included_groups": set()
        }

        # Initialize sub-components
        await self.trajectory_builder.initialize(run_config)
        await self.trainer_adapter.initialize(run_config)

    @abstractmethod
    async def run_epoch(self, epoch: int, run_id: str) -> TrainingJobRef:
        """Run a single epoch of training."""
        raise NotImplementedError

    @abstractmethod
    async def run(self, num_epochs: int) -> list[TrainingJobStatus]:
        """Run the complete finetuning workflow."""
        raise NotImplementedError

    @abstractmethod
    async def get_metrics(self, run_id: str) -> dict[str, Any]:
        """Get training metrics for a run."""
        raise NotImplementedError

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources."""
        raise NotImplementedError

    @abstractmethod
    def log_progress(self, epoch: int, metrics: dict[str, Any], output_dir: str | None = None) -> None:
        """Log training progress."""
        raise NotImplementedError

    async def run_validation_evaluation(self, epoch: int, run_id: str) -> dict[str, Any]:
        """Run evaluation on validation dataset."""
        # Default implementation provided in base class

    def apply_curriculum_learning(self, trajectory_collection: TrajectoryCollection, epoch: int) -> TrajectoryCollection:
        """Apply curriculum learning to filter trajectories."""
        raise NotImplementedError("Override to implement curriculum learning")

    def get_curriculum_state(self) -> dict[str, Any]:
        """Get the current curriculum learning state."""
        # Default implementation provided
```

### Implementing a Custom Trainer

The trainer typically extends the base class and customizes the epoch and run logic. Follow similar steps as before to
define configuration, implement methods, and register the component.

Once you have your `MyTrainer` and `MyTrainerConfig` implemented, register it as follows:

```python
from nat.builder.builder import Builder
from nat.cli.register_workflow import register_trainer

from .my_trainer import MyTrainer, MyTrainerConfig


@register_trainer(config_type=MyTrainerConfig)
async def my_trainer(config: MyTrainerConfig, builder: Builder):
    """Register the custom trainer."""
    yield MyTrainer(trainer_config=config)
```

## Best Practices

### Error Handling

Always handle errors gracefully:

```python
async def run_epoch(self, epoch: int, run_id: str) -> TrainingJobRef | None:
    try:
        # ... epoch logic
    except Exception as e:
        logger.exception("Error in epoch %d", epoch)
        # Return None or raise depending on severity
        raise
```

### Logging

Use structured logging for debugging:

```python
logger.info("Starting epoch %d with %d examples", epoch, num_examples)
logger.debug("Trajectory details: %s", trajectory.metadata)
logger.error("Training failed: %s", error, exc_info=True)
```

### Resource Cleanup

Always implement proper cleanup:

```python
async def cleanup(self) -> None:
    # Cancel pending tasks
    for task in self.pending_tasks.values():
        if not task.done():
            task.cancel()

    # Close connections
    await self.client.aclose()

    # Clear state
    self.pending_tasks.clear()
```

### Testing

Test components in isolation:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_trajectory_builder():
    config = MyTrajectoryBuilderConfig(num_generations=2)
    builder = MyTrajectoryBuilder(trajectory_builder_config=config)

    # Mock the run_eval method
    builder.run_eval = AsyncMock(return_value=mock_eval_output)

    # Test start_run
    await builder.start_run("test-run")
    assert "test-run" in builder.evaluation_runs

    # Test finalize
    result = await builder.finalize("test-run")
    assert isinstance(result, TrajectoryCollection)
```

## Configuration Examples

### Complete YAML Configuration

```yaml
llms:
  my_model:
    _type: openai
    model_name: gpt-4
    base_url: http://localhost:8000/v1

workflow:
  _type: my_agent_workflow
  llm: my_model

eval:
  general:
    max_concurrency: 8
    output_dir: .tmp/finetuning/eval
    dataset:
      _type: json
      file_path: data/train.json

  evaluators:
    my_reward:
      _type: my_custom_evaluator

trajectory_builders:
  my_builder:
    _type: my_traj_builder
    num_generations: 4
    include_tool_calls: true
    min_episode_length: 3

trainer_adapters:
  my_adapter:
    _type: my_trainer_adapter
    backend:
      endpoint: http://training-server:8080
      api_key: ${TRAINING_API_KEY}
      learning_rate: 1e-5
      batch_size: 8
    validate_trajectories: true

trainers:
  my_trainer:
    _type: my_trainer

finetuning:
  enabled: true
  trainer: my_trainer
  trajectory_builder: my_builder
  trainer_adapter: my_adapter
  reward_function:
    name: my_reward
  num_epochs: 20
  output_dir: .tmp/finetuning/output

  curriculum_learning:
    enabled: true
    initial_percentile: 0.3
    increment_percentile: 0.2
    expansion_interval: 5
```

<!-- path-check-skip-end -->

## See Also

- [Finetuning Concepts](../../improve-workflows/finetuning/concepts.md) - Core concepts and architecture
- [OpenPipe ART Integration](../../improve-workflows/finetuning/rl_with_openpipe.md) - Using the ART backend
- [Custom Evaluators](./custom-evaluator.md) - Creating reward functions
