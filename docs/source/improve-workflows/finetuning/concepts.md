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

# Finetuning Harness: Concepts and Architecture

:::{warning}
**Experimental Feature**: The Finetuning Harness is experimental and may change in future releases. Future versions may introduce breaking changes without notice.
:::

The NeMo Agent Toolkit provides a powerful finetuning harness designed for **in-situ reinforcement learning** of agentic [LLM](../../build-workflows/llms/index.md) workflows. This guide introduces the foundational concepts, explains the design philosophy, and provides the background knowledge needed to effectively use the harness.

## What is Finetuning?

**Finetuning** is the process of taking a pre-trained language model and further training it on a specific task or domain. Unlike training from scratch, finetuning leverages the knowledge the model already has and adapts it for your particular use case.

There are several approaches to finetuning:

| Approach | Description | Use Case |
|----------|-------------|----------|
| **Supervised Fine-Tuning (SFT)** | Train on input-output pairs with known correct answers | When you have labeled examples of desired behavior |
| **Reinforcement Learning (RL)** | Train based on reward signals from outcomes | When you can evaluate quality but don't have "correct" answers |
| **Direct Preference Optimization (DPO)** | Train on pairs of preferred vs. rejected outputs | When you have human preference data |
| **RLHF** | RL guided by a learned reward model from human feedback | Complex alignment tasks |

The finetuning harness is designed primarily for **reinforcement learning** approaches, where agents learn through trial and error based on reward signals.

## Reinforcement Learning Fundamentals

To understand the finetuning harness, you need to understand core RL concepts. This section explains them in the context of LLM agents.

### The RL Framework

Reinforcement learning is a paradigm where an **agent** learns to make decisions by interacting with an **environment** and receiving **rewards**.

```
┌─────────────────────────────────────────────────────────────────┐
│                    The RL Loop                                  │
│                                                                 │
│    ┌─────────┐    action     ┌─────────────┐                    │
│    │  Agent  │ ───────────►  │ Environment │                    │
│    │  (LLM)  │               │  (Task/API) │                    │
│    └─────────┘  ◄─────────── └─────────────┘                    │
│         ▲       state, reward                                   │
│         │                                                       │
│         └──── Agent updates policy based on rewards             │
└─────────────────────────────────────────────────────────────────┘
```

In the context of LLM agents:

- **Agent**: The language model making decisions (generating text, calling tools, etc.)
- **Environment**: The task, tools, APIs, or simulated world the agent interacts with
- **State**: The current context (conversation history, tool outputs, etc.)
- **Action**: The agent's response (generated text, tool call, decision)
- **Reward**: A numerical signal indicating how well the agent performed

### Policy

A **policy** is the agent's strategy for choosing actions given a state. For LLMs, the policy is essentially the model's probability distribution over possible next tokens given the conversation history.

When we finetune an LLM with RL, we're adjusting its policy to favor actions that lead to higher rewards.

### Episodes and Trajectories

An **episode** is a complete interaction from start to finish. In a conversational agent, an episode might be:

1. User asks a question
2. Agent thinks and calls tools
3. Agent receives tool results
4. Agent formulates a response
5. User provides feedback or the task completes

A **trajectory** (also called a **rollout**) is the recorded sequence of everything that happened during an episode:

```
Trajectory = [State₀, Action₀, Reward₀, State₁, Action₁, Reward₁, ..., StateₙAction, ₙ, Rewardₙ]
```

For LLM agents, a trajectory typically looks like:

```python
trajectory = [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "What's the weather in Paris?"},
    {"role": "assistant", "content": "<tool_call>get_weather('Paris')</tool_call>"},
    {"role": "tool", "content": "Sunny, 22°C"},
    {"role": "assistant", "content": "The weather in Paris is sunny at 22°C."},
]
# Final reward: 1.0 (correct answer)
```

:::{note}
**Trajectory vs. Rollout**: These terms are often used interchangeably. "Rollout" emphasizes the process of generating the sequence (rolling out the policy), while "trajectory" emphasizes the recorded data. In NeMo Agent toolkit, we use "trajectory" for the data structure.
:::

### Rewards and Returns

A **reward** is the immediate feedback signal after an action. Rewards can be:

- **Sparse**: Only given at the end (e.g., task success = 1, failure = 0)
- **Dense**: Given at each step (e.g., partial credit for intermediate progress)

The **return** is the total accumulated reward over an episode, often with discounting:

```
Return = R₀ + γR₁ + γ²R₂ + ... + γⁿRₙ
```

Where γ (gamma) is the **discount factor** (typically 0.9-0.99). Discounting means:
- Immediate rewards are worth more than future rewards
- Prevents infinite returns in continuing tasks
- Encourages efficient solutions

### Credit Assignment

One of the hardest problems in RL is **credit assignment**: figuring out which actions were responsible for the final outcome.

If your agent had a 10-step conversation and got a reward at the end, which of those 10 steps were good? Which were bad? This is particularly challenging for LLM agents with long conversations.

Common approaches:

1. **Outcome-based**: Assign the same reward to all steps (simple but noisy)
2. **Reward shaping**: Provide intermediate rewards for good behaviors
3. **Advantage estimation**: Use value functions to estimate which actions were better than expected

The harness supports reward shaping through intermediate step metadata, allowing you to record step-quality signals during execution.

### On-Policy vs. Off-Policy Learning

- **On-policy**: The agent learns from trajectories generated by its current policy. The data must be "fresh" because old trajectories were generated by a different policy.

- **Off-policy**: The agent can learn from trajectories generated by any policy, including old versions or even other agents.

Most modern LLM RL methods (like GRPO, PPO) are **on-policy**, meaning you need to regenerate trajectories after each training update. This is why the harness runs evaluation (to collect trajectories) at the start of each epoch.

## Key RL Algorithms for LLMs

### GRPO (Group Relative Policy Optimization)

**GRPO** is the algorithm used by [OpenPipe ART](https://art.openpipe.ai/). Instead of comparing actions to a baseline value function, GRPO compares multiple responses to the same prompt:

```
Given prompt P, generate N responses: [R₁, R₂, ..., Rₙ]
Score each response: [S₁, S₂, ..., Sₙ]
Learn to increase probability of high-scoring responses
Learn to decrease probability of low-scoring responses
```

This is why the harness groups trajectories by example ID—each group contains multiple generations for the same input, enabling GRPO optimization.

**Advantages of GRPO**:
- No need to train a separate value function
- More stable than PPO for language tasks
- Natural fit for LLM generation (sample multiple completions)

### PPO (Proximal Policy Optimization)

**PPO** is a popular RL algorithm that constrains policy updates to prevent large changes:

1. Collect trajectories with current policy
2. Compute advantages (how much better/worse than expected)
3. Update policy, but clip updates to stay close to the old policy
4. Repeat

PPO requires a **value function** (critic) that estimates expected returns, adding complexity compared to GRPO.

### DPO (Direct Preference Optimization)

**DPO** sidesteps RL entirely by treating preference learning as a classification problem:

1. Given pairs of (preferred, rejected) responses
2. Train the model to increase probability of preferred response
3. Simultaneously decrease probability of rejected response

DPO is simpler than RL methods but requires preference data rather than reward signals.

## Curriculum Learning

**Curriculum learning** is a training strategy inspired by how humans learn: starting with easy examples and gradually introducing harder ones.

### Why Curriculum Learning?

Without curriculum learning, your model trains on all examples equally. This can cause problems:

1. **Easy examples dominate**: If 90% of examples are easy, the model focuses on those
2. **Hard examples cause instability**: Difficult examples with high variance can destabilize training
3. **Inefficient learning**: Time spent on already-mastered examples is wasted

### How Curriculum Learning Works

```
Epoch 1-5:   Train on easiest 30% of examples
Epoch 6-10:  Train on easiest 50% of examples
Epoch 11-15: Train on easiest 70% of examples
Epoch 16+:   Train on all examples
```

The harness determines difficulty by the average reward achieved on each example group. Examples where the model already performs well are "easy"; examples where it struggles are "hard."

### Curriculum Learning Configuration

```yaml
finetuning:
  curriculum_learning:
    enabled: true
    initial_percentile: 0.3      # Start with easiest 30%
    increment_percentile: 0.2     # Add 20% more each expansion
    expansion_interval: 5         # Expand every 5 epochs
    min_reward_diff: 0.1         # Skip groups with no variance
    sort_ascending: false         # false = easy-to-hard
```

**Key parameters**:

| Parameter | Description |
|-----------|-------------|
| `initial_percentile` | Fraction of examples to start with (0.0-1.0) |
| `increment_percentile` | How much to add at each expansion |
| `expansion_interval` | Epochs between expansions |
| `min_reward_diff` | Minimum reward variance to include a group |
| `sort_ascending` | `true` for hard-to-easy, `false` for easy-to-hard |

### Filtering Low-Variance Groups

The `min_reward_diff` parameter is crucial. If all trajectories for an example have the same reward, there's no learning signal—the model can't learn what's better or worse.

```
Example A: Trajectories with rewards [0.8, 0.9, 0.7, 0.85]
  → Variance exists, model can learn to prefer 0.9 trajectory

Example B: Trajectories with rewards [1.0, 1.0, 1.0, 1.0]
  → No variance, all trajectories equally good, no learning signal
  → Filtered out if reward_diff < min_reward_diff
```

## Log Probabilities

**Log probabilities** (logprobs) are essential for policy gradient methods. When the model generates a token, it assigns probabilities to all possible tokens. The logprob is the log of that probability.

### Why Log Probabilities Matter

Policy gradient methods update the model by:

1. Looking at what the model generated
2. Checking the probability it assigned to that generation
3. Increasing/decreasing that probability based on reward

Without logprobs, we can't compute this gradient. This is why:
- The harness requires logprobs for assistant messages
- Your LLM inference endpoint must return logprobs
- Trajectories without logprobs are filtered out

### Enabling Log Probabilities

For OpenAI-compatible APIs:
```python
response = client.chat.completions.create(
    model="your-model",
    messages=messages,
    logprobs=True,          # Enable logprobs
    top_logprobs=5          # How many alternative tokens to return
)
```

For vLLM:
```bash
# Start vLLM with logprobs enabled
python -m vllm.entrypoints.openai.api_server \
    --model your-model \
    --enable-log-probs
```

## Design Philosophy

The finetuning harness is built on three foundational principles:

### 1. Decoupled Architecture

The harness is intentionally **decoupled from training backends and optimization algorithms**. This separation allows:

- **Backend Flexibility**: Train with any RL backend (OpenPipe ART, NeMo Aligner, custom implementations)
- **Algorithm Agnosticism**: Support GRPO, PPO, DPO, or SFT without code changes
- **Infrastructure Independence**: Run locally, on cloud GPUs, or across distributed clusters

The decoupling is achieved through abstract interfaces that define *what* needs to happen, not *how*:

```python
# Interface defines the contract
class TrainerAdapter(ABC):
    async def submit(self, trajectories: TrajectoryCollection) -> TrainingJobRef:
        """Submit trajectories for training."""
        raise NotImplementedError

# Implementation handles the specifics
class ARTTrainerAdapter(TrainerAdapter):
    async def submit(self, trajectories: TrajectoryCollection) -> TrainingJobRef:
        # Convert to ART format
        # Submit to ART server
        # Return job reference
```


### 2. Composable Components

The harness uses a **three-component architecture** that separates concerns:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Trainer                                    │
│  (Orchestrates the entire finetuning loop across epochs)                │
│                                                                         │
│  ┌───────────────────────┐         ┌───────────────────────────┐        │
│  │  TrajectoryBuilder    │         │    TrainerAdapter         │        │
│  │                       │         │                           │        │
│  │  - Runs evaluations   │ ──────► │  - Validates trajectories │        │
│  │  - Collects episodes  │         │  - Submits to backend     │        │
│  │  - Computes rewards   │         │  - Monitors training      │        │
│  │  - Groups trajectories│         │  - Reports status         │        │
│  └───────────────────────┘         └───────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │   Remote Training       │
                            │      Backend            │
                            │  (OpenPipe ART, etc.)   │
                            └─────────────────────────┘
```

This architecture ensures:
- **Single responsibility**: Each component does one thing well
- **Independent evolution**: Components can be upgraded separately
- **Easy testing**: Mock any component for unit tests
- **Flexibility**: Mix and match components for different scenarios

## Data Structures

### Trajectories

A **trajectory** in NeMo Agent toolkit represents a complete interaction sequence:

```python
class Trajectory(BaseModel):
    episode: list[EpisodeItem] | list[DPOItem]  # The sequence of messages/actions
    reward: float               # The outcome reward for this trajectory
    shaped_rewards: list[float] | None  # Optional step-wise rewards
    metadata: dict | None       # Additional context
```

### Episode Items

An **episode item** represents a single message or action:

```python
class EpisodeItem(BaseModel):
    role: EpisodeItemRole  # USER, ASSISTANT, SYSTEM, TOOL, etc.
    content: str           # The message content
    logprobs: Any | None   # Log probabilities (required for ASSISTANT)
    metadata: dict | None  # Step-specific metadata
```

The role can be:

| Role | Description |
|------|-------------|
| `USER` | Human or system input to the agent |
| `ASSISTANT` | Model-generated response |
| `SYSTEM` | System prompt or instructions |
| `TOOL` | Tool/function call result |
| `FUNCTION` | Function call (legacy format) |
| `ENVIRONMENT` | Environment state or feedback |


### `DPO` Items

For `DPO` training, a trajectory consists of preferred and rejected responses:

```python
class DPOItem(BaseModel):
    """
    A single step in an episode for DPO training.
    """
    prompt: list[OpenAIMessage] | str = Field(description="The prompt messages leading to the response.")
    chosen_response: str = Field(description="The response chosen as better by the reward model.")
    rejected_response: str = Field(description="The response rejected as worse by the reward model.")
```
The `OpenAIMessage` type is the standard message format used in OpenAI-compatible chat APIs. It consists of:

```python
class OpenAIMessage(BaseModel):
    """
    A message in the OpenAI chat format.
    """
    role: str = Field(description="The role of the message (e.g., 'user', 'assistant').")
    content: str = Field(description="The content of the message.")
```

### Trajectory Collections

Trajectories are organized into **collections** that group related examples:

```python
class TrajectoryCollection(BaseModel):
    trajectories: list[list[Trajectory]]  # Grouped trajectories
    run_id: str                            # Unique identifier
```

The nested list structure (`list[list[Trajectory]]`) is critical:

```
trajectories = [
    # Group 1: All trajectories for "What is Python?"
    [
        Trajectory(episode=[...], reward=0.9),  # Generation 1
        Trajectory(episode=[...], reward=0.7),  # Generation 2
        Trajectory(episode=[...], reward=0.95), # Generation 3
    ],
    # Group 2: All trajectories for "Explain recursion"
    [
        Trajectory(episode=[...], reward=0.6),
        Trajectory(episode=[...], reward=0.8),
        Trajectory(episode=[...], reward=0.5),
    ],
    # ... more groups
]
```

This structure enables:
- **GRPO**: Compare responses to the same prompt
- **Curriculum learning**: Filter groups by average reward
- **Variance analysis**: Identify examples with no learning signal

### Reward Functions

Reward functions determine how well an agent performed. The harness uses the NeMo Agent toolkit [**evaluator system**](../../improve-workflows/evaluate.md) to compute rewards:

```yaml
eval:
  evaluators:
    my_reward:
      _type: custom_evaluator
      # Evaluator configuration...

finetuning:
  reward_function:
    name: my_reward  # References the evaluator above
```

This design allows:
- Reuse of evaluation metrics as training signals
- Complex multi-criteria rewards through evaluator composition
- Consistent scoring between evaluation and training

## The Training Loop

A typical training loop in the NeMo Agent toolkit harness:

```
┌────────────────────────────────────────────────────────────────────────┐
│                         Training Loop                                  │
│                                                                        │
│  for epoch in range(num_epochs):                                       │
│      │                                                                 │
│      ▼                                                                 │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │ 1. TRAJECTORY COLLECTION                                     │      │
│  │    - Run workflow on training dataset                        │      │
│  │    - Generate N trajectories per example                     │      │
│  │    - Compute rewards using configured evaluator              │      │
│  │    - Group trajectories by example ID                        │      │
│  └──────────────────────────────────────────────────────────────┘      │
│      │                                                                 │
│      ▼                                                                 │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │ 2. CURRICULUM FILTERING (if enabled)                         │      │
│  │    - Sort groups by average reward                           │      │
│  │    - Filter out low-variance groups                          │      │
│  │    - Select top percentile of groups                         │      │
│  │    - Expand percentile at intervals                          │      │
│  └──────────────────────────────────────────────────────────────┘      │
│      │                                                                 │
│      ▼                                                                 │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │ 3. TRAINING SUBMISSION                                       │      │
│  │    - Convert trajectories to backend format                  │      │
│  │    - Submit to training backend                              │      │
│  │    - Wait for training to complete                           │      │
│  └──────────────────────────────────────────────────────────────┘      │
│      │                                                                 │
│      ▼                                                                 │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │ 4. LOGGING & MONITORING                                      │      │
│  │    - Record metrics (avg reward, num trajectories, etc.)     │      │
│  │    - Generate visualizations                                 │      │
│  │    - Run validation (if configured)                          │      │
│  └──────────────────────────────────────────────────────────────┘      │
│      │                                                                 │
│      └──────────────────► Next epoch                                   │
└────────────────────────────────────────────────────────────────────────┘
```

## Configuration Reference

### Minimal Configuration

```yaml
llms:
  training_model:
    _type: openai
    model_name: Qwen/Qwen2.5-3B-Instruct
    base_url: http://localhost:8000/v1
    api_key: default

workflow:
  _type: my_workflow
  llm: training_model

eval:
  general:
    max_concurrency: 16
    output_dir: .tmp/nat/finetuning/eval
    dataset:
      _type: json
      file_path: data/training_data.json

  evaluators:
    accuracy:
      _type: my_accuracy_evaluator

trajectory_builders:
  my_builder:
    _type: my_trajectory_builder
    num_generations: 2

trainer_adapters:
  my_adapter:
    _type: my_trainer_adapter

trainers:
  my_trainer:
    _type: my_trainer

finetuning:
  enabled: true
  trainer: my_trainer
  trajectory_builder: my_builder
  trainer_adapter: my_adapter
  reward_function:
    name: accuracy
  num_epochs: 10
  output_dir: .tmp/nat/finetuning
```

### Full Configuration Reference

#### `finetuning` Section

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `false` | Whether finetuning is enabled |
| `trainer` | `str` | - | Name of the trainer to use |
| `trajectory_builder` | `str` | - | Name of the trajectory builder |
| `trainer_adapter` | `str` | - | Name of the trainer adapter |
| `reward_function.name` | `str` | - | Name of the evaluator for rewards |
| `target_functions` | `list[str]` | `["<workflow>"]` | Functions to extract trajectories from |
| `target_model` | `str` | `null` | Specific model to target |
| `num_epochs` | `int` | `1` | Number of training epochs |
| `output_dir` | `Path` | `.tmp/nat/finetuning` | Output directory |
| `curriculum_learning` | `object` | see below | Curriculum learning config |

#### `curriculum_learning` Section

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable curriculum learning |
| `initial_percentile` | `float` | `0.3` | Starting fraction of examples |
| `increment_percentile` | `float` | `0.2` | Fraction to add each expansion |
| `expansion_interval` | `int` | `5` | Epochs between expansions |
| `min_reward_diff` | `float` | `0.1` | Minimum variance threshold |
| `sort_ascending` | `bool` | `false` | Sort direction (false=easy-to-hard) |
| `random_subsample` | `float` | `null` | Optional random subsampling |

## CLI Usage

Run finetuning from the command line:

```bash
nat finetune --config_file=path/to/config.yml
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--config_file` | Path to the configuration file (required) |
| `--dataset` | Override the dataset path from config |
| `--result_json_path` | JSON path to extract results (default: `$`) |
| `--endpoint` | Remote endpoint for workflow execution |
| `--endpoint_timeout` | HTTP timeout in seconds (default: 300) |
| `--override`, `-o` | Override config values |
| `--validation_dataset` | Path to validation dataset |
| `--validation_interval` | Validate every N epochs (default: 5) |
| `--validation_config_file` | Separate config for validation |

### Example Commands

```bash
# Basic finetuning
nat finetune --config_file=configs/finetune.yml

# Override number of epochs
nat finetune --config_file=configs/finetune.yml -o finetuning.num_epochs 20

# With validation
nat finetune --config_file=configs/finetune.yml \
    --validation_dataset=data/val.json \
    --validation_interval=3

# Using remote endpoint
nat finetune --config_file=configs/finetune.yml \
    --endpoint=http://localhost:8000/generate \
    --endpoint_timeout=600
```


<!-- path-check-skip-end -->

## See Also

- [Extending the Finetuning Harness](../../extend/custom-components/finetuning.md) - Creating custom components
- [OpenPipe ART Integration](rl_with_openpipe.md) - Using the ART backend
- [Evaluating Workflows](../evaluate.md) - Understanding evaluators for rewards
