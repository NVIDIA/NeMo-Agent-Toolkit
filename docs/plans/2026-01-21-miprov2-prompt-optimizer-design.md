# MIPROv2 Prompt Optimizer Design

## Overview

Add MIPROv2 (Multiprompt Instruction PRoposal Optimizer Version 2) as an alternative prompt optimization algorithm alongside the existing Genetic Algorithm (GA) optimizer. Users can select which algorithm to use via configuration.

## Design Decisions

| Aspect | Decision |
|--------|----------|
| **Scope** | Instructions only (no few-shot demos) |
| **Selection** | `algorithm: "ga" \| "miprov2"` field |
| **Config model** | Discriminated union - `PromptGAConfig` / `PromptMIPROConfig` |
| **Grounding** | Dataset summary + random tips |
| **Evaluation** | Configurable minibatch (default 35) + periodic full eval |
| **Intensity** | Presets (`light`/`medium`/`heavy`) with optional overrides |
| **Proposer LLM** | Reuses existing `optimizer_llm` |
| **Search** | Optuna TPE (Tree-structured Parzen Estimator) |

## Configuration Model

### Discriminated Union Approach

Each algorithm gets its own focused config model, selected by the `algorithm` discriminator field.

```python
# In src/nat/data_models/optimizer.py

from typing import Literal, Annotated
from pydantic import Field, Discriminator

class PromptGAConfig(BaseModel):
    """GA-specific prompt optimization config."""
    algorithm: Literal["ga"] = "ga"

    prompt_population_init_function: str | None = None
    prompt_recombination_function: str | None = None
    ga_population_size: int = 24
    ga_generations: int = 15
    ga_crossover_rate: float = 0.8
    ga_mutation_rate: float = 0.3
    ga_elitism: int = 2
    ga_selection_method: str = "tournament"
    ga_tournament_size: int = 3
    ga_parallel_evaluations: int = 8
    ga_diversity_lambda: float = 0.0
    # ... oracle feedback fields ...


class PromptMIPROConfig(BaseModel):
    """MIPROv2-specific prompt optimization config."""
    algorithm: Literal["miprov2"] = "miprov2"

    prompt_population_init_function: str | None = None  # reused for proposal
    intensity: Literal["light", "medium", "heavy"] = "medium"
    num_candidates: int | None = None   # override preset
    num_trials: int | None = None       # override preset
    minibatch_size: int = 35
    full_eval_frequency: int = 5
    use_dataset_summary: bool = True
    use_tips: bool = True
    parallel_evaluations: int = 8


PromptOptimizationConfig = Annotated[
    PromptGAConfig | PromptMIPROConfig,
    Discriminator("algorithm")
]
```

### YAML Usage Examples

```yaml
# GA config (default, backward compatible)
optimizer:
  prompt:
    algorithm: "ga"
    ga_population_size: 24
    ga_generations: 15

# MIPROv2 config
optimizer:
  prompt:
    algorithm: "miprov2"
    intensity: "medium"
    num_trials: 50
```

## File Structure

```
src/nat/profiler/parameter_optimization/
├── prompt_optimizer.py          # Existing GA implementation (unchanged)
├── prompt_optimizer_mipro.py    # New MIPROv2 implementation
├── prompt_optimizer_common.py   # Shared utilities (evaluation, normalization)
├── grounded_proposer.py         # New: generates candidate instructions
├── oracle_feedback.py           # Existing (can be reused)
└── update_helpers.py            # Existing
```

## MIPROv2 Implementation Flow

### Stage 1: Grounded Instruction Proposal

```python
async def generate_candidates(prompt_space, dataset, optimizer_llm, config):
    candidates = {}  # param_name -> list of candidate instructions

    for param_name, (original_prompt, purpose) in prompt_space.items():
        # 1. Generate dataset summary (if enabled)
        summary = None
        if config.use_dataset_summary:
            summary = await summarize_dataset(dataset, optimizer_llm)

        # 2. Generate N candidate instructions
        num = get_num_candidates(config)  # from preset or override
        param_candidates = [original_prompt]  # always include original

        for _ in range(num - 1):
            tip = random.choice(PROMPT_TIPS) if config.use_tips else None
            candidate = await propose_instruction(
                original_prompt, purpose, summary, tip, optimizer_llm
            )
            param_candidates.append(candidate)

        candidates[param_name] = param_candidates

    return candidates
```

### Stage 2: Bayesian Optimization with Optuna TPE

```python
async def bayesian_search(candidates, prompt_space, config, evaluate_fn):
    study = optuna.create_study(direction="maximize", sampler=TPESampler())

    for trial_num in range(get_num_trials(config)):
        trial = study.ask()

        # Select candidate for each prompt parameter
        prompts = {
            param: candidates[param][trial.suggest_categorical(param, range(len(candidates[param])))]
            for param in prompt_space
        }

        # Minibatch evaluation
        score = await evaluate_fn(prompts, minibatch_size=config.minibatch_size)
        study.tell(trial, score)

        # Periodic full evaluation
        if trial_num % config.full_eval_frequency == 0:
            await full_validation(study.best_params, ...)

    return study.best_params
```

## Grounded Proposer

### Dataset Summary Generation

```python
DATASET_SUMMARY_PROMPT = """Analyze these evaluation examples and describe:
1. The type of task being performed
2. Common patterns in inputs
3. Expected output characteristics
4. Any edge cases or challenges observed

Examples:
{examples}

Provide a concise summary (2-3 paragraphs) that would help someone write better instructions for this task."""

async def summarize_dataset(dataset: list, llm, sample_size: int = 10) -> str:
    """Generate LLM-based characterization of the dataset."""
    sampled = random.sample(dataset, min(sample_size, len(dataset)))
    examples_text = format_examples(sampled)
    return await llm.ainvoke(DATASET_SUMMARY_PROMPT.format(examples=examples_text))
```

### Random Tips for Diversity

```python
PROMPT_TIPS = [
    "Be concise and direct",
    "Think step by step",
    "Be creative and thorough",
    "Focus on accuracy over speed",
    "Consider edge cases explicitly",
    "Use specific examples in your reasoning",
    "Break down complex problems",
    "Verify your answer before responding",
]
```

### Instruction Proposal Prompt

```python
INSTRUCTION_PROPOSAL_PROMPT = """You are an expert prompt engineer. Generate an improved version of this instruction.

Original instruction:
{original_prompt}

Purpose: {purpose}

{dataset_section}
{tip_section}

Return ONLY the improved instruction text, preserving any {{variables}} exactly."""
```

## Intensity Presets

```python
MIPRO_PRESETS = {
    "light": {
        "num_candidates": 6,
        "num_trials": 20,
        "max_val_size": 100,
    },
    "medium": {
        "num_candidates": 12,
        "num_trials": 50,
        "max_val_size": 300,
    },
    "heavy": {
        "num_candidates": 18,
        "num_trials": 100,
        "max_val_size": 1000,
    },
}

def resolve_mipro_params(config: PromptMIPROConfig) -> dict:
    """Resolve preset + overrides into final parameters."""
    preset = MIPRO_PRESETS[config.intensity]
    return {
        "num_candidates": config.num_candidates or preset["num_candidates"],
        "num_trials": config.num_trials or preset["num_trials"],
        "max_val_size": preset["max_val_size"],  # not overridable
        "minibatch_size": config.minibatch_size,
        "full_eval_frequency": config.full_eval_frequency,
        "use_dataset_summary": config.use_dataset_summary,
        "use_tips": config.use_tips,
        "parallel_evaluations": config.parallel_evaluations,
    }
```

## Entry Point & Dispatcher

```python
from nat.data_models.optimizer import PromptGAConfig, PromptMIPROConfig

async def optimize_prompts_dispatch(
    *,
    base_cfg: Config,
    full_space: dict[str, SearchSpace],
    optimizer_config: OptimizerConfig,
    opt_run_config: OptimizerRunConfig,
) -> None:
    """Route to appropriate prompt optimizer based on algorithm config."""

    prompt_cfg = optimizer_config.prompt

    if isinstance(prompt_cfg, PromptGAConfig):
        from nat.profiler.parameter_optimization.prompt_optimizer import optimize_prompts
        await optimize_prompts(
            base_cfg=base_cfg,
            full_space=full_space,
            optimizer_config=optimizer_config,
            opt_run_config=opt_run_config,
        )

    elif isinstance(prompt_cfg, PromptMIPROConfig):
        from nat.profiler.parameter_optimization.prompt_optimizer_mipro import optimize_prompts_mipro
        await optimize_prompts_mipro(
            base_cfg=base_cfg,
            full_space=full_space,
            optimizer_config=optimizer_config,
            opt_run_config=opt_run_config,
        )
```

## Outputs

Consistent with GA optimizer:

- `optimized_prompts.json` - Final best prompts
- `mipro_history.csv` - Trial history (trial number, params, scores)
- `optimized_prompts_trial{N}.json` - Checkpoints at full-eval intervals

## Files to Create

| File | Description |
|------|-------------|
| `prompt_optimizer_mipro.py` | Main MIPROv2 implementation |
| `prompt_optimizer_common.py` | Shared utilities extracted from GA |
| `grounded_proposer.py` | Instruction candidate generation |

## Files to Modify

| File | Changes |
|------|---------|
| `optimizer.py` | Add discriminated union config models |
| `optimizer_runtime.py` | Add dispatcher logic |

## Dependencies

- `optuna` - Already used for numeric optimization, provides TPE sampler
