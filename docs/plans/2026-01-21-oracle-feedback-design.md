# Oracle Feedback for GA Prompt Optimizer

## Overview

Add context-grounded improvement to the GA prompt optimizer by extracting reasoning from poorly-performing evaluation items and feeding it back into the mutation process. Instead of blind evolution, the optimizer learns *why* certain prompts failed.

## Requirements

1. Extract reasoning from `EvalOutputItem.reasoning` when available
2. Identify worst N items by weighted score across evaluators
3. Inject reasoning into prompt optimizer as oracle feedback
4. Cap feedback at configurable character limit to protect context window
5. Support multiple feedback injection modes (never, always, failing_only, adaptive)

## Configuration

New fields in `PromptGAOptimizationConfig`:

```python
# Oracle feedback mode: when to inject feedback into mutations
oracle_feedback_mode: Literal["never", "always", "failing_only", "adaptive"] = "never"

# Number of worst-scoring items to extract reasoning from
oracle_feedback_worst_n: int = 5

# Maximum characters for oracle feedback in mutation prompt
oracle_feedback_max_chars: int = 4000

# For "failing_only" mode: fitness threshold below which feedback is injected (0-1 normalized)
oracle_feedback_fitness_threshold: float = 0.3

# For "adaptive" mode: generations without improvement before enabling feedback
oracle_feedback_stagnation_generations: int = 3

# For "adaptive" mode: fitness variance threshold for collapse detection
oracle_feedback_fitness_variance_threshold: float = 0.01

# For "adaptive" mode: prompt duplication ratio threshold (0-1)
oracle_feedback_diversity_threshold: float = 0.5
```

### Feedback Modes

| Mode | Behavior |
|------|----------|
| `never` | Current behavior, no feedback injection (default) |
| `always` | Every mutation receives feedback from parent's worst items |
| `failing_only` | Only individuals below fitness threshold receive feedback |
| `adaptive` | Starts without feedback, enables when plateau detected |

### Adaptive Mode Triggers

Feedback is enabled when either condition is met:
- **Fitness stagnation**: Best fitness hasn't improved for N generations
- **Diversity collapse**: Either fitness variance below threshold OR prompt duplication ratio above threshold

## Data Model Changes

### Extended Individual Dataclass

```python
@dataclass
class Individual:
    prompts: dict[str, str]
    metrics: dict[str, float] | None = None
    scalar_fitness: float | None = None
    worst_items_reasoning: list[str] | None = None  # NEW
```

## Multi-Evaluator Reasoning Aggregation

When multiple evaluators are configured, reasoning is aggregated using metric weights:

```python
# Collect items with evaluator weights
weighted_items: list[tuple[float, str, str]] = []  # (weighted_score, reasoning, evaluator_name)

for name, result in run_results:
    evaluator_weight = weights_by_name.get(name, 1.0)
    direction = directions_by_name.get(name, "maximize")

    for item in result.eval_output_items:
        if item.reasoning:
            score = float(item.score)
            if direction == "maximize":
                score = -score  # Invert so sorting ascending gives worst

            # Higher weight = more important failures float to top
            priority_score = score / max(evaluator_weight, 0.01)
            weighted_items.append((priority_score, str(item.reasoning), name))

# Sort by priority (worst weighted failures first)
weighted_items.sort(key=lambda x: x[0])
worst_n = weighted_items[:oracle_feedback_worst_n]

# Format with evaluator context
ind.worst_items_reasoning = [
    f"[{evaluator}] {reasoning}"
    for _, reasoning, evaluator in worst_n
]
```

## Feedback Injection Logic

### Building Feedback String

```python
def _build_oracle_feedback(ind: Individual, max_chars: int) -> str | None:
    """Build truncated feedback string from worst items reasoning."""
    if not ind.worst_items_reasoning:
        return None

    feedback_parts = []
    current_length = 0

    for i, reasoning in enumerate(ind.worst_items_reasoning, 1):
        entry = f"{i}. {reasoning}\n"
        if current_length + len(entry) > max_chars:
            remaining = max_chars - current_length
            if remaining > 20:
                feedback_parts.append(entry[:remaining-3] + "...")
            break
        feedback_parts.append(entry)
        current_length += len(entry)

    return "".join(feedback_parts) if feedback_parts else None
```

### Mode Decision Logic

```python
def _should_inject_feedback(
    ind: Individual,
    mode: str,
    fitness_threshold: float,
    adaptive_enabled: bool,
) -> bool:
    if mode == "never":
        return False
    if mode == "always":
        return True
    if mode == "failing_only":
        return (ind.scalar_fitness or 0.0) < fitness_threshold
    if mode == "adaptive":
        return adaptive_enabled
    return False
```

### Adaptive State Tracking

```python
best_fitness_history: list[float] = []
adaptive_feedback_enabled: bool = False

# After each generation:
current_best = max(ind.scalar_fitness or 0.0 for ind in population)
best_fitness_history.append(current_best)

if not adaptive_feedback_enabled and mode == "adaptive":
    # Check stagnation
    if len(best_fitness_history) >= stagnation_generations:
        recent = best_fitness_history[-stagnation_generations:]
        stagnated = (max(recent) - min(recent)) < 0.001

    # Check diversity collapse
    fitness_values = [ind.scalar_fitness or 0.0 for ind in population]
    fitness_variance = variance(fitness_values)
    variance_collapsed = fitness_variance < fitness_variance_threshold

    prompt_keys = [tuple(sorted(ind.prompts.items())) for ind in population]
    unique_ratio = len(set(prompt_keys)) / len(prompt_keys)
    diversity_collapsed = unique_ratio < (1.0 - diversity_threshold)

    if stagnated or variance_collapsed or diversity_collapsed:
        adaptive_feedback_enabled = True
```

## Prompt Template Changes

### Updated mutator_prompt (prompt.py)

Add placeholder for optional feedback section:

```python
mutator_prompt = """
## CORE DIRECTIVES
... (existing content unchanged) ...

---

## INPUT
Here is the prompt to mutate:
{original_prompt}

## OBJECTIVE
The prompt must achieve the following objective:
{objective}

{oracle_feedback_section}

The modified prompt is:
"""

oracle_feedback_template = """
## FAILURE ANALYSIS
The following are examples of cases where the current prompt performed poorly,
along with reasoning explaining why. Use these insights to improve the prompt:

{oracle_feedback}

Focus on addressing the root causes identified above while maintaining the original objective.
"""
```

### Updated _inner function (register.py)

```python
async def _inner(input_message: PromptOptimizerInputSchema) -> str:
    original_prompt = input_message.original_prompt
    prompt_objective = input_message.objective
    oracle_feedback = input_message.oracle_feedback

    feedback_section = ""
    if oracle_feedback:
        feedback_section = oracle_feedback_template.format(
            oracle_feedback=oracle_feedback
        )

    prompt_extension = (await prompt_extension_template.ainvoke(input={
        "original_prompt": original_prompt,
        "objective": prompt_objective,
        "oracle_feedback_section": feedback_section,
    })).to_string()

    prompt = f"{base_prompt}\n\n{prompt_extension}"
    optimized_prompt = await llm.ainvoke(prompt)
    return optimized_prompt.content
```

## Implementation Plan

### Task 1: Add configuration fields
- File: `src/nat/data_models/optimizer.py`
- Add 7 new fields to `PromptGAOptimizationConfig`

### Task 2: Extend Individual dataclass
- File: `src/nat/profiler/parameter_optimization/prompt_optimizer.py`
- Add `worst_items_reasoning` field to `Individual`

### Task 3: Implement reasoning extraction in _evaluate
- File: `src/nat/profiler/parameter_optimization/prompt_optimizer.py`
- Extract reasoning from EvalOutputItems
- Implement weighted multi-evaluator aggregation
- Store worst N in Individual

### Task 4: Implement feedback decision logic
- File: `src/nat/profiler/parameter_optimization/prompt_optimizer.py`
- Add `_should_inject_feedback` function
- Add `_build_oracle_feedback` function
- Add adaptive state tracking variables and logic

### Task 5: Update _mutate_prompt to accept and use feedback
- File: `src/nat/profiler/parameter_optimization/prompt_optimizer.py`
- Update signature to accept parent Individual
- Call feedback helpers and pass to PromptOptimizerInputSchema

### Task 6: Update prompt templates
- File: `src/nat/agent/prompt_optimizer/prompt.py`
- Add `oracle_feedback_section` placeholder
- Add `oracle_feedback_template`

### Task 7: Update prompt optimizer function
- File: `src/nat/agent/prompt_optimizer/register.py`
- Update `_inner` to build and inject feedback section
- Update PromptTemplate input_variables

### Task 8: Update optimizer documentation
- File: `docs/source/improve-workflows/optimizer.md`
- Document oracle feedback configuration options
- Explain feedback modes and when to use each
- Document evaluator requirements (must populate `reasoning` field in `EvalOutputItem`)
- Add examples of configuring oracle feedback

### Task 9: Add unit tests
- Test feedback extraction with single and multiple evaluators
- Test each feedback mode
- Test adaptive trigger conditions
- Test character truncation
- Test prompt template rendering with and without feedback

## Evaluator Requirements

For oracle feedback to work, evaluators must:
1. Populate the `reasoning` field in `EvalOutputItem` with meaningful explanation
2. Return reasoning that explains *why* an item scored poorly, not just the score

Example evaluator output:
```python
EvalOutputItem(
    id="item_123",
    score=0.2,
    reasoning="The response failed to address the user's question about pricing. "
              "Instead, it provided generic product information without mentioning "
              "any costs or payment options."
)
```
