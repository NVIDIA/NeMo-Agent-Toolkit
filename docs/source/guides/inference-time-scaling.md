# Inference Time Scaling

Inference Time Scaling (ITS) is a collection of strategies and helper functions
for building agent pipelines that explore multiple plans or tool invocations,
then downselect to a final answer.  ITS separates the problem into four
**strategy types**:

1. **Search** – Generate plans or queries.
2. **Editing** – Refine or modify generated items.
3. **Scoring** – Evaluate items using an LLM.
4. **Selection** – Pick or merge the best items.

Each strategy is implemented as a subclass of
`aiq.experimental.inference_time_scaling.models.strategy_base.StrategyBase` and
registered using the `register_its_strategy` decorator.  Strategy classes live in
`aiq.experimental.inference_time_scaling.*` and are loaded by importing the
package's `register` module.

## ITSItem

`ITSItem` is the basic data model passed between strategies.  It stores fields
for the original input, generated output or plan, an optional score and any
metadata.  Strategies receive and return `list[ITSItem]`.

```python
from aiq.experimental.inference_time_scaling.models.its_item import ITSItem

item = ITSItem(input="search term", output=None, metadata={"motivation": "demo"})
```

## Building Strategies with the Builder

The `WorkflowBuilder` can instantiate strategies by name.  After importing the
`register` module all built‑in strategies become available.

```python
import aiq.experimental.inference_time_scaling.register  # registers strategies
from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.experimental.inference_time_scaling.models.stage_enums import (
    PipelineTypeEnum, StageTypeEnum,
)

async with WorkflowBuilder() as builder:
    strategy = await builder.get_its_strategy(
        strategy_name="threshold_selection",
        pipeline_type=PipelineTypeEnum.TOOL_USE,
        stage_type=StageTypeEnum.SELECTION,
    )
```

To add your own strategy subclass simply decorate a function with
`register_its_strategy` and yield the instantiated strategy.

```python
from aiq.cli.register_workflow import register_its_strategy
from aiq.data_models.its_strategy import ITSStrategyBaseConfig
from aiq.experimental.inference_time_scaling.models.strategy_base import StrategyBase

class MyConfig(ITSStrategyBaseConfig, name="MyStrategy"):
    ...

class MySelector(StrategyBase):
    ...

@register_its_strategy(config_type=MyConfig)
async def register_my_selector(config: MyConfig, builder: Builder):
    selector = MySelector(config)
    await selector.build_components(builder)
    yield selector
```

## Chaining Strategies

Strategies can be combined manually or via the helper functions in
`aiq.experimental.inference_time_scaling.functions`.  For example,
`plan_select_execute_function` performs plan generation, optional editing,
scoring and selection before executing a downstream function.

Other helpers include:

- `its_tool_wrapper_function` – convert task descriptions into structured tool
  inputs using an LLM.
- `its_tool_orchestration_function` – orchestrate multiple tool invocations with
  search/edit/score/select steps.
- `execute_score_select_function` – repeatedly execute a function, score the
  outputs and select the best one.

These functions are registered like normal AIQ functions and can be composed in
workflows or called directly.

## Example

```python
async with WorkflowBuilder() as builder:
    # Obtain a plan from the planning strategy
    planner = await builder.get_its_strategy(
        strategy_name="single_shot_multi_plan",
        pipeline_type=PipelineTypeEnum.PLANNING,
        stage_type=StageTypeEnum.SEARCH,
    )
    plans = await planner.ainvoke([ITSItem()], "What should I do?", "Agent")
    print(plans[0].plan)
```

This modular design allows developers to swap strategies at inference time,
experiment with different combinations and build powerful agent pipelines.