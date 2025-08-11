<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Test Time Compute With NVIDIA NeMo Agent Toolkit
Test time compute reallocates compute after a model has been trained, trading extra inference cycles for much better reasoning, factuality, and robustness, often without any additional training data. The new **`aiq.experimental.test_time_compute`** package codifies this idea as four strategy types (Search ▶ Editing ▶ Scoring ▶ Selection) that operate on a lightweight `TTCItem` record.  Developers can compose these strategies manually or use several **pre‑built TTC functions** that wire everything up automatically. To add your own strategy, you can simply follow these steps:
1. Write a config subclass.
2. Implement a `StrategyBase` child. 
3. Register it with the `@register_ttc_strategy` decorator.  
The remainder of this document explains each step in detail.

## Core Design

### Strategy pipeline

| Stage         | Purpose                                                      | Examples                                                                                                                                      |
| ------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Search**    | Generate many alternative plans, prompts, or tool invocations | `single_shot_multi_plan`, `multi_llm_plan`, `multi_query_retrieval_search`                                                                    |
| **Editing**   | Refine or transform the candidates                           | `iterative_plan_refinement`, `llm_as_a_judge_editor`, `motivation_aware_summarization`                                                        |
| **Scoring**   | Assign a numeric quality score                               | `llm_based_plan_scorer`, `llm_based_agent_scorer`, `motivation_aware_scorer`                                                                  |
| **Selection** | Down‑select or merge                                         | `best_of_n_selector`, `threshold_selector`, `llm_based_plan_selector`, `llm_based_output_merging_selector`, `llm_based_agent_output_selector` |

A pipeline type tells a strategy where it is used.

```text
PipelineTypeEnum = { PLANNING, TOOL_USE, AGENT_EXECUTION, CUSTOM }
StageTypeEnum    = { SEARCH, EDITING, SCORING, SELECTION }
```

Each strategy exposes the following methods to the `Builder` to allow the `Builder` to resolve dependencies and ensure type safety:

```python
supported_pipeline_types() -> list[PipelineTypeEnum]
stage_type()                -> StageTypeEnum
```

The `Builder` will ensure that when an `TTC Strategy` is requested, that the stage and pipeline types match the implementation's supported types.

### `StrategyBase`

Every concrete strategy extends `StrategyBase`.

```python
class MyStrategy(StrategyBase):
    async def build_components(self, builder): ...
    async def ainvoke(
            self,
            items: list[TTCItem],
            original_prompt: str | None = None,
            agent_context:  str | None = None,
    ) -> list[TTCItem]:
        ...
```

*Implementation hint*: Use the `Builder` helpers (`get_llm`, `get_function`, …) during `build_components` to resolve references once and cache them.

### `TTCItem`

A **single, interoperable record** passed between stages.

| Field      | Meaning                             |
| ---------- | ----------------------------------- |
| `input`    | Raw user task / tool `args`           |
| `output`   | Generated answer / tool result      |
| `plan`     | Execution plan (planning pipelines) |
| `feedback` | Review comments from editing stages |
| `score`    | Numeric quality metric              |
| `metadata` | Arbitrary auxiliary data            |
| `name`     | Tool name or other identifier       |

Because it is a `pydantic.BaseModel`, you get `.model_dump()` and validation for free.

## Built‑in Strategies

Below is a non‑exhaustive catalog you can use immediately; refer to the inline doc‑strings for full parameter lists.

| Category  | `Config` class                                                    | One‑liner                                                                 |
| --------- | --------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Search    | `SingleShotMultiPlanConfig`                                     | Few‑shot prompt that emits *n* candidate plans at different temperatures. |
|           | `MultiLLMPlanConfig`                                            | Query multiple LLMs in parallel, then concatenate plans.                  |
|           | `MultiQueryRetrievalSearchConfig`                               | Reformulate a retrieval query from diverse perspectives.                  |
| Editing   | `IterativePlanRefinementConfig`                                 | Loop: *plan → critique → edit*.                                           |
|           | `LLMAsAJudgeEditorConfig`                                       | “Feedback LLM + editing LLM” cooperative refinement.                      |
|           | `MotivationAwareSummarizationConfig`                            | Grounded summary that respects user’s “motivation”.                       |
| Scoring   | `LLMBasedPlanScoringConfig`                                     | Judge execution plans on a 1‑10 scale.                                    |
|           | `LLMBasedAgentScoringConfig`                                    | Judge final agent answers.                                                |
|           | `MotivationAwareScoringConfig`                                  | Score w\.r.t. task + motivation context.                                  |
| Selection | `BestOfNSelectionConfig`                                        | Keep the highest‑scoring item.                                            |
|           | `ThresholdSelectionConfig`                                      | Filter by score ≥ τ.                                                      |
|           | `LLMBasedPlanSelectionConfig` / …AgentOutput… / …OutputMerging… | Let an LLM choose or merge.                                               |


## Pre‑Built TTC Functions

NeMo Agent toolkit ships higher‑level wrappers that hide all orchestration.

| Function                              | Use‑case                                                                                                            |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **`ttc_tool_wrapper_function`**       | Turn an arbitrary function into a *tool*; the wrapper asks an LLM to translate free‑text into structured arguments. |
| **`ttc_tool_orchestration_function`** | Accepts a list of tool invocations, optionally runs search/edit/score/select, then executes each tool concurrently. |
| **`execute_score_select_function`**   | Run a function *k* times, score each output, pick the best.                                                         |
| **`plan_select_execute_function`**    | End‑to‑end: plan → optionally edit/score → select plan → feed downstream agent.                                     |

These are declared in `aiq.experimental.test_time_compute.functions.*` and can be referenced in your `AIQConfig` just like any other function.

## Creating and Registering a New Strategy

Follow the steps below to create and register a new strategy.

1. Define a `config` model.

   ```python
   class MyStrategyConfig(TTCStrategyBaseConfig, name="my_strategy"):
       my_param: float = 0.5
   ```

2. Implement the strategy

   ```python
   from aiq.experimental.test_time_compute.models.strategy_base import StrategyBase
   class MyStrategy(StrategyBase):
       ...
   ```

3. Register the strategy.

   ```python
   from aiq.cli.register_workflow import register_ttc_strategy

   @register_ttc_strategy(config_type=MyStrategyConfig)
   async def register_my_strategy(cfg: MyStrategyConfig, builder: Builder):
       strat = MyStrategy(cfg)
       await strat.build_components(builder)
       yield strat
   ```

Your strategy is now discoverable by `TypeRegistry` and can be referenced in `AIQConfig` fields.

---

## Composing Strategies in an `AIQConfig`

TTC Strategies can be part of workflow configurations, just like other components such as `LLMs`. For example, the following configuration excerpt shows how an TTC strategy can be 
configured in a `config.yml` file and used in a workflow function:

```yaml
ttc_strategies:
  selection_strategy:
    _type: llm_based_agent_output_merging
    selection_llm: nim_llm

workflow:
  _type: execute_score_select_function
  selector: selection_strategy
  augmented_fn: react_agent_executor
  num_executions: 3
```

## Extending Tools and Pipelines

* **Multiple stages**: Nothing stops you from chaining *search → edit → search* again, as long as each stage returns `List[TTCItem]`.
* **Streaming**: Strategies themselves are non‑streaming, but you can wrap a streaming LLM in an TTC pipeline by choosing an appropriate pre‑built function such as `plan_select_execute_function`, which keeps streaming support if the downstream agent streams.
* **Debugging**: Log levels are respected through the standard `logging` module; export `AIQ_LOG_LEVEL=DEBUG` for verbose traces, including every intermediate `TTCItem`.


## Testing your strategy

Write isolated unit tests by instantiating your config and strategy directly, then call `ainvoke` with hand‑crafted `TTCItem` lists.  Refer to the companion `tests/` directory for reference tests on `ThresholdSelector` and `BestOfNSelector`.


Happy scaling!