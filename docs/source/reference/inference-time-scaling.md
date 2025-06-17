## 1. Inference‑time scaling at a glance
Inference‑time scaling reallocates compute *after* a model has been trained, trading extra inference cycles for markedly better reasoning, factuality and robustness often without any additional training data. The new **`aiq.experimental.inference_time_scaling`** package codifies this idea as four pluggable *strategy* types (Search ▶ Editing ▶ Scoring ▶ Selection) that operate on a lightweight `ITSItem` record.  Developers can compose these strategies manually or use several **pre‑built ITS functions** that wire everything up automatically.  Adding your own strategy is as simple as (1) writing a config subclass, (2) implementing a `StrategyBase` child and (3) registering it with the `@register_its_strategy` decorator.  The remainder of this document explains each step in detail.

---

## 2. Core design

### 2.1 Strategy pipeline

| Stage         | Purpose                                                      | Examples                                                                                                                                      |
| ------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Search**    | Generate many alternative plans, prompts or tool invocations | `single_shot_multi_plan`, `multi_llm_plan`, `multi_query_retrieval_search`                                                                    |
| **Editing**   | Refine or transform the candidates                           | `iterative_plan_refinement`, `llm_as_a_judge_editor`, `motivation_aware_summarization`                                                        |
| **Scoring**   | Assign a numeric quality score                               | `llm_based_plan_scorer`, `llm_based_agent_scorer`, `motivation_aware_scorer`                                                                  |
| **Selection** | Down‑select or merge                                         | `best_of_n_selector`, `threshold_selector`, `llm_based_plan_selector`, `llm_based_output_merging_selector`, `llm_based_agent_output_selector` |

A **pipeline type** tells a strategy *where* it is used:

```text
PipelineTypeEnum = { PLANNING, TOOL_USE, AGENT_EXECUTION }
StageTypeEnum    = { SEARCH, EDITING, SCORING, SELECTION }
```

Each strategy advertises

```python
supported_pipeline_types() -> list[PipelineTypeEnum]
stage_type()                -> StageTypeEnum
```

so mismatches are caught at build‑time.

### 2.2 `StrategyBase`

Every concrete strategy extends `StrategyBase`:

```python
class MyStrategy(StrategyBase):
    async def build_components(self, builder): ...
    async def ainvoke(
            self,
            items: list[ITSItem],
            original_prompt: str | None = None,
            agent_context:  str | None = None,
    ) -> list[ITSItem]:
        ...
```

*Implementation hint*: use the `Builder` helpers (`get_llm`, `get_function`, …) during `build_components` to resolve references just once and cache them.

### 2.3 `ITSItem`

A **single, interoperable record** passed between stages.

| Field      | Meaning                             |
| ---------- | ----------------------------------- |
| `input`    | Raw user task / tool args           |
| `output`   | Generated answer / tool result      |
| `plan`     | Execution plan (planning pipelines) |
| `feedback` | Review comments from editing stages |
| `score`    | Numeric quality metric              |
| `metadata` | Arbitrary auxiliary data            |
| `name`     | Tool name or other identifier       |

Because it is a `pydantic.BaseModel`, you get `.model_dump()` and validation for free.

---

## 3. Built‑in strategies

Below is a non‑exhaustive catalogue you can use immediately; refer to the inline doc‑strings for full parameter lists.

| Category  | Config class                                                    | One‑liner                                                                 |
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

---

## 4. Pre‑built ITS functions

AgentIQ ships higher‑level wrappers that hide all orchestration:

| Function                              | Use‑case                                                                                                            |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **`its_tool_wrapper_function`**       | Turn an arbitrary function into a *tool*; the wrapper asks an LLM to translate free‑text into structured arguments. |
| **`its_tool_orchestration_function`** | Accepts a list of tool invocations, optionally runs search/edit/score/select, then executes each tool concurrently. |
| **`execute_score_select_function`**   | Run a function *k* times, score each output, pick the best.                                                         |
| **`plan_select_execute_function`**    | End‑to‑end: plan → optionally edit/score → select plan → feed downstream agent.                                     |

These are declared in `aiq.experimental.inference_time_scaling.functions.*` and can be referenced in your `AIQConfig` just like any other function.

---

## 5. Creating and registering a new strategy

1. **Define a config model**

   ```python
   class MyStrategyConfig(ITSStrategyBaseConfig, name="my_strategy"):
       my_param: float = 0.5
   ```

2. **Implement the strategy**

   ```python
   from aiq.experimental.inference_time_scaling.models.strategy_base import StrategyBase
   class MyStrategy(StrategyBase):
       ...
   ```

3. **Register**

   ```python
   from aiq.cli.register_workflow import register_its_strategy

   @register_its_strategy(config_type=MyStrategyConfig)
   async def register_my_strategy(cfg: MyStrategyConfig, builder: Builder):
       strat = MyStrategy(cfg)
       await strat.build_components(builder)
       yield strat
   ```

That’s it — your strategy is now discoverable by `TypeRegistry` and can be referenced in `AIQConfig` fields.

---

## 6. Composing strategies in an `AIQConfig`

```python
from aiq.experimental.inference_time_scaling.models.search_config import SingleShotMultiPlanConfig
from aiq.experimental.inference_time_scaling.models.selection_config import BestOfNSelectionConfig
from aiq.experimental.inference_time_scaling.functions.execute_score_select_function import (
    ExecuteScoreSelectFunctionConfig,
)

cfg = AIQConfig(
    its_strategies = {
        "planner": SingleShotMultiPlanConfig(
            num_plans=5,
            planning_llm="nim_llm",
        ),
        "selector": BestOfNSelectionConfig(),
    },
    functions = {
        "qa": ExecuteScoreSelectFunctionConfig(
            augmented_fn="my_domain.llm_chat",
            scorer=None,
            selector="selector",
            num_executions=3,
        ),
    },
    workflow = ...,
)
```

The builder will:

1. Instantiate `planner` and `selector`.
2. Inject them into the function that needs them.
3. Guarantee type compatibility at build‑time.

---

## 7. Extending tools and pipelines

* **Multiple stages**: Nothing stops you from chaining *search → edit → search* again, as long as each stage returns `List[ITSItem]`.
* **Streaming**: Strategies themselves are non‑streaming, but you can wrap a streaming LLM in an ITS pipeline by choosing an appropriate pre‑built function (e.g., `plan_select_execute_function` keeps streaming support if the downstream agent streams).
* **Debugging**: Log levels are respected through the standard `logging` module; export `AIQ_LOG_LEVEL=DEBUG` for verbose traces, including every intermediate `ITSItem`.

---

## 8. Testing your strategy

*Write isolated unit tests* by instantiating your config and strategy directly, then calling `ainvoke` with hand‑crafted `ITSItem` lists.  See the companion `tests/` directory for reference tests on `ThresholdSelector` and `BestOfNSelector`.

---

## 9. Further reading

* Conceptual overviews of ITS techniques ([sebastianraschka.com][1], [ve3.global][6], [arxiv.org][2], [research.ibm.com][4])
* Practical business cases and cost–accuracy trade‑offs ([redhat.com][3], [dominguezdaniel.medium.com][5], [youtube.com][7])
* Academic analyses of compute vs. quality ([arxiv.org][9])
* Industry commentary on the shift from *training* to *inference* scaling ([barrons.com][10], [reuters.com][8])

These external resources frame the scientific motivations behind the concrete API you now have at your disposal.

---

Happy scaling!

[1]: https://sebastianraschka.com/blog/2025/state-of-llm-reasoning-and-inference-scaling.html?utm_source=chatgpt.com "Inference-Time Compute Scaling Methods to Improve Reasoning ..."
[2]: https://arxiv.org/html/2504.00294v1?utm_source=chatgpt.com "Inference-Time Scaling for Complex Tasks: Where We Stand ... - arXiv"
[3]: https://www.redhat.com/en/blog/smarter-enterprise-ai-inference-time-scaling?utm_source=chatgpt.com "Unlocking smarter enterprise AI with inference-time scaling - Red Hat"
[4]: https://research.ibm.com/blog/inference-scaling-reasoning-ai-model?utm_source=chatgpt.com "Reasoning in Granite 3.2 using inference scaling - IBM Research"
[5]: https://dominguezdaniel.medium.com/understanding-inference-time-compute-c6c6ca2e17a8?utm_source=chatgpt.com "Understanding Inference-Time Compute | by Daniel Dominguez"
[6]: https://www.ve3.global/inference-time-scaling-the-next-frontier-in-ai-performance/?utm_source=chatgpt.com "Inference-Time Scaling: The Next Frontier in AI Performance - VE3"
[7]: https://www.youtube.com/watch?v=seKLD3tdj9g&utm_source=chatgpt.com "What is inference-time scaling? - YouTube"
[8]: https://www.reuters.com/technology/artificial-intelligence/openai-rivals-seek-new-path-smarter-ai-current-methods-hit-limitations-2024-11-11/?utm_source=chatgpt.com "OpenAI and rivals seek new path to smarter AI as current methods hit limitations"
[9]: https://arxiv.org/abs/2504.02495?utm_source=chatgpt.com "[2504.02495] Inference-Time Scaling for Generalist Reward Modeling"
[10]: https://www.barrons.com/articles/ai-scaling-laws-chips-nvda-amd-intel-stock-prices-d20e96d8?utm_source=chatgpt.com "The AI Race Is Entering a New Stage. Nvidia Won't Be the Only Winner This Time."