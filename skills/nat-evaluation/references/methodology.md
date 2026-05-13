# The Agent Evaluation Framework — 8-Step Methodology

An 8-step mental model for evaluating any NeMo Agent Toolkit agent. Use this guide alongside [`agent-eval-framework.md`](agent-eval-framework.md) (the conceptual framework) and the per-evaluator pages under [`evaluators/`](evaluators/).

This guidance is grounded in the Agent Evaluation Framework ([agent-eval-framework.md](agent-eval-framework.md)). The framework establishes:

## Five Dimensions of Agent Behavior

Every agent exhibits behavior across five dimensions. Your evaluator suite should cover as many as are relevant:

| Dimension | What Can Go Wrong | How to Evaluate |
| --------- | ----------------- | --------------- |
| **Reasoning** | Logical errors in chain-of-thought, incorrect conclusions | RAGAS AnswerAccuracy, custom LLM-as-judge |
| **Planning** | Inefficient action sequences, loops, abandoned paths | Trajectory evaluation, step count analysis, profiler |
| **Tool Use** | Wrong tools, incorrect parameters, misinterpreted results | Trajectory evaluation, custom tool selection checks |
| **Memory** | Context overflow, irrelevant retrieval, forgetting details | RAGAS ContextRelevance, ResponseGroundedness |
| **Adaptation** | Overfitting to history, inconsistent persona | Consistency across runs, error recovery evaluation |

## The Evaluation Mental Model

Agent evaluation replaces binary pass/fail with continuous quality scores, exact-match assertions with semantic equivalence, and one-shot testing with continuous monitoring. For the full mindset-shift table and philosophical framing, see [`agent-eval-framework.md § 1.4 The Evaluation Mindset Transformation`](agent-eval-framework.md#14-the-evaluation-mindset-transformation).

## Step 0: Gather Context

Before generating configs, understand what you're evaluating. **Use every source available:**

### Discovery step: list registered evaluators

Before writing any eval config, run `nat info components -t evaluator` to see the evaluator `_type` values actually registered in the current environment. Do not assume an evaluator (e.g. `ragas`, `trajectory`, `tunable_rag_evaluator`, `langsmith_judge`) is available — confirm it is, or install the matching extra (see [../../nat-installation/references/installation.md](../../nat-installation/references/installation.md)). The same applies to dataset readers (`nat info components -t dataset`) and any custom evaluators registered via `register.py`.

### Evaluation surface and downstream consumers

Before picking evaluator metrics, decide whether the user needs native legacy NeMo Agent Toolkit evaluation (`IntermediateStep`), an ATIF output artifact, or ATIF-native evaluator execution. Read [evaluation-surfaces.md](evaluation-surfaces.md) when the task mentions ATIF, `workflow_output_atif.json`, `write_atif_workflow_output`, `enable_atif_evaluator`, canonical trace shape, evaluator support by lane, or possible metadata/state loss.

ATIF is the canonical trajectory format NeMo Agent Toolkit is moving toward, but ATIF state parity and evaluator coverage are still in flight. Prefer ATIF when the user or downstream consumer needs the canonical shape; prefer legacy (`IntermediateStep`) when current evaluator support or full legacy eval state matters more.

If evaluator results will also feed `nat optimize`, GA prompt optimization, finetuning reward/validation flows, red-team workflows, or profiler runtime objectives, read [evaluation-contract.md](evaluation-contract.md) for the score/name/reasoning contract before finalizing evaluator names and metrics.

**From the codebase:**

- Read NeMo Agent Toolkit workflow config YAML for `workflow._type` (react_agent, router, sequential), tools (`functions:`), LLM config (`llms:`), and system prompt
- Check for existing `eval_config.yml`, datasets, or custom evaluators
- Look at `register.py` for custom evaluator registrations
- Check for existing output directories with previous eval results

**From agent traces (if the user provides them):**
Traces are the richest context source. If the user shares NeMo Agent Toolkit output files (`workflow_output.json`, profiler data, or log files):

- Extract actual tool call sequences from `intermediate_steps`
- Identify failure patterns — tool errors, empty retrievals, hallucinated responses, loops
- Use `standardized_data_all.csv` profiler output for latency and token usage patterns
- Use trace data to inform dataset categories (failures become edge case examples)

**From the user's description:**

- Agent purpose, domain, and target users
- Known pain points or failure modes
- Business-level priorities

Summarize your understanding in a brief context table. Only ask for what you cannot determine from configs, traces, or their message.

### Agent Type Adaptations

| Agent Type | NeMo Agent Toolkit Config | Key Adaptations |
| ---------- | ---------- | --------------- |
| **RAG chatbot** | `react_agent` with `webpage_query` | Weight RAGAS AnswerAccuracy + ResponseGroundedness. Add ContextRelevance. Focus dataset on retrieval-heavy cases. |
| **Multi-agent workflow** | `router` or `sequential` | Trajectory evaluation critical for delegation. Add routing accuracy evaluator. |
| **Agent-to-Agent (A2A)** | `nat a2a serve` | Dataset should include cross-agent coordination tasks. Trajectory captures delegation decisions. |
| **Conversational / multi-turn** | `react_agent` with memory | Context retention across turns, goal completion rate. |
| **Safety-critical** | Any workflow type | Safety threshold = 1.0. 30%+ adversarial examples. Multiple safety evaluator layers. |

## Step 1: Define Success Criteria & Quality Dimensions

Identify 3-5 quality dimensions and set numeric thresholds. Infer reasonable defaults from the agent type.

### Quality Dimensions Catalog

**Response Quality:**

| Dimension | Description | NeMo Agent Toolkit Evaluator |
| --------- | ----------- | ------------- |
| Correctness | Factually accurate outputs matching ground truth | `ragas` / `AnswerAccuracy` |
| Groundedness | Response grounded in retrieved context, no hallucination | `ragas` / `ResponseGroundedness` |
| Context Relevance | Retrieved context is relevant to the question | `ragas` / `ContextRelevance` |
| Completeness | Addresses all parts of the question | Custom LLM-as-judge evaluator |
| Coherence | Logically structured, clear communication | Custom LLM-as-judge evaluator |

**Agent Behavior:**

| Dimension | Description | NeMo Agent Toolkit Evaluator |
| --------- | ----------- | ------------- |
| Trajectory Quality | Correct tools in correct order, no wasted steps | `trajectory` evaluator |
| Tool Accuracy | Selects and uses tools correctly | Custom rule-based evaluator |
| Delegation Accuracy | Routes to correct sub-agent (multi-agent) | Custom rule-based evaluator |
| Error Recovery | Handles failures and retries gracefully | Trajectory on error scenarios |

**Safety & Compliance:**

| Dimension | Description | NeMo Agent Toolkit Evaluator |
| --------- | ----------- | ------------- |
| Safety | No harmful, inappropriate, or leaked outputs | Custom `SafetyEvaluator` (rule-based) |
| PII Protection | No exposure of personal information | Custom rule-based evaluator |
| Prompt Injection Resistance | Refuses manipulation attempts | Custom rule-based + adversarial dataset |
| Scope Adherence | Stays within designated capabilities | Custom rule-based evaluator |
| Red-Team Robustness | Attack scenarios fail to induce unsafe tool use, leakage, or policy bypass | `red_teaming_evaluator` + red-team runner/middleware |

**Operational (via NeMo Agent Toolkit Profiler):**

| Dimension | Description | NeMo Agent Toolkit Feature |
| --------- | ----------- | ----------- |
| Latency | Response time within SLA | `profiler.workflow_runtime_forecast` |
| Token Efficiency | Reasonable token usage per task | `profiler.compute_llm_metrics` |
| Token Uniqueness | Cache-friendliness of requests | `profiler.token_uniqueness_forecast` |
| Consistency | Stable results across similar inputs | Variance across k runs (variance@k) |

### Threshold Levels

For each chosen dimension, set three tiers:

| Level | Purpose | Example |
| ----- | ------- | ------- |
| **Minimum** | Below this = don't deploy | AnswerAccuracy > 0.85 |
| **Target** | Production readiness goal | AnswerAccuracy > 0.93 |
| **Stretch** | Aspirational | AnswerAccuracy > 0.97 |

Infer sensible defaults: safety should always be near 1.0, correctness typically 0.85+ minimum, efficiency metrics depend on the use case.

### Determine Reference Availability

Assess what ground truth data exists — this shapes which evaluators you can use:

| Availability | Description | Evaluator Impact |
| ------------ | ----------- | ---------------- |
| **Full Reference** | Correct answers for all test cases | Use reference-based: `ragas` / `AnswerAccuracy`, trajectory matching |
| **Partial Reference** | Reference outputs for some scenarios | Reference-based where available; fall back to reference-free for the rest |
| **No Reference** | No ground truth (e.g., production traffic) | Reference-free only: `ResponseGroundedness`, `ContextRelevance`, `trajectory` (LLM judge), custom safety |

> This assessment directly shapes your evaluator suite. Offline evaluation (Step 7) uses reference-based metrics; online uses reference-free only. Planning this upfront avoids rework when you split configs later.

### Failure Modes

Document failure modes by severity based on what the agent does:

- **High**: Harm, data loss, major user impact (hallucinated facts, leaked credentials, wrong medical/legal/financial advice)
- **Medium**: Degraded experience but recoverable (wrong tool selected, incomplete answer, unnecessary steps)
- **Low**: Minor issues (verbose response, redundant tool call, suboptimal formatting)

## Step 2: Build the Evaluation Golden Dataset

Create a dataset for NeMo Agent Toolkit evaluation. Start small (10-20 examples), cover all relevant categories:

| Category | Purpose | When to Include |
| -------- | ------- | --------------- |
| **Happy Path** | Standard successful interactions | Always |
| **Edge Cases** | Boundary conditions, unusual inputs | Always |
| **Adversarial** | Prompt injection, data extraction attempts | Always (especially safety-critical) |
| **Ambiguous** | Multiple valid interpretations | When agent handles open-ended queries |
| **Multi-Tool** | Queries requiring 2+ tools | When agent has multiple tools |
| **Error Recovery** | Agent handles failures gracefully | When tools can fail |
| **Multi-Turn** | Context-dependent follow-ups | When agent handles conversations |
| **Out-of-Scope** | Queries outside agent's capabilities | When scope boundaries matter |

### Dataset Schema

NeMo Agent Toolkit uses the field names `question` and `answer` — but these are generic input/output fields, not limited to Q&A chatbots. Map them to whatever your agent does:

| NeMo Agent Toolkit Field | What It Represents | Examples Across Agent Types |
| --------- | ------------------ | --------------------------- |
| **question** | The agent's input — any task, query, or instruction | A user question, a code review request, a data extraction task, a planning prompt |
| **answer** | The expected reference output (optional but recommended) | A correct answer, expected code output, extracted schema, generated plan |
| **metadata** | Tags for segmented analysis | `category`, `difficulty`, `expected_tools`, `scenario` |

Think of `question` as **input** and `answer` as **expected output** — they work for any agent type.

> `nat eval` passes each `question` directly to the workflow with no preprocessing. If the agent preprocesses or reformats input before passing it to the workflow, the `question` field must already contain the preprocessed version.

### NeMo Agent Toolkit Dataset Formats

NeMo Agent Toolkit supports JSON (recommended), JSONL, CSV, Parquet, and XLS. For YAML snippets per format, custom parsers, filtering, and field-name remapping, see [`code-patterns.md § Dataset Config Options`](code-patterns.md#dataset-config-options). Generate the dataset file with examples tailored to this specific agent's tools and domain.

### Building Your Dataset

**If the user has existing data** (QA pairs, support tickets, test cases, trace exports), incorporate those as the foundation and augment with edge cases and adversarial examples.

**If starting from scratch**, generate examples that cover:

1. Every tool the agent has (at least 2-3 examples per tool)
2. Multi-tool scenarios if the agent has 2+ tools
3. At least 2-3 adversarial examples (prompt injection, scope violations)
4. Error recovery scenarios (what if a tool returns empty or fails?)

**Bootstrapping larger datasets** — use LLM-generated candidates + human validation:

1. Generate candidates with an LLM given tool descriptions and category requirements
2. Human expert validates and corrects each candidate
3. Augment with adversarial examples
4. After first eval run, use low-scoring examples to identify gaps

## Step 3: Design the Evaluator Suite

Map quality dimensions to concrete NeMo Agent Toolkit evaluators using the Test Pyramid:

### The Agent Evaluation Test Pyramid

```text
              ┌─────────────────────────────┐
              │  L4: End-to-End Scenarios    │  ragas / AnswerAccuracy
              │  (Most realistic)            │  Full task completion
              ├─────────────────────────────┤
              │  L3: Trajectory Evaluation   │  trajectory evaluator
              │  (Decision quality)          │  Tool sequence quality
              ├─────────────────────────────┤
              │  L2: Component & Tool        │  ragas / ResponseGroundedness
              │  (Single capabilities)       │  ragas / ContextRelevance
              ├─────────────────────────────┤
              │  L1: Foundation              │  Custom SafetyEvaluator
              │  (LLM response quality)      │  Custom CompletenessEvaluator
              └─────────────────────────────┘
```

### NeMo Agent Toolkit Built-In Evaluator Types

| NeMo Agent Toolkit `_type` | Metric / Mode | What It Measures | Requires Reference? |
| ----------- | ------------- | ---------------- | ------------------- |
| `ragas` | `AnswerAccuracy` | Semantic accuracy vs. reference answer | Yes |
| `ragas` | `ResponseGroundedness` | Response supported by retrieved context | No |
| `ragas` | `ContextRelevance` | Retrieved context relevant to question | No |
| `trajectory` | *(overall)* | Quality of full tool-call sequence | Uses LLM judge |
| `tunable_rag_evaluator` | *(custom dimensions)* | Customizable LLM-as-judge with adjustable scoring weights and optional custom judge prompt | No |
| `langsmith_judge` | *(prompt name or template)* | LLM-as-judge via openevals prebuilt or custom prompt | Optional |
| `langsmith` | *(openevals short name)* | Built-in openevals metric (e.g. `exact_match`, `levenshtein_distance`) | Depends on metric |
| `langsmith_custom` | *(dotted path)* | Any LangSmith-compatible evaluator function | Depends on fn |
| profiler runtime evaluators | latency / tokens / LLM calls | Runtime metrics as evaluator scores | No |
| `red_teaming_evaluator` | filtered trajectory judge | Red-team scenario success/failure | Uses expected behavior |
| Custom | User-defined via `BaseEvaluator` | Any domain-specific metric | User-defined |

**Additional RAGAS Metrics (available via `_type: ragas`):**

Beyond the three core metrics above, RAGAS exposes additional metrics — change the `metric` field in `eval_config.yml`:

| RAGAS Metric | What It Measures | Requires Reference? | When to Use |
| ------------ | ---------------- | ------------------- | ----------- |
| `FactualCorrectness` | Factual accuracy (accepts `mode` kwarg) | Yes | More granular than AnswerAccuracy — supports different strictness modes |
| `Faithfulness` | Claims inferable from given context | No | Stricter hallucination check than ResponseGroundedness |
| `AnswerRelevancy` | Whether the answer addresses the question | No | Detecting off-topic or evasive responses |
| `ContextPrecision` | Proportion of retrieved context that is relevant | Yes | Optimizing retrieval precision |
| `ContextRecall` | Whether all relevant info was retrieved | Yes | Ensuring retrieval completeness |
| `NoiseSensitivity` | Robustness to irrelevant context | Yes | RAG agents with noisy retrieval |

> Available metrics depend on your `ragas` package version. Check [RAGAS docs](https://docs.ragas.io/) for the full list.

**Discover evaluators available in the current NeMo Agent Toolkit installation:**

```bash
nat info components -t evaluator
```

Full config examples and field reference for each evaluator type:

- [`evaluators/evaluator-ragas.md`](evaluators/evaluator-ragas.md) — RAGAS metrics (requires `nvidia-nat[ragas]`)
- [`evaluators/evaluator-tunable-rag.md`](evaluators/evaluator-tunable-rag.md) — tunable RAG evaluator
- [`evaluators/evaluator-langsmith-judge.md`](evaluators/evaluator-langsmith-judge.md) — LLM-as-judge (requires `nvidia-nat[langchain]`)
- [`evaluators/evaluator-langsmith.md`](evaluators/evaluator-langsmith.md) — built-in openevals + custom evaluators (requires `nvidia-nat[langchain]`)
- [`evaluators/evaluator-trajectory.md`](evaluators/evaluator-trajectory.md) — trajectory evaluation
- [`evaluators/evaluator-red-teaming.md`](evaluators/evaluator-red-teaming.md) — red-team/security evaluator

For ATIF support, trajectory/reference requirements, and downstream suitability by evaluator family, see [evaluation-surfaces.md](evaluation-surfaces.md). For optimizer, finetuning, red-team, and profiler consumer implications, see [evaluation-contract.md](evaluation-contract.md).

### Evaluation Strategy Matrix

| What You Want to Measure | Evaluation Approach | NeMo Agent Toolkit Evaluator(s) | Pyramid Level |
| ------------------------ | ------------------- | ---------------- | ------------- |
| Does the agent solve the task? | Final Response | `ragas` / `AnswerAccuracy` | L4 |
| Does the agent pick the right tools? | Single Step | Custom rule-based tool selection check | L2 |
| Does the agent take an efficient path? | Trajectory | `trajectory` evaluator | L3 |
| Does the agent reason correctly? | Trajectory + Single Step | `trajectory` + custom step-level evaluator | L3 |
| Does the agent recover from errors? | Trajectory | `trajectory` on error-recovery examples | L3 |
| Is the response grounded? | Final Response | `ragas` / `ResponseGroundedness` | L2 |
| Is the response safe? | Final Response | Custom `SafetyEvaluator` | L1 |
| Does the agent resist red-team attacks? | Trajectory + Scenario | `red_teaming_evaluator` with red-team runner/middleware | L4 |

### Critical Design Rule

Use a **different LLM model** for evaluation than the agent uses. Configure separate `llms:` entries in `eval_config.yml` for evaluator judges. Set `max_tokens: 8` for RAGAS metrics, `max_tokens: 1024` for trajectory evaluation.

### Decomposing evaluators by quality dimension

If your evaluation judges multiple **independent** facets of the agent's output, prefer **one evaluator per facet** over a single composite scorer. Two facets are independent when the agent's output can be correct on one and wrong on the other. This matters most when the evaluator suite feeds a downstream consumer such as `nat optimize` or finetuning rewards.

Reasons to decompose:

- The optimizer can search across facets simultaneously (`direction: maximize` per `eval_metrics` entry — see the Multi-Objective Optimization subsection under Optimization below) instead of being blind to trade-offs hidden inside a single scalar.
- Per-facet scores reveal *which* facet improved or regressed when you compare two configs; a composite score absorbs facet-level damage and can look fine on the surface while one facet has degraded.
- When the agent's output format breaks (parse failure, missing fields), per-facet evaluators register zero across the board, making the failure visible. A composite that gives partial credit for any field present can hide format breakage and let the optimizer drift.

Stop decomposing when facets are not actually independent (e.g. one is a strict function of the other) — extra evaluators add noise and judge cost without adding signal. As a rule of thumb: 2–4 evaluators when the task has multiple distinct quality dimensions; 1 when it has a single objective.

## Step 4: Build & Configure Evaluators

Generate the NeMo Agent Toolkit eval config and custom evaluator code. See [code-patterns.md](code-patterns.md) for all templates.

### NeMo Agent Toolkit Evaluation Architecture

| Layer | Config Section | What It Contains |
| ----- | -------------- | ---------------- |
| **LLMs** | `llms` | LLM endpoints for evaluators (separate from agent's LLM) |
| **Dataset** | `eval.general.dataset` | Path and format of golden dataset |
| **Evaluators** | `eval.evaluators` | Named evaluators with `_type` and config; these names may also become optimizer objectives, reward signals, or red-team scores |
| **Profiler** | `eval.general.profiler` | Performance profiling artifacts; profiler runtime evaluators can also produce evaluator scores |

### Parallelism

`nat eval` processes dataset items concurrently via `eval.general.max_concurrency`.

**Set `max_concurrency` to 8.** Decrease only if a run produces 429 rate-limit errors or if explicitly told. DO NOT lower as a precaution.

Wall-clock ≈ `dataset_size × per-item-time / max_concurrency`.

Do NOT shrink the dataset to reduce wall-clock. Raise parallelism first.

### Custom Evaluator Pattern

NeMo Agent Toolkit custom evaluators follow this structure:

1. Config class extending `EvaluatorBaseConfig`
2. Evaluator class extending `BaseEvaluator` with `evaluate_item(EvalInputItem) -> EvalOutputItem`
3. Registration via `@register_evaluator` decorator
4. Import in `register.py` for NeMo Agent Toolkit discovery

> **API note:** NeMo Agent Toolkit is actively evolving. The import paths and class signatures in [code-patterns.md](code-patterns.md) are based on the current API. If you hit `ImportError` or signature mismatches, fetch the latest custom evaluator docs via the External Documentation links above — they are the source of truth for the current API.

Generate evaluators specific to the agent's domain. See [code-patterns.md](code-patterns.md) for safety evaluator, LLM-as-judge, and routing evaluator templates.

### External Evaluator Integration

Custom evaluators have full Python capabilities inside `evaluate_item()` — you can call any external library. NeMo Agent Toolkit also supports automatic evaluator discovery from external packages via its plugin system:

| Integration Method | How It Works | When to Use |
| ------------------ | ------------ | ----------- |
| **`register.py` imports** | Import and register evaluators in a local `register.py` | Project-specific evaluators alongside agent code |
| **Entry points** | External packages register via `nat.plugins` entry point in `pyproject.toml` | Sharing evaluators across teams or projects |
| **Inside `evaluate_item()`** | Call any external Python library (LangChain, LangSmith, HTTP APIs) | Wrapping existing evaluation infrastructure into NeMo Agent Toolkit |

```python
# External package entry point (pyproject.toml)
[project.entry-points."nat.plugins"]
my_evaluators = "my_package.evaluators:register_all"
```

## Step 5: Run Experiments & Capture Results

```bash
# Validate config first
nat validate --config_file=configs/eval_config.yml

# Run evaluation (default parallelism)
nat eval --config_file=configs/eval_config.yml \
    --override eval.general.max_concurrency 8

# You can lower it if you encounter rate limits
nat eval --config_file=configs/eval_config.yml \
    --override eval.general.max_concurrency 1

# Multiple reps for non-deterministic agents
nat eval --config_file=configs/eval_config.yml --reps 3

# If your workflow is interrupted, you can resume an interrupted evaluation (skips already-completed entries)
nat eval --config_file=configs/eval_config.yml --skip_completed_entries --dataset output/workflow_output.json

# Offline: evaluate pre-generated answers without re-running the workflow
nat eval --config_file=configs/eval_config.yml --skip_workflow --dataset output/workflow_output.json
```

> `nat eval` is a long-running command. Runtime depends on dataset size, workflow complexity, and number of reps. Run it with a generous timeout or none.

### NeMo Agent Toolkit Output Artifacts

| File | Contents |
| ---- | -------- |
| `workflow_output.json` | Per-sample: question, expected_answer, generated_answer, intermediate_steps |
| `<evaluator>_output.json` | Per-entry scores + reasoning + average |
| `config_original.yml` | Initial configuration file |
| `config_effective.yml` | Final configuration with CLI overrides applied |
| `config_metadata.json` | Execution metadata and arguments |
| `standardized_data_all.csv` | Profiler: latency, token counts, error flags per request |
| `workflow_profiling_metrics.json` | Aggregated profiler stats (means, percentiles) |
| `workflow_profiling_report.txt` | Human-readable profiler summary |
| `all_requests_profiler_traces.json` | Full per-request trace events |
| `inference_optimization.json` | Inference optimization signals |
| `gantt_chart.png` | Visual timeline of LLM/tool execution spans |

### Non-Determinism Handling

| Metric | Formula | Use When |
| ------ | ------- | -------- |
| **pass@k** | Pass if any of k runs succeeds | Measuring capability ceiling |
| **pass^k** | Pass if all of k runs succeed | Measuring production reliability |
| **mean@k** | Average across k runs | General quality assessment |

```bash
for i in 1 2 3; do
  nat eval --config_file=configs/eval_config.yml \
    --override eval.general.output_dir "./output/run_${i}/"
done
```

## Step 6: Compare & Iterate

Analyze results and generate a prioritized iteration plan:

1. **Score breakdown by category** — parse `<evaluator>_output.json`, segment by dataset metadata
2. **Identify lowest-scoring examples** — sort entries by score, find worst performers
3. **Check safety violations** — P0 critical, blocks deployment
4. **Use profiler data** — `standardized_data_all.csv` for latency bottlenecks, `gantt_chart.png` for execution timeline
5. **Generate iteration plan** — prioritized by severity (P0/P1/P2)

**Regression prevention:** After each fix, add the failure case to the golden dataset, re-run full eval, compare with baseline.

## Step 7: Establish Offline & Online Evaluation Cycles

### Offline (Pre-Deploy)

- Trigger: Every PR / pre-release
- Config: `eval_config.yml` with full suite (RAGAS + trajectory + custom)
- Data: Golden dataset with reference answers
- Decision: Block deploy if below threshold
- Generate: GitHub Actions workflow + threshold checker script

### Online (Production)

- Trigger: Continuous or sampled (5-10% of traffic)
- Config: `online_eval_config.yml` with **reference-free evaluators only**
- Data: Production trace exports
- Decision: Alert on anomalies
- Evaluators: ResponseGroundedness, ContextRelevance, trajectory, safety (NO AnswerAccuracy — no ground truth for live traffic)

## Step 8: Scale Your Evaluation Practice

Always include the maturity progression when discussing Steps 7-8.

### Maturity Progression

| Stage | Dataset | Evaluators | Automation | What to Focus On |
| ----- | ------- | ---------- | ---------- | ---------------- |
| **Prototype** | 10-20 examples | 1-2 RAGAS metrics | Manual `nat eval` | Get accuracy + safety working. Iterate fast. |
| **Alpha** | 50-100 examples | RAGAS + trajectory | PR-triggered | Add trajectory evaluators. Start CI/CD. |
| **Beta** | 200-500 examples | + custom evaluators | CI/CD gated | Full pyramid + profiler. Quality gates block deploys. |
| **Production** | 500+ examples | Full pyramid | Continuous + alerts | Online monitoring. Auto-add failures to dataset. |

### Scaling Checklist

- [ ] CI/CD integration (`nat eval` on every PR)
- [ ] Quality gates (threshold checker blocks deploys)
- [ ] Offline eval config (RAGAS + trajectory + custom evaluators + profiler)
- [ ] Online eval config (reference-free evaluators only)
- [ ] Custom evaluators registered via `@register_evaluator`
- [ ] Dataset growth pipeline (auto-add production failures)
- [ ] Automated alerts (Slack/PagerDuty on score drops)
- [ ] Profiler analysis (latency, token efficiency, gantt charts)

## Common Evaluation Pitfalls

| Pitfall | Better Approach |
| ------- | --------------- |
| Testing only happy paths | Include adversarial, edge cases, error recovery |
| Same LLM evaluating itself | Use different model in `eval.llms` vs workflow `llms` |
| Only RAGAS evaluators | Layer with trajectory + custom rule-based evaluators |
| Ignoring the profiler | Enable `profiler` section — it catches latency/token issues for free |
| Using AnswerAccuracy in production | Online = reference-free metrics only (ResponseGroundedness, ContextRelevance, trajectory) |
| Skipping `nat validate` before eval | Config errors waste eval time — validate first |
| Setting `max_concurrency` too low | Default to `8`; only decrease if you see 429 rate-limit errors |
| Generic evaluators for specialized agents | Build domain-specific evaluators via `BaseEvaluator` |
