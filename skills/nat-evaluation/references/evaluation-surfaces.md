# NeMo Agent toolkit Evaluation Surfaces: Legacy (`IntermediateStep`) And ATIF

Use this page before choosing evaluator metrics or writing eval YAML when the user mentions ATIF, `workflow_output_atif.json`, `write_atif_workflow_output`, `enable_atif_evaluator`, metadata/state loss, canonical trace shape, or evaluator support by lane.

Directionally, **ATIF is the canonical trajectory format NeMo Agent toolkit is moving toward**. Prefer ATIF when the evaluator supports it or when a downstream consumer needs a stable, standard trace shape. Fall back to **legacy `nat eval` (`IntermediateStep`)** when the chosen evaluator does not support ATIF yet or when scoring needs legacy-only state such as `full_dataset_entry`. In current NeMo Agent toolkit config, ATIF still has to be enabled explicitly with `write_atif_workflow_output` and/or evaluator-specific ATIF support such as `enable_atif_evaluator`.

## Quick Decision

| User Intent | Surface | User Experience | State/Metadata |
|:------------|:--------|:----------------|:---------------|
| "Evaluate my NeMo Agent toolkit workflow with built-in evaluators" | ATIF where supported; legacy (`IntermediateStep`) fallback | Configure `eval.general.dataset` and `eval.evaluators`; enable ATIF only for evaluator families that support it; run `nat eval`; inspect `<evaluator>_output.json` and, when enabled, `workflow_output_atif.json`. | ATIF gives canonical shape; legacy preserves full `EvalInputItem` state. |
| "I want normal NeMo Agent toolkit eval plus an ATIF trace artifact" | Legacy `nat eval` (`IntermediateStep`) + ATIF artifact | Add `write_atif_workflow_output: true`; still run normal `nat eval`; also inspect `workflow_output_atif.json`. | Legacy evaluators keep `IntermediateStep` state; ATIF artifact is a projection. |
| "I want to evaluate using ATIF-shaped trajectories" | ATIF evaluator lane | Add evaluator-specific ATIF flags only where supported, usually `enable_atif_evaluator: true`. | ATIF samples may omit legacy-only fields. |
| "I have ATIF samples or need an ATIF-native custom evaluator" | Standalone ATIF evaluation | Use `nvidia-nat-eval` / ATIF-native evaluator APIs rather than full workflow execution. | Depends on how the samples are constructed. |

Care about ATIF when one of these is true:

- The user needs a `workflow_output_atif.json` artifact.
- The user or downstream consumer expects a canonical ATIF trace shape.
- The user needs an ATIF-shaped trajectory contract.
- The user is writing or running ATIF-native custom evaluators.
- The user is comparing a migrated evaluator lane against the legacy lane.

If the scoring logic needs dataset metadata, custom labels, tenant/user fields, or full original rows, prefer the legacy (`IntermediateStep`) lane today unless the ATIF path is explicitly enriched for those fields.

## Terminology

| Term | Meaning |
|:-----|:--------|
| ATIF | Canonical trajectory format NeMo Agent toolkit is moving toward for stable cross-system trace shape. |
| Legacy `nat eval` (`IntermediateStep`) | Default NeMo Agent toolkit eval runtime using `EvalInputItem`; trajectories are lists of `IntermediateStep` objects. |
| ATIF output artifact | An additional ATIF-shaped file, `workflow_output_atif.json`, written by `nat eval` for export/debugging. |
| ATIF evaluator lane | An evaluator path that receives `AtifEvalSample` payloads via `evaluate_atif_fn` instead of legacy `EvalInputItem` payloads. |
| Standalone ATIF evaluation | Evaluation outside normal config-driven workflow execution, usually via `nvidia-nat-eval` and ATIF-native custom evaluators. |

## Config Patterns

### Current Default: Legacy `nat eval` (`IntermediateStep`)

```yaml
eval:
  general:
    dataset:
      _type: json
      file_path: data/golden_dataset.json
  evaluators:
    accuracy:
      _type: ragas
      metric: AnswerAccuracy
      llm_name: judge_llm
```

### Legacy (`IntermediateStep`) Eval With ATIF Artifact

```yaml
eval:
  general:
    output:
      write_atif_workflow_output: true
```

This preserves normal `nat eval` behavior and additionally writes `workflow_output_atif.json`.

### ATIF Evaluator Lane

Use this only for evaluators that expose the flag in the installed NeMo Agent toolkit version:

```yaml
eval:
  general:
    output:
      write_atif_workflow_output: true
  evaluators:
    groundedness:
      _type: ragas
      metric: ResponseGroundedness
      llm_name: judge_llm
      enable_atif_evaluator: true
```

Confirm support before adding `enable_atif_evaluator`. Unsupported flags make the config misleading or invalid. As ATIF coverage expands, prefer ATIF-native evaluator lanes when they preserve the state the evaluator needs.

## Evaluator Support Matrix

Verify with `nat info components -t evaluator` and source/docs for the installed version. In the local NeMo Agent toolkit source reviewed for this skill:

| Evaluator | Package | Legacy (`IntermediateStep`) | ATIF | Requires Trajectory | Requires Reference | Optimization Suitability | Notes |
|:----------|:--------|:------:|:----:|:--------------------|:-------------------|:-------------------------|:------|
| `ragas` | `nvidia-nat[ragas]` | Yes | Optional | Metric-dependent; context metrics need retrieved context from steps/ATIF observations | Metric-dependent | Good for RAG objectives when one metric maps to one objective | `enable_atif_evaluator: true` exists. |
| `trajectory` | `nvidia-nat[langchain]` | Yes | Optional | Yes | No; `expected_trajectory` is optional | Good for tool-path objectives; judge noise may require reps | `enable_atif_evaluator: true` exists. |
| `tunable_rag_evaluator` | `nvidia-nat[langchain]` | Yes | Optional | Legacy no; ATIF lane needs user input recoverable from ATIF trajectory | Default scoring uses expected answer; custom prompt is user-defined | Good when the prompt returns a calibrated numeric score | `enable_atif_evaluator: true` exists. |
| `langsmith_judge` | `nvidia-nat[langchain]` | Yes | No observed ATIF lane | No | Optional, depending on prompt | Good for custom final-response objectives if configured for numeric scoring | Requires judge model with structured output. |
| `langsmith` | `nvidia-nat[langchain]` | Yes | No observed ATIF lane | No | Depends on openevals metric | Good for deterministic string/similarity objectives | Algorithmic; no judge LLM. |
| `langsmith_custom` | `nvidia-nat[langchain]` | Yes | No observed ATIF lane | User-defined | User-defined | Good if the custom function emits stable numeric scores | Wraps existing LangSmith-compatible functions. |
| Profiler runtime evaluators | `nvidia-nat-profiler` | Yes | Yes | Yes | No | Good for latency/token objectives; usually `direction: minimize` | Distinct from `eval.general.profiler` artifacts. |
| `red_teaming_evaluator` | `nvidia-nat-security` | Yes | No observed ATIF lane | Yes | Uses expected behavior from the dataset | Better as a security diagnostic than a primary optimizer objective; use `direction: minimize` when higher means attack success | Security workflow also uses middleware/runner. |
| Custom `BaseEvaluator` | Project/package-specific | Yes | No | User-defined | User-defined | Good if scores are numeric, stable, and failures are explicit | Uses `EvalInputItem`. |
| Custom ATIF evaluator | `nvidia-nat-eval` or project/package-specific | No | Yes | Usually yes | User-defined | Use when downstream consumers accept ATIF lane and metadata needs are handled | Implement `evaluate_atif_fn` / `AtifBaseEvaluator`. |

## State And Metadata Implications

Legacy (`IntermediateStep`) and ATIF do not expose exactly the same shape.

- Legacy evaluators receive `EvalInputItem`, including `id`, `input_obj`, `expected_output_obj`, `output_obj`, `expected_trajectory`, `trajectory`, and `full_dataset_entry`. The `expected_trajectory` and `trajectory` fields are `list[IntermediateStep]`.
- ATIF evaluators receive `AtifEvalSample`, including `item_id`, `trajectory`, `expected_output_obj`, `output_obj`, and `metadata`.
- In the reviewed NeMo Agent toolkit adapter, `metadata` is currently populated as `{}` when converting from legacy eval input.

If the scoring logic needs the full dataset entry, choose legacy unless the ATIF path is explicitly enriched for those fields. If the downstream consumer primarily needs a canonical, stable trace shape, prefer ATIF and document any missing legacy state explicitly.

### Lossy Conversion: What Changes If You Choose ATIF

In current NeMo Agent toolkit, ATIF is built from the legacy `EvalInputItem` and its `trajectory: list[IntermediateStep]`. That conversion is useful because it produces a canonical trajectory shape, but it is not a full copy of the NeMo Agent toolkit eval row.

Not lost in the reviewed adapter:

- Record identity: `EvalInputItem.id` becomes `AtifEvalSample.item_id`.
- Reference and actual answers: `expected_output_obj` and `output_obj` are copied onto the ATIF sample.
- Actual trajectory semantics: workflow input/output, LLM outputs, tool/function calls, observations, selected ancestry/timing, and token metrics are projected into `ATIFTrajectory`.

Lost or reshaped when using ATIF:

| Legacy state | ATIF behavior | End-user impact |
|:-------------|:--------------|:----------------|
| `input_obj` | No top-level field on `AtifEvalSample`; the converter may emit a user step from `WORKFLOW_START.data.input` | Structured dataset input is not guaranteed available to ATIF evaluators. |
| `full_dataset_entry` | Not carried by the reviewed adapter; `metadata={}` | Dataset columns such as category, tenant, risk class, labels, or difficulty are unavailable unless ATIF metadata is enriched. |
| `expected_trajectory` | No corresponding field on `AtifEvalSample` | Expected tool/path comparison remains a legacy-lane concern unless represented separately. |
| Raw `IntermediateStep` event stream | Converted into canonical ATIF steps | ATIF is not a byte-for-byte event log. Use legacy for exact event-boundary assertions. |
| START/CHUNK/SPAN/CUSTOM/TTC events | Usually skipped unless they contribute to a canonical ATIF step | Streaming chunks, span chunks, custom events, and TTC-specific events may not be visible to ATIF evaluators. |
| `parent_id`, raw `UUID` | Not first-class ATIF fields; selected ancestry/timing goes into `extra`, and tool call ids are generated from `UUID` | Use legacy if evaluator logic depends on raw NeMo Agent toolkit ids. |
| `tags`, arbitrary `metadata`, `TraceMetadata` | Only selected pieces are mapped, such as tool definitions | Do not assume custom trace metadata survives conversion. |
| Some usage fields | Token counts mostly map; fields like `num_llm_calls` and `seconds_between_calls` are not mapped today | Use legacy/profiler data for those operational details. |

What you miss if you choose legacy only:

- No canonical ATIF trajectory contract for downstream consumers.
- No `workflow_output_atif.json` artifact unless `write_atif_workflow_output: true` is enabled.
- No ATIF-native evaluator lane unless the evaluator supports it and the config enables it.

Practical rule:

- Prefer ATIF when the evaluator or downstream consumer needs a canonical trajectory contract.
- Use legacy (`IntermediateStep`) when the evaluator needs full dataset state, expected trajectory, arbitrary trace metadata, or exact NeMo Agent toolkit event boundaries.
- If both are needed, run legacy eval with `write_atif_workflow_output: true`; treat ATIF as the exported projection and legacy as the richer NeMo Agent toolkit eval state.

For downstream consumers such as `nat optimize`, finetuning reward flows, or red-team workflows, the state delta matters because the evaluator result may become an objective, reward, or scenario score. See [`evaluation-contract.md`](evaluation-contract.md).

## How To Present This To Users

Do not ask "ATIF or legacy?" as the first question for a routine NeMo Agent toolkit eval. Make both the direction and current tradeoff clear:

> ATIF is the canonical format NeMo Agent toolkit is moving toward. I will use ATIF when the evaluator supports it or a downstream consumer expects it, and fall back to legacy `nat eval` (`IntermediateStep`) only when ATIF coverage or state parity is not sufficient.

If the user asks about evaluator support, answer in terms of the support matrix above and verify against the installed NeMo Agent toolkit version.
