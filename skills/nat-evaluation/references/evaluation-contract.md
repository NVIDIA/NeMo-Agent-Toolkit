# NeMo Agent Toolkit Evaluation Contract

Use this page when evaluator choice affects more than a one-off `nat eval` report. In NeMo Agent Toolkit, `eval.evaluators` is a shared contract: the evaluator key, score shape, reasoning, and available input state can be consumed by other workflows.

## Who Should Care

| Reader | What They Need To Decide |
|:-------|:-------------------------|
| End user configuring eval | Which evaluator keys to define, which scores become objectives or reward signals, whether the evaluator needs reference answers, trajectories, or dataset metadata. |
| User running optimization | Which evaluator averages should be optimized, which direction to use, and whether the score is stable enough for repeated trials. |
| User running red-team or finetuning flows | Which evaluator key represents the security score or reward signal. |
| Evaluator developer or contributor | Whether to implement legacy (`IntermediateStep`), ATIF, or both lanes; what state is available; how errors, numeric scores, and reasoning should be emitted. |

Default to end-user guidance. Include developer details only when they prevent wrong configs, broken optimization, or unusable custom evaluators.

## Shared Consumers

| Consumer | How It Uses Evaluation | DX Implication |
|:---------|:-----------------------|:---------------|
| `nat eval` | Runs configured `eval.evaluators` and writes `workflow_output.json` plus `<evaluator>_output.json`. | Pick evaluators that match the dataset and workflow behavior. |
| `nat optimize` numeric search | `optimizer.eval_metrics.*.evaluator_name` must match a key under `eval.evaluators`; the optimizer reads each evaluator's `average_score`. | Evaluator names are objective names. Scores must be numeric and direction must be correct. |
| GA prompt optimization | Runs the same eval loop for each prompt candidate and can use evaluator reasoning as oracle feedback. | Per-item reasoning should explain failures clearly enough to guide prompt mutation. |
| Finetuning trajectory/reward flows | The configured reward name is matched against evaluation result names; matching per-item scores become reward signals. | The reward evaluator should be stable, numeric, and aligned with the desired training behavior. |
| Red teaming | The red-team runner builds eval runs that include `red_teaming_evaluator` plus red-team middleware scenarios. | Treat red-team scoring as a security diagnostic, not just another quality metric. |
| Profiler | `eval.general.profiler` emits profiler artifacts; profiler runtime evaluators can also be configured as evaluators and produce scores. | Use profiler artifacts for diagnosis; use runtime evaluator scores when optimization needs latency/token objectives. |

## Choosing Evaluators For Downstream Use

For a one-off report, a noisy or composite score may still be useful. For downstream consumers, be stricter:

- Use one evaluator per independent quality dimension when the result feeds `nat optimize`.
- Prefer numeric `0..1` scores for quality objectives and explicit units for operational objectives such as latency or token count.
- Set `direction: maximize` for quality/safety scores and `direction: minimize` for latency, token count, cost-like metrics, or attack-success scores such as `red_teaming_evaluator` when a higher score means the attack succeeded.
- Keep judge LLMs stable across trials. Do not optimize the judge model or judge prompt in the same run.
- Use `reps_per_param_set` or repeated eval runs when the workflow or evaluator is nondeterministic.
- If the evaluator needs dataset metadata, custom labels, tenant/user fields, or `full_dataset_entry`, prefer the legacy (`IntermediateStep`) lane unless the ATIF path explicitly carries those fields.

## Contributor Notes

Evaluator authors should make the contract obvious:

- Document whether the evaluator supports legacy (`IntermediateStep`) `evaluate_fn`, ATIF `evaluate_atif_fn`, or both.
- Return per-item scores and a meaningful `average_score`; downstream consumers generally use the average.
- On evaluator errors, return a clear zero-score item with error reasoning instead of hiding failures.
- Keep reasoning concise but useful; GA prompt optimization may feed it back into mutation prompts.
- State whether the evaluator requires a trajectory, reference answer, retrieved context, or extra dataset fields.
- Avoid depending on fields that are only present in `full_dataset_entry` unless the evaluator is documented as legacy-only.

## Related Pages

- [`evaluation-surfaces.md`](evaluation-surfaces.md) — choose legacy NeMo Agent Toolkit eval (`IntermediateStep`), legacy plus ATIF artifact, or ATIF-native evaluator lane.
- [`methodology.md`](methodology.md) — design datasets, metrics, and offline/online cycles.
- [`../../nat-optimization/references/configuration.md`](../../nat-optimization/references/configuration.md) — wire evaluator names into optimizer objectives.
