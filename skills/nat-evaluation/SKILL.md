---
name: nat-evaluation
description: Use when designing, configuring, running, or troubleshooting NeMo Agent toolkit evaluations, datasets, evaluator selection, ATIF surfaces, quality gates, custom evaluators, and `nat eval`.
metadata:
  version: "0.1.0"
  status: initial
---

# NeMo Agent toolkit Evaluation

Use this skill for measuring agent quality and behavior.

## Workflow

1. Decide the evaluation surface and output format.
2. Decompose quality goals into separate evaluators.
3. Choose built-in evaluators before writing custom evaluators.
4. Keep datasets small and explicit for local validation.
5. Run `nat eval` and inspect generated artifacts.

## References

- `references/operating-mode.md`
- `references/methodology.md`
- `references/agent-eval-framework.md`
- `references/evaluation-surfaces.md`
- `references/evaluation-contract.md`
- `references/evaluators/`
- `references/code-patterns.md`
