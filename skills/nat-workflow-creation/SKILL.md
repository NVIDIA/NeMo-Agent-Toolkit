---
name: nat-workflow-creation
description: Use when creating, editing, validating, running, or troubleshooting NeMo Agent toolkit workflow YAML, component discovery, LLM configuration, and common `nat` CLI commands.
metadata:
  version: "0.1.0"
  status: initial
---

# NeMo Agent toolkit Workflow Creation

Use this skill for workflow YAML and command-line execution.

## Workflow

1. Run component discovery before editing `_type` values:

```bash
uv run nat info components -t function
uv run nat info components -t llm_provider
```

2. Read the reference that matches the task.
3. Keep YAML examples runnable from the repository root.
4. Validate with the smallest useful command, usually:

```bash
uv run nat run --config_file path/to/workflow.yml --input "Test request"
```

## References

- `references/workflow-creation.md`
- `references/cli-reference.md`
- `references/llm-config.md`
