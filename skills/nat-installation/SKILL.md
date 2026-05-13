---
name: nat-installation
description: Use when installing or configuring NVIDIA NeMo Agent toolkit, verifying the `nat` CLI, setting up optional extras, or creating a first hello-world workflow.
metadata:
  version: "0.1.0"
  status: initial
---

# NeMo Agent toolkit Installation

Use this skill when the task is about setup, dependencies, package extras, environment variables, or a first runnable workflow.

## Workflow

1. Read `references/installation.md`.
2. Prefer the smallest package extra that supports the requested workflow.
3. Verify the CLI with `uv run nat --version` or `uv run nat --help`.
4. For a first workflow, adapt `references/hello_world.yaml`.

## Key Commands

```bash
uv run nat --help
uv run nat --version
uv run nat info components
```

## References

- `references/installation.md`
- `references/hello_world.yaml`
