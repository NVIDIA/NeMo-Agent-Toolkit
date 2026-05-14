---
name: nat-agent-configuration
description: Use when selecting, configuring, composing, or troubleshooting NeMo Agent Toolkit agents and control-flow components, including ReAct, tool-calling, ReWOO, reasoning, router, sequential, parallel, and sub-agent patterns.
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---

# NeMo Agent Toolkit Agent Configuration

Use this skill when the task is about choosing or wiring agent components.

## Workflow

1. Classify the workflow shape before choosing an agent.
2. Prefer built-in agents and control-flow components before custom Python.
3. Use sub-agents as tools when composing larger systems.
4. Validate agent behavior with a small `nat run` request before broad tests.

## References

- `references/agents.md`
- `references/additional-agent-types.md`
- `references/subagents.md`
- `references/subagent-patterns.md`
