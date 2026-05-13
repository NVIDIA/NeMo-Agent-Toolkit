---
name: skill-evolution
description: Use when creating, refining, or maintaining AI coding agent skills in this repository, especially after user corrections, repeated failures, stale references, routing gaps, or reusable lessons learned.
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---

# Skill Evolution

Use this skill when a workflow, command, reference, or user correction reveals a reusable improvement for the repository's AI coding agent skills.

## When to Update Skills

Update a skill when any of the following happens:

- A user corrects agent behavior in a way that should generalize.
- A command fails and the recovery path should be remembered.
- A reference link, command, component `_type`, or installation instruction is stale.
- Task routing is too broad, too narrow, or points to the wrong skill.
- A repeated instruction appears in multiple skills and should move to a shared entry point.
- A skill contains too much detail and should move detail into focused references.

Do not update skills for one-off user preferences, temporary local environment quirks, or speculative guidance that was not validated.

## Update Workflow

1. Finish the user's requested task first unless the skill update is the task.
2. Identify the smallest skill that should change.
3. Keep each `SKILL.md` concise and task-oriented.
4. Move detailed examples or long reference material into that skill's `references/` directory.
5. Update `AGENTS.md` when a new skill is added or task routing changes.
6. Run Markdown link checks through pre-commit on the changed skill files.

## Naming and Layout

Use the flat skills layout:

```text
skills/
  nat-installation/
    SKILL.md
    references/
  nat-workflow-creation/
    SKILL.md
    references/
```

Each skill folder must contain one `SKILL.md` with frontmatter:

```yaml
---
name: skill-name
description: Use when ...
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---
```

Use `author` and `license` frontmatter instead of long license headers in `SKILL.md` files.

Use specific names that describe the task surface. Avoid catch-all folders that hide routing information.

## Quality Bar

- Skills should tell agents what to do, what to read next, and what to validate.
- Prefer canonical repository docs over copied long-form explanations.
- Keep cross-skill links relative and valid.
- Use "NVIDIA NeMo Agent Toolkit" on first prose use, then "NeMo Agent Toolkit" or "the toolkit".
- Use `nat` only for technical identifiers such as the CLI, package name, Python namespace, paths, and environment variables.
