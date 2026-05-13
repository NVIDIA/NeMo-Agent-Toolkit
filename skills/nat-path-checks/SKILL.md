---
name: nat-path-checks
description: Use when fixing NeMo Agent Toolkit documentation path-check failures, especially failed `ci/scripts/path_checks.py` output, slash-delimited text mistaken for paths, relative path references, Markdown code escaping, and path-check allowlist decisions.
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---

# NeMo Agent Toolkit Path Checks

Use this skill when CI reports failed path checks from `ci/scripts/path_checks.py`, especially when copied CI output lists entries like:

```text
Failed path checks:
- docs/example.md:40:10 -> data/test1
```

## Workflow

1. Open each reported file at the exact line before editing.
2. Classify the reported token by meaning, not just by shape.
3. Apply the smallest fix that preserves meaning.
4. Re-run path checks and Markdown link checks on the changed files.

## Fix Decision Table

| Reported token means | Preferred fix | Example |
| --- | --- | --- |
| Prose shorthand, not a path | Rewrite in words. Do not hide it in code just to silence CI. | `linux/amd64` -> `linux or amd64`; `pass/fail` -> `pass or fail` |
| Two field names or concepts | Name each item separately, usually as code spans if they are literal fields. | `question/answer` -> `question` and `answer` |
| Exact CLI, protocol, API, config, or method literal | Wrap the literal in inline code, or use a fenced code block for multi-line examples. | `tools/list` -> `tools/list` |
| Placeholder path in prose | Wrap the placeholder in inline code. | `path/to/workflow.yml` -> `path/to/workflow.yml` |
| Placeholder paths in a list or snippet | Use a fenced code block with an appropriate language or `text`. | Put `data/test1` and `data/test2` inside a fenced block. |
| Real repo-relative path | Make the path correct relative to the current file, and link it when useful. | `[README](../../README.md)` |
| Intended generated output path | Use inline code or a fenced block unless the file should already exist in the repo. | `./output/results.json` |
| Repeated false positive across many docs | Prefer a local wording or escaping fix. Add `ALLOWLISTED_WORDS`, `IGNORED_PATHS`, or `ALLOWLISTED_FILE_PATH_PAIRS` only for true checker limitations. | Add an allowlist only after confirming the token should pass everywhere. |

## Important Distinctions

- If a slash token is not meant to be copied exactly, rewrite it as prose.
- If a slash token is meant to be copied exactly, put it in Markdown code.
- If a slash token points to a real file in the repository, fix the target instead of escaping it.
- If `data/test1` is an example input path, use `data/test1`. If it is a real checked-in path, link to the real file or correct the path.
- Do not replace a meaningful relative path with different wording unless the path was never meant to be literal.

## Validation

Run these from the repository root after editing:

```bash
uv run ci/scripts/path_checks.sh
uv run pre-commit run markdown-link-check --files path/to/changed.md
```

If a copied CI log contains many failures, fix one file at a time and re-run path checks before applying the same pattern broadly.
