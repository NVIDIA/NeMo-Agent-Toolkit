# Evaluators: `langsmith` and `langsmith_custom`

**Package:** `nvidia-nat[langchain]` — install with `uv pip install "nvidia-nat[langchain]"`
**Best for:** Integrating openevals built-in metrics (`langsmith`) or wrapping any LangSmith-compatible evaluator function (`langsmith_custom`)

---

## `_type: langsmith` — Built-in openevals evaluators

Use this when you want a **deterministic, rule-based metric** from the openevals library selected by short name.

### When to use

- Exact string matching (`exact_match`)
- Edit distance / string similarity (`levenshtein_distance`)
- Any other openevals built-in metric
- No judge LLM required — these are algorithmic

### Config fields

| Field | Required | Description |
| --- | --- | --- |
| `evaluator` | Yes | Short name of an openevals evaluator (e.g., `exact_match`, `levenshtein_distance`) |
| `extra_fields` | No | Map of evaluator kwarg names → dataset field names for passing extra context |

### Example

```yaml
eval:
  evaluators:
    exact_match:
      _type: langsmith
      evaluator: exact_match

    string_similarity:
      _type: langsmith
      evaluator: levenshtein_distance
```

### Discover available evaluators

```bash
# List all evaluators registered in your NeMo Agent toolkit installation
nat info components -t evaluator
```

To see what's available from openevals specifically, check the openevals package directly:

```bash
python3 -c "import openevals; help(openevals)"
```

---

## `_type: langsmith_custom` — Custom LangSmith evaluator

Use this when you have an **existing LangSmith-compatible evaluator function** you want to wire into NeMo Agent toolkit by dotted path — no need to rewrite it as a `BaseEvaluator`.

### When to use

- You already have a LangSmith evaluator function and want to reuse it
- You want to reference an evaluator from another Python package
- Supports `RunEvaluator` subclasses, `(run, example)` functions, and `(inputs, outputs, reference_outputs)` functions

### Config fields

| Field | Required | Description |
| --- | --- | --- |
| `evaluator` | Yes | Python dotted path to the evaluator callable (e.g., `my_package.evaluators.my_fn`) |
| `extra_fields` | No | Map of evaluator kwarg names → dataset field names. Only works with the openevals `(inputs, outputs, reference_outputs)` calling convention. |

### Example

```yaml
eval:
  evaluators:
    my_custom_eval:
      _type: langsmith_custom
      evaluator: my_project.evaluators.check_tool_selection
```

The function at that path must be importable and follow one of the LangSmith evaluator calling conventions.

---

## Gotchas

- `langsmith` resolves short names from openevals at registration time — if the name doesn't exist, NeMo Agent toolkit raises a `ValueError` listing available names
- `langsmith_custom` auto-detects the calling convention — if detection fails, check that your function signature matches one of the three supported forms
- `extra_fields` only works with `langsmith_custom` when the function uses the `(inputs, outputs, reference_outputs)` convention
- Neither type requires a judge LLM in the config (unlike `langsmith_judge` and `tunable_rag_evaluator`)
