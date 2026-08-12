# Installation & Hello World

Full installation guide for `nvidia-nat`, including extras matrix, conflict matrix, verification checklist, and a runnable hello-world workflow.

## From PyPI (recommended for using NeMo Agent Toolkit)

For agent projects, always install with the `langchain` extra — it is required for most built-in agent types (`react_agent`, `reasoning_agent`, etc.):

```bash
uv add "nvidia-nat[langchain]"
```

For all framework integrations:

```bash
uv add "nvidia-nat[all]"         # All extras
```

## Optional Extras

Some NeMo Agent Toolkit capabilities ship as separate extras. Install only what you need:

| Extra | Provides | Install |
|---|---|---|
| `[eval]` | `nat eval` runtime — workflow execution, dataset readers, evaluators | `uv add "nvidia-nat[eval]"` |
| `[ragas]` | RAGAS evaluators (AnswerAccuracy, ResponseGroundedness, …) | `uv add "nvidia-nat[ragas]"` |
| `[config-optimizer]` | `nat optimize` CLI for hyperparameter tuning (Optuna) and prompt evolution (GA) | `uv add "nvidia-nat[config-optimizer]"` |

Verify each extra with the matching `--help` (e.g. `nat optimize --help`). If the subcommand prints `Error: No such command`, the extra isn't installed yet.

## Installation Verification Checklist

Before marking installation complete:

- [ ] Run `uv sync` again to make sure the newly added NeMo Agent Toolkit dependencies are installed and the virtual environment is up to date
- [ ] Run `uv run nat --version` to verify that the NeMo Agent Toolkit CLI is available and working
- [ ] Run `uv run python main.py` to verify that the project still runs without errors

Can't check all boxes? Review and update the installation until you can check all boxes.

## Hello World Workflow

## Create the NeMo Agent Toolkit workflow configuration file

Copy the [`hello_world.yaml`](hello_world.yaml) workflow file from the skills assets into the working directory.

Key sections in the NeMo Agent Toolkit workflow file:

- **functions**: Tools the agent can call (each has a `_type` matching a registered function)
- **llms**: LLM provider configuration
- **workflow**: Agent workflow and which tools it uses

## Run the workflow

```bash
nat run --config_file hello_world.yaml --input "What's the current date?"
```

## Hello World Verification Checklist

Before marking installation complete:

- [ ] Make sure the `hello_world.yaml` file is present and correctly references the built-in `current_datetime` tool and a valid LLM configuration
- [ ] Run `uv run nat run --config_file hello_world.yaml --input "What's the current date?"` to verify that the Hello World workflow runs without errors and returns today's date in the response
- [ ] Run `uv run python main.py` to verify that the project still runs without errors

Can't check all boxes? Review and update the hello world example until you can check all boxes.
