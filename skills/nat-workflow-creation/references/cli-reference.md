# CLI Reference

Top-level reference for the `nat` CLI. For task-specific deep dives, follow the links into other references.

## Discovery

`nat info components` is the first thing to run before editing any workflow YAML — it lists what's actually registered in the current environment. Do not invent component names from memory.

```bash
nat info components -t function                  # all functions/tools
nat info components -t function -q wiki          # filter by keyword
nat info components -t llm_provider              # LLM provider _types
nat info components -t llm_provider -q nim       # specific component details
nat info components -t logging                   # logging exporters
nat info components -t tracing                   # tracing exporters
nat info components -t evaluator                 # evaluator types
```

If a `_type` you expect isn't listed, the relevant extra is not installed — fix the install, don't guess YAML.

## Running and validating workflows

| Command                                                  | Use                                                       |
| -------------------------------------------------------- | --------------------------------------------------------- |
| `nat validate --config_file <path>`                      | Catch config errors before executing                      |
| `nat run --config_file <path> --input <query>`           | One-off execution                                         |
| `nat run --config_file <path> --input_file <json>`       | Batch execution from a JSON file                          |
| `nat run ... --override <path> <value>`                  | Change a config value without editing YAML                |
| `nat run ... --result_json_path <jsonpath>`              | Extract a nested result (default `$`)                     |

## Serving workflows

`nat serve` is the FastAPI shorthand. `nat start <frontend>` selects a specific protocol — use it when the frontend matters (MCP, FastMCP, A2A).

| Command                                                       | Use                                |
| ------------------------------------------------------------- | ---------------------------------- |
| `nat serve --config_file <path> --host 0.0.0.0 --port 8000`   | FastAPI — the typical dev path     |
| `nat start console --config_file <path>`                      | Interactive console / REPL         |
| `nat start fastapi --config_file <path>`                      | Same as `nat serve`                |
| `nat start mcp --config_file <path>`                          | Expose workflow as an MCP server   |
| `nat start fastmcp --config_file <path>`                      | FastMCP variant                    |
| `nat start a2a --config_file <path>`                          | Agent-to-Agent protocol            |

`nat mcp …`, `nat fastmcp …`, `nat a2a …` are protocol-specific subcommand groups (client + server utilities, e.g. `nat mcp install`).

For FastAPI specifics see [`../../nat-mcp-and-serving/references/fastapi-frontend.md`](../../nat-mcp-and-serving/references/fastapi-frontend.md). For exposing a workflow as an MCP server see [`../../nat-mcp-and-serving/references/mcp.md`](../../nat-mcp-and-serving/references/mcp.md).

## Evaluation

```bash
nat eval --config_file <path>
```

Useful flags: `--dataset`, `--endpoint`, `--result_json_path`, `--skip_workflow`, `--reps`. Full guidance in [`../../nat-evaluation/references/methodology.md`](../../nat-evaluation/references/methodology.md).

## Optimization

```bash
nat optimize --config_file <path>
```

Useful flags: `--dataset`, `--endpoint`, `--result_json_path`, `--endpoint_timeout`. Full guidance in [`../../nat-optimization/references/`](../../nat-optimization/references/) — see `output-and-cli.md` for the full flag table and `choosing-parameters.md` for tuning advice.

> **Never kill `nat optimize` mid-run** — final artifacts are only written when the study finishes cleanly.

## Workflow scaffolding

| Command                          | Use                                                                   |
| -------------------------------- | --------------------------------------------------------------------- |
| `nat workflow create <name>`     | Scaffold a new workflow package (rarely needed for in-app YAML edits) |
| `nat workflow reinstall <name>`  | Re-run install steps for a workflow package                           |
| `nat workflow delete <name>`     | Remove a scaffolded workflow                                          |

For ordinary YAML edits inside an existing app, prefer adapting an example over `nat workflow create`. Full scaffolding flow in [`workflow-creation.md`](workflow-creation.md).

## Registry

`nat registry` manages publishing, pulling, searching, or removing reusable workflow components through a configured registry. Use only when a registry/channel is set up.

```bash
nat registry --help
```

## Common flags

| Flag                  | Where                                                     | Description                                            |
| --------------------- | --------------------------------------------------------- | ------------------------------------------------------ |
| `--config_file`       | most subcommands                                          | Path to workflow YAML                                  |
| `--input`             | `nat run`                                                 | Single query string                                    |
| `--input_file`        | `nat run`                                                 | JSON batch input                                       |
| `--override`          | `nat run`                                                 | `--override <path> <value>` — change config inline     |
| `--result_json_path`  | `nat run`, `nat eval`, `nat optimize`                     | JSONPath to extract from output (default `$`)          |
| `--dataset`           | `nat eval`, `nat optimize`                                | Override dataset path (prefer setting it in config)    |
| `--endpoint`          | `nat eval`, `nat optimize`                                | Run against a served endpoint                          |
| `--reps`              | `nat eval`                                                | Multiple repetitions                                   |
| `--skip_workflow`     | `nat eval`                                                | Re-run evaluators on cached generated answers          |
| `--host`, `--port`    | `nat serve`, `nat start *`                                | Bind address                                           |

## Picking a command

- Just running once → `nat run`
- Catching config errors → `nat validate`
- Exposing over HTTP → `nat serve` (FastAPI), or `nat start <frontend>` for MCP / A2A / FastMCP
- Adding eval or optimization → `nat eval` / `nat optimize`
- Building a new workflow package from scratch → `nat workflow create`
- Sharing / publishing components → `nat registry`
