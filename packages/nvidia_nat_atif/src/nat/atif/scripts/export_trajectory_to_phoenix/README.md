<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# ATIF Trajectory Scripts

Scripts for converting and exporting ATIF trajectory JSON files to
OpenTelemetry-compatible visualization tools such as Phoenix.

## Prerequisites

### Installing the Required Packages

From the repository root, install the required packages from source:

```bash
uv pip install -e packages/nvidia_nat_core -e packages/nvidia_nat_atif -e packages/nvidia_nat_phoenix -e packages/nvidia_nat_opentelemetry
```

### Starting the Phoenix Server

Run a Phoenix instance through Docker:

```bash
docker run -it --rm -p 4317:4317 -p 6006:6006 arizephoenix/phoenix:13.22
```

Once running, the Phoenix UI is available at `http://localhost:6006`.

## Exporting trajectories to Phoenix

```bash
# Single file
python -m nat.atif.scripts.export_trajectory_to_phoenix.export_atif_trajectory_to_phoenix trajectory.json

# Multiple files
python -m nat.atif.scripts.export_trajectory_to_phoenix.export_atif_trajectory_to_phoenix *.json

# Custom endpoint and project
python -m nat.atif.scripts.export_trajectory_to_phoenix.export_atif_trajectory_to_phoenix trajectory.json \
    --endpoint http://localhost:6006/v1/traces \
    --project my-project

# Enable debug logging
python -m nat.atif.scripts.export_trajectory_to_phoenix.export_atif_trajectory_to_phoenix -v trajectory.json
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `files` (positional) | *(required)* | One or more ATIF trajectory JSON files |
| `--endpoint` | `http://localhost:6006/v1/traces` | Phoenix server endpoint URL |
| `--project` | `atif-trajectories` | Phoenix project name for trace grouping |
| `--verbose`, `-v` | off | Enable debug logging |

## Architecture

### Processing flow

```
ATIF Trajectory (dict)
  -> ATIFTrajectorySpanExporter.convert()  -> list[Span]
  -> convert_spans_to_otel_batch()         -> list[OtelSpan]
  -> HTTPSpanExporter.export()             -> Phoenix
```

### Span hierarchy

Each trajectory produces the following span tree:

```
WORKFLOW span  (root -- covers entire trajectory duration)
 |-- LLM span  (agent step with tool_calls)
 |   |-- TOOL span  (tool call 1)
 |   +-- TOOL span  (tool call 2)
 |-- LLM span  (next agent step)
 |   +-- TOOL span
 |-- LLM span  (terminal agent step -- final answer)
 |-- FUNCTION span  (system step with tool_calls, no LLM)
 |   |-- TOOL span  (tool call 1)
 |   +-- TOOL span  (tool call 2, may nest under tool 1)
 +-- FUNCTION span  (terminal system step)
```

- **WORKFLOW** spans wrap the entire trajectory and carry the first
  user message as input and the last agent/system response as output.
- **LLM** spans represent agent steps backed by a language model.
- **FUNCTION** spans represent system/pipeline steps with no LLM involvement.
- **TOOL** spans represent individual tool calls. Nested tool ancestry
  (tool A calls tool B) is preserved via parent-child relationships.

### Design notes

- User steps in external ATIF files typically lack `extra.ancestry`.
  The converter synthesises a root WORKFLOW span from trajectory
  metadata and reparents orphaned agent spans under it.
- `subagent_trajectories` are processed recursively and share
  the parent trace ID so all spans appear in a single Phoenix trace.
- Tool spans that delegate to subagents are linked via
  `subagent_trajectory_ref` in observation results.
- Each call to `export()` creates a new trace (unique trace ID),
  so re-exporting the same file will not produce duplicate spans
  within a single trace.
- Timestamps prefer `extra.invocation` epoch timestamps when available
  and fall back to ISO `step.timestamp` fields only when no invocation
  timestamps exist.

## Modules

| Module | Description |
|--------|-------------|
| `atif_trajectory_exporter.py` | Core converter: ATIF trajectory dict to NAT `Span` objects |
| `atif_trajectory_phoenix_exporter.py` | Phoenix wrapper: converts spans to OTel and exports via HTTP |
| `export_atif_trajectory_to_phoenix.py` | CLI entry point |
