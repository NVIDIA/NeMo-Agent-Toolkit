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

# Creating a New Workflow

Scaffolding a fresh NeMo Agent toolkit workflow with `nat workflow create`.

This is barely needed for most use-cases. Usually it's sufficient to use one of the built-in agents.

## Prerequisites: discover registered components

Before writing or editing the workflow YAML, run the discovery commands in [`cli-reference.md § Discovery`](cli-reference.md#discovery) — do not invent component names from memory. If a `_type` does not appear, install the right extra (see [`../../nat-installation/references/installation.md`](../../nat-installation/references/installation.md)) instead of guessing.

## Scaffold the workflow

```bash
nat workflow create --workflow-dir examples my_workflow
```

This generates:

```text
examples/my_workflow/
├── configs -> src/my_workflow/configs
├── data -> src/my_workflow/data
├── pyproject.toml
└── src/
    └── my_workflow/
        ├── __init__.py
        ├── configs/
        │   └── config.yml
        ├── data/
        ├── register.py
        └── my_workflow.py
```

## Complete `config.yml` scaffold

A workflow YAML always needs three top-level sections — `functions:` (tools the agent can call), `llms:` (LLM definitions), and `workflow:` (the agent itself, referencing the above by name). A minimal but complete example:

```yaml
# configs/config.yml

functions:
  current_datetime:
    _type: current_datetime              # confirm via `nat info components -t function`

llms:
  base_llm:
    _type: nim                            # confirm via `nat info components -t llm_provider`
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0

workflow:
  _type: react_agent                      # confirm via `nat info components -t agent`
  tool_names: [current_datetime]
  llm_name: base_llm
  verbose: true
  parse_agent_response_max_retries: 3
```

A runnable copy of this scaffold lives at [`hello_world.yaml`](../../nat-installation/references/hello_world.yaml). Replace placeholder values, then verify with `nat run`.

## Run or delete

```bash
nat run --config_file=examples/my_workflow/configs/config.yml --input "Hello"
nat workflow delete my_workflow
```
