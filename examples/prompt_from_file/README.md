<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# File-Based Prompt Loading Example

This example demonstrates how to load prompts from external files using the `file://` prefix.

## Structure

```text
examples/prompt_from_file/
├── configs/
│   └── config.yml         # Config using file:// prompts
├── prompts/
│   └── system_prompt.txt  # System prompt loaded from file
├── pyproject.toml
└── README.md
```

## Installation

```bash
# From repository root
uv pip install -e examples/prompt_from_file
```

## How It Works

In `config.yml`, prompts are loaded from files:

```yaml
workflow:
  system_prompt: file://../prompts/system_prompt.txt
```

### Rules

- Field name must end with `prompt` (case-insensitive)
- Value must start with `file://`
- Paths are relative to the config file
- Allowed extensions: `.txt`, `.md`, `.j2`, `.jinja2`, `.jinja`, `.prompt`, `.tpl`, `.template`

## Running the Example

```bash
# Run with console (interactive)
nat start console --config_file examples/prompt_from_file/configs/config.yml --input "What is 5 + 3?"

# Run with FastAPI (HTTP server)
nat start fastapi --config_file examples/prompt_from_file/configs/config.yml
```

## Testing (FastAPI mode)

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"input_message": "What is 25 * 4 + 10?"}'
```

## Benefits

- Edit prompts without modifying YAML
- Track prompt changes in version control
- Share prompts across configuration files
- Use any text editor with syntax highlighting
