<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Test With `nat_test_llm` for NVIDIA NeMo Agent Toolkit

Use `nat_test_llm` to quickly validate workflows during development and CI. It yields deterministic, cycling responses and avoids real API calls. It is not intended for production use.

## Prerequisites

- Install the testing plugin package:

```bash
uv pip install nvidia-nat-test
```

## Minimal YAML

The following YAML config defines a testing LLM and a simple `chat_completion` workflow that uses it.

```yaml
llms:
  main:
    _type: nat_test_llm
    response_seq: [alpha, beta, gamma]
    delay_ms: 0
workflow:
  _type: chat_completion
  llm_name: main
  system_prompt: "Say only the answer."
```

Save this as `config.yml`.

## Run from the CLI

```bash
nat run --config_file config.yml --input "What is 1 + 2?"
```

You should see a response corresponding to the first item in `response_seq` (for example, `alpha`). Repeated runs will cycle through the sequence (`alpha`, `beta`, `gamma`, then repeat).

## Run programmatically

```python
from nat.runtime.loader import load_workflow

async def main():
    async with load_workflow("config.yml") as workflow:
        async with workflow.run("What is 1 + 2?") as runner:
            result = await runner.result()
            print(result)
```

## Notes
- `nat_test_llm` is for development and CI only. Do not use it in production.
- To implement your own provider, see: [Adding an LLM Provider](../extend/adding-an-llm-provider.md).
- For more about configuring LLMs, see: [LLMs](../workflows/llms/index.md).
