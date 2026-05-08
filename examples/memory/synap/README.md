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

# Synap Memory Plugin for NeMo Agent Toolkit

[Synap](https://maximem.ai) is a managed memory layer for AI agents. This example demonstrates how to use the `SynapMemoryEditor` — a drop-in `MemoryEditor` implementation — with NeMo Agent Toolkit workflows.

## Installation

```bash
pip install maximem-synap-nemo-agent-toolkit
```

Get an API key at [`synap.maximem.ai`](https://synap.maximem.ai) and set `SYNAP_API_KEY` in your environment.

## YAML Configuration

Add `synap` to the `memory:` block in your NeMo Agent Toolkit workflow YAML. The snippet below is a minimal complete example — replace `my_react_agent`, `nim_llm`, and the tool names with your own definitions:

```yaml
memory:
    synap:
        _type: synap_memory
        customer_id: "acme_corp"
        mode: "accurate"

llms:
    nim_llm:
        _type: nim

functions:
    my_react_agent:
        _type: react_agent
        llm_name: nim_llm
        tool_names: []

workflow:
    _type: auto_memory_agent
    inner_agent_name: my_react_agent
    memory_name: synap
    llm_name: nim_llm
```

## Programmatic Usage

```python
import asyncio
import os

from maximem_synap import MaximemSynapSDK
from nat.memory.models import MemoryItem
from synap_nemo_agent_toolkit import SynapMemoryEditor


async def main() -> None:
    sdk = MaximemSynapSDK(api_key=os.environ["SYNAP_API_KEY"])
    await sdk.initialize()

    editor = SynapMemoryEditor(sdk=sdk, customer_id="acme_corp")

    # Write
    await editor.add_items([
        MemoryItem(user_id="user_123", memory="User prefers concise answers.")
    ])

    # Search
    items = await editor.search("communication style", top_k=5, user_id="user_123")
    for item in items:
        print(item.memory)


if __name__ == "__main__":
    asyncio.run(main())
```

Memory is scoped to `user_id` and `customer_id`, ensuring strict isolation in multi-tenant applications.

## More Resources

- [Synap Documentation](https://docs.maximem.ai)
- [NeMo Agent Toolkit Integration Guide](https://docs.maximem.ai/integrations/nemo-agent-toolkit)
- [Dashboard](https://synap.maximem.ai)
- [PyPI: `maximem-synap-nemo-agent-toolkit`](https://pypi.org/project/maximem-synap-nemo-agent-toolkit/)
- [Open source integration package](https://github.com/maximem-ai/maximem_synap_sdk/tree/main/packages/integrations)
