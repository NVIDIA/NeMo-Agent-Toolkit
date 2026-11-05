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

# Memanto Memory Plugin for NeMo Agent Toolkit

[Memanto](https://github.com/moorcheh-ai/memanto) is an open-source long-term memory layer for AI agents. The [`nemo-agent-toolkit-memanto`](https://pypi.org/project/nemo-agent-toolkit-memanto/) plugin registers Memanto as a memory backend, `_type: memanto`, that works with the automatic memory wrapper agent and with the `add_memory`, `get_memory`, and `delete_memory` tools.

- **One Memanto agent per user.** Each `user_id` maps to its own Memanto agent, so recall never crosses users.
- **Typed memories, not transcripts.** Conversation turns saved by the automatic memory wrapper are distilled into facts, preferences, and decisions. Small talk stores nothing.
- **Explicit memories stored as written.** Items with `memory` text, such as those from the `add_memory` tool, are stored as is, with their tags.

## Installation

```bash
pip install nemo-agent-toolkit-memanto nvidia-nat-langchain
```

Get a Moorcheh API key at [`console.moorcheh.ai`](https://console.moorcheh.ai/api-keys) and set `MOORCHEH_API_KEY` in your environment:

```bash
export MOORCHEH_API_KEY=<YOUR_MOORCHEH_API_KEY>
```

Check that NeMo Agent Toolkit discovers the plugin:

```bash
nat info components -t memory -q memanto
```

## Automatic Memory

Wrap any agent in the automatic memory wrapper agent. Each user message is saved, relevant memories are retrieved before the inner agent runs, and the agent's reply is saved. The user identity comes from the NeMo Agent Toolkit runtime.

```yaml
memory:
  user_memory:
    _type: memanto

llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.3-70b-instruct

functions:
  current_datetime:
    _type: current_datetime
  my_react_agent:
    _type: react_agent
    llm_name: nim_llm
    tool_names: [current_datetime]

workflow:
  _type: auto_memory_agent
  description: Assistant that remembers each user
  inner_agent_name: my_react_agent
  memory_name: user_memory
  llm_name: nim_llm
  search_params:
    top_k: 5
```

Save the configuration as `config.yml`, set `NVIDIA_API_KEY`, and run two turns:

```bash
nat run --config_file config.yml --input "I'm vegetarian and I live in Toronto."
nat run --config_file config.yml --input "Suggest a restaurant for me."
```

The first turn stores memories such as "User is vegetarian." and "User lives in Toronto." The second turn retrieves them before the agent answers.

## Memory Tools

To let the agent decide what to remember, give it the memory tools:

```yaml
memory:
  user_memory:
    _type: memanto

functions:
  add_memory:
    _type: add_memory
    memory: user_memory
    user_id: alex
  get_memory:
    _type: get_memory
    memory: user_memory
    user_id: alex
  delete_memory:
    _type: delete_memory
    memory: user_memory
    user_id: alex

workflow:
  _type: react_agent
  llm_name: nim_llm
  tool_names: [add_memory, get_memory, delete_memory]
```

A fixed `user_id` suits a single-user deployment. For multiple authenticated users, set `user_id_resolver` instead. Refer to [Memory](../../../docs/source/build-workflows/memory.md) for details.

## Configuration

| Field | Default | Description |
| --- | --- | --- |
| `api_key` | `MOORCHEH_API_KEY` | Moorcheh API key. |
| `agent_prefix` | `nat` | Prefix of the per-user agent IDs. Use a different prefix for each application so they never share memories. |
| `max_memories` | `20` | Maximum memories extracted from one conversation, from 1 to 100. |

## More Resources

- [Memanto documentation for NeMo Agent Toolkit](https://docs.memanto.ai/integrations/nemo-agent-toolkit)
- [Plugin source and README](https://github.com/moorcheh-ai/NeMo-Agent-Toolkit-memanto)
- [PyPI: `nemo-agent-toolkit-memanto`](https://pypi.org/project/nemo-agent-toolkit-memanto/)
- [Memanto on GitHub](https://github.com/moorcheh-ai/memanto)
