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

# MCP (Model Context Protocol) Integration

Consuming MCP tools (`mcp_client`), exposing a NeMo Agent toolkit workflow as an MCP server, transport types, and troubleshooting.

NeMo Agent toolkit supports MCP both as a client (consuming tools from an external MCP server) and as a server (exposing a workflow via MCP). Install the MCP extra:

```bash
uv add "nvidia-nat[mcp]"
```

## Consuming MCP Tools (`mcp_client`)

Use `mcp_client` to connect to an MCP server and auto-discover all its tools as a function group. This is the recommended approach — it replaces the older `mcp_tool_wrapper`.

```yaml
function_groups:
  kb_tools:
    _type: mcp_client
    server:
      transport: sse
      url: ${MCP_SERVER_URL:-http://localhost:8000/sse}

workflow:
  _type: react_agent
  tool_names: [kb_tools]
  llm_name: nim_llm
  verbose: true
  use_native_tool_calling: false
```

All tools from the MCP server are automatically discovered and registered under the function group. The agent sees them as `kb_tools__search_docs`, `kb_tools__get_document`, etc.

### Complete example: MCP client with programmatic API

This is a full working example — workflow YAML + main.py entry point. **Do not bypass the NeMo Agent toolkit framework by using the `mcp` library directly.** Always use `mcp_client` in the YAML config and `WorkflowBuilder.from_config()` in Python.

**workflow.yaml:**

```yaml
function_groups:
  kb_tools:
    _type: mcp_client
    server:
      transport: sse
      url: ${KB_MCP_URL}
    tool_call_timeout: 60
    reconnect_enabled: true

llms:
  nim_llm:
    _type: openai
    model_name: aws/anthropic/claude-haiku-4-5-v1
    base_url: https://inference-api.nvidia.com/v1
    temperature: 0.0
    api_key: $NVIDIA_INFERENCE_API_KEY
    max_tokens: 4096

workflow:
  _type: react_agent
  tool_names: [kb_tools]
  llm_name: nim_llm
  verbose: true
  use_native_tool_calling: false
```

**main.py:**

```python
import argparse
import asyncio
import sys
from pathlib import Path

from nat.builder.workflow_builder import WorkflowBuilder
from nat.runtime.loader import load_config


async def run_agent(question: str) -> str:
    config = load_config("workflow.yaml")
    async with WorkflowBuilder.from_config(config) as builder:
        workflow = await builder.build()
        async with workflow.run(question) as runner:
            result = await runner.result()
    return str(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    if not args.prompt.strip():
        sys.exit(0)

    answer = asyncio.run(run_agent(args.prompt))
    Path("output.md").write_text(answer)


if __name__ == "__main__":
    main()
```

This is all you need. The `mcp_client` function group handles MCP connection, tool discovery, and invocation. The `react_agent` handles reasoning and tool selection. **Do not implement MCP calls manually.**

### Transport types

| Transport | When to use | Config |
| --- | --- | --- |
| `sse` | MCP servers with SSE endpoints | `transport: sse`, `url: http://...` |
| `streamable-http` | Modern MCP servers (recommended for new servers) | `transport: streamable-http`, `url: http://...` |
| `stdio` | Local MCP servers running as subprocesses | `transport: stdio`, `command: python`, `args: ["-m", "my_server"]` |

### Overriding tool names and descriptions

```yaml
function_groups:
  kb_tools:
    _type: mcp_client
    server:
      transport: sse
      url: ${MCP_SERVER_URL}
    tool_overrides:
      search_docs:
        alias: "search"
        description: "Search the knowledge base for documents"
      get_document:
        description: "Fetch full document content by ID"
```

### Additional options

```yaml
function_groups:
  kb_tools:
    _type: mcp_client
    server:
      transport: sse
      url: ${MCP_SERVER_URL}
    tool_call_timeout: 60          # seconds, default 60
    reconnect_enabled: true        # auto-reconnect on connection loss
    reconnect_max_attempts: 2
```

## MCP Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Agent hangs on startup, never calls tools | MCP SSE connection stuck waiting for `\n\n` event terminator | Check that the MCP server sends proper SSE framing (`event: ...\ndata: ...\n\n`). Test with `curl -N <url>` |
| `tools/list` succeeds but `tools/call` times out | MCP server not responding to tool invocations | Increase `tool_call_timeout`. Check server logs for errors on the tool handler |
| Agent generates text but never invokes tools | `use_native_tool_calling: true` with incompatible LLM | Switch to `use_native_tool_calling: false` (text-based ReAct parsing) |
| Intermittent connection resets | MCP server dropping idle SSE connections | Set `reconnect_enabled: true` and `reconnect_max_attempts: 3` |
| `ValueError: url is required` | Missing URL in `mcp_client` server config | Ensure `server.url` is set and the env var resolves (e.g., `${MCP_URL}` is in the environment) |

## Exposing a Workflow as an MCP Server

Set `front_end._type: mcp` to expose the entire workflow as an MCP server:

```yaml
general:
  front_end:
    _type: mcp
    name: "my_agent"
    host: ${MCP_HOST:-0.0.0.0}
    port: ${MCP_PORT:-9001}
    base_path: "/maas/my_agent"
```
