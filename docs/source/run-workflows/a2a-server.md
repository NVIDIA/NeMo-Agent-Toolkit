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

# NVIDIA NeMo Agent Toolkit Workflow as an A2A Server

[Agent-to-Agent (A2A) Protocol](https://a2aproject.org/) is an open standard from the Linux Foundation that enables agent-to-agent communication and collaboration. You can publish NeMo Agent toolkit [workflows](../build-workflows/about-building-workflows.md) as A2A [agents](../components/agents/index.md) so they can be discovered and called by other A2A clients.

This guide covers how to publish NeMo Agent toolkit workflows as A2A servers. For information on connecting to remote A2A agents, refer to [A2A Client](../build-workflows/a2a-client.md).

:::{note}
**Read First**: This guide assumes familiarity with A2A client concepts. Please read [A2A Client](../build-workflows/a2a-client.md) first for foundational understanding.
:::

## Installation

A2A server functionality requires the `nvidia-nat-a2a` package. Install it with:

```bash
uv pip install "nvidia-nat[a2a]"
```

## Basic Usage

The `nat a2a serve` command starts an A2A server that publishes your workflow as an A2A agent.

### Starting an A2A Server

```bash
nat a2a serve --config_file examples/getting_started/simple_calculator/configs/config.yml
```

This command:
1. Loads the workflow configuration
2. Starts an A2A server on `http://localhost:10000` (default)
3. Publishes the workflow as an A2A agent with [functions](../build-workflows/functions-and-function-groups/functions.md) as skills
4. Exposes an Agent Card at `http://localhost:10000/.well-known/agent-card.json`

### Server Options

You can customize the server settings using command-line flags:

```bash
nat a2a serve --config_file examples/getting_started/simple_calculator/configs/config.yml \
  --host 0.0.0.0 \
  --port 11000 \
  --name "Calculator Agent" \
  --description "A calculator agent for mathematical operations"
```

### Configuration File Approach

You can also configure the A2A server directly in your workflow configuration file using the `general.front_end` section:

```yaml
general:
  front_end:
    _type: a2a
    name: "Calculator Agent"
    description: "A calculator agent for mathematical operations"
    host: localhost
    port: 10000
    version: "1.0.0"
```

Then start the server with:

```bash
nat a2a serve --config_file examples/getting_started/simple_calculator/configs/config.yml
```

### Concurrency Control

The A2A server includes built-in concurrency control to prevent resource exhaustion when handling multiple simultaneous requests. You can configure the maximum number of concurrent workflow executions:

```yaml
general:
  front_end:
    _type: a2a
    name: "Calculator Agent"
    max_concurrency: 16  # Maximum concurrent workflow executions (default: 8)
```

When the limit is reached, additional requests wait in a queue until a workflow completes.

### Additional Configuration Options

You can get the complete list of configuration options and their schemas by running:
```bash
nat info components -t front_end -q a2a
```

## How Workflows Map to A2A Agents

When you publish a workflow as an A2A agent:

1. **Workflow becomes an Agent**: The entire workflow is exposed as a single A2A agent
2. **Functions become Skills**: Each [tool](../build-workflows/functions-and-function-groups/functions.md#agents-and-tools) (function) in the workflow becomes an A2A skill
3. **Agent Card is auto-generated**: Metadata is derived from workflow configuration
4. **Natural language interface**: The agent accepts natural language queries and delegates to appropriate functions

### Example Mapping

**Workflow Configuration:**
```yaml
function_groups:
  calculator:
    _type: calculator  # Provides: add, subtract, multiply, divide

workflow:
  _type: react_agent
  tool_names: [calculator]
```

**A2A Agent Card (Generated):**
```json
{
  "name": "Calculator Agent",
  "skills": [
    {"id": "calculator__add", "name": "add", "description": "Add two or more numbers"},
    {"id": "calculator__subtract", "name": "subtract", "description": "Subtract numbers"},
    {"id": "calculator__multiply", "name": "multiply", "description": "Multiply numbers"},
    {"id": "calculator__divide", "name": "divide", "description": "Divide numbers"}
  ]
}
```

### Viewing the Agent Card
When you start an A2A server, it automatically generates an Agent Card that describes the agent's capabilities. The Agent Card is available at:

```text
http://<host>:<port>/.well-known/agent-card.json
```

You can view the Agent Card using the URL above or the CLI.
```bash
export A2A_SERVER_URL=http://localhost:10000
```

```bash
# Using curl
curl $A2A_SERVER_URL/.well-known/agent-card.json | jq

# Using nat CLI
nat a2a client discover --url $A2A_SERVER_URL
```

Sample output:
![Agent Card](../_static/a2a_agent_card.png)


### Invoking the Agent with the CLI

```bash
# Call the agent
nat a2a client call --url $A2A_SERVER_URL --message "What is product of 42 and 67?"
```

Sample output:
```text
Query: What is product of 42 and 67?

The product of 42 and 67 is 2814.0

(0.85s)
```

## Examples

The following example demonstrates A2A server usage:

- Math Assistant A2A Example - NeMo Agent toolkit workflow published as an A2A server. See `examples/A2A/math_assistant_a2a/README.md`.

## Troubleshooting

### Server Won't Start

**Port Already in Use**:
```bash
# Check what's using the port
lsof -i :10000

# Use a different port
nat a2a serve --config_file config.yml --port 11000
```

## Security Considerations

### Authentication

A2A servers can be protected using OAuth2 authentication with JWT token validation. The server validates incoming tokens by checking:

- **Token signature**: Verified using JWKS from the authorization server
- **Issuer validation**: Ensures token was issued by the expected authorization server
- **Expiration**: Rejects expired tokens
- **Scopes**: Validates required scopes are present in the token
- **Audience**: Ensures token is intended for this specific server

For detailed authentication setup and configuration, see [A2A Authentication Documentation](../components/auth/a2a-auth.md).


### Best Practices

- **Use HTTPS in production**: Always use TLS or SSL for production deployments
- **Configure token validation**: Set appropriate issuer, audience, and required scopes
- **Short-lived tokens**: Configure authorization server to issue short-lived access tokens
- **Monitor access**: Track authentication events and token usage patterns

## Protocol Compliance

The A2A server is built on the official [A2A Python SDK](https://github.com/a2aproject/a2a-python) to ensure protocol compliance. For detailed protocol specifications, refer to the [A2A Protocol Documentation](https://a2a-protocol.org/latest/specification/).

## Related Documentation

- [A2A Client Guide](../build-workflows/a2a-client.md) - Connecting to remote A2A agents
- [A2A Authentication](../components/auth/a2a-auth.md) - OAuth2 authentication for A2A servers
