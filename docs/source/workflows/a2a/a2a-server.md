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

# NeMo Agent Toolkit as an A2A Server

[Agent-to-Agent (A2A) Protocol](https://a2aproject.org/) is an open standard from the Linux Foundation that enables agent-to-agent communication and collaboration. You can publish NeMo Agent toolkit workflows as A2A agents, making them discoverable and invokable by other A2A clients.

This guide covers how to publish NAT workflows as A2A servers. For information on connecting to remote A2A agents, refer to [A2A Client](./a2a-client.md). For CLI utilities, refer to [A2A CLI](./a2a-cli.md).

:::{note}
**Read First**: This guide assumes familiarity with A2A client concepts. Please read [A2A Client](./a2a-client.md) first for foundational understanding.
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
3. Publishes the workflow as an A2A agent with functions as skills
4. Exposes an Agent Card at `http://localhost:10000/.well-known/agent-card.json`

### Server Options

You can customize the server settings using command-line flags:

```bash
nat a2a serve --config_file examples/getting_started/simple_calculator/configs/config.yml \
  --host 0.0.0.0 \
  --port 11000 \
  --name "My Calculator Agent" \
  --description "A calculator agent for mathematical operations"
```

**Available options:**
- `--host`: Host to bind the server to (default: `localhost`)
- `--port`: Port to bind the server to (default: `10000`)
- `--name`: Name of the A2A agent (default: workflow name)
- `--description`: Description of the agent's capabilities
- `--version`: Agent version (default: `1.0.0`)

## Configuration File Approach

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

## Configuration Options

You can get the complete list of configuration options and their schemas by running:
```bash
nat info components -t front_end -q a2a
```

## Agent Card Discovery

When you start an A2A server, it automatically generates an Agent Card that describes the agent's capabilities. The Agent Card is available at:

```
http://<host>:<port>/.well-known/agent-card.json
```

### Viewing the Agent Card

```bash
# Using curl
curl http://localhost:10000/.well-known/agent-card.json | jq

# Using NAT CLI
nat a2a client discover --url http://localhost:10000
```

### Agent Card Structure

The Agent Card contains:
- **Agent metadata**: Name, description, version, provider
- **Skills**: Available functions/tools with descriptions and examples
- **Capabilities**: Streaming support, push notifications
- **Content modes**: Supported input/output formats

## How Workflows Map to A2A Agents

When you publish a NAT workflow as an A2A agent:

1. **Workflow becomes an Agent**: The entire workflow is exposed as a single A2A agent
2. **Functions become Skills**: Each function/tool in the workflow becomes an A2A skill
3. **Agent Card is auto-generated**: Metadata is derived from workflow configuration
4. **Natural language interface**: The agent accepts natural language queries and delegates to appropriate functions

### Example Mapping

**NAT Configuration:**
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
    {"id": "calculator.add", "name": "add", "description": "Add two or more numbers"},
    {"id": "calculator.subtract", "name": "subtract", "description": "Subtract numbers"},
    {"id": "calculator.multiply", "name": "multiply", "description": "Multiply numbers"},
    {"id": "calculator.divide", "name": "divide", "description": "Divide numbers"}
  ]
}
```

## Verifying Server Health

### Using the CLI

```bash
# Check if server is running
nat a2a client discover --url http://localhost:10000

# Call the agent
nat a2a client call --url http://localhost:10000 --message "What is product of 42 and 67?"
```

### Using HTTP Endpoints

```bash
# Check agent card
curl http://localhost:10000/.well-known/agent-card.json | jq
```

## Examples

The following example demonstrates A2A server usage:

- Math Assistant A2A Example - NAT workflow published as an A2A server. See `examples/A2A/math_assistant_a2a/README.md`.

## Troubleshooting

### Server Won't Start

**Port Already in Use**:
```bash
# Check what's using the port
lsof -i :10000

# Use a different port
nat a2a serve --config_file config.yml --port 11000
```

**Configuration Errors**:
- Verify your workflow configuration is valid
- Check that all required dependencies are installed
- Review server logs for specific error messages

### Agent Card Not Accessible

**Connection Refused**:
- Verify the server is running
- Check firewall settings
- Ensure you're using the correct host and port

**Empty Skills List**:
- Verify your workflow has functions/tools configured
- Check that function groups are properly defined
- Review workflow configuration for errors

### Performance Issues

**Slow Responses**:
- Check LLM response times
- Review function execution performance
- Monitor server resource usage

## Security Considerations

### Authentication Limitations

**Coming Soon**: Server-side authentication is in progress and will be available shortly.

### Local Development

For local development, use `localhost` or `127.0.0.1` as the host (default). This limits access to your local machine only.

### Production Deployment

For production environments:
- Run the A2A server behind a trusted network or authenticating reverse proxy with HTTPS
- Do not expose the server directly to the public Internet without authentication
- Do not bind to non-localhost addresses (such as `0.0.0.0` or public IP addresses) without proper security measures
- Use a reverse proxy (NGINX, Traefik) with authentication and rate limiting
- Implement network-level security (VPN, firewall rules)

## Protocol Compliance

The A2A server is built on the official [A2A Python SDK](https://github.com/a2aproject/a2a-python) to ensure protocol compliance. For detailed protocol specifications, refer to the [A2A Protocol Documentation](https://a2a-protocol.org/latest/specification/).

## Related Documentation

- [A2A Client Guide](./a2a-client.md) - Connecting to remote A2A agents
- [A2A CLI Guide](./a2a-cli.md) - CLI utilities for testing and debugging
