# Agent 365 MCP Setup

This is the example-local runbook for the Microsoft-hosted Agent 365 tooling
path plus the local MCP servers used by this demo.

## Two MCP Modes In This Example

1. **Microsoft-hosted A365 tooling**
   - discovered from `ToolingManifest.json`
   - uses `A365_BEARER_TOKEN`
   - development discovery reads this example's manifest
   - production discovery uses the A365 tooling gateway

2. **Example-local MCP servers**
   - `graph_mail`
   - `jira`
   - `github`
   - `transcripts`
   - configured directly in `configs/config_a365_bot_with_tooling.yml`

The second path is what powers the end-to-end demo today.

## Microsoft-Hosted Tooling Setup

From this directory:

```bash
a365 develop list-available
a365 develop add-mcp-servers mcp_SampleMCPServer
```

That updates [../ToolingManifest.json](../ToolingManifest.json).

If you use the official Microsoft catalog path, the blueprint also needs MCP
permissions granted by an administrator:

```bash
a365 setup permissions mcp
```

## Bearer Token Scope

For telemetry, use:

```bash
export A365_TOKEN_SCOPE='api://AzureADTokenExchange/.default'
```

For MCP tooling gateway calls, mint a token with the MCP platform scope:

```bash
export A365_TOKEN_SCOPE='ea9ffc3e-8a23-4a7d-836d-234d7c7565c1/.default'
export A365_BEARER_TOKEN="$(uv run python scripts/get_a365_token.py)"
```

## Example-Local MCP Servers

The live demo config wires these servers into the bot:

- `graph_mail`
- `jira`
- `github`
- `transcripts`

Those are backed by:

- [../mcp_servers/graph_mail/server.py](../mcp_servers/graph_mail/server.py)
- [../mcp_servers/jira/server.py](../mcp_servers/jira/server.py)
- [../mcp_servers/github/server.py](../mcp_servers/github/server.py)
- [../mcp_servers/transcripts/server.py](../mcp_servers/transcripts/server.py)

## What To Verify In Logs

On bot startup, look for:

- `Listing MCP tool servers`
- `Loaded ... MCP server configurations`
- `A365 MCP tooling: registered N total tools`

For the local servers, also confirm the tool names you expect are present in the
logs and in `workflow.tool_names`.

## Current Demo Shape

The strongest demo path is not the Microsoft catalog servers. It is the local
cross-system tool stack:

- transcript summaries and retrieval
- mailbox search
- GitHub issue / PR lookups
- Jira reads and writes

That is why the example keeps both `ToolingManifest.json` and explicit local MCP
client configuration.
