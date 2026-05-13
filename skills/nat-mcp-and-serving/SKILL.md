---
name: nat-mcp-and-serving
description: Use when serving NeMo Agent toolkit workflows, exposing workflows through FastAPI, configuring MCP clients or servers, or troubleshooting transport and server setup.
metadata:
  version: "0.1.0"
  status: initial
---

# NeMo Agent toolkit MCP and Serving

Use this skill when a workflow needs to call remote MCP tools or be served as an API.

## Workflow

1. Read the reference for the protocol or server target.
2. Keep serving configuration separate from workflow logic where possible.
3. Validate locally with a small request before adding deployment details.
4. Prefer documented `nat serve` and `nat start` commands over ad hoc servers.

## References

- `references/mcp.md`
- `references/fastapi-frontend.md`
