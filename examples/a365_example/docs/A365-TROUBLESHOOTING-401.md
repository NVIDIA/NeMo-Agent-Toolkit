# Agent 365 Telemetry Troubleshooting

Use this when the A365 telemetry exporter fails with `401` or `403`.

## 401 Checklist

1. Use an app token, not a user token
   - prefer `scripts/get_a365_token.py`
   - avoid assuming `az account get-access-token` is the right token for traces

2. Refresh expired tokens
   - `A365_BEARER_TOKEN` is short-lived

3. Confirm scope
   - usually `api://AzureADTokenExchange/.default`
   - in some setups you may need `A365_FMI_PATH`

4. Confirm the config ids
   - `tenant_id`
   - `agent_id`
   - app registration / token tenant

## 403 After 401 Is Fixed

If the token resolves and the exporter still gets `403`, that is usually an
authorization or tenant-policy issue, not a code-path issue in this repo.

Common causes:

- missing Entra application permissions
- missing admin consent
- wrong `agent_id`
- tenant / cloud mismatch
- A365-side policy restrictions

In practice, trace export and MCP/tooling work often diverge. It is normal for
the bot and local MCP servers to function even when A365 observability still
fails.

## Useful Command

```bash
export A365_BEARER_TOKEN="$(uv run python scripts/get_a365_token.py --decode)"
```

`--decode` prints JWT claims to stderr so you can inspect `aud`, `scp`, and
expiration without changing how the token is exported.
