# Agent 365 smoke test

Minimal NeMo Agent Toolkit example for testing Agent 365 telemetry.

**Effort split (read this):** **Trace export** that still returns **HTTP 403** after a valid app token is usually **blocked on tenant policy and Microsoft**, not on the next toolkit patch—see [docs/A365-DEV-INVENTORY.md](./docs/A365-DEV-INVENTORY.md). **MCP tooling** (`config_telemetry_and_tooling.yml`) is **separate**; it does not fix trace 403 and may need its own scopes.

**Development tenant IDs, services, and config file mapping:** [docs/A365-DEV-INVENTORY.md](./docs/A365-DEV-INVENTORY.md).

**Native Microsoft MCP setup (Agent 365 CLI + `ToolingManifest.json` + admin permissions):** [docs/A365-MCP-NATIVE-SETUP.md](./docs/A365-MCP-NATIVE-SETUP.md).

**Microsoft Teams + Azure Bot** (manifest `botId`, permission policies, Web Chat vs Teams, `isNotificationOnly`): [docs/A365-DEV-INVENTORY.md](./docs/A365-DEV-INVENTORY.md).

## Local telemetry-only smoke test

1. Get a bearer token. **Either**:
   - **App (client credentials) — recommended for traces:** Copy `.env.example` to `.env`, set `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`. For Agent 365 telemetry, set **`A365_TOKEN_SCOPE=api://AzureADTokenExchange/.default`** (blueprint app; see [docs/A365-TROUBLESHOOTING-401.md](./docs/A365-TROUBLESHOOTING-401.md)). If Microsoft’s token endpoint requires it, set **`A365_FMI_PATH`** to your **agent identity** client ID ([autonomous agent tokens](https://learn.microsoft.com/en-us/entra/agent-id/identity-platform/autonomous-agent-request-tokens)).
     ```bash
     set -a && source .env && set +a
     export A365_BEARER_TOKEN=$(uv run python scripts/get_a365_token.py --decode)
     ```
     `--decode` prints JWT `aud` / `scp` / `exp` to **stderr** so you can confirm the audience; the token on **stdout** is what gets exported.
   - **User token (often 401 for traces):** `az account get-access-token ...` — usually not the right token type for the traces API.
2. From this directory, **either**:

   **Option A – use `uv run` (no need to install `nat` on PATH):**
   ```bash
   export A365_BEARER_TOKEN="<your-token>"
   uv run nat serve --config_file configs/config_local_telemetry_only.yml
   ```

   **Option B – install then run `nat`:**  
   From the repo root: `uv pip install -e .` and `uv pip install -e packages/nvidia_nat_a365`, then `uv pip install -e examples/a365_example`. Activate the same environment, then:
   ```bash
   export A365_BEARER_TOKEN="<your-token>"
   nat serve --config_file configs/config_local_telemetry_only.yml
   ```
3. At the console prompt, ask something (e.g. "What time is it?"). Check A365 observability for spans.

Replace `agent_id` and `tenant_id` in the config if yours differ.

## Telemetry + A365 MCP tooling (`config_telemetry_and_tooling.yml`)

1. Install the MCP extra: **`uv sync --extra mcp`** (from this directory).
2. **`export A365_BEARER_TOKEN=…`** (same client-credentials flow as telemetry; used for production gateway discovery).
3. **Development discovery:** With default **`ENVIRONMENT=Development`**, the Microsoft SDK reads **`ToolingManifest.json`** from this directory (see the checked-in example). Each entry’s **`mcpServerUniqueName`** is turned into an MCP platform URL; replace names with those your tenant registers, or switch to production discovery.
4. **Production discovery:** **`export ENVIRONMENT=Production`** so the SDK calls the Agent 365 tooling gateway (requires a token that can access that API).
5. Run: **`uv run nat serve --config_file configs/config_telemetry_and_tooling.yml`**. In logs, confirm **`Discovered N MCP servers`** and **`A365 MCP tooling: registered N total tools`** with **`N > 0`**.
6. **Wire tools into the LLM:** add the registered function names (see logs; often `mcp_client__…`) to **`workflow.tool_names`** in the YAML, then restart. Until you do this, the react agent only uses **`current_datetime`**.

### How to test MCP tooling (quick path)

**What “development” discovery actually calls:** With **`ENVIRONMENT=Development`**, the Microsoft SDK still turns each **`mcpServerUniqueName`** into a **streamable HTTP URL** under the MCP platform, e.g.  
`https://agent365.svc.cloud.microsoft/agents/servers/<mcpServerUniqueName>`  
(see `build_mcp_server_url` in **`microsoft_agents_a365`**). So you are **not** talking to a random string—you must have a **token the MCP endpoint accepts** and (in practice) a **server id your tenant knows**.

1. **Startup is the proof:** Run `nat serve` with **`config_telemetry_and_tooling.yml`** (or your bot config). In the **first few seconds** of logs, look for **`Listing MCP tool servers`**, **`Loaded … MCP server configurations`**, **`Registered MCP server`**, and **`A365 MCP tooling: registered N total tools`**. If **`N == 0`**, read **`ERROR`/`WARNING`** lines right after (MCP handshake/401/skips).
2. **Token scope:** **`A365_BEARER_TOKEN`** must be valid for **MCP HTTP**, not only for Power Platform traces. The SDK’s default MCP scope is **`ea9ffc3e-8a23-4a7d-836d-234d7c7565c1/.default`** unless you set **`MCP_PLATFORM_AUTHENTICATION_SCOPE`**. If traces work but MCP does not, request a token with the **MCP** scope (see Microsoft Agent 365 MCP docs) and export that as **`A365_BEARER_TOKEN`** for the tooling run.
3. **Exercise a tool:** After **`N > 0`**, add the logged **`mcp_client__…`** names to **`workflow.tool_names`**, restart, then call **`POST /generate`** with a prompt that **requires** that tool—or use the **Teams** bot with the same YAML once tools are wired.
4. **Optional local mock:** **`TOOLS_MODE=mockmcpserver`** and **`MOCK_MCP_SERVER_URL`** (default `http://localhost:5309/mcp-mock/agents/servers`) point discovery at a **mock** base URL; you still need a real process listening there unless you only test discovery.

The **same checks** apply to **`config_a365_bot_with_tooling.yml`** in Container Apps: use **Log stream from replica start**, not only mid-chat lines.

## Comprehensive smoke (all components)

Runs three scenarios in sequence: telemetry-only, telemetry + A365 MCP tooling, A365 front-end (Bot on 3978). From this directory:

```bash
# If you use .env, source it so A365_BEARER_TOKEN is set (no need to export again):
#   set -a && source .env && set +a
# Otherwise: export A365_BEARER_TOKEN="<your-token>"
# Optional for scenario 3 (A365 front-end): export A365_APP_ID=... A365_APP_PASSWORD=...
./scripts/smoke_all.sh
```

- **Configs:** `configs/config_local_telemetry_only.yml`, `configs/config_telemetry_and_tooling.yml`, `configs/config_a365_front_end.yml`.
- Scenario 2 requires MCP: from this directory run `uv sync --extra mcp` (or `uv pip install nvidia-nat-mcp`). Tooling discovery may still 401 until the right scope is granted.
- Scenario 3 is skipped unless `A365_APP_ID` and `A365_APP_PASSWORD` are set; it runs `nat start a365` (Bot Framework adapter). See `docs/COMPREHENSIVE_SMOKE_TEST.md` for design details.

**What the smoke script does *not* prove:** A **200** from `/generate` only means the workflow ran. It does **not** mean spans were accepted by Agent 365. Confirm ingestion by checking the `nat serve` logs for **no `HTTP 401`** from the A365 exporter (or by seeing traces in the Microsoft admin / Agent 365 UI). If you still see 401, fix token/scope first—re-running the smoke script alone won’t change that.

### If you get HTTP 401 or HTTP 403 when exporting traces

See **[docs/A365-TROUBLESHOOTING-401.md](./docs/A365-TROUBLESHOOTING-401.md)**. Short version:

1. Use **`A365_TOKEN_SCOPE=api://AzureADTokenExchange/.default`** with the **blueprint** app’s client id/secret; add **`A365_FMI_PATH=<agent-identity-client-id>`** if the token request fails or Microsoft’s flow requires it.
2. Inspect the JWT: `uv run python scripts/get_a365_token.py --decode` — confirm **`aud`** matches what the Power Platform traces URL expects. If it shows Graph, traces will often reject it.
3. If still 401, try **`A365_TOKEN_SCOPE=https://api.powerplatform.com/.default`** (same app, if permitted).
4. **Hosting:** For `TurnContext`-based apps, Microsoft documents `exchange_token` + `get_observability_authentication_scope()` ([Agent observability](https://learn.microsoft.com/en-us/microsoft-agent-365/developer/reference/observability-schema/)); that path is separate from `nat serve` HTTP smoke.

### If the bot works in Web Chat but not in Teams

See **[docs/A365-DEV-INVENTORY.md](./docs/A365-DEV-INVENTORY.md)**. The usual fix is aligning **Teams manifest `botId`** with **Azure Bot Microsoft App ID** and **`A365_APP_ID`**; server logs may show **401 Invalid audience** for **`Microsoft-SkypeBotApi`** when they are misaligned.
