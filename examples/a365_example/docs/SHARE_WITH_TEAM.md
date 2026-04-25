# A365 telemetry smoke test — context for teammates

## Summary

We run **NeMo Agent Toolkit** locally with the **Microsoft Agent 365 telemetry exporter** enabled. The **workflow and HTTP API work** (`POST /generate` → **200**). Export to the **Agent 365 traces** endpoint may return **HTTP 401** or **HTTP 403** even when logs show **“Token resolved successfully”** — the token can be valid for authentication but **wrong audience/scope** (401) or **missing application permissions / roles** (403) for trace ingestion.

Telemetry auth in this example is wired so it can use either a **static bearer** (env / `api_key`) or an **auth provider** (deferred resolution, token cache, refresh).

---

## Where the smoke test lives (repo paths)

| What | Path |
|------|------|
| Telemetry-only config | `examples/a365_example/configs/config_local_telemetry_only.yml` |
| Telemetry + A365 MCP tooling | `examples/a365_example/configs/config_telemetry_and_tooling.yml` |
| A365 Bot front-end (port 3978) | `examples/a365_example/configs/config_a365_front_end.yml` |
| Multi-scenario runner | `examples/a365_example/scripts/smoke_all.sh` |
| Power Platform token helper (client credentials) | `examples/a365_example/scripts/get_a365_token.py` |
| How to run, 401 notes, MCP extra | `examples/a365_example/README.md` |
| Smoke design / scenarios | `examples/a365_example/docs/COMPREHENSIVE_SMOKE_TEST.md` |
| Env template | `examples/a365_example/.env.example` |

---

## Full YAML — `config_local_telemetry_only.yml`

This is the main artifact we use to reproduce telemetry export locally.

```yaml
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# A365 telemetry-only smoke test (no A365 front-end). Use for local run to verify spans reach A365.
# Run: A365_BEARER_TOKEN=<token> nat serve --config_file configs/config_local_telemetry_only.yml
# Then trigger a request (e.g. via console prompt) and check A365 observability for spans.

general:
  telemetry:
    logging:
      console:
        _type: console
        level: INFO
    tracing:
      a365:
        _type: a365
        agent_id: "a185cf21-03c8-4bf1-919a-ec8f0782118d"
        tenant_id: "06938c20-42d5-4112-9f91-643dff159d7f"
        token_resolver: "a365_auth"
        cluster_category: "prod"
  front_end:
    _type: console

authentication:
  a365_auth:
    _type: api_key
    raw_key: ${A365_BEARER_TOKEN}
    auth_scheme: Bearer

function_groups: {}

functions:
  current_datetime:
    _type: current_datetime

llms:
  nim_llm:
    _type: nim
    model_name: nvidia/nemotron-3-nano-30b-a3b
    temperature: 0.0
    max_tokens: 256
    chat_template_kwargs:
      enable_thinking: false

workflow:
  _type: react_agent
  tool_names: [current_datetime]
  llm_name: nim_llm
  verbose: true
  parse_agent_response_max_retries: 3
```

**Note:** With `nat serve`, the NeMo Agent Toolkit uses the **FastAPI** front-end (default for the `serve` subcommand) and listens on **port 8000**. The `front_end: console` in the file is replaced/overridden for that command; the important pieces are **telemetry + authentication + workflow**.

---

## How to run locally

From `examples/a365_example`:

```bash
# Optional: put A365_BEARER_TOKEN (and Azure creds if used) in .env — smoke_all.sh sources .env if present
set -a && source .env && set +a

uv run nat serve --config_file configs/config_local_telemetry_only.yml
```

In another terminal, trigger the workflow (this is what produces spans and kicks off export):

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"input_message": "What time is it?"}'
```

Expected: **HTTP 200** and a JSON body with the agent’s answer.

---

## Auth lanes (implementation context)

- **Bearer via config:** `authentication.a365_auth` with `_type: api_key` and `raw_key: ${A365_BEARER_TOKEN}` (or any toolkit auth provider that yields `BearerTokenCred` / `HeaderCred`).
- **Auth provider path:** The A365 telemetry plugin resolves `token_resolver` **on first export** (after the workflow/auth stack exists), caches the token, and can refresh via the same provider.

So: **token is present and passed to the Microsoft `agent365_exporter`**; **401** or **403** is from the **Power Platform / Agent 365 traces** HTTP API, not from “missing token” in the exporter wiring.

---

## What we see in logs (symptom)

Typical sequence (paraphrased):

1. `Found N identity groups with M total spans to export`
2. `Exporting … to endpoint: https://il-….tenant.api.powerplatform.com/maven/agent365/agents/<agent_id>/traces?api-version=1`
3. `Token resolved successfully for agent <agent_id>`
4. `HTTP 401 non-retryable error. Correlation ID: …`

Correlation IDs change per request; the **401** is consistent until the correct **scope/permission** for trace ingestion is available.

---

## What “passing” the smoke script means — and does not mean

- **`scripts/smoke_all.sh`** (or a manual curl) proving **200** on `/generate` only proves the **workflow** ran.
- It does **not** prove spans were **ingested** by Agent 365. Confirm that via **no 401** in exporter logs and/or traces visible in **Microsoft admin / Agent 365 observability** UI.

---

## Optional: telemetry + MCP tooling smoke

- Config: `configs/config_telemetry_and_tooling.yml` adds `function_groups.a365_tools` with `_type: a365_mcp_tooling`.
- Requires **`uv sync --extra mcp`** (or `nvidia-nat-mcp`) in this example’s environment.
- Tooling discovery may also hit **401** without the right A365/tooling scopes.

---

## Optional: A365 Bot front-end smoke

- Config: `configs/config_a365_front_end.yml` — needs **`A365_APP_ID`**, **`A365_APP_PASSWORD`**, and typically **`nat start a365`** (not `nat serve`).
- Used for Teams/Copilot-style hosting; separate from the FastAPI `/generate` path above.

---

## Ask for the team / Microsoft

1. **Exact OAuth scope or app permission** required to **POST** to the Agent 365 **traces** API (`…/traces?api-version=1`) for a given tenant/agent.
2. Whether a **standalone** `nat serve` process (not Agent Hosting) must use **client credentials** + a specific resource, or a token from **`get_observability_authentication_scope()`**-equivalent flow.
3. Any **tenant / agent registration** step so the identity tied to the token is allowed to write traces for agent `a185cf21-03c8-4bf1-919a-ec8f0782118d`.

---

## References

- [Agent observability (Microsoft Learn)](https://learn.microsoft.com/en-us/microsoft-agent-365/developer/reference/observability-schema/)
- Internal: `examples/a365_example/README.md`; `examples/a365_example/docs/A365-TROUBLESHOOTING-401.md` (HTTP 401 and 403)
