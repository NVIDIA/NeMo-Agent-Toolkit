# Comprehensive A365 Smoke Test Design

## Components to cover

| Component | Config key | What it does | Smoke goal |
|-----------|------------|--------------|------------|
| **Telemetry** | `general.telemetry.tracing.a365` | Exports OTel spans to A365 traces API | Already have: run workflow, assert export attempted (200 or 401). |
| **Front-end (A365)** | `general.front_end._type: a365` | Bot Framework adapter for Teams/Copilot; binds port 3978 | Start the NeMo Agent Toolkit with the A365 front-end (`nat start a365`); optionally send one Bot activity and assert response. |
| **Tooling (A365 MCP)** | `function_groups` with `a365_mcp_tooling` | Discovers MCP servers via A365, registers tools | Start the toolkit with tooling enabled; run workflow that uses a discovered tool (or assert discovery attempted). |

## Approach

### 1. Scenario-based configs

- **Telemetry only** (current): `configs/config_local_telemetry_only.yml` — keep as-is.
- **Telemetry + Tooling**: New config that adds `function_groups` with `a365_mcp_tooling` (agentic_app_id, auth ref). Requires `nvidia-nat-mcp`. Tooling calls A365 to discover servers — may get 401 without correct scope; smoke can still verify “startup + discovery attempted.”
- **A365 front-end**: New config with `front_end: _type: a365` (app_id, app_password, tenant_id). The toolkit runs the Bot endpoint on 3978. Smoke: start server and optionally POST one Bot Framework activity (e.g. from a small script or Bot Framework Emulator) and assert 200.

### 2. Runner options

**Option A – Script (e.g. `scripts/smoke_all.sh`)**  
- Start the toolkit with config 1 → POST /generate → stop.  
- Start the toolkit with config 2 → POST /generate (or a tool-using prompt) → stop.  
- Start the toolkit with config 3 → (optional) POST one activity to 3978 → stop.  
- Exit 0 if no crashes; optional grep of logs for “export” / “discovery” / “activity.”

**Option B – Pytest smoke suite**  
- Reuse existing integration tests in `packages/nvidia_nat_a365/tests/` (telemetry, front_end, tooling) that **mock** the A365 SDK — run them as “smoke” (fast, no real APIs).  
- Add one or two **live** smoke tests (e.g. telemetry export, tooling discovery) that skip if `A365_BEARER_TOKEN` unset or if response is 401, so CI stays green without real credentials.

**Option C – Hybrid**  
- Unit/integration tests (mocked) in pytest for regression.  
- One script or manual flow that runs the three scenarios against real configs (and real A365 when token/scope work).

### 3. Handling 401 / missing credentials

- **Telemetry**: Already the case — workflow 200, export may 401; smoke asserts “no crash + export attempted.”
- **Tooling**: Discovery may 401; smoke can assert “server starts, tooling config loaded, discovery attempted” (log or exception path).
- **Front-end**: Startup only needs app_id/app_password/tenant_id; actual Bot messages may need more entitlements. Smoke can be “server starts and binds 3978” or “one activity returns 200.”

### 4. Minimal new artifacts

- **Configs**: `config_telemetry_and_tooling.yml`, `config_a365_front_end.yml` (templates with placeholders for app_id, agentic_app_id, auth refs).
- **Runner**: Either `scripts/smoke_all.sh` (start server + curl per scenario) or `tests/test_a365_example.py` (pytest that starts a `nat` subprocess or uses existing mocked tests).
- **Docs**: This file; optional one-liner in main README: “Full A365 smoke: see docs/COMPREHENSIVE_SMOKE_TEST.md.”

## Summary

- **Telemetry**: Keep current smoke; optional assertion on export attempt vs success.  
- **Tooling**: Add config + scenario that starts the toolkit with `a365_mcp_tooling`, run one workflow or assert discovery attempted (with or without mock).  
- **Front-end**: Add config + scenario that starts the toolkit with the A365 front-end, optionally send one Bot activity; assert startup (and 200 if you have credentials).  
- Use mocks in pytest for fast, deterministic coverage; use script or live pytest for full-stack smoke when credentials/scope allow.
