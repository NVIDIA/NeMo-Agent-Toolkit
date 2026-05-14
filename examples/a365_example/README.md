<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A365 Worker Example

This example captures the current NAT + Microsoft Agent 365 worker setup used
for the Teams-triggered demo path.

The validated shape is:

1. A Microsoft Agent 365 blueprint and Azure Bot are created in the tenant.
2. The NAT worker is deployed behind `/api/messages`.
3. Teams sends traffic through the Azure Bot into `nat start a365`.
4. The workflow calls a local MCP-backed tool, `graph_mail`.
5. NAT emits A365 telemetry through the configured exporter.

The canonical deployed config is
[configs/a365_worker.yml](./configs/a365_worker.yml).

## What This Example Includes

- A365 front-end config for a Teams / Azure Bot worker
- A365 telemetry config for the worker lane
- local MCP client wiring for the `graph_mail` demo tool
- deployment notes and helper scripts
- sanitized setup notes for rebuilding the tenant-side shape

## If You Are Recreating This Setup

Use this order:

1. Read [docs/SETUP.md](./docs/SETUP.md).
2. Create or confirm the Entra app registration, blueprint, bot, and Azure
   hosting resources.
3. Configure Teams and Azure Bot so the bot ID, endpoint, and audiences align.
4. Copy [`.env.example`](./.env.example) to `.env` and fill in the tenant
   values.
5. Deploy or run the worker with
   [configs/a365_worker.yml](./configs/a365_worker.yml).
6. Validate the path in layers: route health, Teams turn, MCP tools, telemetry.

The setup guide also includes the set of relevant licenses observed in the
working tenant so reviewers can see the approximate Microsoft 365 / Agent 365
footprint required for this demo shape.

## Quick Start

Run the following from `examples/a365_example`.

1. Install the example dependencies:

```bash
uv sync --project examples/a365_example
```

2. Load the local environment:

```bash
cp .env.example .env
set -a && source .env && set +a
```

3. Mint the worker bearer token:

```bash
export A365_BEARER_TOKEN="$(uv run python scripts/get_a365_token.py --decode)"
```

4. Start the worker:

```bash
uv run nat start a365 --config_file configs/a365_worker.yml
```

For a local telemetry-only smoke, run:

```bash
uv run nat serve --config_file configs/a365_telemetry.yml
```

## Teams Setup Checklist

Before expecting a Teams round-trip to work, confirm all of these:

1. The Azure Bot messaging endpoint points at `https://<host>/api/messages`.
2. The Teams app manifest identity fields are aligned:
   `bots[].botId`, `webApplicationInfo.id`, and
   `webApplicationInfo.resource` should match the app-registration and bot
   wiring you actually published in Teams Developer Portal.
3. `A365_APP_ID` remains the worker blueprint app ID, not the bot app ID.
4. `A365_ALLOWED_AUDIENCES` includes the Azure Bot app ID when inbound Teams
   JWT audiences differ from `A365_APP_ID`.
5. The bot is installed or made available to the intended Teams users.

The current demo is Teams-triggered. `graph_mail` is a workflow tool, not the
trigger path.

## Manifest Files

This example involves one manifest concept that sits outside the repo:

1. Teams app manifest
   This is the Teams-side app package you upload or publish in the tenant.
   It is not checked into this example. Its `bots[].botId` and SSO fields must
   align with the app-registration and Azure Bot wiring you publish through
   Teams Developer Portal.

## Included Configs

- [configs/a365_worker.yml](./configs/a365_worker.yml)
  primary validated Teams-triggered worker lane
- [configs/a365_frontend.yml](./configs/a365_frontend.yml)
  smaller Azure Bot / front-end reference
- [configs/a365_email_notifications.yml](./configs/a365_email_notifications.yml)
  email-notification trigger example, included as an unvalidated reference lane
- [configs/a365_telemetry.yml](./configs/a365_telemetry.yml)
  local telemetry smoke path

## Validated Vs Included

| Path | Status |
| --- | --- |
| Teams-triggered worker + local `graph_mail` MCP tool | Validated demo path |
| Local telemetry-only smoke | Included for local validation |
| Email notifications | Included as an unvalidated reference lane |

## Documentation

- [docs/SETUP.md](./docs/SETUP.md) explains identities, permissions, licenses,
  Azure resources, and the rebuild sequence.
- [docs/DEPLOYMENT.md](./docs/DEPLOYMENT.md) explains image build, worker
  deployment, and the smaller front-end reference lane.
- [docs/MCP.md](./docs/MCP.md) explains the local Graph mail MCP server.
- [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md) covers common 401/403,
  audience, Teams, and notification issues.

## Notes

- Email notification routing exists in plugin code but is not the primary
  validated demo path for this example.
- The main end-to-end tool path uses a local or separately deployed Graph mail
  MCP service.
- Do not commit `.env`, generated secrets, token dumps, local logs, or
  generated `a365` config artifacts.
