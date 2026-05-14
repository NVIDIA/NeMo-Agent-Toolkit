<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A365 Deployment

This document describes the intended Azure deployment path for the example.

## Canonical Deployment Config

Use:

- [../configs/a365_worker.yml](../configs/a365_worker.yml)

That config represents the deployed Teams-triggered worker bot with:

- A365 front end
- A365 telemetry
- the local `graph_mail` MCP client

Other included front-end lanes:

- [../configs/a365_frontend.yml](../configs/a365_frontend.yml)
  for a smaller bot-only deployment shape
- [../configs/a365_email_notifications.yml](../configs/a365_email_notifications.yml)
  for the email-notification trigger path

Treat the email-notification config as an included reference lane, not a
primary validated deployment path.

## Runtime Secrets

At minimum, provide:

- `A365_APP_ID`
- `A365_APP_PASSWORD`
- `A365_BEARER_TOKEN`

Commonly also needed:

- `A365_ALLOWED_AUDIENCES`
- `AZURE_TENANT_ID`

Keep these in Azure app settings, Key Vault references, or local `.env` files.
Do not bake them into the image.

## Build

From the repository root:

```bash
docker build -f examples/a365_example/deploy/Dockerfile -t nat-a365-bot:latest .
```

Or use the helper:

```bash
./examples/a365_example/deploy/build_and_push.sh <registry> nat-a365-bot <tag>
```

## Hosting Targets

The example docs and scripts currently align best with:

- Azure App Service
- Azure Container Apps

## Azure Bot

The Azure Bot messaging endpoint must point to:

```text
https://<public-host>/api/messages
```

The Azure Bot Microsoft App ID must stay aligned with:

- Teams manifest `bots[].botId`
- `A365_ALLOWED_AUDIENCES` when the worker blueprint ID is used as `A365_APP_ID`

## Validation

Basic health probe:

```bash
curl -i https://<host>/api/messages
```

Expect `401` or `405` without a valid bot JWT. That confirms the route is
present; it does not prove Teams messaging works.

Real deployment validation should include:

1. successful web app startup
2. successful Teams bot round-trip
3. expected MCP tool registration in logs
4. expected telemetry export behavior
