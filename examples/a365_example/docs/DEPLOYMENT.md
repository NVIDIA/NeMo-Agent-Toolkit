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

# A365 Deployment

This document describes the intended Azure deployment path for the example.

## Canonical Deployment Configuration

Use:

- [../configs/a365_worker.yml](../configs/a365_worker.yml)

That config represents the deployed Teams-triggered worker bot with:

- A365 front end
- A365 telemetry

Other included front-end lanes:

- [../configs/a365_frontend.yml](../configs/a365_frontend.yml)
  for a smaller bot-only deployment shape
- [../configs/a365_email_notifications.yml](../configs/a365_email_notifications.yml)
  for the email-notification trigger path

Treat the email-notification config as an included reference lane, not a
primary validated deployment path.

This example intentionally does not include a custom MCP service deployment.
If you want to integrate managed tooling in a future phase, see
[MCP.md](./MCP.md) for the Work IQ / managed-tooling direction.

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
3. expected telemetry export behavior
