<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A365 Setup

This document captures the sanitized setup state behind the NAT A365 worker
example.

## What This Example Assumes

The example assumes a tenant with:

- Microsoft Agent 365 access
- Azure Bot access
- Teams channel access for testing
- admin ability to grant Entra / Agent 365 permissions
- access to the Azure subscription and resource group hosting the worker

## Observed Relevant Licenses

The working tenant used during setup showed these likely relevant Microsoft 365
/ Agent 365 licenses for this example:

- Agent 365
- Agent 365 IP Preview
- Microsoft Agent 365 Frontier
- Microsoft 365 E5

Of those, the three most important licenses to call out from the working tenant
are:

- Microsoft 365 E5
- Microsoft Agent 365 Frontier
- Agent 365 IP Preview

These are the licenses you should verify first when trying to reproduce the
same tenant shape for the A365 worker example.

The tenant also had other Microsoft licenses present, including:

- Microsoft 365 Copilot
- Microsoft 365 Copilot for Service
- Communications Credits
- Microsoft 365 Audio Conferencing
- Microsoft Fabric (Free)
- Microsoft Power Automate Free
- Microsoft Teams Domestic Calling Plan
- Microsoft Teams Domestic and International Calling Plan
- Microsoft Teams Phone Standard
- Microsoft Teams Rooms Standard
- Office 365 E1

These are the licenses observed in the tenant that successfully hosted the demo
shape. Treat this as a practical reference point, not as a complete Microsoft
licensing contract or minimum entitlement matrix.

For reproduction, the most relevant takeaway is:

- Agent 365 access must be present
- Teams access must be present for the validated chat-triggered lane
- tenant admins need the ability to manage the related Entra, Agent 365, and
  Azure resources

## Identity Map

The deployed worker lane uses multiple identities that should not be confused:

| Purpose | What to record |
| --- | --- |
| Tenant ID | The Microsoft Entra tenant hosting the worker |
| Supporting Entra app registration client ID | The app-registration-side client used for bootstrap and token minting |
| Agent blueprint app ID | The Agent 365 blueprint identity used by the NAT worker lane |
| Runtime / Entra agent ID | The effective runtime agent identity used in the observability path |
| Bot app ID | The Azure Bot / bot registration identity |
| Agent blueprint principal object ID | The service principal object created for the blueprint |
| App Service managed identity principal ID | The managed identity of the deployed worker host, if used |

### How NAT Uses These IDs

- `A365_APP_ID` in the deployed worker config points at the blueprint app ID.
- `A365_ALLOWED_AUDIENCES` should include the Azure Bot app ID when Teams
  sends an inbound audience different from `A365_APP_ID`.
- `AZURE_CLIENT_ID` / `AZURE_CLIENT_SECRET` are used for standalone token
  minting flows.
- The explicit runtime-agent observability config uses the runtime / Entra
  agent ID as the effective `agent_id` in the telemetry path.

## Azure Resources

Record the Azure resources that host and expose the worker:

- subscription ID
- resource group
- App Service plan or other compute target
- web app or service hostname
- Azure Bot messaging endpoint

For App Service deployments, the messaging endpoint typically looks like:

```text
https://<worker-hostname>/api/messages
```

## Blueprint And App Registration State

The working environment included:

- an Agent 365 blueprint
- a supporting Entra app registration
- an Azure Bot registration
- a worker deployment exposing `/api/messages`

Do not commit generated `a365.generated.config.json` files with live client
secrets. Convert any needed values into docs or `.env.example`.

## A365 CLI

Use the `a365` CLI primarily to create and configure the agent blueprint used
by the worker.

The reduced example in this repo does not depend on Microsoft-hosted A365
tooling discovery.

### Blueprint Setup

For end-user setup, treat the CLI as the main tool for creating the Agent 365
blueprint and applying its baseline configuration.

The official CLI reference documents:

- `a365 setup all`
- `a365 setup permissions bot`
- `a365 setup permissions custom`

In practice, this example assumes an `a365.config.json` file exists and carries
tenant-specific inputs such as tenant ID, Azure subscription, resource group,
deployment project path, and related blueprint settings.

For this example, the CLI-driven setup produced runtime values such as:

- blueprint app ID
- bot app ID
- messaging endpoint

### Practical Guidance For This Example

Use the CLI in this order:

1. initialize or confirm the tenant-specific `a365.config.json`
2. create or update the blueprint with the setup commands appropriate for your tenant
3. verify the resulting blueprint, bot, and messaging endpoint values in Entra
   and Agent 365 admin surfaces

## Permission Surfaces

The working tenant had two distinct permission surfaces that should not be
collapsed together.

### 1. Supporting Entra App Registration Permissions

The supporting Entra app registration has its own API permissions in Entra.
Those permissions are relevant for bootstrap, standalone token minting, and
related app-registration-side operations.

From the captured Entra app-registration view, that app had a broad Agent Tools
/ MCP-oriented delegated permission set, including examples such as:

- Agent blueprint create / delete permissions
- MCP server list / publish / unpublish permissions
- Dataverse environment listing permissions
- multiple MCP server delegated scopes such as planner, calendar, files,
  Graph admin, and related Agent Tools capabilities

Treat that permission surface as the supporting app-registration side of the
setup, not as proof that the deployed runtime worker or blueprint inherits all
of those grants.

### 2. Agent Blueprint Granted Permissions

Separately, the Agent blueprint itself had granted permissions in the Agent 365
/ Entra admin experience. This is the more important permission surface for the
runtime worker behavior.

The blueprint-side granted permissions are what align most directly with:

- runtime message handling through the bot / worker path
- A365 observability export
- delegated Microsoft Graph actions and related Microsoft workloads

## Blueprint Permissions Observed In The Working Tenant

The blueprint granted-permissions view included these relevant grants.

### Microsoft Graph

- `Mail.ReadWrite`
- `Mail.Send`
- `Chat.ReadWrite`
- `User.Read.All`
- `Sites.Read.All`

Why they matter:

- `Mail.ReadWrite` / `Mail.Send` are relevant for Outlook / mail-oriented
  tooling scenarios
- `Chat.ReadWrite` is relevant to Teams / chat-adjacent scenarios
- `User.Read.All` and `Sites.Read.All` were part of the working tenant setup
  and may be relevant for broader document / user-context scenarios

### Messaging Bot API

- `Authorization.ReadWrite`
- `user_impersonation`

Why they matter:

- these are part of the bot / message-handling permission surface behind the
  A365 worker front end and Azure Bot integration

### Observability

- delegated `user_impersonation`
- application permissions for Agent 365 telemetry / device telemetry /
  collector-send flows

Why they matter:

- these were table stakes for the worker telemetry lane
- the deployed demo depended on the observability audience and permissions being
  present on the blueprint-side granted-permissions surface, not just somewhere
  on the supporting app registration
- this is the permission family most directly tied to the A365 exporter path

### Power Platform API

- `Connectivity.Connections.Read`

Why it matters:

- this appeared in the blueprint granted-permissions view and was part of the
  successful tenant setup

## Practical Rule

If you are debugging “why does the app registration look right but the worker
still fails,” check the blueprint granted permissions next.

The app-registration-side permissions and the blueprint-side permissions are
related, but they are not interchangeable:

- the regular Entra app registration helps with bootstrap and standalone token
  flows
- the blueprint granted permissions are the more important runtime permission
  surface for the deployed worker lane

## Environment Variables

The main runtime variables are:

- `AZURE_TENANT_ID`
- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`
- `A365_APP_ID`
- `A365_APP_PASSWORD`
- `A365_ALLOWED_AUDIENCES`
- `A365_BEARER_TOKEN`
The example `.env.example` includes the main variables used by the worker and
the narrower telemetry / front-end validation paths in this repo example.

## Teams Alignment Rules

For the API-based worker path, keep the Teams manifest identity fields separate
from the NAT worker identity fields.

In a working Teams Developer Portal setup, record and verify:

- Teams app package ID: the app manifest `id`
- Teams manifest `bots[].botId`: the identity bound to the Teams bot capability
- Teams manifest `webApplicationInfo.id`: the SSO application ID
- Teams manifest `webApplicationInfo.resource`: the SSO resource URI

At the same time, the NAT worker lane still uses:

- `A365_APP_ID` = the blueprint app ID
- `A365_APP_PASSWORD` = the matching blueprint credential
- telemetry `agent_id` = the runtime / Entra agent ID

Treat those as different layers of the same setup, not as interchangeable IDs.

If Teams sends an audience that differs from `A365_APP_ID`, include the
accepted upstream audience values in `A365_ALLOWED_AUDIENCES`.

## Teams Setup

The example assumes a Teams app and Azure Bot path, not a raw Bot Framework
smoke in isolation.

### Teams / Bot Wiring

To make the worker reachable from Teams:

1. Create or reuse the Azure Bot tied to the worker deployment.
2. Set the Azure Bot messaging endpoint to the worker route:

```text
https://<worker-hostname>/api/messages
```

3. Create or open the Teams app in Developer Portal at:

```text
https://dev.teams.microsoft.com/apps
```

4. In Developer Portal, open `Configure -> App package editor` and confirm the
   manifest identity bindings:

   - Teams app package `id`
   - `bots[].botId`
   - `webApplicationInfo.id`
   - `webApplicationInfo.resource`

5. Use `Preview in Teams` for an install / launch check.
6. Run `Publish -> App validation`.
7. Use `Publish -> Publish to org` when the package is ready for tenant use.
8. Use `Distribute` when you need to export or share the app package.
9. Make the Teams app available to the users who will test the worker.

### What Usually Breaks

The most common setup mistake is confusing these identities:

- Teams app package ID: Developer Portal app `id`
- Teams manifest bot ID: `bots[].botId`
- Teams SSO binding: `webApplicationInfo.id` and `webApplicationInfo.resource`
- `A365_APP_ID`: worker blueprint app ID
- telemetry `agent_id`: runtime / Entra agent ID in the worker lane

If those are mixed up, the usual symptom is a healthy `/api/messages` route but
401 audience failures on real Teams traffic.

## Recreate The Environment

Rebuild the environment in this order.

### 1. Activate The Required Admin Access

Before touching Agent ID resources, confirm the operator has the required PIM
or group-based admin access. The captured setup flow included activation
through an Agent ID admin group before managing blueprints and permissions.

### 2. Verify Tenant Licenses And Product Access

In Microsoft 365 admin and Entra:

1. confirm the tenant has Agent 365 access
2. confirm the target users can access Teams / Copilot / agent surfaces
3. confirm the operator can manage app registrations, blueprints, and grants

### 3. Create Or Reuse The Supporting Entra App Registration

Create or identify a regular Entra app registration that will be used for
bootstrap, standalone token minting, or Teams manifest SSO bindings.

Treat this as the app-registration-side identity used for setup-side flows.
Do not confuse it with the Agent blueprint app ID.

### 4. Create The Agent Blueprint

Use `a365.config.json` to carry values such as:

- tenant ID
- subscription ID
- resource group
- location
- app service plan
- web app name
- client app ID / bot app ID

That bootstrap produces blueprint-side outputs such as:

- blueprint app ID
- blueprint service principal object ID
- bot ID / bot Microsoft App ID
- messaging endpoint

### 5. Grant Blueprint Permissions

After blueprint creation, grant the required Graph, Agent Tools, Messaging Bot
API, Observability, and Power Platform permissions described above and ensure
admin consent is completed where required.

### 6. Create The Azure Hosting Resources

Provision or confirm:

- App Service plan
- web app
- Azure Bot

Record the actual resource names for your tenant and deployment target.

### 7. Wire The Bot Endpoint

Set the Azure Bot messaging endpoint to:

```text
https://<worker-hostname>/api/messages
```

Then make sure the Teams manifest `botId` matches the Azure Bot Microsoft App
ID.

### 8. Configure NAT Runtime Environment

Populate the worker deployment environment with:

- `A365_APP_ID`
- `A365_APP_PASSWORD`
- `A365_ALLOWED_AUDIENCES`
- `A365_BEARER_TOKEN`
Use `.env.example` as the sanitized source of truth for local reproduction.

### 9. Deploy The Worker

Use the deployment assets under [../deploy](../deploy) and the canonical config
[../configs/a365_worker.yml](../configs/a365_worker.yml).

### 10. Validate In Layers

Validate in this order:

1. route health at `/api/messages`
2. Teams bot round-trip
3. MCP tool registration
4. telemetry export behavior

Do not assume success in one layer proves the others are working.
