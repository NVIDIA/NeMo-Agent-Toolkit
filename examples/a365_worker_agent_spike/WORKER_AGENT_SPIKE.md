# Worker Agent Spike

This directory is an isolated copy of `examples/a365_example` for a new
API-based / worker-style Agent 365 setup. It is intentionally separate from the
current OBO bot path so the two efforts do not overlap.

## Goal

Create a fresh Agent 365 worker-style deployment with:

- a new agent identity blueprint
- a new Azure Bot
- a new Teams app manifest managed locally
- a new endpoint/App Service or Container App URL

## Pattern To Follow

Use the local workflow/app code as the runtime, but treat all Microsoft-side
objects as new:

- new blueprint
- new bot
- new app secret
- new Teams manifest version/history

Do **not** reuse the current OBO bot identity or current Teams app package for
this spike.

## Microsoft-Side Steps

1. Create a new agent blueprint identity.
   - Type: `API based`
   - Endpoint: point to the new runtime URL for this worker-agent spike

2. In Bot Management, create a new bot.
   - Set the messaging endpoint to the new runtime URL
   - Enable required channels
   - Enable meeting event subscriptions if needed
   - Create a new client secret and save it

3. Update the local Teams manifest.
   - Set the new `botId`
   - Set the new SSO/app values
   - Manage the manifest locally; do not rely on editing it later in the web UI

4. Re-run A365 deploy for the new worker agent package.

5. In Teams:
   - `teams.microsoft.com`
   - `Apps`
   - `Agents for your team`
   - verify the new agent appears with `Create instance`

## Local Separation

Use this directory for the worker-agent spike:

- `examples/a365_worker_agent_spike`

Keep the current OBO bot work in:

- `examples/a365_example`

## Recommended Next Local Changes

Before deploying this spike, update the copied files here to use new names:

- Teams app display name
- manifest version
- bot/client IDs
- any blueprint- or agent-specific IDs
- deployment script defaults (resource names / image tags) if you want fully
  separate Azure resources
