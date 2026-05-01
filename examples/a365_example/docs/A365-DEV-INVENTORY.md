# Agent 365 Development Inventory

This example-local inventory is the staging-safe version of the A365 setup notes.
Keep tenant-specific values in `.env` or Azure secret stores, not in this file.

**Related docs**
- [A365-TROUBLESHOOTING-401.md](./A365-TROUBLESHOOTING-401.md)
- [A365-MCP-NATIVE-SETUP.md](./A365-MCP-NATIVE-SETUP.md)
- [../README.md](../README.md)
- [../deploy/README.md](../deploy/README.md)

## Identity Map

The worker/hireable-agent lane uses two identities:

- Agent 365 worker blueprint ID -> `A365_APP_ID`
- Agent 365 worker blueprint client secret -> `A365_APP_PASSWORD`
- Azure Bot `Microsoft App ID` / `botMsaAppId` -> `A365_ALLOWED_AUDIENCES`
- Azure Bot `Microsoft App ID` / `botMsaAppId` -> Teams manifest `bots[].botId`

For standalone token minting, use the Entra app that owns the worker blueprint:

- Entra app `Application (client) ID` -> `AZURE_CLIENT_ID`
- Entra app secret -> `AZURE_CLIENT_SECRET`
- tenant -> `AZURE_TENANT_ID`

## Azure Services Used

- **Microsoft Entra ID**
  - App registration
  - Certificates & secrets
  - API permissions and admin consent
- **Azure Bot**
  - Messaging endpoint must be `https://<host>/api/messages`
  - Teams channel enabled
- **Azure Container Apps**
  - `nat-a365-bot`
  - `graph-mail-mcp`
  - `jira-mcp`
  - `github-mcp`
  - `transcript-mcp`
  - `transcript-ingest`
- **Azure Blob Storage**
  - transcript persistence
- **Azure Container Registry**
  - image push/pull for bot and service rollouts

## Key Runtime Variables

- `A365_APP_ID`
- `A365_APP_PASSWORD`
- `A365_ALLOWED_AUDIENCES` optional
- `A365_BEARER_TOKEN`
- `A365_MCP_TOKEN` optional
- `AZURE_TENANT_ID`
- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`
- `GRAPH_MAIL_TOKEN`
- `JIRA_EMAIL`
- `JIRA_API_TOKEN`
- `JIRA_SITE`
- `GRAPH_CLIENT_ID`
- `GRAPH_CLIENT_SECRET`
- `GRAPH_TENANT_ID`
- `GRAPH_TRANSCRIPT_NOTIFICATION_URL`
- `GRAPH_TRANSCRIPT_LIFECYCLE_URL`
- `TRANSCRIPT_BLOB_CONNECTION_STRING`
- `TRANSCRIPT_BLOB_CONTAINER`

## Transcript Permissions

Transcript ingestion depends on Graph application permissions plus admin consent:

- `OnlineMeetingTranscript.Read.All`
- `CallTranscripts.Read.All`

The working subscription resource for this example is:

- `communications/onlineMeetings/getAllTranscripts`

## Teams / Bot Alignment

For the API-based worker path:

1. Teams manifest `bots[].botId` must match Azure Bot `Microsoft App ID`.
2. `A365_APP_ID` must be the Agent 365 worker blueprint ID.
3. `A365_APP_PASSWORD` must be the worker blueprint client secret.
4. `A365_ALLOWED_AUDIENCES` must include the Azure Bot `Microsoft App ID`.

If Teams inbound auth works but replies fail, also confirm the app registration
tenant model matches the bot setup. In this demo, accepting the bot ID as an
audience while authenticating outbound agentic requests as the blueprint was the
working fix.

## Operational Split

There are three mostly independent tracks in this example:

1. **Bot transport**
   - Teams -> Azure Bot -> `/api/messages`
2. **Observability**
   - NAT spans -> OTel-backed pipeline -> A365 traces exporter
3. **Tooling**
   - local MCP servers for GitHub, Jira, mail, and transcripts

Do not assume fixing one of those automatically fixes the others.

## Staging Guidance

Reasonable files to stage:

- example configs
- example docs
- deploy scripts
- local MCP servers
- transcript ingest service

Do not stage:

- `.env`
- `.token_err.txt`
- `.nat_a365_startup.log`
- `a365.generated.config.json`
- tenant-local secrets or access tokens
