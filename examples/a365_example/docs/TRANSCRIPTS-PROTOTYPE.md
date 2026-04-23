# Teams Transcript Prototype

This prototype splits transcript handling into two deployable pieces:

1. `services/transcript_ingest`
   Receives Microsoft Graph change notifications for meeting or ad hoc call transcripts,
   fetches transcript metadata and VTT content, and stores both metadata and content in Azure Blob Storage.

2. `mcp_servers/transcripts`
   Exposes Blob-backed transcript artifacts as MCP tools for the Teams bot.

## Components

### Transcript ingest service

Routes:

- `GET /health`
- `POST /graph/notifications`
- `POST /graph/lifecycle`
- `POST /admin/subscriptions/ensure`

Required environment variables:

- `GRAPH_CLIENT_ID`
- `GRAPH_CLIENT_SECRET`
- `GRAPH_TENANT_ID`
- `GRAPH_TRANSCRIPT_NOTIFICATION_URL`
- `TRANSCRIPT_BLOB_CONNECTION_STRING`
  or
- `TRANSCRIPT_BLOB_ACCOUNT_NAME`
- `TRANSCRIPT_BLOB_ACCOUNT_KEY`

Optional:

- `GRAPH_TRANSCRIPT_LIFECYCLE_URL`
- `GRAPH_TRANSCRIPT_RESOURCES`
- `GRAPH_TRANSCRIPT_CLIENT_STATE`
- `TRANSCRIPT_BLOB_CONTAINER`

Default subscription resources:

- `communications/onlineMeetings/getAllTranscripts`
- `communications/adhocCalls/getAllTranscripts`

### Transcript MCP server

Tools:

- `list_transcripts(...)`
- `get_transcript(transcript_id, include_vtt=False)`
- `summarize_transcript(transcript_id, preview_lines=12)`

Required environment variables:

- `TRANSCRIPT_BLOB_CONNECTION_STRING`
  or
- `TRANSCRIPT_BLOB_ACCOUNT_NAME`
- `TRANSCRIPT_BLOB_ACCOUNT_KEY`

Optional:

- `TRANSCRIPT_BLOB_CONTAINER`

## Blob layout

The ingest service stores transcripts under:

`tenant/{tenantId}/date/{yyyy-mm-dd}/source/{onlineMeetings|adhocCalls}/transcript/{transcriptId}/`

Artifacts:

- `metadata.json`
- `transcript.vtt`
- `transcript.txt`

## Azure / Graph setup still required

This scaffold does not grant Graph permissions for you. The app registration used by the ingest service still needs:

- `OnlineMeetingTranscript.Read.All` for online meetings
- `CallTranscripts.Read.All` for ad hoc calls

Those require admin consent.

Notification callback URLs must be public HTTPS endpoints that match the deployed ingest service.

## Suggested bot wiring

Add a new `mcp_client` function group pointing at the deployed transcript MCP URL, then include `transcripts` in the workflow tool list.

Example:

```yaml
function_groups:
  transcripts:
    _type: mcp_client
    server:
      transport: streamable-http
      url: "https://transcript-mcp.<your-fqdn>/mcp"

workflow:
  tool_names: [current_datetime, graph_mail, jira, github, transcripts]
```
