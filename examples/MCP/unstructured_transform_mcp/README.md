<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Document Transformation with the Unstructured Transform MCP Server

**Complexity:** 🟨 Intermediate

This example demonstrates how the NVIDIA NeMo Agent Toolkit connects to a third-party remote MCP server that is protected by static bearer-token authentication: the hosted [Unstructured Transform](https://transform.unstructured.io/get-started) service, which converts documents (PDF, DOCX, PPTX, XLSX, HTML, images, and 40+ other formats) into clean Markdown that agents can reason over.

It also demonstrates a useful composition pattern: the remote MCP server exposes an asynchronous, multi-step protocol, and this example wraps that protocol in a single deterministic custom function so the agent only needs one reliable tool call.

## Key Features

- **Bearer-token MCP authentication:** Uses the `api_key` authentication provider with `auth_scheme: Bearer` to authenticate against a remote MCP server with a static API key supplied through an environment variable. The other MCP examples cover unauthenticated servers and OAuth2 flows; this example covers the common "API key in a header" case.
- **Remote MCP client over streamable HTTP:** Declares the Transform server as an `mcp_client` function group using the `streamable-http` transport.
- **Deterministic composition of MCP tools:** A custom function (`transform_document`) resolves the four Transform MCP tools from the function group and orchestrates the upload, transform, poll, and download flow in plain Python, exposing one dependable tool to the ReAct agent.
- **Document parsing for agents:** Turns binary documents into Markdown the LLM can summarize, query, and extract from.

## How It Works

The Unstructured Transform MCP server exposes an asynchronous job protocol as four tools:

1. `request_file_upload_url`: Returns a pre-signed upload URL and a file reference for a local file.
2. `transform_files`: Starts a transform job for one or more file references (or public HTTP(S) URLs) and returns a job ID.
3. `check_transform_status`: Reports whether the job is `SCHEDULED`, `IN_PROGRESS`, or `COMPLETED` (any other state means the job failed, and the function reports it as an error).
4. `get_transform_results`: Returns a pre-signed download URL for the Markdown output of each transformed file.

Two steps of the protocol are plain HTTP transfers rather than MCP calls: uploading the raw document bytes to the pre-signed upload URL and downloading the Markdown from the pre-signed download URL. An agent cannot perform those byte transfers with MCP tools alone, and letting an LLM drive the polling loop is slow and unreliable. The `transform_document` function in `src/nat_unstructured_transform_mcp/register.py` therefore performs the whole sequence deterministically:

```text
agent -> transform_document(source)
           |-- request_file_upload_url (MCP)      # local files only
           |-- PUT raw bytes to upload URL        # plain HTTP, no bearer token
           |-- transform_files (MCP)
           |-- check_transform_status (MCP)       # polled until COMPLETED
           |-- get_transform_results (MCP)
           `-- GET Markdown from download URL     # plain HTTP, no bearer token
```

The function accepts either a local file path or a public HTTP(S) URL (public URLs are passed directly to `transform_files`, skipping the upload). Transforms take from a few seconds up to several minutes depending on page count, and the maximum file size is 50 MB. Each `transform_document` call processes a single document, and the example always requests the default Markdown output; the Transform service also supports element JSON, HTML, and plain-text output, which would require extending `transform_document`.

> [!IMPORTANT]
> Trust boundary: for a local path, `transform_document` reads that file and uploads its contents to the hosted Transform service. Because the path comes from the agent, a crafted prompt could point it at a sensitive file (for example a private key or a credentials file) and cause that file to leave the host. Run this example with documents and prompts you trust, and if you adapt it for untrusted input, restrict the accepted paths to a designated directory.

## Prerequisites

1. **Unstructured Transform API key:** Sign up and create a key at [transform.unstructured.io/get-started](https://transform.unstructured.io/get-started).
2. **NVIDIA API key:** The workflow uses an NVIDIA NIM hosted model. Get your key from the [NVIDIA API Catalog](https://build.nvidia.com/).
3. **NeMo Agent Toolkit installed from source:** Follow the [installation guide](../../../docs/source/get-started/installation.md#install-from-source).

## Installation and Setup

### Install this Workflow

From the root of the repository, run:

```bash
uv pip install -e examples/MCP/unstructured_transform_mcp
```

### Set Up API Keys

```bash
export UNSTRUCTURED_API_KEY=<YOUR_UNSTRUCTURED_API_KEY>
export NVIDIA_API_KEY=<YOUR_NVIDIA_API_KEY>
```

## Run the Workflow

Ask the agent to transform a document and answer a question about it. The input can be a public HTTPS URL:

```bash
nat run --config_file examples/MCP/unstructured_transform_mcp/configs/config.yml \
  --input "Transform https://arxiv.org/pdf/1706.03762 to Markdown and list the section headings."
```

Or a local file:

<!-- path-check-skip-begin -->
```bash
nat run --config_file examples/MCP/unstructured_transform_mcp/configs/config.yml \
  --input "Transform the document at /path/to/your/document.pdf and summarize it in three bullet points."
```
<!-- path-check-skip-end -->

**Expected Workflow Output**

The transform takes roughly one to three minutes for this 15-page paper, after which the agent answers from the returned Markdown:

```text
Workflow Result:
The section headings of the transformed document are:
- # Attention Is All You Need
- # 2 Background
- # 3 Model Architecture
- # 3.1 Encoder and Decoder Stacks
- # 3.2 Attention
- # 3.2.1 Scaled Dot-Product Attention
- # 3.2.2 Multi-Head Attention
- # 3.3 Position-wise Feed-Forward Networks
- # 3.4 Embeddings and Softmax
- # 3.5 Positional Encoding
- # 4 Why Self-Attention
- # 5 Training
- # 5.1 Training Data and Batching
- # 5.2 Hardware and Schedule
- # 5.3 Optimizer
- # 5.4 Regularization
- # 6 Results
- # 6.1 Machine Translation
- # 6.2 Model Variations
- # 6.3 English Constituency Parsing
- # 7 Conclusion
- # References
```

## Configuration Details

The complete configuration is in `configs/config.yml`. The MCP client and authentication sections are the interesting parts:

```yaml
function_groups:
  unstructured_transform:
    _type: mcp_client
    server:
      transport: streamable-http
      url: https://mcp.transform.unstructured.io/
      auth_provider: unstructured_auth
    include:
      - request_file_upload_url
      - transform_files
      - check_transform_status
      - get_transform_results

authentication:
  unstructured_auth:
    _type: api_key
    raw_key: ${UNSTRUCTURED_API_KEY}
    auth_scheme: Bearer
```

- The `api_key` authentication provider attaches `Authorization: Bearer <UNSTRUCTURED_API_KEY>` to every request the MCP client makes, including the initial handshake.
- `${UNSTRUCTURED_API_KEY}` is interpolated from the environment when the configuration is loaded. See [workflow configuration](../../../docs/source/build-workflows/workflow-configuration.md) for the interpolation syntax.
- The `include` list documents the four tools the example depends on and fails fast if the server stops exposing any of them.
- The `transform_document` function references the function group through its `mcp_group` setting, so the group does not need to appear in the workflow `tool_names` and the agent never sees the low-level tools.

### Alternative: Custom Headers

If you prefer not to define an authentication provider, the MCP client can also send a static header directly. This variant sends the same header but does not retry on authentication failures:

```yaml
function_groups:
  unstructured_transform:
    _type: mcp_client
    server:
      transport: streamable-http
      url: https://mcp.transform.unstructured.io/
      custom_headers:
        Authorization: "Bearer ${UNSTRUCTURED_API_KEY}"
    include:
      - request_file_upload_url
      - transform_files
      - check_transform_status
      - get_transform_results
```

Keep the `include` list in this variant as well; it provides the same fail-fast contract check described above.

### Tuning

The `transform_document` function accepts a few settings in the `functions` section of the configuration:

| Setting | Default | Purpose |
|---|---|---|
| `poll_interval_seconds` | `5.0` | Delay between job status checks. |
| `transform_timeout_seconds` | `900.0` | Maximum time to wait for a transform job. Raise this for very large documents. |
| `http_timeout_seconds` | `120.0` | Timeout for the raw upload and download requests. |
| `max_file_size_bytes` | `52428800` | Maximum document size accepted by the Transform service (50 MB at the time of writing). |
| `max_output_characters` | `50000` | Truncates the returned Markdown to protect the context window of the agent; a short truncation notice is appended. |

## Testing

Unit tests mock the MCP tools and the HTTP transfers, so they run without network access or credentials:

```bash
pytest examples/MCP/unstructured_transform_mcp/tests
```

Integration tests run against the live Transform service and are skipped unless `UNSTRUCTURED_API_KEY` is set (the full workflow test also requires `NVIDIA_API_KEY`):

```bash
pytest --run_integration --run_slow examples/MCP/unstructured_transform_mcp/tests
```

## Troubleshooting

- **Authentication errors during startup:** A missing `UNSTRUCTURED_API_KEY` fails configuration validation immediately with a `raw_key` error, because the environment variable interpolates to an empty string. An invalid key fails the MCP handshake when the workflow is built. In both cases, verify the variable is exported in the shell that runs `nat run`.
- **Transform timed out:** Large documents can take several minutes. Raise `transform_timeout_seconds` in the `transform_document` function configuration.
- **Transform job ended in an unexpected state:** The document could not be processed on the server (for example, a corrupt or unsupported file). The error includes the status payload returned by the service.
- **File too large:** The Transform service accepts files up to 50 MB.
- **Document not found:** The agent passes the file path verbatim, so use an absolute path in the prompt.

## Related Examples and Documentation

- [`simple_auth_mcp`](../simple_auth_mcp/README.md): OAuth2 authentication with remote MCP servers.
- [`service_account_auth_mcp`](../service_account_auth_mcp/README.md): OAuth2 client-credentials (service account) authentication with remote MCP servers.
- [MCP client documentation](../../../docs/source/build-workflows/mcp-client.md): All `mcp_client` configuration options.
- [API authentication documentation](../../../docs/source/components/auth/api-authentication.md): Details of the `api_key` authentication provider.
