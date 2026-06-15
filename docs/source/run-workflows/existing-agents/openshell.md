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

# Run NeMo Agent Toolkit Agents in OpenShell

OpenShell is a good fit when you want to run a NeMo Agent Toolkit workload as an isolated, long-lived service with tighter runtime controls around network access, filesystem access, and credential delivery.

Use this pattern when you want to:

- run a NeMo Agent Toolkit agent as a managed service instead of a one-shot CLI workflow
- expose a frontend such as Teams, webhooks, or another callback-driven channel
- give the agent outbound access to tools, MCP servers, or external APIs without giving it unrestricted egress
- keep long-lived identity material outside the workload when the target system supports brokered or runtime token exchange

This guide focuses on the NeMo Agent Toolkit side of the integration. OpenShell is the runtime boundary. The toolkit still owns the workflow, tool configuration, and frontend behavior.

## Architecture Split

In this deployment model, the responsibilities are split across three layers:

| Layer | Responsibility |
|---|---|
| NeMo Agent Toolkit workload | Agent workflow, tool definitions, frontend integrations, tracing, and business logic |
| OpenShell runtime | Isolated execution, outbound policy enforcement, provider-backed credential delivery, and service exposure |
| Cloud and identity systems | Tenant-specific identity, callback registration, ingress, and cloud resources |

The important boundary is that OpenShell should own runtime controls and credential delivery. The image should own only the agent behavior and the toolkit-side configuration it needs to consume those credentials safely.

## Package the Agent

Package the agent as a container image with a deterministic entrypoint. Treat the container as a long-lived service, not just as a development shell.

A typical image shape is:

- install the NeMo Agent Toolkit project and dependencies
- include the workflow YAML and supporting assets
- expose the frontend port if the agent listens for inbound traffic
- start the toolkit with an explicit config file

For example:

```dockerfile
ENTRYPOINT ["nat"]
CMD ["start", "a365", "--config_file", "/app/configs/a365_worker.yml"]
```

Keep these concerns out of the image when possible:

- long-lived secrets
- environment-specific ingress hostnames
- cluster-specific service wiring
- cloud identity bootstrap details

Those belong in the deployment layer.

## Configure Auth Boundaries

The safest model is to separate:

- agent configuration
- runtime credential delivery
- cloud identity setup

There are two broad auth patterns.

## How OpenShell Providers Map to NeMo Agent Toolkit Auth Providers

The most important integration boundary is the handoff between:

- the **OpenShell provider system**
- the **NeMo Agent Toolkit auth provider or auth configuration**

These are not the same thing.

OpenShell providers own the runtime-side identity contract. They decide:

- where long-lived credential or refresh material is stored
- whether the runtime receives a raw credential or a brokered token contract
- what audiences, resources, or upstream systems the sandbox is allowed to access

NeMo Agent Toolkit auth providers are the application-side consumers of that contract. They decide:

- how the workflow asks for credentials
- how bearer tokens are attached to outbound requests
- how a frontend or tool-facing workflow resolves auth at runtime

The clean mental model is:

- OpenShell providers **produce** an auth contract for the sandbox
- NeMo Agent Toolkit auth providers **consume** that contract inside the workflow

Typical mappings look like this:

| OpenShell runtime contract | Toolkit-side auth shape |
|---|---|
| static environment-variable credential | `api_key` or another environment-driven toolkit auth provider |
| brokered token URL | `openshell_bearer_token` or another callback-driven toolkit auth provider |
| workload-specific callback contract | a custom toolkit auth adapter |

For example, in the Microsoft A365 lane:

- OpenShell owns the `microsoft-agent-s2s` provider
- the runtime injects `A365_TOKEN_PROVIDER_URL`
- the toolkit consumes that contract through an auth block such as:

```yaml
authentication:
  a365_auth:
    _type: openshell_bearer_token
    token_url: ${A365_TOKEN_PROVIDER_URL}
    audience: "<audience>"
```

That keeps the long-lived Microsoft identity material out of the toolkit workload while still letting the workflow obtain a short-lived token when it needs one.

### When the Existing Contract Is Enough

You usually do **not** need new platform work if:

- OpenShell can already expose the credential as an environment variable or a token URL
- the toolkit already has an auth provider that can consume that shape
- the downstream system only needs a standard bearer token or static credential

In that case, the work is mainly:

- provider configuration
- policy for the downstream API
- workflow wiring inside the toolkit

### When You May Need to Extend OpenShell or the Toolkit

You may need additional work when the downstream system expects a more specialized runtime exchange than plain environment-variable injection or a simple bearer-token callback.

Examples include:

- a cloud or enterprise API that needs stricter token-exchange semantics
- a workload that should never see the raw long-lived credential
- a harness that has no callback seam and only understands static auth
- a deployment that needs a stronger local security boundary than a plain HTTP token URL

In those cases, the right move is usually:

- extend **OpenShell** when you need a stronger provider and runtime security boundary
- extend **NeMo Agent Toolkit** when you need a new auth adapter that can consume the runtime contract safely

Treat this as a design boundary, not a hack.

If the secure path cannot be expressed with existing OpenShell provider contracts and toolkit auth providers, add a first-class integration instead of pushing secrets down into the workflow just to make the demo work.

### Static credentials

Use this when the agent or tool expects:

- an API key
- a bot token
- a client secret
- another stable credential presented directly in an environment variable or config

This is the simplest path, but it means the workload is holding the credential value directly.

### Brokered runtime credentials

Use this when the target system supports:

- short-lived access tokens
- service-to-service identity
- a token callback or token URL contract

In this model:

- OpenShell stores the long-lived identity material or refresh material
- the workload receives a local token provider contract
- the workload asks for a token at runtime
- OpenShell validates the request and returns a short-lived token

This is the cleaner model for cloud APIs and non-human worker identities because the long-lived secret does not have to live inside the toolkit container.

### What the Toolkit Needs from the Runtime

For brokered auth to work well, the toolkit workload or adapter layer needs an auth seam such as:

- a token URL
- a token callback
- a pluggable auth provider

If the agent only supports static bearer tokens or hard-coded credential exchange logic, you may need a small adapter before it can consume brokered runtime auth cleanly.

## Allow External Tool Calls

OpenShell sandboxes do not assume unrestricted egress. If your toolkit workflow calls external tools or APIs, you must explicitly allow that traffic.

This is especially important for:

- MCP servers
- GitHub, Jira, Slack, or ServiceNow APIs
- internal HTTP APIs
- vector stores, databases, or search backends
- cloud control-plane APIs

There are two separate pieces to configure:

1. **Policy**
   - which destinations the sandbox can reach
   - which binaries or processes can make those calls
   - whether the traffic is plain L4 or inspected REST or WebSocket traffic

2. **Auth**
   - how the tool or client gets its credentials
   - whether those credentials are static or brokered at runtime

When planning tool access, treat each external integration as its own contract:

- what host or service does the agent need to reach
- what identity should it use
- does the workload need the raw secret, or just a short-lived token

## MCP and Tooling

If the NeMo Agent Toolkit workflow uses MCP or another remote tool host:

- the MCP endpoint must be reachable from inside the OpenShell sandbox
- the sandbox policy must allow that outbound route
- any tool credentials should still come from the runtime or provider layer rather than being baked into the image

In practice, the agent image contains the workflow and tool wiring, while OpenShell controls what the workflow can reach and how secrets cross the boundary.

## Minimal OpenShell Example

Start with a simple non-A365 workflow before you move to Teams, Entra, or callback-driven deployments.

This example shows:

- a toolkit workflow with one outbound HTTP tool
- a container image that runs the workflow
- an OpenShell policy that allows only the required API
- exact `openshell` commands to create the sandbox and test it

The point of this example is not the tool itself. The point is the contract between:

- the toolkit workflow
- the container image
- the OpenShell policy
- the OpenShell sandbox runtime

### Example workflow

Save this as `weather-workflow.yml`:

```yaml
functions:
  current_datetime:
    _type: current_datetime

  weather_api:
    _type: http_function
    url: https://api.weatherapi.com/v1/current.json
    method: GET
    query_params:
      q: "San Francisco"
      key: ${WEATHER_API_KEY}

llms:
  nim_llm:
    _type: nim
    model_name: nvidia/nemotron-3-mini-4b-instruct
    temperature: 0.0
    max_tokens: 512

workflow:
  _type: react_agent
  tool_names: [current_datetime, weather_api]
  llm_name: nim_llm
  verbose: true
  parse_agent_response_max_retries: 3
```

This is deliberately simple:

- `weather_api` is the external tool
- `WEATHER_API_KEY` is the auth boundary
- the workflow can only succeed if the sandbox can both reach the API and resolve the credential

### Example container

Package the workflow into a container image with a stable startup command.

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install nvidia-nat

ENTRYPOINT ["nat"]
CMD ["run", "--config_file", "/app/weather-workflow.yml", "--input", "What is the weather right now?"]
```

This is enough for a local smoke. For a long-lived service workload, switch the command to a service-oriented toolkit entrypoint such as `nat serve` or a channel-specific `nat start ...` command.

### Example OpenShell policy

Save this as `weather-policy.yaml`:

```yaml
version: 1

filesystem_policy:
  include_workdir: true
  read_only:
    - /usr
    - /lib
    - /etc
  read_write:
    - /sandbox
    - /tmp

landlock:
  compatibility: best_effort

process:
  run_as_user: sandbox
  run_as_group: sandbox

network_policies:
  weather_api:
    name: weather-api
    endpoints:
      - host: api.weatherapi.com
        port: 443
        protocol: rest
        enforcement: enforce
        access: read-only
    binaries:
      - path: /usr/local/bin/python
      - path: /usr/local/bin/nat
```

This policy does one important thing: it allows the toolkit workload to reach only `api.weatherapi.com:443` through the expected binaries.

If the workflow needs more tools later, add them intentionally instead of broadening egress all at once.

### Example OpenShell commands

Create a provider or credential source first.

For a static environment-variable credential:

```shell
export WEATHER_API_KEY=<your-key>

openshell provider create \
  --name weather-api \
  --type generic \
  --credential WEATHER_API_KEY
```

Create the sandbox from the image and attach the provider:

```shell
openshell sandbox create \
  --from my-registry.example.com/nat-weather:latest \
  --provider weather-api \
  --policy /path/to/weather-policy.yaml
```

Inspect the sandbox:

```shell
openshell sandbox list
openshell sandbox get <sandbox-name>
```

Check logs:

```shell
openshell logs <sandbox-name> --tail --source sandbox
```

If the workflow is blocked, iterate on policy:

```shell
openshell policy get <sandbox-name> --full > current-policy.yaml
openshell policy set <sandbox-name> --policy current-policy.yaml --wait
```

### What this example proves

This minimal lane proves the most important parts of the deployment model:

- the toolkit image boots correctly inside OpenShell
- the workflow can read its config
- the workflow can consume a runtime-provided credential
- outbound tool access is controlled by explicit policy

Once this works, then you can add:

- more tools
- MCP backends
- brokered runtime auth
- long-lived service commands
- Kubernetes ingress
- Teams, A365, or other callback-driven frontends

### Brokered-auth variant

If the downstream system supports brokered runtime auth, the same workflow pattern can consume a token URL instead of a raw key.

For example:

```yaml
authentication:
  downstream_auth:
    _type: openshell_bearer_token
    token_url: ${DOWNSTREAM_TOKEN_PROVIDER_URL}
    audience: "api://example"
```

In that shape:

- OpenShell owns the long-lived identity material
- the toolkit workflow only asks for short-lived tokens at runtime
- the overall workflow shape stays the same

That is the preferred model when the target API and the toolkit auth seam both support it.

## Deployment Lanes

There are two common lanes for running NeMo Agent Toolkit agents in OpenShell.

### Local or container lane

Use this when you want to iterate quickly on:

- workflow config
- policy
- auth integration
- local testing of outbound tools

The usual flow is:

1. build the toolkit image
2. run OpenShell locally
3. create a sandbox from the image
4. attach the required providers or credentials
5. apply policy for outbound tools and APIs
6. validate the workflow with a small smoke test

This lane is the easiest place to verify that:

- the image boots correctly
- the entrypoint is correct
- the workflow can read the expected config
- outbound tool calls and provider-backed auth work from inside the sandbox

### Kubernetes or cloud lane

Use this when the agent needs:

- stable hosting
- public or tenant-facing callbacks
- integration with cloud identity and ingress
- long-running service operation

In this lane:

- OpenShell hosts the workload as an isolated pod or workload
- cluster-specific ingress and service wiring expose the agent when needed
- tenant-specific identity setup happens outside the generic toolkit image

For callback-driven agents, this layer often includes:

- an ingress path or HTTPS endpoint
- bot or webhook registration
- cloud identity resources
- DNS or public hostnames

These are deployment concerns, not part of the core toolkit workflow.

## Microsoft A365 Example

One validated example of this pattern is a NeMo Agent Toolkit A365 worker hosted inside OpenShell on AKS.

In that shape:

- the toolkit worker listens for a frontend channel such as Teams or another Microsoft-triggered event
- the toolkit workflow runs inside an OpenShell sandbox
- Microsoft or Entra identity is configured in the deployment and provider layers
- the toolkit runtime consumes an OpenShell-managed auth contract rather than minting every token itself
- additional AKS ingress or service plumbing exposes `/api/messages` and related callback paths

The Microsoft-specific pieces are optional. The broader OpenShell hosting model applies to non-Microsoft agents too.

## What Belongs Where

When documenting or implementing this pattern, keep these boundaries clear:

- **NeMo Agent Toolkit owns**
  - workflow YAML
  - tool definitions
  - frontend behavior
  - tracing and evaluation hooks
  - agent-specific auth adapters

- **OpenShell owns**
  - sandbox lifecycle
  - outbound policy
  - provider-backed credential delivery
  - service exposure and runtime boundaries

- **Cloud deployment owns**
  - ingress
  - bot registration
  - tenant setup
  - cloud IAM and cluster resources

Keeping those boundaries clean makes it easier to move the same agent between local, container, and Kubernetes deployment lanes.

## Recommended Deployment Workflow

If you are bringing a NeMo Agent Toolkit workload into OpenShell for the first time, use this order:

1. Package the workload as a container with a stable command.
2. Make the workflow run locally without cloud-specific glue.
3. Identify every outbound tool or API the workflow needs.
4. Decide which credentials can stay static and which should move behind a brokered runtime auth seam.
5. Validate the image in a local OpenShell sandbox first.
6. Move to Kubernetes or cloud deployment only after the local contract is stable.
7. Add ingress, callback registration, and tenant-specific identity last.

This sequence keeps agent behavior, auth, policy, and cloud deployment concerns separate enough to debug them independently.
