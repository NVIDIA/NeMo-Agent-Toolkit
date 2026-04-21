# Deploy A365 smoke Bot (`nat start a365`)

**IDs, Azure services, and tenant example values:** [docs/A365-DEV-INVENTORY.md](../../../docs/A365-DEV-INVENTORY.md) (repo root).

This folder mirrors the **intent** of blueprints like **NVIDIA AI-Q** on your machine: e.g. `~/proj/aiq/deploy/` (**Dockerfile**, **`entrypoint.py`**, **`compose/`**, **`helm/helm-charts-k8s/`**). This example is smaller: **one Dockerfile** + Azure-oriented notes; you can port the same **Helm** patterns from AI-Q when targeting **AKS**.

## What gets deployed

- **Default container command:** `nat start a365` with `configs/config_a365_bot_with_tooling.yml` — **A365 Bot + telemetry + A365 MCP tooling** (requires **`nvidia-nat-mcp`** in the image; the Dockerfile uses `uv sync --frozen --no-dev --extra mcp`).
- **Bot-only (no MCP, no bearer token):** use `configs/config_a365_front_end_azure.yml` by overriding **CMD** (see Dockerfile comments).
- **Listen:** `0.0.0.0:3978`
- **HTTPS path:** Azure provides TLS at **Container Apps / App Service / Ingress**. Your **Azure Bot** “Messaging endpoint” must be `https://<public-host>/api/messages` (same path as local).

**Secrets (runtime env, never in image):**

- `A365_APP_ID` — Entra application (client) ID of the Bot registration
- `A365_APP_PASSWORD` — client secret
- `A365_ALLOWED_AUDIENCES` — optional comma-separated inbound Bot JWT audience aliases if Teams sends an `aud` different from `A365_APP_ID`
- `A365_BEARER_TOKEN` — **Required for the default config** (client-credentials token for A365 telemetry + tooling discovery; same pattern as [get_a365_token.py](../scripts/get_a365_token.py))

Optional: override `tenant_id` in YAML or inject **`AZURE_TENANT_ID`** on the Container App ( **`aca_rollout_mcp.sh`** copies it when set before rollout). For **production** tooling gateway (**`GET …/mcpServers`**), set **`ENVIRONMENT=Production`** (the rollout script sets this). **`config_a365_bot_with_tooling.yml`** should set **`tooling_gateway_tenant_id`** to your Entra **directory (tenant) ID** for NAT’s gateway headers; see [docs/A365-MCP-NATIVE-SETUP.md](../../../docs/A365-MCP-NATIVE-SETUP.md) and [docs/A365-DEV-INVENTORY.md](../../../docs/A365-DEV-INVENTORY.md#a365-mcp-tooling-nat-and-gateway-handoff).

## Build image (repo root)

`nvidia-nat-a365` is shipped from this monorepo (not as a standalone PyPI install). The Dockerfile copies `packages/`, root `pyproject.toml` + `uv.lock`, and `examples/a365_smoke`, then runs **`uv sync --frozen --no-dev --extra mcp`** from the smoke example so **`nvidia-nat-mcp`** is available for A365 MCP tooling.

```bash
cd /path/to/NeMo-Agent-Toolkit
docker build -f examples/a365_smoke/deploy/Dockerfile -t nat-a365-bot:latest .
```

**Build and push in one step** (Docker running; Azure CLI logged in for ACR):

```bash
cd /path/to/NeMo-Agent-Toolkit
./examples/a365_smoke/deploy/build_and_push.sh <your-acr>.azurecr.io nat-a365-bot <tag>
```

### Build-and-push tag mistake and MANIFEST_UNKNOWN

**Always pass three arguments:** `registry`, **`nat-a365-bot`** (short repo name), **`tag`** (e.g. `mcp-202604071200`).

If you only pass two arguments and bake the tag into the “image name” (e.g. `nat-a365-bot/mcp-202604071200`), the script treats that as a **nested ACR repository** and defaults the Docker tag to **`latest`**. You then push **`…/nat-a365-bot/mcp-…:latest`** but set **`ACA_IMAGE=…/nat-a365-bot:mcp-…`**, and Azure returns **`MANIFEST_UNKNOWN`**. The script **fails fast** when it detects **`/`** in the image name with a missing third argument.

**Apple Silicon (M1/M2/M3):** Azure Container Apps needs **`linux/amd64`**. The script sets **`DOCKER_PLATFORM=linux/amd64`** by default. A pure `docker build` without `--platform` produces **arm64** and ACA errors with *no child with platform linux/amd64*.

**Version pretense:** `.git` is usually excluded from the Docker context; the image sets **`SETUPTOOLS_SCM_PRETEND_VERSION`** (build-arg, default `0.0.0+docker`) so `nvidia-nat` can build without git metadata.

**Reproducible builds:** The lockfile is `examples/a365_smoke/uv.lock`. Regenerate it in CI or local development when bumping NeMo Agent Toolkit dependencies before relying on `--frozen` in production pipelines.

## Azure Container Apps (typical path)

**Existing app roll-out (Bot + MCP image + bearer token):** after `az login` and with a valid `A365_BEARER_TOKEN` (see [examples/a365_smoke/README.md](../README.md)):

```bash
# zsh: use separate lines (do not end a line with “# … AZURE_*” — a wrapped “AZURE_*” line errors as a glob).
cd /path/to/NeMo-Agent-Toolkit/examples/a365_smoke
set -a
source .env
set +a
# MCP against agent365.svc.cloud.microsoft: set scope *after* source .env, or .env's A365_TOKEN_SCOPE wins.
export A365_TOKEN_SCOPE='ea9ffc3e-8a23-4a7d-836d-234d7c7565c1/.default'
export A365_BEARER_TOKEN="$(uv run python scripts/get_a365_token.py)"
cd /path/to/NeMo-Agent-Toolkit
./examples/a365_smoke/deploy/aca_rollout_mcp.sh
# or: ./examples/a365_smoke/deploy/aca_rollout_mcp.sh '<jwt>'
```

In **`.env`**, quote shell glob characters (e.g. `VAR='*'`), or `source .env` under zsh can fail with `no matches found`.

Defaults: resource group `a365-nat-dev-rg-1`, app `nat-a365-bot`, image `…/nat-a365-bot:20260327-mcp`. Override with `ACA_RG`, `ACA_APP`, `ACA_IMAGE`, `ACA_BEARER_SECRET` if needed. Token acquisition: [examples/a365_smoke/README.md](../README.md).

---

1. **Azure Container Registry:** `az acr create …` / `az acr login`
2. **Tag & push:** `docker tag nat-a365-bot … && docker push …`
3. **Create Container App** with:
   - **Ingress:** external, **HTTPS**, **target port 3978**
   - **Secrets** → environment variables for `A365_APP_ID`, `A365_APP_PASSWORD`, **`A365_BEARER_TOKEN`** (prefer Key Vault references). If you use the default image **CMD**, the bearer token is **required** at startup.
4. Copy the **FQDN** (e.g. `https://nat-a365-bot.<region>.azurecontainerapps.io`)
5. **Azure Bot** (Configuration) → **Messaging endpoint:**  
   `https://nat-a365-bot.<region>.azurecontainerapps.io/api/messages`
6. Enable **Teams** channel.
7. **Teams app + tenant:** Package **manifest** with **`bots`[] → `botId`** equal to the Azure Bot **Microsoft App ID** (same value as **`A365_APP_ID`**); set **`isNotificationOnly`** to **`false`** for a chat bot. Upload a versioned package, **publish to the org** if required, and ensure **Teams admin center** permission policies allow your testers to use the app. If Web Chat works but Teams does not, re-check **`botId`** and published app version (clients cache manifests).

Verify health: `curl -i https://<fqdn>/api/messages` → expect **401/405** without a Bot JWT (proves route exists); real traffic must come from Bot Framework / Teams.

**Teams symptoms and manifest details:** [docs/A365-DEV-INVENTORY.md](../../../docs/A365-DEV-INVENTORY.md#microsoft-teams-and-azure-bot-lessons-learned).

## Azure App Service (Linux container)

- Configure **container** to expose **3978** (or map **WEBSITES_PORT** if your image honors it — this image uses fixed **3978**).
- Set **application settings** for `A365_APP_ID`, `A365_APP_PASSWORD`.
- Set **HTTPS only**; Bot endpoint uses the App Service **default hostname** or custom domain.

## AI-Q reference (same machine layout)

The **AI-Q** blueprint under `~/proj/aiq/deploy/` uses:

- **Multi-stage Dockerfile** (`builder` → distroless), `entrypoint.py` for Dask + web
- **Helm** under `deploy/helm/helm-charts-k8s/aiq/`

You can template a **Helm Deployment** for `nat-a365-bot` by copying that chart pattern: same **Deployment/Service/Ingress** shape, different **image**, **containerPort 3978**, **env** from `Secret`, **ingress TLS**.

## Notes

- **`enable_notifications: false`** in the front-end YAML avoids a startup failure from the **A365 notifications** package path (**`AgentNotification`** / **`on_lifecycle_notification`** mismatch with current **`microsoft-agents-a365-notifications`**). It is **not** a Teams-specific workaround for messaging.
- **Default image:** `config_a365_bot_with_tooling.yml` runs **Bot + A365 trace export + MCP discovery** in one process. After deployment, check logs for **`A365 MCP tooling: registered N total tools`**, then add each **`mcp_client__…`** function name to **`workflow.tool_names`** in that YAML, rebuild, and redeploy (or mount an overridden config) so the ReAct agent can call MCP tools—not only `current_datetime`.
- **Bot-only image behavior:** override **CMD** to `start a365 --config_file /app/configs/config_a365_front_end_azure.yml` if you do not want MCP/telemetry or cannot inject **`A365_BEARER_TOKEN`** yet.
