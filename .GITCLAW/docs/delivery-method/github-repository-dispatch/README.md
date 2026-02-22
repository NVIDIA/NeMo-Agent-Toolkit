# .GITCLAW 🦞 GitHub Repository Dispatch / API-Driven

### Trigger gitclaw installation remotely via the GitHub API — from any system that can make HTTP requests.

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

---

## Overview

The GitHub Repository Dispatch method uses the GitHub API's `repository_dispatch` event to trigger a workflow that installs gitclaw on a target repository. A centralized "mothership" repository contains the installation workflow, and external systems — CLIs, web portals, chatbots, other workflows — send dispatch events to trigger installations.

This is the most composable delivery method. It separates the "trigger" from the "execution," allowing any system that can make an HTTP POST request to initiate a gitclaw installation. The installation logic lives in one place and can be invoked from anywhere.

---

## How It Works

### Architecture

```
┌──────────────────────┐
│   Trigger Sources     │
│                       │
│  - CLI tool           │
│  - Web portal         │
│  - Slack bot          │
│  - Another workflow   │
│  - API call           │
└────────┬──────────────┘
         │
         │ POST /repos/{owner}/{repo}/dispatches
         ▼
┌──────────────────────┐
│   Mothership Repo     │
│   (japer-technology/  │
│    gitclaw)           │
│                       │
│   Workflow:           │
│   on: repository_     │
│       dispatch        │
│                       │
│   1. Download files   │
│   2. Auth to target   │
│   3. Create branch    │
│   4. Commit .GITCLAW/ │
│   5. Open PR          │
└──────────────────────┘
         │
         │ GitHub API
         ▼
┌──────────────────────┐
│   Target Repository   │
│   (user/my-project)   │
│                       │
│   - New branch        │
│   - .GITCLAW/ committed│
│   - Bootstrap PR open │
└──────────────────────┘
```

### Step 1 — Send a Dispatch Event

Any system can trigger the installation by sending a `repository_dispatch` event to the mothership repository:

```bash
curl -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/japer-technology/gitclaw/dispatches \
  -d '{
    "event_type": "install",
    "client_payload": {
      "target_repo": "user/my-project",
      "provider": "anthropic",
      "model": "claude-sonnet-4-20250514",
      "thinking": "high"
    }
  }'
```

### Step 2 — Mothership Workflow Runs

The dispatch event triggers a workflow in the mothership repository:

```yaml
# .github/workflows/remote-install.yml
name: Remote Install
on:
  repository_dispatch:
    types: [install]

jobs:
  install:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Validate payload
        run: |
          TARGET="${{ github.event.client_payload.target_repo }}"
          if [ -z "$TARGET" ]; then
            echo "Error: target_repo is required"
            exit 1
          fi
          echo "Installing gitclaw on $TARGET"

      - name: Install to target repo
        env:
          GH_TOKEN: ${{ secrets.INSTALL_TOKEN }}
        run: |
          TARGET="${{ github.event.client_payload.target_repo }}"
          PROVIDER="${{ github.event.client_payload.provider || 'anthropic' }}"
          MODEL="${{ github.event.client_payload.model || 'claude-sonnet-4-20250514' }}"

          # Clone target repo
          git clone "https://x-access-token:${GH_TOKEN}@github.com/${TARGET}.git" target
          cd target

          # Copy .GITCLAW/ from mothership
          cp -r ../.GITCLAW/ .GITCLAW/

          # Configure
          # (apply provider, model, and other settings)

          # Create branch
          git checkout -b gitclaw/install
          git add .GITCLAW/
          git commit -m "feat: install gitclaw 🦞"
          git push origin gitclaw/install

          # Open PR
          gh pr create \
            --repo "$TARGET" \
            --title "🦞 Install GitClaw" \
            --body "This PR installs gitclaw..." \
            --base main \
            --head gitclaw/install
```

### Step 3 — PR Appears on Target Repository

The target repository now has a pull request containing the `.GITCLAW/` folder and all necessary configuration. The user reviews, merges, adds their API key, and gitclaw is active.

---

## Dispatch Event Schema

### Install Event

```json
{
  "event_type": "install",
  "client_payload": {
    "target_repo": "owner/repo-name",
    "provider": "anthropic",
    "model": "claude-sonnet-4-20250514",
    "thinking": "high",
    "version": "latest",
    "callback_url": "https://example.com/webhook/install-complete"
  }
}
```

### Update Event

```json
{
  "event_type": "update",
  "client_payload": {
    "target_repo": "owner/repo-name",
    "version": "v1.3.0"
  }
}
```

### Uninstall Event

```json
{
  "event_type": "uninstall",
  "client_payload": {
    "target_repo": "owner/repo-name",
    "cleanup": true
  }
}
```

### Payload Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `target_repo` | string | Yes | Full repo identifier (`owner/name`) |
| `provider` | string | No | LLM provider (default: `anthropic`) |
| `model` | string | No | Model identifier |
| `thinking` | string | No | Thinking mode (`high`, `medium`, `low`) |
| `version` | string | No | gitclaw version (default: `latest`) |
| `callback_url` | string | No | URL to notify when installation is complete |

---

## Authentication and Authorization

### Token Requirements

The dispatch event requires authentication at two levels:

1. **Dispatch token** — The caller needs a token with `repo` scope on the mothership repository to send the dispatch event.
2. **Installation token** — The mothership workflow needs a token with `repo` scope on the target repository to create branches, commit files, and open PRs.

### Token Strategies

| Strategy | How it works | Pros | Cons |
|----------|-------------|------|------|
| **Personal Access Token (PAT)** | User provides their PAT, stored as a secret in the mothership repo | Simple setup | Broad scope, tied to a single user |
| **Fine-Grained PAT** | PAT scoped to specific repos | Better security, limited scope | Must be updated when new repos are targeted |
| **GitHub App Installation Token** | A GitHub App generates short-lived tokens | Best security, scoped per installation | Requires a GitHub App |
| **OAuth Token** | User authenticates via OAuth, token is used for the dispatch | User-specific scope | Requires OAuth flow |

### Recommended Approach

Use a **GitHub App** as the installation mechanism:

1. User installs the GitHub App on their repository (or entire org).
2. The mothership workflow generates an installation access token for the target repo.
3. The token is short-lived (1 hour) and scoped to only the installed repos.

This provides the best security while still being automated.

---

## Trigger Sources

The beauty of the repository dispatch approach is that the trigger can come from anywhere:

### CLI Tool

```bash
# A CLI command that sends a dispatch event
npx gitclaw install --repo user/my-project
```

Behind the scenes, this sends the dispatch event to the mothership:

```typescript
async function triggerInstall(repo: string, config: Config) {
  await fetch(
    'https://api.github.com/repos/japer-technology/gitclaw/dispatches',
    {
      method: 'POST',
      headers: {
        Authorization: `token ${token}`,
        Accept: 'application/vnd.github.v3+json',
      },
      body: JSON.stringify({
        event_type: 'install',
        client_payload: { target_repo: repo, ...config },
      }),
    }
  );
}
```

### Web Portal

A form on the gitclaw website sends a dispatch event when the user clicks "Install":

```javascript
document.getElementById('install-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const repo = document.getElementById('repo').value;
  await triggerInstall(repo, getConfig());
  showSuccess(`Installation triggered on ${repo}`);
});
```

### Slack Bot

A Slack bot that responds to `/gitclaw install user/my-project`:

```
User: /gitclaw install user/my-project
Bot: 🦞 Installing gitclaw on user/my-project...
Bot: ✅ Installation triggered! A PR will appear on your repo shortly.
```

### Another GitHub Workflow

A workflow in a different repository can trigger gitclaw installation:

```yaml
# In any repo's workflow:
- name: Install gitclaw on new repo
  env:
    GH_TOKEN: ${{ secrets.DISPATCH_TOKEN }}
  run: |
    curl -X POST \
      -H "Authorization: token $GH_TOKEN" \
      -H "Accept: application/vnd.github.v3+json" \
      https://api.github.com/repos/japer-technology/gitclaw/dispatches \
      -d '{"event_type":"install","client_payload":{"target_repo":"${{ github.repository }}"}}'
```

### Chatbot / AI Assistant

An AI assistant that can install gitclaw when asked:

```
User: Install gitclaw on my project-alpha repository.
Bot: I'll set that up for you. Sending install request for user/project-alpha...
Bot: Done! A pull request has been created: https://github.com/user/project-alpha/pull/42
```

---

## Callback Mechanism

The dispatch event can include a `callback_url` for notification when the installation is complete:

```yaml
# In the mothership workflow, after successful installation:
- name: Notify callback
  if: github.event.client_payload.callback_url
  run: |
    curl -X POST "${{ github.event.client_payload.callback_url }}" \
      -H "Content-Type: application/json" \
      -d '{
        "status": "success",
        "target_repo": "${{ github.event.client_payload.target_repo }}",
        "pr_url": "$PR_URL",
        "version": "v1.2.0"
      }'
```

This enables the triggering system to show real-time feedback to the user.

---

## Strengths

- **Fully automated** — No manual file copying, no local commands. The entire installation happens server-side.
- **Composable** — The dispatch event can be triggered from any system: CLI, web, Slack, other workflows, scripts, cron jobs.
- **Centralized logic** — Installation logic lives in one place (the mothership repo). Updates to the installation process propagate to all triggers instantly.
- **Scalable** — Can handle installations across many repositories. The mothership workflow is the single source of truth.
- **Auditable** — Every dispatch event and workflow run is logged in GitHub Actions, providing a complete audit trail.
- **Version management** — The mothership always has the latest gitclaw version. Every installation gets the current release.
- **Callback support** — Triggers can receive notifications when installations complete, enabling rich integrations.

---

## Limitations

- **Requires a token with write access** — The user must grant a personal access token (or GitHub App installation) with write permissions on their target repository. This is a significant trust requirement.
- **Dispatch token management** — The token used to send dispatch events must be managed securely.
- **Mothership secrets** — The mothership repository must store tokens or app credentials that can access arbitrary target repos. This is a high-value secret.
- **GitHub Actions minutes** — Each dispatch event triggers a workflow run, consuming Actions minutes. For many installations, this adds up.
- **Asynchronous** — The dispatch event returns immediately (HTTP 204), but the actual installation happens asynchronously. The caller must poll or use a callback to know when it's done.
- **Rate limits** — GitHub API rate limits apply to both the dispatch call and the workflow's API operations.
- **Complex initial setup** — Setting up the mothership workflow, managing secrets, and handling edge cases requires more engineering than simpler methods.

---

## Security Considerations

- **Token scoping** — Use the most restrictive token possible. Fine-grained PATs scoped to specific repos are preferred over classic PATs with broad `repo` scope.
- **Payload validation** — The mothership workflow must validate all payload fields to prevent injection attacks (e.g., in the `target_repo` field).
- **Secret rotation** — Rotate tokens stored in the mothership's secrets regularly.
- **Audit trail** — GitHub Actions provides logs for every workflow run, enabling auditing of all installations.
- **Callback URL validation** — If using callbacks, validate the URL to prevent SSRF (Server-Side Request Forgery).
- **Rate limiting** — Implement rate limiting on who can send dispatch events to prevent abuse.

---

## When to Use This Method

This method is ideal when:

- You want a **single, centralized installation mechanism** that can be triggered from many different sources.
- You are building **integrations** (Slack bots, web portals, CLI tools) that need to programmatically install gitclaw.
- You need **automation at scale** — installing gitclaw across many repos via scripts or CI/CD.
- You want a complete **audit trail** of all installations.

---

## When to Consider Alternatives

Consider a different delivery method when:

- Your users are **uncomfortable granting write access tokens** to external systems.
- You want a **simpler setup** without mothership secrets and workflow management (use [template repo](./github-template-repository.md) or [CLI tool](./cli-tool.md)).
- You need **real-time feedback** during installation (the async nature of dispatches adds latency).

---

## Related Methods

- [GitHub Application](./github-application.md) — Can provide the installation tokens needed by the mothership workflow.
- [GitHub Marketplace Action](./github-marketplace-action.md) — Similar automation, but runs within the target repo instead of a mothership.
- [CLI Tool (npx / bunx)](./cli-tool.md) — Can use dispatch events as its backend installation mechanism.
- [Probot / Webhook Service](./probot-webhook-service.md) — Another event-driven approach that responds to GitHub events.

---

> 🦞 *Repository dispatch turns gitclaw installation into an API call — trigger it from anywhere, and the mothership handles the rest.*
