# .GITCLAW 🦞 GitHub Application (OAuth / GitHub App)

### A registered GitHub App that installs gitclaw into any repository with a single click — no CLI, no fork, no manual file copying.

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

---

## Overview

A GitHub Application is a first-class integration registered on the GitHub platform that can act on behalf of users or organizations. By packaging gitclaw as a GitHub App, installation becomes a one-click process: a user visits the app's installation page, selects which repositories to enable, and the app automatically commits the `.GITCLAW/` folder, opens a bootstrap PR, and optionally configures secrets.

This is the most user-friendly delivery method for gitclaw. It eliminates every manual step from the current fork/import workflow and introduces automatic updates, centralized management, and native GitHub permissions handling.

---

## How It Works

### Step 1 — User Visits the App Installation Page

The GitHub App is registered under the `japer-technology` organization. Its public installation page is accessible at:

```
https://github.com/apps/gitclaw
```

Any GitHub user or organization admin can visit this page and click **Install**.

### Step 2 — Select Repositories

GitHub presents the standard app installation screen where the user chooses:

- **All repositories** — gitclaw is enabled for every repo in the account/org.
- **Only select repositories** — the user picks specific repos.

The user confirms the installation, which grants the app the configured permissions.

### Step 3 — App Receives Installation Event

When the user installs the app, GitHub sends an `installation` webhook event to the app's backend. This event includes:

- The installation ID.
- The list of selected repositories.
- The permissions granted.

### Step 4 — App Bootstraps Each Repository

For each selected repository, the app's backend:

1. **Creates a new branch** (e.g., `gitclaw/install`) using the GitHub API.
2. **Commits the `.GITCLAW/` folder** contents to that branch via the Git Trees and Commits API.
3. **Opens a pull request** from the install branch to the default branch.
4. **Adds a descriptive PR body** with setup instructions, including how to add the API key secret.
5. **(Optional)** If the app has `secrets: write` permission, it can create the API key secret directly.

### Step 5 — User Merges the PR

The user reviews the PR — which contains the entire `.GITCLAW/` folder, workflow files, and configuration — and merges it. Gitclaw is now active.

### Step 6 — Ongoing Management

After installation, the app can:

- **Push updates** — When a new gitclaw version is released, the app opens PRs in all installed repos with the updated files.
- **Monitor health** — The app can check if workflows are functioning, secrets are configured, and the agent is responding.
- **Handle uninstallation** — When the user uninstalls the app, a cleanup workflow can optionally remove `.GITCLAW/` and associated workflows.

---

## Architecture

```
┌─────────────────────────┐
│   GitHub App Backend     │
│   (Serverless Function)  │
│                          │
│   Handles webhooks:      │
│   - installation         │
│   - installation_repos   │
│   - push (for updates)   │
└────────┬─────────────────┘
         │ GitHub API
         ▼
┌─────────────────────────┐
│   Target Repository      │
│                          │
│   1. Create branch       │
│   2. Commit .GITCLAW/    │
│   3. Open PR             │
│   4. (Optional) Set      │
│      secrets             │
└──────────────────────────┘
```

### Backend Options

The app backend can be hosted on any serverless platform:

| Platform | Pros | Cons |
|----------|------|------|
| **Cloudflare Workers** | Free tier, global edge, fast cold starts | Limited runtime (CPU time limits) |
| **Vercel Serverless Functions** | Easy deployment, GitHub integration | Cold starts on free tier |
| **AWS Lambda** | Mature, scalable, extensive ecosystem | More setup required |
| **GitHub Actions** (self-hosted) | No external service needed | Slower response to webhooks |

### Authentication Flow

GitHub Apps authenticate using a two-step process:

1. **App-level authentication** — The app signs a JWT using its private key. This JWT is used to request an installation access token.
2. **Installation access token** — This token is scoped to the specific installation (user/org + repos) and expires after 1 hour.

```
App Private Key → JWT → GitHub API → Installation Access Token → Repository Operations
```

This is more secure than personal access tokens because:
- Tokens are short-lived (1 hour).
- Permissions are explicitly scoped.
- The app's identity is separate from any user.

---

## Required Permissions

The GitHub App needs the following permissions:

| Permission | Access | Purpose |
|------------|--------|---------|
| **Contents** | Read & Write | Commit `.GITCLAW/` files and workflow definitions |
| **Pull Requests** | Read & Write | Open bootstrap and update PRs |
| **Workflows** | Read & Write | Commit workflow YAML files to `.github/workflows/` |
| **Actions** | Read | Verify workflow execution status |
| **Metadata** | Read | Required by default for all apps |
| **Secrets** | Write (optional) | Automatically configure LLM API key secrets |
| **Issues** | Read & Write (optional) | Create initial setup issue with instructions |

### Webhook Events

| Event | Purpose |
|-------|---------|
| `installation` | Triggered when the app is installed — initiates repository setup |
| `installation_repositories` | Triggered when repos are added/removed from an existing installation |
| `push` | (Optional) Monitor for manual changes to `.GITCLAW/` files |
| `release` | (Optional) Trigger updates when a new gitclaw version is released |

---

## Implementation Details

### Committing Files via the GitHub API

The app uses the Git Data API to create commits without needing to clone the repository:

1. **Get the current default branch reference:**
   ```
   GET /repos/{owner}/{repo}/git/ref/heads/main
   ```

2. **Get the commit at the tip of the branch:**
   ```
   GET /repos/{owner}/{repo}/git/commits/{sha}
   ```

3. **Create blobs for each file in `.GITCLAW/`:**
   ```
   POST /repos/{owner}/{repo}/git/blobs
   {
     "content": "<base64-encoded file content>",
     "encoding": "base64"
   }
   ```

4. **Create a tree containing all the blobs:**
   ```
   POST /repos/{owner}/{repo}/git/trees
   {
     "base_tree": "<existing tree sha>",
     "tree": [
       { "path": ".GITCLAW/config.yml", "mode": "100644", "type": "blob", "sha": "<blob sha>" },
       ...
     ]
   }
   ```

5. **Create a commit pointing to the new tree:**
   ```
   POST /repos/{owner}/{repo}/git/commits
   {
     "message": "feat: install gitclaw 🦞",
     "tree": "<tree sha>",
     "parents": ["<parent commit sha>"]
   }
   ```

6. **Create a new branch reference:**
   ```
   POST /repos/{owner}/{repo}/git/refs
   {
     "ref": "refs/heads/gitclaw/install",
     "sha": "<commit sha>"
   }
   ```

7. **Open a pull request:**
   ```
   POST /repos/{owner}/{repo}/pulls
   {
     "title": "🦞 Install GitClaw",
     "head": "gitclaw/install",
     "base": "main",
     "body": "..."
   }
   ```

### Automatic Updates

When a new version of gitclaw is released:

1. The app's backend is notified (via a webhook on the gitclaw source repo, a cron job, or a manual trigger).
2. For each installed repository, the app:
   - Checks the current `.GITCLAW/` version.
   - If outdated, creates a branch, commits the updated files, and opens a PR.
3. The PR body includes a changelog summary so the user knows what changed.

### Version Tracking

The app can track versions by:
- Reading a version identifier from `.GITCLAW/` (e.g., a `version` field in a config file).
- Storing installation metadata in a database (e.g., Cloudflare KV, DynamoDB, or even a GitHub repository as a data store).

---

## User Experience

### Installation Flow

```
User clicks "Install" on GitHub
        │
        ▼
Selects repositories
        │
        ▼
Confirms permissions
        │
        ▼
App receives webhook → commits .GITCLAW/ → opens PR
        │
        ▼
User reviews and merges PR
        │
        ▼
User adds API key secret
        │
        ▼
gitclaw is live 🦞
```

### Management Dashboard

If the app is listed on GitHub Marketplace, users manage installations through:

```
GitHub → Settings → Applications → gitclaw → Configure
```

From here they can:
- Add or remove repositories.
- Suspend or uninstall the app.
- View permissions.

---

## Strengths

- **One-click install** — The simplest possible onboarding experience. No CLI, no terminal, no file copying.
- **Automatic updates** — The app can push new gitclaw versions as PRs, keeping installations current with zero user effort.
- **Centralized management** — Install, update, and uninstall across multiple repositories from a single dashboard.
- **Native permissions** — GitHub's app permission model provides granular access control. Users see exactly what the app can do.
- **Scalable** — Works for individual developers installing on one repo and for organizations deploying across hundreds of repos.
- **Professional presentation** — A registered GitHub App has a profile page, description, and logo — lending credibility to the project.

---

## Limitations

- **Requires a backend** — Even a minimal one. The app needs a server (or serverless function) to receive webhooks and make API calls.
- **Hosting costs** — While serverless platforms have generous free tiers, there is some operational overhead and potential cost as usage scales.
- **App review process** — If published to the GitHub Marketplace, the app must pass GitHub's review process. This adds time to the initial release but ensures quality.
- **Permission sensitivity** — Requesting `contents: write` and `workflows: write` on user repositories is a significant trust ask. Clear documentation and minimal scoping are essential.
- **Secret management** — If the app sets up API key secrets, it must handle those values securely. This adds complexity and security considerations.
- **Maintenance burden** — The backend must be kept running, monitored for errors, and updated when GitHub changes its API or webhook format.

---

## Security Considerations

- **Private key storage** — The app's private key must be stored securely (environment variables, secret manager) and never committed to source code.
- **Token expiry** — Installation access tokens expire after 1 hour, limiting the blast radius of a token leak.
- **Minimal permissions** — Request only the permissions absolutely needed. Avoid `admin` or broad organization-level scopes.
- **Webhook signature verification** — Always verify the `X-Hub-Signature-256` header on incoming webhooks to ensure they originate from GitHub.
- **Rate limiting** — The GitHub API has rate limits (5000 requests/hour for app installations). Batch operations across many repos must respect these limits.

---

## When to Use This Method

This method is ideal when:

- You want the **simplest possible installation experience** for end users.
- You need to support **automatic updates** across many repositories.
- You are deploying gitclaw to an **organization** with many repos and want centralized control.
- You want a **professional, polished** distribution channel.

---

## When to Consider Alternatives

Consider a different delivery method when:

- You don't want to host or maintain a backend service.
- Your users are comfortable with the CLI and prefer a **manual, transparent** installation.
- You need the installation to work in **air-gapped or restricted environments** where external apps cannot be installed.

---

## Related Methods

- [Fork / Import + Installer](./fork-import-installer.md) — The manual approach this method automates.
- [GitHub Marketplace Action](./github-marketplace-action.md) — Another GitHub-native approach that doesn't require a backend.
- [Third-Party Website](./third-party-website.md) — A richer UI that could use a GitHub App under the hood.
- [Probot / Webhook Service](./probot-webhook-service.md) — An event-driven alternative built on the Probot framework.

---

> 🦞 *A GitHub App turns gitclaw from a folder you copy into a service you install. One click, and it's there.*
