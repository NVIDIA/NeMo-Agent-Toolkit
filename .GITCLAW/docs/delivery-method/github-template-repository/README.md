# .GITCLAW 🦞 GitHub Template Repository

### Mark the gitclaw repository (or a dedicated starter repo) as a template — users click "Use this template" and get a new repo with gitclaw pre-installed.

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

---

## Overview

A GitHub Template Repository is a standard GitHub repository that has been flagged as a "template." This enables a special button on the repo page — **"Use this template"** — that allows anyone to create a new repository from the template with all its files and directory structure, but with a clean commit history.

For gitclaw, this means a user can create a brand-new repository that comes pre-loaded with the `.GITCLAW/` folder, workflow files, issue templates, and configuration — all without forking, cloning, or running any commands.

This is the lowest-effort delivery method for new repositories.

---

## How It Works

### Step 1 — Create or Designate the Template Repository

A repository is prepared that contains:

- The complete `.GITCLAW/` folder.
- Pre-configured `.github/workflows/` with gitclaw's workflow files.
- Pre-configured `.github/ISSUE_TEMPLATE/` with gitclaw's issue templates.
- A `README.md` with getting-started instructions.
- A `.gitignore` tailored to the intended use case.
- (Optional) Example issues, starter code, or configuration files.

This repository is then marked as a template:

1. Go to the repository **Settings**.
2. Under **General**, check **"Template repository"**.
3. Save.

### Step 2 — User Creates a New Repository from the Template

1. The user navigates to the template repository on GitHub.
2. Clicks the green **"Use this template"** button.
3. Selects **"Create a new repository"**.
4. Fills in:
   - **Repository name** — Their desired name.
   - **Description** — Optional description.
   - **Visibility** — Public or private.
   - **Owner** — Their personal account or an organization.
5. Clicks **"Create repository"**.

GitHub creates a new repository with:
- All files and directories from the template.
- A clean, single-commit history (no fork relationship).
- No connection to the template repository's issues, PRs, or branches.

### Step 3 — User Configures gitclaw

After the repository is created:

1. The user adds their LLM API key as a repository secret (**Settings** → **Secrets and variables** → **Actions**).
2. (Optional) The user customizes gitclaw's configuration files in `.GITCLAW/`.
3. The user opens their first issue to interact with the gitclaw agent.

Since the workflow files are already in place, gitclaw is ready to go as soon as the secret is configured.

---

## Template Repository Structure

```
template-repo/
├── .GITCLAW/
│   ├── install/
│   │   └── GITCLAW-INSTALLER.ts
│   ├── docs/
│   │   └── ...
│   ├── config/
│   │   └── ...
│   └── ...
├── .github/
│   ├── workflows/
│   │   ├── gitclaw-agent.yml
│   │   ├── gitclaw-bootstrap.yml
│   │   └── ...
│   ├── ISSUE_TEMPLATE/
│   │   ├── gitclaw-task.yml
│   │   └── ...
│   └── PULL_REQUEST_TEMPLATE.md
├── README.md
├── .gitignore
└── LICENSE
```

### Key Design Decisions

**Pre-run the installer vs. include raw `.GITCLAW/` only:**

| Approach | Pros | Cons |
|----------|------|------|
| **Pre-installed** (workflows already in `.github/`) | Zero-step setup — just add the API key | Template is tightly coupled to a specific gitclaw version |
| **Raw `.GITCLAW/` only** (user runs installer) | Installer generates files tailored to the repo | Adds a manual step; defeats the "zero CLI" benefit |

**Recommendation:** Use the **pre-installed** approach. The entire point of the template method is to eliminate manual steps. The user should only need to add their API key.

**Single template vs. multiple templates:**

Consider offering multiple templates for different use cases:

| Template | Description |
|----------|-------------|
| `gitclaw-starter` | Minimal repo with just gitclaw, ready for any project |
| `gitclaw-typescript` | TypeScript project with gitclaw pre-configured |
| `gitclaw-python` | Python project with gitclaw pre-configured |
| `gitclaw-docs` | Documentation-focused repo with gitclaw |

---

## API-Based Template Creation

Templates can also be used programmatically via the GitHub API:

```bash
curl -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/japer-technology/gitclaw-template/generate \
  -d '{
    "owner": "my-username",
    "name": "my-new-project",
    "description": "My project with gitclaw",
    "private": false,
    "include_all_branches": false
  }'
```

This enables automation tools, CLIs, and web portals to create gitclaw-enabled repositories programmatically.

---

## Keeping the Template Updated

When a new gitclaw version is released, the template repository must be updated to include the latest files. Options:

### Manual Update

1. Pull the latest gitclaw release.
2. Replace the `.GITCLAW/` folder and workflow files in the template repo.
3. Commit and push.

### Automated Update via GitHub Actions

A workflow in the template repository can automatically pull updates:

```yaml
name: Update GitClaw
on:
  repository_dispatch:
    types: [gitclaw-release]
  schedule:
    - cron: '0 0 * * 1'  # Weekly

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Download latest gitclaw
        run: |
          VERSION=$(curl -s https://api.github.com/repos/japer-technology/gitclaw/releases/latest | jq -r .tag_name)
          curl -sL "https://github.com/japer-technology/gitclaw/releases/download/${VERSION}/gitclaw.tar.gz" | tar xz
          # Replace .GITCLAW/ with updated version

      - name: Commit updates
        run: |
          git config user.name "gitclaw[bot]"
          git config user.email "gitclaw[bot]@users.noreply.github.com"
          git add .
          git commit -m "chore: update gitclaw to ${VERSION}" || exit 0
          git push
```

---

## Strengths

- **Zero CLI required** — The entire process is browser-based. No terminal, no git commands, no runtime installations.
- **Clean commit history** — Unlike forks, template-created repos start with a fresh history. No upstream baggage.
- **Instant setup** — The repository is created with all files in place. The user only needs to add their API key secret.
- **GitHub-native UX** — The "Use this template" button is a standard GitHub feature that developers already understand.
- **No backend required** — No servers, no hosted services, no OAuth. It's just a repository setting.
- **Zero cost** — Templates are a free GitHub feature.
- **Discoverable** — Template repositories appear in GitHub search results and can be promoted via topics and descriptions.

---

## Limitations

- **New repositories only** — This method only works when creating a **new** repository. It does not solve the problem of adding gitclaw to an existing project.
- **No automatic updates** — Repositories created from a template are independent copies. When the template is updated, existing repos created from it are not affected. Users must manually update their gitclaw installation.
- **Template maintenance** — The template repository must be kept up to date with the latest gitclaw version. If it falls behind, users will install an outdated version.
- **One-size-fits-all** — The template contains a fixed configuration. Users who need different settings must modify files after creation.
- **No fork relationship** — Unlike forks, template-created repos have no upstream connection. This means no easy pull of updates from the source.

---

## Combining with Other Methods

The Template Repository method is most powerful when combined with other delivery methods:

| Combination | Use Case |
|-------------|----------|
| **Template + Marketplace Action** | Template for new repos, Action for adding to existing repos |
| **Template + CLI Tool** | Template for browser-based setup, CLI for terminal-based setup |
| **Template + GitHub App** | Template for quick starts, App for managed installations with updates |

This "new repos use the template, existing repos use X" pattern covers the full spectrum of use cases.

---

## When to Use This Method

This method is ideal when:

- You are creating a **brand-new repository** and want gitclaw from the start.
- You want the **simplest possible setup** with no CLI or technical knowledge required.
- You are running a **workshop, tutorial, or demo** and need participants to get started instantly.
- You are an **organization** that wants all new projects to start with gitclaw pre-configured.

---

## When to Consider Alternatives

Consider a different delivery method when:

- You need to add gitclaw to an **existing repository** (this method only works for new repos).
- You need **automatic updates** when new gitclaw versions are released.
- You need **customized configurations** per repository that can't be handled by a single template.

---

## Related Methods

- [Fork / Import + Installer](./fork-import-installer.md) — For adding gitclaw to existing repos.
- [GitHub Marketplace Action](./github-marketplace-action.md) — Complements templates by handling the "existing repo" case.
- [CLI Tool (npx / bunx)](./cli-tool.md) — Another way to add gitclaw to existing repos.
- [GitHub Application](./github-application.md) — Provides automatic updates that templates lack.

---

> 🦞 *Click "Use this template." Name your repo. Add your API key. You're done. That's the template experience.*
