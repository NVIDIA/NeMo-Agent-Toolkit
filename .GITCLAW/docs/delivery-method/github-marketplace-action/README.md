# .GITCLAW 🦞 GitHub Marketplace Action

### A published GitHub Action that bootstraps gitclaw when added to any workflow — users add one YAML file, the action handles everything else.

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

---

## Overview

A GitHub Marketplace Action packages the gitclaw installation logic into a reusable, versioned GitHub Action that any repository can invoke. Instead of manually copying files and running an installer, a user creates a single workflow YAML file that references the gitclaw action. When triggered, the action downloads the latest `.GITCLAW/` folder, commits it to the repository, and opens a bootstrap pull request.

This approach is native to the GitHub ecosystem, discoverable via the GitHub Marketplace, and follows a pattern that millions of developers already use daily.

---

## How It Works

### Step 1 — User Adds a Workflow File

The user creates a single file in their repository:

```yaml
# .github/workflows/install-gitclaw.yml
name: Install GitClaw
on: workflow_dispatch

jobs:
  install:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
    steps:
      - uses: actions/checkout@v4

      - uses: japer-technology/gitclaw-action@v1
        with:
          provider: anthropic
          model: claude-sonnet-4-20250514
```

### Step 2 — User Triggers the Workflow

The user navigates to the **Actions** tab in their repository, selects the "Install GitClaw" workflow, and clicks **Run workflow**. Alternatively, the workflow can be configured to trigger on other events (push, schedule, issue comment, etc.).

### Step 3 — The Action Executes

Inside the GitHub Actions runner, the action performs:

1. **Downloads the latest `.GITCLAW/` release artifact** from the gitclaw repository (or a CDN/package registry).
2. **Copies the `.GITCLAW/` folder** into the repository's working directory.
3. **Runs the installer** (`GITCLAW-INSTALLER.ts`) to set up workflows, templates, and configuration.
4. **Creates a new branch** (e.g., `gitclaw/install`).
5. **Commits all changes** to the branch.
6. **Opens a pull request** with a descriptive title and body that includes next steps (e.g., adding the API key secret).

### Step 4 — User Merges the PR

The user reviews the generated PR — which contains the full `.GITCLAW/` folder and all bootstrapped files — and merges it into the default branch.

### Step 5 — Add API Key and Start Using

The user adds their LLM API key as a repository secret and begins interacting with gitclaw through issues.

---

## Action Architecture

### Action Metadata (`action.yml`)

The action is defined by an `action.yml` file in the `japer-technology/gitclaw-action` repository:

```yaml
name: 'Install GitClaw'
description: 'Bootstrap gitclaw in your repository — AI-powered assistant via GitHub Actions'
author: 'japer-technology'

branding:
  icon: 'terminal'
  color: 'red'

inputs:
  provider:
    description: 'LLM provider (anthropic, openai, etc.)'
    required: false
    default: 'anthropic'
  model:
    description: 'Model identifier'
    required: false
    default: 'claude-sonnet-4-20250514'
  branch:
    description: 'Branch name for the install PR'
    required: false
    default: 'gitclaw/install'
  version:
    description: 'gitclaw version to install (tag or "latest")'
    required: false
    default: 'latest'

runs:
  using: 'composite'
  steps:
    - name: Download gitclaw
      shell: bash
      run: |
        VERSION="${{ inputs.version }}"
        if [ "$VERSION" = "latest" ]; then
          VERSION=$(curl -s https://api.github.com/repos/japer-technology/gitclaw/releases/latest | jq -r .tag_name)
        fi
        curl -sL "https://github.com/japer-technology/gitclaw/releases/download/${VERSION}/gitclaw.tar.gz" | tar xz

    - name: Run installer
      shell: bash
      run: |
        npx tsx .GITCLAW/install/GITCLAW-INSTALLER.ts

    - name: Create PR
      shell: bash
      env:
        GH_TOKEN: ${{ github.token }}
      run: |
        git config user.name "gitclaw[bot]"
        git config user.email "gitclaw[bot]@users.noreply.github.com"
        git checkout -b ${{ inputs.branch }}
        git add .
        git commit -m "feat: install gitclaw 🦞"
        git push origin ${{ inputs.branch }}
        gh pr create \
          --title "🦞 Install GitClaw" \
          --body "This PR installs gitclaw into your repository..." \
          --base main \
          --head ${{ inputs.branch }}
```

### Composite vs. JavaScript vs. Docker Action

| Type | Pros | Cons |
|------|------|------|
| **Composite** | Simple, uses shell commands, no build step | Limited error handling, verbose YAML |
| **JavaScript** | Rich GitHub API integration via `@actions/core` and `@actions/github`, better error handling | Requires bundling, `node_modules` management |
| **Docker** | Full environment control, reproducible | Slower startup (container pull), larger size |

**Recommendation:** Start with a **composite action** for simplicity. Migrate to a **JavaScript action** if more sophisticated logic is needed (e.g., API-driven PR creation, conditional updates, rich output summaries).

---

## Publishing to the Marketplace

### Requirements

1. The action repository must be **public**.
2. The `action.yml` file must be in the root of the repository.
3. The action must have a `name`, `description`, and `branding` section.
4. A release tag (e.g., `v1`, `v1.0.0`) must be created.

### Steps

1. Create the `japer-technology/gitclaw-action` repository.
2. Add the `action.yml` and supporting files.
3. Create a release and tag it (`v1.0.0`).
4. On the release page, check **"Publish this Action to the GitHub Marketplace"**.
5. Fill in the Marketplace listing details (category, description, screenshots).

### Version Management

Follow the GitHub Actions versioning convention:

- **Major tag** (`v1`) — Points to the latest `v1.x.x` release. Users referencing `@v1` get bug fixes and minor features automatically.
- **Specific tag** (`v1.2.3`) — For users who want pinned, reproducible behavior.
- **`main` branch** — For users who want bleeding-edge (not recommended for production).

```bash
# After releasing v1.2.3:
git tag -fa v1 -m "Update v1 tag"
git push origin v1 --force
```

---

## Advanced Features

### Update Mode

The action can support an `update` mode that checks for new gitclaw versions and opens update PRs:

```yaml
- uses: japer-technology/gitclaw-action@v1
  with:
    mode: update
```

This mode:
1. Reads the current gitclaw version from the repository.
2. Checks the latest available version.
3. If an update is available, creates a branch with the new files and opens a PR with a changelog.

### Scheduled Updates

Combine the update mode with a scheduled trigger for fully automatic updates:

```yaml
name: Update GitClaw
on:
  schedule:
    - cron: '0 9 * * 1'  # Every Monday at 9 AM

jobs:
  update:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
      - uses: japer-technology/gitclaw-action@v1
        with:
          mode: update
```

### Configuration Inputs

The action can accept configuration inputs that customize the gitclaw installation:

| Input | Description | Default |
|-------|-------------|---------|
| `provider` | LLM provider | `anthropic` |
| `model` | Model identifier | `claude-sonnet-4-20250514` |
| `thinking` | Thinking mode (high, medium, low) | `high` |
| `personality` | Agent personality preset | `default` |
| `version` | gitclaw version to install | `latest` |
| `mode` | `install` or `update` | `install` |
| `branch` | Branch name for the PR | `gitclaw/install` |

---

## Strengths

- **GitHub-native** — Uses the same mechanism that powers the entire GitHub Actions ecosystem. Developers are already familiar with `uses:` syntax.
- **Discoverable** — Published on the GitHub Marketplace, searchable by anyone looking for AI coding assistants.
- **Versioned** — Tags provide clear version pinning and upgrade paths. Users control when they update.
- **No backend required** — Everything runs inside GitHub Actions. No external servers, no hosting costs.
- **Composable** — Can be combined with other actions in a workflow. For example, running after `actions/checkout` or before deployment steps.
- **Automatic updates** — With a scheduled workflow, updates happen without any manual intervention.
- **Transparent** — The action's source code is open. Users can audit exactly what it does.

---

## Limitations

- **Requires the initial YAML file** — The "add one workflow file" step still needs to happen. Users must create `.github/workflows/install-gitclaw.yml` manually, which means they already need to be somewhat GitHub Actions-savvy.
- **Bootstrap chicken-and-egg** — The workflow that installs gitclaw must exist before gitclaw is installed. This can be solved by pairing with a template repo or documenting the YAML clearly.
- **Runner time costs** — Each action run consumes GitHub Actions minutes. For public repos this is free; for private repos, it counts against the account's minute quota.
- **Network dependency** — The action needs to download the gitclaw release artifact during execution, requiring network access from the runner.

---

## Security Considerations

- **Permissions scoping** — The workflow should request only `contents: write` and `pull-requests: write`. Avoid `admin` or `secrets` permissions in the workflow file.
- **Tag immutability** — While major tags (`v1`) are updated, release tags (`v1.2.3`) should be immutable to prevent supply-chain attacks.
- **Release artifact integrity** — The downloaded gitclaw archive should be verified against a checksum or signature to ensure it hasn't been tampered with.
- **`GITHUB_TOKEN` scope** — The action uses the workflow's `GITHUB_TOKEN`, which is automatically scoped to the repository. No personal access tokens are needed.

---

## When to Use This Method

This method is ideal when:

- Your team is **already using GitHub Actions** and comfortable with workflow YAML files.
- You want a **zero-backend** approach that stays entirely within GitHub's infrastructure.
- You want **versioned installations** with clear upgrade paths.
- You want **automatic updates** via scheduled workflows.

---

## When to Consider Alternatives

Consider a different delivery method when:

- Your users are **not familiar with GitHub Actions** and would struggle to create the initial YAML file.
- You need a **one-click** experience with no file creation required (consider a [GitHub App](./github-application.md)).
- You want to install gitclaw **without any git operations** (consider the [CLI tool](./cli-tool.md) or [GitHub Pages portal](./github-pages-self-service-portal.md)).

---

## Related Methods

- [Fork / Import + Installer](./fork-import-installer.md) — The manual process this action automates.
- [GitHub Application](./github-application.md) — A one-click alternative that requires a backend.
- [GitHub Template Repository](./github-template-repository.md) — Pairs well with this action (template for new repos, action for existing repos).
- [Package Registry](./package-registry.md) — An alternative distribution mechanism for the gitclaw files.

---

> 🦞 *One YAML file. One workflow run. gitclaw is installed. That's the Marketplace Action promise.*
