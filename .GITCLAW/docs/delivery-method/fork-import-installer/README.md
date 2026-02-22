# .GITCLAW 🦞 Fork / Import + Installer

### The foundational delivery method — manually copy the `.GITCLAW/` folder into your repository and run the installer script.

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

---

## Overview

The Fork / Import + Installer method is the original and current way gitclaw reaches a repository. It relies on a developer manually obtaining the `.GITCLAW/` folder — either by forking the gitclaw repository, importing it, or directly copying the folder — and then running a local installer script that bootstraps the necessary GitHub Actions workflows, issue templates, and configuration files into the target repository.

This approach is intentionally simple: no external services, no third-party dependencies, no hosted backends. Everything lives inside the repository itself. The trade-off is that it requires a few manual steps and some familiarity with the command line.

---

## How It Works

### Step 1 — Obtain the `.GITCLAW/` Folder

There are several ways to get the `.GITCLAW/` folder into your repository:

**Option A — Fork the Repository**

1. Navigate to the [gitclaw repository](https://github.com/japer-technology/gitclaw) on GitHub.
2. Click the **Fork** button to create a copy under your GitHub account.
3. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/gitclaw.git
   ```
4. The `.GITCLAW/` folder is already present in the cloned repository.

**Option B — Import the Repository**

1. Use GitHub's **Import Repository** feature (`https://github.com/new/import`).
2. Provide the gitclaw repository URL.
3. GitHub creates a new repository with all the contents, including `.GITCLAW/`.

**Option C — Direct Copy**

1. Clone or download the gitclaw repository.
2. Copy the `.GITCLAW/` folder into the root of your target repository:
   ```bash
   cp -r gitclaw/.GITCLAW/ /path/to/your-repo/.GITCLAW/
   ```

### Step 2 — Run the Installer

The installer is a TypeScript script that reads the contents of `.GITCLAW/` and sets up the required GitHub Actions workflows, issue templates, and PR templates inside the `.github/` directory.

```bash
cd /path/to/your-repo
bun .GITCLAW/install/GITCLAW-INSTALLER.ts
```

Alternatively, if you prefer Node.js:
```bash
npx tsx .GITCLAW/install/GITCLAW-INSTALLER.ts
```

The installer performs the following actions:

| Action | Description |
|--------|-------------|
| Copy workflow files | Places GitHub Actions workflow YAML files into `.github/workflows/` |
| Copy issue templates | Places issue template YAML files into `.github/ISSUE_TEMPLATE/` |
| Copy PR templates | Places pull request template files into `.github/` |
| Generate configuration | Creates or updates configuration files as needed |

### Step 3 — Push Changes

After the installer completes, commit and push all changes to your repository:

```bash
git add .
git commit -m "feat: install gitclaw"
git push origin main
```

### Step 4 — Bootstrap

Once the changes are pushed:

1. The bootstrap GitHub Actions workflow automatically triggers.
2. It creates an initial pull request with any remaining setup steps.
3. Merge the PR to activate gitclaw.

### Step 5 — Add API Key Secret

The final step is to add your LLM provider API key as a GitHub repository secret:

1. Go to your repository **Settings** → **Secrets and variables** → **Actions**.
2. Click **New repository secret**.
3. Add the secret name and value as specified in gitclaw's configuration (e.g., `ANTHROPIC_API_KEY`).
4. Save the secret.

Gitclaw is now live. Open an issue to interact with the agent.

---

## Detailed Workflow

```
┌─────────────────────┐
│  Developer Machine  │
│                     │
│  1. Fork / Clone    │
│  2. Copy .GITCLAW/  │
│  3. Run installer   │
│  4. git push        │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   GitHub Repository  │
│                      │
│  .GITCLAW/           │
│  .github/workflows/  │
│  .github/ISSUE_TEMPLATE/ │
│                      │
│  5. Bootstrap workflow│
│     triggers         │
│  6. Creates setup PR │
└────────┬─────────────┘
         │
         ▼
┌─────────────────────┐
│  Developer merges PR │
│  Adds API key secret │
│  gitclaw is active   │
└──────────────────────┘
```

---

## Requirements

| Requirement | Details |
|-------------|---------|
| **Runtime** | [Bun](https://bun.sh/) or [Node.js](https://nodejs.org/) with `tsx` |
| **Git** | Git CLI installed locally |
| **GitHub Account** | Repository write access |
| **LLM API Key** | API key for the chosen LLM provider (e.g., Anthropic, OpenAI) |
| **Operating System** | macOS, Linux, or Windows (with WSL recommended) |

---

## Strengths

- **Zero external dependencies** — No hosted services, no backends, no third-party integrations. Everything is self-contained within the repository.
- **Full control** — The developer sees exactly what is being installed and can review every file before committing.
- **Transparency** — No black-box automation. Every workflow, template, and configuration is a readable file in the repo.
- **No permissions required** — Beyond normal repo write access, no OAuth tokens, GitHub App installations, or API scopes are needed.
- **Works offline** — Once the `.GITCLAW/` folder is obtained, the installer runs entirely locally.
- **Universal compatibility** — Works with any GitHub repository, regardless of language, framework, or existing CI/CD setup.

---

## Limitations

- **Manual process** — Requires multiple steps: obtain the folder, run the installer, push, merge, add secrets. Each step is a potential point of confusion or error.
- **Requires CLI familiarity** — The developer must be comfortable running commands in a terminal. Not accessible to non-technical users.
- **Runtime dependency** — Requires Bun or Node.js installed on the developer's machine.
- **No automatic updates** — When gitclaw releases a new version, the developer must manually re-copy the `.GITCLAW/` folder and re-run the installer.
- **Fork baggage** — If using the fork method, the forked repo retains the full commit history of the gitclaw repository, which may be unwanted.
- **Import limitations** — GitHub's import feature can be slow for large repositories and doesn't preserve fork relationships.

---

## When to Use This Method

This method is ideal when:

- You want **maximum control** over what gets installed in your repository.
- You are comfortable with the command line and git operations.
- You prefer a **zero-dependency** approach with no external services.
- You are setting up gitclaw for a **single repository** and don't need to manage multiple installations.
- You are evaluating gitclaw and want to understand its internals before committing to it.

---

## When to Consider Alternatives

Consider a different delivery method when:

- You need to install gitclaw across **many repositories** and want automation.
- Your team includes **non-technical users** who cannot run CLI commands.
- You want **automatic updates** when new gitclaw versions are released.
- You need a **one-click** installation experience.

---

## Troubleshooting

### Installer fails with "bun: command not found"

Install Bun:
```bash
curl -fsSL https://bun.sh/install | bash
```

Or use Node.js with tsx instead:
```bash
npx tsx .GITCLAW/install/GITCLAW-INSTALLER.ts
```

### Workflows don't trigger after pushing

- Ensure the workflow files are in `.github/workflows/` (not `.GITCLAW/`).
- Check that the repository has GitHub Actions enabled in **Settings** → **Actions** → **General**.
- Verify the workflow trigger events match what you're doing (e.g., `on: issues` for issue-based triggers).

### Bootstrap PR is not created

- Check the Actions tab for any failed workflow runs.
- Ensure the `GITHUB_TOKEN` has sufficient permissions (usually automatic for Actions).
- Verify the workflow YAML syntax is valid.

### API key secret not working

- Ensure the secret name matches exactly what the workflow references.
- Check that the secret is set at the repository level, not the environment level (unless the workflow specifies an environment).
- Verify the API key is valid and has sufficient quota.

---

## Related Methods

- [GitHub Template Repository](./github-template-repository.md) — For new repos, avoids the fork/copy step entirely.
- [CLI Tool (npx / bunx)](./cli-tool.md) — Wraps this same process into a single command.
- [Git Submodule / Subtree](./git-submodule-subtree.md) — Alternative to copying files, with built-in update support.

---

> 🦞 *The fork-and-install path is where every gitclaw journey begins. It's the most hands-on way to understand how gitclaw works — and that understanding is a feature, not a limitation.*
