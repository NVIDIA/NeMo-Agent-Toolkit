# .GITCLAW 🦞 Install

### These files are installed by GITCLAW-INSTALLER.yml

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

The `install/` directory contains the **installable payload** for `.GITCLAW`

Everything in this folder is intentionally flat (no nested subfolders) so it can be copied, vendored, or inspected quickly.

## Files in this folder

- `GITCLAW-INSTALLER.ts` — one-time installer script.
- `GITCLAW-WORKFLOW-AGENT.yml` — GitHub Actions workflow template copied to `.github/workflows/GITCLAW-WORKFLOW-AGENT.yml`.
- `GITCLAW-TEMPLATE-HATCH.md` — issue template copied to `.github/ISSUE_TEMPLATE/hatch.md`.
- `GITCLAW-AGENTS.md` — default agent identity/instructions copied to `.GITCLAW/AGENTS.md`.
- `package.json` and `package-lock.json` — runtime dependencies for the scripts under `.GITCLAW/`.

## Install process (step-by-step)

### 1) Place `.GITCLAW` at your repository root

The expected layout is:

```text
<repo>/
  .GITCLAW/
    install/
      GITCLAW-INSTALLER.ts
      GITCLAW-WORKFLOW-AGENT.yml
      GITCLAW-TEMPLATE-HATCH.md
      GITCLAW-AGENTS.md
      package.json
      package-lock.json
    lifecycle/
      GITCLAW-AGENT.ts
      GITCLAW-INDICATOR.ts
      GITCLAW-ENABLED.ts
```

### 2) Run the installer

From the repository root:

```bash
bun .GITCLAW/install/GITCLAW-INSTALLER.ts
```

The installer is **non-destructive**:

- If a destination file already exists, it skips it.
- If a destination file is missing, it installs it.

### 3) What `GITCLAW-INSTALLER.ts` installs

The script installs the following resources:

1. `.GITCLAW/install/GITCLAW-WORKFLOW-AGENT.yml` → `.github/workflows/GITCLAW-WORKFLOW-AGENT.yml`
2. `.GITCLAW/install/GITCLAW-TEMPLATE-HATCH.md` → `.github/ISSUE_TEMPLATE/hatch.md`
3. `.GITCLAW/install/GITCLAW-AGENTS.md` → `.GITCLAW/AGENTS.md`
4. Ensures `.gitattributes` contains:

```text
memory.log merge=union
```

That merge rule keeps the memory log append-only merge behavior safe when multiple branches update it.

### 4) Install dependencies

```bash
cd .GITCLAW
bun install
```

### 5) Configure secrets and push

1. Add `ANTHROPIC_API_KEY` in: **Repository Settings → Secrets and variables → Actions**.
2. Commit the new/installed files.
3. Push to GitHub.

### 6) (Optional) Enable the automated installer workflow

`GITCLAW-INSTALLER.yml` is a reusable GitHub Actions workflow that bootstraps gitclaw automatically whenever changes to `.GITCLAW/**` are pushed, or on demand via `workflow_dispatch`.

To activate it:

1. Copy `.GITCLAW/GITCLAW-INSTALLER.yml` into your `.github/workflows/` folder:

   ```bash
   cp .GITCLAW/GITCLAW-INSTALLER.yml .github/workflows/GITCLAW-INSTALLER.yml
   ```

2. Commit and push:

   ```bash
   git add .github/workflows/GITCLAW-INSTALLER.yml
   git commit -m "chore: add GITCLAW installer workflow"
   git push
   ```

3. To trigger it manually, go to **Actions → GITCLAW Bootstrap → Run workflow** in your GitHub repository.

### 7) Start using the agent

Open a GitHub issue. The workflow picks it up and the agent responds in issue comments.

## Why this structure exists

Keeping installable assets in `install/` provides:

- a single source of truth for what gets installed,
- a predictable payload for distribution,
- easier auditing of installation-time files,
- simpler automation for future installers.
