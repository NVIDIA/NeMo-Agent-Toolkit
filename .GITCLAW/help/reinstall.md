# ♻️ Reinstall GitClaw

[← Back to Help](README.md)

---

Reset or upgrade an existing GitClaw installation. Use this when you want a fresh start, need to update to a newer version, or want to fix a broken installation.

## When to Reinstall

- **Upgrading** — a newer version of GitClaw is available with new features or fixes
- **Broken state** — session files or configuration got corrupted
- **Fresh start** — you want to clear all conversation history and start over
- **Workflow changes** — the workflow template has been updated upstream

## Upgrade — Preserve Configuration and History

To upgrade GitClaw while keeping your existing configuration, personality, and conversation history:

### 1. Back up your customizations

```bash
# Save your agent identity
cp .GITCLAW/AGENTS.md /tmp/AGENTS.md.bak

# Save your LLM settings
cp .GITCLAW/.pi/settings.json /tmp/settings.json.bak

# Save your system prompt (if customized)
cp .GITCLAW/.pi/APPEND_SYSTEM.md /tmp/APPEND_SYSTEM.md.bak
```

### 2. Replace the `.GITCLAW` folder

Remove the old folder and copy in the new version:

```bash
rm -rf .GITCLAW
# Copy the new .GITCLAW folder from the latest release
```

### 3. Restore your customizations

```bash
cp /tmp/AGENTS.md.bak .GITCLAW/AGENTS.md
cp /tmp/settings.json.bak .GITCLAW/.pi/settings.json
cp /tmp/APPEND_SYSTEM.md.bak .GITCLAW/.pi/APPEND_SYSTEM.md
```

### 4. Run the installer and install dependencies

```bash
bun .GITCLAW/install/GITCLAW-INSTALLER.ts
cd .GITCLAW && bun install
```

> The installer never overwrites existing files — it only creates missing ones. This means your existing `GITCLAW-WORKFLOW-AGENT.yml` workflow will remain as-is. If the workflow template has changed, delete `.github/workflows/GITCLAW-WORKFLOW-AGENT.yml` before running the installer to get the latest version.

### 5. Commit and push

```bash
git add -A
git commit -m "Upgrade gitclaw"
git push
```

## Clean Reinstall — Start Fresh

To completely reset GitClaw, including all conversation history:

### 1. Remove everything

```bash
rm -rf .GITCLAW
rm -f .github/workflows/GITCLAW-WORKFLOW-AGENT.yml
rm -f .github/ISSUE_TEMPLATE/hatch.md
```

### 2. Copy in a fresh `.GITCLAW` folder

Copy the `.GITCLAW` folder from the latest release into your repo root.

### 3. Run the full installation

```bash
bun .GITCLAW/install/GITCLAW-INSTALLER.ts
cd .GITCLAW && bun install
```

### 4. Reconfigure

- Add your API key secret in **Settings → Secrets and variables → Actions** (if not already present)
- Edit `.GITCLAW/.pi/settings.json` to set your preferred provider and model (see [Configure](configure.md))

### 5. Commit and push

```bash
git add -A
git commit -m "Reinstall gitclaw"
git push
```

## Fixing a Broken Workflow

If only the GitHub Actions workflow is broken or outdated:

```bash
# Remove the old workflow
rm .github/workflows/GITCLAW-WORKFLOW-AGENT.yml

# Re-run the installer to regenerate it from the template
bun .GITCLAW/install/GITCLAW-INSTALLER.ts

git add -A
git commit -m "Regenerate gitclaw workflow"
git push
```

## Resetting Session State Only

If you want to clear conversation history without reinstalling:

```bash
rm -rf .GITCLAW/state/sessions/*
rm -rf .GITCLAW/state/issues/*

git add -A
git commit -m "Clear gitclaw session history"
git push
```

The agent will start fresh conversations for all issues going forward.

## See Also

- [🔧 Install](install.md) — first-time installation guide
- [🗑️ Uninstall](uninstall.md) — complete removal
- [⚙️ Configure](configure.md) — customize after reinstalling
