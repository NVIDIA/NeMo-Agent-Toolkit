# ▶️ Enable GitClaw

[← Back to Help](README.md)

---

Re-activate a previously disabled GitClaw agent, or confirm that GitClaw is currently enabled.

## How Enabling Works

GitClaw uses a **fail-closed** security model controlled by a single sentinel file:

```
.GITCLAW/GITCLAW-ENABLED.md
```

When this file exists in the repository, the agent is **enabled** and will respond to issues and comments. When it's missing, the agent is **disabled** and all workflow runs exit immediately at the Guard step.

## Enable the Agent

### If the sentinel file was deleted

Recreate the sentinel file and push:

```bash
cat > .GITCLAW/GITCLAW-ENABLED.md << 'EOF'
# .GITCLAW 🦞 Enabled

### Delete or rename this file to disable .GITCLAW
EOF

git add .GITCLAW/GITCLAW-ENABLED.md
git commit -m "Enable gitclaw"
git push
```

### If the sentinel file was renamed

Rename it back:

```bash
mv .GITCLAW/GITCLAW-DISABLED.md .GITCLAW/GITCLAW-ENABLED.md
git add -A
git commit -m "Enable gitclaw"
git push
```

### If the workflow was disabled via GitHub UI

1. Go to your repository on GitHub
2. Click the **Actions** tab
3. Select the **GITCLAW-WORKFLOW-AGENT** workflow in the left sidebar
4. Click **Enable workflow**

> **Note:** If both the sentinel file is missing AND the workflow is disabled, you need to do both: restore the sentinel file and re-enable the workflow.

## Verify GitClaw Is Enabled

Check that all of the following are in place:

| Check | How to Verify |
|-------|---------------|
| Sentinel file exists | `ls .GITCLAW/GITCLAW-ENABLED.md` — file should be present |
| Workflow exists | `ls .github/workflows/GITCLAW-WORKFLOW-AGENT.yml` — file should be present |
| Workflow is active | Go to **Actions** tab — workflow should not show "This workflow is disabled" |
| API key is set | Go to **Settings → Secrets and variables → Actions** — the provider secret should be listed |

## Testing After Enabling

Open a test issue or comment on an existing one. You should see:

1. A workflow run appears in the **Actions** tab
2. The Guard step passes (shows "GitClaw enabled — GITCLAW-ENABLED.md found.")
3. A 👀 reaction appears on the issue
4. The agent posts a reply

## Enabling for the First Time

If you haven't installed GitClaw yet, see [Install](install.md) instead. The sentinel file is included in the `.GITCLAW` folder by default — no extra steps are needed to enable it during initial setup.

## See Also

- [⏸️ Disable](disable.md) — temporarily stop the agent
- [⚙️ Configure](configure.md) — adjust settings after enabling
- [🔧 Install](install.md) — first-time setup
