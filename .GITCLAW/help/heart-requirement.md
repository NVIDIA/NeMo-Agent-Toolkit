# ❤️ Heart Requirement

[← Back to Help](README.md)

---

Optionally require a ❤️ heart emoji in the body of new issues for the GitClaw agent to process them. This feature provides an additional layer of control over which issues trigger the agent.

## How It Works

GitClaw looks for a file matching `.GITCLAW/GITCLAW-HEART-REQUIRED.*` (any extension) in the repository. If such a file exists:

- **New issues** (`issues.opened` events) must contain a heart emoji (❤️) in the issue body. If the heart emoji is missing, the workflow exits at the Heart Guard step and the agent does not respond.
- **Comments on existing issues** (`issue_comment.created` events) are always processed regardless of this setting.

By default, the repository ships with `.GITCLAW/GITCLAW-HEART-NOT-REQUIRED.md`, which means the heart requirement is **not active** — all new issues are processed normally.

## Supported Heart Emojis

The guard recognizes a variety of heart emoji characters, including:

❤️ 💙 💚 💛 🧡 💜 🖤 🤍 🤎 💗 💖 💝 💘 💕 💞 💓 ♥️

## Enable Heart Requirement

Rename the sentinel file and push:

```bash
cd .GITCLAW
mv GITCLAW-HEART-NOT-REQUIRED.md GITCLAW-HEART-REQUIRED.md
git add -A
git commit -m "Require heart emoji for new issues"
git push
```

Once enabled, new issues without a heart emoji in the body will be skipped by the agent.

## Disable Heart Requirement

Rename the file back:

```bash
cd .GITCLAW
mv GITCLAW-HEART-REQUIRED.md GITCLAW-HEART-NOT-REQUIRED.md
git add -A
git commit -m "Remove heart emoji requirement"
git push
```

## How to Use

When the heart requirement is active, include a heart emoji anywhere in your issue body:

```
Please refactor the auth module ❤️
```

Or add it on its own line — the guard simply checks for the presence of any heart emoji character in the issue body.

## Workflow Position

The Heart Guard runs in the workflow as step 2, after the main Guard (GITCLAW-ENABLED) and before the Preinstall indicator:

```
1. Guard         (GITCLAW-ENABLED.ts)       — verify opt-in sentinel exists
2. Heart Guard   (GITCLAW-HEART-GUARD.ts)   — check heart emoji requirement
3. Preinstall    (GITCLAW-INDICATOR.ts)      — add 👀 reaction indicator
4. Install       (bun install)               — install npm/bun dependencies
5. Run           (GITCLAW-AGENT.ts)          — execute the AI coding agent
```

## See Also

- [💬 Issues Management](issues-management.md) — how issues and conversations work
- [⏸️ Disable](disable.md) — temporarily stop the agent
- [▶️ Enable](enable.md) — re-activate the agent
