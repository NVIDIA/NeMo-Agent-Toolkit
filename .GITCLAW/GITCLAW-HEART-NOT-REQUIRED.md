# .GITCLAW 🦞 Heart Not Required

### Rename this file to `GITCLAW-HEART-REQUIRED.md` to require a ❤️ heart emoji in the issue body for the agent to process new issues.

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

## File existence behavior

The workflow runs `.GITCLAW/lifecycle/GITCLAW-HEART-GUARD.ts` as a guard step. If a file matching `GITCLAW-HEART-REQUIRED.*` exists in the `.GITCLAW/` directory, the guard checks that new issues contain a ❤️ heart emoji in their body. If the heart emoji is missing, the guard exits non-zero and prints:

> GitClaw heart guard — issue does not contain a ❤️ heart emoji. Skipping.

This check only applies to `issues.opened` events. Comments on existing issues are always processed regardless of this setting.

## How to enable

```bash
cd .GITCLAW
mv GITCLAW-HEART-NOT-REQUIRED.md GITCLAW-HEART-REQUIRED.md
git add -A
git commit -m "Require heart emoji for new issues"
git push
```

## How to disable

```bash
cd .GITCLAW
mv GITCLAW-HEART-REQUIRED.md GITCLAW-HEART-NOT-REQUIRED.md
git add -A
git commit -m "Remove heart emoji requirement for new issues"
git push
```
