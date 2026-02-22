# .GITCLAW 🦞 Enabled

### Delete or rename this file to disable .GITCLAW

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

## File existence behavior

All `GITCLAW-*` workflows run `.GITCLAW/lifecycle/GITCLAW-ENABLED.ts` as the first blocking guard step. If this file is missing, the guard exits non-zero and prints:

> GitClaw disabled by missing GITCLAW-ENABLED.md

That fail-closed guard blocks all subsequent GITCLAW workflow logic.
