# .GITCLAW 🦞 Since Forking

### How japer-technology/gitclaw diverged from SawyerHood/gitclaw

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

> This report compares the current state of `japer-technology/gitclaw` against its
> upstream fork origin `SawyerHood/gitclaw`, documenting every structural change,
> addition, and removal since the fork was created.

---

## At a Glance

| Metric | Value |
|---|---|
| **Commits ahead** | **285** |
| **Commits behind** | **0** (fully up to date with upstream) |
| **Files changed** | 85 |
| **Lines added** | +8,372 |
| **Lines removed** | -322 |
| **Merge base** | `c544212` (upstream's latest: "Tighten up README") |

---

## 🏗️ Original Upstream (SawyerHood/gitclaw)

A minimal **22-file**, **3-commit** repo:

- `.pi/` — Pi agent config (settings, skills, bootstrap)
- `lifecycle/` — `main.ts` and `preinstall.ts`
- `.github/workflows/GITCLAW-WORKFLOW-AGENT.yml` — single workflow
- `AGENTS.md`, `README.md`, `LICENSE`, `banner.jpeg`
- `package.json` + `bun.lock`

Only 3 commits: "Initial commit", "Update acknowledgments", "Tighten up README".

---

## 🚀 Fork Changes (japer-technology/gitclaw)

### 1. Complete Restructure into `.GITCLAW/` Drop In Folder

The original `lifecycle/`, `.pi/`, `AGENTS.md`, `LICENSE`, and `banner.jpeg` were moved and reorganized into a self-contained `.GITCLAW/` directory:

- **`.GITCLAW/lifecycle/`** — Renamed: `GITCLAW-AGENT.ts`, `GITCLAW-ENABLED.ts`, `GITCLAW-INDICATOR.ts`
- **`.GITCLAW/install/`** — New installer: `GITCLAW-INSTALLER.ts`, `.gitattributes`, `.gitignore`, `GITCLAW-WORKFLOW-AGENT.yml`
- **`.GITCLAW/.pi/`** — Pi config preserved inside `.GITCLAW`
- **`.GITCLAW/build/`** — Build directory with README
- **`.GITCLAW/package.json`** — Dedicated package.json

### 2. Extensive Documentation (15+ new docs)

- `GITCLAW-QUICKSTART.md`, `GITCLAW-ENABLED.md`, `GITCLAW-NOT-INSTALLED.md`
- `docs/GITCLAW-The-Idea.md`, `GITCLAW-Roadmap.md`, `GITCLAW-Possibilities.md`
- `docs/GITCLAW-Interactive-Possibilities.md`, `GITCLAW-The-GitHub-Possibilities.md`
- `docs/GITCLAW-Communication-Channels.md`, `GITCLAW-Internal-Mechanics.md`, `GITCLAW-Loves-Pi.md`
- Pi-specific: `GITCLAW-Pi-Architecture.md`, `GITCLAW-Pi-Capabilities.md`, `GITCLAW-Pi-Configuration.md`, `GITCLAW-Pi-Personality.md`, `GITCLAW-Pi-Skills.md`

### 3. New GitHub Workflows

- **`GITCLAW-INSTALLER.yml`** — Auto-install workflow (279 lines) triggered on push/workflow_run
- **`GITCLAW-WORKFLOW-AGENT.yml`** — Replacement for the original `agent.yml`
- Issue templates: `GITCLAW-TEMPLATE-HATCH.md`, `GITCLAW-ISSUE-TEMPLATE-README.md`
- PR template: `GITCLAW-PR-TEMPLATE-README.md`

### 4. State Management & Session Tracking

- `.GITCLAW/state/issues/` — JSON state files for 11+ tracked issues
- `.GITCLAW/state/sessions/` — 11 JSONL session logs
- `.GITCLAW/state/user.md` — User state file

### 5. Testing

- `.GITCLAW/tests/phase0.test.js` — 369-line test suite for Phase 0 validation

### 6. Files Removed from Upstream

| Upstream File | What Happened |
|---|---|
| `LICENSE` | Moved to `.GITCLAW/LICENSE.md` |
| `banner.jpeg` | Replaced with `.GITCLAW/GITCLAW-LOGO.png` |
| `lifecycle/main.ts` | Refactored into `.GITCLAW/lifecycle/GITCLAW-AGENT.ts` |
| `lifecycle/preinstall.ts` | Refactored into `.GITCLAW/install/GITCLAW-INSTALLER.ts` |
| `.github/workflows/agent.yml` | Replaced by `GITCLAW-WORKFLOW-AGENT.yml` |

### 7. README.md

Significantly rewritten with new branding, emoji, and documentation of the `.GITCLAW` drop in approach.

---

## 🔑 Key Takeaway

The fork transformed the original minimal 3-commit Pi agent scaffold into a **full-featured, self-contained `.GITCLAW/` drop in system** with its own installer workflow, extensive documentation, lifecycle management, state tracking, issue automation, and tests — all while keeping 0 commits behind upstream.

---

_Generated on February 20, 2026 by fetching `upstream/main` from `SawyerHood/gitclaw` and diffing against `HEAD` of `japer-technology/gitclaw`._
