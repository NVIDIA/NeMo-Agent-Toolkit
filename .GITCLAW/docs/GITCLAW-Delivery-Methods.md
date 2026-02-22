# .GITCLAW 🦞 Delivery Methods

> How gitclaw reaches repositories — current approach, alternatives, and everything in between.

gitclaw is a self-contained `.GITCLAW` folder that turns any GitHub repo into an AI-powered assistant. The delivery question is: **how does that folder get into a repo in the first place?**

This document maps out every viable channel — from what exists today to ideas that could reshape how gitclaw is distributed.

---

## Table of Contents

1. [Current Method — Fork / Import + Installer](#1-current-method--fork--import--installer)
2. [GitHub Application (OAuth / GitHub App)](#2-github-application-oauth--github-app)
3. [GitHub Marketplace Action](#3-github-marketplace-action)
4. [GitHub Template Repository](#4-github-template-repository)
5. [CLI Tool (npx / bunx)](#5-cli-tool-npx--bunx)
6. [GitHub Pages — Self-Service Portal](#6-github-pages--self-service-portal)
7. [Third-Party Website](#7-third-party-website)
8. [Email-Based Delivery](#8-email-based-delivery)
9. [GitHub Repository Dispatch / API-Driven](#9-github-repository-dispatch--api-driven)
10. [Browser Extension](#10-browser-extension)
11. [Git Submodule / Subtree](#11-git-submodule--subtree)
12. [Package Registry (npm / GitHub Packages)](#12-package-registry-npm--github-packages)
13. [GitHub Codespaces / Dev Container](#13-github-codespaces--dev-container)
14. [Probot / Webhook Service](#14-probot--webhook-service)
15. [Comparison Matrix](#15-comparison-matrix)
16. [Recommendations](#16-recommendations)

---

## 1. Current Method — Fork / Import + Installer

| Step | What happens |
|------|-------------|
| Copy `.GITCLAW/` folder into your repo | Manual or via fork/import |
| Run `bun .GITCLAW/install/GITCLAW-INSTALLER.ts` | Copies workflows + templates into `.github/` |
| Push changes | Bootstrap workflow picks up and creates a PR |
| Merge PR, add API key secret | gitclaw is live |

**Strengths:** Zero external dependencies. Everything stays in the repo. Full control.

**Limitations:** Requires manual folder copy. Users need bun or node installed locally. Multi-step process with room for error. Updates require manual re-copy.

---

## 2. GitHub Application (OAuth / GitHub App)

> A registered GitHub App that installs gitclaw into any repo with a single click.

### How it works

1. User visits the GitHub App installation page.
2. Selects which repositories to enable.
3. The app uses the GitHub API to:
   - Create a branch in the target repo.
   - Commit the `.GITCLAW/` folder contents.
   - Open a bootstrap PR with setup instructions.
   - Optionally create the API key secret (if granted permission).

### Why this is compelling

- **One-click install** — no CLI, no local setup, no fork.
- **Automatic updates** — the app can open PRs when gitclaw has new versions.
- **Centralized management** — install/uninstall across multiple repos from one dashboard.
- **Permission model** — GitHub's native app permissions handle authorization cleanly.

### Considerations

- Requires hosting a small backend (could be serverless — Cloudflare Workers, Vercel, AWS Lambda).
- Needs careful scoping of permissions (contents write, workflows write, secrets write).
- App review process if published to the GitHub Marketplace.

---

## 3. GitHub Marketplace Action

> A published GitHub Action that bootstraps gitclaw when added to any workflow.

### How it works

```yaml
# .github/workflows/install-gitclaw.yml
name: Install GitClaw
on: workflow_dispatch
jobs:
  install:
    runs-on: ubuntu-latest
    steps:
      - uses: japer-technology/gitclaw-action@v1
```

The action:
1. Downloads the latest `.GITCLAW/` release artifact.
2. Commits it into the repo on a new branch.
3. Opens a PR with the bootstrap changes.

### Why this is compelling

- Users only need to add **one workflow file** — the action handles everything.
- Native to the GitHub ecosystem. Discoverable via the Marketplace.
- Version pinning with tags (`@v1`, `@v2`).
- Updates can be as simple as bumping the action version.

### Considerations

- Requires packaging gitclaw as a release artifact or embedding it in the action.
- The "add one YAML file" step still needs to happen somehow (documentation, or paired with a template repo).

---

## 4. GitHub Template Repository

> Mark the gitclaw repo (or a dedicated starter repo) as a **template**.

### How it works

1. User clicks **"Use this template"** → **"Create a new repository"** on GitHub.
2. Gets a fresh repo with `.GITCLAW/` pre-installed and workflows ready.
3. Adds their API key secret and opens their first issue.

### Why this is compelling

- **Zero CLI required** — entirely browser-based.
- GitHub-native UX that developers already understand.
- The new repo starts with a clean commit history (no fork baggage).
- Could include a starter `README.md`, example issues, and pre-configured settings.

### Considerations

- Only works for **new** repos. Doesn't solve "add gitclaw to my existing project."
- The template repo needs to stay updated with the latest gitclaw version.
- Could be combined with a Marketplace Action for the "add to existing repo" path.

---

## 5. CLI Tool (npx / bunx)

> A single command that installs gitclaw into the current directory.

### How it works

```bash
npx gitclaw init
# or
bunx gitclaw init
```

The CLI:
1. Downloads the latest `.GITCLAW/` folder from the release or repo.
2. Copies it into the working directory.
3. Runs the installer to set up workflows and templates.
4. Prompts for configuration (provider, model, etc.).

### Why this is compelling

- **One command** — familiar pattern for developers (`npx create-react-app`, `npx astro init`).
- No need to fork, clone, or manually copy anything.
- Could support flags: `--provider anthropic --model claude-sonnet-4-20250514 --thinking high`.
- Could include an interactive setup wizard.
- Easy to version and distribute via npm.

### Considerations

- Requires node/bun on the user's machine (but so does the current method).
- Needs a published npm package or a download mechanism.

---

## 6. GitHub Pages — Self-Service Portal

> A website hosted on GitHub Pages that generates and delivers gitclaw configurations.

### How it works

1. User visits `japer-technology.github.io/gitclaw`.
2. Fills out a form: target repo, LLM provider, model, agent personality.
3. The site either:
   - **Generates a downloadable ZIP** of a pre-configured `.GITCLAW/` folder.
   - **Uses the GitHub API** (via OAuth) to directly commit to the user's repo and open a PR.
   - **Generates a one-line install command** customized with their choices.

### Why this is compelling

- **Visual, guided experience** — no terminal needed for the initial decision-making.
- Configuration happens before installation, reducing post-install setup.
- Could include live previews of the agent personality and workflow configuration.
- Free hosting via GitHub Pages.

### Considerations

- Direct repo integration requires OAuth and a backend for token exchange.
- The ZIP download path is simple but still requires manual steps.
- The generated install command path combines well with the CLI approach.

---

## 7. Third-Party Website

> A dedicated website (e.g., `gitclaw.dev`) that serves as a full-featured installation hub.

### How it works

1. User signs in with GitHub OAuth.
2. Browses their repos and selects one.
3. Configures gitclaw settings through a UI.
4. Clicks "Install" — the site commits `.GITCLAW/` via the GitHub API and opens a PR.
5. Dashboard shows installed repos, agent activity, and update availability.

### Why this is compelling

- **Richest user experience** — onboarding, configuration, monitoring all in one place.
- Could aggregate analytics across repos (usage stats, conversation counts, cost tracking).
- Natural home for documentation, tutorials, and a community showcase.
- Could offer a "gallery" of agent personalities and skill packages.

### Considerations

- Requires hosting and maintenance (though serverless keeps costs near zero).
- Adds an external dependency to what is currently a self-contained system.
- OAuth token management and security considerations.
- Domain registration and upkeep.

---

## 8. Email-Based Delivery

> Send a setup email containing everything needed to get started.

### How it works

**Option A — Attachment delivery:**
1. User requests gitclaw via a form or mailing list.
2. Receives an email with a ZIP of `.GITCLAW/` attached.
3. Extracts, copies into their repo, and follows the included README.

**Option B — Magic link delivery:**
1. User provides their GitHub username and target repo.
2. Receives an email with a unique link.
3. Clicking the link triggers a GitHub App or API call that opens a bootstrap PR on their repo.

**Option C — Invite-based delivery:**
1. User receives a GitHub repository invitation.
2. Accepting the invite gives them access to a private repo with setup automation.
3. A workflow in that repo pushes `.GITCLAW/` to their target repo.

### Why this is compelling

- Email is universal — reaches developers who may not be browsing GitHub Marketplace.
- Magic links feel effortless.
- Invite-based delivery leverages GitHub's own collaboration model.
- Good for onboarding non-technical users who prefer guided experiences.

### Considerations

- Attachment delivery is clunky and prone to being caught by spam filters.
- Magic links require a backend to process.
- Email is a slower, less developer-native channel.

---

## 9. GitHub Repository Dispatch / API-Driven

> Trigger gitclaw installation remotely via the GitHub API.

### How it works

1. A "mothership" repo contains a workflow that accepts `repository_dispatch` events.
2. User sends an API call (or uses a simple UI) specifying their target repo.
3. The workflow:
   - Forks or creates a branch in the target repo.
   - Commits `.GITCLAW/`.
   - Opens a bootstrap PR.

```bash
curl -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/japer-technology/gitclaw/dispatches \
  -d '{"event_type":"install","client_payload":{"target_repo":"user/my-repo"}}'
```

### Why this is compelling

- **Fully automated** — no manual file copying.
- Can be triggered from any system that can make HTTP requests (CI/CD, Slack bots, other workflows).
- The "mothership" pattern centralizes installation logic and version management.
- Composable — could be triggered by a GitHub Pages form, a CLI tool, or a chatbot.

### Considerations

- Requires the user to grant a personal access token with repo write permissions.
- The dispatching repo needs appropriate secrets configured.
- More complex to set up initially, but powerful at scale.

---

## 10. Browser Extension

> A Chrome/Firefox extension that adds an "Install GitClaw" button to GitHub repo pages.

### How it works

1. User installs the browser extension.
2. When viewing any GitHub repo, a button appears: **"🦞 Add GitClaw"**.
3. Clicking it authenticates via GitHub OAuth and opens a PR with `.GITCLAW/` on that repo.

### Why this is compelling

- **Contextual** — appears exactly where the user is already working.
- One-click from any repo page.
- Could show gitclaw status on repos where it's already installed.

### Considerations

- Browser extensions have a small audience and require maintenance across browsers.
- Extension store review processes add friction.
- OAuth token management in extensions requires careful security.

---

## 11. Git Submodule / Subtree

> Include gitclaw as a git submodule or subtree rather than copying files.

### How it works

```bash
# Submodule
git submodule add https://github.com/japer-technology/gitclaw.git .GITCLAW

# Subtree
git subtree add --prefix .GITCLAW https://github.com/japer-technology/gitclaw.git main --squash
```

Then run the installer as usual.

### Why this is compelling

- **Updates are trivial** — `git submodule update --remote` or `git subtree pull`.
- Clear separation between gitclaw code and the host repo.
- Version pinning to specific commits or tags.
- Familiar pattern for developers who use submodules.

### Considerations

- Submodules add complexity to `git clone` (need `--recurse-submodules`).
- GitHub Actions `checkout` needs `submodules: true` configuration.
- Subtree merges can create noisy commit history.
- The installer workflow needs to reference the submodule path correctly.

---

## 12. Package Registry (npm / GitHub Packages)

> Publish `.GITCLAW/` as a versioned package.

### How it works

```bash
npm install --save-dev @japer-technology/gitclaw
npx gitclaw init
```

Or via GitHub Packages:

```bash
npm install --save-dev @japer-technology/gitclaw --registry=https://npm.pkg.github.com
```

The package's `postinstall` script (or explicit `init` command) copies `.GITCLAW/` into the repo root and runs the installer.

### Why this is compelling

- **Versioned delivery** via a mature package ecosystem.
- Dependency management tools handle updates (`npm update`, Dependabot, Renovate).
- Could be listed as a `devDependency` — making gitclaw part of the project manifest.
- GitHub Packages keeps everything within the GitHub ecosystem.

### Considerations

- Adds a `package.json` dependency to repos that may not be JavaScript projects.
- The "copy folder" postinstall pattern is unconventional for npm packages.
- Needs thoughtful handling of non-JS repos.

---

## 13. GitHub Codespaces / Dev Container

> Pre-configure gitclaw in a dev container definition.

### How it works

1. A `.devcontainer/devcontainer.json` is published as a template or feature.
2. When a user opens a Codespace, gitclaw is automatically installed.
3. Could include a post-create command that runs the installer.

```json
{
  "name": "GitClaw-enabled",
  "features": {
    "ghcr.io/japer-technology/gitclaw:1": {}
  },
  "postCreateCommand": "npx gitclaw init"
}
```

### Why this is compelling

- **Zero local setup** — everything happens in the cloud.
- Pairs naturally with GitHub's Codespaces push.
- Could be a "Dev Container Feature" — installable in any Codespace with one line.
- Great for workshops, demos, and onboarding.

### Considerations

- Only useful for Codespaces users (growing but not universal).
- Dev Container Features are a relatively new spec.
- The agent still runs via GitHub Actions, not inside the Codespace itself.

---

## 14. Probot / Webhook Service

> A hosted webhook service that responds to GitHub events to install and manage gitclaw.

### How it works

1. User installs a lightweight Probot app.
2. When a specific event occurs (e.g., a label `gitclaw:install` is added to the repo, or a special issue is opened), the service:
   - Commits `.GITCLAW/` to a new branch.
   - Opens a bootstrap PR.
   - Responds with instructions in the issue/PR.

### Why this is compelling

- **Event-driven** — installs happen in response to natural GitHub actions.
- Could handle updates the same way (label `gitclaw:update` triggers a version bump PR).
- Probot is a well-established GitHub automation framework.

### Considerations

- Requires hosting (Probot needs a server, though it can run on serverless platforms).
- Adds operational complexity compared to the current zero-infrastructure model.

---

## 15. Comparison Matrix

| Method | New Repo | Existing Repo | No CLI | No Backend | Auto-Updates | One-Click | GitHub-Native |
|--------|:--------:|:-------------:|:------:|:----------:|:------------:|:---------:|:-------------:|
| Current (Fork + Installer) | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |
| GitHub App | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| Marketplace Action | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Template Repo | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| CLI (npx) | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| GitHub Pages Portal | ✅ | ✅ | ✅ | ⚠️ | ❌ | ✅ | ✅ |
| Third-Party Website | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| Email | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Repository Dispatch | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Browser Extension | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| Submodule / Subtree | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ |
| Package Registry | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ⚠️ |
| Codespaces / Dev Container | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Probot / Webhook | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |

> ⚠️ = Partial. GitHub Pages needs a backend for direct repo integration. GitHub Packages is GitHub-native but npm-specific.

---

## 16. Recommendations

### Quick wins (low effort, high impact)

1. **Template Repository** — Mark the gitclaw repo as a template. Zero development cost, instant "Use this template" button for new repos. Pairs with the current installer for existing repos.

2. **CLI tool (`npx gitclaw init`)** — Wrap the current installer in an npm package. The logic already exists in `GITCLAW-INSTALLER.ts`; this just wraps it with a download step. Familiar developer pattern.

3. **Marketplace Action** — Publish the installer workflow as a standalone GitHub Action. Users add one YAML file, the action does the rest.

### Medium-term (moderate effort, transformative)

4. **GitHub App** — The most user-friendly path. One-click install, centralized updates, native permissions. Needs a small backend but unlocks a fundamentally better onboarding experience.

5. **GitHub Pages Portal** — A visual configuration + install experience. Could generate custom CLI commands or ZIP downloads without needing a separate backend.

### Long-term (higher effort, ecosystem play)

6. **Third-Party Website + Dashboard** — Full management plane for gitclaw across repos. Install, configure, monitor, update — all from one place. Makes sense once gitclaw has significant adoption.

### The compound approach

These methods are not mutually exclusive. The most effective strategy is **layered delivery**:

- **Template Repo** for new projects (today).
- **CLI tool** for developers adding to existing repos (this week).
- **Marketplace Action** for CI-native teams (this month).
- **GitHub App** for the widest possible audience (this quarter).

Each layer catches a different segment of users. The `.GITCLAW/` folder remains the universal constant — every delivery method ultimately puts that folder into a repo. The only question is how it gets there.

---

> 🦞 *gitclaw is the folder. The delivery method is just the door.*
