# .GITCLAW 🦞 The GitHub Possibilities

### A comprehensive analysis of every GitHub platform feature and what it means for a repo-native AI agent

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

> GitHub is not a single product. It is a constellation of features — issue trackers, code
> review surfaces, automation engines, security scanners, project boards, package registries,
> wikis, discussions, releases, environments, and more — unified by a single identity and
> permission model. GitClaw's central thesis is that an AI agent can participate in **all**
> of these surfaces using nothing but the repository itself as its runtime.
>
> This document maps that constellation. For every major GitHub feature, it asks three
> questions: **What is it?** **How does GitClaw relate to it today?** **What could GitClaw
> do with it tomorrow?**

---

## Table of Contents

1. [Core Platform: Git & Repositories](#1-core-platform-git--repositories)
2. [Issues](#2-issues)
3. [Pull Requests & Code Review](#3-pull-requests--code-review)
4. [GitHub Actions](#4-github-actions)
5. [Checks & Status API](#5-checks--status-api)
6. [Labels, Milestones & Projects](#6-labels-milestones--projects)
7. [Discussions](#7-discussions)
8. [Releases & Tags](#8-releases--tags)
9. [Security: Code Scanning, Dependabot & Secret Scanning](#9-security-code-scanning-dependabot--secret-scanning)
10. [Branch Protection & Rulesets](#10-branch-protection--rulesets)
11. [CODEOWNERS](#11-codeowners)
12. [GitHub Pages](#12-github-pages)
13. [Wikis](#13-wikis)
14. [Packages & Container Registry](#14-packages--container-registry)
15. [Environments & Deployments](#15-environments--deployments)
16. [GitHub Apps & OAuth](#16-github-apps--oauth)
17. [Webhooks & Repository Dispatch](#17-webhooks--repository-dispatch)
18. [OIDC & Federated Credentials](#18-oidc--federated-credentials)
19. [GitHub API (REST & GraphQL)](#19-github-api-rest--graphql)
20. [Notifications & Subscriptions](#20-notifications--subscriptions)
21. [Forks, Stars & Social Graph](#21-forks-stars--social-graph)
22. [GitHub Copilot & AI Ecosystem](#22-github-copilot--ai-ecosystem)
23. [Cross-Cutting Analysis](#23-cross-cutting-analysis)
24. [The Combinatorial Explosion](#24-the-combinatorial-explosion)
25. [Conclusion](#25-conclusion)

---

## 1. Core Platform: Git & Repositories

### What GitHub Provides

At its foundation, GitHub is a hosted Git service. Every repository is a full version-control system with branches, tags, commits, diffs, blame, and a complete history graph. GitHub layers a web UI, an API, and access controls on top of that foundation.

Key primitives:

| Primitive | Description |
|-----------|-------------|
| **Commits** | Immutable snapshots of the entire tree, with author, timestamp, and message |
| **Branches** | Lightweight pointers to commits — parallel lines of development |
| **Tags** | Named pointers to specific commits — release markers |
| **Diffs** | Computed differences between any two commits |
| **Blame** | Per-line attribution of authorship across time |
| **Refs** | The namespace system (`refs/heads/`, `refs/tags/`, `refs/pull/`) that ties everything together |

### How GitClaw Uses It Today

GitClaw's entire state model is built on git:

- **Sessions are JSONL files** committed to `state/sessions/`. Every conversation turn is a commit.
- **Issue mappings are JSON files** in `state/issues/`. Each issue number points to its session.
- **The agent's memory is `git log`** — diffable, blameable, revertable.
- **Conflict resolution is `git pull --rebase`** with a retry loop, handling concurrent writes gracefully.

The repository is not just where the agent's code lives — it is the agent's database, its filesystem, and its synchronization medium. Forking the repo forks the agent's mind. Branching branches its reality.

### What Could Be Done

| Possibility | Description |
|-------------|-------------|
| **Branch-per-investigation** | The agent creates a feature branch for exploratory work, keeping `main` clean until a human merges |
| **Commit-level audit trails** | Structured commit messages with metadata (token usage, model, skill invoked) for forensic analysis |
| **Blame-driven context** | When reviewing code, the agent uses `git blame` to understand who wrote what and when, enriching its analysis |
| **Diff-based change summaries** | Automatically summarize what changed between any two points using `git diff` — useful for release notes and review summaries |
| **Subtree-scoped agents** | Multiple agents operating on different directory subtrees of the same repo, with git as the coordination layer |

> **Core insight:** Git is not just GitClaw's storage backend — it is an active participant in the agent's reasoning. Every git primitive (blame, log, diff, branch) is a potential tool in the agent's toolkit.

---

## 2. Issues

### What GitHub Provides

Issues are GitHub's general-purpose work-tracking system. Each issue has a number, title, body, labels, assignees, milestones, projects, comments, reactions, and a timeline of events. Issues support Markdown, task lists, file attachments, and cross-references.

### How GitClaw Uses It Today

Issues are GitClaw's **primary conversation interface**. This is the surface where human and agent meet:

1. **Opening an issue** starts a new conversation. The title and body become the first prompt.
2. **Commenting on an issue** continues the conversation. The agent resumes from its persisted session.
3. **The 👀 reaction** signals that the agent is working.
4. **The agent's response** appears as a comment on the issue thread.

The mapping `issue #N → state/issues/N.json → state/sessions/<session>.jsonl` is the trick that makes everything work.

### What Could Be Done

| Possibility | Description |
|-------------|-------------|
| **Issue templates as skill selectors** | Different issue templates invoke different agent skills — "Bug Report" triggers triage, "Feature Request" triggers design analysis, "Code Review" triggers structured review |
| **Task list automation** | The agent reads Markdown task lists in issue bodies and posts progress updates as items are completed |
| **Sub-issue decomposition** | The agent breaks a large request into sub-issues, creating a hierarchy of tracked work items |
| **Issue search as memory recall** | Instead of only reading its own session files, the agent searches closed issues to find prior decisions, rejected approaches, and resolved questions |
| **Reaction-based feedback** | Users react to agent comments (👍 for good, 👎 for bad) and the agent uses this signal to calibrate future responses |
| **Auto-close stale conversations** | The agent identifies issues that have been idle and posts a summary-and-close comment |
| **Issue transfer** | When the agent determines an issue belongs in a different repo, it transfers it programmatically |
| **Pinned issue dashboards** | The agent maintains a pinned issue with a live dashboard — open investigations, recent decisions, memory highlights |

> **Design space:** Issues are the most natural GitHub surface for human-AI conversation. Every enhancement to issue-based interaction directly improves the core GitClaw experience.

---

## 3. Pull Requests & Code Review

### What GitHub Provides

Pull requests are GitHub's code-review and merge mechanism. A PR has:

- A **diff** between two branches
- A **conversation thread** (comments, reviews, review comments)
- **Review verdicts** (`APPROVE`, `REQUEST_CHANGES`, `COMMENT`)
- **Line-level annotations** tied to specific file locations in the diff
- **Check suites and status checks** that gate merging
- **Merge methods** (merge commit, squash, rebase)
- **Draft PRs** for work-in-progress signaling
- **Auto-merge** that merges once all requirements are met

### How GitClaw Uses It Today

GitClaw does not currently trigger on pull request events. It operates exclusively through issues.

### What Could Be Done

This is one of the richest expansion surfaces:

| Possibility | Description |
|-------------|-------------|
| **Auto-review on PR open** | When a PR is opened, the agent reads the diff, analyzes it, and posts a structured review with line-level comments |
| **Review verdict logic** | The agent posts `APPROVE` for clean changes, `REQUEST_CHANGES` for issues found, with a consistent rubric |
| **Diff summarization** | Large PRs get a top-level summary — what changed, why it matters, what to pay attention to |
| **Review thread participation** | The agent responds to unresolved review threads, offering clarification or suggesting fixes |
| **PR-to-session mapping** | Mirror the issue-session model: `state/pull-requests/<pr>.json → state/sessions/<session>.jsonl` |
| **Draft PR creation** | The agent creates draft PRs from issue discussions — "implement what we discussed in #42" |
| **Merge conflict resolution** | The agent detects merge conflicts, analyzes both sides, and proposes a resolution |
| **PR description generation** | From the diff alone, generate a structured PR description with summary, motivation, and testing notes |
| **Suggested changes** | Use GitHub's "suggested change" syntax in review comments so users can apply fixes with one click |

> **Key API surfaces:** `pull_request` events, Pull Request Reviews API, Pull Request Review Comments API, and the `pull_request_review_comment.created` webhook.

---

## 4. GitHub Actions

### What GitHub Provides

GitHub Actions is a CI/CD and workflow automation platform built into every repository. Key capabilities:

| Capability | Description |
|------------|-------------|
| **Event triggers** | 35+ event types: push, PR, issue, schedule, dispatch, deployment, release, and more |
| **Workflow files** | YAML definitions in `.github/workflows/` |
| **Runner environments** | Ubuntu, macOS, Windows, self-hosted, and larger runners |
| **Secrets & variables** | Encrypted secrets and plaintext variables scoped to repo, environment, or org |
| **Caching** | Dependency and build caching via `actions/cache` |
| **Artifacts** | Upload/download files between jobs and workflow runs |
| **Matrix builds** | Parallel execution across multiple configurations |
| **Reusable workflows** | Call workflows from other workflows, enabling composition |
| **OIDC tokens** | Short-lived credentials for cloud authentication |
| **Concurrency controls** | Group and cancel in-progress runs to prevent conflicts |

### How GitClaw Uses It Today

Actions is GitClaw's **compute engine**. The entire agent lifecycle runs inside a single workflow:

```
issues.opened / issue_comment.created
  → Authorize (permission check)
  → Checkout (fetch repo)
  → Setup Bun (install runtime)
  → Guard (verify sentinel file)
  → Preinstall (add 👀 reaction)
  → Install dependencies (bun install)
  → Run agent (bun lifecycle/GITCLAW-AGENT.ts)
```

The workflow is defined in `.github/workflows/GITCLAW-WORKFLOW-AGENT.yml` and uses `ubuntu-latest` runners.

### What Could Be Done

| Possibility | Description |
|-------------|-------------|
| **Multi-event workflows** | Extend triggers to `pull_request`, `discussion`, `schedule`, `workflow_dispatch`, and `repository_dispatch` |
| **Scheduled agent tasks** | Cron-triggered runs for periodic triage, dependency audits, documentation freshness checks, and health reports |
| **Workflow dispatch with inputs** | Manual triggers with parameters — task type, target issue, skill override, model selection |
| **Matrix-based multi-agent** | Use matrix strategies to run multiple agent personas in parallel on different issues |
| **Caching for faster starts** | Cache `node_modules` / `bun` dependencies between runs to reduce cold-start time |
| **Artifact-based reporting** | Upload agent reports (review summaries, audit results, usage metrics) as workflow artifacts |
| **Reusable workflow distribution** | Package the agent workflow as a reusable workflow that other repos call with `uses:` |
| **Concurrency groups** | Prevent multiple agent runs on the same issue from racing with `concurrency:` groups |
| **Self-hosted runners** | Run on dedicated hardware for faster execution, GPU access, or air-gapped environments |
| **Composite actions** | Package the guard + indicator + agent sequence as a composite action for cleaner workflow files |

> **Untapped triggers:** `schedule`, `workflow_dispatch`, `repository_dispatch`, `release`, `deployment_status`, `check_suite`, `project`, `star`, and `fork` events are all available but unused today.

---

## 5. Checks & Status API

### What GitHub Provides

GitHub offers two complementary systems for signaling code quality:

**Check Runs** (Checks API):
- Rich output with markdown summaries and inline annotations
- File-level and line-level annotations with severity levels
- `success`, `failure`, `neutral`, `action_required`, `skipped` conclusions
- Progress updates (in-progress → completed)
- Re-run capability

**Commit Statuses** (Status API):
- Simple `pending`, `success`, `error`, `failure` states
- Context strings for identification (e.g., `gitclaw/review`)
- Description and target URL

### How GitClaw Uses It Today

GitClaw does not currently post check runs or commit statuses.

### What Could Be Done

| Possibility | Description |
|-------------|-------------|
| **Agent-posted check runs** | After reviewing a PR, post a Check Run with inline annotations on problem lines |
| **Required status checks** | Register `gitclaw/review` as a required check, gating merges on agent approval |
| **Multi-check reporting** | Separate checks for different concerns: `gitclaw/style`, `gitclaw/security`, `gitclaw/tests` |
| **Progress indication** | Use in-progress check runs to show the agent is working (richer than 👀 reactions) |
| **Annotation-based feedback** | File-level annotations appear directly in the PR diff view — no need to scroll through comments |
| **Action-required gates** | Post `action_required` conclusions when the agent finds issues that need human judgment |

> **Merge-gating power:** Check runs and status checks are the mechanism by which the agent can enforce quality standards. A required `gitclaw/review` check means no PR merges without agent approval.

---

## 6. Labels, Milestones & Projects

### What GitHub Provides

**Labels** — colored tags on issues and PRs for categorization:
- Arbitrary name/color/description
- Filterable in search and on boards
- Assignable via API

**Milestones** — time-boxed groupings of issues/PRs:
- Title, description, due date
- Open/closed issue counts
- Progress percentage

**Projects (v2)** — flexible project boards powered by GraphQL:
- Custom fields (text, number, date, single-select, iteration)
- Views (table, board, roadmap)
- Automated workflows (auto-add, auto-archive)
- Cross-repo item tracking

### How GitClaw Uses It Today

GitClaw does not currently interact with labels, milestones, or projects.

### What Could Be Done

| Feature | Possibility |
|---------|-------------|
| **Labels** | Auto-label issues based on content analysis (`bug`, `feature`, `question`, `documentation`); use labels to route to different agent skills (`agent:review`, `agent:triage`, `agent:docs`) |
| **Labels** | Enforce a label taxonomy — warn when issues lack required labels or have conflicting ones |
| **Labels** | Priority detection — analyze urgency signals in issue text and apply priority labels |
| **Milestones** | Auto-assign issues to milestones based on content and project phase |
| **Milestones** | Generate milestone progress reports — burn-down charts, scope creep warnings, overdue item alerts |
| **Projects** | Move items through project board columns based on agent analysis (triage → in-progress → review → done) |
| **Projects** | Set custom field values (effort estimates, priority scores) based on issue complexity analysis |
| **Projects** | Auto-add items to project boards when issues match configured criteria |
| **Projects** | Generate sprint summaries by reading project board state |

> **Organization multiplier:** Labels + Projects + agent intelligence = automated project management. The agent becomes a project coordinator, not just a code assistant.

---

## 7. Discussions

### What GitHub Provides

GitHub Discussions is a forum-style feature for long-form conversations:

- Categories (General, Ideas, Q&A, Show and Tell, Polls)
- Threaded replies
- Upvoting and marking answers
- Pinning and locking
- Labels and search
- Separate from issues — designed for open-ended conversation rather than tracked work

### How GitClaw Uses It Today

GitClaw does not currently interact with Discussions.

### What Could Be Done

| Possibility | Description |
|-------------|-------------|
| **Q&A responder** | In Q&A categories, the agent answers questions by searching the codebase and citing specific files and functions |
| **RFC summarizer** | For long design discussions, the agent posts periodic summaries: key arguments, open questions, emerging consensus |
| **Discussion-to-issue converter** | When a discussion reaches a decision, the agent creates a tracked issue with the agreed-upon requirements |
| **Knowledge base builder** | The agent monitors discussions and compiles recurring answers into a searchable FAQ or documentation page |
| **Poll analysis** | Analyze poll results and post structured summaries with visualizations |
| **Ideas triage** | In Ideas categories, the agent evaluates feasibility, estimates effort, and suggests related existing issues |

> **Cultural fit:** Discussions are where communities think out loud. An agent that can participate thoughtfully in discussions — summarizing, answering, connecting dots — becomes an invaluable community moderator.

---

## 8. Releases & Tags

### What GitHub Provides

**Releases** — packaged versions of the software:
- Tied to a git tag
- Release notes (Markdown)
- Binary assets (uploadable files)
- Pre-release and draft states
- Auto-generated release notes from merged PRs

**Tags** — named pointers to specific commits:
- Lightweight or annotated
- Often follow semantic versioning (v1.2.3)
- The foundation for release workflows

### How GitClaw Uses It Today

GitClaw does not currently interact with releases or tags.

### What Could Be Done

| Possibility | Description |
|-------------|-------------|
| **Changelog generation** | Read merged PRs and closed issues since the last tag; generate a structured changelog categorized by labels |
| **Release note drafting** | Create a draft GitHub Release with generated notes, letting humans review before publishing |
| **Semantic version detection** | Analyze the nature of changes (breaking, feature, fix) and recommend the next version number |
| **Tag creation** | After human approval, create an annotated tag and publish the release |
| **Asset coordination** | Trigger build workflows and attach binary artifacts to the release |
| **Release announcement** | Post release summaries to a pinned discussion or a dedicated issue |
| **Upgrade guides** | For breaking changes, generate migration instructions based on the diff between tagged versions |

> **End-to-end release pipeline:** The agent can orchestrate the full release lifecycle — from determining the version number to generating notes to publishing the release — all triggered by a single slash command or scheduled check.

---

## 9. Security: Code Scanning, Dependabot & Secret Scanning

### What GitHub Provides

GitHub offers three integrated security features:

**Code Scanning (CodeQL):**
- Static analysis that finds security vulnerabilities
- Alerts with severity, location, and remediation guidance
- Configurable queries and languages
- SARIF format for third-party tool integration

**Dependabot:**
- Alerts for known vulnerabilities in dependencies
- Automated version update PRs
- Configurable update schedule and grouping
- Compatibility scores for updates

**Secret Scanning:**
- Detects exposed credentials (API keys, tokens, passwords)
- Alerts on push
- Partner integrations for automatic revocation
- Custom patterns for organization-specific secrets

### How GitClaw Uses It Today

GitClaw does not currently interact with any security scanning features.

### What Could Be Done

| Feature | Possibility |
|---------|-------------|
| **CodeQL** | Triage alerts by severity — auto-close informational alerts, escalate critical ones to dedicated issues with fix proposals |
| **CodeQL** | Generate fix PRs for common vulnerability patterns (SQL injection, XSS, path traversal) |
| **CodeQL** | Track alert trends — detect regressions where new alerts appear after a merge |
| **Dependabot** | Auto-approve low-risk updates (patch version bumps, no breaking changes) |
| **Dependabot** | Summarize high-risk alerts with impact analysis — what code paths are affected, what's the blast radius |
| **Dependabot** | Group related updates and test them together |
| **Secret Scanning** | Immediately open a high-priority issue on secret detection with containment instructions |
| **Secret Scanning** | Post provider-specific rotation guidance (AWS key? Here's how to rotate it) |
| **Secret Scanning** | Audit access logs to assess exposure window |
| **All** | Weekly security digest — consolidated report of all open alerts across categories with trend analysis |

> **Security posture management:** The agent can serve as a security operations center (SOC) for the repository — triaging alerts, coordinating fixes, tracking trends, and ensuring nothing falls through the cracks.

---

## 10. Branch Protection & Rulesets

### What GitHub Provides

**Branch Protection Rules:**
- Require pull request reviews before merging
- Require status checks to pass
- Require signed commits
- Restrict who can push
- Enforce linear history
- Require deployments to succeed

**Repository Rulesets:**
- Organization-level policy enforcement
- Target branches, tags, or push events
- Stack multiple rules with bypass permissions
- More granular than branch protection

### How GitClaw Uses It Today

GitClaw pushes directly to `main` with a retry-on-conflict strategy. It operates within whatever branch protection rules are configured but does not inspect or enforce them.

### What Could Be Done

| Possibility | Description |
|-------------|-------------|
| **Protection audit** | The agent inspects branch protection settings and reports gaps (missing required reviews, no status checks, unsigned commits allowed) |
| **Rule recommendations** | Based on repository activity patterns, suggest optimal protection rules |
| **Compliance reporting** | Periodically verify that protection rules meet organizational standards |
| **Agent-as-required-reviewer** | Register the agent as a required reviewer for specific paths, providing consistent automated review |
| **Ruleset validation** | For organizations, audit cross-repo rulesets for consistency and coverage |
| **PR-based state changes** | Instead of pushing directly to `main`, the agent creates PRs that flow through protection rules — human-in-the-loop by default |

> **Governance alignment:** Branch protection and rulesets are GitHub's governance layer. An agent that understands and respects these rules — and can even help configure them — earns trust in enterprise environments.

---

## 11. CODEOWNERS

### What GitHub Provides

The `CODEOWNERS` file maps file paths to responsible individuals or teams. When a PR touches files matching a pattern, the listed owners are automatically requested as reviewers.

### How GitClaw Uses It Today

GitClaw does not interact with CODEOWNERS.

### What Could Be Done

| Possibility | Description |
|-------------|-------------|
| **CODEOWNERS generation** | Analyze `git blame` across the codebase and generate a CODEOWNERS file reflecting actual ownership patterns |
| **Stale owner detection** | Flag CODEOWNERS entries that reference users who have left the organization or haven't contributed recently |
| **Coverage gap analysis** | Identify directories or file types not covered by any CODEOWNERS entry |
| **Review routing** | When the agent identifies an issue related to specific code, it can tag the appropriate owners based on CODEOWNERS |
| **Ownership reports** | Generate ownership distribution reports — who owns how much, where are the bus-factor risks |

---

## 12. GitHub Pages

### What GitHub Provides

GitHub Pages is a static site hosting service built into every repository:

- Deploy from a branch (typically `main` or `gh-pages`) or GitHub Actions
- Custom domain support
- HTTPS by default
- Jekyll built-in, or deploy any static site generator via Actions

### How GitClaw Uses It Today

GitClaw does not currently use GitHub Pages.

### What Could Be Done

| Possibility | Description |
|-------------|-------------|
| **Documentation site** | The agent generates and maintains a documentation site from repo content, deployed via Pages |
| **Agent dashboard** | A live dashboard showing agent activity: recent conversations, open investigations, memory highlights, usage metrics |
| **Skill catalog** | A browsable catalog of available agent skills, generated from skill files in `.pi/skills/` |
| **Session viewer** | A web-based viewer for agent conversation sessions — richer than raw JSONL files |
| **Project status page** | Auto-generated project status page with open issues, PR state, release history, and security posture |
| **Interactive FAQ** | A searchable FAQ built from the agent's past Q&A interactions across issues and discussions |

> **Public presence:** Pages transforms the repository from a developer tool into a public-facing resource. An agent that can build and maintain its own website is remarkably self-sufficient.

---

## 13. Wikis

### What GitHub Provides

Each repository can have a wiki — a collection of Markdown pages with their own git repository under the hood:

- Sidebar navigation
- Page history and diffs
- Cloneable as a separate git repo
- Edit via web UI or git push

### How GitClaw Uses It Today

GitClaw does not interact with wikis.

### What Could Be Done

| Possibility | Description |
|-------------|-------------|
| **Auto-generated wiki** | The agent generates wiki pages from code analysis — architecture overviews, API references, configuration guides |
| **Wiki-as-knowledge-base** | The agent maintains a structured knowledge base in the wiki, updated as it learns from conversations |
| **Decision log** | Each significant decision discussed in issues is summarized and added to a wiki decision log |
| **Sync with docs** | Keep wiki content synchronized with in-repo documentation — changes in one propagate to the other |

---

## 14. Packages & Container Registry

### What GitHub Provides

**GitHub Packages** — a package hosting service for multiple ecosystems:
- npm, Maven, Gradle, NuGet, RubyGems, Docker containers
- Scoped to users or organizations
- Integrated with Actions for publishing
- Version management and access controls

**GitHub Container Registry (ghcr.io):**
- OCI-compliant container image hosting
- Public and private images
- Fine-grained permissions via GitHub Apps or PATs

### How GitClaw Uses It Today

GitClaw does not publish packages or container images.

### What Could Be Done

| Possibility | Description |
|-------------|-------------|
| **Skill packages** | Publish GitClaw skills as npm packages — installable via `bun add` or a future `gitclaw install` command |
| **Container-based execution** | Package the agent runtime as a container image for reproducible, isolated execution |
| **Package version monitoring** | Track published package versions and alert when new releases are available or vulnerabilities are disclosed |
| **Build artifact publishing** | When the agent generates reports or documentation, publish them as versioned packages |
| **Self-contained distribution** | Ship the entire `.GITCLAW` folder as an npm package or container image for easy adoption |

---

## 15. Environments & Deployments

### What GitHub Provides

**Environments** — named deployment targets with protection rules:
- Required reviewers before deployment
- Wait timers
- Deployment branches restrictions
- Environment-specific secrets and variables

**Deployments API:**
- Track deployment state (pending, in-progress, success, failure)
- Associate deployments with commits
- Deployment status history

### How GitClaw Uses It Today

GitClaw does not use environments or deployments.

### What Could Be Done

| Possibility | Description |
|-------------|-------------|
| **Staged agent rollout** | Use environments to gate agent capability upgrades — test new skills in `staging` before `production` |
| **Deployment tracking** | When the agent creates or approves a release, track the deployment through environments |
| **Environment-scoped secrets** | Different LLM API keys or model selections per environment (cheaper models for staging, best models for production) |
| **Approval-gated operations** | Destructive agent actions (file deletion, force push, release publishing) require environment approval |
| **Deployment status reporting** | Post deployment status updates that appear in the PR timeline |

---

## 16. GitHub Apps & OAuth

### What GitHub Provides

**GitHub Apps** — first-class integrations with fine-grained permissions:
- Installation-level access (per-repo or org-wide)
- Short-lived installation tokens
- Webhook delivery
- Dedicated bot identity (`app-name[bot]`)
- Granular permission scoping (read/write per resource type)
- Marketplace listing

**OAuth Apps** — user-level authentication for third-party applications.

### How GitClaw Uses It Today

GitClaw uses `GITHUB_TOKEN` (the default Actions token) and `github-actions[bot]` as its identity. There is no dedicated GitHub App.

### What Could Be Done

| Possibility | Description |
|-------------|-------------|
| **Dedicated identity** | A GitHub App gives the agent its own identity (`gitclaw[bot]`) — distinct from `github-actions[bot]` and easier to audit |
| **Fine-grained permissions** | App installation tokens can be scoped to exactly the permissions needed — no more, no less |
| **Webhook-based triggers** | Apps receive webhooks directly, enabling lower-latency responses than Action-based triggers |
| **Cross-repo installation** | An org-level App installation lets a single agent serve multiple repositories |
| **Marketplace distribution** | List GitClaw on the GitHub Marketplace for one-click installation |
| **User consent flows** | OAuth integration for features that need to act on behalf of a specific user |

> **Identity matters:** A dedicated GitHub App identity makes agent actions auditable, filterable, and distinguishable from CI. It is the professional-grade path for agent identity.

---

## 17. Webhooks & Repository Dispatch

### What GitHub Provides

**Webhooks** — HTTP callbacks fired when events occur:
- Configurable per-event delivery
- Payload includes full event context
- Delivery history and retry logic

**Repository Dispatch** — API-triggered workflow events:
- Custom `event_type` strings
- Arbitrary JSON `client_payload`
- Enables external systems to trigger workflows

### How GitClaw Uses It Today

GitClaw triggers only on `issues` and `issue_comment` events via Actions workflow definitions.

### What Could Be Done

| Possibility | Description |
|-------------|-------------|
| **External event ingestion** | CI failures, monitoring alerts, deployment events, or customer reports trigger the agent via `repository_dispatch` |
| **Cross-service integration** | Slack, Discord, Linear, Jira, or PagerDuty events dispatch agent tasks |
| **Typed event routing** | Define event types (`agent-task`, `security-alert`, `deploy-request`) with structured payloads that route to specific skills |
| **Webhook-based real-time response** | With a GitHub App, receive webhooks directly for faster response than Action polling |
| **Event batching** | Aggregate multiple events into a single agent invocation for efficiency |

---

## 18. OIDC & Federated Credentials

### What GitHub Provides

GitHub Actions can request **OIDC tokens** — short-lived JWT tokens that prove the workflow's identity to external services:

- No long-lived secrets needed
- Claims include repo, branch, workflow, actor
- Supported by AWS, GCP, Azure, HashiCorp Vault, and others
- Scoped to specific repos, branches, and environments

### How GitClaw Uses It Today

GitClaw uses a static `ANTHROPIC_API_KEY` secret for LLM access.

### What Could Be Done

| Possibility | Description |
|-------------|-------------|
| **Secretless LLM access** | If LLM providers adopt OIDC (or via a proxy), eliminate static API keys entirely |
| **Cloud resource access** | The agent accesses AWS, GCP, or Azure resources using short-lived OIDC tokens — no static credentials to rotate |
| **Credential audit** | The agent scans repo settings for long-lived secrets and recommends OIDC alternatives |
| **Scoped access** | OIDC claims restrict agent access to specific repos and branches — defense in depth |
| **Zero-trust agent** | Combined with environment protections and OIDC, the agent operates in a zero-trust model where every action is authenticated and scoped |

---

## 19. GitHub API (REST & GraphQL)

### What GitHub Provides

Two comprehensive API surfaces:

**REST API:**
- Full CRUD for issues, PRs, repos, releases, labels, milestones, and more
- Pagination, filtering, and sorting
- Rate limits (5,000 requests/hour for authenticated users)

**GraphQL API:**
- Single-endpoint querying with precise field selection
- Mutations for creating and updating resources
- Nested relationships in a single request
- Required for Projects v2 and some advanced features

### How GitClaw Uses It Today

GitClaw uses the `gh` CLI (which wraps the REST API) for:
- Posting comments on issues
- Adding/removing reactions
- Checking collaborator permissions

### What Could Be Done

| Possibility | Description |
|-------------|-------------|
| **Full platform participation** | Use the API to interact with every GitHub feature — PRs, projects, releases, security alerts, discussions |
| **Batch operations** | GraphQL enables fetching issue + labels + milestone + project status in a single request, reducing latency |
| **Cross-repo queries** | Query multiple repositories in an organization for consolidated views |
| **Webhook subscription management** | Programmatically configure which events the agent listens to |
| **Rate-limit-aware scheduling** | Monitor rate limit headers and schedule agent tasks to avoid throttling |
| **API as tool** | Expose the GitHub API as an agent tool — let the LLM decide which API calls to make based on the user's request |

> **The API is the control surface.** Every possibility in this document ultimately maps to one or more API calls. The API is what turns conversation into action.

---

## 20. Notifications & Subscriptions

### What GitHub Provides

GitHub's notification system alerts users to activity:
- Subscriptions (watching repos, issues, PRs)
- @mentions and team mentions
- Review requests
- CI/CD status
- Notification filtering and custom routing

### How GitClaw Uses It Today

GitClaw does not interact with notifications. Humans see agent responses via normal GitHub notification flows.

### What Could Be Done

| Possibility | Description |
|-------------|-------------|
| **Smart notification digest** | The agent generates a daily/weekly digest of important activity — summarizing instead of flooding |
| **@mention handling** | The agent responds when @mentioned in any context (issue, PR, discussion, commit comment) |
| **Escalation routing** | When the agent detects high-priority issues, it @mentions the appropriate team or individual |
| **Notification deduplication** | Consolidate related notifications into summaries to reduce notification fatigue |

---

## 21. Forks, Stars & Social Graph

### What GitHub Provides

GitHub's social features connect developers:
- **Forks** — independent copies of repositories
- **Stars** — bookmarking and signaling interest
- **Followers** — social connections between users
- **Contributors** — commit-based participation tracking
- **Network graph** — visualizing fork relationships

### How GitClaw Uses It Today

GitClaw does not interact with social features. However, when a repo is forked, the `.GITCLAW` folder is included — and the agent's memory comes along.

### What Could Be Done

| Possibility | Description |
|-------------|-------------|
| **Fork-aware memory** | Detect when running in a fork and adjust behavior — offer to start fresh or inherit upstream memory |
| **Contributor onboarding** | Detect new contributors (first-time PRs) and provide personalized onboarding guidance |
| **Star milestone celebrations** | Post celebratory messages or generate a community update when star milestones are reached |
| **Upstream sync assistance** | In forks, help users stay synchronized with the upstream repository |
| **Community health analysis** | Track contributor patterns, response times, and engagement metrics |

---

## 22. GitHub Copilot & AI Ecosystem

### What GitHub Provides

GitHub's AI ecosystem is expanding rapidly:
- **Copilot** — AI code completion in editors
- **Copilot Chat** — conversational AI in IDE and on GitHub.com
- **Copilot Workspace** — AI-driven development environment
- **Copilot for Pull Requests** — AI-generated PR summaries and reviews
- **GitHub Models** — API access to multiple LLMs via GitHub

### How GitClaw Relates

GitClaw occupies a different niche in the AI ecosystem:

| Dimension | GitHub Copilot | GitClaw |
|-----------|---------------|---------|
| **Where it runs** | Editor / cloud | Repository (Actions) |
| **Interface** | IDE / web chat | GitHub Issues |
| **Memory** | Session-based | Git-committed, permanent |
| **Customization** | Limited | Full (personality, skills, model) |
| **State persistence** | Ephemeral | Immutable git history |
| **Cost model** | Subscription | Pay-per-use (LLM API + Actions minutes) |
| **Self-hosting** | No | Yes (any runner + any LLM provider) |

### What Could Be Done

| Possibility | Description |
|-------------|-------------|
| **Complementary roles** | Copilot assists in the editor; GitClaw assists in the issue tracker and review process — different surfaces, different strengths |
| **GitHub Models integration** | Use GitHub Models API as an LLM provider, enabling GitClaw to run on GitHub-hosted models without external API keys |
| **Copilot Extensions** | Build a Copilot Extension that queries GitClaw's session history — "what did the agent say about this code?" |
| **Workflow handoff** | Copilot drafts code in the editor; the user opens a PR; GitClaw reviews it — a complete AI-assisted development loop |

> **Coexistence, not competition.** GitClaw and Copilot solve different problems at different layers. Together, they create an AI-assisted development experience that spans from keystroke to merge.

---

## 23. Cross-Cutting Analysis

### Feature × Agent Interaction Matrix

How each GitHub feature maps to agent interaction patterns:

| GitHub Feature | Trigger | Read | Write | Memory | Gate |
|----------------|---------|------|-------|--------|------|
| **Issues** | ✅ Event | ✅ Body, comments | ✅ Comments, reactions | ✅ Session mapping | — |
| **Pull Requests** | 🔮 Event | 🔮 Diff, files, threads | 🔮 Reviews, comments | 🔮 PR-session mapping | 🔮 Required check |
| **Actions** | ✅ Workflow | ✅ Logs, artifacts | ✅ Dispatch | ✅ State commits | ✅ Sentinel file |
| **Checks** | — | — | 🔮 Check runs, annotations | — | 🔮 Required check |
| **Labels** | — | 🔮 Issue labels | 🔮 Add/remove labels | — | 🔮 Label-based dispatch |
| **Projects** | — | 🔮 Board state | 🔮 Move items, set fields | — | — |
| **Milestones** | — | 🔮 Progress | 🔮 Assign issues | — | — |
| **Discussions** | 🔮 Event | 🔮 Threads | 🔮 Replies, answers | 🔮 Discussion-session mapping | — |
| **Releases** | 🔮 Event | 🔮 Release notes | 🔮 Draft/publish | — | 🔮 Environment gate |
| **Tags** | — | ✅ Git tags | 🔮 Create tags | — | — |
| **Code Scanning** | 🔮 Alert event | 🔮 Alerts | 🔮 Dismiss/comment | — | — |
| **Dependabot** | 🔮 Alert event | 🔮 Alerts, PRs | 🔮 Approve, merge | — | — |
| **Secret Scanning** | 🔮 Alert event | 🔮 Alerts | 🔮 Resolve | — | — |
| **Branch Protection** | — | 🔮 Rules | — | — | ✅ Existing rules respected |
| **CODEOWNERS** | — | 🔮 Ownership map | 🔮 Generate/update | — | — |
| **Pages** | — | — | 🔮 Deploy content | — | — |
| **Packages** | — | 🔮 Versions | 🔮 Publish | — | — |
| **Environments** | — | 🔮 Config | — | — | 🔮 Approval gates |
| **OIDC** | — | — | — | — | 🔮 Auth tokens |
| **GitHub Apps** | 🔮 Webhooks | ✅ Via API | ✅ Via API | — | 🔮 Installation perms |
| **Webhooks** | 🔮 External events | — | — | — | — |
| **Notifications** | — | 🔮 Activity feed | 🔮 @mentions | — | — |

✅ = used today | 🔮 = future possibility

### Permission Growth Path

As the agent participates in more surfaces, its required permissions expand:

```
Phase 0 (Today):     contents:write  issues:write  actions:write
Phase 1 (PRs):       + pull-requests:write  checks:write  statuses:write
Phase 2 (Projects):  + (GraphQL project mutations via existing token)
Phase 3 (Releases):  + (contents:write already covers tags/releases)
Phase 4 (Security):  + security-events:write  secret-scanning-alerts:read
Phase 5 (Identity):  GitHub App replaces GITHUB_TOKEN with scoped installation tokens
Phase 6 (Observe):   + id-token:write (for OIDC)
```

The GitHub App model (Phase 5) is the natural endpoint — it replaces the growing permission block with fine-grained, per-installation scoping.

### Session Model Universality

The `number → mapping file → session file` pattern works for every conversation surface:

```
state/
  issues/42.json           → state/sessions/2026-02-20T..._abc.jsonl
  pull-requests/17.json    → state/sessions/2026-02-20T..._def.jsonl
  discussions/5.json       → state/sessions/2026-02-20T..._ghi.jsonl
```

One pattern, many surfaces. The session model is GitClaw's most portable abstraction.

---

## 24. The Combinatorial Explosion

The real power is not in any single integration — it is in the **combinations**.

Consider what happens when just three features are connected:

```
Issues + Labels + Projects
  → A user opens an issue
  → The agent analyzes content, applies labels (bug, P1, area:auth)
  → The issue is automatically added to the project board in the "Triage" column
  → The agent posts a diagnostic comment citing relevant code
  → The project board reflects real-time repository health
```

Now add more:

```
Pull Requests + Checks + Code Scanning + Releases
  → A developer opens a PR
  → The agent reviews the diff, posts a Check Run with annotations
  → CodeQL alerts on the PR are triaged by the agent
  → Once approved and merged, the agent drafts release notes
  → The release is published after environment approval
```

And more:

```
Issues + Discussions + Wiki + Pages
  → A user asks a question in Discussions
  → The agent answers, citing code and past issues
  → The answer is summarized and added to the wiki
  → The wiki powers a searchable FAQ on GitHub Pages
  → Future questions are answered by searching the FAQ first
```

Each integration multiplies every existing one. The complexity stays linear — it is all files and API calls — but the capability surface grows **combinatorially**.

| N features integrated | Possible feature combinations |
|----------------------:|------------------------------:|
| 3 | 1 |
| 5 | 10 |
| 10 | 120 |
| 15 | 1,365 |
| 22 | 7,315 |

Most of these combinations are meaningful. That is the design space GitClaw opens up.

---

## 25. Conclusion

GitHub is not a code host. It is a **development platform** — a unified surface for code, conversation, review, automation, security, project management, releases, documentation, and community. It is the most feature-rich environment most developers will ever work in.

GitClaw's thesis is that an AI agent can participate in **every layer** of this platform using nothing but the repository itself as its runtime. No servers, no databases, no external services. Just a folder, a workflow, and an API key.

This document has mapped 22 distinct GitHub feature areas and their relationship to GitClaw. The analysis reveals several patterns:

1. **Issues are the proven foundation.** GitClaw's issue-driven conversation model works today and extends naturally to every other conversation surface (PRs, discussions, review threads).

2. **The session model is universal.** The `number → mapping → session` pattern works for issues, PRs, discussions, and any future surface where conversations happen.

3. **Git is the perfect state backend.** Every agent interaction — conversations, decisions, file changes, memory — is captured in git. Diffable, auditable, revertable, forkable.

4. **GitHub Actions is a sufficient compute engine.** With 35+ event triggers, secret management, caching, artifacts, environments, and OIDC, Actions provides everything an agent needs to participate in the full platform.

5. **The GitHub API is the control surface.** Every feature in this document maps to API calls. The agent's capability is bounded only by what the API exposes — and GitHub's API is extraordinarily comprehensive.

6. **Combinations compound.** Each new integration multiplies every existing one. The design space is combinatorial, but the implementation complexity stays linear.

7. **A GitHub App is the natural identity endpoint.** As the agent participates in more surfaces, a dedicated App identity replaces the growing permission block with fine-grained, auditable, marketplace-distributable scoping.

GitClaw today is an issue-comment assistant. The GitHub platform provides the surface area for it to become a full development collaborator — a participant in every stage from idea to deployment.

The possibilities are not theoretical. Every feature described in this document has a concrete API, a concrete trigger, and a concrete implementation path. The only question is which possibilities to pursue first.

The [Roadmap](GITCLAW-Roadmap.md) answers that question phase by phase. This document provides the map. The territory is GitHub itself.

---

### Related Documents

- [GITCLAW-The-Idea.md](GITCLAW-The-Idea.md) — The philosophical vision behind GitClaw
- [GITCLAW-Possibilities.md](GITCLAW-Possibilities.md) — Current and future use cases
- [GITCLAW-Interactive-Possibilities.md](GITCLAW-Interactive-Possibilities.md) — Deep analysis of interactive workflows
- [GITCLAW-Roadmap.md](GITCLAW-Roadmap.md) — Phased implementation plan
- [GITCLAW-Internal-Mechanics.md](GITCLAW-Internal-Mechanics.md) — Architecture and internals
- [GITCLAW-Loves-Pi.md](GITCLAW-Loves-Pi.md) — The Pi agent engine

---

_Last updated: 2026-02-20_
