# .GITCLAW 🦞 Roadmap

### Deep GitHub integration — from issue bot to full-platform agent

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

> This roadmap builds on [GITCLAW-Possibilities.md](GITCLAW-Possibilities.md) and lays out concrete
> phases for turning GitClaw from an issue-comment assistant into an agent that
> participates in every layer of the GitHub platform — pull requests, reviews,
> checks, projects, releases, security scanning, and more.

---

## Guiding Principles

| Principle | Meaning |
|---|---|
| **Git-native state** | All agent memory, decisions, and artifacts stay in the repo — diffable, auditable, revertable. |
| **Zero external infrastructure** | Everything runs on GitHub Actions, the GitHub API, and the repository itself. |
| **Incremental opt-in** | Each capability is a separate workflow/skill that a repo owner enables explicitly. |
| **Fail-closed security** | The agent does nothing unless the correct sentinel files, permissions, and gates are in place. |
| **Human-in-the-loop by default** | The agent proposes; humans approve. Automation never bypasses review. |

---

## Phase 0 — Foundation (current state)

What GitClaw already does today:

- [x] Trigger on `issues.opened` and `issue_comment.created`
- [x] Authorization gating (`OWNER` / `MEMBER` / `COLLABORATOR`)
- [x] Multi-turn sessions persisted as JSONL in `state/sessions/`
- [x] Issue → session mapping in `state/issues/`
- [x] 👀 reaction indicator while working
- [x] Commit + push state to `main` with retry-on-conflict
- [x] Modular skill system and configurable personality

---

## Phase 1 — Pull Request Lifecycle

> **Goal:** Make the agent a first-class participant in code review and merging.

### 1.1 Pull Request Triggers

Extend the Actions workflow to listen on `pull_request` and `pull_request_review` events in addition to issues.

- [ ] Add `pull_request: [opened, synchronize, reopened]` trigger to `GITCLAW-WORKFLOW-AGENT.yml`
- [ ] Add `pull_request_review: [submitted]` and `pull_request_review_comment: [created]` triggers
- [ ] Map PR number → session (parallel to the existing issue → session model)
- [ ] Store PR sessions under `state/pull-requests/<pr>.json`

### 1.2 Pull Request Reviews

Use the GitHub API to post structured review objects (not just comments).

- [ ] Post `APPROVE`, `REQUEST_CHANGES`, or `COMMENT` reviews via `gh api`
- [ ] Attach file-specific review comments to the correct diff hunks
- [ ] Track unresolved review threads and resurface them on new pushes
- [ ] Summarize large diffs as a top-level review body before line-level comments

### 1.3 PR Comments & Threads

Participate in line-level review discussions the same way the agent participates in issue threads today.

- [ ] Listen for `pull_request_review_comment.created` events
- [ ] Reply inline to unresolved threads with analysis or suggested fixes
- [ ] Maintain a per-PR "open threads" summary in the session state
- [ ] Allow `/resolve` slash command to mark a thread as addressed

### 1.4 Check Runs

Post structured pass/fail results and inline annotations on commits.

- [ ] Create Check Runs via the Checks API (`POST /repos/{owner}/{repo}/check-runs`)
- [ ] Report `success`, `failure`, `neutral`, or `action_required` conclusions
- [ ] Attach inline annotations (file, line, message, severity) for code issues
- [ ] Update Check Run output with markdown summaries and progress counts
- [ ] Add `checks: write` permission to the workflow

### 1.5 Status Checks

Gate merges on agent-posted results.

- [ ] Register a required status check (e.g., `gitclaw/review`) in branch protection
- [ ] Post commit status via the Status API to signal policy compliance
- [ ] Block merge when the agent identifies critical issues (security, lint, tests)
- [ ] Clear/pass the status once issues are resolved and review is approved

---

## Phase 2 — Project Management & Triage

> **Goal:** Let the agent organize, label, and track work — not just respond to it.

### 2.1 Labels

Route and classify work using a managed label taxonomy.

- [ ] Build a `label-taxonomy` skill defining standard label categories (type, priority, area, status)
- [ ] Auto-label new issues based on title/body analysis
- [ ] Enforce label hygiene — warn or auto-fix when taxonomy rules are violated
- [ ] Support label-driven agent dispatch: `agent:review`, `agent:docs`, `agent:triage`

### 2.2 Projects (v2)

Integrate with GitHub Projects to move issues through workflow stages.

- [ ] Read project board state via the GraphQL API (`ProjectV2` schema)
- [ ] Move issues between columns/status fields based on agent analysis
- [ ] Set custom field values (priority, effort estimate, sprint)
- [ ] Auto-create project items when new issues match configured criteria
- [ ] Build a `project-sync` skill that reconciles board state with issue labels

### 2.3 Milestones

Group issues by release target and keep scope consistent.

- [ ] Assign issues to milestones based on label or content analysis
- [ ] Generate milestone progress reports (open/closed, burn-down, scope creep alerts)
- [ ] Warn when a milestone's scope changes significantly between check-ins
- [ ] Build a `milestone-report` skill triggered on schedule or slash command

### 2.4 Discussions

Participate in GitHub Discussions for long-form Q&A and RFCs.

- [ ] Add `discussion` and `discussion_comment` triggers to the workflow
- [ ] Summarize long discussion threads into actionable decisions
- [ ] Extract action items and open issues from discussion conclusions
- [ ] Build an `rfc-summarizer` skill that posts a structured summary as a pinned comment

---

## Phase 3 — Releases & Versioning

> **Goal:** Automate the release pipeline from changelog to tag to published release.

### 3.1 Releases

Generate changelogs and release notes from merged issues and PRs.

- [ ] Build a `release-notes` skill that collects merged PRs since last tag
- [ ] Categorize changes by label (features, fixes, breaking, internal)
- [ ] Draft a GitHub Release via the Releases API with generated notes
- [ ] Support both Keep-a-Changelog and Conventional Commits formats
- [ ] Allow human review before publishing (draft → published gate)

### 3.2 Tags

Mark version points and coordinate tagging with release notes.

- [ ] Create annotated git tags via `git tag -a` with structured messages
- [ ] Enforce semantic versioning by analyzing change categories (major/minor/patch)
- [ ] Coordinate tag creation with release note publication in a single workflow
- [ ] Build a `version-bump` skill that updates version files and creates the tag

---

## Phase 4 — Security & Compliance

> **Goal:** Turn the agent into a security-aware collaborator that triages alerts, enforces policy, and contains incidents.

### 4.1 Code Scanning (CodeQL)

Triage CodeQL alerts and suggest fixes.

- [ ] Read Code Scanning alerts via the REST API
- [ ] Post triage comments on alerts (dismiss with reason, or propose a fix)
- [ ] Open fix PRs for auto-remediable vulnerability patterns
- [ ] Track alert trends and surface regressions in a periodic report
- [ ] Build a `codeql-triage` skill with severity-based prioritization

### 4.2 Dependabot Alerts & PRs

Coordinate dependency updates and security fixes.

- [ ] Monitor Dependabot alerts via the API
- [ ] Auto-approve low-risk Dependabot PRs (patch bumps, no breaking changes)
- [ ] Summarize high-risk alerts in a triage issue with impact analysis
- [ ] Build a `dependency-review` skill that evaluates upgrade paths
- [ ] Add `security_advisory` event trigger for real-time response

### 4.3 Secret Scanning

Detect leaked credentials and trigger containment.

- [ ] Monitor Secret Scanning alerts via the API
- [ ] Immediately open a high-priority issue on new secret detection
- [ ] Post containment instructions (revoke, rotate, audit access logs)
- [ ] Auto-close alerts that are resolved or marked as false positives
- [ ] Build a `secret-response` skill with provider-specific rotation guidance

### 4.4 CODEOWNERS

Enforce reviewer requirements for sensitive paths.

- [ ] Generate and maintain a `CODEOWNERS` file based on git blame and team structure
- [ ] Validate that PRs touching protected paths have required reviewers assigned
- [ ] Warn when CODEOWNERS entries reference users who have left the organization
- [ ] Build a `codeowners-audit` skill that reports coverage gaps

### 4.5 Branch Protection Rules

Enforce review, checks, and signed commits before merge.

- [ ] Audit branch protection configuration and report gaps
- [ ] Recommend protection rules based on repository activity patterns
- [ ] Verify that agent-created PRs respect all protection requirements
- [ ] Surface protection rule violations in Check Run annotations

### 4.6 Rulesets

Org-level policy enforcement across multiple repositories.

- [ ] Read repository and org-level rulesets via the API
- [ ] Validate agent actions against active rulesets before execution
- [ ] Report ruleset compliance status in milestone and release reports
- [ ] Build a `policy-check` skill that audits cross-repo rule consistency

---

## Phase 5 — Advanced Workflow & Identity

> **Goal:** Give the agent a proper identity, secure credentials, and flexible triggering.

### 5.1 GitHub Apps

Fine-grained permissions and auditability for the agent identity.

- [ ] Package GitClaw as an installable GitHub App
- [ ] Use App installation tokens instead of `GITHUB_TOKEN` for tighter scoping
- [ ] Separate agent identity from workflow bot (`gitclaw[bot]` vs `github-actions[bot]`)
- [ ] Enable per-repo permission grants (issues only, issues+PRs, full access)
- [ ] Add webhook-based triggers alongside Actions for lower-latency responses

### 5.2 OIDC Credentials

Short-lived cloud auth without long-lived secrets.

- [ ] Configure OIDC trust between GitHub Actions and cloud providers (AWS, GCP, Azure)
- [ ] Replace static API keys with short-lived OIDC tokens in agent workflows
- [ ] Scope OIDC claims to specific repos, branches, and workflow steps
- [ ] Document OIDC setup for each major cloud provider in a `cloud-auth` guide
- [ ] Build a `credential-audit` skill that flags long-lived secrets in repo settings

### 5.3 Actions Environments

Add approval gates and protected deployment contexts.

- [ ] Define `staging` and `production` environments with reviewer requirements
- [ ] Run agent release skills through environment approval gates
- [ ] Add environment-specific secrets (deploy keys, API endpoints) scoped per stage
- [ ] Enforce environment protection rules on agent-initiated deployments

### 5.4 Workflow Dispatch

Manually trigger agent runs with parameters.

- [ ] Add `workflow_dispatch` trigger with configurable input parameters
- [ ] Define inputs for task type, target issue/PR, skill override, and model selection
- [ ] Build a dispatch UI that maps to common agent tasks
- [ ] Allow scheduled skills to also be triggered on-demand via dispatch

### 5.5 Repository Dispatch

Trigger workflows from external events without polling.

- [ ] Add `repository_dispatch` trigger with typed event payloads
- [ ] Define event types: `agent-task`, `security-alert`, `deploy-request`, `external-review`
- [ ] Document webhook integration for external CI, monitoring, and alerting systems
- [ ] Build an `event-router` skill that dispatches to the correct handler based on event type

---

## Phase 6 — Artifacts & Observability

> **Goal:** Store, inspect, and report on everything the agent does.

### 6.1 Artifacts

Store reports, summaries, and build outputs for later inspection.

- [ ] Upload agent reports as workflow artifacts via `actions/upload-artifact`
- [ ] Generate structured outputs: review reports, triage summaries, audit results
- [ ] Attach artifact links to issue/PR comments for easy access
- [ ] Implement artifact retention policies (e.g., keep last 10 runs per skill)
- [ ] Build a `report-archive` skill that indexes artifacts by type and date

### 6.2 Agent Observability

Track what the agent does, how long it takes, and what it costs.

- [ ] Log token usage, model, and latency per session turn
- [ ] Publish per-run metrics as artifact JSON files
- [ ] Build a `usage-report` skill that summarizes weekly/monthly agent activity
- [ ] Surface cost and performance trends in a GitHub Pages dashboard

---

## Phase Summary

| Phase | Focus | Key Outcomes |
|---|---|---|
| **0** | Foundation (done) | Issue-driven agent with git-backed state |
| **1** | Pull Request Lifecycle | Agent reviews PRs, posts checks, gates merges |
| **2** | Project Management & Triage | Auto-labeling, project board sync, milestone tracking |
| **3** | Releases & Versioning | Automated changelogs, tags, and GitHub Releases |
| **4** | Security & Compliance | CodeQL triage, Dependabot coordination, secret response, policy enforcement |
| **5** | Advanced Workflow & Identity | GitHub App identity, OIDC auth, dispatch triggers, environments |
| **6** | Artifacts & Observability | Structured reports, usage metrics, cost tracking |

---

## Integration Map

How each GitHub feature connects to GitClaw:

```
                         ┌─────────────────────────────────────────────┐
                         │              GitHub Platform                │
                         └─────────────────────────────────────────────┘
                                            │
        ┌───────────────┬───────────────┬───┴───┬───────────────┬──────────────┐
        │               │               │       │               │              │
   ┌────▼────┐   ┌──────▼──────┐  ┌─────▼────┐ │       ┌───────▼───────┐      │
   │ Issues  │   │Pull Requests│  │  Actions  │ │       │   Security    │      │
   │         │   │  & Reviews  │  │           │ │       │               │      │
   │ Phase 0 │   │  Phase 1    │  │ Phase 5   │ │       │  Phase 4      │      │
   └────┬────┘   └──────┬──────┘  └─────┬────┘ │       └───────┬───────┘      │
        │               │               │       │               │              │
        │         ┌─────▼──────┐  ┌─────▼────┐  │        ┌──────▼──────┐       │
        │         │Check Runs &│  │Dispatch & │  │        │CodeQL       │       │
        │         │Status      │  │Environ-   │  │        │Dependabot   │       │
        │         │Checks      │  │ments      │  │        │Secrets      │       │
        │         │ Phase 1    │  │ Phase 5   │  │        │ Phase 4     │       │
        │         └────────────┘  └──────────┘   │        └─────────────┘       │
        │                                        │                              │
   ┌────▼──────────────────┐           ┌─────────▼──────────┐          ┌───────▼───────┐
   │ Labels, Milestones,   │           │  Releases & Tags   │          │  Artifacts &  │
   │ Projects, Discussions │           │                    │          │  Observability│
   │                       │           │    Phase 3         │          │               │
   │       Phase 2         │           │                    │          │   Phase 6     │
   └───────────────────────┘           └────────────────────┘          └───────────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │   CODEOWNERS,         │
                                    │   Branch Protection,  │
                                    │   Rulesets             │
                                    │                       │
                                    │      Phase 4          │
                                    └───────────────────────┘
```

---

## Feature × Phase Matrix

| Feature | Phase | Trigger / API | New Permissions | Key Skill |
|---|---|---|---|---|
| Pull Requests | 1 | `pull_request` event | `pull-requests: write` | `pr-review` |
| PR Reviews | 1 | `pull_request_review` event | `pull-requests: write` | `pr-review` |
| PR Comments / Threads | 1 | `pull_request_review_comment` event | `pull-requests: write` | `pr-review` |
| Check Runs | 1 | Checks API | `checks: write` | `check-reporter` |
| Status Checks | 1 | Status API | `statuses: write` | `status-gate` |
| Labels | 2 | Labels API | `issues: write` | `label-taxonomy` |
| Projects (v2) | 2 | GraphQL `ProjectV2` | `project: write` (App) | `project-sync` |
| Milestones | 2 | Milestones API | `issues: write` | `milestone-report` |
| Discussions | 2 | `discussion` event | `discussions: write` | `rfc-summarizer` |
| Releases | 3 | Releases API | `contents: write` | `release-notes` |
| Tags | 3 | Git + Tags API | `contents: write` | `version-bump` |
| Code Scanning | 4 | Code Scanning API | `security-events: write` | `codeql-triage` |
| Dependabot | 4 | Dependabot API | `security-events: read` | `dependency-review` |
| Secret Scanning | 4 | Secret Scanning API | `secret-scanning-alerts: read` | `secret-response` |
| CODEOWNERS | 4 | File in repo | `contents: write` | `codeowners-audit` |
| Branch Protection | 4 | Branch Protection API | `administration: read` | `policy-check` |
| Rulesets | 4 | Rulesets API | `administration: read` | `policy-check` |
| GitHub Apps | 5 | App Installation API | App-managed | — |
| OIDC Credentials | 5 | OIDC provider config | `id-token: write` | `credential-audit` |
| Environments | 5 | Environments API | `deployments: write` | — |
| Workflow Dispatch | 5 | `workflow_dispatch` event | — | — |
| Repository Dispatch | 5 | `repository_dispatch` event | — | `event-router` |
| Artifacts | 6 | `actions/upload-artifact` | `actions: write` | `report-archive` |

---

## Implementation Notes

### Session model extension

The current `state/issues/<n>.json` → `state/sessions/<session>.jsonl` model extends naturally:

```
state/
  issues/
    42.json                 # issue #42 → session mapping
  pull-requests/
    17.json                 # PR #17 → session mapping
  discussions/
    5.json                  # discussion #5 → session mapping
  sessions/
    2026-02-20T..._abc.jsonl
```

Each event source (issue, PR, discussion) maps to a session via the same pointer pattern.

### Workflow file evolution

The single `GITCLAW-WORKFLOW-AGENT.yml` workflow grows to handle multiple event types:

```yaml
on:
  issues:
    types: [opened]
  issue_comment:
    types: [created]
  pull_request:
    types: [opened, synchronize, reopened]
  pull_request_review:
    types: [submitted]
  pull_request_review_comment:
    types: [created]
  discussion:
    types: [created]
  discussion_comment:
    types: [created]
  workflow_dispatch:
    inputs:
      task:
        description: "Task for the agent"
        required: true
      skill:
        description: "Skill override"
        required: false
  repository_dispatch:
    types: [agent-task, security-alert, deploy-request]
  schedule:
    - cron: "0 9 * * 1"  # weekly triage
```

### Permission growth

As capabilities expand, the workflow permission block grows:

```yaml
permissions:
  contents: write
  issues: write
  pull-requests: write
  checks: write
  statuses: write
  actions: write
  discussions: write
  security-events: write
  id-token: write
```

The GitHub App model (Phase 5) replaces this with fine-grained per-installation scoping.

### Skill naming convention

New skills follow a consistent naming pattern:

```
.GITCLAW/.pi/skills/
  pr-review.md
  check-reporter.md
  label-taxonomy.md
  project-sync.md
  milestone-report.md
  rfc-summarizer.md
  release-notes.md
  version-bump.md
  codeql-triage.md
  dependency-review.md
  secret-response.md
  codeowners-audit.md
  policy-check.md
  credential-audit.md
  event-router.md
  report-archive.md
  usage-report.md
```

Each skill is a self-contained Markdown file with optional shell scripts, following the existing skill system conventions.

---

## Dependencies Between Phases

```
Phase 0 (Foundation)
  └──▶ Phase 1 (PR Lifecycle)
         ├──▶ Phase 2 (Project Management) ──▶ Phase 3 (Releases)
         └──▶ Phase 4 (Security)
                └──▶ Phase 5 (Workflow & Identity) ──▶ Phase 6 (Observability)
```

- **Phase 1** depends on Phase 0: PR triggers reuse the session and commit/push model.
- **Phase 2** depends on Phase 1: label-driven dispatch and project tracking build on PR awareness.
- **Phase 3** depends on Phase 2: release notes are generated from labeled, milestoned PRs.
- **Phase 4** can start in parallel with Phase 2: security scanning is independent of project management.
- **Phase 5** depends on Phase 4: GitHub App identity is needed for advanced permissions.
- **Phase 6** depends on Phase 5: observability needs dispatch triggers and artifact uploads.

---

_Last updated: 2026-02-20_
