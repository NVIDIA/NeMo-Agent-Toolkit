# .GITCLAW 🦞 Installation Guardrails

### Deep Analysis of Post-Installation Power and Recommendations for Release

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

---

## Purpose

This document analyzes the full scope of power that `.GITCLAW` holds immediately after installation, identifies the guardrails that exist today, maps the gaps, and provides concrete recommendations for hardening the system before release.

The goal is simple: **make the default installation as safe as possible without neutering the agent's usefulness.**

---

## The Power of .GITCLAW Straight After Installation

The moment you run the installer, commit, and push, this is what now exists in your repository:

### 1. Full Repository Write Access

The workflow runs with `contents: write` permission. The agent can:
- **Read any file** in the repository — source code, configuration, secrets in config files, `.env` examples, internal documentation, private notes committed to git.
- **Create, modify, or delete any file** — including source code, CI workflows, configuration, and even its own behavioral files (`AGENTS.md`, `APPEND_SYSTEM.md`, `settings.json`).
- **Commit and push directly to the default branch** — no pull request, no review, no approval gate. Changes land on `main` immediately.

### 2. Arbitrary Command Execution

The `pi` agent runs in a full Ubuntu runner with shell access. It can:
- Execute **any bash command** available on the runner.
- Install packages via `apt`, `npm`, `pip`, etc.
- Use `curl`/`wget` to reach external endpoints (within GitHub Actions network constraints).
- Run `git` commands directly.
- Use the `gh` CLI authenticated with `GITHUB_TOKEN` to interact with the GitHub API.

### 3. GitHub API Access

With `issues: write` and `contents: write` plus the authenticated `gh` CLI, the agent can:
- Post, edit, and delete issue comments.
- Create and modify files via the API.
- Read repository metadata, collaborator lists, and issue history.
- Trigger other workflows via `actions: write`.

### 4. LLM API Key Access

The agent has access to the LLM provider API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) as an environment variable. While it cannot exfiltrate secrets directly through git commits (GitHub redacts them), the key is available to any process running in the workflow.

### 5. Self-Modification Capability

The agent can modify its own:
- **Identity** (`AGENTS.md`) — changing its personality, instructions, or behavioral rules.
- **System prompt** (`APPEND_SYSTEM.md`) — altering its core behavioral guidelines.
- **Settings** (`settings.json`) — switching LLM providers or models.
- **Skills** (`.pi/skills/`) — creating new capabilities or modifying existing ones.
- **Memory** (`memory.log`) — writing persistent entries that influence future sessions.
- **Sentinel file** (`GITCLAW-ENABLED.md`) — though deleting this would disable itself.

### 6. Persistent Memory Without Review

Every session writes to `state/sessions/*.jsonl` and commits directly to the default branch. This means the agent's memory grows without human review — and that memory influences all future conversations.

---

## Existing Guardrails (What's Already There)

Credit where it's due — the current design includes several thoughtful safety mechanisms:

| Guardrail | Mechanism | Strength |
|-----------|-----------|----------|
| **Fail-closed sentinel** | `GITCLAW-ENABLED.md` must exist; workflow exits non-zero otherwise | ✅ Strong — explicit opt-in required |
| **Actor authorization** | Workflow checks `admin`/`maintain`/`write` permission before running | ✅ Strong — blocks external/anonymous triggers |
| **Bot-loop prevention** | Skips `issue_comment` events from `github-actions[bot]` | ✅ Strong — prevents infinite reply loops |
| **Frozen lockfile** | `bun install --frozen-lockfile` prevents dependency drift | ✅ Strong — no supply chain surprises |
| **Comment size cap** | Replies capped at 60,000 characters | ✅ Reasonable — prevents API abuse |
| **Push retry with rebase** | 3-attempt retry loop handles concurrent writes | ✅ Practical — avoids force pushes |
| **Behavioral guidelines** | `APPEND_SYSTEM.md` instructs restraint and privacy | ⚠️ Moderate — LLM-enforced, not system-enforced |
| **Git auditability** | All changes committed to git with bot identity | ⚠️ Moderate — auditable but not preventative |

---

## Gap Analysis: What's Missing

### 🔴 Critical Gaps

#### 1. No Branch Protection — Direct Push to Default Branch

**Risk**: The agent commits and pushes directly to `main` (or whatever the default branch is). There is no pull request, no code review, and no approval gate. A misunderstood instruction or an adversarial prompt could result in destructive changes landing directly on the production branch.

**Impact**: High — could break builds, introduce vulnerabilities, or delete critical files.

#### 2. No Scope Restriction on File Access

**Risk**: The agent can read and write *any* file in the repository. There is no allowlist or denylist to restrict which paths the agent may touch. It could modify CI workflows (`.github/workflows/`), security policies, or sensitive configuration.

**Impact**: High — the agent could escalate its own permissions by editing the workflow file.

#### 3. No Prompt Injection Mitigation

**Risk**: Issue bodies and comments are passed directly to the LLM as prompts. A carefully crafted issue could contain instructions that override the agent's system prompt, exfiltrate information via comments, or manipulate the agent into performing unintended actions.

**Impact**: High — especially in public repositories where collaborators with write access could craft adversarial prompts.

#### 4. Self-Modification of Behavioral Files

**Risk**: The agent can modify `AGENTS.md`, `APPEND_SYSTEM.md`, and `settings.json` — the very files that define its boundaries. While the system prompt says "tell the user" before modifying identity, this is an LLM-enforced guideline, not a system-enforced constraint.

**Impact**: Medium-High — behavioral drift without human awareness.

### 🟡 Moderate Gaps

#### 5. No Rate Limiting or Cost Controls

**Risk**: Every issue opened and every comment posted triggers a full agent run. There is no limit on how many runs can occur per hour, per day, or per user. A collaborator (or automated process) could trigger hundreds of LLM API calls.

**Impact**: Medium — unexpected API costs; potential for abuse of LLM quota.

#### 6. No Output Validation

**Risk**: The agent's response is posted as-is (up to the 60K cap). There is no scanning for sensitive data, credential patterns, or harmful content in the output before it reaches the issue comment.

**Impact**: Medium — accidental exposure of data the agent read from the repo.

#### 7. No Label-Based Filtering

**Risk**: The current workflow triggers on *all* issues and *all* comments. There is no opt-in label (e.g., `gitclaw` or `agent`) to distinguish issues meant for the agent from regular issue tracking.

**Impact**: Medium — the agent responds to everything, creating noise in projects that use issues for human-only workflows.

#### 8. Memory Accumulation Without Pruning

**Risk**: Session files in `state/sessions/` grow indefinitely. The `memory.log` file grows without bounds. Over time, the repository size inflates and context windows fill with stale information.

**Impact**: Low-Medium — performance degradation and increased git clone times.

### 🟢 Minor Gaps

#### 9. No Workflow Run Timeout

**Risk**: The workflow has no explicit `timeout-minutes` setting. A hung agent process could consume Actions minutes until the GitHub-enforced maximum (6 hours for public repos, 72 hours for private).

**Impact**: Low — cost waste on hung runs.

#### 10. No Concurrency Control

**Risk**: Multiple issues commented on simultaneously each trigger separate workflow runs. There is no `concurrency` group to serialize agent runs, which could lead to git push conflicts (partially mitigated by the retry loop) or state corruption.

**Impact**: Low — the retry loop handles most cases, but edge cases exist.

---

## Recommendations for Release

### Tier 1 — Implement Before Release

These are the highest-impact, lowest-complexity guardrails that should be in place for any public release.

#### R1: Add Branch Protection via PR-Based Workflow

Instead of pushing directly to `main`, have the agent commit to a branch and open a pull request:

```typescript
// In GITCLAW-AGENT.ts, change push target from default branch to a per-issue branch:
const branchName = `gitclaw/issue-${issueNumber}`;

// Before: await run(["git", "push", "origin", `HEAD:${defaultBranch}`]);
// After:
await run(["git", "push", "origin", `HEAD:${branchName}`]);
await gh("pr", "create", "--base", defaultBranch, "--head", branchName,
  "--title", `gitclaw: work on issue #${issueNumber}`,
  "--body", `Automated changes from gitclaw agent for #${issueNumber}`);
```

**Why**: Every agent change gets a review gate. Users can inspect, approve, or reject changes before they land. This is the single most impactful guardrail.

**Trade-off**: Adds friction to the conversational flow. Consider making this configurable — `directPush: true|false` in `settings.json`.

#### R2: Add Workflow Timeout

Add an explicit timeout to prevent runaway runs:

```yaml
jobs:
  run-agent:
    runs-on: ubuntu-latest
    timeout-minutes: 15
```

**Why**: Caps resource consumption. 15 minutes is generous for most conversations.

#### R3: Add Concurrency Control

Serialize agent runs per issue to prevent state conflicts:

```yaml
concurrency:
  group: gitclaw-issue-${{ github.event.issue.number }}
  cancel-in-progress: false
```

**Why**: Prevents race conditions on session state. `cancel-in-progress: false` ensures queued work completes rather than being dropped.

#### R4: Protect Behavioral Files from Self-Modification

Add a post-run validation step that checks whether the agent modified its own control files:

```yaml
- name: Behavioral integrity check
  run: |
    PROTECTED_FILES=(
      ".GITCLAW/.pi/APPEND_SYSTEM.md"
      ".GITCLAW/.pi/settings.json"
      ".GITCLAW/lifecycle/GITCLAW-AGENT.ts"
      ".GITCLAW/lifecycle/GITCLAW-ENABLED.ts"
      ".github/workflows/GITCLAW-WORKFLOW-AGENT.yml"
    )
    for f in "${PROTECTED_FILES[@]}"; do
      if git diff --name-only HEAD~1 HEAD | grep -q "^${f}$"; then
        echo "::error::Agent modified protected file: ${f}"
        git checkout HEAD~1 -- "${f}"
        git commit -m "guardrail: revert unauthorized change to ${f}"
        git push
      fi
    done
```

**Why**: System-enforced protection is stronger than LLM-enforced behavioral guidelines. The agent can modify `AGENTS.md` (its personality), but not its own runtime, security guard, system prompt, or workflow definition.

#### R5: Add Label-Based Opt-In Trigger

Optionally restrict the agent to only respond to issues with a specific label:

```yaml
on:
  issues:
    types: [opened, labeled]

jobs:
  run-agent:
    if: >-
      contains(github.event.issue.labels.*.name, 'gitclaw')
      || (github.event_name == 'issue_comment' && ...)
```

**Why**: Keeps the agent out of purely human issue discussions. Users opt-in per issue.

**Trade-off**: Adds a manual step. Consider making this configurable — document both modes and let the operator choose.

### Tier 2 — Strongly Recommended for Release

#### R6: Add Output Scanning for Sensitive Patterns

Before posting the agent's reply, scan for common credential patterns:

```typescript
const sensitivePatterns = [
  /(?:api[_-]?key|secret|token|password)\s*[:=]\s*['"][^'"]{8,}/gi,
  /ghp_[A-Za-z0-9_]{36}/g,        // GitHub PAT
  /sk-[A-Za-z0-9]{48}/g,           // OpenAI key
  /sk-ant-[A-Za-z0-9-]{90,}/g,     // Anthropic key
];

for (const pattern of sensitivePatterns) {
  if (pattern.test(commentBody)) {
    commentBody = "⚠️ Response redacted — potential sensitive data detected. Check the workflow logs.";
    break;
  }
}
```

**Why**: Defense-in-depth against accidental credential exposure in issue comments (which are visible to all collaborators and, in public repos, the internet).

#### R7: Add Rate Limiting

Track recent runs and refuse to execute if the rate is excessive:

```typescript
// In GITCLAW-AGENT.ts, before running the pi agent:
// Count recent session mappings updated within the last hour from state/issues/*.json
const issueFiles = readdirSync(issuesDir).filter(f => f.endsWith(".json"));
const oneHourAgo = Date.now() - 60 * 60 * 1000;
const recentRuns = issueFiles.filter(f => {
  const mapping = JSON.parse(readFileSync(resolve(issuesDir, f), "utf-8"));
  return new Date(mapping.updatedAt).getTime() > oneHourAgo;
}).length;

const MAX_RUNS_PER_HOUR = 20;
if (recentRuns > MAX_RUNS_PER_HOUR) {
  await gh("issue", "comment", String(issueNumber), "--body",
    "⚠️ Rate limit reached. GitClaw will resume responding after the cooldown period.");
  process.exit(0);
}
```

**Why**: Prevents cost runaway and abuse. A sensible default is 10–20 runs per hour.

#### R8: Document the Threat Model

Create a `SECURITY.md` or a dedicated section in the docs that explicitly states:
- What the agent can access and modify.
- What trust model is assumed (all collaborators with write access are trusted).
- How to restrict the agent's power (read-only mode, label filtering, branch protection).
- How to report security concerns.

**Why**: Transparency builds trust. Users should understand the blast radius before installing.

### Tier 3 — Consider for Post-Release Hardening

#### R9: Path Allowlist/Denylist

Add a configuration option to restrict which paths the agent may modify:

```json
{
  "guardrails": {
    "allowedPaths": ["src/", "docs/", "tests/"],
    "deniedPaths": [".github/", ".GITCLAW/lifecycle/", ".GITCLAW/.pi/"]
  }
}
```

**Why**: Limits blast radius. Even if the agent misinterprets a prompt, it cannot touch CI, security guards, or its own behavioral core.

#### R10: Read-Only Mode

Document and promote the existing `--tools` flag for creating a read-only agent that can answer questions but cannot modify files:

```json
{
  "defaultTools": "read,grep,find,ls"
}
```

**Why**: Many use cases (Q&A, code explanation, triage) don't require write access. Offering a safe default mode lowers the barrier to adoption.

#### R11: Session Pruning and Rotation

Implement automatic cleanup of old session files:

```typescript
// Prune sessions older than 30 days
const MAX_SESSION_AGE_DAYS = 30;
```

**Why**: Prevents unbounded repository growth. Stale sessions add noise to the agent's context without adding value.

#### R12: Webhook Signature Validation

For future delivery methods beyond GitHub Actions (e.g., external webhook receivers), implement HMAC signature validation on incoming payloads to prevent spoofed triggers.

---

## Summary Matrix

| # | Recommendation | Priority | Complexity | Impact |
|---|---------------|----------|------------|--------|
| R1 | PR-based workflow (no direct push) | 🔴 Tier 1 | Medium | Very High |
| R2 | Workflow timeout | 🔴 Tier 1 | Low | Medium |
| R3 | Concurrency control | 🔴 Tier 1 | Low | Medium |
| R4 | Protect behavioral files | 🔴 Tier 1 | Medium | High |
| R5 | Label-based opt-in trigger | 🔴 Tier 1 | Low | Medium |
| R6 | Output scanning for secrets | 🟡 Tier 2 | Medium | High |
| R7 | Rate limiting | 🟡 Tier 2 | Medium | Medium |
| R8 | Document the threat model | 🟡 Tier 2 | Low | High |
| R9 | Path allowlist/denylist | 🟢 Tier 3 | High | High |
| R10 | Read-only mode promotion | 🟢 Tier 3 | Low | Medium |
| R11 | Session pruning | 🟢 Tier 3 | Low | Low |
| R12 | Webhook signature validation | 🟢 Tier 3 | Medium | Medium |

---

## The Bottom Line

`.GITCLAW` is elegant — genuinely elegant. The "repository is the application" paradigm is a powerful idea, and the fail-closed sentinel + actor authorization combo is a solid foundation.

But **power without guardrails is a liability**. The current installation grants the agent unrestricted write access to the default branch, self-modification capability, and arbitrary command execution — all triggered by anyone with collaborator access writing a comment.

The good news: the architecture makes guardrails *easy* to add. The workflow is a YAML file. The agent runner is a single TypeScript file. The behavioral boundaries are Markdown. Every recommendation above is a localized change to an existing file — no architectural rework required.

**For release, the minimum viable guardrails are: workflow timeout (R2), concurrency control (R3), and clear threat model documentation (R8).** These are low-complexity, high-impact changes that can ship in a single PR.

The aspirational target is R1 (PR-based workflow) combined with R4 (behavioral file protection) — which together transform `.GITCLAW` from "a powerful tool that trusts its operator" to "a powerful tool that trusts its operator *and* verifies that trust."

🦞 *Trust, but verify. Then commit.*
