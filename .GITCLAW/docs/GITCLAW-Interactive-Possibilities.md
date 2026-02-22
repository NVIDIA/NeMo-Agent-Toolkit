# .GITCLAW 🦞 Interactive Possibilities

### A deep analysis of issue processing and the full design space of interactive GitHub AI

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

> This document examines how GitClaw processes issues today, then maps the wider
> design space of **interactive** AI usage across the GitHub platform — every
> surface where a human and an AI agent can hold a real-time, stateful dialogue
> within the tools developers already use.

---

## 1. How .GITCLAW Processes Issues Today — A Deep Look

Before exploring what's possible, it's worth understanding exactly what happens when a user opens an issue or posts a comment. The pipeline is deceptively simple on the surface, but the interactions between its stages create a robust interactive system.

### 1.1 The Trigger Layer

The workflow (`GITCLAW-WORKFLOW-AGENT.yml`) listens on two event types:

```yaml
on:
  issues:
    types: [opened]
  issue_comment:
    types: [created]
```

This creates two distinct interactive entry points:

| Event | Meaning | Interactive Implication |
|---|---|---|
| `issues.opened` | A human starts a new conversation | Fresh context; the agent has no prior memory for this thread |
| `issue_comment.created` | A human continues an existing conversation | The agent resumes from its full prior context for this issue |

The key insight is that **the issue thread IS the conversation interface**. There's no separate chat window, no external tool, no context switch. The developer stays in GitHub and the agent meets them there.

### 1.2 Authorization as Interactive Gatekeeping

Before any AI interaction occurs, the workflow performs a permission check:

```bash
PERM=$(gh api "repos/{owner}/{repo}/collaborators/{actor}/permission" --jq '.permission')
```

Only users with `admin`, `maintain`, or `write` permission can trigger the agent. This is an **interactive security boundary** — it determines _who_ can have conversations with the agent, not just _what_ the agent can do. On a public repository, this prevents arbitrary users from consuming LLM tokens or manipulating the agent's memory through crafted prompts.

### 1.3 The Fail-Closed Guard

`GITCLAW-ENABLED.ts` checks for the sentinel file `.GITCLAW/GITCLAW-ENABLED.md`. If it's missing, the entire pipeline halts silently. This is a **kill switch for interactivity itself** — a single `git rm` disables every AI conversation across every issue, instantly and atomically.

### 1.4 The Activity Indicator — Visual Feedback Loop

`GITCLAW-INDICATOR.ts` adds a 👀 reaction to the triggering issue or comment _before_ dependencies are installed. This is a critical UX detail: it creates a **feedback loop** between the human and the system.

```
User posts comment → 👀 appears (seconds) → User knows agent is working → Response appears (30–120s) → 👀 disappears
```

Without this indicator, the user would be staring at a silent issue thread with no idea whether anything is happening. The 👀 reaction transforms a potentially frustrating wait into a clear interactive signal. The reaction is:
- Added to the _specific comment_ that triggered the run (not just the issue)
- Guaranteed to be removed in a `finally` block, even on error
- Non-fatal if it fails — the agent still processes the request

### 1.5 Session Resolution — The Memory of Interaction

The core of GitClaw's interactivity is session continuity. When `GITCLAW-AGENT.ts` runs, it performs this resolution:

```
Issue #N → state/issues/N.json → state/sessions/<timestamp>.jsonl
```

**If a mapping exists and the session file is present:** the agent resumes the full conversation history. Every prior exchange — the user's questions, the agent's reasoning, tool calls, file edits — is loaded into context. The conversation picks up exactly where it left off.

**If no mapping exists (or the session file is missing):** the agent starts fresh. The issue title and body become the initial prompt.

This creates a fundamentally different interaction model from stateless chatbots:

| Property | Stateless Bot | GitClaw |
|---|---|---|
| Memory across messages | None | Full conversation history via JSONL sessions |
| Memory across days/weeks | None | Sessions persisted as git commits |
| Memory of file changes | None | Agent can see its own prior edits in the repo |
| Conversation forks | Not possible | Implicit via Pi's session tree structure |
| Auditability | Ephemeral logs | Every exchange is a diffable git commit |

### 1.6 Prompt Construction — Translating Interaction to Agent Input

The prompt the agent receives depends on the event type:

- **New issue:** `"${title}\n\n${body}"` — the full issue content becomes the opening message
- **Comment:** `event.comment.body` — the raw comment text, allowing natural conversational follow-up

This means users interact with the agent using the same Markdown they'd use to talk to human collaborators. No special syntax, no command prefixes, no structured forms. The issue thread is a natural-language conversation interface.

### 1.7 Agent Execution — The Thinking Phase

The `pi` agent runs in JSON streaming mode:

```
pi --mode json --provider <provider> --model <model> --session-dir ./state/sessions -p <prompt>
```

Output is piped through `tee` to simultaneously:
1. Stream events to the GitHub Actions log (observable in real-time by anyone watching the workflow)
2. Persist raw JSONL to `/tmp/agent-raw.jsonl` for post-processing

During this phase, the agent can:
- Read any file in the repository
- Write or edit files
- Execute shell commands
- Search code with grep/find
- Reason through multi-step problems

All of this happens within the context of the conversation — the agent's tool calls and results become part of the session transcript.

### 1.8 Response Extraction and Posting

After the agent finishes, the pipeline:
1. Reverses the JSONL output (`tac`)
2. Extracts the final assistant text via `jq` (finding the last `message_end` event with text content)
3. Caps the response at 60,000 characters (GitHub's limit is ~65,535)
4. Posts the response as a new comment on the originating issue

The response appears as a normal GitHub comment — rendered Markdown, with code blocks, tables, links, and all the formatting developers expect. The agent's reply is indistinguishable in format from a human collaborator's response.

### 1.9 State Persistence — Committing the Interaction

After posting, the agent:
1. Updates the issue → session mapping (`state/issues/N.json`)
2. Stages all changes (`git add -A`) — including any files the agent created or edited
3. Commits with message `gitclaw: work on issue #N`
4. Pushes to the default branch with a retry-on-conflict loop (up to 3 attempts with `git pull --rebase`)

This commit is the **durable record of the interaction**. It captures:
- The updated session transcript (full conversation history)
- The issue mapping (so the next interaction can resume)
- Any repository changes the agent made (code, docs, config files)

### 1.10 Cleanup — Closing the Feedback Loop

The `finally` block removes the 👀 reaction, signaling to the user that the agent has finished. This completes the interactive cycle:

```
Human speaks → Agent acknowledges (👀) → Agent thinks → Agent responds (comment) → Agent signals done (👀 removed)
```

---

## 2. Interactive Patterns That Emerge From This Design

The issue-processing pipeline described above creates a foundation for several interactive patterns that go beyond simple Q&A.

### 2.1 Multi-Turn Investigations

Because sessions persist, a user can conduct an extended investigation across multiple comments:

```
Comment 1: "There's a bug in the authentication module. Users are getting 403 errors after token refresh."
Comment 2: "Can you check if the token expiry calculation accounts for clock skew?"
Comment 3: "Good catch. Now look at the retry logic — does it handle the case where the refresh token itself has expired?"
Comment 4: "Write a fix for this and update the tests."
```

Each comment builds on the prior context. The agent remembers what it found, what the user asked, and what files it examined. This is a genuine interactive debugging session — the kind that would normally happen over pair programming or a screen share.

### 2.2 Iterative Document Drafting

A user can collaboratively write documentation with the agent:

```
Comment 1: "Draft a README for the new API client module."
Comment 2: "Good start, but add a section on error handling and include code examples."
Comment 3: "The rate limiting section needs more detail. Also, move the configuration section before the usage section."
Comment 4: "Perfect. Commit this as docs/api-client.md."
```

The agent acts as a writing partner who remembers every revision and can apply incremental feedback — all within a single issue thread.

### 2.3 Exploratory Code Analysis

When investigating an unfamiliar codebase, a user can have a guided tour:

```
Comment 1: "Give me an overview of the authentication system architecture."
Comment 2: "How does the OAuth flow work? Walk me through the code path."
Comment 3: "What happens if the OAuth provider is down? Show me the fallback logic."
Comment 4: "Are there any security concerns with this approach?"
```

Each answer builds on the prior exploration, and the agent can reference files and functions it already examined.

### 2.4 Planning and Decision-Making

Issues become structured decision records:

```
Comment 1: "We need to choose between PostgreSQL and DynamoDB for the session store. Analyze the tradeoffs for our use case."
Comment 2: "What about cost at our expected scale of 10K concurrent sessions?"
Comment 3: "Good analysis. We're going with PostgreSQL. Document this decision and create a migration plan."
```

The issue thread captures the full decision process — question, analysis, decision, and action — all searchable and linkable.

---

## 3. Interactive Possibilities Across GitHub Surfaces

GitHub offers far more interactive surfaces than just issues. Each one represents a potential conversation interface for an AI agent.

### 3.1 Pull Request Conversations

Pull requests are GitHub's richest interactive surface. They combine code diffs, review threads, status checks, and general comments in a single view.

**Inline Review Threads:**
A reviewer (human or agent) can start a conversation on a specific line of code. The agent could:
- Post a review with line-specific comments (not just a top-level summary)
- Respond to replies within a thread ("Why do you think this is a problem?" → agent explains its reasoning with references)
- Mark threads as resolved when the author addresses the feedback
- Track unresolved threads across force-pushes and provide updated assessments

**Interactive Code Suggestions:**
```
Agent: "This function has O(n²) complexity due to the nested loop on line 47. Consider using a Set for O(n) lookup."
Human: "Show me what the refactored version would look like."
Agent: [posts a suggested change block with the refactored code]
Human: "What about the edge case where the input array is empty?"
Agent: "Good point — here's the updated version with an empty-array guard."
```

This is interactive pair programming embedded directly in the review interface.

**PR Summary and Explanation:**
When a PR is opened, the agent could post a structured summary:
- What changed and why (inferred from commit messages, diff, and linked issues)
- Impact analysis (which other modules are affected)
- Test coverage assessment
- Risk rating with justification

Reviewers can then ask follow-up questions directly on the summary comment, creating a dialogue about the change as a whole.

### 3.2 Discussion Forums

GitHub Discussions are designed for long-form, threaded conversations — a natural fit for AI interaction.

**RFC Facilitation:**
A team posts a Request for Comments as a Discussion. The agent could:
- Summarize the proposal's key points
- Identify potential conflicts with existing architecture
- Answer technical questions from team members who haven't read the full RFC
- Track open questions and action items as the discussion evolves
- Post a final summary when the RFC is accepted or rejected

**Q&A Category:**
Discussions have a "Q&A" category with answer marking. The agent could:
- Answer questions with codebase-specific context
- Mark its own answers when it's confident, or explicitly ask for human validation
- Update answers when the codebase changes (on a schedule or trigger)
- Link to related questions to build a knowledge web

**Polls and Surveys:**
The agent could create structured polls within discussions, collect responses, and summarize results — useful for architecture decisions, priority ranking, or feature requests.

### 3.3 GitHub Projects (v2)

Projects are interactive Kanban/Table/Roadmap views. An AI agent could participate as a project manager:

**Interactive Triage:**
```
Human: "@agent triage the new issues from this week"
Agent: [analyzes 15 new issues, adds labels, sets priority fields, assigns to milestones, moves cards to appropriate columns]
Agent: "Triaged 15 issues. 3 are P1 bugs assigned to the current sprint, 8 are feature requests added to the backlog, 4 need clarification — I've posted questions on each."
```

**Sprint Planning:**
The agent could analyze the backlog, estimate effort based on codebase complexity, and propose sprint contents — with the team interacting via issue comments to adjust.

**Burn-Down Tracking:**
On a schedule, the agent could post sprint health updates: progress, blockers, scope changes, and risk assessments.

### 3.4 GitHub Actions as Interactive Infrastructure

Actions aren't just CI/CD — they're an event-driven compute platform that can power interactive workflows.

**Workflow Dispatch with Interactive Inputs:**
```yaml
workflow_dispatch:
  inputs:
    task:
      description: "What should the agent do?"
      type: string
    scope:
      description: "Which area of the codebase?"
      type: choice
      options: [frontend, backend, infrastructure, docs]
    urgency:
      description: "Priority level"
      type: choice
      options: [low, medium, high, critical]
```

This turns the Actions "Run workflow" button into an interactive command interface with structured inputs.

**Repository Dispatch for External Triggers:**
External systems (monitoring alerts, CI failures, deployment events) can trigger the agent via `repository_dispatch`, creating an interactive bridge between operational infrastructure and the codebase.

**Slash Commands as Action Triggers:**
Issue comments containing commands like `/review`, `/deploy`, `/rollback`, `/analyze` could trigger specific workflows, each running the agent with a targeted skill. This creates a command-line-like interface within the conversational issue thread.

### 3.5 Check Runs and Status Checks

Check Runs are interactive annotations on commits and pull requests. The agent could:

- Post detailed analysis as check run annotations (file, line, message, severity)
- Include a summary tab with markdown-formatted reports
- Set pass/fail status that gates merges
- Update annotations as the PR evolves (new pushes, resolved threads)

**Interactive Check Negotiation:**
```
Check: "FAIL — 3 security issues found in dependency updates"
Human: [clicks "Details" on the check] → sees full analysis with line-specific annotations
Human: [comments on PR] "The lodash vulnerability is mitigated by our input validation layer. Can you re-evaluate?"
Agent: [re-runs analysis considering the mitigation] → "Updated: 2 issues remaining. lodash vulnerability accepted with documented mitigation."
```

This transforms automated checks from binary pass/fail signals into interactive negotiations between humans and the AI.

### 3.6 Releases as Interactive Milestones

A release workflow could be an interactive process:

```
Human: [opens issue] "Prepare release v2.3.0"
Agent: "Here's what's changed since v2.2.0: [summary of 23 merged PRs, categorized by type]"
Agent: "I've drafted release notes. 2 PRs are missing descriptions — should I use their commit messages instead?"
Human: "Yes, and add a migration note for the database schema change in PR #147."
Agent: "Updated. Here are the final release notes: [preview]. Ready to tag?"
Human: "Ship it."
Agent: [creates tag, publishes release, posts announcement]
```

The release isn't automated blindly — it's a guided, interactive process where the human maintains control over the final product.

### 3.7 Security Scanning as Interactive Triage

Security alerts (CodeQL, Dependabot, secret scanning) could trigger interactive triage sessions:

**CodeQL Alert Triage:**
```
Agent: [opens issue] "New CodeQL alert: SQL injection in api/users.ts:42. Severity: High."
Agent: "Analysis: The user input flows from req.query.name through buildQuery() without sanitization. The fix is to use parameterized queries. Here's a suggested patch: [code]"
Human: "The buildQuery function is also called from the admin endpoint. Does that path have the same issue?"
Agent: "Checking... Yes, admin/reports.ts:78 has the same vulnerability. Updated patch covers both call sites."
```

**Secret Scanning Response:**
```
Agent: [opens issue, labels: priority:critical] "🚨 Leaked AWS access key detected in commit abc1234"
Agent: "Immediate actions needed: 1) Rotate the key in AWS IAM 2) Check CloudTrail for unauthorized access since [commit date]"
Agent: "I've prepared a PR that removes the key and adds it to .gitignore. The commit that introduced it was authored by [user] on [date]."
Human: "Key rotated. CloudTrail shows no unauthorized access."
Agent: "Good. Closing the alert. I've also added a pre-commit hook to prevent future secret commits."
```

### 3.8 Wiki and Pages as Interactive Documentation

While GitHub Wikis and Pages are less event-driven, the agent could maintain them interactively:

```
Human: [opens issue] "The API documentation is out of date. Update it to match the current codebase."
Agent: [reads current API code, compares with wiki pages]
Agent: "Found 12 endpoints that have changed since the docs were last updated. 3 are new, 7 have modified parameters, 2 have been deprecated. Here's a summary of changes."
Human: "Update the wiki pages and add deprecation notices."
Agent: [uses GitHub API to update wiki pages, posts before/after comparison]
```

---

## 4. Advanced Interactive Patterns

### 4.1 Agent-to-Human Escalation

Not every interaction starts with a human. The agent can initiate interactive sessions:

**Proactive Alerts:**
The agent detects an issue during a scheduled run and opens a new issue:
```
Agent: "During the weekly dependency audit, I found that `jsonwebtoken` v8.5.1 has a critical vulnerability (CVE-2025-XXXX). The fix requires upgrading to v9.0.0, which has breaking changes affecting 3 modules."
Agent: "I've analyzed the migration path and prepared a draft PR. Key changes: [summary]. Should I proceed?"
```

The human is brought into the conversation at the right moment with full context, rather than having to discover the problem themselves.

**Threshold-Based Escalation:**
The agent runs autonomously for routine tasks but escalates when it encounters something beyond its confidence level:
```
Agent: [auto-labeled 47 issues this week, auto-approved 12 low-risk Dependabot PRs]
Agent: [opens issue] "I need human input on PR #234. The refactoring changes the public API in a way that could break downstream consumers. Here are the affected packages and the specific API changes: [analysis]."
```

### 4.2 Multi-Agent Interactive Chains

Multiple agents with different specializations can interact through issues:

```
Triage Agent: [labels issue as `type:security`, `priority:high`]
Security Agent: [triggered by label] [posts analysis of the vulnerability]
Review Agent: [triggered by PR created by Security Agent] [reviews the fix]
Human: [approves PR after reviewing all agents' analysis]
```

Each agent operates in its own session but contributes to the same issue thread, creating a multi-perspective interactive workflow.

### 4.3 Interactive Templates

Issue templates could be designed for specific interactive workflows:

**Bug Report Template with Agent Guidance:**
```markdown
### Bug Report (Agent-Assisted)

**Describe the bug:**
<!-- The agent will analyze your description and ask clarifying questions -->

**Steps to reproduce:**
<!-- Provide steps; the agent will attempt to trace the code path -->

**Expected behavior:**

**Actual behavior:**

**Environment:**
```

When a user fills out this template, the agent immediately:
1. Analyzes the description for relevant code areas
2. Asks targeted clarifying questions if the report is ambiguous
3. Attempts to reproduce the issue by tracing the code path
4. Posts preliminary findings

The template becomes the start of an interactive diagnostic session, not just a static form.

### 4.4 Conversation Branching

Pi's session tree structure enables a powerful interactive pattern: conversation branching.

```
Issue thread (linear):
  Comment 1: "Should we use REST or GraphQL for the new API?"
  Comment 2: [Agent analysis]
  Comment 3: "Explore the REST approach in detail."
  Comment 4: [Agent designs REST API]
  Comment 5: "Actually, also explore the GraphQL approach."
  Comment 6: [Agent designs GraphQL API, with full memory of the REST exploration for comparison]
```

Under the hood, the session contains both exploration branches. The agent can reference and compare them because both paths exist in its session tree. This is architectural exploration with full context preservation.

### 4.5 Interactive Onboarding

For new contributors, the agent can serve as a personalized interactive guide:

```
New Contributor: [opens issue] "I want to contribute to the authentication module. Where do I start?"
Agent: "Welcome! Here's an overview of the auth module: [architecture summary]. The best first issues are #45 (add rate limiting) and #52 (improve error messages). Would you like a walkthrough of either one?"
New Contributor: "Walk me through #45."
Agent: "Rate limiting needs to be added to the /api/login endpoint in auth/login.ts. Here's the current flow: [code walkthrough]. The recommended approach is to use a sliding window counter. Here's a skeleton implementation: [code]. Want me to explain any part in more detail?"
New Contributor: "How does the sliding window algorithm work?"
Agent: [explains with examples, references to the codebase, and suggested test cases]
```

This is interactive mentoring — patient, contextual, and available 24/7.

### 4.6 Interactive Refactoring Sessions

Large refactoring efforts can be conducted as interactive sessions:

```
Comment 1: "We need to migrate from callbacks to async/await across the data access layer."
Comment 2: [Agent surveys all callback-based functions, categorizes by complexity]
Comment 3: "Start with the simple ones. Show me the plan for the first 5."
Comment 4: [Agent shows before/after for 5 functions, with test updates]
Comment 5: "Looks good. Proceed with these and show me the next batch."
Comment 6: [Agent applies changes, shows the next 5, notes one function that requires special handling]
Comment 7: "Explain the special case."
Comment 8: [Agent details the race condition that the current callback avoids, suggests an async-safe alternative]
```

The refactoring proceeds as a guided, interactive process where the human reviews each batch and makes decisions about complex cases.

---

## 5. The Interaction Model: Why Issues Are Powerful

The choice to use GitHub Issues as the conversation interface is a deeper design decision than it might appear. Issues provide properties that purpose-built chat interfaces lack:

### 5.1 Persistent, Linkable Conversations

Every issue has a permanent URL. Conversations can be referenced from:
- Other issues (`#42`)
- Pull requests
- Commit messages
- External documentation
- Slack/Discord messages
- Email notifications

This makes AI conversations **first-class artifacts** in the development process, not ephemeral chat messages that disappear.

### 5.2 Built-In Notification System

GitHub's notification system means:
- Users are notified when the agent responds (email, mobile, web)
- Team members can be @mentioned into conversations
- Subscription controls let users follow or mute specific threads
- Organization-wide notification routing works automatically

The agent doesn't need to build its own notification system — it inherits GitHub's.

### 5.3 Search and Discovery

Issue conversations are indexed by GitHub's search:
- Full-text search across all agent conversations
- Filter by label, milestone, author, date
- Sort by activity, creation date, or comment count

This turns the sum of all agent interactions into a searchable knowledge base.

### 5.4 Access Control and Audit

Every comment has an author and timestamp. GitHub's audit log tracks:
- Who triggered the agent
- When the agent responded
- What the agent changed (via linked commits)

For regulated environments, this provides a complete audit trail of AI-assisted decisions.

### 5.5 Markdown as a Universal Interface

Issues render Markdown, which means agent responses can include:
- Formatted text with headers, lists, and emphasis
- Code blocks with syntax highlighting
- Tables for structured data
- Images and diagrams (via URLs or Mermaid)
- Task lists for actionable items
- Collapsible sections for detailed information

The agent's output is as rich and readable as a human's — because it uses the same rendering system.

---

## 6. Interactive Possibilities Beyond Code

### 6.1 Product Management

```
PM: "Analyze our open issues and suggest which ones should go in the next release."
Agent: [categorizes 89 open issues by theme, estimates complexity, identifies dependencies]
Agent: "Recommended release scope: [12 issues]. These are self-contained, well-defined, and collectively deliver the 'improved error handling' theme that 4 customers have requested."
PM: "What if we also include the dashboard redesign?"
Agent: "That adds 3 more issues and increases the estimated timeline by 2 weeks. Here's the dependency graph showing why: [diagram]."
```

### 6.2 Incident Response

```
On-Call: "Production is returning 500 errors on the /api/payments endpoint since the last deploy."
Agent: [identifies the deploy commit, diffs changes, traces the affected code path]
Agent: "The error is caused by a null pointer in payment-processor.ts:147. The deploy changed the PaymentResponse type to make the `transactionId` field optional, but the downstream handler assumes it's always present."
Agent: "Immediate fix: add a null check at line 147. Root cause: the type change in PR #89 wasn't caught because the test mocks always include transactionId."
On-Call: "Apply the fix and open a PR."
Agent: [creates branch, commits fix, opens PR with full context linked to this issue]
```

### 6.3 Compliance and Governance

```
Compliance Officer: "We need to verify that all API endpoints validate input according to our security policy."
Agent: [scans all route handlers, checks for validation middleware, identifies gaps]
Agent: "Found 47 endpoints. 39 have proper input validation. 8 are missing validation: [list with file and line references]. 3 of those handle sensitive data (PII)."
Compliance Officer: "Prioritize the 3 PII endpoints. What validation is needed for each?"
Agent: [provides endpoint-specific validation requirements based on the data types handled]
```

### 6.4 Architecture Decision Records (ADRs)

Interactive creation of ADRs that capture the full decision process:

```
Architect: "We need to decide on the caching strategy for the recommendation engine."
Agent: [analyzes the recommendation module, identifies hot paths, estimates cache hit rates]
Agent: "Three options: 1) Redis with 15-min TTL, 2) In-process LRU cache, 3) CDN edge caching. Here's the analysis: [comparison table]."
Architect: "Redis seems right. What about cache invalidation?"
Agent: [details invalidation strategies for each data source]
Architect: "Document this as ADR-0023."
Agent: [creates docs/adr/0023-recommendation-caching.md with full context from this conversation, linked to this issue]
```

---

## 7. Design Space: Interactive Modalities

### 7.1 Synchronous vs. Asynchronous Interaction

| Mode | Current | Possible |
|---|---|---|
| **Synchronous (real-time)** | Not supported — Actions has 30–120s latency | Streaming responses via progressive comment edits; WebSocket-based real-time interaction via GitHub App webhooks |
| **Asynchronous (turn-based)** | ✅ Full support via issue comments | Extended with scheduled check-ins, proactive notifications, and batch processing |
| **Hybrid** | Partially supported (👀 indicator) | Agent posts "thinking..." comments that update progressively, then finalizes |

### 7.2 Human-Initiated vs. Agent-Initiated

| Initiator | Current | Possible |
|---|---|---|
| **Human → Agent** | ✅ Issues and comments | Extended to PR reviews, discussions, slash commands, workflow dispatch |
| **Agent → Human** | Not supported | Proactive issue creation, @mention escalations, scheduled reports, alert notifications |
| **Agent → Agent** | Not supported | Cross-issue references, label-driven dispatch, repository dispatch chains |
| **External → Agent** | Not supported | Repository dispatch from monitoring, CI, deployment, and security systems |

### 7.3 Interaction Depth Levels

```
Level 0: One-shot    — User asks, agent answers, done
Level 1: Multi-turn  — User and agent exchange multiple messages ← GitClaw today
Level 2: Multi-issue — Agent references and connects conversations across issues
Level 3: Multi-surface — Agent operates across issues, PRs, discussions, and checks simultaneously
Level 4: Multi-repo  — Agent coordinates across repositories in an organization
Level 5: Multi-agent — Multiple specialized agents collaborate through GitHub's event system
```

Each level multiplies the interactive possibilities of the previous one.

---

## 8. The Conversational Repository

The deepest possibility that GitClaw opens up is the concept of the **conversational repository** — a repo where every artifact has an associated dialogue that explains, evolves, and maintains it.

### 8.1 Every File Has a Story

Imagine a repository where:
- Every module has an associated issue thread where the agent explains its design and answers questions
- Every architectural decision is captured in an interactive ADR session
- Every bug fix has a linked diagnostic conversation
- Every deployment has an interactive runbook

The repository becomes not just a collection of files, but a **living knowledge system** where the reasoning behind every change is preserved, searchable, and interactive.

### 8.2 Institutional Memory That Talks Back

Traditional institutional knowledge is static: wikis, docs, comments in code. GitClaw's interactive model creates institutional knowledge that **responds to questions**:

```
New Team Member: "Why did we choose gRPC over REST for the internal services?"
Agent: [loads the relevant session history, the ADR, the original issue discussion]
Agent: "The decision was made in issue #234 on 2025-11-15. The key factors were: [summary]. The original analysis compared latency, streaming support, and type safety. Since then, the team has also found that gRPC's code generation reduces boilerplate by ~40%. Want to see the full original discussion?"
```

The agent doesn't just retrieve the answer — it synthesizes context from multiple sources and presents it in a way that's relevant to the question.

### 8.3 Self-Documenting Processes

When processes are conducted through interactive issue sessions, documentation is a natural byproduct:

- **Release process:** Captured in the issue thread for each release
- **Onboarding steps:** Recorded in each new contributor's onboarding issue
- **Incident response:** Documented in real-time during the incident issue
- **Security reviews:** Preserved in the triage issue for each alert

No one needs to write process documentation after the fact — the interactive sessions ARE the documentation.

---

## 9. Technical Enablers for Richer Interaction

### 9.1 Progressive Comment Updates

Instead of waiting for the full response, the agent could:
1. Post an initial "Analyzing..." comment
2. Progressively update it with findings as they emerge
3. Finalize the comment when processing is complete

This would use the GitHub API's `PATCH /repos/{owner}/{repo}/issues/comments/{id}` endpoint to edit a comment in place.

### 9.2 Structured Responses with Collapsible Sections

For complex analyses, the agent could use GitHub's `<details>` tag:

```markdown
## Analysis Summary
Found 3 issues. 1 critical, 2 informational.

<details>
<summary>🔴 Critical: SQL injection in api/users.ts</summary>

[full analysis, code references, suggested fix]
</details>

<details>
<summary>ℹ️ Informational: Unused import in utils/helpers.ts</summary>

[details]
</details>
```

This keeps the interactive surface clean while allowing deep exploration.

### 9.3 Task Lists for Interactive Checklists

Agent responses could include interactive task lists:

```markdown
## Migration Checklist

- [x] Update dependency versions
- [x] Fix breaking API changes in module A
- [ ] Fix breaking API changes in module B
- [ ] Update integration tests
- [ ] Run full test suite
- [ ] Update documentation
```

Users can check items off as they complete them, and the agent can read the updated checkbox states in the next interaction.

### 9.4 Reactions as Lightweight Feedback

Beyond 👀, reactions could serve as a feedback channel:
- 👍 on agent comments = positive feedback (agent calibrates)
- 👎 = negative feedback (agent adjusts approach)
- 🚀 = proceed with suggested action
- ❓ = need more explanation

The agent could read these reactions and adapt its behavior — a form of non-verbal interactive feedback.

---

## 10. Summary: The Interactive Design Space

GitClaw's current issue-processing pipeline is a **complete interactive system**: trigger → acknowledge → think → respond → persist → cleanup. Every step is designed for stateful, multi-turn, auditable human-AI dialogue.

But issue comments are just the beginning. The interactive possibilities span:

| Surface | Interaction Type | Key Value |
|---|---|---|
| **Issues** | Multi-turn conversations | Long-running investigations, planning, decisions |
| **Pull Requests** | Code-level dialogue | Interactive review, pair programming, refactoring guidance |
| **Review Threads** | Line-specific discussions | Contextual code analysis with back-and-forth |
| **Discussions** | Long-form Q&A and RFCs | Knowledge building, architectural decisions |
| **Check Runs** | Annotated analysis | Pass/fail with interactive negotiation |
| **Projects** | Board management | Interactive triage, sprint planning, tracking |
| **Releases** | Guided publishing | Interactive changelog generation and release coordination |
| **Security Alerts** | Incident response | Interactive triage, containment, and remediation |
| **Workflow Dispatch** | On-demand tasks | Structured interactive commands with parameters |
| **Repository Dispatch** | External integration | Bridge between operational systems and interactive AI |

The unifying principle: **meet developers where they already work, and make every interaction stateful, auditable, and resumable.**

GitClaw proves that a folder, a workflow, and an LLM API key can create an interactive AI collaborator as natural as talking to a teammate. The only question is which surfaces to light up next.

---

_Last updated: 2026-02-20_
