# 🦞 GITCLAW .pi Configuration

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

How to tune the agent's behavior — from model selection to tool access.

## settings.json

**Location**: `.GITCLAW/.pi/settings.json`

The default LLM configuration:

```json
{
  "defaultProvider": "anthropic",
  "defaultModel": "claude-opus-4-6",
  "defaultThinkingLevel": "high"
}
```

| Field | Description | Default |
|-------|-------------|---------|
| `defaultProvider` | LLM provider | `anthropic` |
| `defaultModel` | Model identifier | `claude-opus-4-6` |
| `defaultThinkingLevel` | Reasoning depth (`low`, `medium`, `high`) | `high` |

These are the defaults used when the workflow doesn't specify overrides.

## Workflow Overrides

**Location**: `.github/workflows/GITCLAW-WORKFLOW-AGENT.yml`

The workflow file is the primary place to customize runtime behavior. Edit the `pi` command in the `Run` step:

### Model Selection

```yaml
- name: Run
  run: bun .GITCLAW/lifecycle/GITCLAW-AGENT.ts
  # The pi agent reads settings.json by default.
  # Override with --provider and --model flags in lifecycle/GITCLAW-AGENT.ts
```

To change the model, edit the `piArgs` array in `lifecycle/GITCLAW-AGENT.ts` or update `settings.json`.

### Thinking Level

Higher thinking levels give the agent more reasoning depth but use more tokens and time.

| Level | Use case |
|-------|----------|
| `low` | Simple tasks, quick responses |
| `medium` | Standard work, balanced speed/quality |
| `high` | Complex problems, multi-step reasoning |

Update `defaultThinkingLevel` in `settings.json`, or pass `--thinking <level>` in the agent args.

### Tool Restrictions

By default, the agent has access to all tools (shell, file I/O, git, GitHub CLI). You can restrict access:

```
--tools read,grep,find,ls
```

This creates a **read-only agent** that can analyze the repo but not modify it. Useful for code review or analysis tasks.

## Workflow Triggers

**Location**: `.github/workflows/GITCLAW-WORKFLOW-AGENT.yml`

The `on:` block controls when the agent activates:

```yaml
on:
  issues:
    types: [opened]
  issue_comment:
    types: [created]
```

You can customize triggers to:
- Filter by **labels**: Only respond to issues with specific labels
- Filter by **assignees**: Only respond when assigned to specific users
- Add **workflow_dispatch**: Allow manual triggering
- Modify **event types**: Respond to edits, not just new issues

### Access Control

The workflow includes a security check:

```yaml
if: >
  contains(fromJSON('["OWNER","MEMBER","COLLABORATOR"]'),
    github.event.issue.author_association)
```

Only repository owners, members, and collaborators can trigger the agent. This prevents random users from running up your API bill on public repos.

## Permissions

The workflow requests these GitHub permissions:

```yaml
permissions:
  contents: write    # Read/write repo files, push commits
  issues: write      # Post comments, add reactions
  actions: write     # Workflow management
```

Reduce permissions if your use case doesn't need them all. For a read-only agent, `contents: read` and `issues: write` (for posting replies) may suffice.

## Environment Variables

| Variable | Required | Source | Purpose |
|----------|----------|--------|---------|
| `ANTHROPIC_API_KEY` | Yes | Repository secret | LLM API authentication |
| `GITHUB_TOKEN` | Auto | GitHub Actions | Repo access, issue comments, reactions |

### Setting Up the API Key

1. Go to **Settings → Secrets and variables → Actions**
2. Create a secret named `ANTHROPIC_API_KEY`
3. Paste your Anthropic API key

## Agent Identity Files

These files shape behavior but aren't in `.pi/` itself:

| File | Location | Purpose |
|------|----------|---------|
| `AGENTS.md` | `.GITCLAW/AGENTS.md` | Primary identity — name, nature, vibe, instructions |
| `AGENTS.md` | repo root (optional) | Project-specific overrides/additions |

The agent reads `.GITCLAW/AGENTS.md` first, then checks for a root-level `AGENTS.md` for project-local additions. This lets you keep the core identity in `.GITCLAW/` while adding project-specific context at the repo root.

## Dependencies

**Location**: `.GITCLAW/package.json`

```json
{
  "dependencies": {
    "@mariozechner/pi-coding-agent": "^0.52.5"
  }
}
```

The `pi` coding agent is the core runtime. It's installed via `bun install` and provides the `pi` binary that `lifecycle/GITCLAW-AGENT.ts` invokes.

## Common Customization Recipes

### Read-Only Analysis Agent

Restrict tools so the agent can only read, not write:
```
--tools read,grep,find,ls
```

### Quick-Response Agent

Use a smaller model and lower thinking level for fast replies:
```json
{
  "defaultProvider": "anthropic",
  "defaultModel": "claude-sonnet-4-20250514",
  "defaultThinkingLevel": "low"
}
```

### Label-Gated Agent

Only respond to issues with a specific label by adjusting the workflow `if:` condition:
```yaml
if: >
  github.event.issue.labels.*.name contains 'agent'
```
