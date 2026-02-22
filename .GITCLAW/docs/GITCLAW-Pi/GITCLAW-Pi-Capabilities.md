# 🦞 GITCLAW .pi Capabilities

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

Everything `.pi` enables the agent to do — from basic conversation to self-modification.

## Conversational Intelligence

### Issue-Driven Conversations

The agent communicates through GitHub Issues. Every issue is a conversation thread:

- **Open an issue** → the agent reads the title and body, processes the request, and replies as a comment.
- **Comment on an issue** → the agent resumes the same session with full prior context and continues the conversation.
- The agent can handle multi-turn conversations across any number of comments on a single issue.

### Context Awareness

The agent reads its identity (`AGENTS.md`), system prompt (`APPEND_SYSTEM.md`), and any relevant skills before every response. When resuming a session, it has access to the entire prior conversation history, making it context-aware across multiple interactions.

## File Operations

### Reading

The agent can read any file in the repository:
- Source code, configuration, documentation
- Its own session logs and memory files
- Other agents' output and state files

### Writing

The agent can create and modify files:
- Write new code, documentation, or configuration
- Update existing files with targeted edits
- Create entire directory structures (e.g., scaffold a project)

### Searching

The agent has access to search tools:
- `grep`/`rg` for content search across files
- `find`/`ls` for file discovery
- Can search its own session history and memory logs

All file changes are committed to git after each session, giving you full version history and the ability to review or roll back any change.

## Persistent Memory

### Long-Term Memory (`memory.log`)

An append-only log file where the agent records important facts:

```
[2026-02-06 14:30] User prefers TypeScript over JavaScript.
[2026-02-07 09:15] Project uses PostgreSQL for the database.
[2026-02-07 11:00] Deploy target is Cloudflare Workers.
```

The agent writes to memory when:
- The user says "remember this" or "remember: X"
- Important preferences, decisions, or facts emerge
- Project context that future sessions need
- Corrections to previous assumptions

The agent searches memory at session start and whenever historical context would help.

### Session History

Full conversation transcripts are stored as JSONL files in `state/sessions/`. The agent can:
- Search past conversations with `rg`
- Extract specific messages (user, assistant, tool calls)
- Review what tools it used and what decisions it made
- Summarize or reference prior work

### Continuity Across Sessions

Each session starts fresh — no hidden state, no magic. Continuity comes entirely from files:
- `AGENTS.md` tells it who it is
- `APPEND_SYSTEM.md` tells it how to behave
- `memory.log` tells it what it knows
- Session files tell it what happened before

This makes the agent fully auditable. Everything it "remembers" is a file you can read.

## Skill System

Skills are modular packages that extend the agent's capabilities. See [GITCLAW-Pi-Skills.md](GITCLAW-Pi-Skills.md) for the full guide.

### Built-In Skills

| Skill | What it does |
|-------|-------------|
| **memory** | Search and recall information from past sessions and memory logs |
| **skill-creator** | Create, validate, and package new skills |

### What Skills Enable

- Domain-specific workflows (e.g., PDF processing, database queries)
- Tool integrations (e.g., working with specific file formats or APIs)
- Bundled scripts for deterministic, repeatable tasks
- Reference material loaded on demand to keep context efficient

## Self-Modification

### Identity Evolution

The agent can update its own identity and configuration files:
- Modify `AGENTS.md` to refine its personality or add instructions
- Update `APPEND_SYSTEM.md` (with user notification — "it's your soul")
- Write new memories to `memory.log`
- Create or update skills

### Project Evolution

Since the agent can read and write files, it can iteratively build and evolve a project:
- Set up a GitHub Pages site, then improve it issue by issue
- Scaffold a project structure, then add features through conversation
- Write and refine documentation based on ongoing discussion

## Tool Access

The agent has access to standard development tools through the `pi` coding agent:

- **Shell commands**: `bash`, standard Unix utilities
- **File operations**: read, write, create, delete, search
- **Git operations**: the agent's changes are committed automatically
- **GitHub CLI** (`gh`): interact with issues, PRs, and repo metadata

Tool access can be restricted in the workflow configuration. For example, setting `--tools read,grep,find,ls` creates a read-only agent that can analyze but not modify the repo.

## Security and Boundaries

### Access Control

The workflow only responds to repository **owners, members, and collaborators**. Random users cannot trigger the agent on public repos.

### Behavioral Boundaries

From the system prompt, the agent follows these rules:
- Private things stay private
- Ask before acting externally (emails, public posts)
- Never send half-baked replies to messaging surfaces
- Be careful in group contexts — the agent is not the user's voice

### Auditability

Every interaction is versioned in git:
- Session logs show exactly what the agent said, thought, and did
- File changes are visible in git history
- Any change can be reviewed or rolled back

## Configuration-Driven Behavior

The agent's behavior can be tuned through configuration (see [GITCLAW-Pi-Configuration.md](GITCLAW-Pi-Configuration.md)):

- **Model selection**: Choose the LLM provider and model
- **Thinking level**: Adjust reasoning depth (low, medium, high)
- **Tool restrictions**: Limit which tools the agent can use
- **Trigger filtering**: Control which issues/comments activate the agent
