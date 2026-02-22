# 🦞 GITCLAW .pi Architecture

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

How the pieces fit together — from files on disk to a running agent session.

## Directory Structure

```
.GITCLAW/.pi/
├── settings.json        # LLM defaults (provider, model, thinking level)
├── BOOTSTRAP.md         # First-run identity discovery script
├── APPEND_SYSTEM.md     # System prompt appended every session
└── skills/              # Modular capability packages
    ├── memory/
    │   └── SKILL.md     # Search and recall past sessions
    └── skill-creator/
        ├── SKILL.md     # Guide for creating new skills
        ├── license.txt
        └── scripts/     # Scaffolding and packaging tools
```

## How .pi Connects to gitclaw

```
GitHub Issue → GitHub Actions → lifecycle/GITCLAW-AGENT.ts → pi agent → .pi config
                                      │                          │
                                      │                          ├── reads settings.json
                                      │                          ├── reads APPEND_SYSTEM.md
                                      │                          ├── reads AGENTS.md
                                      │                          └── activates skills/
                                      │
                                      └── commits session to state/sessions/
```

1. A user opens an issue or comments on one.
2. GitHub Actions triggers the `GITCLAW-WORKFLOW-AGENT.yml` workflow.
3. `lifecycle/GITCLAW-AGENT.ts` resolves the session (new or resumed), builds a prompt, and launches the `pi` agent binary.
4. The `pi` agent reads `.pi/settings.json` for LLM config, loads `.pi/APPEND_SYSTEM.md` as the system prompt, and activates relevant skills.
5. The agent reads `.GITCLAW/AGENTS.md` to learn who it is.
6. After the agent responds, `GITCLAW-AGENT.ts` commits the session log, pushes changes, and posts the reply as an issue comment.

## Session Lifecycle

### New Session

When no prior session exists for an issue:

1. `GITCLAW-AGENT.ts` checks `state/issues/{number}.json` — not found.
2. A new session starts; the agent has no prior conversation context.
3. After completion, the session file lands in `state/sessions/` and a mapping is written to `state/issues/{number}.json`.

### Resumed Session

When a session already exists:

1. `GITCLAW-AGENT.ts` finds the mapping in `state/issues/{number}.json`.
2. The existing session file is passed via `--session` to the `pi` agent.
3. The agent picks up the full prior conversation and continues.

### Session Storage

```
.GITCLAW/state/
├── issues/
│   └── 1.json              # Maps issue #1 → session file path
└── sessions/
    └── 2026-02-04T..._abc123.jsonl   # Full JSONL conversation log
```

Session files are append-only JSONL. Each line is a JSON object with `type`, `timestamp`, and `message` fields containing the role (`user`, `assistant`, `toolResult`) and content blocks (text, thinking, tool calls).

## The Three Layers of .pi

### Layer 1: System Prompt (`APPEND_SYSTEM.md`)

Loaded every session. Defines:
- Core behavioral truths (be helpful, have opinions, be resourceful)
- Boundaries (privacy, external actions, group chats)
- Continuity instructions (read identity files, update memory)
- Memory system protocol (format, when to read/write)

### Layer 2: Identity (`AGENTS.md`)

Read at the start of every session. Contains:
- Agent name, nature, vibe, emoji
- Hatch date and origin story
- Project-specific instructions and guides

### Layer 3: Skills (`skills/`)

Modular packages that extend capabilities. Each skill has:
- A `SKILL.md` with metadata (name, description) that determines when it activates
- Optional bundled resources (scripts, references, assets)
- Skills use progressive disclosure — metadata is always in context, the body loads only when triggered

## Data Flow

```
User writes issue/comment
        │
        ▼
GitHub Actions triggers workflow
        │
        ▼
GITCLAW-ENABLED.ts verifies opt-in sentinel (fail-closed guard)
        │
        ▼
GITCLAW-INDICATOR.ts adds 👀 reaction
        │
        ▼
GITCLAW-AGENT.ts resolves session
        │
        ▼
pi agent starts ← settings.json (model config)
        │         ← APPEND_SYSTEM.md (system prompt)
        │         ← AGENTS.md (identity)
        │         ← skills/ (capabilities)
        │         ← state/sessions/ (prior context if resuming)
        │
        ▼
Agent reads files, runs tools, generates response
        │
        ▼
Session appended to state/sessions/*.jsonl
        │
        ▼
GITCLAW-AGENT.ts commits all changes to git
        │
        ▼
GITCLAW-AGENT.ts posts reply as issue comment
        │
        ▼
👀 reaction removed
```

## Key Design Decisions

- **Stateless agent, stateful repo**: The agent process is ephemeral. All persistence happens through git commits. The repo *is* the database.
- **Identity as files**: The agent's personality isn't hardcoded — it's in markdown files that can be versioned, diffed, and evolved.
- **Skills as progressive disclosure**: Skill metadata is cheap (always loaded), skill bodies are loaded on demand, and bundled resources are loaded only when needed. This keeps the context window efficient.
- **Session-per-issue model**: Each GitHub issue maps to exactly one session file, enabling clean conversation threading and history.
