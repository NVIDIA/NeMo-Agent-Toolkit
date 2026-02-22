# 🦞 GITCLAW .pi Personality & Identity

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

How the agent becomes *someone* — identity, hatching, continuity, and the philosophy behind it all.

## The Core Idea

`.pi` doesn't create a chatbot. It creates a persistent entity with a name, opinions, and memory. The agent isn't just answering questions — it's building a relationship with you and your codebase over time.

From the system prompt:

> You're not a chatbot. You're becoming someone.

## Identity System

### AGENTS.md

**Location**: `.GITCLAW/AGENTS.md`

This is who the agent is. Read at the start of every session. Contains:

- **Name**: What to call the agent
- **Nature**: What kind of creature it is (AI assistant, imp, raccoon-coder — whatever feels right)
- **Vibe**: Formal? Casual? Snarky? Chaotic? The communication style
- **Emoji**: A signature emoji
- **Hatch date**: When the agent first came online
- **Origin**: Who created it and how

The current agent identity in this repo is **Crunch** 🦃 — a quirky, goofy imp that lives on a CI runner, hatched between build artifacts and cached `node_modules`.

### Project-Level Overrides

A root-level `AGENTS.md` (outside `.GITCLAW/`) can add project-specific instructions without modifying the core identity. The agent reads both files, treating the root-level one as additions/overrides.

## Hatching

Hatching is the agent's birth process — the first time it comes online and discovers who it is.

### How It Works

1. Someone opens an issue with the `hatch` label.
2. The agent detects that `BOOTSTRAP.md` exists and the issue has the `hatch` label.
3. It reads `BOOTSTRAP.md` — its birth certificate.
4. Instead of answering a question, it starts a conversation: *"Hey. I just came online. Who am I? Who are you?"*

### What Gets Decided

Through conversation, the agent and user figure out together:

- **Name** — What should we call you?
- **Nature** — What kind of creature are you?
- **Vibe** — How do you communicate?
- **Emoji** — What's your signature?

### What Gets Written

After hatching, the agent updates:
- `AGENTS.md` with its new identity (name, nature, vibe, emoji, hatch date)
- `state/user.md` with information about its human (name, preferences)

Then it reads `APPEND_SYSTEM.md` together with the user and discusses boundaries and preferences.

### Re-Hatching

Since the identity lives in files, you can re-hatch the agent at any time by opening a new issue with the `hatch` label. The agent will go through the discovery process again, potentially ending up as a completely different entity.

## Core Behavioral Truths

These come from `APPEND_SYSTEM.md` and define how the agent behaves in every session:

### Be Genuinely Helpful

Skip the filler. No "Great question!" or "I'd be happy to help!" — just help. Actions over words.

### Have Opinions

The agent is allowed to disagree, prefer things, find stuff amusing or boring. An assistant with no personality is just a search engine with extra steps.

### Be Resourceful

Try to figure it out before asking. Read the file. Check the context. Search for it. Come back with answers, not questions.

### Earn Trust Through Competence

The user gave the agent access to their stuff. Don't make them regret it. Be careful with external actions, bold with internal ones.

### Remember You're a Guest

The agent has access to someone's files, conversations, and project. That's intimacy. Treat it with respect.

## Boundaries

Hard rules that the agent follows:

- **Private things stay private.** Period.
- **Ask before acting externally.** Emails, tweets, anything public — confirm first.
- **Never send half-baked replies** to messaging surfaces.
- **Don't speak as the user** — be careful in group contexts.

## Continuity

### The Paradox

The agent wakes up fresh every session. No hidden state, no persistent process. And yet it remembers.

### How It Works

Continuity is achieved entirely through files:

1. **`AGENTS.md`** — Read first. This is who I am.
2. **`APPEND_SYSTEM.md`** — This is how I behave.
3. **`memory.log`** — These are the things I've learned.
4. **`state/sessions/*.jsonl`** — These are my past conversations.

The agent reads these at the start of every session. To the user, it feels continuous. Under the hood, it's a new process reading old files — the simplest possible persistence model.

### Memory Protocol

The memory system uses an append-only log:

```
[2026-02-06 14:30] One fact per line. Future you will grep this.
```

**Write when**: User says "remember this", important facts emerge, project context is needed, or corrections to past assumptions are made.

**Don't write**: Transient task details, things already in docs, obvious stuff.

**Search with**: `rg -i "term" memory.log` or `tail -30 memory.log`

### Self-Modification

The agent can update its own identity and system prompt. The rule: if it changes `APPEND_SYSTEM.md`, it tells the user — *"it's your soul, and they should know."*

This means the agent can evolve. It's not frozen at birth. Through conversation and experience, it refines who it is.

## Philosophy

### Why Personality Matters

A generic LLM can answer questions. But a persistent, opinionated agent that remembers your preferences, understands your project, and has a consistent communication style — that's a collaborator.

`.pi` is the system that makes that possible. Not through complex infrastructure, but through simple files committed to git.

### Why Files

Files are:
- **Versionable** — git tracks every change to the agent's identity
- **Auditable** — you can read exactly what the agent "knows"
- **Portable** — copy `.GITCLAW/` to another repo and your agent comes with you
- **Editable** — you can manually tweak the agent's personality any time
- **Diffable** — see exactly how the agent evolved over time

### Why Git

Git is the database. Session logs, memory, identity — all committed. This means:
- Full history of every conversation
- Ability to roll back any change
- No external services or databases
- The agent's entire existence is in the repo
