# .GITCLAW 🦞 Loves Pi 🥧

### How a minimal terminal coding agent became the engine of something extraordinary

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

---

## What Is Pi?

[Pi](https://github.com/badlogic/pi-mono) is a minimal terminal coding harness created by [Mario Zechner](https://github.com/badlogic). That sentence undersells it by an order of magnitude.

Pi is what happens when someone looks at the entire landscape of AI coding agents — the bloated SaaS platforms, the framework-of-the-week churn, the opinionated monoliths that insist you work *their* way — and says: **no**. What if the agent was small, composable, radically extensible, and got out of your way?

The result is a tool that ships with exactly four built-in tools (`read`, `write`, `edit`, `bash`) and an extension system so powerful that *everything else* — sub-agents, plan mode, permission gates, MCP support, custom compaction, even Doom — can be built on top without touching pi's internals.

Pi doesn't have features. It has *the ability to grow any feature you need*.

---

## The Architecture of Restraint

Most agent frameworks are additive: they start with everything and let you turn things off. Pi is subtractive: it starts with almost nothing and lets you build up.

### The Core

At its heart, pi is a conversation loop. You talk to an LLM. The LLM can call tools. Tools produce results. The cycle continues. Sessions are saved as JSONL files with a tree structure — every message has an `id` and `parentId`, enabling in-place branching without creating new files.

That's it. That's the core.

### The Extension Points

Everything else is plugged in through four orthogonal systems:

| System | What It Does | How It's Shared |
|--------|-------------|-----------------|
| **Extensions** | TypeScript modules that add tools, commands, keyboard shortcuts, event handlers, and full TUI components | npm or git via Pi Packages |
| **Skills** | On-demand capability packages following the [Agent Skills standard](https://agentskills.io) — specialized workflows as Markdown files | npm or git via Pi Packages |
| **Prompt Templates** | Reusable prompts with variable interpolation, invoked via `/name` | npm or git via Pi Packages |
| **Themes** | Visual customization of the terminal UI, with hot-reload | npm or git via Pi Packages |

The pattern is consistent: define a thing as a file, put it in the right directory, and pi discovers it. Bundle multiple things into a Pi Package and share it with the world via `pi install`.

### What Pi Deliberately Doesn't Have

This is the part that reveals the philosophy:

- **No MCP.** Build a CLI tool with a README, or write an extension that adds MCP support.
- **No sub-agents.** Spawn pi instances via tmux, or build your own orchestration with extensions.
- **No permission popups.** Run in a container, or build your own confirmation flow.
- **No plan mode.** Write plans to files, or build it with extensions.
- **No built-in to-dos.** They confuse models. Use a `TODO.md` file.
- **No background bash.** Use tmux. Full observability, direct interaction.

Every missing feature is a *design decision*. Pi trusts you to know what you need and gives you the tools to build it yourself. The agent adapts to your workflow — not the other way around.

---

## The Four Modes

Pi runs in four modes, each serving a different integration pattern:

| Mode | Use Case |
|------|----------|
| **Interactive** | You sit at a terminal and have a conversation. Full TUI with editor, message history, model switching, session branching. |
| **Print / JSON** | One-shot: send a prompt, get a response, exit. JSON mode emits structured events as JSONL — perfect for piping into other tools. |
| **RPC** | Stdin/stdout protocol for non-Node.js integrations. Speak JSON, get JSON back. |
| **SDK** | Embed pi directly in your TypeScript application. Full programmatic control over sessions, tools, events, and rendering. |

**GitClaw uses JSON mode and the SDK.** The lifecycle orchestrator (`GITCLAW-AGENT.ts`) spawns pi with `--mode json`, streams structured events, extracts the final assistant response, and posts it back to the GitHub issue. The full power of pi — tool calling, session management, thinking levels — is available through a clean, parseable interface.

---

## Why Pi Is Perfect for GitClaw

GitClaw needed an agent engine with very specific properties:

### 1. Session Persistence That Just Works

Pi stores sessions as JSONL files. GitClaw stores those files in git. The combination gives the agent persistent memory across workflow runs without a database, a cache, or any external service. When a user comments on issue #42 three weeks later, GitClaw loads the session file from `state/sessions/`, passes it to pi via `--session`, and the conversation continues seamlessly.

Pi's session model wasn't designed for GitClaw — but it fits like it was.

### 2. Headless Execution

Pi doesn't assume a human is sitting at a terminal. JSON mode and the SDK emit structured events that any consumer can process. GitClaw runs in a GitHub Actions runner with no TTY, no keyboard, no screen. Pi doesn't care. It produces the same quality output whether you're watching it think in real-time or capturing its output in a CI log.

### 3. Provider Agnosticism

Pi supports a staggering number of LLM providers out of the box:

- Anthropic (Claude)
- OpenAI (GPT, Codex)
- Google (Gemini)
- xAI (Grok)
- DeepSeek (via OpenRouter)
- Mistral
- Groq
- Cerebras
- Amazon Bedrock
- Azure OpenAI
- Google Vertex
- Hugging Face
- ...and any OpenAI/Anthropic/Google-compatible API via custom provider config

GitClaw inherits all of this for free. Change a line in `.pi/settings.json` and your repo's agent switches from Claude to Gemini to Grok without touching a single workflow file. The agent is model-agnostic because pi is model-agnostic.

### 4. The Skill System

Pi's skill system is the natural extension point for GitClaw's capabilities. Each skill is a self-contained Markdown file — instructions, workflows, and context that the agent loads on demand. GitClaw can ship skills for code review, release notes, triage, security scanning, and anything else, all as simple Markdown files in `.pi/skills/`.

Skills follow the [Agent Skills standard](https://agentskills.io), which means skills written for pi work in other harnesses too — and vice versa. It's an open ecosystem, not a walled garden.

### 5. Extensibility Without Forking

The extension system means GitClaw can customize pi's behavior deeply — custom tools, event interception, state management, custom compaction — without maintaining a fork. When pi releases a new version, GitClaw picks it up. When GitClaw needs a new capability, it writes an extension. The boundary is clean.

---

## The Extension System (The Crown Jewel)

Pi's extension system deserves its own section because it's genuinely remarkable in its scope.

An extension is a TypeScript file that receives an `ExtensionAPI` object and can do *anything*:

```typescript
export default function (pi: ExtensionAPI) {
  // Register a tool the LLM can call
  pi.registerTool({ name: "deploy", ... });

  // Register a command the user can invoke
  pi.registerCommand("stats", { ... });

  // Intercept tool calls before they execute
  pi.on("tool_call", async (event, ctx) => {
    if (isDangerous(event)) return { block: true, reason: "Nope" };
  });

  // React to lifecycle events
  pi.on("session_start", async (_event, ctx) => { ... });

  // Store persistent state in the session
  pi.appendEntry({ type: "custom", data: myState });
}
```

Extensions can:

- **Add custom tools** — or replace built-in tools entirely
- **Intercept and modify tool calls** — permission gates, logging, sandboxing
- **Customize compaction** — control how long conversations get summarized
- **Build full TUI components** — custom editors, overlays, status lines, interactive dialogs
- **Manage persistent state** — entries survive session restarts
- **Register keyboard shortcuts** — integrate into the interactive workflow
- **Render custom content** — control how tool calls and results display

The examples directory includes extensions for git checkpointing, path protection, conversation summarization, interactive Q&A tools, and yes — Snake and Doom running inside the terminal while you wait for the agent to think.

This isn't a plugin system. It's a *platform*.

---

## The Session Tree (Branching Without Branches)

One of pi's most elegant design choices is session branching. Every message in a session has an `id` and `parentId`, forming a tree structure — all in a single JSONL file. You can:

- **`/tree`** — Navigate the full session tree visually, jump to any point, and continue from there
- **`/fork`** — Create a new session file from any branch point
- **`/compact`** — Summarize older messages while keeping recent ones, with full history preserved for revisiting

For GitClaw, this means a single issue's conversation can have implicit branches — the agent explored one approach, the user asked it to try another, and both paths are preserved in the session file. Git versioning adds another layer: you can see how the session evolved over time via `git log`.

---

## Compaction (Infinite Conversations)

Long conversations eventually exhaust context windows. Pi handles this with compaction — a summarization pass that condenses older messages while keeping recent ones verbatim.

Compaction can be:
- **Manual** — invoke with `/compact` or `/compact <custom instructions>`
- **Automatic** — triggers on context overflow (recovers and retries) or proactively when approaching the limit

The full uncompacted history remains in the JSONL file. Compaction is lossy for the LLM's context but lossless for the human record.

For GitClaw, automatic compaction means conversations can span weeks or months without hitting context limits. The agent gracefully compresses its earlier memories while retaining full access to recent context.

---

## The Numbers

A quick snapshot of pi's reach:

| Dimension | Scope |
|-----------|-------|
| **Built-in providers** | 17+ (Anthropic, OpenAI, Google, xAI, Mistral, Groq, Bedrock, Vertex, OpenRouter, and more) |
| **Authentication** | API keys *and* OAuth subscriptions (Claude Pro/Max, ChatGPT Plus/Pro, Copilot, Gemini CLI) |
| **Built-in tools** | 7 (`read`, `write`, `edit`, `bash`, `grep`, `find`, `ls`) — selectable per session |
| **Extension capabilities** | Tools, commands, keyboard shortcuts, event handlers, TUI components, state management, custom rendering |
| **Skill standard** | [agentskills.io](https://agentskills.io) — interoperable across harnesses |
| **Session format** | Tree-structured JSONL with branching, compaction, and full history preservation |
| **Package ecosystem** | npm and git-based installation, sharing, and versioning |

---

## What This Means for GitClaw

GitClaw's ambition is to be an AI agent that participates in every layer of the GitHub platform — issues, pull requests, reviews, checks, releases, security scanning, project management. That's a lot of surface area.

Pi makes this tractable because:

1. **New capabilities are skills and extensions**, not core changes. Adding PR review support means writing a skill file and maybe an extension — not rewriting the agent.
2. **Provider flexibility** means teams aren't locked in. Use Claude for complex reasoning, GPT for speed, Gemini for cost optimization — switch with a config change.
3. **Session persistence is a solved problem.** Pi's JSONL sessions + GitClaw's git storage = durable, auditable, resumable conversations with zero operational overhead.
4. **The SDK enables deeper integration** when JSON mode isn't enough. Future GitClaw features can embed pi sessions programmatically, spawn sub-agents, or build custom tool orchestration.
5. **The community compounds.** Every pi extension, skill, prompt template, and theme is potentially useful to GitClaw — and vice versa. The ecosystem grows in both directions.

---

## The Philosophy Match

Pi's philosophy is: *adapt the agent to your workflow, not the other way around.*

GitClaw's philosophy is: *the repository is the platform, and the agent lives inside it.*

These two ideas are deeply complementary. Pi provides the engine — minimal, extensible, provider-agnostic. GitClaw provides the chassis — git-native state, issue-driven conversation, zero-infrastructure deployment. Together, they prove that a powerful AI coding agent doesn't need a server farm, a subscription dashboard, or a venture-backed startup behind it.

It just needs a folder, a conversation, and a good foundation to build on.

Pi is that foundation.

---

## Links

- **Pi repository:** [github.com/badlogic/pi-mono](https://github.com/badlogic/pi-mono)
- **Pi on npm:** [@mariozechner/pi-coding-agent](https://www.npmjs.com/package/@mariozechner/pi-coding-agent)
- **Agent Skills standard:** [agentskills.io](https://agentskills.io)
- **Pi blog post:** [mariozechner.at/posts/2025-11-30-pi-coding-agent](https://mariozechner.at/posts/2025-11-30-pi-coding-agent/)
- **GitClaw README:** [README.md](../README.md)
- **GitClaw Possibilities:** [GITCLAW-Possibilities.md](GITCLAW-Possibilities.md)

---

*GitClaw is the claw. Pi is the muscle behind it.* 🦞🥧
