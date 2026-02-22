# .GITCLAW 🦞 The Idea

### An AI Agent That Lives in Your Repo

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

What if your repository wasn't just a place where code sleeps between deploys — but a living, breathing collaborator that *thinks*?

---

## The Elevator Pitch

Drop a single folder into any GitHub repository. Push. Open an issue. *Something answers.*

No servers. No databases. No infrastructure to provision, monitor, babysit, or pay for beyond what you already have. Just a folder called `.GITCLAW/`, a GitHub Actions workflow, and an LLM API key. That's the entire stack.

Your repo becomes sentient. Or at least, *conversant*.

---

## The Audacious Simplicity

Most AI agent platforms look like this: a SaaS dashboard, a database cluster, a queue system, webhook receivers, auth middleware, monitoring dashboards, and a pricing page that makes you wince. You deploy the agent somewhere *else*, and it reaches into your repo like a stranger rummaging through your filing cabinet.

`.GITCLAW` inverts this completely.

The agent doesn't live *outside* your repo and reach *in*. It lives **inside** your repo. Its memory is git commits. Its interface is GitHub Issues. Its compute is GitHub Actions. Its personality is a Markdown file. Its skills are more Markdown files. Everything is diffable, auditable, revertable, and forkable.

**The repository _is_ the application.**

---

## How It Works (The Beautiful Part)

1. **You open an issue.** Write anything — a question, a request, a half-formed thought.
2. **The agent wakes up.** GitHub Actions triggers a workflow. A 👀 emoji appears. Something is thinking.
3. **It responds.** A comment appears on your issue with a thoughtful, context-aware reply. It can read your entire codebase. It can write files. It can search its own memory.
4. **You reply.** Comment on the same issue. The conversation continues. The agent picks up exactly where it left off — not because it's running a server with session state, but because the conversation is *committed to git*.
5. **Everything is saved.** Sessions are JSONL files in `state/sessions/`. Issue-to-session mappings live in `state/issues/`. Every interaction is a commit. Every commit is history. History is memory.

The agent remembers because git remembers. It's memory you can `git log`, `git diff`, and `git blame`.

---

## The Trick That Makes It All Work

Here's the core insight, the elegant hack at the heart of `.GITCLAW`:

> **Each issue number is a stable conversation key.**
>
> `issue #N` → `state/issues/N.json` → `state/sessions/<session>.jsonl`

When you comment on issue #42 three weeks after opening it, the agent loads that linked session and continues. No database lookups. No session cookies. No Redis. Just a JSON pointer in a git-tracked file pointing to a git-tracked conversation log.

The repo is both the memory and the synchronization medium. That's the trick. That's the whole trick. And it's *brilliant*.

---

## What Makes It Special

### 🗂️ Git-Native State
Every conversation, every decision, every file the agent touches — it's all in git. You can review it in a PR. You can revert it. You can fork the entire agent's memory along with the code. Try doing that with ChatGPT.

### 🧩 Composable Behavior
The agent's behavior is decomposed into three orthogonal axes:
- **Skills** define *what* it can do (code review, release notes, triage)
- **Personality** defines *how* it does it (terse and analytical? warm and mentoring?)
- **Settings** define *which model powers it* (Claude, GPT, Gemini, Grok, DeepSeek, Mistral — your pick)

Mix and match. It's configuration, not code.

### 🔒 Fail-Closed Security
The agent does *nothing* unless a sentinel file (`GITCLAW-ENABLED.md`) exists. Delete it, and every workflow silently exits. Only repo owners, members, and collaborators can trigger it. Random drive-by users on public repos can't hijack your AI.

### 🥚 Personality Hatching
Use the "Hatch" issue template to have a conversation *with* the agent *about* the agent — choosing its name, its vibe, its emoji. It's a delightful onboarding ritual that turns setup into play.

### 🔄 Conflict-Resilient Persistence
Multiple issues being worked simultaneously? The agent retries pushes with `git pull --rebase`, handling concurrent writes gracefully. It's not distributed systems engineering — it's just *pragmatic git*.

---

## The Vision: Repository as Platform

`.GITCLAW` is a proof of concept for something larger than itself.

It proves that **AI agents don't need infrastructure**. Git + Actions + an API key is enough.

It proves that **conversations are data** — and they deserve the same versioning, auditability, and collaboration workflows that code gets.

It proves that **agent behavior is configuration** — files in a repo, managed with the same tools developers already use.

And it opens a combinatorial design space that's frankly dizzying:

- **Pull request reviews** where the agent posts structured `APPROVE` / `REQUEST_CHANGES` verdicts with line-level annotations
- **Automated triage** that labels, prioritizes, and routes issues based on content analysis
- **Release automation** that generates changelogs from merged PRs and drafts GitHub Releases
- **Security response** that triages CodeQL alerts, coordinates Dependabot updates, and triggers containment workflows for leaked secrets
- **Cross-repo agents** where a hub repository monitors an entire organization
- **Agent-to-agent conversations** where a triage bot opens issues that a review bot responds to
- **A skill marketplace** where the community publishes and discovers modular capabilities

Each new skill multiplies every existing one. Each new trigger multiplies every existing skill. The surface area grows combinatorially, but the complexity stays linear — because it's all just files in a folder.

---

## The Philosophical Bit

There's something poetic about an AI agent whose memory is `git log`.

Most AI systems treat conversation history as ephemeral — cached in RAM, maybe persisted to a database, eventually garbage-collected. `.GITCLAW` treats it as *source code*. Every exchange is a commit. Every commit has an author, a timestamp, a diff. You can trace the agent's evolution the same way you trace your codebase's evolution.

The agent doesn't just *use* your repository. It *is part of* your repository.

When you fork the repo, you fork the agent's mind. When you branch, you branch its reality. When you revert, you revert its memories. The metaphors aren't strained — they're *structural*.

---

## In One Sentence

`.GITCLAW` is the idea that a single folder, dropped into any GitHub repository, can turn issues into conversations with an AI agent that remembers everything, touches any file, and needs nothing but git and a willingness to try.

🦞 *The claw is the repo. The repo is the claw.*
