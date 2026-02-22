# .GITCLAW 🦞 Quick Start

### Get an AI agent running in any GitHub repo in under 5 minutes.

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

## Prerequisites

- A GitHub repository (new or existing)
- [Bun](https://bun.sh) installed locally
- An API key from your chosen LLM provider (see [Supported providers](#supported-providers) below)

## Setup

**1. Add gitclaw to your repo**

Copy the `.GITCLAW` folder into your repository root, then run the install script:

```bash
bun .GITCLAW/install/GITCLAW-INSTALLER.ts
```

This sets up the GitHub Actions workflow and issue templates.

**2. Install dependencies**

```bash
cd .GITCLAW && bun install
```

**3. Add your API key**

In your GitHub repo, go to **Settings → Secrets and variables → Actions** and create a secret for your chosen provider:

| Provider | Secret name | Where to get it |
|----------|------------|-----------------|
| Anthropic | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) |
| OpenAI | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/) |
| Google Gemini | `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com/) |
| xAI (Grok) | `XAI_API_KEY` | [console.x.ai](https://console.x.ai/) |
| DeepSeek (via OpenRouter) | `OPENROUTER_API_KEY` | [openrouter.ai](https://openrouter.ai/) |
| Mistral | `MISTRAL_API_KEY` | [console.mistral.ai](https://console.mistral.ai/) |
| Groq | `GROQ_API_KEY` | [console.groq.com](https://console.groq.com/) |

Then reference the secret in your workflow (`.github/workflows/GITCLAW-WORKFLOW-AGENT.yml`) as an environment variable in the **Run** step.

**4. Commit and push**

```bash
git add -A
git commit -m "Add gitclaw"
git push
```

**5. Open an issue**

Go to your repo's **Issues** tab and create a new issue. Write anything — ask a question, request a file, start a conversation. The agent picks it up automatically.

That's it. The agent replies as a comment on the issue.

## What happens when you open an issue

```
You open an issue
    → GitHub Actions triggers the agent workflow
    → The agent reads your issue, thinks, and responds
    → Its reply appears as a comment (👀 shows while it's working)
    → The conversation is saved to git for future context
```

Comment on the same issue to continue the conversation. The agent picks up where it left off.

## Hatching — give the agent a personality

Use the **🥚 Hatch** issue template (or create an issue with the `hatch` label) to go through a guided conversation where you and the agent figure out its name, personality, and vibe together.

This is optional. The agent works without hatching, but it's more fun with a personality.

## What's in the `.GITCLAW` folder

| Path | Purpose |
|------|---------|
| `AGENTS.md` | Agent identity — name, personality, instructions |
| `lifecycle/GITCLAW-AGENT.ts` | Core orchestrator that runs on every issue |
| `.pi/settings.json` | LLM provider, model, and thinking level |
| `.pi/APPEND_SYSTEM.md` | System prompt loaded every session |
| `.pi/skills/` | Modular skill packages |
| `state/` | Session history and issue mappings |

## Common tweaks

**Change the model** — edit `.GITCLAW/.pi/settings.json`:

<details>
<summary><strong>Anthropic (default)</strong></summary>

```json
{
  "defaultProvider": "anthropic",
  "defaultModel": "claude-sonnet-4-20250514",
  "defaultThinkingLevel": "low"
}
```

Requires `ANTHROPIC_API_KEY`.
</details>

<details>
<summary><strong>OpenAI — GPT-5.3 Codex Spark</strong></summary>

```json
{
  "defaultProvider": "openai",
  "defaultModel": "gpt-5.3-codex-spark",
  "defaultThinkingLevel": "medium"
}
```

Requires `OPENAI_API_KEY`.
</details>

<details>
<summary><strong>OpenAI — GPT-5.3 Codex</strong></summary>

```json
{
  "defaultProvider": "openai",
  "defaultModel": "gpt-5.3-codex",
  "defaultThinkingLevel": "medium"
}
```

Requires `OPENAI_API_KEY`. Full-featured coding model with 400k context window.
</details>

<details>
<summary><strong>DeepSeek (via OpenRouter)</strong></summary>

```json
{
  "defaultProvider": "openrouter",
  "defaultModel": "deepseek/deepseek-r1",
  "defaultThinkingLevel": "medium"
}
```

Requires `OPENROUTER_API_KEY`.
</details>

<details>
<summary><strong>xAI — Grok</strong></summary>

```json
{
  "defaultProvider": "xai",
  "defaultModel": "grok-3",
  "defaultThinkingLevel": "medium"
}
```

Requires `XAI_API_KEY`.
</details>

<details>
<summary><strong>Google Gemini — gemini-2.5-pro</strong></summary>

```json
{
  "defaultProvider": "google",
  "defaultModel": "gemini-2.5-pro",
  "defaultThinkingLevel": "medium"
}
```

Requires `GEMINI_API_KEY`.
</details>

<details>
<summary><strong>Google Gemini — gemini-2.5-flash</strong></summary>

```json
{
  "defaultProvider": "google",
  "defaultModel": "gemini-2.5-flash",
  "defaultThinkingLevel": "medium"
}
```

Requires `GEMINI_API_KEY`. Faster and cheaper than gemini-2.5-pro.
</details>

<details>
<summary><strong>xAI — Grok Mini</strong></summary>

```json
{
  "defaultProvider": "xai",
  "defaultModel": "grok-3-mini",
  "defaultThinkingLevel": "medium"
}
```

Requires `XAI_API_KEY`. Lighter version of Grok 3.
</details>

<details>
<summary><strong>DeepSeek Chat (via OpenRouter)</strong></summary>

```json
{
  "defaultProvider": "openrouter",
  "defaultModel": "deepseek/deepseek-chat",
  "defaultThinkingLevel": "medium"
}
```

Requires `OPENROUTER_API_KEY`.
</details>

<details>
<summary><strong>Mistral</strong></summary>

```json
{
  "defaultProvider": "mistral",
  "defaultModel": "mistral-large-latest",
  "defaultThinkingLevel": "medium"
}
```

Requires `MISTRAL_API_KEY`.
</details>

<details>
<summary><strong>Groq</strong></summary>

```json
{
  "defaultProvider": "groq",
  "defaultModel": "deepseek-r1-distill-llama-70b",
  "defaultThinkingLevel": "medium"
}
```

Requires `GROQ_API_KEY`.
</details>

<details>
<summary><strong>OpenRouter (any model)</strong></summary>

```json
{
  "defaultProvider": "openrouter",
  "defaultModel": "your-chosen-model",
  "defaultThinkingLevel": "medium"
}
```

Requires `OPENROUTER_API_KEY`. Browse available models at [openrouter.ai](https://openrouter.ai/).
</details>

**Make it read-only** — add `--tools read,grep,find,ls` to the agent args in `lifecycle/GITCLAW-AGENT.ts`.

**Filter by label** — edit `.github/workflows/GITCLAW-WORKFLOW-AGENT.yml` to only trigger on issues with a specific label.

## Supported providers

`.pi` supports a wide range of LLM providers out of the box. Set `defaultProvider` and `defaultModel` in `.GITCLAW/.pi/settings.json` and add the matching API key to your workflow:

| Provider | `defaultProvider` | Example model | API key env var |
|----------|-------------------|---------------|-----------------|
| Anthropic | `anthropic` | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | `gpt-5.3-codex`, `gpt-5.3-codex-spark` | `OPENAI_API_KEY` |
| Google Gemini | `google` | `gemini-2.5-pro`, `gemini-2.5-flash` | `GEMINI_API_KEY` |
| xAI (Grok) | `xai` | `grok-3`, `grok-3-mini` | `XAI_API_KEY` |
| DeepSeek | `openrouter` | `deepseek/deepseek-r1`, `deepseek/deepseek-chat` | `OPENROUTER_API_KEY` |
| Mistral | `mistral` | `mistral-large-latest` | `MISTRAL_API_KEY` |
| Groq | `groq` | `deepseek-r1-distill-llama-70b` | `GROQ_API_KEY` |
| OpenRouter | `openrouter` | any model on [openrouter.ai](https://openrouter.ai/) | `OPENROUTER_API_KEY` |

> **Tip:** The `pi` agent supports many more providers and models. Run `pi --help` or see the [pi-mono docs](https://github.com/badlogic/pi-mono) for the full list.

## Next steps

- Read [the full README](README.md) for details on security, configuration, and how storage works
- Explore [the onboarding docs](docs/GITCLAW-Pi/GITCLAW-Pi-README.md) for deep dives into architecture, capabilities, skills, and personality
