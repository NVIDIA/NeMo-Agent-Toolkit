# ⚙️ Configure GitClaw

[← Back to Help](README.md)

---

Customize the LLM provider, model, agent personality, thinking level, tool access, and workflow behavior.

## Configuration Files Overview

| File | Purpose |
|------|---------|
| `.GITCLAW/.pi/settings.json` | LLM provider, model, and thinking level |
| `.GITCLAW/AGENTS.md` | Agent identity — name, personality, instructions |
| `.GITCLAW/.pi/APPEND_SYSTEM.md` | System prompt loaded every session |
| `.GITCLAW/.pi/BOOTSTRAP.md` | First-run identity prompt (used during hatching) |
| `.GITCLAW/.pi/skills/` | Modular skill packages |
| `.github/workflows/GITCLAW-WORKFLOW-AGENT.yml` | Workflow triggers, permissions, and environment variables |

## Change the LLM Provider and Model

Edit `.GITCLAW/.pi/settings.json`:

```json
{
  "defaultProvider": "anthropic",
  "defaultModel": "claude-sonnet-4-20250514",
  "defaultThinkingLevel": "low"
}
```

### Supported Providers

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

> **Important:** After changing the provider, make sure the matching API key secret is added under **Settings → Secrets and variables → Actions** and referenced in your workflow file.

## Adjust Thinking Level

The `defaultThinkingLevel` controls how much reasoning the model does before responding:

| Level | Best For |
|-------|----------|
| `"low"` | Quick responses, simple tasks, lower cost |
| `"medium"` | Balanced reasoning for most use cases |
| `"high"` | Complex tasks requiring deep analysis |

```json
{
  "defaultThinkingLevel": "medium"
}
```

## Customize Agent Personality

Edit `.GITCLAW/AGENTS.md` to define the agent's identity, name, personality, and behavioral instructions. This file is read at the start of every session.

You can also use the **🥚 Hatch** issue template to set up a personality interactively — create an issue with the `hatch` label and the agent will guide you through the process.

## Modify the System Prompt

Edit `.GITCLAW/.pi/APPEND_SYSTEM.md` to change the system-level instructions loaded on every session. This file controls:

- How the agent introduces itself
- Core behavioral guidelines
- Memory system usage
- Session continuity rules

## Make the Agent Read-Only

To restrict the agent to read-only operations (no file edits, no git pushes of code changes), add `--tools read,grep,find,ls` to the agent arguments in `lifecycle/GITCLAW-AGENT.ts`.

This is useful when you want the agent to answer questions about the codebase without making modifications.

## Add or Modify Skills

Skills are modular capability packages stored as Markdown files in `.GITCLAW/.pi/skills/`. Each skill defines specific behaviors, instructions, or capabilities the agent can use.

To add a new skill, create a `.md` file in the skills directory. The agent automatically loads all skill files at session start.

## Manage API Key Secrets

API keys are stored as GitHub repository secrets and passed to the workflow as environment variables.

### Add or update a secret

1. Go to **Settings → Secrets and variables → Actions**
2. Click **New repository secret** (or edit an existing one)
3. Name it according to the provider (e.g., `ANTHROPIC_API_KEY`)
4. Paste your API key as the value

### Reference the secret in the workflow

In `.github/workflows/GITCLAW-WORKFLOW-AGENT.yml`, add the secret as an environment variable in the **Run** step:

```yaml
- name: Run
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: bun .GITCLAW/lifecycle/GITCLAW-AGENT.ts
```

If you switch providers, update this block to reference the new secret name.

## After Making Changes

Commit and push your configuration changes:

```bash
git add -A
git commit -m "Update gitclaw configuration"
git push
```

The next issue or comment will use the updated configuration.

## See Also

- [🔧 Install](install.md) — first-time setup
- [🚀 Action Management](action-management.md) — customize the workflow
- [💬 Issues Management](issues-management.md) — how conversations work
