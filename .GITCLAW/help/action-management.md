# 🚀 Action Management

[← Back to Help](README.md)

---

Manage the GitHub Actions workflow that powers GitClaw. This covers the workflow lifecycle, customization, debugging, and advanced configuration.

## The GitClaw Workflow

GitClaw runs as a GitHub Actions workflow defined in `.github/workflows/GITCLAW-WORKFLOW-AGENT.yml`. It triggers automatically when:

- A new issue is opened (`issues.opened`)
- A comment is added to an issue (`issue_comment.created`)

### Workflow Steps

Each workflow run executes these steps in order:

| Step | File | Purpose |
|------|------|---------|
| **Authorize** | (inline bash) | Verifies the actor has `admin`, `maintain`, or `write` permission |
| **Checkout** | `actions/checkout@v4` | Checks out the repository at the default branch |
| **Setup Bun** | `oven-sh/setup-bun@v2` | Installs the Bun runtime |
| **Guard** | `lifecycle/GITCLAW-ENABLED.ts` | Checks the sentinel file — exits if missing |
| **Preinstall** | `lifecycle/GITCLAW-INDICATOR.ts` | Adds the 👀 reaction to the issue |
| **Install** | `bun install` | Installs npm/bun dependencies |
| **Run** | `lifecycle/GITCLAW-AGENT.ts` | Runs the AI agent and posts the reply |

## View Workflow Runs

1. Go to your repository on GitHub
2. Click the **Actions** tab
3. Select **GITCLAW-WORKFLOW-AGENT** in the left sidebar
4. Click any run to see its logs

Each run shows which issue triggered it and the full execution log including the agent's reasoning.

## Enable and Disable the Workflow

### Via GitHub UI

1. Go to **Actions** tab
2. Select the workflow in the left sidebar
3. Click the **⋯** menu → **Disable workflow** (or **Enable workflow**)

### Via sentinel file

The workflow also respects the `.GITCLAW/GITCLAW-ENABLED.md` sentinel file. See [Disable](disable.md) and [Enable](enable.md) for details.

## Re-run a Failed Workflow

1. Go to the **Actions** tab
2. Click the failed run
3. Click **Re-run all jobs** (top right)

This is useful when a run failed due to a transient error (e.g., API timeout, rate limit).

## Customize Workflow Triggers

### Filter by label

To make the agent respond only to issues with a specific label, edit `.github/workflows/GITCLAW-WORKFLOW-AGENT.yml`:

```yaml
on:
  issues:
    types: [opened, labeled]
  issue_comment:
    types: [created]

jobs:
  run-agent:
    if: >-
      (github.event_name == 'issues' && contains(github.event.issue.labels.*.name, 'agent'))
      || (github.event_name == 'issue_comment'
          && github.event.comment.user.login != 'github-actions[bot]'
          && contains(github.event.issue.labels.*.name, 'agent'))
```

This example only triggers on issues labeled `agent`.

### Filter by user

To restrict the agent to specific users, add a condition to the `if` clause:

```yaml
if: >-
  github.actor == 'your-username'
  && (
    (github.event_name == 'issues')
    || (github.event_name == 'issue_comment' && github.event.comment.user.login != 'github-actions[bot]')
  )
```

## Customize Workflow Permissions

The default workflow requests these permissions:

```yaml
permissions:
  contents: write    # Push commits (session state, agent edits)
  issues: write      # Post comments and add reactions
  actions: write     # Manage workflow state
```

Reduce permissions if the agent doesn't need write access to certain resources.

## Add Environment Variables

To pass additional environment variables to the agent, add them to the **Run** step:

```yaml
- name: Run
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    CUSTOM_VAR: "my-value"
  run: bun .GITCLAW/lifecycle/GITCLAW-AGENT.ts
```

## Switch API Key Provider

When changing LLM providers, update the workflow to reference the correct secret:

```yaml
- name: Run
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}  # Changed from ANTHROPIC_API_KEY
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: bun .GITCLAW/lifecycle/GITCLAW-AGENT.ts
```

Also update `.GITCLAW/.pi/settings.json` to match (see [Configure](configure.md)).

## Regenerate the Workflow

If the workflow file gets corrupted or you want to reset it to the template:

```bash
rm .github/workflows/GITCLAW-WORKFLOW-AGENT.yml
bun .GITCLAW/install/GITCLAW-INSTALLER.ts
```

The installer copies the workflow template from `.GITCLAW/install/GITCLAW-WORKFLOW-AGENT.yml`.

## Troubleshooting

### Workflow never triggers

- Ensure the workflow file exists at `.github/workflows/GITCLAW-WORKFLOW-AGENT.yml`
- Check that the workflow is not disabled (Actions tab → select workflow → look for "Enable workflow" button)
- Verify the trigger events match your use case (`issues.opened`, `issue_comment.created`)

### Workflow fails at the Guard step

The sentinel file `.GITCLAW/GITCLAW-ENABLED.md` is missing. See [Enable](enable.md) to restore it.

### Workflow fails at the Authorize step

Only users with `admin`, `maintain`, or `write` permission can trigger the agent. Verify the triggering user's role under **Settings → Collaborators and teams**.

### Workflow fails at the Install step

- Check that `.GITCLAW/package.json` and `.GITCLAW/bun.lock` are present and valid
- Try running `cd .GITCLAW && bun install` locally to reproduce the error

### Agent runs but no comment appears

- Check the workflow logs for errors in the **Run** step
- Verify the API key secret exists and is correctly named
- Look for comment size limit issues (the agent caps replies at 60,000 characters)

### Push conflict errors

The agent retries pushes up to 3 times with `--rebase`. If conflicts persist, check for:
- Multiple issues triggering the agent simultaneously
- Manual pushes to the default branch during a run

## See Also

- [⚙️ Configure](configure.md) — LLM model and agent settings
- [💬 Issues Management](issues-management.md) — how conversations work
- [⏸️ Disable](disable.md) / [▶️ Enable](enable.md) — control the agent
