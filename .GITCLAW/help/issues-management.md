# 💬 Issues Management

[← Back to Help](README.md)

---

GitHub Issues are your primary interface for interacting with the GitClaw agent. Every issue becomes a persistent, multi-turn conversation with full memory and context.

## How Conversations Work

```
You open an issue
    → GitHub Actions triggers the agent workflow
    → The agent reads your issue title and body
    → It thinks, reasons, and formulates a response
    → Its reply appears as a comment (👀 shows while it's working)
    → The conversation session is committed to git

You comment on the same issue
    → The agent loads the full conversation history
    → It reads your new comment with all prior context
    → It responds as another comment
    → The updated session is committed to git
```

Each issue maps to exactly one conversation session. The agent remembers everything from previous exchanges in that issue.

## Starting a Conversation

1. Go to your repo's **Issues** tab
2. Click **New issue**
3. Write your request — ask a question, request a code change, start a discussion
4. Submit the issue

The agent picks it up automatically. You'll see:
- A 👀 reaction on the issue (indicates the agent is working)
- A comment with the agent's reply (once processing is complete)
- The 👀 reaction is removed when the agent finishes

## Continuing a Conversation

Comment on any existing issue to continue the conversation. The agent loads the full session history and responds with full awareness of everything discussed previously.

This works even if days or weeks have passed since the last interaction — the session file is stored in git and loaded on every run.

## Session Storage

All conversation state lives in the repository:

```
.GITCLAW/state/
  issues/
    1.json           # Maps issue #1 → its session file
    42.json          # Maps issue #42 → its session file
  sessions/
    2026-02-04T..._abc123.jsonl    # Full conversation for issue #1
    2026-02-10T..._def456.jsonl    # Full conversation for issue #42
```

### Issue mapping files (`state/issues/*.json`)

Each file maps an issue number to its session:

```json
{
  "issueNumber": 1,
  "sessionPath": ".GITCLAW/state/sessions/2026-02-04T12-00-00-000Z_abc123.jsonl",
  "updatedAt": "2026-02-04T12:00:00.000Z"
}
```

### Session files (`state/sessions/*.jsonl`)

JSONL (JSON Lines) files containing the complete conversation transcript. Each line is a structured event from the `pi` agent — messages, tool calls, reasoning traces, and more.

These files are plain text and can be inspected with standard tools:

```bash
# View the last few events in a session
tail -5 .GITCLAW/state/sessions/*.jsonl

# Search all sessions for a keyword
grep -r "search term" .GITCLAW/state/sessions/
```

## Personality Hatching

Use the **🥚 Hatch** issue template to give the agent a personality through a guided conversation:

1. Go to **Issues → New issue**
2. Select the **🥚 Hatch** template (or create an issue with the `hatch` label)
3. The agent walks you through choosing a name, personality, and behavioral style
4. The result is saved to `.GITCLAW/AGENTS.md`

Hatching is optional — the agent works without it, but it's more fun with a personality.

## What You Can Ask

GitClaw can handle a wide range of tasks through issues:

| Category | Examples |
|----------|---------|
| **Code changes** | "Add a login page", "Refactor the database module", "Fix the bug in auth.ts" |
| **Questions** | "How does the payment system work?", "What does this function do?" |
| **File operations** | "Create a README", "Organize the docs folder", "Set up a GitHub Pages site" |
| **Research** | "What are the best practices for API rate limiting?", "Compare REST vs GraphQL" |
| **Iterative work** | Start with "Set up a basic Express server", then follow up with "Add authentication", "Add tests", etc. |

The agent can read and write files in the repository, so changes it makes are committed and pushed automatically.

## Managing Multiple Issues

Each issue maintains its own independent conversation. You can have many active conversations running in parallel:

- **Issue #1**: Building a new feature
- **Issue #2**: Debugging a bug
- **Issue #3**: Writing documentation

Each one has its own session file and context. They don't interfere with each other.

> **Note:** If two issues trigger the agent simultaneously, the push-retry mechanism handles any git conflicts. The agent retries up to 3 times with `--rebase`.

## Clearing Conversation History

### For a single issue

Delete the mapping and session files:

```bash
rm .GITCLAW/state/issues/<number>.json
# Find and delete the corresponding session file
rm .GITCLAW/state/sessions/<timestamp>_<id>.jsonl

git add -A
git commit -m "Clear session for issue #<number>"
git push
```

The next comment on that issue will start a fresh conversation.

### For all issues

```bash
rm -rf .GITCLAW/state/sessions/*
rm -rf .GITCLAW/state/issues/*

git add -A
git commit -m "Clear all gitclaw sessions"
git push
```

## Authorization

Only users with **admin**, **maintain**, or **write** permission on the repository can trigger the agent. This is enforced by the Authorize step in the workflow.

Random users on public repositories cannot trigger the agent by opening issues or posting comments.

## Heart Emoji Requirement

GitClaw supports an optional heart emoji gate for new issues. When enabled (by renaming `.GITCLAW/GITCLAW-HEART-NOT-REQUIRED.md` to `.GITCLAW/GITCLAW-HEART-REQUIRED.md`), the agent only processes new issues that contain a ❤️ heart emoji in the issue body. Comments on existing issues are always processed.

See [❤️ Heart Requirement](heart-requirement.md) for full details.

## Comment Size Limits

GitHub enforces a ~65,535 character limit on issue comments. The agent caps its replies at 60,000 characters to stay within this limit. If a response is truncated, check the workflow logs for the full output.

## The 👀 Indicator

When the agent starts processing an issue or comment:
- A 👀 (eyes) reaction is added to the issue or comment
- This indicates the agent is actively working

When the agent finishes (or encounters an error):
- The 👀 reaction is removed
- This guarantees cleanup even if the agent crashes

If you see 👀 on an issue, the agent is still running. Check the **Actions** tab for live progress.

## Tips for Effective Conversations

- **Be specific** — "Add input validation to the signup form in `src/auth/signup.ts`" works better than "fix the form"
- **Provide context** — mention file paths, function names, or expected behavior
- **Iterate** — start simple and build up through follow-up comments
- **Use issues for different topics** — one issue per project or feature keeps context clean
- **Review the agent's changes** — the agent commits to git, so you can always `git diff` or revert

## See Also

- [⚙️ Configure](configure.md) — customize the agent's model and behavior
- [🚀 Action Management](action-management.md) — manage the workflow
- [🔧 Install](install.md) — first-time setup
