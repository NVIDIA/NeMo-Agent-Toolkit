/**
 * GITCLAW-INDICATOR.ts — Adds a 👀 reaction to signal that the agent is working.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * PURPOSE
 * ─────────────────────────────────────────────────────────────────────────────
 * This script serves as the "activity indicator" for GitClaw.  It runs
 * *before* dependency installation (hence "pre-install") so that users
 * receive immediate visual feedback on the triggering issue or comment —
 * in the form of a 👀 (eyes) emoji reaction — the moment the workflow starts.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * LIFECYCLE POSITION
 * ─────────────────────────────────────────────────────────────────────────────
 * Workflow step order:
 *   1. Guard       (GITCLAW-ENABLED.ts)   — verify opt-in sentinel exists
 *   2. Preinstall  (GITCLAW-INDICATOR.ts) ← YOU ARE HERE
 *   3. Install     (bun install)            — install npm/bun dependencies
 *   4. Run         (GITCLAW-AGENT.ts)      — execute the AI coding agent
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * REACTION STATE HANDOFF
 * ─────────────────────────────────────────────────────────────────────────────
 * After adding the reaction this script persists the reaction metadata
 * (reaction ID, target type, comment ID if applicable) to a temporary JSON
 * file at `/tmp/reaction-state.json`.
 *
 * `GITCLAW-AGENT.ts` reads that file in its `finally` block and uses the
 * stored IDs to DELETE the 👀 reaction once the agent finishes — regardless
 * of whether the run succeeded or failed.  This guarantees the indicator is
 * always cleaned up.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * EVENT HANDLING
 * ─────────────────────────────────────────────────────────────────────────────
 * The script handles two GitHub event types differently:
 *
 *   issue_comment  → reacts to the COMMENT that triggered the workflow
 *                    (uses the /issues/comments/{id}/reactions endpoint)
 *   issues (opened)→ reacts to the ISSUE itself
 *                    (uses the /issues/{number}/reactions endpoint)
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * FAULT TOLERANCE
 * ─────────────────────────────────────────────────────────────────────────────
 * Failures to add the reaction are caught and logged but do NOT abort the
 * workflow — a missing indicator emoji is not a critical error.  The state
 * file is always written (with `reactionId: null` on failure) so that
 * `GITCLAW-AGENT.ts` does not crash when it tries to read it.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * DEPENDENCIES
 * ─────────────────────────────────────────────────────────────────────────────
 * - Node.js built-in `fs` module  (readFileSync, writeFileSync)
 * - GitHub CLI (`gh`)             — must be authenticated via GITHUB_TOKEN
 * - Bun runtime                   — for Bun.spawn and top-level await
 */

import { readFileSync, writeFileSync } from "fs";

// ─── Read GitHub Actions event context ────────────────────────────────────────
// GitHub Actions injects GITHUB_EVENT_PATH pointing to a JSON file that
// contains the full webhook payload for the triggering event.
const event = JSON.parse(readFileSync(process.env.GITHUB_EVENT_PATH!, "utf-8"));

// GITHUB_EVENT_NAME is 'issues' for newly-opened issues or
// 'issue_comment' for comments posted on existing issues.
const eventName = process.env.GITHUB_EVENT_NAME!;

// GITHUB_REPOSITORY is formatted as "owner/repo" (e.g. "acme/my-app").
const repo = process.env.GITHUB_REPOSITORY!;

// The issue number is present on both event shapes under `event.issue.number`.
const issueNumber: number = event.issue.number;

// ─── GitHub CLI helper ────────────────────────────────────────────────────────
// Thin wrapper around the `gh` CLI binary.  Spawns a subprocess, captures its
// stdout, waits for it to exit, and returns the trimmed text.  stderr is
// inherited so that authentication errors and API failures remain visible in
// the Actions log.
async function gh(...args: string[]): Promise<string> {
  const proc = Bun.spawn(["gh", ...args], { stdout: "pipe", stderr: "inherit" });
  const stdout = await new Response(proc.stdout).text();
  await proc.exited;
  return stdout.trim();
}

// ─── Add 👀 reaction ──────────────────────────────────────────────────────────
// Track three pieces of information that `GITCLAW-AGENT.ts` needs for cleanup:
//   reactionId     — the numeric GitHub reaction ID returned by the API
//   reactionTarget — "comment" or "issue" (determines which API endpoint to DELETE)
//   commentId      — the comment's ID, only set when reactionTarget === "comment"
let reactionId: string | null = null;
let reactionTarget: "comment" | "issue" = "issue";
let commentId: number | null = null;

try {
  if (eventName === "issue_comment") {
    // ── React to the comment that triggered the workflow ─────────────────────
    // POST /repos/{owner}/{repo}/issues/comments/{comment_id}/reactions
    commentId = event.comment.id;
    reactionId = await gh(
      "api", `repos/${repo}/issues/comments/${commentId}/reactions`,
      "-f", "content=eyes", "--jq", ".id"
    );
    reactionTarget = "comment";
  } else {
    // ── React to the issue itself (e.g. newly opened) ────────────────────────
    // POST /repos/{owner}/{repo}/issues/{issue_number}/reactions
    reactionId = await gh(
      "api", `repos/${repo}/issues/${issueNumber}/reactions`,
      "-f", "content=eyes", "--jq", ".id"
    );
  }
} catch (e) {
  // A failed reaction is non-fatal — log it but do not abort the workflow.
  // The agent will still run; the user just won't see the 👀 indicator.
  console.error("Failed to add reaction:", e);
}

// ─── Persist reaction state for GITCLAW-AGENT.ts cleanup ────────────────────
// Write all fields to a well-known temp path.  `GITCLAW-AGENT.ts` reads this
// file inside its `finally` block and uses the IDs to DELETE the reaction
// once the agent finishes — ensuring the indicator is always cleaned up,
// even if the agent itself throws an error.
writeFileSync("/tmp/reaction-state.json", JSON.stringify({
  reactionId,
  reactionTarget,
  commentId,
  issueNumber,
  repo,
}));
