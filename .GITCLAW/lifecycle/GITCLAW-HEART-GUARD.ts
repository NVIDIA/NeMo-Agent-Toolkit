/**
 * GITCLAW-HEART-GUARD.ts — Optional guard that requires a ❤️ heart emoji
 * in the issue body for new issues to be processed.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * PURPOSE
 * ─────────────────────────────────────────────────────────────────────────────
 * This script provides an optional gate for incoming issues.  When a file
 * matching `.GITCLAW/GITCLAW-HEART-REQUIRED.*` (any extension) exists in
 * the repository, the guard enforces that newly opened issues contain a ❤️
 * heart emoji somewhere in their body.  If the emoji is absent, the workflow
 * exits early and the agent does not process the issue.
 *
 * When no `GITCLAW-HEART-REQUIRED.*` file is found (the default — the repo
 * ships with `GITCLAW-HEART-NOT-REQUIRED.md`), the guard passes immediately
 * and all issues are processed normally.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * SCOPE
 * ─────────────────────────────────────────────────────────────────────────────
 * The heart requirement applies ONLY to `issues.opened` events (i.e. new
 * issues).  Comments on existing issues (`issue_comment.created`) are always
 * allowed through — requiring a heart on every follow-up comment would be
 * disruptive and unnecessary.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * ACTIVATION
 * ─────────────────────────────────────────────────────────────────────────────
 * To ENABLE  the heart requirement:
 *   Rename `GITCLAW-HEART-NOT-REQUIRED.md` → `GITCLAW-HEART-REQUIRED.md`
 *   (or create any file named `GITCLAW-HEART-REQUIRED.*`).
 *
 * To DISABLE the heart requirement:
 *   Rename `GITCLAW-HEART-REQUIRED.md` → `GITCLAW-HEART-NOT-REQUIRED.md`
 *   (or delete all `GITCLAW-HEART-REQUIRED.*` files).
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * DEPENDENCIES
 * ─────────────────────────────────────────────────────────────────────────────
 * - Node.js built-in `fs` module  (readdirSync, readFileSync)
 * - Node.js built-in `path` module (resolve)
 * - Bun runtime (for `import.meta.dir` support)
 *
 * No external packages are required; this file intentionally has zero
 * third-party dependencies so it can run before `bun install`.
 */

import { readdirSync, readFileSync } from "fs";
import { resolve } from "path";

// ─── Resolve paths ────────────────────────────────────────────────────────────
// `import.meta.dir` resolves to `.GITCLAW/lifecycle/`.  Step one level up to
// reach `.GITCLAW/`.
const gitclawDir = resolve(import.meta.dir, "..");

// ─── Check if heart requirement is enabled ────────────────────────────────────
// Scan the `.GITCLAW/` directory for any file matching `GITCLAW-HEART-REQUIRED.*`.
const heartRequiredFile = readdirSync(gitclawDir).find((f) =>
  /^GITCLAW-HEART-REQUIRED\..+$/.test(f)
);

if (!heartRequiredFile) {
  // No GITCLAW-HEART-REQUIRED.* file found — heart is not required.
  console.log("GitClaw heart guard — heart requirement is not active. Passing.");
  process.exit(0);
}

console.log(`GitClaw heart guard — heart requirement is active (${heartRequiredFile}).`);

// ─── Only enforce on issues.opened events ─────────────────────────────────────
const eventName = process.env.GITHUB_EVENT_NAME!;

if (eventName !== "issues") {
  // Comments on existing issues are always allowed through.
  console.log("GitClaw heart guard — event is not issues.opened. Passing.");
  process.exit(0);
}

// ─── Read the issue body from the event payload ───────────────────────────────
const event = JSON.parse(readFileSync(process.env.GITHUB_EVENT_PATH!, "utf-8"));
const issueBody: string = event.issue?.body ?? "";

// ─── Check for heart emoji ────────────────────────────────────────────────────
// Match common heart emoji characters: ❤️ ❤ 💓 💕 💖 💗 💘 💙 💚 💛 💜 💝 💞 💟 🖤 🤍 🤎 🧡 ♥️ ♥
const heartPattern = /\u2764\uFE0F?|[\u{1F493}-\u{1F49F}]|\u{1F5A4}|\u{1F90D}|\u{1F90E}|\u{1F9E1}|\u2665\uFE0F?/u;

if (heartPattern.test(issueBody)) {
  console.log("GitClaw heart guard — ❤️ heart emoji found in issue body. Passing.");
  process.exit(0);
}

// ─── Heart emoji not found — block processing ────────────────────────────────
console.error(
  "GitClaw heart guard — issue does not contain a ❤️ heart emoji. Skipping.\n" +
  "To process this issue, edit it to include a heart emoji (❤️) in the body.\n" +
  "To disable this requirement, rename `.GITCLAW/GITCLAW-HEART-REQUIRED.md` to `.GITCLAW/GITCLAW-HEART-NOT-REQUIRED.md`."
);
process.exit(1);
