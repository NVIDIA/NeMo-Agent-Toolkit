/**
 * GITCLAW-INSTALLER.ts — One-time setup script for gitclaw.
 *
 * Copies the GitHub Actions workflow, issue templates, and git attributes
 * from `.GITCLAW/install` into the standard locations the repo needs to function.
 * Existing files are never overwritten — only missing ones are installed.
 *
 * Usage:
 *   bun .GITCLAW/install/GITCLAW-INSTALLER.ts
 */

import { existsSync, mkdirSync, cpSync, readFileSync, writeFileSync } from "fs";
import { resolve } from "path";

/** Directory containing the installable resources. */
const bootstrapDir = import.meta.dir;

/** Repository root — two levels above `.GITCLAW/install/`. */
const repoRoot = resolve(bootstrapDir, "..", "..");

/** Create a directory (and parents) if it does not already exist. */
function ensureDir(dir: string) {
  mkdirSync(dir, { recursive: true });
}

/** Copy `src` to `dest` only when `dest` is absent; logs the outcome. */
function copyIfMissing(src: string, dest: string, label: string) {
  if (existsSync(dest)) {
    console.log(`  ⏭  ${label} already exists, skipping`);
  } else {
    cpSync(src, dest, { recursive: true });
    console.log(`  ✅ ${label} installed`);
  }
}

/**
 * Ensure that `.gitattributes` contains `attributeRule`.
 * Creates the file if absent; appends the rule if not already present.
 */
function ensureAttribute(filePath: string, attributeRule: string) {
  if (!existsSync(filePath)) {
    writeFileSync(filePath, `${attributeRule}\n`, "utf-8");
    console.log(`  ✅ .gitattributes created with: ${attributeRule}`);
    return;
  }

  const currentContent = readFileSync(filePath, "utf-8");
  const lines = currentContent
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !line.startsWith("#"));

  if (lines.includes(attributeRule)) {
    console.log(`  ⏭  .gitattributes already contains: ${attributeRule}`);
    return;
  }

  const separator = currentContent.endsWith("\n") || currentContent.length === 0 ? "" : "\n";
  writeFileSync(filePath, `${currentContent}${separator}${attributeRule}\n`, "utf-8");
  console.log(`  ✅ Added to .gitattributes: ${attributeRule}`);
}

console.log("🔧 Installing gitclaw into this repository...\n");

// --- Install workflow -------------------------------------------------
console.log("Workflows:");
ensureDir(resolve(repoRoot, ".github", "workflows"));
copyIfMissing(
  resolve(bootstrapDir, "GITCLAW-WORKFLOW-AGENT.yml"),
  resolve(repoRoot, ".github", "workflows", "GITCLAW-WORKFLOW-AGENT.yml"),
  ".github/workflows/GITCLAW-WORKFLOW-AGENT.yml"
);

// --- Issue templates --------------------------------------------------
console.log("\nIssue templates:");
ensureDir(resolve(repoRoot, ".github", "ISSUE_TEMPLATE"));
copyIfMissing(
  resolve(bootstrapDir, "GITCLAW-TEMPLATE-HATCH.md"),
  resolve(repoRoot, ".github", "ISSUE_TEMPLATE", "hatch.md"),
  ".github/ISSUE_TEMPLATE/hatch.md"
);

// --- Agent identity ---------------------------------------------------
console.log("\nAgent identity:");
ensureDir(resolve(repoRoot, ".GITCLAW"));
copyIfMissing(resolve(bootstrapDir, "GITCLAW-AGENTS.md"), resolve(repoRoot, ".GITCLAW", "AGENTS.md"), ".GITCLAW/AGENTS.md");

// --- Git attributes --------------------------------------------------
// `memory.log merge=union` tells git to union-merge the append-only
// memory log so concurrent writes from different branches don't conflict.
console.log("\nGit attributes:");
ensureAttribute(resolve(repoRoot, ".gitattributes"), "memory.log merge=union");

console.log("\n✨ gitclaw installed!\n");
console.log("Next steps:");
console.log("  1. Add ANTHROPIC_API_KEY to Settings → Secrets and variables → Actions");
console.log("  2. Run: cd .GITCLAW/install && bun install");
console.log("  3. Commit and push the changes");
console.log("  4. Open an issue to start chatting with the agent\n");
