/**
 * Phase 0 validation tests — verify that all Foundation-layer features
 * described in .GITCLAW/docs/GITCLAW-Roadmap.md are structurally present.
 *
 * Run with: node --test .GITCLAW/tests/phase0.test.js
 *        or: bun test .GITCLAW/tests/phase0.test.js
 */

const { describe, it } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const REPO_ROOT = path.resolve(__dirname, "..", "..");
const GITCLAW = path.resolve(REPO_ROOT, ".GITCLAW");

function readFile(relPath) {
  return fs.readFileSync(path.resolve(REPO_ROOT, relPath), "utf-8");
}

// ── 1. Trigger on issues.opened and issue_comment.created ──────────────────

describe("Workflow triggers", () => {
  const workflow = readFile(".github/workflows/GITCLAW-WORKFLOW-AGENT.yml");

  it("triggers on issues.opened", () => {
    assert.match(workflow, /issues:\s*\n\s*types:\s*\[.*opened.*\]/);
  });

  it("triggers on issue_comment.created", () => {
    assert.match(workflow, /issue_comment:\s*\n\s*types:\s*\[.*created.*\]/);
  });

  it("workflow template also has correct triggers", () => {
    const template = readFile(".GITCLAW/install/GITCLAW-WORKFLOW-AGENT.yml");
    assert.match(template, /issues:\s*\n\s*types:\s*\[.*opened.*\]/);
    assert.match(template, /issue_comment:\s*\n\s*types:\s*\[.*created.*\]/);
  });
});

// ── 2. Authorization gating ────────────────────────────────────────────────

describe("Authorization gating", () => {
  const workflow = readFile(".github/workflows/GITCLAW-WORKFLOW-AGENT.yml");

  it("has an Authorize step that checks collaborator permission", () => {
    assert.ok(workflow.includes("name: Authorize"));
    assert.ok(workflow.includes("collaborators"));
    assert.ok(workflow.includes("permission"));
  });

  it("gates on admin permission", () => {
    assert.ok(workflow.includes("admin"));
  });

  it("gates on write permission", () => {
    assert.ok(workflow.includes("write"));
  });

  it("excludes github-actions[bot] from comment triggers", () => {
    assert.ok(workflow.includes("github-actions[bot]"));
  });

  it("Authorize step runs before Checkout", () => {
    const authorizeIdx = workflow.indexOf("name: Authorize");
    const checkoutIdx = workflow.indexOf("name: Checkout");
    assert.ok(authorizeIdx > 0 && checkoutIdx > 0);
    assert.ok(
      authorizeIdx < checkoutIdx,
      "Authorize must run before Checkout"
    );
  });
});

// ── 3. Multi-turn sessions persisted as JSONL in state/sessions/ ───────────

describe("Session persistence", () => {
  it("state/sessions directory exists", () => {
    assert.ok(fs.existsSync(path.join(GITCLAW, "state", "sessions")));
  });

  it("agent script references sessions directory", () => {
    const agent = readFile(".GITCLAW/lifecycle/GITCLAW-AGENT.ts");
    assert.ok(agent.includes("state/sessions"));
  });

  it("agent script uses JSONL session format", () => {
    const agent = readFile(".GITCLAW/lifecycle/GITCLAW-AGENT.ts");
    assert.ok(agent.includes(".jsonl"));
  });

  it("agent script handles session resumption", () => {
    const agent = readFile(".GITCLAW/lifecycle/GITCLAW-AGENT.ts");
    assert.ok(agent.includes("--session"));
    assert.ok(agent.includes('mode = "resume"'));
  });
});

// ── 4. Issue → session mapping in state/issues/ ────────────────────────────

describe("Issue-session mapping", () => {
  it("state/issues directory exists", () => {
    assert.ok(fs.existsSync(path.join(GITCLAW, "state", "issues")));
  });

  it("agent script writes mapping files", () => {
    const agent = readFile(".GITCLAW/lifecycle/GITCLAW-AGENT.ts");
    assert.ok(agent.includes("mappingFile"));
    assert.ok(agent.includes("writeFileSync"));
  });

  it("mapping includes issueNumber and sessionPath", () => {
    const agent = readFile(".GITCLAW/lifecycle/GITCLAW-AGENT.ts");
    assert.ok(agent.includes("issueNumber"));
    assert.ok(agent.includes("sessionPath"));
  });
});

// ── 5. 👀 reaction indicator while working ─────────────────────────────────

describe("Reaction indicator", () => {
  it("indicator script exists", () => {
    assert.ok(
      fs.existsSync(path.join(GITCLAW, "lifecycle", "GITCLAW-INDICATOR.ts"))
    );
  });

  it("indicator adds eyes reaction", () => {
    const indicator = readFile(".GITCLAW/lifecycle/GITCLAW-INDICATOR.ts");
    assert.ok(indicator.includes("content=eyes"));
  });

  it("indicator persists reaction state to /tmp", () => {
    const indicator = readFile(".GITCLAW/lifecycle/GITCLAW-INDICATOR.ts");
    assert.ok(indicator.includes("/tmp/reaction-state.json"));
  });

  it("agent script removes reaction in finally block", () => {
    const agent = readFile(".GITCLAW/lifecycle/GITCLAW-AGENT.ts");
    assert.ok(agent.includes("finally"));
    assert.ok(agent.includes("reactionId"));
    assert.ok(agent.includes("DELETE"));
  });

  it("workflow runs indicator before install", () => {
    const workflow = readFile(".github/workflows/GITCLAW-WORKFLOW-AGENT.yml");
    const indicatorIdx = workflow.indexOf("GITCLAW-INDICATOR");
    const installIdx = workflow.indexOf("bun install");
    assert.ok(indicatorIdx > 0 && installIdx > 0);
    assert.ok(
      indicatorIdx < installIdx,
      "Indicator must run before dependency install"
    );
  });
});

// ── 6. Commit + push state to main with retry-on-conflict ──────────────────

describe("Commit and push with retry", () => {
  const agent = readFile(".GITCLAW/lifecycle/GITCLAW-AGENT.ts");

  it("stages all changes with git add", () => {
    assert.ok(agent.includes('"git", "add"'));
  });

  it("commits with descriptive message", () => {
    assert.ok(agent.includes("git commit"));
    assert.ok(agent.includes("gitclaw: work on issue"));
  });

  it("pushes to default branch", () => {
    assert.ok(agent.includes("git push"));
    assert.ok(agent.includes("defaultBranch"));
  });

  it("retries push on conflict", () => {
    assert.ok(agent.includes("retrying"));
    assert.ok(agent.includes('"git", "pull"'));
    assert.ok(agent.includes('"--rebase"'));
  });

  it("has a retry limit", () => {
    assert.match(agent, /for\s*\(\s*let\s+i\s*=\s*1;\s*i\s*<=\s*3/);
  });
});

// ── 7. Modular skill system and configurable personality ───────────────────

describe("Skill system", () => {
  const skillsDir = path.join(GITCLAW, ".pi", "skills");

  it("skills directory exists", () => {
    assert.ok(fs.existsSync(skillsDir));
  });

  it("memory skill exists with SKILL.md", () => {
    const skillFile = path.join(skillsDir, "memory", "SKILL.md");
    assert.ok(fs.existsSync(skillFile));
    const content = fs.readFileSync(skillFile, "utf-8");
    assert.ok(content.startsWith("---"), "SKILL.md must have YAML frontmatter");
    assert.ok(content.includes("name:"));
    assert.ok(content.includes("description:"));
  });

  it("skill-creator skill exists with SKILL.md", () => {
    const skillFile = path.join(skillsDir, "skill-creator", "SKILL.md");
    assert.ok(fs.existsSync(skillFile));
    const content = fs.readFileSync(skillFile, "utf-8");
    assert.ok(content.startsWith("---"), "SKILL.md must have YAML frontmatter");
    assert.ok(content.includes("name:"));
    assert.ok(content.includes("description:"));
  });
});

describe("Configurable personality", () => {
  it("settings.json exists with provider and model config", () => {
    const settings = JSON.parse(
      readFile(".GITCLAW/.pi/settings.json")
    );
    assert.ok(settings.defaultProvider);
    assert.ok(settings.defaultModel);
    assert.ok(settings.defaultThinkingLevel);
  });

  it("APPEND_SYSTEM.md exists (system prompt)", () => {
    assert.ok(
      fs.existsSync(path.join(GITCLAW, ".pi", "APPEND_SYSTEM.md"))
    );
  });

  it("BOOTSTRAP.md exists (first-run identity)", () => {
    assert.ok(
      fs.existsSync(path.join(GITCLAW, ".pi", "BOOTSTRAP.md"))
    );
  });

  it("AGENTS.md exists (agent identity)", () => {
    assert.ok(fs.existsSync(path.join(GITCLAW, "AGENTS.md")));
  });
});

// ── Fail-closed guard (prerequisite for all Phase 0 features) ──────────────

describe("Fail-closed guard", () => {
  it("sentinel file exists", () => {
    assert.ok(
      fs.existsSync(path.join(GITCLAW, "GITCLAW-ENABLED.md"))
    );
  });

  it("guard script exists", () => {
    assert.ok(
      fs.existsSync(path.join(GITCLAW, "lifecycle", "GITCLAW-ENABLED.ts"))
    );
  });

  it("guard checks for sentinel file", () => {
    const guard = readFile(".GITCLAW/lifecycle/GITCLAW-ENABLED.ts");
    assert.ok(guard.includes("GITCLAW-ENABLED.md"));
    assert.ok(guard.includes("existsSync"));
  });

  it("guard exits non-zero when sentinel missing", () => {
    const guard = readFile(".GITCLAW/lifecycle/GITCLAW-ENABLED.ts");
    assert.ok(guard.includes("process.exit(1)"));
  });

  it("workflow runs guard before indicator and agent", () => {
    const workflow = readFile(".github/workflows/GITCLAW-WORKFLOW-AGENT.yml");
    const guardIdx = workflow.indexOf("GITCLAW-ENABLED");
    const indicatorIdx = workflow.indexOf("GITCLAW-INDICATOR");
    const agentIdx = workflow.indexOf("GITCLAW-AGENT");
    assert.ok(guardIdx > 0);
    assert.ok(guardIdx < indicatorIdx, "Guard must run before indicator");
    assert.ok(guardIdx < agentIdx, "Guard must run before agent");
  });
});

// ── Install template integrity ─────────────────────────────────────────────

describe("Install templates", () => {
  it("hatch template has valid frontmatter", () => {
    const template = readFile(
      ".GITCLAW/install/GITCLAW-TEMPLATE-HATCH.md"
    );
    assert.ok(
      template.startsWith("---"),
      "Hatch template must start with YAML frontmatter delimiter"
    );
    assert.ok(template.includes('name: "🥚 Hatch"'));
    assert.ok(template.includes("labels:"));
  });

  it("workflow template has checkout with ref and fetch-depth", () => {
    const template = readFile(
      ".GITCLAW/install/GITCLAW-WORKFLOW-AGENT.yml"
    );
    assert.ok(
      template.includes("github.event.repository.default_branch"),
      "Checkout should reference default_branch"
    );
    assert.ok(
      template.includes("fetch-depth: 0"),
      "Checkout should fetch full history"
    );
  });

  it("workflow template matches live workflow triggers", () => {
    const template = readFile(
      ".GITCLAW/install/GITCLAW-WORKFLOW-AGENT.yml"
    );
    const live = readFile(".github/workflows/GITCLAW-WORKFLOW-AGENT.yml");
    // Both should have the same trigger structure
    assert.ok(template.includes("issues:"));
    assert.ok(template.includes("issue_comment:"));
    assert.ok(live.includes("issues:"));
    assert.ok(live.includes("issue_comment:"));
  });

  it("workflow template name matches live workflow name", () => {
    const template = readFile(
      ".GITCLAW/install/GITCLAW-WORKFLOW-AGENT.yml"
    );
    const live = readFile(".github/workflows/GITCLAW-WORKFLOW-AGENT.yml");
    const templateName = template.match(/^name:\s*(.+)$/m)?.[1];
    const liveName = live.match(/^name:\s*(.+)$/m)?.[1];
    assert.ok(templateName, "Template should have a name field");
    assert.ok(liveName, "Live workflow should have a name field");
    assert.strictEqual(liveName, templateName, "Live workflow name must match template");
  });
});

// ── Heart emoji guard ──────────────────────────────────────────────────────

describe("Heart emoji guard", () => {
  it("heart guard script exists", () => {
    assert.ok(
      fs.existsSync(path.join(GITCLAW, "lifecycle", "GITCLAW-HEART-GUARD.ts"))
    );
  });

  it("default sentinel file is GITCLAW-HEART-NOT-REQUIRED.md", () => {
    assert.ok(
      fs.existsSync(path.join(GITCLAW, "GITCLAW-HEART-NOT-REQUIRED.md"))
    );
  });

  it("guard checks for GITCLAW-HEART-REQUIRED file pattern", () => {
    const guard = readFile(".GITCLAW/lifecycle/GITCLAW-HEART-GUARD.ts");
    assert.ok(guard.includes("GITCLAW-HEART-REQUIRED"));
  });

  it("guard reads issue body from event payload", () => {
    const guard = readFile(".GITCLAW/lifecycle/GITCLAW-HEART-GUARD.ts");
    assert.ok(guard.includes("GITHUB_EVENT_PATH"));
    assert.ok(guard.includes("issue"));
    assert.ok(guard.includes("body"));
  });

  it("guard checks for heart emoji pattern", () => {
    const guard = readFile(".GITCLAW/lifecycle/GITCLAW-HEART-GUARD.ts");
    assert.ok(guard.includes("heartPattern"));
  });

  it("guard only enforces on issues.opened events", () => {
    const guard = readFile(".GITCLAW/lifecycle/GITCLAW-HEART-GUARD.ts");
    assert.ok(guard.includes("GITHUB_EVENT_NAME"));
    assert.ok(guard.includes('"issues"'));
  });

  it("guard exits non-zero when heart emoji is missing", () => {
    const guard = readFile(".GITCLAW/lifecycle/GITCLAW-HEART-GUARD.ts");
    assert.ok(guard.includes("process.exit(1)"));
  });

  it("guard passes when heart requirement is not active", () => {
    const guard = readFile(".GITCLAW/lifecycle/GITCLAW-HEART-GUARD.ts");
    assert.ok(guard.includes("process.exit(0)"));
  });

  it("workflow runs heart guard after main guard and before preinstall", () => {
    const workflow = readFile(".github/workflows/GITCLAW-WORKFLOW-AGENT.yml");
    const guardIdx = workflow.indexOf("GITCLAW-ENABLED");
    const heartGuardIdx = workflow.indexOf("GITCLAW-HEART-GUARD");
    const indicatorIdx = workflow.indexOf("GITCLAW-INDICATOR");
    assert.ok(guardIdx > 0, "Main guard must be present");
    assert.ok(heartGuardIdx > 0, "Heart guard must be present");
    assert.ok(indicatorIdx > 0, "Indicator must be present");
    assert.ok(guardIdx < heartGuardIdx, "Main guard must run before heart guard");
    assert.ok(heartGuardIdx < indicatorIdx, "Heart guard must run before indicator");
  });

  it("workflow template also has heart guard step", () => {
    const template = readFile(".GITCLAW/install/GITCLAW-WORKFLOW-AGENT.yml");
    assert.ok(template.includes("GITCLAW-HEART-GUARD"));
    const guardIdx = template.indexOf("GITCLAW-ENABLED");
    const heartGuardIdx = template.indexOf("GITCLAW-HEART-GUARD");
    const indicatorIdx = template.indexOf("GITCLAW-INDICATOR");
    assert.ok(guardIdx < heartGuardIdx, "Template: main guard before heart guard");
    assert.ok(heartGuardIdx < indicatorIdx, "Template: heart guard before indicator");
  });

  it("help documentation for heart requirement exists", () => {
    assert.ok(
      fs.existsSync(path.join(GITCLAW, "help", "heart-requirement.md"))
    );
  });

  it("help index links to heart requirement page", () => {
    const helpIndex = readFile(".GITCLAW/help/README.md");
    assert.ok(helpIndex.includes("heart-requirement.md"));
  });
});

// ── Error handling and observability ───────────────────────────────────────

describe("Error handling", () => {
  const agent = readFile(".GITCLAW/lifecycle/GITCLAW-AGENT.ts");

  it("gh() helper checks exit code", () => {
    assert.ok(agent.includes("exitCode !== 0"));
    assert.ok(agent.includes("throw new Error"));
  });

  it("pi agent stderr is not silenced", () => {
    assert.ok(
      !agent.includes('stderr: "ignore"'),
      "pi agent stderr should not be silenced — use 'inherit' for observability"
    );
  });

  it("validates provider API key is set", () => {
    assert.ok(
      agent.includes("providerKeyMap"),
      "Agent should validate that the required API key for the configured provider is present"
    );
    assert.ok(agent.includes("ANTHROPIC_API_KEY"));
  });

  it("checks pi agent exit code and throws on failure", () => {
    assert.ok(agent.includes("piExitCode"));
    assert.ok(agent.includes("throw new Error") && agent.includes("piExitCode"));
  });

  it("handles empty agent response", () => {
    assert.ok(
      agent.includes("did not produce a response"),
      "Agent should post an error message when response is empty"
    );
  });
});
