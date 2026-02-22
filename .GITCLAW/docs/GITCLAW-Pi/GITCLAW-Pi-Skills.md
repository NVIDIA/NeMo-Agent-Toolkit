# 🦞 GITCLAW .pi Skills

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

Skills are modular, self-contained packages that extend the agent's capabilities. They transform the agent from a general-purpose assistant into a specialized one with domain expertise, reusable scripts, and procedural knowledge.

## How Skills Work

### Progressive Disclosure

Skills use a three-level loading system to manage the context window efficiently:

1. **Metadata** (name + description) — Always in context (~100 words). This is how the agent decides whether to activate a skill.
2. **SKILL.md body** — Loaded when the skill triggers (<5k words). Contains instructions and workflows.
3. **Bundled resources** — Loaded as needed by the agent. Scripts can be executed without reading into context.

This means a repo can have dozens of skills installed without bloating the agent's context. Only the relevant ones load, and only as deep as needed.

### Activation

The agent reads all skill metadata (name + description) at the start of every session. When a user request matches a skill's description, the agent loads the full `SKILL.md` body and any relevant resources.

**Example**: If the user asks "what did we talk about last week?", the `memory` skill triggers because its description mentions "remember when" and "past sessions."

## Built-In Skills

### Memory

**Location**: `.pi/skills/memory/`

Enables the agent to search and recall information from past sessions.

**Capabilities**:
- Search `memory.log` for specific facts, preferences, or decisions
- Query session JSONL files for past conversations
- Extract user messages, assistant responses, or tool usage from sessions
- List sessions by date
- Get conversation overviews (skipping thinking blocks and tool calls)

**Key commands**:
```bash
rg -i "search term" state/memory.log # Search memory
tail -30 state/memory.log # Recent memories
rg -i "search term" state/sessions/ # Search all sessions
```

### Skill Creator

**Location**: `.pi/skills/skill-creator/`

A meta-skill for creating new skills. Includes scaffolding scripts and comprehensive guidance on skill design.

**Capabilities**:
- Initialize new skills with `scripts/init_skill.py`
- Validate skill structure with `scripts/quick_validate.py`
- Package skills for distribution with `scripts/package_skill.py`
- Guidance on naming, structure, progressive disclosure, and best practices

## Skill Anatomy

Every skill follows this structure:

```
skill-name/
├── SKILL.md              # Required — metadata + instructions
└── (optional resources)
    ├── scripts/          # Executable code (Python, Bash, etc.)
    ├── references/       # Documentation loaded into context as needed
    └── assets/           # Files used in output (templates, images, etc.)
```

### SKILL.md

The only required file. Contains:

**Frontmatter** (YAML):
```yaml
---
name: my-skill
description: What this skill does and when to use it. Be specific about triggers.
---
```

The `description` is the primary triggering mechanism. It should include both what the skill does and when to activate it.

**Body** (Markdown): Instructions, workflows, and references to bundled resources. Only loaded after the skill triggers.

### Bundled Resources

| Type | Purpose | When to use |
|------|---------|-------------|
| **scripts/** | Deterministic, repeatable code | Same code gets rewritten every time |
| **references/** | Documentation loaded on demand | Schemas, API docs, domain knowledge |
| **assets/** | Files used in output | Templates, images, boilerplate |

## Creating a New Skill

The recommended workflow:

### 1. Understand the Use Cases

Start with concrete examples of how the skill will be used. What would a user say that should trigger it? What does the agent need to do in response?

### 2. Plan the Resources

For each use case, identify:
- Scripts that would prevent rewriting the same code
- Reference material the agent needs (schemas, docs, policies)
- Assets that should ship with the skill (templates, images)

### 3. Initialize

```bash
scripts/init_skill.py my-skill --path skills/public --resources scripts,references
```

This creates the directory structure and a `SKILL.md` template with TODO placeholders.

### 4. Implement

- Write the scripts, references, and assets
- Update `SKILL.md` with instructions and references to bundled resources
- Test scripts by actually running them

### 5. Package

```bash
scripts/package_skill.py path/to/skill-folder
```

Validates the skill structure and creates a distributable `.skill` file (a zip with a `.skill` extension).

### 6. Iterate

Use the skill on real tasks. Notice struggles or inefficiencies. Update and re-package.

## Design Principles

### Keep SKILL.md Lean

Under 500 lines. If it's getting long, split content into reference files and link to them from `SKILL.md`.

### Write Good Descriptions

The description is everything. If it doesn't clearly explain when to use the skill, the agent won't activate it at the right time.

**Good**: "Search and recall information from past sessions and memory logs. Use when you need to remember something, reference past conversations, ask 'what did we talk about', 'remember when', or need historical context."

**Bad**: "Memory skill."

### Match Freedom to Fragility

- **High freedom** (text instructions): Multiple approaches are valid
- **Medium freedom** (pseudocode/parameterized scripts): A preferred pattern exists
- **Low freedom** (specific scripts): Operations are fragile, consistency is critical

### Avoid Deep Nesting

Keep references one level deep from `SKILL.md`. All reference files should link directly from the skill's main document.

### Don't Duplicate

Information should live in either `SKILL.md` or reference files, not both. Keep core workflow in `SKILL.md`; move detailed reference material to `references/`.
