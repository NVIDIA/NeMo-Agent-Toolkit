# .GITCLAW 🦞 Installation Personalities

### Deep Analysis: Repo-Type-Specific Guardrails at Installation Time

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

---

## The Insight

Right now, `.GITCLAW` installs the same way into every repository. A Python web application gets the same agent, the same system prompt, the same guardrails, and the same skills as a browser extension, a novel manuscript, or a data analysis notebook. The agent is powerful from the moment it lands — but it's _generically_ powerful. It doesn't know what kind of project it's in until someone tells it.

That's a missed opportunity.

The moment of installation is the highest-leverage point for inserting guardrails. The operator is present, engaged, and making a decision. If the installer asks _"What kind of project is this?"_ and configures the agent accordingly, every subsequent interaction starts from the right context — with appropriate constraints, relevant skills, and a useful default personality.

This document analyzes how to make that happen.

---

## The Power at Installation

Before designing personalities, we need to understand what's at stake. The moment `.GITCLAW` is installed and pushed, the agent has:

| Capability | Implication |
|---|---|
| **Full repository read/write** | Can modify any file — source, config, CI, docs, its own behavioral files |
| **Arbitrary shell execution** | Can run any command available on the GitHub Actions Ubuntu runner |
| **GitHub API access** | Can post comments, interact with issues, read metadata |
| **LLM API key access** | Can make unbounded API calls against the configured provider |
| **Direct push to default branch** | Changes land on `main` without review (unless branch protection is configured) |
| **Self-modification** | Can rewrite its own identity, system prompt, and settings |

This power is appropriate for some project types and excessive for others. A repository containing a novel manuscript doesn't need shell execution. A single-file HTML app doesn't need access to CI workflow files. A data analysis repo might need read access to everything but write access to nothing outside `output/`.

**Personality-based installation means right-sizing this power from the start.**

---

## Design: The Personality System

### How It Works

The installer (`GITCLAW-INSTALLER.ts`) gains a new step: personality selection. After copying the workflow and templates, it presents a menu of repo-type personalities. Each personality is a pre-configured bundle that sets:

1. **`settings.json` overrides** — model selection, thinking level, tool restrictions
2. **`APPEND_SYSTEM.md` additions** — repo-type-specific behavioral guidelines
3. **Path guardrails** — which directories the agent may read/write
4. **Skill pre-loading** — relevant skills activated by default
5. **`.gitclaw-personality.json`** — a manifest that records the choice and its constraints

### Where Personalities Live

```
.GITCLAW/install/personalities/
  single-file-html/
    personality.json        # metadata, description, guardrail config
    APPEND_SYSTEM.md        # additional system prompt content
    skills/                 # pre-bundled skills for this type
  browser-extension/
    personality.json
    APPEND_SYSTEM.md
    skills/
  python-website/
    personality.json
    APPEND_SYSTEM.md
    skills/
  writing/
    personality.json
    APPEND_SYSTEM.md
    skills/
  analysis/
    personality.json
    APPEND_SYSTEM.md
    skills/
  general/
    personality.json        # the current default — no restrictions
    APPEND_SYSTEM.md
```

### The Manifest: `personality.json`

Each personality defines its constraints declaratively:

```json
{
  "name": "python-website",
  "label": "Python Website (Django/Flask/FastAPI)",
  "description": "Web application with Python backend, templates, static assets, and tests",
  "guardrails": {
    "allowedWritePaths": ["src/", "app/", "templates/", "static/", "tests/", "docs/", "migrations/"],
    "deniedWritePaths": [".github/workflows/", ".GITCLAW/lifecycle/", ".GITCLAW/.pi/APPEND_SYSTEM.md"],
    "allowedReadPaths": ["*"],
    "deniedCommands": [],
    "maxFileSize": "500KB",
    "branchProtection": "recommended"
  },
  "settings": {
    "defaultThinkingLevel": "high",
    "preferredSkills": ["code-review", "test-generator", "dependency-audit"]
  },
  "systemPromptAppend": "APPEND_SYSTEM.md"
}
```

---

## Programming Personalities

### 1. Single File HTML Application

**Context**: A project where the entire application lives in one (or very few) HTML files. Think creative coding experiments, interactive demos, data visualizations, single-page tools, and submissions for game jams or code golf competitions. The repo is simple, the scope is tight, and the blast radius of a mistake is contained.

**What the agent should know**:
- The entire application is probably `index.html` (or a small handful of files)
- Changes to _the_ file are changes to _the_ application — there's no abstraction layer
- The developer likely thinks in terms of inline `<style>` and `<script>` rather than build toolchains
- Dependencies, if any, are loaded via CDN `<script>` tags, not package managers
- Testing may be manual (open in browser) rather than automated
- Performance and file size may be core constraints

**Guardrails**:

```json
{
  "name": "single-file-html",
  "label": "Single File HTML Application",
  "guardrails": {
    "allowedWritePaths": ["*.html", "*.css", "*.js", "assets/", "docs/", "README.md"],
    "deniedWritePaths": [".github/workflows/", ".GITCLAW/lifecycle/", ".GITCLAW/.pi/"],
    "deniedCommands": ["npm install", "yarn add", "bun add"],
    "maxFileSize": "1MB",
    "branchProtection": "optional"
  },
  "settings": {
    "defaultThinkingLevel": "medium",
    "preferredSkills": ["code-review", "accessibility-check"]
  }
}
```

**System prompt additions**:
> You are working in a single-file HTML application repository. The entire application likely lives in one or a few HTML files. Respect this simplicity — don't introduce build tools, package managers, or framework dependencies unless the user explicitly asks. When suggesting changes, always consider that the user may be testing by opening the file directly in a browser. Prioritize inline solutions. If adding external dependencies, prefer CDN links over npm packages. Keep things small, portable, and self-contained.

**Why these guardrails matter**:
- Blocking `npm install` prevents the agent from turning a 50KB project into a 200MB `node_modules` disaster
- Restricting write paths keeps the agent focused on the actual application files
- Lower thinking level is appropriate — these projects rarely need deep architectural reasoning
- The agent should think like a creative coder, not an enterprise architect

---

### 2. Browser Extension

**Context**: A Chrome, Firefox, or cross-browser extension. These projects have strict structural requirements (manifests, content scripts, background workers, popup pages), security-sensitive APIs (tabs, storage, cookies, webRequest), and a review/publishing pipeline. The agent needs to understand the extension architecture without being able to accidentally break the manifest or escalate permissions.

**What the agent should know**:
- `manifest.json` (v3) or `manifest.json` (v2) is the central configuration file — changes here affect permissions, CSP, and store review
- Content scripts run in web page contexts; background scripts run in the extension's isolated context
- The `permissions` array in the manifest is a security boundary — adding permissions triggers user consent prompts and closer store review
- Web store review policies are strict: no obfuscated code, no unnecessary permissions, no remote code execution
- Testing often involves `chrome://extensions` reload cycles, not CI pipelines

**Guardrails**:

```json
{
  "name": "browser-extension",
  "label": "Browser Extension (Chrome/Firefox/Cross-browser)",
  "guardrails": {
    "allowedWritePaths": ["src/", "popup/", "options/", "background/", "content/", "icons/", "tests/", "docs/", "README.md"],
    "deniedWritePaths": [".github/workflows/", ".GITCLAW/lifecycle/", ".GITCLAW/.pi/"],
    "protectedFiles": ["manifest.json"],
    "deniedCommands": [],
    "maxFileSize": "2MB",
    "branchProtection": "recommended"
  },
  "settings": {
    "defaultThinkingLevel": "high",
    "preferredSkills": ["code-review", "security-audit", "manifest-validator"]
  }
}
```

**Protected files concept**: `manifest.json` is marked as protected — the agent can _read_ it and _propose_ changes, but modifications require explicit user confirmation. This is a new guardrail tier between "allowed" and "denied": the agent can suggest a diff but must not commit the change without approval.

**System prompt additions**:
> You are working in a browser extension repository. The `manifest.json` file is the most sensitive file in this project — changes to it affect permissions, security policy, and web store review outcomes. Never add permissions to the manifest without explaining why they are needed and what the user-facing consent impact will be. When writing content scripts, remember they run in the context of arbitrary web pages — treat all DOM data as untrusted. When writing background scripts, remember they have access to powerful browser APIs — minimize the permissions you request. Always consider cross-browser compatibility unless the user specifies a single target. Be aware of Manifest V3 requirements (service workers instead of persistent background pages, declarativeNetRequest instead of webRequest blocking).

**Why these guardrails matter**:
- Protecting `manifest.json` prevents accidental permission escalation
- The security-audit skill catches common extension vulnerabilities (XSS in popup, overly broad host permissions, eval() usage)
- High thinking level is warranted — extension APIs are complex and mistakes are hard to debug
- The agent needs to think about store review implications, not just code correctness

---

### 3. Python Website

**Context**: A web application built with Django, Flask, FastAPI, or similar Python frameworks. These projects have structured directory layouts (views, models, templates, static files, migrations), dependency management (pip/poetry/pipenv), testing infrastructure (pytest), and deployment considerations (WSGI/ASGI, Docker, environment variables).

**What the agent should know**:
- The project has a framework — respect its conventions (Django's `manage.py`, Flask's app factory, FastAPI's router pattern)
- Database migrations are order-dependent and sensitive — generating or modifying them carelessly can break production
- Environment variables (`.env`, secrets) contain credentials — never commit them, never expose them in comments
- Template files (Jinja2, Django templates) are rendered server-side — injection vulnerabilities are real
- Virtual environments (`venv/`, `.venv/`) should never be committed or modified
- Requirements files (`requirements.txt`, `pyproject.toml`, `Pipfile`) are the dependency source of truth

**Guardrails**:

```json
{
  "name": "python-website",
  "label": "Python Website (Django/Flask/FastAPI)",
  "guardrails": {
    "allowedWritePaths": ["src/", "app/", "templates/", "static/", "tests/", "docs/", "migrations/", "*.py", "*.html", "*.css", "*.js"],
    "deniedWritePaths": [".github/workflows/", ".GITCLAW/lifecycle/", ".GITCLAW/.pi/", ".env", ".env.*", "venv/", ".venv/"],
    "protectedFiles": ["requirements.txt", "pyproject.toml", "Pipfile", "Dockerfile", "docker-compose.yml"],
    "deniedCommands": ["rm -rf /", "sudo"],
    "maxFileSize": "2MB",
    "branchProtection": "recommended"
  },
  "settings": {
    "defaultThinkingLevel": "high",
    "preferredSkills": ["code-review", "test-generator", "dependency-audit", "security-audit"]
  }
}
```

**System prompt additions**:
> You are working in a Python web application repository. Respect the framework's conventions — don't fight the framework's patterns, extend them. When modifying models, always consider the migration impact and warn the user before generating migrations for destructive changes (column drops, type changes). Never read, expose, or commit `.env` files or their contents — treat all environment variables as secrets. When writing templates, be aware of server-side template injection risks — always use the framework's auto-escaping. When suggesting dependency changes, check for version compatibility with the existing stack. Prefer `pytest` for testing unless the project uses a different framework. When reviewing code, pay special attention to SQL injection, CSRF protection, authentication bypass, and insecure deserialization.

**Why these guardrails matter**:
- Denying `.env` writes prevents credential exposure
- Protecting dependency files (requirements.txt, pyproject.toml) prevents supply chain issues from unvetted additions
- The security-audit skill is critical — web applications face OWASP Top 10 threats
- Migration awareness prevents database disasters

---

### 4. Node.js / TypeScript Application

**Context**: A server-side or full-stack JavaScript/TypeScript application (Express, Next.js, Nest, etc.). Deep dependency trees, build toolchains, configuration sprawl, and fast-moving ecosystem.

**Guardrails**:

```json
{
  "name": "node-typescript",
  "label": "Node.js / TypeScript Application",
  "guardrails": {
    "allowedWritePaths": ["src/", "lib/", "pages/", "components/", "api/", "tests/", "docs/", "public/", "styles/"],
    "deniedWritePaths": [".github/workflows/", ".GITCLAW/lifecycle/", ".GITCLAW/.pi/", ".env", ".env.*", "node_modules/"],
    "protectedFiles": ["package.json", "package-lock.json", "tsconfig.json", ".eslintrc.*", "next.config.*", "Dockerfile"],
    "deniedCommands": ["rm -rf /", "sudo"],
    "maxFileSize": "2MB",
    "branchProtection": "recommended"
  },
  "settings": {
    "defaultThinkingLevel": "high",
    "preferredSkills": ["code-review", "test-generator", "dependency-audit", "type-checker"]
  }
}
```

**System prompt additions**:
> You are working in a Node.js/TypeScript application repository. Respect the existing build toolchain and configuration — don't introduce competing tools (e.g., don't add Webpack if the project uses Vite). When suggesting dependencies, check for bundle size impact and prefer lighter alternatives when they exist. Never commit `node_modules/`. When modifying `tsconfig.json`, explain the implications of compiler option changes. Pay attention to the distinction between `dependencies` and `devDependencies`. If the project uses a monorepo structure (workspaces, lerna, turborepo), respect package boundaries.

---

### 5. Mobile Application (React Native / Flutter)

**Context**: Cross-platform mobile applications with native build requirements, platform-specific code, and app store considerations.

**Guardrails**:

```json
{
  "name": "mobile-app",
  "label": "Mobile Application (React Native / Flutter)",
  "guardrails": {
    "allowedWritePaths": ["src/", "lib/", "components/", "screens/", "services/", "tests/", "docs/", "assets/"],
    "deniedWritePaths": [".github/workflows/", ".GITCLAW/lifecycle/", ".GITCLAW/.pi/", "android/app/build.gradle", "ios/*.xcodeproj/", "*.keystore", "*.jks", "*.mobileprovision"],
    "protectedFiles": ["app.json", "pubspec.yaml", "package.json", "AndroidManifest.xml", "Info.plist"],
    "deniedCommands": ["rm -rf /", "sudo"],
    "maxFileSize": "5MB",
    "branchProtection": "strongly-recommended"
  },
  "settings": {
    "defaultThinkingLevel": "high",
    "preferredSkills": ["code-review", "test-generator", "platform-compatibility"]
  }
}
```

**System prompt additions**:
> You are working in a mobile application repository. Native configuration files (Gradle, Xcode project files, AndroidManifest.xml, Info.plist) are extremely sensitive — changes can break builds, alter app permissions, or affect store review. Never modify signing configurations, keystores, or provisioning profiles. When writing platform-specific code, always consider both iOS and Android unless told otherwise. Be aware of platform-specific API differences, permission models, and lifecycle behaviors. Prefer cross-platform solutions when the abstraction cost is low.

---

## Non-Programming Personalities

This is where the personality system gets genuinely interesting. `.GITCLAW` isn't just a coding agent — it's an AI agent that lives in a git repository. Git repositories can hold anything: prose, research, data, legal documents, music, art direction. The agent's power should match the medium.

### 6. Writing (Novel / Screenplay / Blog / Documentation)

**Context**: A repository used primarily for writing — manuscripts, screenplays, blog posts, documentation, or other prose-heavy content. The files are Markdown, plain text, LaTeX, or similar formats. There is no "build" in the traditional sense, though there may be rendering (Markdown → HTML, LaTeX → PDF). The value is in the words, not the code.

**What the agent should know**:
- This is a creative or editorial project, not a software project
- The human is an author, not a developer (though they may be both)
- Continuity, tone, and voice consistency matter more than technical correctness
- Suggesting structural changes to a manuscript is more invasive than suggesting a code refactor — approach with care
- Spelling, grammar, and style are the equivalent of lint and tests
- The agent should be a thoughtful editor, not a code reviewer

**Guardrails**:

```json
{
  "name": "writing",
  "label": "Writing (Novel / Screenplay / Blog / Documentation)",
  "guardrails": {
    "allowedWritePaths": ["chapters/", "drafts/", "content/", "posts/", "docs/", "notes/", "*.md", "*.txt", "*.tex", "*.rst", "outline.*", "README.md"],
    "deniedWritePaths": [".github/workflows/", ".GITCLAW/lifecycle/", ".GITCLAW/.pi/"],
    "deniedCommands": ["npm", "pip", "bun", "cargo", "go", "make", "gcc", "javac"],
    "maxFileSize": "5MB",
    "branchProtection": "optional"
  },
  "settings": {
    "defaultThinkingLevel": "high",
    "preferredSkills": ["editorial-review", "continuity-check", "style-guide"]
  },
  "toolRestrictions": {
    "shellAccess": "minimal",
    "fileOps": "read-write",
    "gitOps": "read-only"
  }
}
```

**System prompt additions**:
> You are working in a writing repository. Your role is editorial, not technical. The files here contain creative or professional writing — manuscripts, blog posts, documentation, or other prose. Treat the author's voice with respect. When suggesting edits, explain _why_ (clarity, pacing, consistency, grammar) rather than just offering replacements. Never rewrite large sections without asking — small, targeted suggestions are more useful than wholesale rewrites. Track character names, plot points, terminology, and style choices in your memory so you can flag inconsistencies across chapters. If the project has a style guide, follow it. If it doesn't, suggest creating one after you understand the author's preferences. You are an editor, not a ghostwriter — your job is to make the author's work better, not to replace their voice.

**Skills for writing**:
- **`editorial-review`**: Line-by-line prose review focusing on clarity, grammar, pacing, and style consistency
- **`continuity-check`**: Track named entities, timelines, and facts across files; flag contradictions
- **`style-guide`**: Maintain and enforce a project-specific style guide (Oxford comma? Serial comma? Title case?)
- **`outline-manager`**: Help structure and reorganize content at the outline level
- **`word-count-tracker`**: Track word counts per file, per chapter, and overall — useful for authors with targets

**Why this personality matters**:
- Blocking build/package commands prevents the agent from trying to "build" a manuscript
- Shell access is minimal because there's nothing to compile, test, or deploy
- High thinking level is warranted — editorial feedback requires nuance and context
- The agent becomes a genuine creative collaborator instead of a confused code assistant

---

### 7. Research & Analysis

**Context**: A repository used for research — academic papers, market analysis, competitive intelligence, literature reviews, or investigative journalism. The content may be a mix of writing (reports, papers) and data (CSV, JSON, spreadsheets). The work is structured around questions, evidence, and conclusions rather than features and bugs.

**What the agent should know**:
- Accuracy is paramount — every claim should be traceable to evidence
- The research process has phases: question formulation → data gathering → analysis → synthesis → writing
- Citations and sourcing matter — the agent should track where information comes from
- Data files may be large and should not be casually modified
- The output is a document or presentation, not a running application
- Bias awareness is important — the agent should flag when evidence is one-sided

**Guardrails**:

```json
{
  "name": "analysis",
  "label": "Research & Analysis (Academic / Business / Data)",
  "guardrails": {
    "allowedWritePaths": ["research/", "analysis/", "reports/", "drafts/", "notes/", "data/processed/", "docs/", "*.md", "*.txt", "*.tex", "*.csv", "*.json", "README.md"],
    "deniedWritePaths": [".github/workflows/", ".GITCLAW/lifecycle/", ".GITCLAW/.pi/", "data/raw/"],
    "protectedFiles": ["data/raw/*"],
    "deniedCommands": ["rm -rf", "sudo"],
    "maxFileSize": "10MB",
    "branchProtection": "optional"
  },
  "settings": {
    "defaultThinkingLevel": "high",
    "preferredSkills": ["source-tracker", "fact-checker", "data-summarizer", "citation-manager"]
  }
}
```

**System prompt additions**:
> You are working in a research and analysis repository. Accuracy and intellectual honesty are your highest priorities. When making claims, cite your sources. When analyzing data, show your reasoning. When you are uncertain, say so — never present speculation as fact. The `data/raw/` directory contains source data and must never be modified; work in `data/processed/` or `analysis/` instead. Help the researcher organize findings, identify gaps in evidence, spot logical fallacies, and structure arguments. If asked to summarize, preserve the nuance — don't flatten complex findings into oversimplified bullet points. Track sources, methodology decisions, and key findings in your memory for cross-session continuity.

**Skills for research**:
- **`source-tracker`**: Maintain a bibliography of sources referenced in the project; flag unsourced claims
- **`fact-checker`**: Cross-reference claims against cited sources; flag contradictions or unsupported statements
- **`data-summarizer`**: Generate statistical summaries of CSV/JSON data files without modifying the originals
- **`citation-manager`**: Format citations in APA, MLA, Chicago, or custom styles; maintain a references file
- **`methodology-log`**: Track research methodology decisions and their rationale for reproducibility

**Why this personality matters**:
- Protecting raw data prevents accidental corruption of source material
- The fact-checking and source-tracking skills enforce intellectual rigor
- High thinking level is essential — research analysis requires deep, careful reasoning
- The agent becomes a research assistant, not a code monkey

---

### 8. Data Science / Machine Learning

**Context**: Jupyter notebooks, Python scripts, model training pipelines, and datasets. The work involves experimentation, iteration, and reproducibility concerns.

**Guardrails**:

```json
{
  "name": "data-science",
  "label": "Data Science / Machine Learning",
  "guardrails": {
    "allowedWritePaths": ["notebooks/", "src/", "models/", "experiments/", "reports/", "tests/", "docs/", "*.py", "*.ipynb", "*.md"],
    "deniedWritePaths": [".github/workflows/", ".GITCLAW/lifecycle/", ".GITCLAW/.pi/", "data/raw/", "*.pkl", "*.h5", "*.pt"],
    "protectedFiles": ["requirements.txt", "pyproject.toml", "Dockerfile", "dvc.yaml"],
    "deniedCommands": ["rm -rf /", "sudo"],
    "maxFileSize": "10MB",
    "branchProtection": "recommended"
  },
  "settings": {
    "defaultThinkingLevel": "high",
    "preferredSkills": ["code-review", "experiment-tracker", "data-validation", "notebook-reviewer"]
  }
}
```

**System prompt additions**:
> You are working in a data science / machine learning repository. Reproducibility is critical — always be explicit about random seeds, data splits, hyperparameters, and environment versions. Never modify raw data files or trained model binaries directly. When reviewing notebooks, check for cell execution order dependencies, hidden state, and missing documentation. When suggesting model changes, explain the trade-offs (accuracy vs. latency, complexity vs. interpretability). Track experiment parameters and results in your memory to enable comparison across runs. Be cautious with large computations — suggest profiling before scaling up.

---

### 9. Legal / Compliance / Policy

**Context**: A repository containing legal documents, compliance policies, regulatory filings, or contract templates. Precision of language is paramount. Changes have real-world legal consequences.

**Guardrails**:

```json
{
  "name": "legal-compliance",
  "label": "Legal / Compliance / Policy",
  "guardrails": {
    "allowedWritePaths": ["drafts/", "reviews/", "notes/", "templates/", "docs/", "*.md", "*.txt", "*.docx"],
    "deniedWritePaths": [".github/workflows/", ".GITCLAW/lifecycle/", ".GITCLAW/.pi/", "final/", "signed/", "executed/"],
    "protectedFiles": ["final/*", "signed/*", "executed/*"],
    "deniedCommands": ["npm", "pip", "bun", "cargo", "go", "make"],
    "maxFileSize": "5MB",
    "branchProtection": "strongly-recommended"
  },
  "settings": {
    "defaultThinkingLevel": "high",
    "preferredSkills": ["editorial-review", "defined-terms-tracker", "cross-reference-checker"]
  },
  "toolRestrictions": {
    "shellAccess": "none",
    "fileOps": "read-write",
    "gitOps": "read-only"
  }
}
```

**System prompt additions**:
> You are working in a legal/compliance repository. Precision of language is paramount — every word may have legal significance. Never modify files in `final/`, `signed/`, or `executed/` directories. When suggesting edits, explain the legal rationale and flag any ambiguity you introduce. Track defined terms and ensure consistent usage across documents. When reviewing contracts or policies, check for internal contradictions, undefined terms, missing definitions, and vague language. You are not a lawyer and must never present your analysis as legal advice — always recommend human legal review for substantive changes. Focus on structural clarity, consistency, and completeness.

---

### 10. Design System / UI Library

**Context**: A component library, design system, or Storybook-based UI repository focused on visual consistency, accessibility, and documentation.

**Guardrails**:

```json
{
  "name": "design-system",
  "label": "Design System / UI Component Library",
  "guardrails": {
    "allowedWritePaths": ["src/components/", "src/tokens/", "stories/", "tests/", "docs/", "*.tsx", "*.ts", "*.css", "*.scss", "*.mdx"],
    "deniedWritePaths": [".github/workflows/", ".GITCLAW/lifecycle/", ".GITCLAW/.pi/"],
    "protectedFiles": ["package.json", "tsconfig.json"],
    "deniedCommands": ["rm -rf /", "sudo"],
    "maxFileSize": "2MB",
    "branchProtection": "recommended"
  },
  "settings": {
    "defaultThinkingLevel": "high",
    "preferredSkills": ["code-review", "accessibility-check", "visual-consistency", "documentation-generator"]
  }
}
```

**System prompt additions**:
> You are working in a design system / UI component library. Every component must be accessible (WCAG 2.1 AA minimum), well-documented, and visually consistent with the design tokens. When creating or modifying components, always include proper ARIA attributes, keyboard navigation, and screen reader support. When reviewing, check for hardcoded colors (should use tokens), hardcoded spacing (should use scale), and missing documentation. Story coverage is as important as test coverage — every component variant needs a story.

---

## Implementation Architecture

### Installer Flow

The enhanced installer flow adds a personality selection step:

```
$ bun .GITCLAW/install/GITCLAW-INSTALLER.ts

🔧 Installing gitclaw into this repository...

What kind of project is this?

  1. Single File HTML Application
  2. Browser Extension (Chrome/Firefox)
  3. Python Website (Django/Flask/FastAPI)
  4. Node.js / TypeScript Application
  5. Mobile Application (React Native / Flutter)
  6. Writing (Novel / Screenplay / Blog)
  7. Research & Analysis
  8. Data Science / Machine Learning
  9. Legal / Compliance / Policy
  10. Design System / UI Library
  11. General (no restrictions)
  12. Custom (I'll configure manually)

> 3

✅ Personality: Python Website
   - Write access: src/, app/, templates/, static/, tests/, docs/, migrations/
   - Protected files: requirements.txt, pyproject.toml, Dockerfile
   - Skills: code-review, test-generator, dependency-audit, security-audit
   - Branch protection: recommended

Workflows:
  ✅ .github/workflows/agent.yml installed

Issue templates:
  ✅ .github/ISSUE_TEMPLATE/hatch.md installed

Agent identity:
  ✅ .GITCLAW/AGENTS.md installed

Personality:
  ✅ .GITCLAW/.gitclaw-personality.json installed
  ✅ System prompt additions applied

✨ gitclaw installed with python-website personality!
```

### Runtime Enforcement

Personality guardrails must be enforced at runtime, not just documented. There are three enforcement layers:

#### Layer 1: System Prompt (LLM-enforced)

The personality's `APPEND_SYSTEM.md` additions are injected into the system prompt. This is soft enforcement — the LLM follows the guidelines but could theoretically be convinced to violate them.

**Strength**: Handles nuance and context well. The agent understands _why_ it shouldn't modify raw data, not just that it shouldn't.

**Weakness**: Prompt injection could override. Not reliable as a sole constraint.

#### Layer 2: Pre-Commit Validation (System-enforced)

Before committing, a validation step checks that the agent's changes conform to the personality's path guardrails:

```typescript
function validateChanges(personality: Personality, changedFiles: string[]): ValidationResult {
  const violations: string[] = [];

  for (const file of changedFiles) {
    // Check denied write paths
    if (personality.guardrails.deniedWritePaths.some(p => matchGlob(file, p))) {
      violations.push(`Denied path: ${file}`);
      continue;
    }

    // Check allowed write paths
    if (!personality.guardrails.allowedWritePaths.some(p => matchGlob(file, p))) {
      violations.push(`Outside allowed paths: ${file}`);
    }

    // Check protected files
    if (personality.guardrails.protectedFiles?.some(p => matchGlob(file, p))) {
      violations.push(`Protected file modified: ${file}`);
    }
  }

  return { valid: violations.length === 0, violations };
}
```

**Strength**: Hard enforcement. The agent cannot commit changes that violate the guardrails regardless of what the LLM decides.

**Weakness**: Binary — either allows or blocks, no nuance. Protected files need a "propose, don't commit" pathway.

#### Layer 3: Post-Push Audit (Detective control)

A lightweight post-push check records any guardrail-relevant activity:

```typescript
// After push, log personality compliance
const auditEntry = {
  timestamp: new Date().toISOString(),
  issueNumber,
  filesChanged: changedFiles,
  personalityName: personality.name,
  violations: validationResult.violations,
  overrides: [] // any user-approved exceptions
};

appendFileSync(resolve(stateDir, "personality-audit.jsonl"),
  JSON.stringify(auditEntry) + "\n");
```

**Strength**: Full audit trail. Even if guardrails are relaxed, the record exists.

**Weakness**: Not preventative — detective only.

### The Three Layers Together

| Layer | Enforcement | Handles Nuance | Bypassable | When |
|---|---|---|---|---|
| System Prompt | LLM-enforced | ✅ Yes | ⚠️ Possible | During generation |
| Pre-Commit Validation | System-enforced | ❌ No | ❌ No | Before commit |
| Post-Push Audit | Detective | N/A | N/A | After push |

The combination is robust: the LLM tries to do the right thing (Layer 1), the system prevents mistakes (Layer 2), and everything is auditable (Layer 3).

---

## Personality Evolution

### Re-Personality (Like Re-Hatching)

Just as the agent can re-hatch to get a new identity, it should be possible to re-personality to change the project type:

```bash
bun .GITCLAW/install/GITCLAW-INSTALLER.ts --personality browser-extension
```

This updates `.gitclaw-personality.json` and the system prompt additions without affecting the agent's identity, memory, or session history.

### Personality Composition

Some projects are hybrids. A project might be a Python website _and_ a documentation project _and_ a data analysis pipeline. The personality system should support composition:

```json
{
  "personalities": ["python-website", "analysis"],
  "mergeStrategy": "union-allow-intersect-deny"
}
```

- **Allowed paths**: union of both personalities (widest access)
- **Denied paths**: intersection of both (strictest constraints)
- **Protected files**: union of both (most protection)
- **Skills**: union of both (most capabilities)

### Custom Personalities

For projects that don't fit any template, the `custom` option lets the user configure guardrails manually:

```bash
bun .GITCLAW/install/GITCLAW-INSTALLER.ts --personality custom
```

This generates a blank `.gitclaw-personality.json` with all fields commented and documented, ready for manual configuration.

---

## What This Changes About .GITCLAW

### Before Personalities

1. Install `.GITCLAW/`
2. Agent has full, unrestricted access
3. Guardrails are generic
4. The agent discovers the project type through conversation
5. Every mistake is possible until the operator learns to guide the agent

### After Personalities

1. Install `.GITCLAW/` with a personality
2. Agent has scoped, appropriate access from the first session
3. Guardrails match the project type
4. The agent understands the project type before the first conversation
5. Whole categories of mistakes are structurally prevented

The insight is that **installation is a configuration event**, not just a deployment event. Right now, the installer copies files. With personalities, the installer _configures a context-aware agent_ — and that configuration persists as a guardrail for every subsequent session.

---

## Priority Personalities for First Release

Not all personalities need to ship at once. The highest-impact ones for a first release:

| Priority | Personality | Rationale |
|---|---|---|
| 🔴 P0 | **General** (current default) | Must exist — backward compatibility |
| 🔴 P0 | **Writing** | Largest non-obvious use case; proves .GITCLAW isn't just for code |
| 🟡 P1 | **Python Website** | Most common web framework; good template for other language personalities |
| 🟡 P1 | **Single File HTML** | Simplest programming project type; good starting point |
| 🟡 P1 | **Research & Analysis** | High-value non-programming use case |
| 🟢 P2 | **Browser Extension** | Niche but strong guardrail story |
| 🟢 P2 | **Node.js / TypeScript** | Natural second language personality after Python |
| 🟢 P2 | **Data Science / ML** | Growing demand; strong guardrail needs |
| 🟢 P2 | **Legal / Compliance** | Demonstrates .GITCLAW versatility beyond tech |
| 🟢 P2 | **Design System** | Good component library use case |
| 🟢 P2 | **Mobile Application** | Complex guardrail needs; demonstrates depth |

---

## The Bottom Line

The personality system transforms `.GITCLAW` from "an AI agent that happens to live in your repo" to "an AI agent that _understands_ your repo from the moment it wakes up."

The technical implementation is modest — a selection prompt in the installer, a JSON manifest, a pre-commit validation step, and per-personality system prompt additions. The architecture already supports everything needed: `APPEND_SYSTEM.md` handles behavioral guidance, `settings.json` handles configuration, and the skill system handles capabilities.

What changes is the _default experience_. Instead of an omnipotent agent that needs to be taught boundaries through trial and error, you get a scoped collaborator that already knows what kind of project it's in, what files it should and shouldn't touch, and what skills are most relevant.

The non-programming personalities are especially important. They prove that `.GITCLAW` is not a coding tool — it's a _git-native AI collaboration system_. A novelist, a researcher, a legal team, and a data scientist can all use the same infrastructure, with guardrails that match their domain.

**Power is most useful when it's focused.** Personalities focus the power.

🦞 _Different repos, different claws._

---

_Last updated: 2026-02-22_
