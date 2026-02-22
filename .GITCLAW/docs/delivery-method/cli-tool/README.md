# .GITCLAW 🦞 CLI Tool (npx / bunx)

### A single command that downloads and installs gitclaw into the current directory — the developer-native way to bootstrap an AI assistant.

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

---

## Overview

The CLI Tool delivery method wraps the existing gitclaw installer into a publishable npm package that can be executed with a single command. Developers run `npx gitclaw init` (or `bunx gitclaw init`) in their project directory, and the tool handles everything: downloading the latest `.GITCLAW/` folder, running the installer, configuring workflows, and optionally guiding the user through an interactive setup wizard.

This follows the well-established pattern used by popular tools like `create-react-app`, `astro init`, `nuxt init`, and `shadcn-ui` — a single command that scaffolds an entire project or feature.

---

## How It Works

### Basic Usage

```bash
# Using npx (Node.js)
npx gitclaw init

# Using bunx (Bun)
bunx gitclaw init
```

### What the CLI Does

When `gitclaw init` runs, it performs the following steps:

```
1. Check prerequisites
   ├── Verify git repository exists
   ├── Verify .GITCLAW/ doesn't already exist
   └── Check runtime (Node.js or Bun)

2. Download latest .GITCLAW/
   ├── Fetch latest release from GitHub API
   ├── Download release artifact (tar.gz)
   └── Extract to current directory

3. Run installer
   ├── Execute GITCLAW-INSTALLER.ts
   ├── Copy workflows to .github/workflows/
   ├── Copy templates to .github/ISSUE_TEMPLATE/
   └── Generate configuration files

4. Interactive configuration (optional)
   ├── Select LLM provider
   ├── Choose model
   ├── Set thinking mode
   └── Configure agent personality

5. Post-install instructions
   ├── Display next steps
   ├── Remind to add API key secret
   └── Link to documentation
```

### Interactive Mode

When run without flags, the CLI enters an interactive mode:

```
$ npx gitclaw init

  🦞 gitclaw installer v1.2.0

  ? Select your LLM provider:
  ❯ Anthropic (Claude)
    OpenAI (GPT)
    Other

  ? Select model:
  ❯ claude-sonnet-4-20250514
    claude-sonnet-4-20250514
    claude-3-5-haiku-20241022

  ? Thinking mode:
  ❯ high
    medium
    low

  ✅ Downloaded .GITCLAW/ (v1.2.0)
  ✅ Installed workflows to .github/workflows/
  ✅ Installed templates to .github/ISSUE_TEMPLATE/
  ✅ Generated configuration

  Next steps:
  1. Add your API key as a repository secret:
     → Settings → Secrets → Actions → New secret
     → Name: ANTHROPIC_API_KEY
     → Value: <your-api-key>

  2. Commit and push:
     git add . && git commit -m "feat: install gitclaw" && git push

  3. Open an issue to talk to your agent! 🦞
```

### Flag-Based Mode

For scripting and CI/CD, the CLI supports flags:

```bash
npx gitclaw init \
  --provider anthropic \
  --model claude-sonnet-4-20250514 \
  --thinking high \
  --no-interactive
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `gitclaw init` | Install gitclaw in the current repository |
| `gitclaw update` | Update gitclaw to the latest version |
| `gitclaw doctor` | Check the installation for common issues |
| `gitclaw version` | Display the installed and latest versions |
| `gitclaw config` | View or edit configuration |

### `gitclaw update`

Downloads the latest `.GITCLAW/` folder and re-runs the installer, preserving user-modified configuration:

```bash
$ npx gitclaw update

  🦞 Checking for updates...
  Current: v1.1.0
  Latest:  v1.2.0

  ? Update to v1.2.0? (Y/n) Y

  ✅ Updated .GITCLAW/ to v1.2.0
  ✅ Updated workflows
  ✅ Configuration preserved

  Review changes with: git diff
  Commit with: git add . && git commit -m "chore: update gitclaw to v1.2.0"
```

### `gitclaw doctor`

Diagnoses common installation issues:

```bash
$ npx gitclaw doctor

  🦞 gitclaw doctor

  ✅ .GITCLAW/ folder exists
  ✅ Workflows installed in .github/workflows/
  ✅ Issue templates installed
  ⚠️  API key secret not detected (cannot verify from local machine)
  ✅ gitclaw version: v1.2.0 (latest)
  ✅ Git repository detected
  ✅ Remote: origin → github.com/user/repo

  1 warning. See documentation for details.
```

---

## Package Structure

### npm Package Layout

```
gitclaw/
├── package.json
├── bin/
│   └── gitclaw.js          # CLI entry point
├── src/
│   ├── cli.ts              # Command parser and router
│   ├── commands/
│   │   ├── init.ts          # Init command logic
│   │   ├── update.ts        # Update command logic
│   │   ├── doctor.ts        # Doctor command logic
│   │   └── config.ts        # Config command logic
│   ├── download.ts          # Release artifact downloader
│   ├── installer.ts         # Wrapper around GITCLAW-INSTALLER.ts
│   ├── prompts.ts           # Interactive prompts (using inquirer/prompts)
│   └── utils.ts             # Shared utilities
├── templates/
│   └── .GITCLAW/            # Bundled fallback (optional)
├── tsconfig.json
└── README.md
```

### `package.json`

```json
{
  "name": "gitclaw",
  "version": "1.2.0",
  "description": "Install and manage gitclaw — AI-powered GitHub assistant",
  "bin": {
    "gitclaw": "./bin/gitclaw.js"
  },
  "keywords": ["github", "ai", "assistant", "copilot", "agent"],
  "repository": "japer-technology/gitclaw",
  "license": "MIT",
  "engines": {
    "node": ">=18.0.0"
  },
  "dependencies": {
    "commander": "^12.0.0",
    "@clack/prompts": "^0.7.0"
  }
}
```

---

## Implementation Details

### Download Logic

The CLI downloads the latest `.GITCLAW/` release from GitHub:

```typescript
async function downloadLatest(version: string = 'latest'): Promise<string> {
  const releaseUrl = version === 'latest'
    ? 'https://api.github.com/repos/japer-technology/gitclaw/releases/latest'
    : `https://api.github.com/repos/japer-technology/gitclaw/releases/tags/${version}`;

  const release = await fetch(releaseUrl).then(r => r.json());
  const asset = release.assets.find(a => a.name === 'gitclaw.tar.gz');

  const response = await fetch(asset.browser_download_url);
  const buffer = await response.arrayBuffer();

  // Extract to current directory
  await extractTarGz(buffer, process.cwd());

  return release.tag_name;
}
```

### Configuration Preservation During Updates

When updating, the CLI:

1. Reads the user's current configuration from `.GITCLAW/`.
2. Downloads and extracts the new version.
3. Merges user-modified settings back into the new configuration.
4. Reports any configuration changes that need attention.

```typescript
async function update() {
  // 1. Save current config
  const currentConfig = await readConfig('.GITCLAW/config');

  // 2. Download new version
  await downloadLatest();

  // 3. Restore user config
  await mergeConfig('.GITCLAW/config', currentConfig);

  // 4. Re-run installer
  await runInstaller();
}
```

### Error Handling

The CLI handles common failure modes gracefully:

| Error | Behavior |
|-------|----------|
| Not in a git repository | Displays an error with instructions to run `git init` |
| `.GITCLAW/` already exists | Suggests using `gitclaw update` instead |
| Network failure | Falls back to a bundled version if available, or provides offline instructions |
| Unsupported Node.js version | Displays minimum version requirement |
| Permission denied | Suggests checking file permissions or running with appropriate access |

---

## Distribution

### Publishing to npm

```bash
# Build the package
npm run build

# Publish to npm
npm publish
```

Once published, the package is available globally:

```bash
npx gitclaw init        # Downloads and runs without global install
npm install -g gitclaw   # Global install for frequent use
```

### Publishing to GitHub Packages

For staying within the GitHub ecosystem:

```bash
npm publish --registry=https://npm.pkg.github.com
```

Users install with:

```bash
npx @japer-technology/gitclaw init
```

---

## Strengths

- **One command** — A single `npx gitclaw init` replaces the multi-step fork/copy/install process. Familiar to any developer who has used `npx create-*` tools.
- **Interactive wizard** — Guides users through configuration choices, reducing setup errors.
- **Flag-based scripting** — Supports non-interactive mode for CI/CD and automation scripts.
- **Built-in updates** — `gitclaw update` makes upgrading painless, with configuration preservation.
- **Diagnostics** — `gitclaw doctor` helps users troubleshoot issues without reading documentation.
- **No fork required** — Works directly in any existing repository.
- **Version control** — The npm package is versioned, so users can install specific versions.
- **Offline fallback** — If a bundled version is included, the CLI can work without network access.

---

## Limitations

- **Requires Node.js or Bun** — The CLI needs a JavaScript runtime installed on the developer's machine. This is the same requirement as the current installer.
- **Not a one-click experience** — Still requires opening a terminal and running a command. Less accessible than a GitHub App or template.
- **npm dependency** — Adds a dependency on the npm registry (or GitHub Packages). If the registry is down, `npx` won't work.
- **No GUI** — Purely terminal-based. Developers who prefer visual interfaces should use the [GitHub Pages portal](./github-pages-self-service-portal.md) or [third-party website](./third-party-website.md).
- **Package maintenance** — The npm package must be kept in sync with gitclaw releases. This is an additional release step.

---

## Security Considerations

- **npm supply-chain security** — The package should be published with 2FA enabled on the npm account. Consider using npm provenance for transparency.
- **Download verification** — The CLI should verify checksums of downloaded artifacts to prevent tampering.
- **No credential handling** — The CLI should never ask for or store API keys. It should only instruct the user to add them as GitHub secrets.
- **Minimal dependencies** — Keep the dependency tree small to reduce the attack surface. Audit dependencies regularly.

---

## When to Use This Method

This method is ideal when:

- You are a developer comfortable with the **command line**.
- You want to add gitclaw to an **existing repository** quickly.
- You want an **interactive setup** experience that guides you through configuration.
- You want to **script** gitclaw installation in CI/CD or automation workflows.
- You prefer a **familiar pattern** (like `npx create-react-app` or `npx astro init`).

---

## When to Consider Alternatives

Consider a different delivery method when:

- Your users are **not comfortable with the terminal** (consider a [GitHub App](./github-application.md) or [template repository](./github-template-repository.md)).
- You need a **zero-install, browser-only** experience (consider [GitHub Pages portal](./github-pages-self-service-portal.md)).
- You want **automatic updates** without any user action (consider a [GitHub App](./github-application.md)).

---

## Related Methods

- [Fork / Import + Installer](./fork-import-installer.md) — The manual process this CLI automates.
- [Package Registry](./package-registry.md) — Uses npm as a distribution channel, but as a dependency rather than a CLI.
- [GitHub Marketplace Action](./github-marketplace-action.md) — Similar automation, but runs in GitHub Actions instead of locally.
- [GitHub Pages Portal](./github-pages-self-service-portal.md) — A visual alternative that could generate CLI commands.

---

> 🦞 *`npx gitclaw init` — three words, one command, and your repo has an AI assistant.*
