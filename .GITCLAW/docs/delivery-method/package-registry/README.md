# .GITCLAW 🦞 Package Registry (npm / GitHub Packages)

### Publish the `.GITCLAW/` folder as a versioned package on npm or GitHub Packages — leveraging the mature package ecosystem for distribution and updates.

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

---

## Overview

The Package Registry method publishes gitclaw as a versioned npm package (or GitHub Package). Developers install it as a dev dependency in their project, and a postinstall script or explicit `init` command copies the `.GITCLAW/` folder into the repository root and runs the installer.

This approach treats gitclaw as a project dependency — a familiar concept for JavaScript/TypeScript developers — and leverages npm's version management, update detection, and ecosystem tooling (Dependabot, Renovate, `npm outdated`).

---

## How It Works

### Step 1 — Install the Package

```bash
# From npm
npm install --save-dev @japer-technology/gitclaw

# From GitHub Packages
npm install --save-dev @japer-technology/gitclaw --registry=https://npm.pkg.github.com

# Using Bun
bun add --dev @japer-technology/gitclaw
```

### Step 2 — Run the Init Command

After installing the package, run the init command to copy `.GITCLAW/` and set up workflows:

```bash
npx gitclaw init
```

Or, if using a postinstall script (automatic):

```json
{
  "scripts": {
    "postinstall": "gitclaw init --silent"
  }
}
```

### Step 3 — Init Command Behavior

The `init` command:

1. Copies the `.GITCLAW/` folder from `node_modules/@japer-technology/gitclaw/.GITCLAW/` to the repository root.
2. Runs the installer to set up `.github/workflows/` and `.github/ISSUE_TEMPLATE/`.
3. Displays next steps (add API key, commit, push).

```
$ npx gitclaw init

  🦞 gitclaw v1.2.0

  ✅ Copied .GITCLAW/ to repository root
  ✅ Installed workflows to .github/workflows/
  ✅ Installed templates to .github/ISSUE_TEMPLATE/

  Next steps:
  1. git add . && git commit -m "feat: install gitclaw"
  2. Add your API key as a repository secret
  3. Open an issue to interact with gitclaw!
```

### Step 4 — Commit and Push

```bash
git add .
git commit -m "feat: install gitclaw"
git push
```

### Step 5 — Updates

When a new gitclaw version is released:

```bash
npm update @japer-technology/gitclaw
npx gitclaw init  # Re-run to copy updated files
```

Or automatically via Dependabot/Renovate.

---

## Package Structure

### npm Package Layout

```
@japer-technology/gitclaw/
├── package.json
├── bin/
│   └── gitclaw.js              # CLI entry point
├── src/
│   ├── cli.ts                  # CLI command parser
│   ├── commands/
│   │   ├── init.ts             # Init command
│   │   ├── update.ts           # Update command (re-copies files)
│   │   └── doctor.ts           # Health check
│   └── utils/
│       ├── copy.ts             # File copy logic
│       └── version.ts          # Version comparison
├── .GITCLAW/                   # The actual gitclaw folder
│   ├── install/
│   │   └── GITCLAW-INSTALLER.ts
│   ├── docs/
│   ├── config/
│   └── ...
└── README.md
```

### `package.json`

```json
{
  "name": "@japer-technology/gitclaw",
  "version": "1.2.0",
  "description": "AI-powered GitHub assistant — install and manage gitclaw",
  "bin": {
    "gitclaw": "./bin/gitclaw.js"
  },
  "files": [
    "bin/",
    "src/",
    ".GITCLAW/"
  ],
  "keywords": [
    "github",
    "ai",
    "assistant",
    "agent",
    "copilot",
    "gitclaw"
  ],
  "repository": {
    "type": "git",
    "url": "https://github.com/japer-technology/gitclaw.git"
  },
  "license": "MIT",
  "engines": {
    "node": ">=18.0.0"
  }
}
```

---

## Publishing

### To npm

```bash
# Build the package
npm run build

# Publish (scoped package)
npm publish --access public

# The package is now available at:
# https://www.npmjs.com/package/@japer-technology/gitclaw
```

### To GitHub Packages

Add to `package.json`:

```json
{
  "publishConfig": {
    "@japer-technology:registry": "https://npm.pkg.github.com"
  }
}
```

Then publish:

```bash
npm publish
```

Users configure their `.npmrc` to access GitHub Packages:

```ini
@japer-technology:registry=https://npm.pkg.github.com
```

### Release Automation

A GitHub Actions workflow can automatically publish on release:

```yaml
name: Publish Package
on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          registry-url: https://registry.npmjs.org

      - run: npm ci
      - run: npm run build
      - run: npm publish --access public
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

---

## Dependency Management Integration

### Dependabot

Dependabot can automatically detect and propose gitclaw updates:

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
    allow:
      - dependency-name: "@japer-technology/gitclaw"
```

When a new version is published, Dependabot opens a PR:

```
Bump @japer-technology/gitclaw from 1.2.0 to 1.3.0

Release notes:
- Added new agent personality options
- Fixed workflow template for Node.js 20
- Improved error messages

Dependabot will resolve any conflicts with this PR as long as you don't alter it yourself.
```

### Renovate

Renovate provides similar functionality:

```json
{
  "extends": ["config:base"],
  "packageRules": [
    {
      "matchPackageNames": ["@japer-technology/gitclaw"],
      "automerge": true,
      "automergeType": "pr"
    }
  ]
}
```

### npm outdated

Users can check for updates:

```bash
$ npm outdated

Package                   Current  Wanted  Latest
@japer-technology/gitclaw  1.2.0   1.2.0   1.3.0
```

---

## postinstall Script Approach

An alternative to the explicit `init` command is using npm's `postinstall` lifecycle script:

```json
{
  "scripts": {
    "postinstall": "gitclaw init --silent --no-overwrite"
  }
}
```

This automatically copies `.GITCLAW/` files after `npm install`. The `--no-overwrite` flag ensures user-modified files aren't replaced.

### Pros of postinstall

- **Automatic** — Files are copied every time dependencies are installed.
- **No extra step** — Users don't need to remember to run `npx gitclaw init`.
- **Keeps files in sync** — Every `npm install` ensures `.GITCLAW/` matches the package version.

### Cons of postinstall

- **Unexpected behavior** — postinstall scripts modifying files outside `node_modules/` is unconventional and may surprise developers.
- **CI/CD side effects** — In CI pipelines, `npm install` would modify the working tree, which may cause issues.
- **Security concerns** — postinstall scripts are a known attack vector. Security-conscious teams may disable them (`npm install --ignore-scripts`).

**Recommendation:** Use the explicit `init` command by default. Offer postinstall as an opt-in for teams who prefer automatic synchronization.

---

## Handling Non-JavaScript Repositories

A key challenge: gitclaw is useful for **any** GitHub repository, not just JavaScript projects. Adding an npm dependency to a Python, Go, or Rust project feels wrong.

### Solutions

| Approach | Pros | Cons |
|----------|------|------|
| **Standalone CLI** (`npx gitclaw init`) | Works without adding a dependency; uses `npx` for one-time execution | Still requires Node.js |
| **GitHub Package (universal)** | GitHub Packages supports non-npm ecosystems (Docker, Maven, etc.) | Less mature than npm for generic distribution |
| **Optional devDependency** | Listed in `package.json` only for JS projects; other projects use the CLI directly | Two different installation paths |
| **Docker image** | `docker run ghcr.io/japer-technology/gitclaw init` | Requires Docker |

**Recommendation:** Position the npm package primarily for **JavaScript/TypeScript projects** where a `package.json` already exists. For other languages, recommend the [CLI tool](./cli-tool.md) (which uses `npx` without adding a dependency) or other delivery methods.

---

## Version Synchronization

The package version should track the gitclaw version:

| Package Version | gitclaw Version | Notes |
|----------------|----------------|-------|
| `1.0.0` | `v1.0.0` | Initial release |
| `1.1.0` | `v1.1.0` | New features |
| `1.1.1` | `v1.1.0` | Package-only fix (CLI bug) |
| `1.2.0` | `v1.2.0` | gitclaw update |

The `.GITCLAW/` folder bundled in the package should always match a specific gitclaw release tag.

---

## Strengths

- **Versioned delivery** — npm provides a mature, battle-tested versioning system with semver, changelogs, and release notes.
- **Dependency management tools** — Dependabot, Renovate, and `npm outdated` automatically detect and propose updates.
- **Familiar pattern** — For JavaScript developers, `npm install` is second nature. Adding gitclaw feels natural in the project workflow.
- **`devDependency` semantics** — gitclaw is a development tool, so listing it as a `devDependency` is semantically correct.
- **GitHub Packages integration** — Publishing to GitHub Packages keeps everything within the GitHub ecosystem.
- **Reproducible installs** — `package-lock.json` ensures the exact same version is installed across environments.
- **npm ecosystem** — Discoverability via npm search, integration with CI/CD tools, and ecosystem familiarity.

---

## Limitations

- **JavaScript-centric** — Adding an npm dependency to a non-JavaScript project is awkward. The presence of `package.json` and `node_modules/` in a Go or Python project feels out of place.
- **Unconventional pattern** — npm packages typically install into `node_modules/` and are imported into code. A package that copies files to the repo root is unusual.
- **postinstall concerns** — If using postinstall scripts, they modify the working tree outside `node_modules/`, which may surprise developers and break CI expectations.
- **Requires Node.js** — The package and its CLI require Node.js to be installed. This is a non-issue for JS projects but a limitation for others.
- **Package size** — The `.GITCLAW/` folder's full contents are bundled in the package, increasing download size.
- **Two-step process** — Install the package, then run `init`. This is better than the current method but still not one-click.

---

## Security Considerations

- **npm account security** — The npm account publishing the package must use 2FA and strong credentials. A compromised account could push malicious packages.
- **npm provenance** — Enable npm provenance to link published packages to their source commits, providing supply-chain transparency.
- **Dependency audit** — Keep the package's dependency tree minimal and audit regularly with `npm audit`.
- **postinstall script review** — If using postinstall, the script should be minimal and well-documented. Users should be able to verify exactly what it does.
- **Scoped package** — Using a scoped name (`@japer-technology/gitclaw`) prevents name squatting and clarifies the package owner.
- **lockfile integrity** — Users should commit their `package-lock.json` to ensure the integrity of installed packages.

---

## When to Use This Method

This method is ideal when:

- Your project is **JavaScript/TypeScript** and already has a `package.json`.
- You want gitclaw updates managed by **Dependabot or Renovate**.
- You want a **versioned, reproducible** installation via the npm ecosystem.
- Your team prefers managing tools as **project dependencies**.

---

## When to Consider Alternatives

Consider a different delivery method when:

- Your project is **not a JavaScript project** and adding `package.json` feels wrong.
- You want a **one-click** experience without CLI commands (consider [GitHub App](./github-application.md)).
- You want a **git-native** update mechanism (consider [submodule/subtree](./git-submodule-subtree.md)).
- You prefer to avoid **npm's supply-chain risks** and want a self-contained approach (consider [fork/import installer](./fork-import-installer.md)).

---

## Related Methods

- [CLI Tool (npx / bunx)](./cli-tool.md) — Uses npm as distribution but without adding a dependency. Ideal for non-JS projects.
- [Git Submodule / Subtree](./git-submodule-subtree.md) — A git-native alternative for versioned inclusion.
- [GitHub Marketplace Action](./github-marketplace-action.md) — Another distribution channel that avoids local dependencies.
- [Fork / Import + Installer](./fork-import-installer.md) — The manual approach this method improves upon.

---

> 🦞 *`npm install @japer-technology/gitclaw` — gitclaw becomes a dependency, and your package manager handles the rest.*
