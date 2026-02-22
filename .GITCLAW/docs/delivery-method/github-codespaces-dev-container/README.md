# .GITCLAW 🦞 GitHub Codespaces / Dev Container

### Pre-configure gitclaw in a dev container definition — so that when a user opens a Codespace, gitclaw is automatically installed and ready to use.

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

---

## Overview

GitHub Codespaces provides cloud-hosted development environments that can be launched instantly from any GitHub repository. Dev Containers are the configuration standard that defines what tools, extensions, and setup steps are included in these environments.

The Codespaces / Dev Container delivery method pre-configures gitclaw as part of a dev container definition. When a user opens a Codespace (or any Dev Container-compatible environment), gitclaw is automatically installed — no manual steps, no CLI commands, no local setup.

This method is particularly powerful for workshops, demos, onboarding, and teams that want every developer to have gitclaw available from the moment they open their development environment.

---

## How It Works

### Option A — Dev Container Feature

Dev Container Features are self-contained units of installation code that can be added to any `devcontainer.json`. Publishing gitclaw as a Dev Container Feature makes it installable with a single line:

#### Usage

```json
// .devcontainer/devcontainer.json
{
  "name": "My Project",
  "image": "mcr.microsoft.com/devcontainers/base:ubuntu",
  "features": {
    "ghcr.io/japer-technology/gitclaw/gitclaw:1": {
      "provider": "anthropic",
      "model": "claude-sonnet-4-20250514",
      "thinking": "high"
    }
  }
}
```

When the Codespace or Dev Container is created, the Feature's install script runs automatically:

1. Downloads the latest `.GITCLAW/` folder.
2. Copies it into the workspace.
3. Runs the installer to set up workflows and templates.
4. Configures the agent with the specified options.

#### Feature Definition

```
gitclaw-feature/
├── devcontainer-feature.json
├── install.sh
└── README.md
```

**`devcontainer-feature.json`:**

```json
{
  "id": "gitclaw",
  "version": "1.2.0",
  "name": "GitClaw",
  "description": "AI-powered GitHub assistant",
  "documentationURL": "https://github.com/japer-technology/gitclaw",
  "options": {
    "provider": {
      "type": "string",
      "default": "anthropic",
      "description": "LLM provider"
    },
    "model": {
      "type": "string",
      "default": "claude-sonnet-4-20250514",
      "description": "Model identifier"
    },
    "thinking": {
      "type": "string",
      "default": "high",
      "enum": ["high", "medium", "low"],
      "description": "Thinking mode"
    },
    "version": {
      "type": "string",
      "default": "latest",
      "description": "gitclaw version to install"
    }
  },
  "installsAfter": [
    "ghcr.io/devcontainers/features/node:1"
  ]
}
```

**`install.sh`:**

```bash
#!/bin/bash
set -e

VERSION="${VERSION:-latest}"
PROVIDER="${PROVIDER:-anthropic}"
MODEL="${MODEL:-claude-sonnet-4-20250514}"
THINKING="${THINKING:-high}"

echo "🦞 Installing gitclaw ${VERSION}..."

# Download gitclaw
if [ "$VERSION" = "latest" ]; then
  VERSION=$(curl -s https://api.github.com/repos/japer-technology/gitclaw/releases/latest | jq -r .tag_name)
fi

curl -sL "https://github.com/japer-technology/gitclaw/releases/download/${VERSION}/gitclaw.tar.gz" -o /tmp/gitclaw.tar.gz
mkdir -p /tmp/gitclaw
tar xzf /tmp/gitclaw.tar.gz -C /tmp/gitclaw

# Create post-create script that runs in the workspace context
cat > /usr/local/share/gitclaw-setup.sh << 'EOF'
#!/bin/bash
if [ ! -d "${containerWorkspaceFolder:-.}/.GITCLAW" ]; then
  echo "🦞 Setting up gitclaw in workspace..."
  cp -r /tmp/gitclaw/.GITCLAW/ "${containerWorkspaceFolder:-.}/.GITCLAW/"

  if command -v bun &> /dev/null; then
    bun "${containerWorkspaceFolder:-.}/.GITCLAW/install/GITCLAW-INSTALLER.ts"
  elif command -v npx &> /dev/null; then
    npx tsx "${containerWorkspaceFolder:-.}/.GITCLAW/install/GITCLAW-INSTALLER.ts"
  fi

  echo "🦞 gitclaw installed! Add your API key secret and open an issue to get started."
fi
EOF
chmod +x /usr/local/share/gitclaw-setup.sh

echo "🦞 gitclaw feature installed. Will set up workspace on first open."
```

### Option B — postCreateCommand

A simpler approach without a custom Feature — use the `postCreateCommand` in `devcontainer.json`:

```json
{
  "name": "My Project with GitClaw",
  "image": "mcr.microsoft.com/devcontainers/typescript-node:20",
  "postCreateCommand": "npx gitclaw init --no-interactive --provider anthropic --model claude-sonnet-4-20250514",
  "features": {
    "ghcr.io/devcontainers/features/node:1": {}
  }
}
```

This runs after the container is created, installing gitclaw into the workspace.

### Option C — Dockerfile-Based

For teams that maintain their own dev container images:

```dockerfile
# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/typescript-node:20

# Pre-install gitclaw globally
RUN npm install -g @japer-technology/gitclaw

# The init command runs via postCreateCommand
```

```json
{
  "name": "My Project",
  "build": {
    "dockerfile": "Dockerfile"
  },
  "postCreateCommand": "gitclaw init --no-interactive"
}
```

---

## Dev Container Lifecycle

Understanding when gitclaw setup runs in the container lifecycle:

```
┌─────────────────────────────────────────────┐
│           Dev Container Lifecycle            │
│                                              │
│  1. Build image                              │
│     └── Features are installed               │
│         └── gitclaw Feature's install.sh     │
│                                              │
│  2. Create container                         │
│     └── Container starts with features       │
│                                              │
│  3. Clone/mount workspace                    │
│     └── Repository files available           │
│                                              │
│  4. postCreateCommand                        │
│     └── gitclaw init runs here               │
│         ├── Copies .GITCLAW/ to workspace    │
│         ├── Runs installer                   │
│         └── Reports success                  │
│                                              │
│  5. postStartCommand                         │
│     └── Runs on every container start        │
│                                              │
│  6. postAttachCommand                        │
│     └── Runs when user attaches              │
│                                              │
│  Developer is ready to work                  │
└─────────────────────────────────────────────┘
```

### Lifecycle Hooks

| Hook | When it runs | Best for |
|------|-------------|----------|
| `postCreateCommand` | Once, after container creation | Installing gitclaw (first time setup) |
| `postStartCommand` | Every time container starts | Verifying gitclaw is configured |
| `postAttachCommand` | Every time a user attaches | Displaying welcome message with gitclaw status |
| Feature `install.sh` | During image build | Pre-installing gitclaw binaries/dependencies |

---

## Publishing as a Dev Container Feature

### Feature Repository Structure

```
gitclaw-devcontainer-features/
├── src/
│   └── gitclaw/
│       ├── devcontainer-feature.json
│       ├── install.sh
│       └── README.md
├── test/
│   └── gitclaw/
│       ├── test.sh
│       └── scenarios.json
└── .github/
    └── workflows/
        ├── test.yml
        └── release.yml
```

### Publishing to GHCR (GitHub Container Registry)

Dev Container Features are published as OCI artifacts to container registries:

```yaml
# .github/workflows/release.yml
name: Release Features
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

      - name: Publish Features
        uses: devcontainers/action@v1
        with:
          publish-features: true
          base-path-to-features: src
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

Once published, the Feature is available at:

```
ghcr.io/japer-technology/gitclaw-devcontainer-features/gitclaw:1
```

---

## Compatible Environments

Dev Container Features work in multiple environments, not just GitHub Codespaces:

| Environment | Support | Notes |
|-------------|---------|-------|
| **GitHub Codespaces** | ✅ Full support | Native integration |
| **VS Code Dev Containers** | ✅ Full support | Local Docker-based |
| **JetBrains Gateway** | ✅ Supported | Remote development |
| **DevPod** | ✅ Supported | Open-source alternative |
| **Gitpod** | ⚠️ Partial | Different config format, but concepts apply |
| **CodeSandbox** | ⚠️ Partial | Different config format |

---

## Template Repository with Dev Container

Combine the Dev Container approach with a template repository:

```
gitclaw-codespace-template/
├── .devcontainer/
│   └── devcontainer.json       # Pre-configured with gitclaw Feature
├── .GITCLAW/                   # Pre-installed gitclaw
├── .github/
│   ├── workflows/              # Pre-installed workflows
│   └── ISSUE_TEMPLATE/         # Pre-installed templates
├── README.md                   # Getting started guide
└── .gitignore
```

Users click **"Use this template"** → **"Open in a codespace"** → gitclaw is immediately available.

```
┌──────────────────────────────────────┐
│  GitHub Repository Page               │
│                                       │
│  [Use this template ▼]               │
│  ├── Create a new repository          │
│  └── Open in a codespace ← This one │
└──────────────────────────────────────┘
```

---

## Strengths

- **Zero local setup** — Everything happens in the cloud. No local installations, no runtime requirements, no configuration.
- **Instant onboarding** — Open a Codespace and gitclaw is ready. Ideal for new team members, workshop participants, and quick demos.
- **Consistent environments** — Every developer gets the exact same gitclaw setup, eliminating "works on my machine" issues.
- **GitHub-native** — Codespaces is a GitHub product, so the integration feels seamless.
- **Dev Container Feature ecosystem** — Features are a growing standard supported by multiple tools and platforms.
- **No runtime maintenance** — The Codespace environment includes everything needed; users don't manage Node.js or Bun versions.
- **Pairs with templates** — A template repo with a Dev Container creates a "one-click to fully working environment" experience.

---

## Limitations

- **Codespaces adoption** — Not all developers use Codespaces. Those working with local development environments won't benefit from this method directly (though VS Code Dev Containers help).
- **Dev Container Features are relatively new** — The spec is still evolving, and the ecosystem is smaller than npm or GitHub Actions.
- **gitclaw runs via Actions, not in the Codespace** — The Dev Container sets up the configuration files, but the gitclaw agent still runs via GitHub Actions workflows. The Codespace is for development, not for running the agent.
- **Cost** — Codespaces have usage-based pricing (free tier for personal accounts, charged for organizations). The gitclaw Feature adds a few seconds to container creation time.
- **Internet dependency** — The Feature downloads gitclaw during container creation, requiring internet access.
- **Rebuild required for updates** — To get a new gitclaw version, the Codespace needs to be rebuilt (or the user runs an update command manually).

---

## Security Considerations

- **Feature source trust** — Only use Features from trusted registries. The `ghcr.io/japer-technology/` namespace should be verified.
- **Install script auditing** — The Feature's `install.sh` runs with elevated permissions during container build. Keep the script minimal and auditable.
- **No secrets in Dev Container config** — Never put API keys or tokens in `devcontainer.json`. Use Codespaces Secrets (encrypted, per-user).
- **Network access** — The Feature downloads files during build. Ensure the download URLs are HTTPS and the artifacts are verified.
- **Container isolation** — Codespaces containers are isolated per user. One user's gitclaw setup cannot affect another's.

---

## When to Use This Method

This method is ideal when:

- Your team uses **GitHub Codespaces** or **VS Code Dev Containers** as their primary development environment.
- You want **zero-friction onboarding** for new developers or workshop participants.
- You need **consistent, reproducible** gitclaw setups across all developers.
- You are creating a **template repository** that should include a complete development environment with gitclaw.

---

## When to Consider Alternatives

Consider a different delivery method when:

- Your team works **locally** and doesn't use Codespaces or Dev Containers.
- You need gitclaw on **existing repositories** that don't have Dev Container configurations.
- You want the **simplest possible method** and your team is comfortable with CLI (use [CLI tool](./cli-tool.md)).

---

## Related Methods

- [GitHub Template Repository](./github-template-repository.md) — Combine with a Dev Container for the ultimate "one-click" experience.
- [CLI Tool (npx / bunx)](./cli-tool.md) — What the `postCreateCommand` runs under the hood.
- [Package Registry](./package-registry.md) — The Feature could install gitclaw from npm during container build.
- [GitHub Marketplace Action](./github-marketplace-action.md) — The workflows that gitclaw installs still run via Actions, regardless of how the files were delivered.

---

> 🦞 *Open a Codespace, and gitclaw is already there — zero setup, instant productivity.*
