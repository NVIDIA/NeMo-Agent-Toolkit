# .GITCLAW 🦞 Git Submodule / Subtree

### Include gitclaw as a git submodule or subtree rather than copying files — enabling trivial updates and clear separation between gitclaw code and the host repository.

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

---

## Overview

Instead of copying the `.GITCLAW/` folder into a repository, the Git Submodule / Subtree method uses Git's built-in mechanisms for including external repositories as part of another repository. A submodule maintains a reference to a specific commit in the gitclaw repository, while a subtree merges the gitclaw files directly into the host repository's tree.

Both approaches provide a clear update path: when gitclaw releases a new version, updating is a single git command rather than manually re-copying files.

---

## How It Works

### Option A — Git Submodule

A git submodule is a reference to another repository at a specific commit, stored as a pointer inside the host repository.

#### Adding gitclaw as a Submodule

```bash
cd /path/to/your-repo

# Add the submodule
git submodule add https://github.com/japer-technology/gitclaw.git .GITCLAW

# This creates:
# - .GITCLAW/  (the submodule directory, pointing to gitclaw's repo)
# - .gitmodules (configuration file tracking the submodule)

# Commit the submodule reference
git add .gitmodules .GITCLAW
git commit -m "feat: add gitclaw as submodule"
```

#### What `.gitmodules` Looks Like

```ini
[submodule ".GITCLAW"]
    path = .GITCLAW
    url = https://github.com/japer-technology/gitclaw.git
    branch = main
```

#### Running the Installer

After adding the submodule, run the installer to set up workflows:

```bash
bun .GITCLAW/install/GITCLAW-INSTALLER.ts
git add .github/
git commit -m "feat: install gitclaw workflows"
git push
```

#### Cloning a Repository with a Submodule

When someone clones the repository, they need to initialize the submodule:

```bash
# Option 1: Clone with submodules
git clone --recurse-submodules https://github.com/user/my-project.git

# Option 2: Initialize after cloning
git clone https://github.com/user/my-project.git
cd my-project
git submodule init
git submodule update
```

#### Updating gitclaw via Submodule

```bash
# Update to the latest commit on the tracked branch
git submodule update --remote .GITCLAW

# Or update to a specific tag
cd .GITCLAW
git checkout v1.3.0
cd ..

# Commit the updated reference
git add .GITCLAW
git commit -m "chore: update gitclaw to v1.3.0"
git push
```

#### Pinning to a Specific Version

```bash
cd .GITCLAW
git checkout v1.2.0  # Pin to a specific tag
cd ..
git add .GITCLAW
git commit -m "chore: pin gitclaw to v1.2.0"
```

### Option B — Git Subtree

A git subtree merges the contents of an external repository directly into the host repository's file tree. Unlike submodules, the files are fully present — no special initialization is needed when cloning.

#### Adding gitclaw as a Subtree

```bash
cd /path/to/your-repo

# Add the subtree
git subtree add --prefix .GITCLAW https://github.com/japer-technology/gitclaw.git main --squash

# --prefix .GITCLAW    → Place files under .GITCLAW/
# --squash             → Squash external history into one commit
```

This creates a single commit that adds all of gitclaw's files under `.GITCLAW/`.

#### Running the Installer

Same as with submodules:

```bash
bun .GITCLAW/install/GITCLAW-INSTALLER.ts
git add .github/
git commit -m "feat: install gitclaw workflows"
git push
```

#### Cloning a Repository with a Subtree

No special handling is needed. A standard `git clone` includes all subtree files:

```bash
git clone https://github.com/user/my-project.git
# .GITCLAW/ is already there, fully materialized
```

#### Updating gitclaw via Subtree

```bash
# Pull the latest changes from the gitclaw repo
git subtree pull --prefix .GITCLAW https://github.com/japer-technology/gitclaw.git main --squash

# This creates a merge commit with the updated files
```

---

## Submodule vs. Subtree Comparison

| Feature | Submodule | Subtree |
|---------|-----------|---------|
| **Files in host repo** | Pointer only (SHA reference) | Fully materialized files |
| **Clone behavior** | Requires `--recurse-submodules` or `submodule init` | Standard `git clone` works |
| **Update command** | `git submodule update --remote` | `git subtree pull` |
| **Commit history** | Separate (external repo) | Merged into host repo |
| **`.gitmodules` file** | Yes | No |
| **GitHub Actions checkout** | Needs `submodules: true` | Works with default checkout |
| **Familiarity** | Moderate (submodules have a learning curve) | Lower (subtree is less common) |
| **Version pinning** | By commit SHA or tag | By commit at pull time |
| **Offline access** | Requires fetch from remote | Files always available locally |
| **Repo size impact** | Minimal (pointer only) | Increases by gitclaw's file size |

---

## GitHub Actions Configuration

### With Submodules

The `actions/checkout` step needs the `submodules` option:

```yaml
steps:
  - uses: actions/checkout@v4
    with:
      submodules: true          # Initialize and checkout submodules
      # or: submodules: recursive  # For nested submodules
```

Without this, the `.GITCLAW/` directory will be empty in the Actions runner, and workflows that reference gitclaw files will fail.

### With Subtrees

No special configuration is needed — standard checkout works:

```yaml
steps:
  - uses: actions/checkout@v4
  # .GITCLAW/ files are already in the repo
```

---

## Automated Updates

### Dependabot (Submodules)

GitHub's Dependabot supports submodule updates. Add to `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "gitsubmodule"
    directory: "/"
    schedule:
      interval: "weekly"
```

Dependabot will open PRs when the gitclaw repository has new commits.

### GitHub Actions (Both)

A scheduled workflow can check for and apply updates:

```yaml
name: Update GitClaw
on:
  schedule:
    - cron: '0 9 * * 1'  # Weekly on Monday

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: true

      - name: Update submodule
        run: |
          git submodule update --remote .GITCLAW
          if git diff --quiet; then
            echo "No updates available"
            exit 0
          fi

      - name: Create PR
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          git config user.name "gitclaw[bot]"
          git config user.email "gitclaw[bot]@users.noreply.github.com"
          BRANCH="gitclaw/update-$(date +%Y%m%d)"
          git checkout -b "$BRANCH"
          git add .GITCLAW
          git commit -m "chore: update gitclaw submodule"
          git push origin "$BRANCH"
          gh pr create \
            --title "🦞 Update GitClaw" \
            --body "Updates gitclaw to the latest version." \
            --base main \
            --head "$BRANCH"
```

---

## Detailed Workflow: Submodule Setup

```
Step 1: Add submodule
┌──────────────────────────────────────┐
│ git submodule add <url> .GITCLAW     │
│                                      │
│ Creates:                             │
│ ├── .gitmodules (submodule config)   │
│ └── .GITCLAW/ (pointer to commit)   │
└──────────────────────────────────────┘
                │
                ▼
Step 2: Run installer
┌──────────────────────────────────────┐
│ bun .GITCLAW/install/INSTALLER.ts    │
│                                      │
│ Creates:                             │
│ ├── .github/workflows/*.yml          │
│ └── .github/ISSUE_TEMPLATE/*.yml     │
└──────────────────────────────────────┘
                │
                ▼
Step 3: Commit and push
┌──────────────────────────────────────┐
│ git add .                            │
│ git commit -m "feat: add gitclaw"    │
│ git push                             │
└──────────────────────────────────────┘
                │
                ▼
Step 4: Configure Actions checkout
┌──────────────────────────────────────┐
│ In all workflows:                    │
│   - uses: actions/checkout@v4        │
│     with:                            │
│       submodules: true               │
└──────────────────────────────────────┘
                │
                ▼
Step 5: Add API key secret
┌──────────────────────────────────────┐
│ Settings → Secrets → Actions         │
│ → ANTHROPIC_API_KEY                  │
└──────────────────────────────────────┘
```

---

## Detailed Workflow: Subtree Setup

```
Step 1: Add subtree
┌──────────────────────────────────────┐
│ git subtree add --prefix .GITCLAW    │
│   <url> main --squash                │
│                                      │
│ Creates:                             │
│ └── .GITCLAW/ (full file copy)       │
└──────────────────────────────────────┘
                │
                ▼
Step 2: Run installer
┌──────────────────────────────────────┐
│ bun .GITCLAW/install/INSTALLER.ts    │
│                                      │
│ Creates:                             │
│ ├── .github/workflows/*.yml          │
│ └── .github/ISSUE_TEMPLATE/*.yml     │
└──────────────────────────────────────┘
                │
                ▼
Step 3: Commit and push
┌──────────────────────────────────────┐
│ git add .                            │
│ git commit -m "feat: add gitclaw"    │
│ git push                             │
└──────────────────────────────────────┘
                │
                ▼
Step 4: Add API key secret
┌──────────────────────────────────────┐
│ Settings → Secrets → Actions         │
│ → ANTHROPIC_API_KEY                  │
└──────────────────────────────────────┘
```

---

## Strengths

- **Trivial updates** — Updating gitclaw is a single command (`git submodule update --remote` or `git subtree pull`). No manual file copying or re-downloading.
- **Clear separation** — gitclaw code lives in its own directory with a clear boundary from the host project's code.
- **Version pinning** — Submodules pin to a specific commit SHA, ensuring reproducible builds. Subtrees pin at pull time.
- **Familiar git patterns** — Developers who have used submodules or subtrees will immediately understand the setup.
- **Dependabot support** — Submodules can be auto-updated by Dependabot, creating PRs when new versions are available.
- **No external tools** — Only requires git, which is already installed on every developer's machine.
- **Works for existing repos** — Can be added to any existing repository.

---

## Limitations

- **Submodule complexity** — Submodules add complexity to the `git clone` workflow. Contributors must know to use `--recurse-submodules`, and forgetting this leads to an empty `.GITCLAW/` directory and confusing errors.
- **GitHub Actions configuration** — With submodules, every workflow's checkout step must include `submodules: true`. Missing this in any workflow breaks gitclaw.
- **Subtree merge noise** — Subtree pulls create merge commits that can clutter the commit history, especially with frequent updates.
- **Subtree unfamiliarity** — Many developers have never used `git subtree`. The command syntax is verbose and unintuitive.
- **No configuration customization** — Unlike the CLI tool or GitHub Pages portal, the submodule/subtree approach doesn't offer interactive configuration. Users must edit files manually after adding.
- **Submodule network dependency** — Submodule content is fetched from the remote, so cloning requires network access to the gitclaw repository.
- **Shallow clone issues** — Some CI environments use shallow clones, which can interact poorly with submodules.

---

## Security Considerations

- **Submodule URL trust** — The `.gitmodules` file specifies the URL to clone the submodule from. If an attacker modifies this, they could redirect the submodule to a malicious repository. Review `.gitmodules` changes in PRs.
- **Commit SHA pinning** — Submodules pin to a specific commit. This provides security: even if the remote repository is compromised, the host repo won't pull malicious code until the submodule reference is updated.
- **Subtree code review** — Since subtree pulls merge code directly into the repo, all changes should be reviewed in the resulting PR or commit.
- **Branch protection** — Enable branch protection on the host repo to ensure submodule updates go through PR review.

---

## When to Use This Method

This method is ideal when:

- You want **easy updates** with a single git command.
- Your team is **familiar with submodules or subtrees** and comfortable with the workflow.
- You want **version pinning** for reproducible builds.
- You need a **git-native** solution with no external tools or services.
- You want **Dependabot integration** for automatic update PRs (submodules).

---

## When to Consider Alternatives

Consider a different delivery method when:

- Your **team or contributors are unfamiliar** with submodules and the added complexity would cause confusion.
- You want a **one-click or zero-CLI** experience (consider [GitHub App](./github-application.md) or [template repo](./github-template-repository.md)).
- Your CI/CD uses **shallow clones** that don't work well with submodules.
- You want **interactive configuration** during installation (consider [CLI tool](./cli-tool.md)).

---

## Related Methods

- [Fork / Import + Installer](./fork-import-installer.md) — The manual copy approach that submodules/subtrees replace.
- [CLI Tool (npx / bunx)](./cli-tool.md) — A CLI that could use subtree-add under the hood.
- [Package Registry](./package-registry.md) — Another versioned distribution approach, using npm instead of git.
- [GitHub Marketplace Action](./github-marketplace-action.md) — An alternative that handles updates via GitHub Actions instead of git commands.

---

> 🦞 *Submodule or subtree — let git handle the versioning, and updating gitclaw becomes just another pull.*
