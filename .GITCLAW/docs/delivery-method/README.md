# .GITCLAW 🦞 Delivery Methods

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

This folder documents every viable delivery method for getting the `.GITCLAW/` folder into a repository. Each document is a deep-dive into a specific approach — how it works, its architecture, strengths, limitations, and when to use it.

## Why Delivery Methods Matter

gitclaw is a self-contained `.GITCLAW/` folder that turns any GitHub repo into an AI-powered assistant. The delivery question is: **how does that folder get into a repo in the first place?**

These documents map out every channel — from what exists today to ideas that could reshape how gitclaw is distributed.

## Documentation Map

| Document | Description |
|----------|-------------|
| [fork-import-installer](fork-import-installer/) | The current method — manual fork or import plus the installer script |
| [github-application](github-application/) | A registered GitHub App that installs gitclaw with a single click |
| [github-marketplace-action](github-marketplace-action/) | A published GitHub Action that bootstraps gitclaw from any workflow |
| [github-template-repository](github-template-repository/) | Mark the gitclaw repo as a template for instant new-repo creation |
| [cli-tool](cli-tool/) | A single `npx gitclaw init` command to install locally |
| [github-pages-self-service-portal](github-pages-self-service-portal/) | A GitHub Pages site that generates and delivers configurations |
| [third-party-website](third-party-website/) | A dedicated website serving as a full-featured installation hub |
| [email-based-delivery](email-based-delivery/) | Email-based delivery via attachments, magic links, or invitations |
| [github-repository-dispatch](github-repository-dispatch/) | API-driven installation via GitHub repository dispatch events |
| [browser-extension](browser-extension/) | A browser extension that adds an install button to GitHub repo pages |
| [git-submodule-subtree](git-submodule-subtree/) | Include gitclaw as a git submodule or subtree |
| [package-registry](package-registry/) | Publish `.GITCLAW/` as a versioned npm or GitHub Package |
| [github-codespaces-dev-container](github-codespaces-dev-container/) | Pre-configure gitclaw in a dev container definition |
| [probot-webhook-service](probot-webhook-service/) | A hosted webhook service that responds to GitHub events |

## See Also

- [GITCLAW-Delivery-Methods.md](../GITCLAW-Delivery-Methods.md) — Overview and comparison matrix of all delivery methods in a single document.

---

> 🦞 *gitclaw is the folder. The delivery method is just the door.*
