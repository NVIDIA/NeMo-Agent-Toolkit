# .GITCLAW 🦞 Email-Based Delivery

### Deliver gitclaw via email — attachments, magic links, or repository invitations — reaching developers through the most universal communication channel.

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

---

## Overview

Email-Based Delivery uses email as the channel through which gitclaw reaches a user's repository. This method has three distinct variants: sending a ZIP attachment containing the `.GITCLAW/` folder, sending a magic link that triggers an automated installation, or sending a GitHub repository invitation that gives the user access to setup automation.

While email is not the most developer-native channel, it is the most universal one. Every GitHub user has an email address, and email can reach people who aren't actively browsing the GitHub Marketplace or searching for tools.

---

## How It Works

### Option A — Attachment Delivery

The simplest variant: package the `.GITCLAW/` folder as a ZIP file and send it directly to the user.

#### Flow

1. **User requests gitclaw** via a sign-up form, mailing list subscription, or direct request.
2. **An automated system** (or manual process) sends an email with:
   - A ZIP attachment containing the pre-configured `.GITCLAW/` folder.
   - A README with step-by-step installation instructions.
   - Links to documentation.
3. **User downloads the attachment**, extracts it into their repository, and follows the included instructions.

#### Email Content

```
Subject: 🦞 Your gitclaw installation package

Hi [name],

Thanks for your interest in gitclaw! Attached is your installation
package.

Getting started:
1. Extract gitclaw.zip into your repository root
2. Run: bun .GITCLAW/install/GITCLAW-INSTALLER.ts
3. Commit and push the changes
4. Add your API key as a repository secret
5. Open an issue to talk to your agent!

Full documentation: https://github.com/japer-technology/gitclaw

— The gitclaw team 🦞
```

#### Configuration Customization

The email system can generate customized ZIP files based on information provided during sign-up:

| User provides | ZIP is customized with |
|--------------|----------------------|
| LLM provider | Pre-configured provider setting |
| Preferred model | Pre-configured model selection |
| Repository URL | Pre-set remote references |
| Organization name | Customized agent personality |

### Option B — Magic Link Delivery

A more automated approach: send the user a unique link that, when clicked, triggers an API-driven installation on their repository.

#### Flow

1. **User provides their GitHub username and target repository** via a form on the gitclaw website, a GitHub issue, or a direct message.
2. **The system generates a unique, time-limited magic link** and sends it to the user's email.
3. **User clicks the link**, which:
   - Redirects to a confirmation page.
   - Authenticates the user via GitHub OAuth (if not already authenticated).
   - Triggers the gitclaw installation API to create a branch, commit `.GITCLAW/`, and open a PR on their repository.
4. **User is redirected to the PR** on GitHub, ready to review and merge.

#### Magic Link Structure

```
https://install.gitclaw.dev/magic?token=abc123def456&repo=user/my-project&expires=1709500800
```

| Parameter | Purpose |
|-----------|---------|
| `token` | Unique, cryptographically secure token |
| `repo` | Target repository (pre-filled from user input) |
| `expires` | Expiration timestamp (e.g., 24 hours) |

#### Security Model

- **Tokens are single-use** — Once clicked, the token is invalidated.
- **Tokens expire** — Typically within 24-48 hours.
- **OAuth confirmation** — Even with a valid token, the user must authenticate via GitHub OAuth to confirm their identity.
- **No stored credentials** — The magic link system doesn't store long-lived tokens. OAuth happens at click time.

### Option C — Invite-Based Delivery

Leverage GitHub's repository invitation system to share access to a setup repository.

#### Flow

1. **User provides their GitHub username** via any channel (form, email, chat).
2. **The system invites the user as a collaborator** to a private setup repository (e.g., `japer-technology/gitclaw-setup`).
3. **User accepts the invitation** on GitHub.
4. **The setup repository contains:**
   - An automated workflow that, when triggered, pushes `.GITCLAW/` to the user's target repo.
   - Configuration forms via GitHub Issues.
   - Documentation.
5. **User opens an issue** in the setup repo specifying their target repository.
6. **A workflow runs** that:
   - Forks or creates a branch in the target repo.
   - Commits `.GITCLAW/`.
   - Opens a bootstrap PR.
7. **User is notified** via the issue thread with a link to the PR.

#### Setup Repository Structure

```
gitclaw-setup/ (private)
├── .github/
│   ├── workflows/
│   │   └── install-gitclaw.yml    # Triggered by issue creation
│   └── ISSUE_TEMPLATE/
│       └── install-request.yml     # Form for specifying target repo
├── README.md                       # Instructions for new collaborators
└── docs/
    └── getting-started.md
```

---

## Email Delivery System Architecture

### For Attachment and Magic Link Delivery

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Sign-up     │     │  Email       │     │  User        │
│  Form        │────▶│  Service     │────▶│  Inbox       │
│  (Website/   │     │  (SendGrid/  │     │              │
│   GitHub     │     │   SES/       │     │  Opens email │
│   Issue)     │     │   Resend)    │     │  Clicks link │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                     ┌──────────────┐              │
                     │  GitHub API  │◀─────────────┘
                     │  (Create     │  (Magic link triggers
                     │   branch,    │   installation)
                     │   commit,    │
                     │   open PR)   │
                     └──────────────┘
```

### Email Service Options

| Service | Pros | Cons |
|---------|------|------|
| **Resend** | Modern API, great DX, generous free tier | Newer service |
| **SendGrid** | Mature, feature-rich, good deliverability | Complex pricing |
| **AWS SES** | Very cheap at scale, reliable | More setup required |
| **Postmark** | Excellent deliverability, focused on transactional email | Smaller free tier |
| **GitHub Notifications** | No external service needed | Limited formatting, not customizable |

---

## Email Templates

### Welcome Email (Attachment)

```html
<div style="font-family: system-ui; max-width: 600px; margin: 0 auto;">
  <h1>🦞 Welcome to gitclaw</h1>
  <p>Your AI-powered GitHub assistant is ready to install.</p>

  <h2>Quick Start</h2>
  <ol>
    <li>Download and extract the attached ZIP file into your repo root</li>
    <li>Run the installer: <code>bun .GITCLAW/install/GITCLAW-INSTALLER.ts</code></li>
    <li>Commit and push your changes</li>
    <li>Add your API key as a repository secret</li>
    <li>Open an issue — gitclaw will respond!</li>
  </ol>

  <a href="https://github.com/japer-technology/gitclaw/docs">
    Full Documentation →
  </a>
</div>
```

### Magic Link Email

```html
<div style="font-family: system-ui; max-width: 600px; margin: 0 auto;">
  <h1>🦞 Install gitclaw on {{repo}}</h1>
  <p>Click the button below to install gitclaw. A pull request will be
     created on your repository with everything you need.</p>

  <a href="{{magic_link}}"
     style="background: #e74c3c; color: white; padding: 12px 24px;
            text-decoration: none; border-radius: 6px;">
    Install gitclaw →
  </a>

  <p style="color: #666; font-size: 12px;">
    This link expires in 24 hours. If you didn't request this,
    you can safely ignore this email.
  </p>
</div>
```

---

## Strengths

- **Universal reach** — Every developer has an email address. Email reaches people who aren't actively browsing marketplaces or package registries.
- **Personalized delivery** — Emails can include customized configurations, personalized instructions, and targeted messaging.
- **Magic links feel effortless** — Click a link, confirm, and a PR appears on your repo. No terminal, no configuration.
- **Invite-based delivery leverages GitHub** — Uses GitHub's own collaboration model, so the experience is familiar.
- **Asynchronous** — The user can act on the email whenever convenient, unlike real-time installation flows.
- **Good for onboarding campaigns** — Email sequences can guide users through setup, configuration, and first use over multiple days.
- **Works for non-technical users** — The magic link and invite approaches require minimal technical knowledge.

---

## Limitations

- **Attachment delivery is clunky** — ZIP attachments are often caught by spam filters, email file size limits may apply, and the user still needs to extract and install manually.
- **Magic links require a backend** — The system needs a server to generate tokens, send emails, and process installations.
- **Email is a slower channel** — Unlike instant CLI or one-click installations, email introduces delay (delivery time, user checking inbox).
- **Spam risk** — Automated emails can end up in spam folders, especially if the sending domain is new or has low reputation.
- **Less developer-native** — Most developers expect to install tools via CLI or GitHub, not email. This channel may feel unusual.
- **Token security** — Magic link tokens must be carefully managed (single-use, time-limited, cryptographically secure) to prevent abuse.
- **Invite management** — The invite-based approach requires managing collaborator lists and access to the setup repository.
- **No automatic updates** — Email delivery is a one-time action. Updates require re-sending or a separate mechanism.

---

## Security Considerations

- **Magic link token security** — Tokens must be cryptographically random, single-use, and time-limited (e.g., 24-hour expiry).
- **Email authentication** — Configure SPF, DKIM, and DMARC records for the sending domain to prevent spoofing.
- **No credentials in email** — Never include API keys, passwords, or tokens in email content. Only include links and instructions.
- **Rate limiting** — Limit the number of magic links that can be generated per user per hour to prevent abuse.
- **Unsubscribe compliance** — Include unsubscribe links in all marketing/automated emails per CAN-SPAM and GDPR requirements.
- **Attachment scanning** — If sending ZIP attachments, ensure they pass common email security scanners.

---

## When to Use This Method

This method is ideal when:

- You are running an **onboarding campaign** or **outreach effort** to introduce gitclaw to new users.
- Your target audience includes people who are **not actively browsing GitHub Marketplace** or developer tools.
- You want to provide a **personalized, guided experience** delivered asynchronously.
- You need to reach users who prefer **non-technical channels** and may not be comfortable with CLI tools.
- You are using **invite-based access control** for a private beta or limited release.

---

## When to Consider Alternatives

Consider a different delivery method when:

- Your users are **active GitHub users** who would find email delivery slow or unusual (use [GitHub App](./github-application.md) or [Marketplace Action](./github-marketplace-action.md)).
- You want **instant, self-service installation** without waiting for emails (use [CLI tool](./cli-tool.md) or [template repo](./github-template-repository.md)).
- You want to **avoid managing an email service** and token system (use [fork/import installer](./fork-import-installer.md)).

---

## Related Methods

- [GitHub Pages Portal](./github-pages-self-service-portal.md) — The sign-up form that captures user information for email delivery.
- [GitHub Repository Dispatch](./github-repository-dispatch.md) — The API-driven installation that magic links trigger behind the scenes.
- [GitHub Application](./github-application.md) — A more direct, real-time alternative to email-based installation.
- [Third-Party Website](./third-party-website.md) — Could incorporate email delivery as one of its channels.

---

> 🦞 *Email is the original delivery channel. A magic link in the inbox, and gitclaw is a click away.*
