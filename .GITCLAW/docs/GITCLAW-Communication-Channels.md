# .GITCLAW 🦞 Communication Channels

### Every mechanism by which GitHub can send an email notification — and how GitClaw can exploit each one

<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/japer-technology/gitclaw/main/.GITCLAW/GITCLAW-LOGO.png" alt="GitClaw" width="500">
  </picture>
</p>

> GitHub is, among many things, an email delivery system. Nearly every meaningful event on the
> platform can trigger a notification email to one or more users. These emails are not a side
> effect — they are a **communication channel** that GitClaw can leverage to reach humans
> outside of the GitHub web interface.
>
> This document is an exhaustive catalogue of every built-in mechanism by which GitHub
> sends email notifications. For each mechanism, it describes **exactly what triggers it**,
> **exactly who receives it**, **exactly what the email contains**, and **how GitClaw can
> use it** as an outbound communication channel.

---

## Table of Contents

1. [How GitHub Email Notifications Work](#1-how-github-email-notifications-work)
2. [Issue Events](#2-issue-events)
3. [Issue Comment Events](#3-issue-comment-events)
4. [Pull Request Events](#4-pull-request-events)
5. [Pull Request Review Events](#5-pull-request-review-events)
6. [Pull Request Review Comment Events](#6-pull-request-review-comment-events)
7. [@Mentions](#7-mentions)
8. [Team Mentions](#8-team-mentions)
9. [Commit Comment Events](#9-commit-comment-events)
10. [Release Events](#10-release-events)
11. [Discussion Events](#11-discussion-events)
12. [Discussion Comment Events](#12-discussion-comment-events)
13. [Security Alert Events](#13-security-alert-events)
14. [GitHub Actions Workflow Events](#14-github-actions-workflow-events)
15. [Deployment Review Events](#15-deployment-review-events)
16. [Collaborator Invitation Events](#16-collaborator-invitation-events)
17. [Organization & Team Membership Events](#17-organization--team-membership-events)
18. [Team Discussion Events](#18-team-discussion-events)
19. [GitHub Pages Build Failure Events](#19-github-pages-build-failure-events)
20. [Sponsorship Events](#20-sponsorship-events)
21. [Repository Transfer Events](#21-repository-transfer-events)
22. [Assignment Events](#22-assignment-events)
23. [Label Subscription Events](#23-label-subscription-events)
24. [Scheduled Reminder Events](#24-scheduled-reminder-events)
25. [The Reply-To Channel](#25-the-reply-to-channel)
26. [Email Anatomy](#26-email-anatomy)
27. [Recipient Determination Logic](#27-recipient-determination-logic)
28. [Notification Preference Controls](#28-notification-preference-controls)
29. [Constraints, Limits & Risks](#29-constraints-limits--risks)
30. [GitClaw Communication Patterns](#30-gitclaw-communication-patterns)
31. [Trigger-by-Trigger API Reference](#31-trigger-by-trigger-api-reference)
32. [Conclusion](#32-conclusion)

---

## 1. How GitHub Email Notifications Work

Before cataloguing every trigger, it is essential to understand the underlying system.

### The Notification Pipeline

When an event occurs on GitHub (an issue is opened, a comment is posted, a review is submitted), the platform runs a notification pipeline:

1. **Event fires.** Something happens — a REST API call, a git push, a UI action.
2. **Recipient list is computed.** GitHub determines who should be notified based on subscription state, mention parsing, assignment, team membership, and repository watching preferences.
3. **Per-recipient filtering.** Each potential recipient's notification settings are checked. Users can opt out of specific event types, mute specific threads, or disable email notifications entirely.
4. **Email is constructed.** GitHub generates an HTML email with the event content, rendered Markdown, relevant context links, and threading headers.
5. **Email is delivered.** The email is sent from `notifications@github.com` to the recipient's configured notification email address.

### Key Principle: Notifications Are Opt-In by Default

A user only receives email notifications if:

- They are **watching** the repository (and their watch level includes the event type), OR
- They are **subscribed** to the specific thread (by participating, being assigned, being mentioned, or manually subscribing), OR
- They are **directly targeted** by the event (assigned, review-requested, @mentioned)

This means GitClaw cannot email arbitrary external addresses. It can only reach **GitHub users who have opted into receiving notifications** from the repository or thread in question. This is a fundamental constraint — and also a feature, because it means the channel is consent-based.

### What "Watching" Means

Users can watch a repository at several levels:

| Watch Level | What Triggers Email |
|-------------|-------------------|
| **Participating and @mentions** (default) | Only events on threads the user has commented on, been assigned to, been @mentioned in, or manually subscribed to |
| **All Activity** | Every event on the repository — issues, PRs, comments, pushes, releases, discussions |
| **Releases Only** | Only published release events |
| **Ignoring** | Nothing — the repository is muted entirely |
| **Custom** | User selects which event types they want: Issues, Pull Requests, Releases, Discussions, Security Alerts |

### What "Subscribed to a Thread" Means

A user becomes subscribed to a specific issue, PR, or discussion thread when:

- They **created** the thread (authored the issue, opened the PR)
- They **commented** on the thread
- They were **assigned** to the thread
- They were **@mentioned** in the thread or a comment on it
- They were **requested as a reviewer** (for PRs)
- They **manually clicked "Subscribe"** on the thread
- They are a member of a **team that was @mentioned**

Once subscribed, a user receives email for every subsequent event on that thread until they unsubscribe or mute it.

---

## 2. Issue Events

### 2a. Issue Created

**What triggers it:**
A new issue is opened on the repository. This can happen via the GitHub web UI, the GitHub CLI (`gh issue create`), or the REST API (`POST /repos/{owner}/{repo}/issues`).

**Who receives the email:**
- Every user watching the repository with "All Activity" or "Issues" custom watch level.
- Every user @mentioned in the issue body (see [§7 @Mentions](#7-mentions)).
- Every user assigned to the issue at creation time (see [§22 Assignment Events](#22-assignment-events)).

**What the email contains:**
- **Subject:** `[owner/repo] Issue title (#number)`
- **Body:** The full issue body, rendered from Markdown to HTML. Includes images, code blocks, tables, and task lists as they would appear on GitHub.
- **Links:** Direct link to the issue on GitHub.
- **Metadata:** Repository name, issue author, labels (if applied at creation), assignees.

**How GitClaw can use it:**
The agent can programmatically create issues via the GitHub API to send notifications to all repository watchers. The issue title becomes the email subject line. The issue body becomes the email body. This is the most natural outbound notification mechanism — the issue serves double duty as both a tracked work item and a notification.

```
API: POST /repos/{owner}/{repo}/issues
Body: { "title": "Alert: deployment failed", "body": "Details here..." }
Result: All watchers receive an email with the alert content.
```

### 2b. Issue Closed

**What triggers it:**
An issue is closed. This can happen via the web UI, CLI, API (`PATCH /repos/{owner}/{repo}/issues/{number}` with `"state": "closed"`), or by a commit/PR that references it with closing keywords (e.g., "Fixes #42").

**Who receives the email:**
- Every user subscribed to that specific issue thread.

**What the email contains:**
- **Subject:** `Re: [owner/repo] Issue title (#number)` (threaded with the original)
- **Body:** A notice that the issue was closed, including who closed it and whether it was closed by a commit or PR (with a link).

**How GitClaw can use it:**
Closing an issue re-notifies all thread subscribers. This can be used as a "resolution" notification — the agent closes an issue with a summary comment, and everyone who participated receives the closure email.

### 2c. Issue Reopened

**What triggers it:**
A previously closed issue is reopened via the web UI, CLI, or API (`PATCH` with `"state": "open"`).

**Who receives the email:**
- Every user subscribed to that specific issue thread.

**What the email contains:**
- A notice that the issue was reopened, including who reopened it.

**How GitClaw can use it:**
Reopening an issue is a way to re-alert all thread subscribers. The agent can use a close/reopen cycle on a dedicated "notification" issue to repeatedly notify the same group of subscribers. Each cycle generates a new email while keeping the conversation in a single thread.

### 2d. Issue Transferred

**What triggers it:**
An issue is transferred from one repository to another via the web UI or API.

**Who receives the email:**
- All subscribers of the original issue thread. They remain subscribed in the destination repository.

**What the email contains:**
- A notice that the issue was transferred, with a link to its new location.

**How GitClaw can use it:**
Limited utility for communication, but the agent could use issue transfers to route notifications across repositories in an organization.

### 2e. Issue Milestone Changed

**What triggers it:**
A milestone is added to, removed from, or changed on an issue via the web UI or API (`PATCH` with `"milestone"` field).

**Who receives the email:**
- All subscribers of that issue thread.

**What the email contains:**
- A notice of the milestone change.

**How GitClaw can use it:**
Minimal standalone communication value, but milestone changes on subscribed issues generate notification emails — useful if the agent is coordinating release planning and wants subscribers to know an issue's timeline has changed.

---

## 3. Issue Comment Events

### 3a. Comment Added to an Issue

**What triggers it:**
A new comment is posted on an existing issue. This can happen via the web UI, CLI (`gh issue comment`), or API (`POST /repos/{owner}/{repo}/issues/{issue_number}/comments`).

**Who receives the email:**
- Every user subscribed to that issue thread. This includes:
  - The issue author
  - Anyone who has previously commented on the issue
  - Anyone assigned to the issue
  - Anyone @mentioned in the issue or its comments
  - Anyone who manually subscribed
  - Anyone watching the repo with "All Activity" or "Issues" enabled
- Additionally, any user @mentioned in the new comment who was not previously subscribed — they receive the email AND become subscribed.

**What the email contains:**
- **Subject:** `Re: [owner/repo] Issue title (#number)` (threaded with the original issue email)
- **Body:** The full comment body, rendered from Markdown to HTML.
- **Context:** The comment author's username and avatar.
- **Reply-To:** A unique reply address (see [§25 The Reply-To Channel](#25-the-reply-to-channel)).

**What makes this the most important trigger:**
This is GitClaw's **primary outbound communication mechanism today**. When the agent posts a comment on an issue, every thread subscriber receives an email containing the agent's full response. The Markdown renders beautifully in email — code blocks, tables, images, and links all work. The email is threaded with the original issue, so the entire conversation reads naturally in any email client.

**How GitClaw can use it:**
This is the mechanism GitClaw already uses. Every time the agent responds to a user via an issue comment, that response is delivered as an email to all thread subscribers. The agent's comment IS the email body. This is the most reliable, most flexible, and most content-rich notification channel available.

```
API: POST /repos/{owner}/{repo}/issues/{issue_number}/comments
Body: { "body": "## Analysis Complete\n\nHere are the findings..." }
Result: All issue subscribers receive an email with the rendered Markdown.
```

### 3b. Comment Edited

**What triggers it:**
An existing comment is edited via the web UI or API (`PATCH /repos/{owner}/{repo}/issues/comments/{comment_id}`).

**Who receives the email:**
- **Nobody.** GitHub does not send email notifications for comment edits.

**GitClaw implication:**
Editing a comment is silent. The agent can update previously posted comments without generating additional notifications. This is useful for correcting errors or updating status without spamming subscribers.

### 3c. Comment Deleted

**What triggers it:**
A comment is deleted via the web UI or API (`DELETE /repos/{owner}/{repo}/issues/comments/{comment_id}`).

**Who receives the email:**
- **Nobody.** GitHub does not send email notifications for comment deletions.

**GitClaw implication:**
The agent can clean up temporary or superseded comments without notification noise.

---

## 4. Pull Request Events

### 4a. Pull Request Opened

**What triggers it:**
A new pull request is opened via the web UI, CLI (`gh pr create`), or API (`POST /repos/{owner}/{repo}/pulls`).

**Who receives the email:**
- Every user watching the repository with "All Activity" or "Pull Requests" custom watch level.
- Every user @mentioned in the PR body.
- Every user assigned to the PR at creation time.
- Every user requested as a reviewer at creation time.
- CODEOWNERS who match files touched by the PR (if auto-review is configured).

**What the email contains:**
- **Subject:** `[owner/repo] PR title (#number)`
- **Body:** The full PR description, rendered from Markdown to HTML. Includes a link to the diff.
- **Metadata:** Branch names (head → base), author, reviewers requested, labels.

**How GitClaw can use it:**
The agent can create pull requests to notify watchers and reviewers. The PR description becomes the email body. This is particularly useful when the agent has made code changes and wants human review — the PR creation email serves as both a review request and a notification.

### 4b. Pull Request Closed / Merged

**What triggers it:**
A PR is closed (with or without merging) via the web UI, CLI, or API (`PATCH` with `"state": "closed"` or `PUT /repos/{owner}/{repo}/pulls/{number}/merge`).

**Who receives the email:**
- Every user subscribed to that PR thread.

**What the email contains:**
- Whether the PR was merged or closed without merging.
- Who performed the action.
- The merge commit SHA (if merged).

**How GitClaw can use it:**
Merging a PR notifies all thread subscribers. The agent can use this as a "completion" signal — after code changes are merged, all participants are emailed.

### 4c. Commits Pushed to PR Branch

**What triggers it:**
New commits are pushed to the head branch of an open pull request. This happens via `git push`.

**Who receives the email:**
- Every user subscribed to that PR thread.

**What the email contains:**
- A list of new commits with their messages.
- A link to the updated diff.

**How GitClaw can use it:**
The agent can push commits to a PR branch, and every PR subscriber receives a notification with the commit messages. The commit messages themselves become notification content.

---

## 5. Pull Request Review Events

### 5a. Review Submitted

**What triggers it:**
A user (or bot) submits a pull request review with a verdict. This can happen via the web UI or API (`POST /repos/{owner}/{repo}/pulls/{number}/reviews` with `"event"` set to `"APPROVE"`, `"REQUEST_CHANGES"`, or `"COMMENT"`).

**Who receives the email:**
- The PR author (always).
- Every user subscribed to the PR thread.

**What the email contains:**
- **Subject:** `Re: [owner/repo] PR title (#number)`
- **Body:** The review body text (if any), rendered from Markdown. The review verdict (approved, changes requested, or commented) is indicated.
- **Inline comments:** If the review includes file-level comments, they are included in the email with file paths and line numbers.

**How GitClaw can use it:**
This is the most powerful PR notification mechanism. The agent submits a formal review with a body and inline comments. The PR author and all subscribers receive a richly formatted email containing:
- The overall review verdict and summary
- Every inline comment with its file context

This is how the agent delivers structured code review feedback via email.

```
API: POST /repos/{owner}/{repo}/pulls/{number}/reviews
Body: {
  "body": "## Review Summary\n\nTwo issues found.",
  "event": "REQUEST_CHANGES",
  "comments": [
    { "path": "src/main.ts", "position": 15, "body": "This introduces a race condition." }
  ]
}
Result: PR author receives an email with the full review, including inline annotations.
```

### 5b. Review Requested

**What triggers it:**
A user or team is requested as a reviewer on a pull request. This can happen via the web UI or API (`POST /repos/{owner}/{repo}/pulls/{number}/requested_reviewers`).

**Who receives the email:**
- The specific user requested as a reviewer.
- Every member of a team requested as a reviewer (each member individually).

**What the email contains:**
- A notification that a review has been requested.
- A link to the PR.
- The PR title and description summary.

**How GitClaw can use it:**
The agent can request specific individuals or teams as reviewers. Each requested person receives a direct email. This is a targeted notification mechanism — the agent can "page" specific people by requesting their review on a PR.

### 5c. Review Request Removed

**What triggers it:**
A review request is removed from a user or team.

**Who receives the email:**
- The user whose review request was removed.

**GitClaw implication:**
Removing a review request generates a notification. The agent could theoretically use add/remove review request cycles for repeated notifications, though this would be an abuse of the mechanism.

---

## 6. Pull Request Review Comment Events

### 6a. Inline Review Comment Posted

**What triggers it:**
A comment is posted on a specific line of a pull request diff, outside of a formal review submission. This can happen via the web UI or API (`POST /repos/{owner}/{repo}/pulls/{number}/comments`).

**Who receives the email:**
- Every user subscribed to the PR thread.

**What the email contains:**
- The comment text, rendered from Markdown.
- The file path and line context where the comment was placed.
- A code snippet showing the surrounding diff context.

**How GitClaw can use it:**
The agent can post targeted inline comments on specific lines of a PR diff. Each comment generates an email to all PR subscribers with the precise file and line context. This enables granular, location-specific feedback delivered via email.

### 6b. Review Thread Replied To

**What triggers it:**
A reply is posted to an existing inline review thread.

**Who receives the email:**
- Every user subscribed to the PR thread.
- Every user who has participated in that specific review thread.

**What the email contains:**
- The reply text with the thread context.

**How GitClaw can use it:**
The agent can participate in review thread discussions. Replies to existing threads generate emails that are properly threaded in email clients, maintaining conversational context.

---

## 7. @Mentions

### 7a. @User Mention

**What triggers it:**
The text `@username` appears in any of the following:
- Issue body
- Issue comment
- PR body
- PR comment
- PR review body
- PR review comment (inline)
- Discussion body
- Discussion comment
- Commit comment
- Commit message (when pushed)

GitHub parses the rendered Markdown for `@username` patterns and treats each as a mention.

**Who receives the email:**
- The mentioned user — **regardless of whether they are watching the repository or subscribed to the thread**. An @mention overrides all watch/subscription settings (unless the user has explicitly muted the thread or disabled all email notifications globally).

**What the email contains:**
- The same email that would be generated by the parent event (comment, issue, review, etc.), but the `X-GitHub-Reason` header is set to `mention`.

**What makes this significant:**
@mentions are the **most aggressive notification mechanism** available. They can reach any GitHub user, even if they are not watching the repository and have never interacted with it. The only requirement is that the mentioned user has email notifications enabled in their global GitHub settings.

**How GitClaw can use it:**
The agent can @mention specific users in its comments to ensure those users receive email notifications. This is the most reliable way to reach a specific person.

```
API: POST /repos/{owner}/{repo}/issues/{number}/comments
Body: { "body": "@alice @bob Urgent: the production database migration failed. Details below..." }
Result: alice and bob both receive email notifications, even if they don't watch the repo.
```

**Constraints:**
- The mentioned user must have a GitHub account.
- The mentioned user must have email notifications enabled in their GitHub settings.
- The agent must know the user's GitHub username.
- The mentioned user can mute the thread after receiving the first notification, blocking subsequent ones.

### 7b. @Mention in Commit Message

**What triggers it:**
A commit message contains `@username` and the commit is pushed to the repository.

**Who receives the email:**
- The mentioned user (if their notification settings allow it).

**What the email contains:**
- A notification about the commit, including the commit message, SHA, and repository.

**How GitClaw can use it:**
The agent can include @mentions in commit messages when pushing code changes. This is a subtle notification channel — the commit itself becomes the notification vehicle.

---

## 8. Team Mentions

### 8a. @Org/Team Mention

**What triggers it:**
The text `@organization/team-name` appears in any of the same surfaces as user @mentions (issue body, comment, PR body, PR comment, review, discussion, commit comment).

**Who receives the email:**
- **Every member of the mentioned team** — individually. Each team member receives their own email notification. This is equivalent to @mentioning every member of the team individually.

**Requirements:**
- The person writing the @mention must have **read access** to the team (i.e., they must be a member of the organization and able to see the team).
- If the agent is using `GITHUB_TOKEN` from GitHub Actions, it operates as `github-actions[bot]`, which may or may not have visibility into organization teams depending on the token's scope and the organization's settings.

**What the email contains:**
- The same email as the parent event, with `X-GitHub-Reason` set to `team_mention`.

**How GitClaw can use it:**
Team mentions are a **broadcast mechanism**. The agent can notify an entire team with a single @mention. This is the closest thing to a mailing list that GitHub provides natively.

```
API: POST /repos/{owner}/{repo}/issues/{number}/comments
Body: { "body": "@myorg/platform-team The nightly build has been failing for 3 consecutive days. Summary..." }
Result: Every member of the platform-team receives an email notification.
```

**Constraints:**
- Only works within organizations (not for personal repos).
- The agent must have visibility into the team.
- Team members can individually mute the thread.
- Organization admins can restrict who can @mention teams.

---

## 9. Commit Comment Events

### 9a. Comment on a Commit

**What triggers it:**
A comment is posted on a specific commit. This can happen via the web UI (clicking on a commit and adding a comment) or the API (`POST /repos/{owner}/{repo}/commits/{sha}/comments`).

**Who receives the email:**
- The commit author.
- Anyone who has previously commented on that commit.
- Anyone @mentioned in the comment.
- Anyone subscribed to the commit thread.

**What the email contains:**
- **Subject:** `Re: [owner/repo] Commit message (SHA)`
- **Body:** The comment text, rendered from Markdown. If the comment is on a specific line, the file path and line context are included.
- **Context:** The commit SHA, author, and message.

**How GitClaw can use it:**
The agent can comment on specific commits to reach the commit author. This is useful for post-merge feedback, security notifications about specific changes, or compliance flagging. The commit author receives the email even if they are not watching the repository.

```
API: POST /repos/{owner}/{repo}/commits/{sha}/comments
Body: { "body": "This commit introduces a dependency with a known CVE. See details..." }
Result: The commit author receives an email about their specific commit.
```

### 9b. Line-Level Commit Comment

**What triggers it:**
A comment is posted on a specific line of a specific file in a commit. Triggered via API with additional `path` and `position` parameters.

**Who receives the email:**
- Same recipients as a general commit comment.

**What the email contains:**
- Same as above, plus the file path and line context.

**How GitClaw can use it:**
The agent can leave precise, line-level feedback on any commit in the repository's history. This is useful for retroactive code review or security annotation.

---

## 10. Release Events

### 10a. Release Published

**What triggers it:**
A release is published on the repository. This can happen via the web UI, CLI (`gh release create`), or API (`POST /repos/{owner}/{repo}/releases` with `"draft": false`). Also triggered when a draft release is edited to no longer be a draft.

**Who receives the email:**
- Every user watching the repository with **any** watch level that includes releases:
  - "All Activity" watchers
  - "Releases Only" watchers
  - "Custom" watchers who have selected the "Releases" checkbox

**What the email contains:**
- **Subject:** `[owner/repo] Release title`
- **Body:** The full release notes, rendered from Markdown to HTML. Includes the tag name, release author, and whether it's a pre-release.
- **Assets:** Links to any attached binary assets.
- **Changelog:** If auto-generated release notes are used, the full categorized changelog appears.

**What makes this significant:**
Release notifications reach the **broadest audience** of any notification type. Users who watch with "Releases Only" receive NO other notifications — only releases. This means release notifications reach users who have deliberately opted out of all other repo noise but still want to hear about releases.

**How GitClaw can use it:**
Publishing a release is a **broadcast to the entire watcher community**. The release notes are the email body. The agent can use this as a newsletter-style communication channel:

```
API: POST /repos/{owner}/{repo}/releases
Body: {
  "tag_name": "v1.0.42",
  "name": "Weekly Status Update",
  "body": "## What happened this week\n\n- Feature X shipped\n- Bug Y fixed\n- 3 security alerts resolved",
  "draft": false,
  "prerelease": false
}
Result: Every watcher (including "Releases Only" watchers) receives an email.
```

**Constraints:**
- Each release must have a unique tag. Tags are permanent git objects.
- Excessive release publishing may cause users to unwatch.
- Release emails cannot be targeted — they go to ALL qualifying watchers.

### 10b. Pre-Release Published

**What triggers it:**
Same as release published, but with `"prerelease": true`.

**Who receives the email:**
- Same as release published. Pre-releases trigger the same notification pipeline.

**How GitClaw can use it:**
Pre-releases can serve as a "softer" broadcast — communicating that something is available for testing without the weight of a full release.

### 10c. Release Edited

**What triggers it:**
An existing published release is edited (notes changed, assets updated).

**Who receives the email:**
- **Nobody.** Editing a release does NOT re-trigger notification emails. The email is sent once, at publication time.

**GitClaw implication:**
The agent cannot re-notify by editing a release. Each notification requires a new release publication.

---

## 11. Discussion Events

_Requires GitHub Discussions to be enabled on the repository._

### 11a. Discussion Created

**What triggers it:**
A new discussion is created in any category. This can happen via the web UI or GraphQL API (`createDiscussion` mutation). There is no REST API for creating discussions — **GraphQL is required**.

**Who receives the email:**
- Every user watching the repository with "All Activity" or "Discussions" custom watch level.
- Every user @mentioned in the discussion body.

**What the email contains:**
- **Subject:** `[owner/repo] Discussion title (#number)`
- **Body:** The full discussion body, rendered from Markdown to HTML.
- **Category:** The discussion category (General, Ideas, Q&A, etc.).

**How GitClaw can use it:**
Discussions are a natural forum-style communication channel. The agent can create discussions for topics that don't fit the issue tracker — announcements, design proposals, Q&A, polls.

```graphql
mutation {
  createDiscussion(input: {
    repositoryId: "R_xxxxx"
    categoryId: "DIC_xxxxx"
    title: "RFC: New authentication approach"
    body: "## Proposal\n\nHere's what I'm thinking..."
  }) {
    discussion { id number }
  }
}
```

### 11b. Discussion Answer Marked

**What triggers it:**
In a Q&A category discussion, an answer is marked as the accepted answer by the discussion author or a maintainer.

**Who receives the email:**
- All discussion thread subscribers.

**What the email contains:**
- A notice that an answer was marked, with a link to the accepted answer.

**How GitClaw can use it:**
The agent can mark answers in Q&A discussions, notifying all thread subscribers that the question has been resolved. This signals closure and delivers the answer to anyone following the thread.

---

## 12. Discussion Comment Events

### 12a. Comment Added to Discussion

**What triggers it:**
A comment (or reply) is posted on an existing discussion. Via web UI or GraphQL (`addDiscussionComment` mutation).

**Who receives the email:**
- Every user subscribed to that discussion thread.
- Every user @mentioned in the comment.

**What the email contains:**
- **Subject:** `Re: [owner/repo] Discussion title (#number)` (threaded)
- **Body:** The comment text, rendered from Markdown.

**How GitClaw can use it:**
Same communication pattern as issue comments. The agent posts a discussion comment, and all thread subscribers receive the content via email. Discussions may be preferred over issues for ongoing conversations that don't represent trackable work items.

---

## 13. Security Alert Events

### 13a. Dependabot Vulnerability Alert Created

**What triggers it:**
GitHub's dependency graph detects a known vulnerability (CVE) in a repository dependency. This is automatic — no API call triggers it. However, Dependabot must be enabled on the repository.

**Who receives the email:**
- Repository administrators.
- Organization owners.
- Users or teams configured as security managers in the organization.
- Anyone who has enabled "Vulnerability alerts" in their repository notification settings.

**What the email contains:**
- The affected dependency name and version.
- The vulnerability severity (critical, high, medium, low).
- The CVE identifier and description.
- Remediation guidance (upgrade to version X).

**How GitClaw can use it:**
GitClaw cannot directly trigger Dependabot alerts, but it can react to them. When a Dependabot alert appears, the agent can:
- Create a follow-up issue with triage analysis and remediation steps
- @mention the relevant code owners
- The follow-up issue/comment generates its own notification emails

### 13b. Secret Scanning Alert Created

**What triggers it:**
GitHub detects a known secret pattern (API key, access token, private key) in repository content. This is automatic on push.

**Who receives the email:**
- Repository administrators.
- Organization security managers.
- The user who committed the secret (if identifiable).

**What the email contains:**
- The type of secret detected.
- The file and location.
- Remediation instructions (revoke and rotate).

**How GitClaw can use it:**
Same as Dependabot — the agent cannot trigger these alerts, but it can amplify them. When a secret is detected, the agent can create an urgent issue with @mentions to relevant team members, generating additional notification emails with detailed containment instructions.

### 13c. Code Scanning (CodeQL) Alert Created

**What triggers it:**
A CodeQL analysis or third-party SARIF upload identifies a security vulnerability in the code.

**Who receives the email:**
- Repository administrators.
- Organization security managers.
- Users who have enabled code scanning notifications.

**What the email contains:**
- The vulnerability type and severity.
- The affected file and line.
- A description of the vulnerability pattern.

**How GitClaw can use it:**
The agent can triage CodeQL alerts and create issues or PR comments with detailed analysis, @mentioning affected code owners. Each of these actions generates its own notification emails, amplifying the security alert to the right people.

---

## 14. GitHub Actions Workflow Events

### 14a. Workflow Run Failed

**What triggers it:**
A GitHub Actions workflow run completes with a `failure` status.

**Who receives the email:**
- The user who triggered the workflow. This is determined by the event:
  - `push` → the pusher
  - `pull_request` → the PR author
  - `workflow_dispatch` → the user who clicked "Run workflow"
  - `schedule` → the repository owner or the last user to edit the workflow file

**Requirements for email delivery:**
- The user must have **Actions email notifications enabled** in their GitHub notification settings (Settings → Notifications → GitHub Actions → check "Send notifications for failed workflows only" or similar).
- This is **not enabled by default**. Many users never enable it.

**What the email contains:**
- The workflow name and run number.
- The failure status.
- A link to the workflow run logs.
- The triggering event and commit.

**How GitClaw can use it:**
The agent runs inside GitHub Actions. If the agent's own workflow fails (due to an error, timeout, or crash), the triggering user receives a failure notification email. This serves as an unintentional but useful "the agent couldn't respond" notification.

The agent could also intentionally trigger other workflows (via `workflow_dispatch` or `repository_dispatch`) that are designed to fail, as a notification mechanism. However, this is a hack — there are better channels.

### 14b. Workflow Run Recovered (Fixed After Failure)

**What triggers it:**
A workflow that previously failed now succeeds on a subsequent run.

**Who receives the email:**
- Same as the failure notification — the triggering user, if they have Actions notifications enabled.

**What the email contains:**
- A "fixed" notification indicating the workflow is passing again.

---

## 15. Deployment Review Events

### 15a. Deployment Review Requested

**What triggers it:**
A GitHub Actions workflow attempts to deploy to an environment that has **required reviewers** configured. The workflow pauses and a review is requested.

**Who receives the email:**
- Every user or team member listed as a required reviewer for that environment. This is configured in the repository settings under Settings → Environments → [environment name] → Protection rules → Required reviewers.

**What the email contains:**
- The workflow name and environment name.
- A link to the pending deployment review.
- The triggering commit and branch.

**How GitClaw can use it:**
If the agent's workflow includes deployment steps gated by environment approvals, the required reviewers receive email notifications. This creates an **approval-gated notification channel** — the agent can request human approval for sensitive operations, and the approval request is delivered via email.

This is particularly useful for high-consequence actions: "GitClaw wants to publish a release to production. Approve?"

```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production  # Has required reviewers configured
    steps:
      - run: echo "This step only runs after a human approves via email"
```

---

## 16. Collaborator Invitation Events

### 16a. Repository Collaborator Invitation

**What triggers it:**
A user is invited to collaborate on a repository. This can happen via the web UI or API (`PUT /repos/{owner}/{repo}/collaborators/{username}`).

**Who receives the email:**
- The invited user — at their **GitHub-registered email address**.

**What the email contains:**
- An invitation to collaborate on the repository.
- The permission level being granted (read, triage, write, maintain, admin).
- Accept/decline buttons.

**How GitClaw can use it:**
This is a highly targeted notification mechanism. The agent can invite a user as a collaborator, and that user receives a personalized email. However, this is an extreme measure — it changes repository access controls. It should only be used when the agent genuinely needs to grant access, not as a notification hack.

---

## 17. Organization & Team Membership Events

### 17a. Organization Invitation

**What triggers it:**
A user is invited to join an organization. Triggered via the web UI or API (`POST /orgs/{org}/invitations`).

**Who receives the email:**
- The invited user.

**What the email contains:**
- An invitation to join the organization.
- The organization name and a join link.

### 17b. Team Membership Added

**What triggers it:**
A user is added to a team within an organization. Triggered via the web UI or API (`PUT /orgs/{org}/teams/{team_slug}/memberships/{username}`).

**Who receives the email:**
- The added user.

**What the email contains:**
- A notification of team membership addition.
- The team name and organization.

**GitClaw implication:**
These are administrative actions with significant access control implications. They should not be used as notification mechanisms, but they do generate emails as a side effect.

---

## 18. Team Discussion Events

### 18a. Team Discussion Created

**What triggers it:**
A discussion is created within an organization team. Triggered via API (`POST /orgs/{org}/teams/{team_slug}/discussions`).

**Who receives the email:**
- Every member of the team (unless they have muted team notifications).

**What the email contains:**
- **Subject:** `[org/team] Discussion title`
- **Body:** The discussion body, rendered from Markdown.

**How GitClaw can use it:**
Team discussions are a **private broadcast channel**. The agent can post a discussion to a specific team, and every team member receives an email. This is private (not visible to non-team-members), targeted (only the specified team), and content-rich (full Markdown support).

```
API: POST /orgs/{org}/teams/{team_slug}/discussions
Body: { "title": "Weekly Agent Report", "body": "## Summary\n\n..." }
Result: Every team member receives an email with the report.
```

### 18b. Team Discussion Comment Added

**What triggers it:**
A comment is posted on an existing team discussion. Triggered via API (`POST /orgs/{org}/teams/{team_slug}/discussions/{discussion_number}/comments`).

**Who receives the email:**
- Every team member subscribed to that discussion thread (all team members are subscribed by default).

---

## 19. GitHub Pages Build Failure Events

### 19a. Pages Build Failed

**What triggers it:**
A GitHub Pages build fails. This happens when content is pushed to the Pages source branch and the build (Jekyll or custom) fails.

**Who receives the email:**
- Repository administrators.

**What the email contains:**
- The build failure details.
- A link to the build log.

**How GitClaw can use it:**
Extremely limited as a communication channel. The agent could intentionally push broken Pages content to trigger a failure notification to admins, but this is absurd. Noted only for completeness.

---

## 20. Sponsorship Events

### 20a. New Sponsorship

**What triggers it:**
A user begins sponsoring another user or organization via GitHub Sponsors.

**Who receives the email:**
- The sponsored user/organization.

### 20b. Sponsorship Tier Changed

**What triggers it:**
A sponsor changes their sponsorship tier.

**Who receives the email:**
- The sponsored user/organization.

### 20c. Sponsorship Cancelled

**What triggers it:**
A sponsor cancels their sponsorship.

**Who receives the email:**
- The sponsored user/organization.

**GitClaw implication:**
Sponsorship events are not useful as communication channels. They are financial transactions that happen to generate notifications.

---

## 21. Repository Transfer Events

### 21a. Repository Transferred

**What triggers it:**
A repository is transferred from one owner to another.

**Who receives the email:**
- The previous owner.
- The new owner.

**GitClaw implication:**
Not useful as a communication channel.

---

## 22. Assignment Events

### 22a. Issue Assigned

**What triggers it:**
A user is assigned to an issue. Triggered via the web UI or API (`PATCH /repos/{owner}/{repo}/issues/{number}` with `"assignees": ["username"]`, or `POST /repos/{owner}/{repo}/issues/{number}/assignees`).

**Who receives the email:**
- The assigned user.

**What the email contains:**
- A notification that they have been assigned to the issue.
- The issue title and a link.

**How GitClaw can use it:**
Assigning a user to an issue is a **targeted notification** — the assigned user receives a direct email. The agent can use assignment as a way to "page" a specific person about a specific issue. The assigned user also becomes subscribed to the issue thread, so they will receive all future comments.

```
API: POST /repos/{owner}/{repo}/issues/{number}/assignees
Body: { "assignees": ["alice"] }
Result: alice receives an email that she has been assigned, and is now subscribed to all future activity on the issue.
```

### 22b. Pull Request Assigned

**What triggers it:**
Same mechanism as issue assignment, applied to a pull request.

**Who receives the email:**
- The assigned user.

**How GitClaw can use it:**
Same pattern as issue assignment. Assigning a user to a PR pages them and subscribes them to the PR thread.

---

## 23. Label Subscription Events

### 23a. Issue Created or Labeled with a Subscribed Label

**What triggers it:**
A user has configured **custom notification routing** in their repository watch settings to receive notifications for issues with specific labels. When an issue is created with that label, or an existing issue has that label added, the user is notified.

**Who receives the email:**
- Users who have subscribed to that specific label in their watch settings for the repository.

**What the email contains:**
- The standard issue creation or labeling notification.

**How GitClaw can use it:**
If users have subscribed to specific labels (e.g., `urgent`, `security`, `agent-alert`), the agent can apply those labels to issues to trigger targeted notifications. This is a **topic-based subscription** model — users opt into specific categories of notifications.

```
API: POST /repos/{owner}/{repo}/issues
Body: { "title": "Security alert", "body": "...", "labels": ["security", "urgent"] }
Result: Users subscribed to the "security" or "urgent" labels receive email.
```

---

## 24. Scheduled Reminder Events

### 24a. Pending Review Reminder

**What triggers it:**
An organization admin has configured **Scheduled Reminders** (Settings → Scheduled Reminders). These are periodic reminders sent to teams about pending pull request reviews.

**Who receives the email:**
- Members of the team configured in the scheduled reminder. Delivery can be via Slack or email.

**What the email contains:**
- A list of PRs awaiting review by the team.
- Age of each pending review.

**How GitClaw can use it:**
The agent cannot directly trigger scheduled reminders, but it can ensure that PRs it creates have pending review requests. Those PRs will then appear in scheduled reminder emails to the appropriate teams. This is an indirect notification channel — the agent creates the work, and GitHub's reminder system nags about it.

---

## 25. The Reply-To Channel

This deserves its own section because it creates a **bidirectional email communication channel**.

### How It Works

Every notification email GitHub sends includes a special `Reply-To` header:

```
Reply-To: reply+AHTK3LRZV7TGBJNQBFMZ6PTVKX7TAJ5EVBNHHYQDTYZQ@reply.github.com
```

The token in this address is unique to the specific thread and recipient. When a user replies to a GitHub notification email from their email client, the reply is:

1. Received by GitHub's mail servers at `reply.github.com`.
2. Authenticated using the unique token (verifying the sender).
3. Posted as a **comment on the original thread** (issue, PR, discussion, or commit).

### What This Means

A user can participate in GitHub conversations **entirely from their email client**. They never need to visit github.com. They receive the agent's messages as emails, type a reply in their email client, and that reply appears as a comment on the issue — which triggers the agent to respond — which sends another email.

**The entire human-agent conversation can happen over email.**

### Constraints

- The reply must come from the same email address registered to the user's GitHub account.
- Attachments are supported (images are uploaded and embedded).
- Email signatures may be included in the comment (GitHub attempts to strip them but is not always successful).
- HTML formatting in the reply is converted to Markdown.
- The reply must be to the most recent notification in the thread (replying to an old email may fail or create an out-of-order comment).

### GitClaw Implication

This is profound. It means that GitClaw can serve users who never open GitHub in a browser. The agent posts a comment, the user gets an email, the user replies to the email, the reply becomes a comment, the agent processes the comment, and the cycle continues. The email client IS the interface.

---

## 26. Email Anatomy

Every GitHub notification email follows a consistent structure. Understanding this structure is essential for designing effective agent-to-human communication.

### Headers

| Header | Value | Purpose |
|--------|-------|---------|
| `From` | `notifications@github.com` | Sender address — always the same |
| `Reply-To` | `reply+<token>@reply.github.com` | Unique per-thread, per-recipient reply address |
| `To` | `repo-name@noreply.github.com` | Repository-scoped address |
| `Cc` | `<user>@noreply.github.com` | Reason-specific CC (mention, assign, etc.) |
| `Subject` | `[owner/repo] Title (#number)` or `Re: [owner/repo] Title (#number)` | Threaded subject line |
| `Message-ID` | `<owner/repo/issues/42/1234567890@github.com>` | Unique message identifier |
| `In-Reply-To` | `<owner/repo/issues/42@github.com>` | Threading reference |
| `References` | Parent message IDs | Full thread chain |
| `List-ID` | `owner/repo <repo.owner.github.com>` | Mailing list identifier for filtering |
| `List-Unsubscribe` | One-click unsubscribe URL | Allows email clients to show unsubscribe button |
| `X-GitHub-Reason` | `mention`, `assign`, `team_mention`, `subscribed`, `review_requested`, `author`, `comment`, `state_change`, `push`, `ci_activity` | Why this user received this email |
| `X-GitHub-Sender` | Username of the person who triggered the event | Who caused the notification |

### Body Structure

The email body is **multipart/alternative** with both:
- **text/plain** — A plain-text rendering of the content, suitable for terminal email clients.
- **text/html** — A fully rendered HTML version with GitHub's styling, including:
  - Syntax-highlighted code blocks
  - Rendered Markdown tables
  - Embedded images (as linked URLs, not inline attachments)
  - Clickable links
  - User avatar images
  - Repository and issue/PR context bar

### Subject Line Format

| Event Type | Subject Format |
|------------|---------------|
| Issue created | `[owner/repo] Title (#42)` |
| Issue comment | `Re: [owner/repo] Title (#42)` |
| PR created | `[owner/repo] Title (#17)` |
| PR review | `Re: [owner/repo] Title (#17)` |
| Release | `[owner/repo] Release title` |
| Discussion | `[owner/repo] Title (#5)` |
| Commit comment | `Re: [owner/repo] Commit message (abc1234)` |
| Team discussion | `[org/team] Title` |
| Assignment | `Re: [owner/repo] Title (#42)` |

**GitClaw control over subject line:**
The agent controls the **title** of the issue, PR, release, or discussion. This title becomes the subject line. This means the agent can craft subject lines that are informative, urgent, or attention-grabbing — within the `[owner/repo]` prefix constraint.

---

## 27. Recipient Determination Logic

This section describes the precise algorithm GitHub uses to determine who receives an email for each event type.

### Step 1: Compute the Candidate Set

```
candidates = ∅

IF event is on a specific thread (issue, PR, discussion):
    candidates += all users subscribed to that thread
    candidates += all users @mentioned in the event content
    candidates += all team members of @mentioned teams

IF event is repo-wide (issue created, PR opened, release published):
    candidates += all users watching the repo at a qualifying level

IF event targets specific users (assignment, review request):
    candidates += targeted users
```

### Step 2: Apply Per-User Filters

```
FOR each candidate:
    IF user has globally disabled email notifications → REMOVE
    IF user has muted this specific thread → REMOVE
    IF user has muted this repository → REMOVE
    IF user is the actor who triggered the event → REMOVE (by default; configurable)
    IF user's custom routing rules exclude this event type → REMOVE
```

### Step 3: Deduplicate

```
Each remaining user receives exactly ONE email per event, even if they qualify
through multiple paths (e.g., both @mentioned AND watching the repo).
```

### Step 4: Set X-GitHub-Reason

```
The X-GitHub-Reason header is set to the HIGHEST-PRIORITY reason:
    mention > team_mention > assign > review_requested > subscribed > author > ...
```

### Important: The Actor Exclusion

By default, **the user who triggers the event does NOT receive a notification email for that event**. This means if the agent (as `github-actions[bot]`) creates an issue, `github-actions[bot]` does not receive a notification. This is irrelevant for the bot identity but important to understand: the agent's actions never generate self-notifications.

However, if a human user triggers the agent (by commenting on an issue), and the agent's response is posted via the API using `GITHUB_TOKEN`, the response is attributed to `github-actions[bot]`, not the human — so the **human DOES receive the notification** of the agent's response. This is the correct behavior.

---

## 28. Notification Preference Controls

Users have fine-grained control over their notification preferences. Understanding these controls helps predict who will and won't receive agent communications.

### Global Settings (Settings → Notifications)

| Setting | Effect |
|---------|--------|
| **Default notification email** | Which email address receives notifications |
| **Custom routing** | Different repos can send to different email addresses |
| **Participating** | Email for threads the user is directly involved in |
| **Watching** | Email for all activity on watched repos |
| **GitHub Actions** | Email for workflow run failures (off by default) |
| **Dependabot alerts** | Email for vulnerability alerts |
| **Email preferences: Include own updates** | Whether to receive notifications for events the user triggered (off by default) |

### Per-Repository Settings

| Setting | Effect |
|---------|--------|
| **Watch level** | All Activity / Participating / Releases Only / Ignoring / Custom |
| **Custom watch: Issues** | Receive email for issue events |
| **Custom watch: Pull Requests** | Receive email for PR events |
| **Custom watch: Releases** | Receive email for release events |
| **Custom watch: Discussions** | Receive email for discussion events |
| **Custom watch: Security alerts** | Receive email for security events |

### Per-Thread Settings

| Setting | Effect |
|---------|--------|
| **Subscribe** | Receive all future notifications for this thread |
| **Unsubscribe** | Stop receiving notifications for this thread |
| **Mute** | Block ALL notifications for this thread, even @mentions |

### GitClaw Implication

The agent cannot control whether a user has email notifications enabled. It can only control:
- **Which events it triggers** (issue creation, comments, reviews, releases, etc.)
- **Who it targets** (@mentions, assignments, review requests)
- **What content it delivers** (the Markdown body of comments, issues, reviews, etc.)

The rest is up to the user's notification preferences.

---

## 29. Constraints, Limits & Risks

### Hard Constraints

| Constraint | Description |
|------------|-------------|
| **GitHub users only** | Emails can only reach GitHub users. There is no mechanism to send to arbitrary email addresses. |
| **Consent-based** | Users must have opted into notifications. The agent cannot force email delivery. |
| **No custom From address** | All emails come from `notifications@github.com`. The agent cannot spoof a sender. |
| **No custom subject line** | Subject lines follow GitHub's format: `[owner/repo] Title (#number)`. The agent controls only the title portion. |
| **No custom email template** | The email body format is controlled by GitHub. The agent controls only the Markdown content within that format. |
| **No BCC** | All recipients are visible (to the extent that thread subscriptions are visible). |
| **No delivery confirmation** | The agent cannot verify whether a notification email was delivered or read. |

### Rate Limits

| Limit | Value | Notes |
|-------|-------|-------|
| **API rate limit** | 5,000 requests/hour (REST) or 5,000 points/hour (GraphQL) | Per authenticated user/token |
| **Secondary rate limit** | ~80 content-creating requests/minute | Applies to creating issues, comments, reviews, etc. |
| **Notification throttling** | Undocumented | GitHub may delay or suppress notifications for accounts generating excessive activity |
| **Abuse detection** | Undocumented | Rapid-fire @mentions or issue creation may trigger abuse detection and temporary throttling |

### Risks

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Notification fatigue** | If the agent sends too many notifications, users will unwatch the repo or mute threads | Rate-limit agent notifications; batch updates; use reactions instead of comments for minor updates |
| **Spam perception** | Excessive @mentions or issue creation may be perceived as spam by users or by GitHub's abuse detection | Mention users only when truly necessary; prefer thread subscriptions over @mentions |
| **Content visibility** | Issue and PR content is visible to anyone with repo access (or publicly, for public repos) | Use private repos for sensitive communications; use team discussions for private team comms |
| **Thread pollution** | Using issues or PRs purely as notification vehicles clutters the repository | Use dedicated labels to distinguish notification-issues from real issues; close automatically |
| **Unsubscribe cascade** | A poorly calibrated agent that sends too many notifications may cause a mass unsubscribe from the repo | Monitor watch counts; implement notification budgets |

---

## 30. GitClaw Communication Patterns

Combining the mechanisms above, here are the concrete communication patterns GitClaw can implement.

### Pattern 1: Direct Response (Current)

```
Trigger:    User comments on an issue
Mechanism:  Agent posts a comment on the same issue (§3a)
Recipients: All issue thread subscribers
Channel:    Issue comment → email notification → reply-to for continued conversation
```

This is what GitClaw does today. It is the most natural and effective pattern.

### Pattern 2: Targeted Page

```
Trigger:    Agent determines a specific person needs to be notified
Mechanism:  Agent @mentions the user in a comment (§7a) OR assigns them to an issue (§22a)
Recipients: The specific targeted user
Channel:    @mention or assignment → direct email notification
```

Use case: The agent detects a critical security issue and pages the security team lead.

### Pattern 3: Team Broadcast

```
Trigger:    Agent has information for an entire team
Mechanism:  Agent @mentions @org/team in a comment (§8a) OR creates a team discussion (§18a)
Recipients: Every member of the team
Channel:    Team mention → individual emails to all team members
```

Use case: The agent posts a weekly summary to the platform team.

### Pattern 4: Community Broadcast

```
Trigger:    Agent has information for all repository watchers
Mechanism:  Agent publishes a release (§10a) OR creates an issue (§2a)
Recipients: All repository watchers at the qualifying watch level
Channel:    Release or issue creation → email to all watchers
```

Use case: The agent announces a new version or a major change.

### Pattern 5: Review Request Page

```
Trigger:    Agent has created a PR that needs human review
Mechanism:  Agent creates a PR and requests specific reviewers (§5b)
Recipients: The requested reviewers
Channel:    Review request → direct email to each reviewer
```

Use case: The agent has implemented a fix and needs human approval before merge.

### Pattern 6: Approval Gate

```
Trigger:    Agent needs human approval for a high-consequence action
Mechanism:  Agent's workflow requires environment approval (§15a)
Recipients: Required reviewers for the environment
Channel:    Deployment review request → email to approvers
```

Use case: The agent wants to publish to production and needs sign-off.

### Pattern 7: Commit Author Notification

```
Trigger:    Agent needs to notify the author of a specific commit
Mechanism:  Agent posts a comment on the commit (§9a)
Recipients: The commit author
Channel:    Commit comment → email to the author
```

Use case: The agent detects a security issue in a specific commit and notifies the author.

### Pattern 8: Email-Only Conversation

```
Trigger:    User interacts entirely via email
Mechanism:  Agent posts comments (§3a); user replies via email Reply-To (§25)
Recipients: The user, via email
Channel:    Bidirectional: agent comment → email → user email reply → GitHub comment → agent processes → repeat
```

Use case: A team member who lives in their email client and never opens GitHub.

### Pattern 9: Label-Based Routing

```
Trigger:    Agent creates or labels an issue with a specific label
Mechanism:  Users have subscribed to that label in their watch settings (§23a)
Recipients: Users subscribed to the label
Channel:    Label application → email to label subscribers
```

Use case: The agent applies a `security` label, and all users subscribed to security notifications receive email.

### Pattern 10: Escalation Chain

```
Step 1:     Agent posts a comment on the issue (§3a) — notifies thread subscribers
Step 2:     No response within N minutes → agent @mentions the issue assignee (§7a)
Step 3:     No response within N more minutes → agent @mentions @org/oncall-team (§8a)
Step 4:     No response within N more minutes → agent creates a new high-priority issue and assigns the team lead (§2a + §22a)
```

Use case: Progressive escalation for critical issues, with each step reaching a wider audience via more aggressive notification mechanisms.

---

## 31. Trigger-by-Trigger API Reference

A quick reference for every mechanism, the API call that triggers it, and the resulting email.

| # | Mechanism | API Call | Email Recipients |
|---|-----------|----------|-----------------|
| 1 | Issue created | `POST /repos/{o}/{r}/issues` | Repo watchers |
| 2 | Issue comment | `POST /repos/{o}/{r}/issues/{n}/comments` | Thread subscribers |
| 3 | Issue closed | `PATCH /repos/{o}/{r}/issues/{n}` `state:closed` | Thread subscribers |
| 4 | Issue reopened | `PATCH /repos/{o}/{r}/issues/{n}` `state:open` | Thread subscribers |
| 5 | Issue assigned | `POST /repos/{o}/{r}/issues/{n}/assignees` | Assigned user |
| 6 | Issue labeled | `POST /repos/{o}/{r}/issues/{n}/labels` | Label subscribers |
| 7 | PR created | `POST /repos/{o}/{r}/pulls` | Repo watchers |
| 8 | PR comment | `POST /repos/{o}/{r}/issues/{n}/comments` (PRs use issues API) | Thread subscribers |
| 9 | PR review submitted | `POST /repos/{o}/{r}/pulls/{n}/reviews` | PR author + thread subscribers |
| 10 | PR review requested | `POST /repos/{o}/{r}/pulls/{n}/requested_reviewers` | Requested reviewer(s) |
| 11 | PR inline comment | `POST /repos/{o}/{r}/pulls/{n}/comments` | Thread subscribers |
| 12 | PR merged | `PUT /repos/{o}/{r}/pulls/{n}/merge` | Thread subscribers |
| 13 | PR commits pushed | `git push` to PR head branch | Thread subscribers |
| 14 | @user mention | Include `@username` in any body/comment text | Mentioned user |
| 15 | @team mention | Include `@org/team` in any body/comment text | All team members |
| 16 | Commit comment | `POST /repos/{o}/{r}/commits/{sha}/comments` | Commit author + thread subscribers |
| 17 | Release published | `POST /repos/{o}/{r}/releases` `draft:false` | All release watchers |
| 18 | Discussion created | GraphQL `createDiscussion` mutation | Repo watchers (with Discussions) |
| 19 | Discussion comment | GraphQL `addDiscussionComment` mutation | Thread subscribers |
| 20 | Discussion answered | GraphQL `markDiscussionCommentAsAnswer` mutation | Thread subscribers |
| 21 | Team discussion created | `POST /orgs/{o}/teams/{t}/discussions` | All team members |
| 22 | Team discussion comment | `POST /orgs/{o}/teams/{t}/discussions/{n}/comments` | Discussion subscribers |
| 23 | Collaborator invited | `PUT /repos/{o}/{r}/collaborators/{u}` | Invited user |
| 24 | Org invitation | `POST /orgs/{o}/invitations` | Invited user |
| 25 | Environment review requested | Workflow hits `environment:` with required reviewers | Environment reviewers |
| 26 | Workflow failed | Workflow run completes with `failure` | Triggering user (if opted in) |
| 27 | Security advisory created | `POST /repos/{o}/{r}/security-advisories` | Repo admins + security managers |

---

## 32. Conclusion

GitHub's notification system is a **comprehensive, consent-based, bidirectional email communication channel** that GitClaw can exploit without any external email infrastructure.

### Key Takeaways

1. **Issue comments are the primary channel.** Every comment the agent posts on an issue becomes an email to all thread subscribers. This is the mechanism GitClaw uses today, and it is the most versatile and reliable channel.

2. **@mentions are the most aggressive targeting mechanism.** They override watch/subscription defaults and reach any GitHub user with email notifications enabled. Use them for urgent, targeted communication.

3. **Team mentions are the broadcast mechanism.** A single `@org/team` mention emails every team member. This is the closest thing to a mailing list.

4. **Releases are the widest broadcast.** They reach "Releases Only" watchers — users who have opted out of all other notifications. Use releases for community-wide announcements.

5. **The Reply-To channel makes conversations bidirectional.** Users can reply to notification emails from their email client, and those replies become GitHub comments that the agent can process. The entire human-agent conversation can happen over email.

6. **Assignment and review requests are precision pages.** They target specific individuals with direct email notifications and subscribe those individuals to future thread activity.

7. **The agent cannot email arbitrary addresses.** All recipients must be GitHub users with email notifications enabled. This is a fundamental constraint of the platform.

8. **Content control is through Markdown.** The agent controls the Markdown body of its comments, issues, reviews, and releases. GitHub controls the email template, headers, and subject format.

9. **Notification fatigue is the primary risk.** An agent that over-communicates will cause users to unwatch and disengage. Calibrate frequency and targeting carefully.

10. **No external infrastructure is needed.** GitHub IS the email system. The agent triggers events via the GitHub API, and GitHub handles email construction, delivery, threading, and unsubscription. Zero additional services required.

### The Communication Stack

```
┌─────────────────────────────────────────────┐
│  Human's Email Client                        │
│  (receives notifications, sends replies)     │
├─────────────────────────────────────────────┤
│  GitHub Notification System                  │
│  (constructs emails, routes to recipients,   │
│   processes reply-to responses)              │
├─────────────────────────────────────────────┤
│  GitHub API                                  │
│  (issues, comments, reviews, releases,       │
│   @mentions, assignments, team discussions)  │
├─────────────────────────────────────────────┤
│  GitClaw Agent                               │
│  (decides what to communicate, to whom,      │
│   and through which mechanism)               │
├─────────────────────────────────────────────┤
│  GitHub Actions                              │
│  (triggers agent execution on events)        │
└─────────────────────────────────────────────┘
```

No servers. No SMTP configuration. No SendGrid. No Mailgun. Just the GitHub platform doing what it already does — sending notification emails — with GitClaw deciding what triggers them.

🦞 *The claw doesn't need a mail server. The claw has GitHub.*

---

### Related Documents

- [GITCLAW-The-Idea.md](GITCLAW-The-Idea.md) — The vision and philosophy
- [GITCLAW-The-GitHub-Possibilities.md](GITCLAW-The-GitHub-Possibilities.md) — Complete GitHub feature analysis
- [GITCLAW-Interactive-Possibilities.md](GITCLAW-Interactive-Possibilities.md) — Interactive workflow patterns
- [GITCLAW-Possibilities.md](GITCLAW-Possibilities.md) — Use case catalogue
- [GITCLAW-Internal-Mechanics.md](GITCLAW-Internal-Mechanics.md) — Architecture and internals
- [GITCLAW-Roadmap.md](GITCLAW-Roadmap.md) — Implementation plan

---

_Last updated: 2026-02-20_
