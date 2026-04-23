# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GitHub MCP Server — exposes GitHub REST API as MCP tools (read-only).
#
# Usage:
#   export GITHUB_TOKEN="github_pat_..."
#   uv run python server.py
#
# Listens on http://0.0.0.0:8102 (streamable-http) at path /mcp

from __future__ import annotations

import logging
import os
from typing import Optional

import aiohttp
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_port = int(os.environ.get("GITHUB_PORT", "8102"))
mcp = FastMCP("github", host="0.0.0.0", port=_port)

GITHUB_API = "https://api.github.com"


def _auth_headers() -> dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise RuntimeError("GITHUB_TOKEN environment variable is not set")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def _get(path: str, params: dict | None = None) -> dict | list:
    url = f"{GITHUB_API}{path}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=_auth_headers(), params=params or {}) as resp:
            body = await resp.json()
            if resp.status >= 400:
                raise RuntimeError(f"GitHub API error {resp.status}: {body.get('message', body)}")
            return body


@mcp.tool()
async def search_issues(
    query: str,
    repo: Optional[str] = None,
    max_results: int = 20,
) -> str:
    """Search GitHub issues and pull requests using GitHub's search syntax.

    Args:
        query: Search query. Examples:
            - 'mcp client bug' — keyword search
            - 'is:open label:bug' — open bugs
            - 'is:pr is:open review-requested:@me' — PRs awaiting your review
            - 'author:afourniernv is:open' — open issues/PRs by a user
        repo: Optional repo to scope search, e.g. 'nvidia/NeMo-Agent-Toolkit'.
        max_results: Max results to return. Defaults to 20.

    Returns:
        Formatted list of matching issues/PRs with number, title, state, and URL.
    """
    q = query
    if repo:
        q = f"repo:{repo} {q}"

    data = await _get("/search/issues", params={"q": q, "per_page": str(min(max_results, 30))})
    items = data.get("items", [])
    total = data.get("total_count", 0)

    if not items:
        return f"No results found for: {q}"

    lines = [f"Found {len(items)} of {total} result(s):\n"]
    for item in items:
        pr_label = " [PR]" if "pull_request" in item else ""
        lines.append(
            f"#{item['number']}{pr_label} [{item['state'].upper()}] {item['title']}\n"
            f"   Repo: {item['repository_url'].split('repos/')[-1]}\n"
            f"   By: {item['user']['login']} | {item['created_at'][:10]}\n"
            f"   URL: {item['html_url']}\n"
        )
    return "\n".join(lines)


@mcp.tool()
async def list_open_prs(repo: str, max_results: int = 20) -> str:
    """List open pull requests in a GitHub repository.

    Args:
        repo: Repository in 'owner/name' format, e.g. 'nvidia/NeMo-Agent-Toolkit'.
        max_results: Max number of PRs to return. Defaults to 20.

    Returns:
        List of open PRs with number, title, author, and review status.
    """
    data = await _get(
        f"/repos/{repo}/pulls",
        params={"state": "open", "per_page": str(min(max_results, 30)), "sort": "updated"},
    )
    if not data:
        return f"No open PRs found in {repo}."

    lines = [f"Open PRs in {repo} ({len(data)}):\n"]
    for pr in data:
        reviewers = [r["login"] for r in pr.get("requested_reviewers", [])]
        labels = [l["name"] for l in pr.get("labels", [])]
        lines.append(
            f"#{pr['number']} {pr['title']}\n"
            f"   Author: {pr['user']['login']} | Updated: {pr['updated_at'][:10]}\n"
            + (f"   Reviewers: {', '.join(reviewers)}\n" if reviewers else "")
            + (f"   Labels: {', '.join(labels)}\n" if labels else "")
            + f"   URL: {pr['html_url']}\n"
        )
    return "\n".join(lines)


@mcp.tool()
async def get_pr(repo: str, pr_number: int) -> str:
    """Get full details of a specific pull request.

    Args:
        repo: Repository in 'owner/name' format, e.g. 'nvidia/NeMo-Agent-Toolkit'.
        pr_number: PR number.

    Returns:
        PR details including description, status checks, reviewers, and changed files.
    """
    pr = await _get(f"/repos/{repo}/pulls/{pr_number}")
    files_data = await _get(f"/repos/{repo}/pulls/{pr_number}/files", params={"per_page": "20"})
    reviews_data = await _get(f"/repos/{repo}/pulls/{pr_number}/reviews")

    files = [f["filename"] for f in files_data] if isinstance(files_data, list) else []
    reviews = {}
    if isinstance(reviews_data, list):
        for r in reviews_data:
            reviews[r["user"]["login"]] = r["state"]

    body = (pr.get("body") or "(no description)")[:500]
    reviewers = [r["login"] for r in pr.get("requested_reviewers", [])]

    return (
        f"PR #{pr['number']}: {pr['title']}\n"
        f"State: {pr['state'].upper()} | Merged: {pr.get('merged', False)}\n"
        f"Author: {pr['user']['login']} | Branch: {pr['head']['ref']} → {pr['base']['ref']}\n"
        f"Created: {pr['created_at'][:10]} | Updated: {pr['updated_at'][:10]}\n"
        f"Additions: +{pr.get('additions', '?')} Deletions: -{pr.get('deletions', '?')} "
        f"Changed files: {pr.get('changed_files', '?')}\n"
        + (f"Requested reviewers: {', '.join(reviewers)}\n" if reviewers else "")
        + (f"Review decisions: {reviews}\n" if reviews else "")
        + f"\nDescription:\n{body}\n"
        + (f"\nChanged files ({len(files)}):\n" + "\n".join(f"  {f}" for f in files[:20]) if files else "")
        + f"\n\nURL: {pr['html_url']}"
    )


@mcp.tool()
async def get_issue(repo: str, issue_number: int) -> str:
    """Get details of a specific GitHub issue.

    Args:
        repo: Repository in 'owner/name' format.
        issue_number: Issue number.

    Returns:
        Issue details including description, labels, assignees, and recent comments.
    """
    issue = await _get(f"/repos/{repo}/issues/{issue_number}")
    comments_data = await _get(
        f"/repos/{repo}/issues/{issue_number}/comments",
        params={"per_page": "3"},
    )

    labels = [l["name"] for l in issue.get("labels", [])]
    assignees = [a["login"] for a in issue.get("assignees", [])]
    body = (issue.get("body") or "(no description)")[:500]

    comments = ""
    if isinstance(comments_data, list):
        for c in comments_data[-3:]:
            comments += f"\n  [{c['user']['login']}]: {(c.get('body') or '')[:200]}"

    return (
        f"Issue #{issue['number']}: {issue['title']}\n"
        f"State: {issue['state'].upper()}\n"
        f"Author: {issue['user']['login']} | Created: {issue['created_at'][:10]}\n"
        + (f"Assignees: {', '.join(assignees)}\n" if assignees else "")
        + (f"Labels: {', '.join(labels)}\n" if labels else "")
        + f"\nDescription:\n{body}"
        + (f"\n\nRecent comments:{comments}" if comments else "")
        + f"\n\nURL: {issue['html_url']}"
    )


@mcp.tool()
async def list_repo_issues(
    repo: str,
    state: str = "open",
    label: Optional[str] = None,
    max_results: int = 20,
) -> str:
    """List issues in a GitHub repository.

    Args:
        repo: Repository in 'owner/name' format.
        state: 'open', 'closed', or 'all'. Defaults to 'open'.
        label: Optional label to filter by (e.g. 'bug', 'enhancement').
        max_results: Max results. Defaults to 20.

    Returns:
        List of issues with number, title, author, and labels.
    """
    params: dict = {"state": state, "per_page": str(min(max_results, 30)), "sort": "updated"}
    if label:
        params["labels"] = label

    data = await _get(f"/repos/{repo}/issues", params=params)
    # Filter out PRs (GitHub returns PRs in issues endpoint)
    issues = [i for i in (data if isinstance(data, list) else []) if "pull_request" not in i]

    if not issues:
        return f"No {state} issues found in {repo}."

    lines = [f"{state.capitalize()} issues in {repo} ({len(issues)}):\n"]
    for issue in issues:
        labels = [l["name"] for l in issue.get("labels", [])]
        lines.append(
            f"#{issue['number']} {issue['title']}\n"
            f"   By: {issue['user']['login']} | {issue['updated_at'][:10]}"
            + (f" | Labels: {', '.join(labels)}" if labels else "")
            + f"\n   URL: {issue['html_url']}\n"
        )
    return "\n".join(lines)


@mcp.tool()
async def get_repo_summary(repo: str) -> str:
    """Get a high-level summary of a GitHub repository.

    Args:
        repo: Repository in 'owner/name' format, e.g. 'nvidia/NeMo-Agent-Toolkit'.

    Returns:
        Repo description, stats, recent activity summary.
    """
    r = await _get(f"/repos/{repo}")
    open_prs = await _get(f"/repos/{repo}/pulls", params={"state": "open", "per_page": "5"})

    return (
        f"{r['full_name']}\n"
        f"{r.get('description', '(no description)')}\n\n"
        f"Stars: {r['stargazers_count']} | Forks: {r['forks_count']} | "
        f"Open issues: {r['open_issues_count']}\n"
        f"Default branch: {r['default_branch']} | Language: {r.get('language', 'N/A')}\n"
        f"Last pushed: {r['pushed_at'][:10]}\n"
        f"Open PRs (up to 5): {len(open_prs if isinstance(open_prs, list) else [])}\n"
        f"URL: {r['html_url']}"
    )


if __name__ == "__main__":
    logger.info("Starting GitHub MCP server on port %d", _port)
    mcp.run(transport="streamable-http")
