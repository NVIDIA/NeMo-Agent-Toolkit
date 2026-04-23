# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# JIRA MCP Server — exposes Atlassian JIRA Cloud REST API as MCP tools.
#
# Usage:
#   export JIRA_API_TOKEN="<your-atlassian-api-token>"
#   export JIRA_EMAIL="<your-atlassian-email>"
#   export JIRA_SITE="<yourname>.atlassian.net"
#   uv run python server.py
#
# The server listens on http://0.0.0.0:8101 (streamable-http) at path /mcp/.
# Register in NAT config as an mcp_client with url: http://host:8101/mcp/

from __future__ import annotations

import base64
import logging
import os
from typing import Optional

import aiohttp
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_port = int(os.environ.get("JIRA_PORT", "8101"))

mcp = FastMCP("jira", host="0.0.0.0", port=_port)


def _auth_headers() -> dict[str, str]:
    email = os.environ.get("JIRA_EMAIL", "")
    token = os.environ.get("JIRA_API_TOKEN", "")
    site = os.environ.get("JIRA_SITE", "")
    if not all([email, token, site]):
        raise RuntimeError("JIRA_EMAIL, JIRA_API_TOKEN, and JIRA_SITE must all be set")
    creds = base64.b64encode(f"{email}:{token}".encode()).decode()
    return {
        "Authorization": f"Basic {creds}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _base_url() -> str:
    site = os.environ.get("JIRA_SITE", "")
    return f"https://{site}/rest/api/3"


def _agile_base_url() -> str:
    site = os.environ.get("JIRA_SITE", "")
    return f"https://{site}/rest/agile/1.0"


async def _get(path: str, params: dict | None = None, *, agile: bool = False) -> dict:
    url = f"{_agile_base_url() if agile else _base_url()}{path}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=_auth_headers(), params=params or {}) as resp:
            body = await resp.json()
            if resp.status >= 400:
                raise RuntimeError(f"JIRA API error {resp.status}: {body.get('errorMessages', body)}")
            return body


async def _post(path: str, payload: dict, *, agile: bool = False) -> dict:
    url = f"{_agile_base_url() if agile else _base_url()}{path}"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=_auth_headers(), json=payload) as resp:
            body = await resp.json()
            if resp.status >= 400:
                raise RuntimeError(f"JIRA API error {resp.status}: {body.get('errorMessages', body)}")
            return body


def _format_issue(issue: dict) -> str:
    fields = issue.get("fields", {})
    assignee = fields.get("assignee") or {}
    reporter = fields.get("reporter") or {}
    status = fields.get("status", {}).get("name", "Unknown")
    priority = fields.get("priority", {}).get("name", "None")
    issue_type = fields.get("issuetype", {}).get("name", "")
    sprint_info = ""
    sprints = fields.get("customfield_10020")
    if sprints and isinstance(sprints, list):
        active = [s for s in sprints if s.get("state") == "active"]
        if active:
            sprint_info = f"\n   Sprint: {active[0].get('name', '')}"

    return (
        f"[{issue['key']}] {fields.get('summary', '')}\n"
        f"   Type: {issue_type} | Status: {status} | Priority: {priority}\n"
        f"   Assignee: {assignee.get('displayName', 'Unassigned')} | "
        f"Reporter: {reporter.get('displayName', 'Unknown')}"
        f"{sprint_info}\n"
        f"   URL: https://{os.environ.get('JIRA_SITE', '')}/browse/{issue['key']}\n"
    )


@mcp.tool()
async def search_issues(
    jql: str,
    max_results: int = 20,
) -> str:
    """Search JIRA issues using JQL (JIRA Query Language).

    Args:
        jql: JQL query string. Examples:
            - 'project = NAT AND status != Done'
            - 'assignee = currentUser() AND sprint in openSprints()'
            - 'priority = High AND status = "In Progress"'
            - 'text ~ "authentication" AND created >= -7d'
        max_results: Maximum number of issues to return. Defaults to 20.

    Returns:
        Formatted list of matching issues with key, summary, status, assignee.
    """
    data = await _get("/search/jql", params={
        "jql": jql,
        "maxResults": str(min(max_results, 50)),
        "fields": "summary,status,assignee,reporter,priority,issuetype,customfield_10020",
    })
    issues = data.get("issues", [])
    total = data.get("total", 0)

    if not issues:
        return f"No issues found for JQL: {jql}"

    lines = [f"Found {len(issues)} of {total} issue(s):\n"]
    for issue in issues:
        lines.append(_format_issue(issue))
    return "\n".join(lines)


@mcp.tool()
async def get_issue(issue_key: str) -> str:
    """Get full details of a specific JIRA issue by its key (e.g. NAT-42).

    Args:
        issue_key: The JIRA issue key (e.g. 'PROJ-123').

    Returns:
        Full issue details including description, comments, and metadata.
    """
    data = await _get(
        f"/issue/{issue_key}",
        params={"fields": "summary,description,status,assignee,reporter,priority,issuetype,comment,customfield_10020"},
    )
    fields = data.get("fields", {})

    # Extract description text
    desc = fields.get("description")
    desc_text = ""
    if desc and isinstance(desc, dict):
        for block in desc.get("content", []):
            for inline in block.get("content", []):
                if inline.get("type") == "text":
                    desc_text += inline.get("text", "")
            desc_text += "\n"
    desc_text = desc_text.strip() or "(no description)"

    # Extract recent comments
    comments = fields.get("comment", {}).get("comments", [])[-3:]
    comment_text = ""
    for c in comments:
        author = c.get("author", {}).get("displayName", "Unknown")
        body_content = c.get("body", {}).get("content", [])
        text = ""
        for block in body_content:
            for inline in block.get("content", []):
                if inline.get("type") == "text":
                    text += inline.get("text", "")
        comment_text += f"\n  [{author}]: {text.strip()}"

    return (
        f"{_format_issue(data)}"
        f"\nDescription:\n{desc_text}"
        + (f"\n\nRecent comments:{comment_text}" if comment_text else "")
    )


@mcp.tool()
async def list_my_open_issues(project_key: Optional[str] = None) -> str:
    """List all open JIRA issues assigned to the current user.

    Args:
        project_key: Optional project key to filter by (e.g. 'NAT'). If omitted, searches all projects.

    Returns:
        List of open issues assigned to me, grouped by status.
    """
    jql = "assignee = currentUser() AND status != Done ORDER BY priority DESC, updated DESC"
    if project_key:
        jql = f"project = {project_key} AND {jql}"
    return await search_issues(jql=jql, max_results=30)


@mcp.tool()
async def get_active_sprint(project_key: str) -> str:
    """Get all issues in the active sprint for a given project.

    Args:
        project_key: The JIRA project key (e.g. 'NAT', 'DEMO').

    Returns:
        All issues in the current active sprint, with status and assignee.
    """
    jql = f"project = {project_key} AND sprint in openSprints() ORDER BY status ASC, priority DESC"
    return await search_issues(jql=jql, max_results=50)


@mcp.tool()
async def list_boards(
    project_key: Optional[str] = None,
    board_type: Optional[str] = None,
    max_results: int = 20,
) -> str:
    """List Jira boards, optionally filtered by project and board type.

    Args:
        project_key: Optional project key to filter boards (e.g. 'NAT').
        board_type: Optional board type: 'scrum', 'kanban', or 'simple'.
        max_results: Maximum number of boards to return. Defaults to 20.

    Returns:
        A formatted list of matching boards with ids and project context.
    """
    params: dict[str, str] = {"maxResults": str(min(max_results, 50))}
    if project_key:
        params["projectKeyOrId"] = project_key
    if board_type:
        params["type"] = board_type

    data = await _get("/board", params=params, agile=True)
    boards = data.get("values", [])

    if not boards:
        filter_desc = f" for project {project_key}" if project_key else ""
        return f"No boards found{filter_desc}."

    lines = [f"Found {len(boards)} board(s):\n"]
    for board in boards:
        location = board.get("location") or {}
        project_name = location.get("projectName", "Unknown project")
        project_key_value = location.get("projectKey", "")
        lines.append(
            f"[{board.get('id')}] {board.get('name', '(unnamed)')} "
            f"(type={board.get('type', 'unknown')}, project={project_name}"
            + (f" [{project_key_value}]" if project_key_value else "")
            + ")"
        )
    return "\n".join(lines)


@mcp.tool()
async def create_sprint(
    board_id: int,
    name: str,
    goal: str = "",
    start_date: str = "",
    end_date: str = "",
) -> str:
    """Create a Jira sprint on a specific board.

    Args:
        board_id: The numeric Jira board id. Use `list_boards` to discover it.
        name: Sprint name.
        goal: Optional sprint goal.
        start_date: Optional ISO-8601 start datetime, e.g. '2026-04-22T09:00:00.000Z'.
        end_date: Optional ISO-8601 end datetime, e.g. '2026-05-06T09:00:00.000Z'.

    Returns:
        Created sprint id, name, and current state.
    """
    payload: dict[str, object] = {
        "name": name,
        "originBoardId": board_id,
    }
    if goal:
        payload["goal"] = goal
    if start_date:
        payload["startDate"] = start_date
    if end_date:
        payload["endDate"] = end_date

    data = await _post("/sprint", payload, agile=True)
    return (
        f"Created sprint [{data.get('id')}] {data.get('name', name)} "
        f"on board {board_id} with state={data.get('state', 'future')}."
    )


@mcp.tool()
async def create_issue(
    project_key: str,
    summary: str,
    description: str = "",
    issue_type: str = "Task",
    priority: str = "Medium",
) -> str:
    """Create a new JIRA issue.

    Args:
        project_key: The JIRA project key (e.g. 'NAT').
        summary: Issue title/summary.
        description: Detailed description of the issue.
        issue_type: Type of issue — 'Task', 'Bug', 'Story', or 'Epic'. Defaults to 'Task'.
        priority: Priority — 'Highest', 'High', 'Medium', 'Low', 'Lowest'. Defaults to 'Medium'.

    Returns:
        The created issue key and URL.
    """
    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "issuetype": {"name": issue_type},
            "priority": {"name": priority},
        }
    }
    if description:
        payload["fields"]["description"] = {
            "type": "doc",
            "version": 1,
            "content": [{"type": "paragraph", "content": [{"type": "text", "text": description}]}],
        }

    data = await _post("/issue", payload)
    key = data.get("key", "")
    site = os.environ.get("JIRA_SITE", "")
    return f"Created {key}: https://{site}/browse/{key}"


@mcp.tool()
async def add_comment(issue_key: str, comment: str) -> str:
    """Add a comment to an existing JIRA issue.

    Args:
        issue_key: The JIRA issue key (e.g. 'NAT-42').
        comment: The comment text to add.

    Returns:
        Confirmation that the comment was added.
    """
    payload = {
        "body": {
            "type": "doc",
            "version": 1,
            "content": [{"type": "paragraph", "content": [{"type": "text", "text": comment}]}],
        }
    }
    await _post(f"/issue/{issue_key}/comment", payload)
    return f"Comment added to {issue_key}."


@mcp.tool()
async def get_sprint_summary(project_key: str) -> str:
    """Get a high-level summary of the active sprint: total issues, done vs in-progress vs todo.

    Args:
        project_key: The JIRA project key (e.g. 'NAT').

    Returns:
        Sprint health summary with counts by status and a list of blockers/high-priority items.
    """
    data = await _get("/search/jql", params={
        "jql": f"project = {project_key} AND sprint in openSprints()",
        "maxResults": "100",
        "fields": "summary,status,assignee,priority,issuetype",
    })
    issues = data.get("issues", [])
    if not issues:
        return f"No active sprint found for project {project_key}."

    by_status: dict[str, list] = {}
    blockers = []
    for issue in issues:
        fields = issue.get("fields", {})
        status = fields.get("status", {}).get("name", "Unknown")
        by_status.setdefault(status, []).append(issue)
        if fields.get("priority", {}).get("name") in ("Highest", "High"):
            blockers.append(issue)

    lines = [f"Sprint summary for {project_key} ({len(issues)} total issues):\n"]
    for status, items in sorted(by_status.items()):
        lines.append(f"  {status}: {len(items)}")

    if blockers:
        lines.append(f"\nHigh/Highest priority ({len(blockers)}):")
        for issue in blockers[:5]:
            fields = issue.get("fields", {})
            assignee = (fields.get("assignee") or {}).get("displayName", "Unassigned")
            lines.append(f"  [{issue['key']}] {fields.get('summary', '')} — {assignee}")

    return "\n".join(lines)


if __name__ == "__main__":
    logger.info("Starting JIRA MCP server on port %d", _port)
    mcp.run(transport="streamable-http")
