# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Seed script: creates a JIRA demo project with realistic sprint data for NAT agent demos.
#
# Usage (from examples/a365_example):
#   uv run python scripts/seed_jira_demo.py
#
# Requires: JIRA_EMAIL, JIRA_API_TOKEN, JIRA_SITE in environment or .env file.
# What it creates:
#   - Project: "NAT Demo" (key: NATDEMO)
#   - 1 active sprint with ~12 issues across Epic/Story/Task/Bug
#   - Mix of statuses, priorities, and assignees for a realistic standup demo

import base64
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

EMAIL = os.environ.get("JIRA_EMAIL") or os.environ["ATLASSIAN_EMAIL"]
TOKEN = os.environ.get("JIRA_API_TOKEN") or os.environ["ATLASSIAN_API_TOKEN"]
SITE = os.environ.get("JIRA_SITE") or os.environ["ATLASSIAN_SITE"]

BASE = f"https://{SITE}/rest/api/3"
AGILE_BASE = f"https://{SITE}/rest/agile/1.0"
AUTH = base64.b64encode(f"{EMAIL}:{TOKEN}".encode()).decode()
HEADERS = {
    "Authorization": f"Basic {AUTH}",
    "Accept": "application/json",
    "Content-Type": "application/json",
}

PROJECT_KEY = "KAN"
PROJECT_NAME = "NAT Demo"


def api(method, path, base=BASE, **kwargs):
    resp = requests.request(method, f"{base}{path}", headers=HEADERS, **kwargs)
    if resp.status_code >= 400:
        print(f"ERROR {resp.status_code} {method} {path}: {resp.text[:300]}")
        resp.raise_for_status()
    return resp.json() if resp.content else {}


def doc(text):
    """Atlassian Document Format wrapper for plain text."""
    return {
        "type": "doc",
        "version": 1,
        "content": [{"type": "paragraph", "content": [{"type": "text", "text": text}]}],
    }


def get_or_create_project():
    try:
        data = api("GET", f"/project/{PROJECT_KEY}")
        print(f"Project {PROJECT_KEY} already exists.")
        return data["id"], data["key"]
    except requests.HTTPError:
        pass

    print(f"Creating project {PROJECT_NAME}...")
    # Get current user account ID
    me = api("GET", "/myself")
    account_id = me["accountId"]

    data = api("POST", "/project", json={
        "key": PROJECT_KEY,
        "name": PROJECT_NAME,
        "projectTypeKey": "software",
        "projectTemplateKey": "com.pyxis.greenhopper.jira:gh-simplified-scrum-agility",
        "description": "Demo project for NAT agent toolkit demos",
        "leadAccountId": account_id,
        "assigneeType": "PROJECT_LEAD",
    })
    print(f"Created project: {data['key']} (id={data['id']})")
    return data["id"], data["key"]


def get_issue_types():
    types = api("GET", f"/project/{PROJECT_KEY}/statuses")
    # Simpler: fetch from issue type endpoint
    all_types = api("GET", "/issuetype")
    return {t["name"]: t["id"] for t in all_types if not t.get("subtask")}


def create_issue(summary, issue_type, priority, description, assignee_id=None):
    payload = {
        "fields": {
            "project": {"key": PROJECT_KEY},
            "summary": summary,
            "issuetype": {"name": issue_type},
            "priority": {"name": priority},
            "description": doc(description),
        }
    }
    if assignee_id:
        payload["fields"]["assignee"] = {"accountId": assignee_id}
    data = api("POST", "/issue", json=payload)
    print(f"  Created {data['key']}: {summary}")
    time.sleep(0.3)  # avoid rate limit
    return data["key"]


def get_board_id():
    data = api("GET", f"/board?projectKeyOrId={PROJECT_KEY}", base=AGILE_BASE)
    values = data.get("values", [])
    if not values:
        print("No board found — sprint creation skipped.")
        return None
    return values[0]["id"]


def create_sprint(board_id, name):
    try:
        data = api("POST", "/sprint", base=AGILE_BASE, json={
            "name": name,
            "originBoardId": board_id,
            "goal": "Ship NAT A365 integration demo + JIRA MCP tooling",
        })
        print(f"Created sprint: {data['name']} (id={data['id']})")
        return data["id"]
    except requests.HTTPError as e:
        print(f"Could not create sprint (may need board admin): {e}")
        return None


def start_sprint(sprint_id):
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=14)
    try:
        api("POST", f"/sprint/{sprint_id}", base=AGILE_BASE, json={
            "state": "active",
            "startDate": now.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "endDate": end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        })
        print(f"Sprint {sprint_id} started.")
    except requests.HTTPError as e:
        print(f"Could not start sprint: {e}")


def add_issues_to_sprint(sprint_id, issue_keys):
    try:
        api("POST", f"/sprint/{sprint_id}/issue", base=AGILE_BASE, json={"issues": issue_keys})
        print(f"Added {len(issue_keys)} issues to sprint.")
    except requests.HTTPError as e:
        print(f"Could not add issues to sprint: {e}")


def transition_issue(issue_key, status_name):
    """Move issue to a status by name."""
    transitions = api("GET", f"/issue/{issue_key}/transitions")
    for t in transitions.get("transitions", []):
        if t["name"].lower() == status_name.lower() or t["to"]["name"].lower() == status_name.lower():
            api("POST", f"/issue/{issue_key}/transitions", json={"transition": {"id": t["id"]}})
            print(f"  Transitioned {issue_key} → {t['to']['name']}")
            time.sleep(0.2)
            return
    print(f"  Could not find transition to '{status_name}' for {issue_key}")


def add_comment(issue_key, text):
    api("POST", f"/issue/{issue_key}/comment", json={"body": doc(text)})


def main():
    print(f"\n=== Seeding JIRA demo data on {SITE} ===\n")

    # 1. Project
    get_or_create_project()

    # 2. Get my account ID for assignee
    me = api("GET", "/myself")
    my_id = me["accountId"]

    # 3. Create issues
    print("\nCreating issues...")
    issues = {}

    # Epics
    issues["epic_agent"] = create_issue(
        "NAT Agent A365 Integration",
        "Epic", "High",
        "End-to-end integration of NVIDIA Agent Toolkit with Microsoft Agent 365 platform, "
        "including Teams front-end, telemetry, and MCP tooling discovery.",
        assignee_id=my_id,
    )
    issues["epic_mcp"] = create_issue(
        "MCP Server Ecosystem",
        "Epic", "High",
        "Build and deploy MCP servers for enterprise data sources: Graph Mail, JIRA, GitHub, SharePoint.",
        assignee_id=my_id,
    )

    # Stories / Tasks — in progress
    issues["story_telemetry"] = create_issue(
        "Implement A365 telemetry exporter with token refresh",
        "Story", "High",
        "Add OTel exporter that ships traces to A365 observability endpoint. "
        "Needs token caching, refresh on 401, and cluster_category support.",
        assignee_id=my_id,
    )
    issues["task_mcp_client"] = create_issue(
        "Add streamable-http transport to NAT MCP client",
        "Task", "High",
        "Extend MCP client plugin to support stateless streamable-http transport "
        "to fix 307-redirect and session-terminated issues with Azure Container Apps.",
        assignee_id=my_id,
    )
    issues["task_jira_mcp"] = create_issue(
        "Build JIRA MCP server for sprint standup agent",
        "Task", "Medium",
        "FastMCP server wrapping Atlassian REST API v3. Tools: search_issues, get_issue, "
        "get_active_sprint, get_sprint_summary, create_issue, add_comment.",
        assignee_id=my_id,
    )
    issues["task_graph_mail"] = create_issue(
        "Deploy Graph Mail MCP server to Azure Container Apps",
        "Task", "Medium",
        "Dockerize and push graph_mail MCP server. Configure ACA ingress, health probe, "
        "and GRAPH_MAIL_TOKEN secret via Key Vault reference.",
        assignee_id=my_id,
    )

    # Stories — todo
    issues["story_github_mcp"] = create_issue(
        "Integrate GitHub MCP server for PR review agent",
        "Story", "Medium",
        "Wire up GitHub MCP server (github/github-mcp-server) as a function_group in NAT config. "
        "Enable PR summary, issue search, and repo context tools.",
    )
    issues["task_seed_script"] = create_issue(
        "Write JIRA demo seed script for standup demo",
        "Task", "Low",
        "Script that creates project NATDEMO with realistic sprint data so the agent demo "
        "has something interesting to query on day one.",
        assignee_id=my_id,
    )
    issues["story_cron"] = create_issue(
        "Add cron front-end config for daily sprint digest",
        "Story", "Medium",
        "Configure NAT cron front-end to fire agent every morning at 9am, "
        "pull JIRA sprint summary + open PRs, and post digest to Teams channel.",
    )

    # Bugs
    issues["bug_401"] = create_issue(
        "MCP tooling gateway returns 401 on ENVIRONMENT=Production",
        "Bug", "Highest",
        "The A365 tooling gateway GET /mcpServers returns 401 when ENVIRONMENT=Production. "
        "Root cause: SDK omits x-ms-tenant-id header. Workaround: set tooling_gateway_tenant_id in config.",
        assignee_id=my_id,
    )
    issues["bug_redirect"] = create_issue(
        "MCP client follows 307 redirect and loses POST body",
        "Bug", "High",
        "aiohttp follows 307 but changes POST to GET, causing session-terminated errors "
        "with stateless streamable-http MCP servers on ACA. Fix: stateless_http=True in FastMCP.",
        assignee_id=my_id,
    )
    issues["task_smoke"] = create_issue(
        "Write comprehensive smoke test for all A365 configs",
        "Task", "Medium",
        "smoke_all.sh should cover: console-only, telemetry-only, telemetry+tooling, "
        "A365 front-end. Each scenario validates startup and optionally hits /generate.",
        assignee_id=my_id,
    )

    # 4. Board + sprint
    print("\nSetting up sprint...")
    board_id = get_board_id()
    sprint_id = None
    if board_id:
        sprint_id = create_sprint(board_id, "Sprint 1 — A365 + MCP Demo")
        if sprint_id:
            start_sprint(sprint_id)
            # Add most issues to sprint (leave epic and one story in backlog)
            sprint_issues = [
                issues["story_telemetry"],
                issues["task_mcp_client"],
                issues["task_jira_mcp"],
                issues["task_graph_mail"],
                issues["story_github_mcp"],
                issues["task_seed_script"],
                issues["bug_401"],
                issues["bug_redirect"],
                issues["task_smoke"],
            ]
            add_issues_to_sprint(sprint_id, sprint_issues)

    # 5. Transition some issues to simulate in-progress sprint
    print("\nSetting issue statuses...")
    for key in [issues["story_telemetry"], issues["task_mcp_client"]]:
        transition_issue(key, "In Progress")

    for key in [issues["task_seed_script"]]:
        transition_issue(key, "Done")

    # 6. Add some comments for realism
    print("\nAdding comments...")
    add_comment(issues["bug_401"], "Confirmed fix: adding tooling_gateway_tenant_id to config resolves this.")
    add_comment(issues["story_telemetry"], "Token refresh logic working in local test. Pushing for review.")
    add_comment(issues["bug_redirect"], "Fixed upstream in FastMCP server — stateless_http=True resolves 307 issue.")

    print(f"\n=== Done! ===")
    print(f"Project: https://{SITE}/jira/software/projects/{PROJECT_KEY}/boards")
    print(f"\nTry these agent prompts:")
    print(f'  "Give me a sprint summary for project NATDEMO"')
    print(f'  "What are my open issues?"')
    bug_401_key = issues["bug_401"]
    print(f'  "Get the details of {bug_401_key}"')
    print(f'  "What bugs are in the active sprint?"')


if __name__ == "__main__":
    main()
