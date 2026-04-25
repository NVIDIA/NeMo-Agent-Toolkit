# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Graph Mail MCP Server — exposes Microsoft Graph API mail endpoints as MCP tools.
#
# Usage:
#   export GRAPH_MAIL_TOKEN="<delegated bearer token for https://graph.microsoft.com>"
#   uv run python server.py
#
# The server listens on http://0.0.0.0:8100 (streamable-http) at path /mcp/.
# Register it in NAT config as an mcp_client with url: http://host:8100/mcp/
#
# Best practices applied:
#   - stateless_http=True: avoids Starlette 307-redirect / session-terminated issues
#   - @mcp.custom_route for health endpoint: no Starlette wrapper needed
#   - mcp.run() for clean startup; uvicorn handles process lifecycle

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiohttp
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GRAPH_BASE = "https://graph.microsoft.com/v1.0"

_port = int(os.environ.get("GRAPH_MAIL_PORT", "8100"))

# Stateful mode (default): supports SSE-based session initialization required by
# the NAT MCP client (streamable_http_client). mcp.run() handles routing internally
# without a Starlette wrapper, avoiding the 307-redirect / Session terminated issue.
mcp = FastMCP("graph_mail", host="0.0.0.0", port=_port)


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """ACA liveness / readiness probe endpoint."""
    return JSONResponse({"status": "ok", "service": "graph-mail-mcp"})


def _auth_headers() -> dict[str, str]:
    token = os.environ.get("GRAPH_MAIL_TOKEN", "")
    if not token:
        raise RuntimeError("GRAPH_MAIL_TOKEN environment variable is not set")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


async def _graph_get(path: str, params: dict | None = None) -> dict:
    url = f"{GRAPH_BASE}{path}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=_auth_headers(), params=params or {}) as resp:
            body = await resp.json()
            if resp.status >= 400:
                raise RuntimeError(f"Graph API error {resp.status}: {body.get('error', body)}")
            return body


@mcp.tool()
async def search_emails(
    subject_contains: Optional[str] = None,
    sender_contains: Optional[str] = None,
    days_back: int = 7,
    max_results: int = 20,
) -> str:
    """Search emails in the mailbox by subject keyword, sender, and date range.

    Args:
        subject_contains: Filter emails whose subject contains this string (case-insensitive).
        sender_contains: Filter emails whose sender name or address contains this string.
        days_back: How many days back to search. Defaults to 7.
        max_results: Maximum number of emails to return. Defaults to 20.

    Returns:
        A formatted list of matching emails with sender, date, subject, and a short preview.
    """
    since = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    since = since - timedelta(days=days_back)
    since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")

    filters = [f"receivedDateTime ge {since_str}"]
    if subject_contains:
        filters.append(f"contains(subject, '{subject_contains}')")
    if sender_contains:
        filters.append(
            f"contains(from/emailAddress/name, '{sender_contains}') "
            f"or contains(from/emailAddress/address, '{sender_contains}')"
        )

    params = {
        "$filter": " and ".join(filters),
        "$select": "id,subject,from,receivedDateTime,bodyPreview",
        "$orderby": "receivedDateTime desc",
        "$top": str(min(max_results, 50)),
    }

    data = await _graph_get("/me/messages", params)
    emails = data.get("value", [])

    if not emails:
        return "No emails found matching the criteria."

    lines = [f"Found {len(emails)} email(s):\n"]
    for i, email in enumerate(emails, 1):
        sender = email.get("from", {}).get("emailAddress", {})
        lines.append(
            f"{i}. [{email.get('receivedDateTime', '')[:10]}] "
            f"From: {sender.get('name', '')} <{sender.get('address', '')}>\n"
            f"   Subject: {email.get('subject', '(no subject)')}\n"
            f"   ID: {email.get('id', '')}\n"
            f"   Preview: {email.get('bodyPreview', '')[:200]}\n"
        )
    return "\n".join(lines)


@mcp.tool()
async def get_email_content(email_id: str) -> str:
    """Get the full text content of a specific email by its ID.

    Args:
        email_id: The email ID returned by search_emails.

    Returns:
        The full email body as plain text, along with sender, recipients, and date.
    """
    data = await _graph_get(
        f"/me/messages/{email_id}",
        params={"$select": "subject,from,toRecipients,receivedDateTime,body"},
    )

    sender = data.get("from", {}).get("emailAddress", {})
    to_list = [
        r.get("emailAddress", {}).get("address", "")
        for r in data.get("toRecipients", [])
    ]
    body = data.get("body", {})
    content = body.get("content", "")
    if body.get("contentType") == "html":
        import re
        content = re.sub(r"<[^>]+>", " ", content)
        content = re.sub(r"\s+", " ", content).strip()

    return (
        f"Subject: {data.get('subject', '')}\n"
        f"From: {sender.get('name', '')} <{sender.get('address', '')}>\n"
        f"To: {', '.join(to_list)}\n"
        f"Date: {data.get('receivedDateTime', '')}\n"
        f"\n{content}"
    )


@mcp.tool()
async def list_recent_top5_emails(days_back: int = 14, max_results: int = 30) -> str:
    """List recent emails with 'Top 5' in the subject — useful for org-wide update digests.

    Args:
        days_back: How many days back to look. Defaults to 14.
        max_results: Maximum number of emails to return. Defaults to 30.

    Returns:
        A list of Top 5 emails with sender, date, and subject.
    """
    return await search_emails(
        subject_contains="Top 5",
        days_back=days_back,
        max_results=max_results,
    )


@mcp.tool()
async def answer_from_emails(question: str, days_back: int = 14) -> str:
    """Search recent Top 5 emails and return relevant excerpts to help answer a question.

    Fetches recent Top 5 emails and returns the content most likely to answer the question.
    The calling agent should synthesize the excerpts into a final answer.

    Args:
        question: The natural language question to answer (e.g. 'What are teams saying about NemoClaw?').
        days_back: How many days of emails to search. Defaults to 14.

    Returns:
        Relevant email excerpts that may help answer the question.
    """
    listing = await search_emails(subject_contains="Top 5", days_back=days_back, max_results=20)
    if "No emails found" in listing:
        return "No Top 5 emails found in the specified time range."

    import re
    ids = re.findall(r"ID: (.+)", listing)

    if not ids:
        return listing

    excerpts = []
    for email_id in ids[:10]:  # Cap at 10 emails to avoid token explosion
        try:
            content = await get_email_content(email_id.strip())
            if len(content) > 2000:
                content = content[:2000] + "\n... [truncated]"
            excerpts.append(content)
        except Exception as e:
            logger.warning("Failed to fetch email %s: %s", email_id, e)

    if not excerpts:
        return "Could not retrieve email content."

    separator = "\n" + "=" * 60 + "\n"
    return f"Relevant email content for: '{question}'\n\n" + separator.join(excerpts)


if __name__ == "__main__":
    logger.info("Starting Graph Mail MCP server on port %d", _port)
    mcp.run(transport="streamable-http")
