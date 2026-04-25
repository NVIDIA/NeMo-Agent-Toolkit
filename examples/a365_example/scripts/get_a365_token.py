# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Obtain an Azure AD access token via client credentials (for A365 telemetry, MCP, etc.).
#
# Environment:
#   AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET — required
#   A365_TOKEN_SCOPE — OAuth scope (see below)
#   A365_FMI_PATH — optional; agent **identity** app (client) ID for blueprint token requests
#                   (Microsoft: fmi_path on the blueprint client_credentials call). See Learn:
#   https://learn.microsoft.com/en-us/entra/agent-id/identity-platform/autonomous-agent-request-tokens
#
# Scopes to try for traces 401 (in order — use what matches your tenant / agent setup):
#   1. api://AzureADTokenExchange/.default  — agent identity blueprint (typical for A365 traces)
#   2. https://api.powerplatform.com/.default — Dataverse / Power Platform API
#   3. https://graph.microsoft.com/.default — Graph only; often wrong audience for traces
#
# Agent 365 MCP tooling (catalog servers on agent365.svc.cloud.microsoft) often uses:
#   ea9ffc3e-8a23-4a7d-836d-234d7c7565c1/.default
# See docs/A365-MCP-NATIVE-SETUP.md and a365 develop list-available.
#
# Usage:
#   export A365_TOKEN_SCOPE=api://AzureADTokenExchange/.default
#   export A365_FMI_PATH=<agent-identity-client-id>   # if required by your blueprint
#   export A365_BEARER_TOKEN=$(uv run python scripts/get_a365_token.py)
#
#   uv run python scripts/get_a365_token.py --decode   # prints JWT payload to stderr, token to stdout

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def _decode_jwt_payload(token: str) -> dict:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Token does not look like a JWT (expected three segments).")
    payload_b64 = parts[1]
    pad = 4 - len(payload_b64) % 4
    if pad != 4:
        payload_b64 += "=" * pad
    raw = base64.urlsafe_b64decode(payload_b64.encode("ascii"))
    return json.loads(raw.decode("utf-8"))


def _request_token(
    tenant: str,
    client_id: str,
    client_secret: str,
    scope: str,
    fmi_path: str | None,
) -> str:
    url = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
    parts = [
        "grant_type=client_credentials",
        f"client_id={urllib.parse.quote(client_id, safe='')}",
        f"client_secret={urllib.parse.quote(client_secret, safe='')}",
        f"scope={urllib.parse.quote(scope, safe='')}",
    ]
    if fmi_path:
        parts.append(f"fmi_path={urllib.parse.quote(fmi_path, safe='')}")
    body = "&".join(parts).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
        token = data.get("access_token")
        if not token:
            raise RuntimeError("Response missing access_token")
        return token


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Client-credentials access token for Azure AD (A365 / smoke tests)."
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Print JWT payload (aud, scp, exp, …) to stderr after a successful token request.",
    )
    args = parser.parse_args()

    tenant = os.environ.get("AZURE_TENANT_ID")
    client_id = os.environ.get("AZURE_CLIENT_ID")
    client_secret = os.environ.get("AZURE_CLIENT_SECRET")
    scope = os.environ.get("A365_TOKEN_SCOPE", "https://graph.microsoft.com/.default")
    fmi_path = os.environ.get("A365_FMI_PATH") or os.environ.get("AGENT_IDENTITY_CLIENT_ID")

    if not tenant or not client_id or not client_secret:
        print(
            "Set AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET (e.g. in .env).",
            file=sys.stderr,
        )
        return 1

    if scope.endswith("graph.microsoft.com/.default"):
        print(
            "Note: Default scope is Graph. For Agent 365 traces, try:\n"
            "  export A365_TOKEN_SCOPE=api://AzureADTokenExchange/.default\n"
            "  Optional: export A365_FMI_PATH=<agent-identity-client-id>\n"
            "See ../../../docs/A365-TROUBLESHOOTING-401.md (repo root).",
            file=sys.stderr,
        )

    try:
        token = _request_token(tenant, client_id, client_secret, scope, fmi_path)
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        print(f"Token request failed: {e.code} {body}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.decode:
        try:
            claims = _decode_jwt_payload(token)
            print(json.dumps(claims, indent=2), file=sys.stderr)
        except Exception as e:
            print(f"Could not decode JWT: {e}", file=sys.stderr)

    print(token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
