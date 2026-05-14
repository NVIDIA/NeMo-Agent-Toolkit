#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Get a delegated Microsoft Graph token via MSAL device code flow for the
# graph_mail MCP example.
#
# Usage:
#   uv run python scripts/get_graph_mail_token.py
# Environment:
#   GRAPH_TENANT_ID      Optional. Defaults to AZURE_TENANT_ID if set.
#   GRAPH_APP_CLIENT_ID  Optional. Defaults to the public Graph Explorer client id.

from __future__ import annotations

import os
import sys

try:
    import msal
except ImportError:
    print("msal not installed. Run: uv add msal")
    sys.exit(1)

TENANT_ID = os.environ.get("GRAPH_TENANT_ID") or os.environ.get("AZURE_TENANT_ID")

# Microsoft Graph Explorer public client (well-known, works for delegated flows)
# Or use your own app registration that has the required delegated permissions.
CLIENT_ID = os.environ.get("GRAPH_APP_CLIENT_ID", "14d82eec-204b-4c2f-b7e8-296a70dab67e")

SCOPES = [
    "https://graph.microsoft.com/Mail.Read",
    "https://graph.microsoft.com/Mail.ReadWrite",
    "https://graph.microsoft.com/Mail.Send",
    "https://graph.microsoft.com/User.Read",
]


def main() -> None:
    if not TENANT_ID:
        print("Set GRAPH_TENANT_ID or AZURE_TENANT_ID before requesting a delegated Graph token.")
        sys.exit(1)

    authority = f"https://login.microsoftonline.com/{TENANT_ID}"
    app = msal.PublicClientApplication(CLIENT_ID, authority=authority)

    # Try silent first (cached)
    accounts = app.get_accounts()
    result = None
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])

    if not result:
        flow = app.initiate_device_flow(scopes=SCOPES)
        if "user_code" not in flow:
            print("Failed to create device flow:", flow)
            sys.exit(1)

        print("\n" + "=" * 60)
        print(flow["message"])
        print("=" * 60 + "\n")
        result = app.acquire_token_by_device_flow(flow)

    if "access_token" not in result:
        print("Failed to get token:", result.get("error_description", result))
        sys.exit(1)

    token = result["access_token"]

    # Save token to file so other scripts can read it without copy-paste
    token_file = "/tmp/graph_mail_token.txt"
    with open(token_file, "w") as f:
        f.write(token)

    print("\nToken acquired successfully!")
    print(f"Expires in: {result.get('expires_in', '?')} seconds")
    print(f"Token saved to: {token_file}")
    print("\nNext steps:")
    print("  1. Export it for local use:")
    print(f"     export GRAPH_MAIL_TOKEN=\"$(cat {token_file})\"")
    print("  2. Or inject it into the graph_mail MCP service environment used by your deployment.")


if __name__ == "__main__":
    main()
