#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Get a delegated Microsoft Graph token with Mail.Read and Mail.ReadWrite scopes
# via MSAL device code flow (no browser required — just open a URL on any device).
#
# Usage:
#   uv run python scripts/get_graph_mail_token.py
#   # Follow the prompt, then copy the token into ACA secret and seed script.

from __future__ import annotations

import os
import sys

try:
    import msal
except ImportError:
    print("msal not installed. Run: uv add msal")
    sys.exit(1)

# Use the nvidiadev.onmicrosoft.com tenant
TENANT_ID = "06938c20-42d5-4112-9f91-643dff159d7f"

# Microsoft Graph Explorer public client (well-known, works for delegated flows)
# Or use your own AAD app registration that has Mail.Read/Write delegated permissions
CLIENT_ID = os.environ.get("GRAPH_APP_CLIENT_ID", "14d82eec-204b-4c2f-b7e8-296a70dab67e")  # Graph Explorer

SCOPES = [
    "https://graph.microsoft.com/Mail.Read",
    "https://graph.microsoft.com/Mail.ReadWrite",
    "https://graph.microsoft.com/Mail.Send",
    "https://graph.microsoft.com/User.Read",
]


def main() -> None:
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
    print("  1. Seed emails:  uv run python scripts/seed_top5_emails.py")
    print("  2. Update ACA:   az containerapp secret set -g a365-nat-dev-rg-1 -n graph-mail-mcp \\")
    print(f"                       --secrets \"graph-mail-token=$(cat {token_file})\"")
    print("                   az containerapp update -g a365-nat-dev-rg-1 -n graph-mail-mcp \\")
    print("                       --image a365natdevacr1.azurecr.io/graph-mail-mcp:latest")


if __name__ == "__main__":
    main()
