# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Example service token functions for demonstration purposes.

In production environments, these functions would:
- Fetch tokens from secure vaults (e.g., HashiCorp Vault, AWS Secrets Manager)
- Use AIQContext to access request metadata
- Implement token caching and refresh logic
- Handle errors and retries appropriately
"""

import os


async def get_jira_service_token(**kwargs) -> tuple[str, str]:
    """
    Example function that returns service token header name and value.

    This simple example reads from environment variables. In production,
    you would fetch from a secure vault or token service.

    Configuration example:
        ```yaml
        service_token:
          function: "nat_service_account_auth_mcp.scripts.service_tokens.get_jira_service_token"
          kwargs:
            vault_path: "secrets/jira"  # Optional custom parameters
        ```

    Args:
        **kwargs: Optional additional arguments from config (such as vault_path, region)

    Returns:
        tuple[str, str]: (header_name, token_value)

    Raises:
        ValueError: If JIRA_SERVICE_TOKEN environment variable is not set

    Example production implementation:
        ```python
        from nat.builder.context import AIQContext

        async def get_jira_service_token(vault_path: str = "secrets/jira", **kwargs):
            # Access runtime context if needed
            context = AIQContext.get()

            # Fetch from secure vault
            token = await fetch_from_vault(vault_path)
            header = os.getenv("SERVICE_TOKEN_HEADER")

            return (header, token)
        ```
    """
    # Read header name from environment (with default)
    header = os.getenv("SERVICE_TOKEN_HEADER")

    if not header:
        raise ValueError("SERVICE_TOKEN_HEADER environment variable not set. "
                         "In production, this would be set to the header name used by the service.")

    # Read token from environment
    token = os.getenv("JIRA_SERVICE_TOKEN")

    if not token:
        raise ValueError("JIRA_SERVICE_TOKEN environment variable not set. "
                         "In production, this would fetch from a secure vault.")

    return (header, token)
