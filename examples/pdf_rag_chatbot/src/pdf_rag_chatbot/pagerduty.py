# SPDX-FileCopyrightText: Copyright (c) 2025, Your Organization. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PagerDuty API client module.
Provides functionality for interacting with the PagerDuty API and managing incidents.
"""

import logging
import re
from typing import Dict, Any, Optional
import os
from pathlib import Path
from dotenv import load_dotenv

import httpx
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

class PagerDutyClientConfig(FunctionBaseConfig, name="pagerduty_client"):
    """Configuration for PagerDuty API client."""
    api_token: Optional[str] = Field(default=None)
    api_version: str = Field(default="2")
    timeout: int = Field(default=30)

@register_function(config_type=PagerDutyClientConfig)
async def pagerduty_client(config: PagerDutyClientConfig, builder: Builder):
    """Register the PagerDuty API client function."""

    async def fetch_incident(incident_url: str) -> Dict[str, Any]:
        try:
            # Extract incident ID from the URL
            match = re.search(r'/incidents?/([a-zA-Z0-9]+)', incident_url, re.IGNORECASE)
            if not match:
                return {
                    "status": "error",
                    "error": "Invalid PagerDuty incident URL format"
                }

            incident_id = match.group(1)
            api_token = config.api_token or os.getenv('PAGERDUTY_API_TOKEN')
            if not api_token:
                return {
                    "status": "error",
                    "error": "PagerDuty API token is required"
                }

            # Headers
            headers = {
                "Authorization": f"Token token={api_token}",
                "Accept": f"application/vnd.pagerduty+json;version={config.api_version}"
            }

            # Base URL - fallback to default if not found
            base_url = os.getenv('PAGERDUTY_API_URL') or os.getenv('PAGERDUTY_INSTANCE') or 'https://api.pagerduty.com'
            api_url = f"{base_url}/incidents/{incident_id}"

            logger.debug(f"Final API URL: {api_url}")
            logger.debug(f"Headers: {headers}")

            # HTTP Request
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                response = await client.get(api_url, headers=headers)

                logger.debug(f"API Response Status: {response.status_code}")
                logger.debug(f"API Response Text: {response.text}")

                if response.status_code == 200:
                    incident = response.json().get("incident", {})
                    details = {
                        "id": incident.get("id"),
                        "title": incident.get("title"),
                        "status": incident.get("status"),
                        "urgency": incident.get("urgency"),
                        "created_at": incident.get("created_at"),
                        "last_status_change": incident.get("last_status_change_at"),
                        "service": incident.get("service", {}).get("summary"),
                        "description": incident.get("description", "No description available"),
                        "assigned_to": [a["assignee"]["summary"] for a in incident.get("assignments", [])]
                    }

                    return {
                        "status": "success",
                        "incident_id": incident_id,
                        "details": details
                    }

                elif response.status_code == 401:
                    return {"status": "error", "error": "Unauthorized: Invalid PagerDuty API token."}
                elif response.status_code == 403:
                    return {"status": "error", "error": "Forbidden: Check PagerDuty API token permissions."}
                elif response.status_code == 404:
                    return {"status": "error", "error": f"Incident not found: {incident_id}"}
                else:
                    return {
                        "status": "error",
                        "error": f"Unexpected error {response.status_code}: {response.text}"
                    }

        except httpx.TimeoutException:
            logger.error("Timeout accessing PagerDuty API")
            return {
                "status": "error",
                "error": f"Timeout: Could not access PagerDuty API within {config.timeout} seconds"
            }

        except Exception as e:
            logger.exception(f"Error fetching PagerDuty incident: {e}")
            return {
                "status": "error",
                "error": f"Unexpected error: {str(e)}"
            }

    yield FunctionInfo.from_fn(
        fetch_incident,
        description="Fetches incident information from PagerDuty URLs. Requires API token for authentication."
    )