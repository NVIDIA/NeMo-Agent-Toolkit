# SPDX-FileCopyrightText: Copyright (c) 2025, Your Organization. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PagerDuty API client module.
Provides functionality for interacting with the PagerDuty API and managing incidents.
"""

import logging
import re
from typing import Dict, Any, Optional, List
import os
from pathlib import Path
from datetime import datetime, timedelta
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

            # Base URL - fallback to NVIDIA PagerDuty instance
            base_url = os.getenv('PAGERDUTY_API_URL') or os.getenv('PAGERDUTY_INSTANCE') or 'https://api.nvidia.pagerduty.com'
            
            # Fetch incident details
            incident_url = f"{base_url}/incidents/{incident_id}"
            timeline_url = f"{base_url}/incidents/{incident_id}/log_entries"
            alerts_url = f"{base_url}/incidents/{incident_id}/alerts"

            logger.debug(f"Final API URL: {incident_url}")
            logger.debug(f"Timeline API URL: {timeline_url}")
            logger.debug(f"Alerts API URL: {alerts_url}")
            logger.debug(f"Headers: {headers}")

            async with httpx.AsyncClient(timeout=config.timeout) as client:
                # Fetch incident details
                incident_response = await client.get(incident_url, headers=headers)
                logger.debug(f"Incident API Response Status: {incident_response.status_code}")
                logger.debug(f"Incident API Response Text: {incident_response.text}")
                
                if incident_response.status_code != 200:
                    return {
                        "status": "error",
                        "error": f"Failed to fetch incident details: {incident_response.text}"
                    }

                response_json = incident_response.json()
                if not response_json or "incident" not in response_json:
                    return {
                        "status": "error",
                        "error": f"Invalid response format from PagerDuty API: {response_json}"
                    }

                incident = response_json.get("incident", {})
                if not incident:
                    return {
                        "status": "error",
                        "error": "No incident data found in the response"
                    }

                # Get service ID for fetching related incidents
                service_id = incident.get("service", {}).get("id")
                service_name = incident.get("service", {}).get("summary")
                
                # Fetch timeline
                timeline_response = await client.get(timeline_url, headers=headers)
                logger.debug(f"Timeline API Response Status: {timeline_response.status_code}")
                logger.debug(f"Timeline API Response Text: {timeline_response.text}")
                
                timeline_entries = []
                if timeline_response.status_code == 200:
                    timeline_data = timeline_response.json()
                    for entry in timeline_data.get("log_entries", []):
                        timeline_entries.append({
                            "type": entry.get("type"),
                            "created_at": entry.get("created_at"),
                            "agent": entry.get("agent", {}).get("summary"),
                            "channel": entry.get("channel", {}).get("type"),
                            "summary": entry.get("summary"),
                            "details": entry.get("details", {})
                        })

                # Fetch alerts
                alerts_response = await client.get(alerts_url, headers=headers)
                logger.debug(f"Alerts API Response Status: {alerts_response.status_code}")
                logger.debug(f"Alerts API Response Text: {alerts_response.text}")
                
                alerts = []
                if alerts_response.status_code == 200:
                    alerts_data = alerts_response.json()
                    for alert in alerts_data.get("alerts", []):
                        alerts.append({
                            "id": alert.get("id"),
                            "type": alert.get("type"),
                            "status": alert.get("status"),
                            "created_at": alert.get("created_at"),
                            "body": alert.get("body", {}),
                            "severity": alert.get("severity"),
                            "source": alert.get("source", {}).get("name"),
                            "summary": alert.get("summary"),
                            "details": alert.get("details", {})
                        })

                # Fetch past incidents for the same service
                past_incidents = []
                if service_id:
                    # Calculate date range (last 30 days)
                    end_date = datetime.utcnow()
                    start_date = end_date - timedelta(days=30)
                    
                    # Format dates for API
                    since = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
                    until = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
                    
                    # Fetch past incidents
                    past_incidents_url = f"{base_url}/incidents"
                    params = {
                        "service_ids[]": service_id,
                        "since": since,
                        "until": until,
                        "status": "resolved",
                        "limit": 5  # Limit to 5 most recent incidents
                    }
                    
                    past_incidents_response = await client.get(
                        past_incidents_url,
                        headers=headers,
                        params=params
                    )
                    
                    if past_incidents_response.status_code == 200:
                        past_data = past_incidents_response.json()
                        for past_inc in past_data.get("incidents", []):
                            if past_inc.get("id") != incident_id:  # Exclude current incident
                                past_incidents.append({
                                    "id": past_inc.get("id"),
                                    "title": past_inc.get("title"),
                                    "status": past_inc.get("status"),
                                    "created_at": past_inc.get("created_at"),
                                    "resolved_at": past_inc.get("resolved_at"),
                                    "urgency": past_inc.get("urgency"),
                                    "priority": past_inc.get("priority", {}).get("summary") if past_inc.get("priority") else None
                                })

                # Compile detailed incident information
                details = {
                    "id": incident.get("id"),
                    "title": incident.get("title"),
                    "status": incident.get("status"),
                    "urgency": incident.get("urgency"),
                    "priority": incident.get("priority", {}).get("summary") if incident.get("priority") else None,
                    "created_at": incident.get("created_at"),
                    "last_status_change": incident.get("last_status_change_at"),
                    "service": {
                        "id": service_id,
                        "name": service_name
                    },
                    "description": incident.get("description", "No description available"),
                    "assigned_to": [a["assignee"]["summary"] for a in incident.get("assignments", []) if a.get("assignee")],
                    "acknowledgers": [a["acknowledger"]["summary"] for a in incident.get("acknowledgements", []) if a.get("acknowledger")],
                    "last_status_change_by": incident.get("last_status_change_by", {}).get("summary") if incident.get("last_status_change_by") else None,
                    "incident_key": incident.get("incident_key"),
                    "timeline": timeline_entries,
                    "past_incidents": past_incidents,
                    "alerts": alerts
                }

                return {
                    "status": "success",
                    "incident_id": incident_id,
                    "details": details
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
        description="Fetches incident information, timeline, alerts, and past incidents from PagerDuty URLs. Requires API token for authentication."
    ) 