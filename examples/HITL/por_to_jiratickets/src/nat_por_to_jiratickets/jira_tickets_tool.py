# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import json
import logging
import os
import re

import httpx
import requests

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


def get_epics_tool(root_path: str) -> str:
    """
    Return a string listing the epics from the last extraction.
    """
    filename = root_path + "epics_tasks.json"
    try:
        with open(filename, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            logger.debug("Data successfully loaded from %s", filename)
    except Exception as e:
        logger.exception("An error occurred while loading the file: %s", e)
        return None

    return data


class JiraTool:

    def __init__(self, domain: str, project_key: str, ticket_type: str):
        self.domain = domain
        self.userid = os.getenv("JIRA_USERID")
        self.token = os.getenv("JIRA_TOKEN")
        self.ticket_type = ticket_type
        self.project_key = project_key
        self.url = f"{self.domain}/rest/api/2/issue"

    async def get_priority_name(self, priority: str):
        if priority == 'P0':
            return priority + " - Must have"
        if priority == 'P1':
            return priority + " - Should have"
        if priority == 'P2':
            return priority + " - Nice to have"

    async def create_epic(self, client: httpx.AsyncClient, ticket_data: dict) -> str:
        """
        Creates a Jira Epic and returns the epic key (e.g. "PROJ-123").
        """
        title = ticket_data.get("name", "Untitled")
        epic_description = ticket_data.get("description", "")
        logger.debug("Creating Epic in Jira: %s", title)
        payload = {
            "fields": {
                "project": {
                    "key": self.project_key
                },
                "summary": title,
                "description": epic_description,
                "issuetype": {
                    "name": "Epic"
                },
                "customfield_10006": title
            }
        }
        try:
            r = await client.post(
                self.url,
                json=payload,
                auth=(self.userid, self.token),
                headers={"Content-Type": "application/json"},
            )

            r.raise_for_status()  # Raise error for 4xx/5xx
        except httpx.HTTPStatusError as err:
            return {"error": f"HTTP error: {err.response.status_code}", "details": err.response.text}
        except httpx.RequestError as err:
            return {
                "error": "Request error",
                "message": str(err),
                "request_url": str(err.request.url) if err.request else "N/A"
            }

        data = r.json()
        return data["key"], data["self"]

    async def create_task(self, client: httpx.AsyncClient, ticket_data: dict):
        """
        Creates a Task Type with assigned priority and story points.
        """
        title = ticket_data.get("title", "Untitled Story")
        description = ticket_data.get("description", "")
        priority = ticket_data.get("priority", "")
        story_points = ticket_data.get("storypoints", "")
        logger.debug("Creating Tasks in Jira: %s for priority %s with story point %s", title, priority, story_points)
        priority_name = await self.get_priority_name(priority)
        payload = {
            "fields": {
                "project": {
                    "key": self.project_key
                },
                "summary": title,
                "description": description,
                "issuetype": {
                    "name": "Task"
                },
                "priority": {
                    "name": priority_name
                }
            }
        }
        try:
            r = await client.post(
                self.url,
                json=payload,
                auth=(self.userid, self.token),
                headers={"Content-Type": "application/json"},
            )

            r.raise_for_status()  # Raise error for 4xx/5xx
        except httpx.HTTPStatusError as err:
            return {"error": f"HTTP error: {err.response.status_code}", "details": err.response.text}
        except httpx.RequestError as err:
            return {
                "error": "Request error",
                "message": str(err),
                "request_url": str(err.request.url) if err.request else "N/A"
            }
        data = r.json()
        return data["key"], data["self"]

    async def create_bug(self, client: httpx.AsyncClient, ticket_data: dict):
        """
        Creates a Bug Type with assigned priority and story points.
        """
        title = ticket_data.get("title", "Untitled Story")
        description = ticket_data.get("description", "")
        priority = ticket_data.get("priority", "")
        story_points = ticket_data.get("storypoints", "")
        logger.debug("Creating Tasks in Jira: %s for priority %s with story point %s", title, priority, story_points)
        priority_name = await self.get_priority_name(priority)
        payload = {
            "fields": {
                "project": {
                    "key": self.project_key
                },
                "summary": title,
                "description": description,
                "issuetype": {
                    "name": "Bug"
                },
                "priority": {
                    "name": priority_name
                },
                "customfield_10002":
                    int(story_points)  # Update with the desired story points
            }
        }
        try:
            r = await client.post(
                self.url,
                json=payload,
                auth=(self.userid, self.token),
                headers={"Content-Type": "application/json"},
            )

            r.raise_for_status()  # Raise error for 4xx/5xx
        except httpx.HTTPStatusError as err:
            return {"error": f"HTTP error: {err.response.status_code}", "details": err.response.text}
        except httpx.RequestError as err:
            return {
                "error": "Request error",
                "message": str(err),
                "request_url": str(err.request.url) if err.request else "N/A"
            }
        data = r.json()
        return data["key"], data["self"]

    async def create_feature(self, client: httpx.AsyncClient, ticket_data: dict):
        """
        Creates a Feature Type with assigned priority and story points.
        """
        title = ticket_data.get("title", "Untitled Story")
        description = ticket_data.get("description", "")
        priority = ticket_data.get("priority", "")
        story_points = ticket_data.get("storypoints", "")
        logger.debug("Creating Tasks in Jira: %s for priority %s with story point %s", title, priority, story_points)
        priority_name = await self.get_priority_name(priority)
        payload = {
            "fields": {
                "project": {
                    "key": self.project_key
                },
                "summary": title,
                "description": description,
                "issuetype": {
                    "name": "New Feature"
                },
                "priority": {
                    "name": priority_name
                },
                "customfield_10002":
                    int(story_points)  # Update with the desired story points
            }
        }
        try:
            r = await client.post(
                self.url,
                json=payload,
                auth=(self.userid, self.token),
                headers={"Content-Type": "application/json"},
            )

            r.raise_for_status()  # Raise error for 4xx/5xx
        except httpx.HTTPStatusError as err:
            return {"error": f"HTTP error: {err.response.status_code}", "details": err.response.text}
        except httpx.RequestError as err:
            return {
                "error": "Request error",
                "message": str(err),
                "request_url": str(err.request.url) if err.request else "N/A"
            }
        data = r.json()
        return data["key"], data["self"]


def process_input_text(input_text):
    input_text = input_text.replace('Observ', "")
    input_text = re.sub(r"\s+", "", input_text)
    return input_text


class CreateJiraToolConfig(FunctionBaseConfig, name="create_jira_tickets_tool"):
    root_path: str
    jira_domain: str
    jira_project_key: str
    timeout: float
    connect: float
    hitl_approval_fn: FunctionRef


@register_function(config_type=CreateJiraToolConfig)
async def create_jira_tickets_tool(config: CreateJiraToolConfig, builder: Builder):

    hitl_approval_fn = builder.get_function(config.hitl_approval_fn)

    async def _arun(input_text: str) -> str:

        # Get user confirmation first
        try:
            selected_option = await hitl_approval_fn.acall_invoke()

            if not selected_option:
                return "Did not receive user confirmation to upload to Jira. You can exit with a final answer."

        except Exception as e:
            logger.error("An error occurred when getting interaction content: %s", e)
            logger.info("Defaulting to not uploading to Jira")
            return ("Did not upload to Jira because human confirmation was not received. "
                    "You can exit with a final answer")

        logger.debug("Creating %s in Jira", input_text)
        # input_text = process_input_text(input_text)
        jira_issues = get_epics_tool(config.root_path)
        logger.debug("Creating %s in Jira", input_text)
        jira = JiraTool(domain=config.jira_domain, project_key=config.jira_project_key, ticket_type=input_text)
        timeout_config = httpx.Timeout(config.timeout, connect=config.connect)
        lines = ["### Created " + str(input_text) + ":"]
        results = []
        if input_text == 'epics':
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                tickets = [jira.create_epic(client, t_data) for t_data in jira_issues[input_text]]
                results = await asyncio.gather(*tickets)
        elif input_text == 'tasks':
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                tickets = [jira.create_task(client, t_data) for t_data in jira_issues[input_text]]
                results = await asyncio.gather(*tickets)
        elif input_text == 'bugs':
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                tickets = [jira.create_bug(client, t_data) for t_data in jira_issues[input_text]]
                results = await asyncio.gather(*tickets)
        elif input_text == 'features':
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                tickets = [jira.create_feature(client, t_data) for t_data in jira_issues[input_text]]
                results = await asyncio.gather(*tickets)

        for _, result in enumerate(results, start=1):
            lines.append(f"- **{result[0]}**: {config.jira_domain + '/browse/' + str(result[0])}")

        output_file = config.root_path + str(input_text) + "_data.json"
        with open(output_file, "w", encoding='utf-8') as json_file:
            json.dump(results, json_file, indent=4)

        return "\n".join(lines)

    yield FunctionInfo.from_fn(
        _arun,
        description=("This tool will import data that has already been extracted on epics,features, bugs and "
                     "tasks and will create jira tickets. If the user asks create epics, then pass epics as input or "
                     "if user asks features pass features as input and so on for other types"))


class GetJiraToolConfig(FunctionBaseConfig, name="get_jira_tickets_tool"):
    root_path: str
    jira_domain: str
    jira_project_key: str


@register_function(config_type=GetJiraToolConfig)
async def get_jira_tickets_tool(config: GetJiraToolConfig, builder: Builder):

    headers = {"Authorization": f"Bearer {os.getenv('JIRA_TOKEN')}", "Accept": "application/json"}

    # JIRA API endpoint to fetch issues
    api_endpoint = f"{config.jira_domain}/rest/api/2/search"

    # Query parameters to fetch all issues from the project
    query_params = {
        "jql": f"project={config.jira_project_key}",
        "maxResults": 100,  # Adjust as needed
        "fields": "summary,issuetype,priority,customfield_10016,description,epic",  # Modify if needed
    }

    async def _arun(input_text: str) -> str:
        response = requests.get(api_endpoint, headers=headers, params=query_params, timeout=30)

        if response.status_code == 200:
            data = response.json()["issues"]
            result = {"tasks": [], "epics": [], "new_features": [], "bugs": []}

            # Map JIRA issue types to categories in the result dictionary
            issue_type_mapping = {
                "Task": "tasks",
                "Epic": "epics",
                "New Feature": "new_features",
                "Bug": "bugs",
            }

            for issue in data:
                issue_type = issue["fields"]["issuetype"]["name"]
                category = issue_type_mapping.get(issue_type, "tasks")
                formatted_issue = {
                    "title": issue["fields"]["summary"],
                    "epic": issue["fields"].get("epic", "None"),
                    "priority": issue["fields"]["priority"]["name"] if issue["fields"].get("priority") else "None",
                    "storypoints": issue["fields"].get("customfield_10016", "None"),
                    "description": issue["fields"].get("description", "None"),
                }

                result[category].append(formatted_issue)

            # Save the result to a JSON file
            with open(config.root_path + "jira_tickets.json", "w", encoding='utf-8') as json_file:
                json.dump(result, json_file, indent=4)

            return "JIRA issues have been successfully saved to jira_tickets.json"

    yield FunctionInfo.from_fn(
        _arun,
        description=("This tool will get all jira tickets associated with a particular project. The project "
                     "name the user will provide and processes that and save into a json format file"))
