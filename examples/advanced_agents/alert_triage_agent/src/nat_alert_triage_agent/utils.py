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

import json
import logging
import math
import os

import ansible_runner
import pandas as pd

from nat.builder.framework_enum import LLMFrameworkEnum

logger = logging.getLogger("nat_alert_triage_agent")

# module‐level variable; loaded on first use
_DATA_CACHE: dict[str, pd.DataFrame | dict | None] = {
    'offline_data': None,
    'benign_fallback_offline_data': None,
}

# Cache LLMs by name and wrapper type
_LLM_CACHE = {}


async def _get_llm(builder, llm_name, wrapper_type):
    """
    Get an LLM from cache or create and cache a new one.

    Args:
        builder: The builder instance to create new `llm`
        llm_name: Name of the LLM to get/create
        wrapper_type: Type of LLM wrapper framework to use

    Returns:
        The cached or newly created LLM instance
    """
    cache_key = (llm_name, wrapper_type)
    if cache_key not in _LLM_CACHE:
        _LLM_CACHE[cache_key] = await builder.get_llm(llm_name=llm_name, wrapper_type=wrapper_type)
    return _LLM_CACHE[cache_key]


async def llm_ainvoke(config, builder, user_prompt, system_prompt=None):
    """
    A helper function to invoke an LLM with a system prompt and user prompt.
    Uses a cached LLM instance if one exists for the given name and wrapper type.
    """
    from langchain_core.messages import HumanMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.prompts import MessagesPlaceholder
    llm = await _get_llm(builder, config.llm_name, LLMFrameworkEnum.LANGCHAIN)

    if system_prompt:
        prompt = ChatPromptTemplate([("system", system_prompt), MessagesPlaceholder("msgs")])
    else:
        prompt = ChatPromptTemplate([MessagesPlaceholder("msgs")])
    chain = prompt | llm
    result = await chain.ainvoke({"msgs": [HumanMessage(content=user_prompt)]})
    return result.content


def log_header(log_str: str, dash_length: int = 100, level: int = logging.DEBUG):
    """Logs a centered header with '=' dashes at the given log level."""
    left = math.floor((dash_length - len(log_str)) / 2)
    right = dash_length - len(log_str) - left
    header = "=" * left + log_str + "=" * right
    logger.log(level, header)


def log_footer(dash_length: int = 100, level: int = logging.DEBUG):
    """Logs a full line of '=' dashes at the given log level."""
    footer = "=" * dash_length
    logger.log(level, footer)


def preload_offline_data(offline_data_path: str | None, benign_fallback_data_path: str | None):
    """
    Preloads test data from CSV and JSON files into module-level cache.

    Args:
        offline_data_path (str): Path to the test data CSV file
        benign_fallback_data_path (str): Path to the benign fallback data JSON file
    """
    if offline_data_path is None:
        raise ValueError("offline_data_path must be provided")

    if benign_fallback_data_path is None:
        raise ValueError("benign_fallback_data_path must be provided")

    _DATA_CACHE['offline_data'] = pd.read_csv(offline_data_path)
    logger.info("Preloaded test data from: %s", offline_data_path)

    with open(benign_fallback_data_path, "r", encoding="utf-8") as f:
        _DATA_CACHE['benign_fallback_offline_data'] = json.load(f)
    logger.info("Preloaded benign fallback data from: %s", benign_fallback_data_path)


def get_offline_data() -> pd.DataFrame:
    """Returns the preloaded test data."""
    if _DATA_CACHE['offline_data'] is None:
        raise ValueError("Test data not preloaded. Call `preload_offline_data` first.")
    return pd.DataFrame(_DATA_CACHE['offline_data'])


def _get_static_data():
    """Returns the preloaded benign fallback test data."""
    if _DATA_CACHE['benign_fallback_offline_data'] is None:
        raise ValueError("Benign fallback test data not preloaded. Call `preload_offline_data` first.")
    return _DATA_CACHE['benign_fallback_offline_data']


def load_column_or_static(df, host_id, column):
    """
    Attempts to load data from a DataFrame column, falling back to static JSON if needed.

    The function assumes that in the test dataset, host_ids are unique and used to locate
    specific tool return values. This means each host_id should appear in at most one row.

    Args:
        df (pandas.DataFrame): DataFrame containing test data
        host_id (str): Host ID to look up in the DataFrame
        column (str): Column name to retrieve data from

    Returns:
        The value from either the DataFrame or static JSON for the given column.

    Raises:
        KeyError: If column not found in static data or DataFrame, or if host_id not found in DataFrame
        ValueError: If multiple rows found for the same host_id in DataFrame
    """
    if column not in df.columns:
        # Column missing from DataFrame, try loading from static JSON file
        static_data = _get_static_data()
        try:
            return static_data[column]  # pylint: disable=unsubscriptable-object
        except KeyError as exc:
            raise KeyError(f"Column '{column}' not found in test and benign fallback data") from exc
    # Column exists in DataFrame, get value for this host
    # Assumption: In test dataset, host_ids are unique and used to locate specific tool return values
    # If multiple rows found for a host_id, this indicates data inconsistency
    subset = df.loc[df["host_id"] == host_id, column]
    if subset.empty:
        raise KeyError(f"No row for host_id='{host_id}' in DataFrame")
    if len(subset) > 1:
        raise ValueError(f"Multiple rows found for host_id='{host_id}' in DataFrame. Expected unique host_ids.")

    data = subset.values[0]
    if pd.isna(data) or (data == ""):
        # If data is None, empty, or NaN, try loading from static JSON file
        static_data = _get_static_data()
        try:
            return static_data[column]  # pylint: disable=unsubscriptable-object
        except KeyError as exc:
            raise KeyError(f"Column '{column}' not found in static data") from exc

    return data


async def run_ansible_playbook(playbook: list,
                               ansible_host: str,
                               ansible_user: str,
                               ansible_port: int,
                               ansible_private_key_path: str) -> dict:
    """
    Execute an Ansible playbook against a remote host and return structured output.

    Args:
        playbook (list): Ansible playbook to execute
        ansible_host (str): Target host to run playbook against
        ansible_user (str): SSH username for connection
        ansible_port (int): SSH port number
        ansible_private_key_path (str): Path to SSH private key file

    Returns:
        dict: Structured output containing playbook execution results
    """
    # Define inventory dictionary with connection details for target host
    inventory = {
        "all": {
            "hosts": {
                "host1": {
                    "ansible_host": ansible_host,
                    "ansible_user": ansible_user,
                    "ansible_ssh_private_key_file": ansible_private_key_path,
                    "ansible_port": ansible_port,
                }
            }
        }
    }

    # Get current directory to use as private data dir
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Execute the ansible playbook using ansible-runner
    runner = ansible_runner.run(private_data_dir=current_dir, playbook=playbook, inventory=inventory)

    # Initialize output dictionary with basic run info
    output = {"ansible_status": runner.status, "return_code": runner.rc, "task_results": []}

    # If no events available, return raw stdout output
    if not hasattr(runner, "events") or not runner.events:
        output["raw_output"] = runner.stdout.read() if runner.stdout else "No output captured."
        return output

    # Process each event and extract task results
    for event in runner.events:
        # Only process successful or failed task events
        if event.get("event") not in ["runner_on_ok", "runner_on_failed"]:
            continue

        # Extract event data and build task result dictionary
        event_data = event["event_data"]
        task_result = {
            "task": event_data.get("task", "unknown"),
            "host": event_data.get("host", "unknown"),
            "status": event.get("event"),
            "stdout": event.get("stdout", ""),
            "result": event_data.get("res", {})
        }
        output["task_results"].append(task_result)

    return output
