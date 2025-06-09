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
import os
from datetime import datetime

import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

from . import utils
from .prompts import MaintenanceCheckPrompts

NO_ONGOING_MAINTENANCE_STR = "No ongoing maintenance found for the host."


class MaintenanceCheckToolConfig(FunctionBaseConfig, name="maintenance_check"):
    description: str = Field(default=MaintenanceCheckPrompts.TOOL_DESCRIPTION, description="Description of the tool.")
    llm_name: LLMRef
    prompt: str = Field(default=MaintenanceCheckPrompts.PROMPT,
                        description="Main prompt for the maintenance check task.")
    static_data_path: str | None = Field(
        default="examples/alert_triage_agent/data/maintenance_static_dataset.csv",
        description=(
            "Path to the static maintenance data CSV file. If not provided, the tool will not check for maintenance."))


def _load_maintenance_data(path: str) -> pd.DataFrame:
    """
    Load maintenance window data from a CSV file into a pandas DataFrame.

    NOTE: This is an example implementation using a CSV file to demonstrate the maintenance
    check functionality. In a production environment, users should modify this function to
    integrate with their organization's maintenance tracking system or database.

    The input CSV must contain these columns:
      - host_id (str): Hostname or identifier of the system under maintenance.
      - maintenance_start (str): Start timestamp of the maintenance window
        in "YYYY-MM-DD HH:MM:SS" format.
      - maintenance_end (str): End timestamp of the maintenance window
        in "YYYY-MM-DD HH:MM:SS" format. This column must be present in the
        CSV but may contain empty values; empty or invalid entries will be
        coerced to NaT to indicate ongoing maintenance.

    Parameters:
      path (str): File path to the CSV containing maintenance data.

    Returns:
      pd.DataFrame: The loaded data with
        - maintenance_start (datetime64[ns])
        - maintenance_end   (datetime64[ns])
      Columns converted to datetime, with parsing errors coerced to NaT.

    Raises:
      ValueError: If any required column (host_id, maintenance_start, maintenance_end) is missing.
    """
    df = pd.read_csv(path)

    # Verify required columns
    required = {"host_id", "maintenance_start", "maintenance_end"}
    missing = required - set(df.columns)
    if missing:
        missing = sorted(missing)
        utils.logger.error("Missing required columns: %s", ", ".join(missing))
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df["maintenance_start"] = pd.to_datetime(df["maintenance_start"], errors="coerce")
    df["maintenance_end"] = pd.to_datetime(df["maintenance_end"], errors="coerce")

    return df


def _parse_alert_data(input_message: str) -> dict | None:
    """
    Parse alert data from an input message containing JSON into a dictionary.

    This function extracts and parses a JSON object from a text message that may contain
    additional text before and/or after the JSON. It handles both double and single quoted
    JSON strings and can parse nested JSON structures.

    Args:
        input_message (str): Input message containing a JSON object, which may be surrounded
                            by additional text. The JSON object should contain alert details
                            like host_id and timestamp.

    Returns:
        dict | None: The parsed alert data as a dictionary if successful parsing,
                    containing fields like host_id and timestamp.
                    Returns None if no valid JSON object is found or parsing fails.
    """
    # Extract everything between first { and last }
    start = input_message.find("{")
    end = input_message.rfind("}") + 1
    if start == -1 or end == 0:
        utils.logger.error("No JSON object found in input message")
        return None

    alert_json_str = input_message[start:end]
    try:
        return json.loads(alert_json_str.replace("'", '"'))
    except Exception as e:
        utils.logger.error("Failed to parse alert from input message: %s", e)
        return None


def _get_active_maintenance(df: pd.DataFrame, host_id: str, alert_time: datetime) -> tuple[str, str] | None:
    """
    Find the active maintenance record for a given host at a specific time.

    Parameters:
        df (pd.DataFrame): DataFrame containing maintenance records with columns:
            - host_id (str): Hostname or identifier of the system
            - maintenance_start (datetime64[ns]): Start timestamp of maintenance window
            - maintenance_end (datetime64[ns]): End timestamp of maintenance window (NaT if ongoing)
        host_id (str): Host identifier to check for maintenance
        alert_time (datetime): Timestamp to check for active maintenance

    Returns:
        tuple[str, str] | None: If maintenance is active, returns a tuple containing:
            - maintenance_start (str): Start time in "YYYY-MM-DD HH:MM:SS" format
            - maintenance_end (str): End time in "YYYY-MM-DD HH:MM:SS" format, or empty string if ongoing
            Returns None if no maintenance is active for the host at alert_time.
    """
    # Filter for records that match either host_id
    host_maintenance = df[(df["host_id"] == host_id)]

    # Check if alert_time falls within maintenance period or if maintenance_end is NaN (ongoing)
    ongoing = host_maintenance[(host_maintenance["maintenance_start"] <= alert_time)
                               & ((host_maintenance["maintenance_end"].isna())
                                  | (host_maintenance["maintenance_end"] > alert_time))]

    if ongoing.empty:
        return None

    # Get the first ongoing maintenance record
    active_maintenance = ongoing.iloc[0]

    # Convert to formatted string
    timestamp_format = "%Y-%m-%d %H:%M:%S"
    start_time_str = active_maintenance["maintenance_start"].strftime(timestamp_format)
    end_time_str = (active_maintenance["maintenance_end"].strftime(timestamp_format)
                    if pd.notna(active_maintenance["maintenance_end"]) else "")

    return start_time_str, end_time_str


def _summarize_alert(llm, prompt_template, alert, maintenance_start_str, maintenance_end_str):
    """
    Generate a summary report for an alert when the affected host is under maintenance.

    Args:
        llm: The language model to use for generating the summary
        prompt_template: The prompt template to use for generating the summary
        alert (dict): Dictionary containing the alert details
        maintenance_start_str (str): Start time of maintenance window in "YYYY-MM-DD HH:MM:SS" format
        maintenance_end_str (str): End time of maintenance window in "YYYY-MM-DD HH:MM:SS" format,
            or empty string if maintenance is ongoing

    Returns:
        str: A markdown-formatted report summarizing the alert and maintenance status
    """
    sys_prompt = prompt_template.format(maintenance_start_str=maintenance_start_str,
                                        maintenance_end_str=maintenance_end_str)
    prompt_template = ChatPromptTemplate([("system", sys_prompt), MessagesPlaceholder("msgs")])
    summarization_chain = prompt_template | llm
    alert_json_str = json.dumps(alert)
    result = summarization_chain.invoke({"msgs": [HumanMessage(content=alert_json_str)]}).content
    return result


@register_function(config_type=MaintenanceCheckToolConfig)
async def maintenance_check(config: MaintenanceCheckToolConfig, builder: Builder):
    # Set up LLM
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    maintenance_data_path = config.static_data_path

    async def _arun(input_message: str) -> str:
        # NOTE: This is just an example implementation of maintenance status checking using a CSV file.
        # Users should implement their own maintenance check logic specific to their environment
        # and infrastructure setup. The key is to check if a host is under maintenance during
        # the time of an alert, to help determine if the alert can be deprioritized.

        utils.log_header("Maintenance Checker")

        if not maintenance_data_path:
            utils.logger.info("No maintenance data path provided, skipping maintenance check")
            return NO_ONGOING_MAINTENANCE_STR  # the triage agent will run as usual

        if not os.path.exists(maintenance_data_path):
            utils.logger.info("Maintenance data file does not exist: %s. Skipping maintenance check.",
                              maintenance_data_path)
            return NO_ONGOING_MAINTENANCE_STR  # the triage agent will run as usual

        alert = _parse_alert_data(input_message)
        if alert is None:
            utils.logger.info("Failed to parse alert from input message, skipping maintenance check")
            return NO_ONGOING_MAINTENANCE_STR

        host = alert.get("host_id")
        alert_time_str = alert.get("timestamp")
        if not (alert and host and alert_time_str):
            utils.logger.info(
                "Failed to parse alert or the host or alert time from input message, skipping maintenance check")
            return NO_ONGOING_MAINTENANCE_STR

        try:
            alert_time = datetime.strptime(alert_time_str, "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError as e:
            utils.logger.error("Failed to parse alert time from input message: %s, skipping maintenance check", e)
            return NO_ONGOING_MAINTENANCE_STR

        maintenance_df = _load_maintenance_data(maintenance_data_path)
        maintenance_info = _get_active_maintenance(maintenance_df, host, alert_time)
        if not maintenance_info:
            utils.logger.info("Host: [%s] is NOT under maintenance according to the maintenance database", host)
            return NO_ONGOING_MAINTENANCE_STR

        try:
            maintenance_start_str, maintenance_end_str = maintenance_info
        except ValueError:
            utils.logger.error(
                "Failed to parse maintenance info into start and end times: %s, skipping maintenance check",
                maintenance_info)
            return NO_ONGOING_MAINTENANCE_STR

        # maintenance info found, summarize alert and return a report (agent execution will be skipped)
        utils.logger.info("Host: [%s] is under maintenance according to the maintenance database", host)

        report = _summarize_alert(llm=llm,
                                  prompt_template=config.prompt,
                                  alert=alert,
                                  maintenance_start_str=maintenance_start_str,
                                  maintenance_end_str=maintenance_end_str)

        utils.log_footer()
        return report

    yield FunctionInfo.from_fn(_arun, description=config.description)
