# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Prompt purpose descriptions for the alert triage agent's prompt optimizer.

This module defines prompt purposes used by the optimizer to understand and improve
system prompts for the alert triage workflow and its sub-agents.
"""


class OptimizerPrompts:
    """
    Prompt purpose descriptions for optimizer-enabled agent configurations.

    This class defines the purpose and expected behavior of system prompts used
    in the alert triage workflow, enabling prompt optimization through detailed
    guidance on what each prompt should accomplish.
    """
    AGENT_PROMPT_PURPOSE = """This is the system prompt that instructs the Alert Triage Agent on how to behave and respond to system alerts. It is used as a SystemMessage that's prepended to every LLM conversation, providing the agent with its role and behavior guidelines.

The prompt should be well-structured and provide specific instructions to help the agent:
- Analyze incoming alerts and identify their type (e.g., InstanceDown, HighCPUUsage)
- Select and use the appropriate diagnostic tools for each alert type (hardware_check, host_performance_check, network_connectivity_check, telemetry_metrics_analysis_agent, monitoring_process_check)
- Avoid calling the same tool repeatedly during a single alert investigation
- Correlate collected data from multiple tools to determine root causes
- Distinguish between true issues, false positives, and benign anomalies
- Generate structured markdown triage reports with clear sections: Alert Summary, Collected Metrics, Analysis, Recommended Actions, and Alert Status

The prompt should give the agent clear security context and explicit instructions on the expected final report format to ensure consistent, actionable output for system analysts."""
    TELEMETRY_AGENT_PROMPT_PURPOSE = """This is the system prompt for the Telemetry Metrics Analysis Agent, a specialized sub-agent within the alert triage system. It is used as a SystemMessage for a nested agent that the main Alert Triage Agent can call to analyze remotely collected telemetry data.

This sub-agent receives two inputs (`host_id` and `alert_type`) and is responsible for selecting and using the appropriate telemetry analysis tools to investigate the alert. It has access to two specialized telemetry tools:
- `telemetry_metrics_host_heartbeat_check`: Checks server heartbeat to determine if the host is up and responsive
- `telemetry_metrics_host_performance_check`: Analyzes CPU usage trends over the past 14 days to identify patterns

The prompt should provide clear instructions to help the agent:
- Understand the alert type and associated host_id provided as input
- Select the correct tool based on the alert type (heartbeat check for instance down alerts, performance check for high CPU usage alerts)
- Execute the selected tool exactly once to gather telemetry data
- Analyze the collected data to identify patterns such as periodic behavior, anomalous peaks, or normal fluctuations
- Return raw data from the tool along with a concise summary of findings
- Highlight any signs that indicate benign (non-critical) behavior, such as normal periodic spikes or consistent uptime, to help de-escalate false alarms
- Provide insights or hypotheses that explain whether the telemetry supports or contradicts the triggered alert

The prompt should ensure the agent delivers actionable intelligence that helps the main Alert Triage Agent distinguish between genuine issues and false positives."""
