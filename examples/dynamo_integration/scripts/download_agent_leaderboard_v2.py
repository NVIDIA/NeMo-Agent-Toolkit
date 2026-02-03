#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Download and transform agent leaderboard v2 dataset from Hugging Face for NAT evaluation framework.
This version uses domain-specific scenarios (banking, healthcare, etc.).
"""

import json
import logging
from pathlib import Path
from typing import Any

from datasets import load_dataset

logger = logging.getLogger(__name__)


def convert_tool_json_strings(tool_record: dict) -> dict:
    """Convert tool JSON strings to proper dictionaries."""
    tool = dict(tool_record)

    # Convert 'properties' from JSON string to dict
    if "properties" in tool and isinstance(tool["properties"], str):
        tool["properties"] = json.loads(tool["properties"])

    # Convert 'response_schema' from JSON string to dict
    if "response_schema" in tool and isinstance(tool["response_schema"], str):
        tool["response_schema"] = json.loads(tool["response_schema"])

    return tool


def derive_expected_tool_calls(user_goals: list[str], tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Derive expected tool calls from user goals by matching goal keywords to tool names and descriptions.

    This is a heuristic approach that:
    1. Extracts keywords from user goals
    2. Matches keywords against tool names and descriptions
    3. Returns a list of expected tool calls with parameter placeholders

    Args:
        user_goals: List of user goal descriptions
        tools: Available tools with their schemas

    Returns:
        List of expected tool calls with format: [{"tool": "tool_name", "parameters": {...}}]
    """
    expected_calls = []

    # Common keyword mappings to tool patterns
    keyword_mappings = {
        "balance": ["balance", "check", "account"],
        "transfer": ["transfer", "send", "move", "pay"],
        "transaction": ["transaction", "history", "statement"],
        "payment": ["payment", "pay", "bill"],
        "card": ["card", "credit", "debit"],
        "loan": ["loan", "mortgage", "credit"],
        "dispute": ["dispute", "challenge", "report"],
        "limit": ["limit", "increase", "decrease"],
        "block": ["block", "freeze", "lock"],
        "unblock": ["unblock", "unfreeze", "unlock"],
        "statement": ["statement", "report", "summary"],
        "contact": ["contact", "phone", "email", "address"],
        "beneficiary": ["beneficiary", "recipient", "payee"],
        "standing": ["standing", "recurring", "automatic"],
        "wire": ["wire", "international", "swift"],
    }

    # Process each goal
    for goal in user_goals:
        goal_lower = goal.lower()
        matched_tools = []

        # Try to match goal keywords to tools
        for tool in tools:
            tool_name = tool.get("title", "").lower()
            tool_desc = tool.get("description", "").lower()

            # Check if any keywords match
            for keyword, patterns in keyword_mappings.items():
                if keyword in goal_lower:
                    # Check if tool name or description contains any pattern
                    if any(pattern in tool_name or pattern in tool_desc for pattern in patterns):
                        # Extract required parameters from tool schema
                        params = {}
                        properties = tool.get("properties", {})
                        required = tool.get("required", [])

                        for param_name in required:
                            param_info = properties.get(param_name, {})
                            param_type = param_info.get("type", "string")

                            # Create placeholder based on type
                            if param_type == "string":
                                params[param_name] = f"<{param_name}>"
                            elif param_type == "integer":
                                params[param_name] = 0
                            elif param_type == "number":
                                params[param_name] = 0.0
                            elif param_type == "boolean":
                                params[param_name] = True
                            else:
                                params[param_name] = None

                        matched_tools.append({
                            "tool": tool.get("title", ""),
                            "parameters": params,
                            "goal": goal,  # Keep track of which goal this satisfies
                        })
                        break  # Only match once per keyword

        # Add matched tools for this goal
        expected_calls.extend(matched_tools)

    # Remove duplicates while preserving order
    seen = set()
    unique_calls = []
    for call in expected_calls:
        tool_sig = call["tool"]
        if tool_sig not in seen:
            seen.add(tool_sig)
            unique_calls.append(call)

    return unique_calls


def transform_scenario_to_nat_format(
    scenario: dict[str, Any],
    tools: list[dict[str, Any]],
    personas: list[dict[str, Any]],
    domain: str,
    index: int,
) -> dict[str, Any]:
    """
    Transform agent leaderboard v2 scenario to NAT evaluation format.

    Args:
        scenario: Scenario from adaptive_tool_use config
        tools: Available tools for the domain
        personas: Available personas for the domain
        domain: Domain name (banking, healthcare, etc.)
        index: Scenario index for generating unique IDs

    Returns:
        NAT-formatted evaluation entry
    """
    # Extract scenario details (v2 structure uses different field names)
    persona_index = scenario.get("persona_index", index)
    first_message = scenario.get("first_message", "")
    user_goals = scenario.get("user_goals", [])

    # Get persona details if available
    persona_info = None
    if persona_index < len(personas):
        persona_info = personas[persona_index]

    # Format ground truth from user goals
    if user_goals:
        ground_truth = "User goals:\n" + "\n".join(f"- {goal}" for goal in user_goals)
    else:
        ground_truth = "Complete the user's banking tasks."

    # Derive expected tool calls from user goals
    expected_tool_calls = derive_expected_tool_calls(user_goals, tools)

    # Build NAT entry
    nat_entry = {
        "id": f"{domain}_scenario_{index:03d}",
        "question": first_message,
        "ground_truth": ground_truth,
        "metadata": {
            "benchmark": "agent-leaderboard-v2",
            "domain": domain,
            "persona_index": persona_index,
            "persona_name": persona_info.get("name", "") if persona_info else "",
            "num_goals": len(user_goals),
        },
        "user_goals": user_goals,
        "available_tools": tools,  # All domain tools available
        "expected_tool_calls": expected_tool_calls,  # Derived from goals
    }

    return nat_entry


def download_and_transform_v2_dataset(
    output_dir: Path,
    domains: list[str] | None = None,
) -> None:
    """
    Download agent leaderboard v2 dataset and transform it to NAT format.

    Args:
        output_dir: Directory to save transformed datasets
        domains: List of domains to download (banking, healthcare, insurance, investment, telecom)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default to all domains if none specified
    available_domains = ["banking", "healthcare", "insurance", "investment", "telecom"]
    if domains is None:
        domains = ["banking"]  # Start with just one for testing
        logger.info("No domains specified, using default: %s", domains)

    logger.info("Loading agent leaderboard v2 dataset from Hugging Face...")

    all_entries = []

    for domain in domains:
        if domain not in available_domains:
            logger.warning("Domain '%s' not in available domains: %s", domain, available_domains)
            continue

        try:
            logger.info("Loading domain: %s", domain)

            # Load all three configs for this domain
            tools_ds = load_dataset("galileo-ai/agent-leaderboard-v2", "tools", split=domain)
            personas_ds = load_dataset("galileo-ai/agent-leaderboard-v2", "personas", split=domain)
            scenarios_ds = load_dataset("galileo-ai/agent-leaderboard-v2", "adaptive_tool_use", split=domain)

            logger.info("Loaded %d tools, %d personas, %d scenarios for %s",
                        len(tools_ds),
                        len(personas_ds),
                        len(scenarios_ds),
                        domain)

            # Convert tools
            tools = [convert_tool_json_strings(dict(tool)) for tool in tools_ds]
            personas = [dict(persona) for persona in personas_ds]

            # Transform each scenario
            for idx, scenario in enumerate(scenarios_ds):
                nat_entry = transform_scenario_to_nat_format(dict(scenario), tools, personas, domain, idx)
                all_entries.append(nat_entry)

            # Save domain-specific file
            domain_file = output_dir / f"agent_leaderboard_v2_{domain}.json"
            domain_entries = [e for e in all_entries if e["metadata"]["domain"] == domain]
            with open(domain_file, "w") as f:
                json.dump(domain_entries, f, indent=2)
            logger.info("Saved %d entries to %s", len(domain_entries), domain_file)

            # Also save raw domain data for reference
            raw_dir = output_dir / "raw" / domain
            raw_dir.mkdir(parents=True, exist_ok=True)

            with open(raw_dir / "tools.json", "w") as f:
                json.dump(tools, f, indent=2)
            with open(raw_dir / "personas.json", "w") as f:
                json.dump(personas, f, indent=2)
            with open(raw_dir / "adaptive_tool_use.json", "w") as f:
                json.dump([dict(s) for s in scenarios_ds], f, indent=2)

            logger.info("Saved raw data to %s", raw_dir)

        except Exception:
            logger.exception("Failed to load domain: %s", domain)
            continue

    # Save combined file
    if all_entries:
        combined_file = output_dir / "agent_leaderboard_v2_all.json"
        with open(combined_file, "w") as f:
            json.dump(all_entries, f, indent=2)
        logger.info("Saved %d total entries to %s", len(all_entries), combined_file)
    else:
        logger.warning("No entries were loaded from any domain")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Download and transform agent leaderboard v2 dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Output directory for transformed datasets",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        help="Domains to download (banking, healthcare, insurance, investment, telecom)",
    )

    args = parser.parse_args()

    # Set cache location if not already set
    import os
    if "HF_HOME" not in os.environ:
        default_hf_home = os.path.expanduser("~/.cache/huggingface")
        logger.info("HF_HOME not set, using default: %s", default_hf_home)
        os.environ["HF_HOME"] = default_hf_home

    download_and_transform_v2_dataset(args.output_dir, args.domains)
