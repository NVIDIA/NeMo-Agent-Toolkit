# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent Leaderboard v2 dataset loader.

Downloads from HuggingFace galileo-ai/agent-leaderboard-v2 and transforms
scenarios into a DataFrame for NAT's eval runner.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from nat.builder.builder import EvalBuilder
from nat.builder.dataset_loader import DatasetLoaderInfo
from nat.cli.register_workflow import register_dataset_loader

from .config import AgentLeaderboardDatasetConfig

logger = logging.getLogger(__name__)


def _convert_tool_json_strings(tool_record: dict) -> dict:
    """Convert tool JSON string fields to proper dicts."""
    tool = dict(tool_record)
    for field in ("properties", "response_schema"):
        if field in tool and isinstance(tool[field], str):
            tool[field] = json.loads(tool[field])
    return tool


def _derive_expected_tool_calls(user_goals: list[str], tools: list[dict]) -> list[dict]:
    """Heuristic: match goal keywords to tool names/descriptions."""
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
        "appointment": ["appointment", "schedule", "book"],
        "contact": ["contact", "phone", "email", "address", "update"],
        "wire": ["wire", "international", "swift"],
        "exchange": ["exchange", "rate", "currency", "convert"],
        "investment": ["investment", "portfolio", "stock"],
        "insurance": ["insurance", "policy", "claim", "coverage"],
        "health": ["health", "medical", "prescription", "doctor"],
    }

    expected = []
    seen = set()
    for goal in user_goals:
        goal_lower = goal.lower()
        for tool in tools:
            tool_name = tool.get("title", "")
            tool_desc = tool.get("description", "").lower()
            tool_name_lower = tool_name.lower()

            for keyword, patterns in keyword_mappings.items():
                if keyword in goal_lower and any(
                    p in tool_name_lower or p in tool_desc for p in patterns
                ):
                    if tool_name not in seen:
                        seen.add(tool_name)
                        expected.append({"tool": tool_name, "parameters": {}})
                    break

    return expected


def load_agent_leaderboard_dataset(
    file_path: str | Path,
    domains: list[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load the Galileo Agent Leaderboard v2 dataset.

    If file_path points to an existing JSON file, loads from disk.
    Otherwise, downloads from HuggingFace.
    """
    file_path = Path(file_path)

    # If a pre-downloaded file exists, use it
    if file_path.is_file():
        with open(file_path, encoding="utf-8") as f:
            entries = json.load(f)
        logger.info("Loaded %d entries from %s", len(entries), file_path)
    else:
        # Download from HuggingFace
        entries = _download_from_huggingface(domains or ["banking"])
        logger.info("Downloaded %d entries from HuggingFace", len(entries))

    if limit:
        entries = entries[:limit]

    rows = []
    for entry in entries:
        rows.append({
            "id": entry.get("id", ""),
            "question": json.dumps(entry),
            "answer": json.dumps(entry.get("expected_tool_calls", [])),
        })

    return pd.DataFrame(rows)


def _download_from_huggingface(domains: list[str]) -> list[dict]:
    """Download and transform from galileo-ai/agent-leaderboard-v2."""
    from datasets import load_dataset

    all_entries = []
    for domain in domains:
        try:
            tools_ds = load_dataset("galileo-ai/agent-leaderboard-v2", "tools", split=domain)
            personas_ds = load_dataset("galileo-ai/agent-leaderboard-v2", "personas", split=domain)
            scenarios_ds = load_dataset("galileo-ai/agent-leaderboard-v2", "adaptive_tool_use", split=domain)

            tools = [_convert_tool_json_strings(dict(t)) for t in tools_ds]
            personas = [dict(p) for p in personas_ds]

            for idx, scenario in enumerate(scenarios_ds):
                scenario = dict(scenario)
                user_goals = scenario.get("user_goals", [])
                expected_tool_calls = _derive_expected_tool_calls(user_goals, tools)

                persona_idx = scenario.get("persona_index", idx)
                persona = personas[persona_idx] if persona_idx < len(personas) else {}

                all_entries.append({
                    "id": f"{domain}_scenario_{idx:03d}",
                    "question": scenario.get("first_message", ""),
                    "ground_truth": "User goals:\n" + "\n".join(f"- {g}" for g in user_goals),
                    "user_goals": user_goals,
                    "available_tools": tools,
                    "expected_tool_calls": expected_tool_calls,
                    "metadata": {
                        "domain": domain,
                        "persona_name": persona.get("name", ""),
                        "num_goals": len(user_goals),
                    },
                })

            logger.info("Loaded %d scenarios from domain '%s'", len(scenarios_ds), domain)

        except Exception:
            logger.exception("Failed to load domain '%s'", domain)

    return all_entries


@register_dataset_loader(config_type=AgentLeaderboardDatasetConfig)
async def register_agent_leaderboard_dataset_loader(
    config: AgentLeaderboardDatasetConfig, builder: EvalBuilder
):
    yield DatasetLoaderInfo(
        config=config,
        load_fn=lambda fp, **kw: load_agent_leaderboard_dataset(
            fp, domains=config.domains, limit=config.limit, **kw
        ),
        description="Galileo Agent Leaderboard v2 dataset loader",
    )
