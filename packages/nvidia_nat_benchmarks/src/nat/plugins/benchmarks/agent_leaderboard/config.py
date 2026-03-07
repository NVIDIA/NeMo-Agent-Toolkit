# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration types for Galileo Agent Leaderboard v2 benchmark."""

from collections.abc import Callable

from pydantic import Field

from nat.data_models.agent import AgentBaseConfig
from nat.data_models.dataset_handler import EvalDatasetBaseConfig
from nat.data_models.evaluator import EvaluatorBaseConfig

AVAILABLE_DOMAINS = ["banking", "healthcare", "insurance", "investment", "telecom"]


class AgentLeaderboardDatasetConfig(EvalDatasetBaseConfig, name="agent_leaderboard"):
    """Dataset config for Galileo Agent Leaderboard v2.

    Downloads scenarios from HuggingFace galileo-ai/agent-leaderboard-v2.
    Each scenario has a user message, user_goals, and available tools.
    """

    domains: list[str] = Field(
        default=["banking"],
        description=f"Domains to include: {AVAILABLE_DOMAINS}",
    )
    limit: int | None = Field(
        default=None,
        description="Max scenarios to load (for testing)",
    )

    def parser(self) -> tuple[Callable, dict]:
        from .dataset import load_agent_leaderboard_dataset
        return load_agent_leaderboard_dataset, {"domains": self.domains, "limit": self.limit}


class AgentLeaderboardWorkflowConfig(AgentBaseConfig, name="agent_leaderboard_workflow"):
    """Workflow config for Agent Leaderboard evaluation.

    Uses tool_calling_agent with stub tools from the dataset.
    Tool calls are captured via ToolIntentBuffer for TSQ scoring.
    """

    description: str = Field(default="Agent Leaderboard Workflow")
    max_steps: int = Field(
        default=10,
        description="Maximum tool-calling steps per scenario",
    )
    system_prompt: str | None = Field(
        default=(
            "You are a tool-calling agent. Select the correct tools to handle the user's request.\n"
            "Focus on selecting the RIGHT TOOL for each step. Use placeholder values for parameters.\n"
            "Tool responses are simulated — focus on tool choice, not data quality."
        ),
        description="System prompt for the agent",
    )


class TSQEvaluatorConfig(EvaluatorBaseConfig, name="agent_leaderboard_tsq"):
    """Tool Selection Quality evaluator for Agent Leaderboard.

    Computes F1 score between predicted and expected tool calls.
    """

    tool_weight: float = Field(default=1.0, description="Weight for tool selection accuracy")
    parameter_weight: float = Field(default=0.0, description="Weight for parameter accuracy")
