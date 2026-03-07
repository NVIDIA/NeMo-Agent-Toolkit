# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path

from pydantic import Field

from nat.data_models.agent import AgentBaseConfig
from nat.data_models.dataset_handler import EvalDatasetBaseConfig
from nat.data_models.evaluator import EvaluatorBaseConfig


class ToolTalkDatasetConfig(EvalDatasetBaseConfig, name="tooltalk"):
    """Dataset config for ToolTalk benchmark.

    file_path should point to a ToolTalk data directory (e.g. tooltalk/data/easy/)
    containing one JSON file per conversation.
    """

    database_dir: str = Field(
        description="Path to ToolTalk database directory (contains Account.json, Alarm.json, etc.)",
    )

    def parser(self) -> tuple[Callable, dict]:
        from .dataset import load_tooltalk_dataset
        return load_tooltalk_dataset, {}


class ToolTalkWorkflowConfig(AgentBaseConfig, name="tooltalk_workflow"):
    """Workflow config for ToolTalk benchmark.

    Runs multi-turn conversations using NAT's LLM with ToolTalk's simulated tool backends.
    """

    description: str = Field(default="ToolTalk Benchmark Workflow")
    database_dir: str = Field(
        description="Path to ToolTalk database directory",
    )
    api_mode: str = Field(
        default="all",
        description="Which API docs to include: 'exact' (only APIs in conversation), "
        "'suite' (all APIs in used suites), or 'all'",
    )
    disable_documentation: bool = Field(
        default=False,
        description="If True, send empty descriptions in tool schemas",
    )
    max_tool_calls_per_turn: int = Field(
        default=10,
        description="Maximum tool calls per assistant turn before forcing a text response",
    )


class ToolTalkEvaluatorConfig(EvaluatorBaseConfig, name="tooltalk_evaluator"):
    """Evaluator config for ToolTalk benchmark.

    Uses ToolTalk's built-in metrics: recall, action_precision, bad_action_rate, success.
    """

    database_dir: str = Field(
        description="Path to ToolTalk database directory",
    )
