# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Configuration types for the BFCL benchmark."""

from collections.abc import Callable
from enum import StrEnum
from typing import Literal

from pydantic import Field
from pydantic import field_validator

from nat.data_models.agent import AgentBaseConfig
from nat.data_models.dataset_handler import EvalDatasetBaseConfig
from nat.data_models.evaluator import EvaluatorBaseConfig


class BFCLTestCategory(StrEnum):
    """Valid BFCL v3 single-turn AST test categories."""

    SIMPLE = "simple"
    MULTIPLE = "multiple"
    PARALLEL = "parallel"
    PARALLEL_MULTIPLE = "parallel_multiple"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    LIVE_SIMPLE = "live_simple"
    LIVE_MULTIPLE = "live_multiple"
    LIVE_PARALLEL = "live_parallel"
    LIVE_PARALLEL_MULTIPLE = "live_parallel_multiple"
    IRRELEVANCE = "irrelevance"
    LIVE_IRRELEVANCE = "live_irrelevance"
    LIVE_RELEVANCE = "live_relevance"


# Type alias for the supported AST parsing languages
BFCLLanguage = Literal["Python", "Java", "JavaScript"]


class BFCLDatasetConfig(EvalDatasetBaseConfig, name="bfcl"):
    """Dataset config for BFCL benchmark.

    file_path should point to a BFCL v3 JSONL test file (e.g. BFCL_v3_simple.json).
    """

    test_category: BFCLTestCategory = Field(
        default=BFCLTestCategory.SIMPLE,
        description="BFCL test category: simple, multiple, parallel, parallel_multiple, "
        "java, javascript, irrelevance, live_simple, etc.",
    )
    possible_answer_file: str | None = Field(
        default=None,
        description="Path to the BFCL possible_answer file. If None, auto-resolves from bfcl package.",
    )

    def parser(self) -> tuple[Callable, dict]:
        from .dataset import load_bfcl_dataset

        return load_bfcl_dataset, {"test_category": self.test_category}


class BFCLASTWorkflowConfig(AgentBaseConfig, name="bfcl_ast_workflow"):
    """Workflow config for BFCL AST (prompting) evaluation.

    The LLM receives function schemas as text in the system prompt and outputs
    raw function call text (e.g. ``func_name(param=value)``). No tools= parameter.
    """

    description: str = Field(default="BFCL AST Prompting Workflow")


class BFCLFCWorkflowConfig(AgentBaseConfig, name="bfcl_fc_workflow"):
    """Workflow config for BFCL Native FC evaluation.

    Uses ``llm.bind_tools(schemas)`` + ``ainvoke()`` — Native Function Calling.
    Extracts tool_calls from AIMessage and formats as BFCL expected output.
    """

    description: str = Field(default="BFCL Native FC Workflow")


class BFCLEvaluatorConfig(EvaluatorBaseConfig, name="bfcl_evaluator"):
    """Evaluator config for BFCL AST/FC benchmark.

    Calls BFCL's ast_checker directly in-process for scoring.
    """

    test_category: BFCLTestCategory = Field(
        default=BFCLTestCategory.SIMPLE,
        description="BFCL test category (must match the dataset config).",
    )
    language: BFCLLanguage = Field(
        default="Python",
        description="Programming language for AST parsing: Python, Java, or JavaScript.",
    )

    @field_validator("language")
    @classmethod
    def _validate_language_category_match(cls, v: str) -> str:
        """Validate that language is a supported value."""
        valid = {"Python", "Java", "JavaScript"}
        if v not in valid:
            raise ValueError(f"language must be one of {valid}, got '{v}'")
        return v
