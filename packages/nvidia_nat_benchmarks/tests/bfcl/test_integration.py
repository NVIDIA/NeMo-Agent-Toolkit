# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for BFCL benchmark pipeline.

Requires NVIDIA_API_KEY and bfcl package with data files.
Run with: pytest --run_integration --run_slow
"""

import json
import os
import subprocess

import pytest

_SKIP_REASON = None

try:
    from bfcl.constant import PROMPT_PATH, POSSIBLE_ANSWER_PATH
    _SIMPLE_FILE = os.path.join(PROMPT_PATH, "BFCL_v3_simple.json")
    if not os.path.isfile(_SIMPLE_FILE):
        _SKIP_REASON = "BFCL_v3_simple.json not found"
except ImportError:
    _SKIP_REASON = "bfcl not installed"

if not os.environ.get("NVIDIA_API_KEY"):
    _SKIP_REASON = "NVIDIA_API_KEY not set"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.timeout(300),
    pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or ""),
]


@pytest.fixture
def small_bfcl_dataset(tmp_path):
    """Create a 5-entry subset of BFCL_v3_simple for fast testing."""
    with open(_SIMPLE_FILE, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()][:5]

    test_file = tmp_path / "BFCL_v3_simple.json"
    answer_dir = tmp_path / "possible_answer"
    answer_dir.mkdir()
    answer_file = answer_dir / "BFCL_v3_simple.json"

    with open(test_file, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    # Copy matching answers
    answer_path = os.path.join(POSSIBLE_ANSWER_PATH, "BFCL_v3_simple.json")
    if os.path.isfile(answer_path):
        entry_ids = {e["id"] for e in entries}
        with open(answer_path, encoding="utf-8") as f:
            answers = [json.loads(line) for line in f if line.strip()]
        with open(answer_file, "w") as f:
            for a in answers:
                if a["id"] in entry_ids:
                    f.write(json.dumps(a) + "\n")

    return str(test_file)


class TestBFCLIntegration:

    @pytest.mark.asyncio
    async def test_end_to_end_bfcl_ast_eval(self, small_bfcl_dataset, tmp_path):
        """Full e2e: BFCL AST prompting → LLM → ast_checker → metrics."""
        import yaml
        output_dir = str(tmp_path / "output")

        config_dict = {
            "llms": {
                "nim_llm": {
                    "_type": "nim",
                    "model_name": "meta/llama-3.3-70b-instruct",
                    "api_key": os.environ["NVIDIA_API_KEY"],
                    "base_url": os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                    "max_tokens": 512,
                    "temperature": 0.0,
                },
            },
            "workflow": {
                "_type": "bfcl_ast_workflow",
                "llm_name": "nim_llm",
            },
            "eval": {
                "general": {
                    "output_dir": output_dir,
                    "workflow_alias": "bfcl_ast_integration_test",
                    "per_input_user_id": False,
                    "max_concurrency": 3,
                    "dataset": {
                        "_type": "bfcl",
                        "file_path": small_bfcl_dataset,
                        "test_category": "simple",
                        "structure": {"question_key": "question", "answer_key": "answer"},
                    },
                },
                "evaluators": {
                    "bfcl": {
                        "_type": "bfcl_evaluator",
                        "test_category": "simple",
                        "language": "Python",
                    },
                },
            },
        }

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config_dict, default_flow_style=False))

        result = subprocess.run(
            ["nat", "eval", "--config_file", str(config_path)],
            capture_output=True, text=True, timeout=240,
            env={**os.environ, "NVIDIA_API_KEY": os.environ["NVIDIA_API_KEY"]},
        )

        assert result.returncode == 0, f"nat eval failed:\n{result.stderr[-2000:]}"

        eval_output = os.path.join(output_dir, "bfcl_output.json")
        assert os.path.isfile(eval_output), f"Missing bfcl_output.json in {output_dir}"

        with open(eval_output) as f:
            eval_data = json.load(f)

        assert "average_score" in eval_data
        assert len(eval_data["eval_output_items"]) == 5

        # With a 70B model on simple tasks, we expect at least some correct
        accuracy = eval_data["average_score"]
        assert accuracy > 0, f"Expected accuracy > 0 on simple tasks, got {accuracy}"

    @pytest.mark.asyncio
    async def test_end_to_end_bfcl_react_eval(self, small_bfcl_dataset, tmp_path):
        """Full e2e: BFCL ReAct workflow → bind_tools loop → intent capture → ast_checker."""
        import yaml
        output_dir = str(tmp_path / "react_output")

        config_dict = {
            "llms": {
                "nim_llm": {
                    "_type": "nim",
                    "model_name": "meta/llama-3.3-70b-instruct",
                    "api_key": os.environ["NVIDIA_API_KEY"],
                    "base_url": os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                    "max_tokens": 512,
                    "temperature": 0.0,
                },
            },
            "workflow": {
                "_type": "bfcl_react_workflow",
                "llm_name": "nim_llm",
                "max_steps": 3,
            },
            "eval": {
                "general": {
                    "output_dir": output_dir,
                    "workflow_alias": "bfcl_react_integration_test",
                    "per_input_user_id": False,
                    "max_concurrency": 3,
                    "dataset": {
                        "_type": "bfcl",
                        "file_path": small_bfcl_dataset,
                        "test_category": "simple",
                        "structure": {"question_key": "question", "answer_key": "answer"},
                    },
                },
                "evaluators": {
                    "bfcl": {
                        "_type": "bfcl_evaluator",
                        "test_category": "simple",
                        "language": "Python",
                    },
                },
            },
        }

        config_path = tmp_path / "react_config.yaml"
        config_path.write_text(yaml.dump(config_dict, default_flow_style=False))

        result = subprocess.run(
            ["nat", "eval", "--config_file", str(config_path)],
            capture_output=True, text=True, timeout=240,
            env={**os.environ, "NVIDIA_API_KEY": os.environ["NVIDIA_API_KEY"]},
        )

        assert result.returncode == 0, f"nat eval failed:\n{result.stderr[-2000:]}"

        eval_output = os.path.join(output_dir, "bfcl_output.json")
        assert os.path.isfile(eval_output), f"Missing bfcl_output.json in {output_dir}"

        with open(eval_output) as f:
            eval_data = json.load(f)

        assert "average_score" in eval_data
        assert len(eval_data["eval_output_items"]) == 5

        # ReAct should also get at least some correct on simple tasks
        accuracy = eval_data["average_score"]
        assert accuracy > 0, f"Expected accuracy > 0 on simple ReAct tasks, got {accuracy}"
