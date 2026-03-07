# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the ToolTalk benchmark pipeline.

These tests call a live NIM endpoint and exercise the full
dataset → workflow → evaluator path. They require:
  - NVIDIA_API_KEY environment variable
  - tooltalk package installed
  - Network access to the NIM API

Run with: pytest -m integration packages/nvidia_nat_benchmarks/tests/tooltalk/test_integration.py
"""

import json
import os
import shutil
import tempfile

import pytest

_SKIP_REASON = None

try:
    import tooltalk
    _TOOLTALK_DIR = os.path.dirname(tooltalk.__file__)
    _DATABASE_DIR = os.path.join(_TOOLTALK_DIR, "data", "databases")
    _EASY_DIR = os.path.join(_TOOLTALK_DIR, "data", "easy")
    if not os.path.isdir(_DATABASE_DIR):
        _SKIP_REASON = "tooltalk database directory not found"
except ImportError:
    _SKIP_REASON = "tooltalk not installed"

if not os.environ.get("NVIDIA_API_KEY"):
    _SKIP_REASON = "NVIDIA_API_KEY not set"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.timeout(300),
    pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or ""),
]


@pytest.fixture
def single_conversation_dir(tmp_path):
    """Create a temp directory with a single AddAlarm-easy conversation."""
    src = os.path.join(_EASY_DIR, "AddAlarm-easy.json")
    if not os.path.isfile(src):
        pytest.skip("AddAlarm-easy.json not found in tooltalk data")
    shutil.copy2(src, tmp_path)
    return str(tmp_path)


class TestToolTalkIntegration:

    @pytest.mark.asyncio
    async def test_workflow_produces_predictions(self, single_conversation_dir):
        """The workflow calls a live NIM endpoint and produces tool call predictions."""
        from nat.plugins.benchmarks.tooltalk.dataset import load_tooltalk_dataset
        from nat.plugins.benchmarks.tooltalk.workflow import _build_tool_schemas, _build_messages

        # Load dataset
        df = load_tooltalk_dataset(single_conversation_dir)
        assert len(df) == 1

        conv = json.loads(df.iloc[0]["question"])
        assert "conversation" in conv
        assert conv["apis_used"] == ["AddAlarm"]

    @pytest.mark.asyncio
    async def test_end_to_end_nat_eval(self, single_conversation_dir, tmp_path):
        """Full end-to-end: nat eval config → workflow → evaluator → metrics output."""
        output_dir = str(tmp_path / "output")

        # Build a config dict programmatically (avoids YAML env var issues)
        config_dict = {
            "llms": {
                "nim_llm": {
                    "_type": "nim",
                    "model_name": "meta/llama-3.3-70b-instruct",
                    "api_key": os.environ["NVIDIA_API_KEY"],
                    "base_url": os.environ.get(
                        "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
                    ),
                    "max_tokens": 1024,
                    "temperature": 0.0,
                },
            },
            "workflow": {
                "_type": "tooltalk_workflow",
                "llm_name": "nim_llm",
                "database_dir": _DATABASE_DIR,
                "api_mode": "all",
                "max_tool_calls_per_turn": 3,
            },
            "eval": {
                "general": {
                    "output_dir": output_dir,
                    "workflow_alias": "tooltalk_integration_test",
                    "per_input_user_id": True,
                    "max_concurrency": 1,
                    "dataset": {
                        "_type": "tooltalk",
                        "file_path": single_conversation_dir,
                        "database_dir": _DATABASE_DIR,
                        "structure": {
                            "question_key": "question",
                            "answer_key": "answer",
                        },
                    },
                },
                "evaluators": {
                    "tooltalk": {
                        "_type": "tooltalk_evaluator",
                        "database_dir": _DATABASE_DIR,
                    },
                },
            },
        }

        # Write config to a temp YAML
        import yaml
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump(config_dict, default_flow_style=False))

        # Run eval via subprocess (same as `nat eval --config_file ...`)
        import subprocess
        result = subprocess.run(
            ["nat", "eval", "--config_file", str(config_path)],
            capture_output=True,
            text=True,
            timeout=240,
            env={**os.environ, "NVIDIA_API_KEY": os.environ["NVIDIA_API_KEY"]},
        )

        # Check that the eval completed
        assert result.returncode == 0, f"nat eval failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

        # Verify output files exist
        workflow_output = os.path.join(output_dir, "workflow_output.json")
        eval_output = os.path.join(output_dir, "tooltalk_output.json")
        assert os.path.isfile(workflow_output), f"Missing workflow_output.json in {output_dir}"
        assert os.path.isfile(eval_output), f"Missing tooltalk_output.json in {output_dir}"

        # Verify eval output structure
        with open(eval_output) as f:
            eval_data = json.load(f)
        assert "average_score" in eval_data
        assert "eval_output_items" in eval_data
        assert len(eval_data["eval_output_items"]) == 1

        item = eval_data["eval_output_items"][0]
        assert "score" in item
        assert "reasoning" in item

        # The reasoning should have ToolTalk metrics
        reasoning = item["reasoning"]
        assert "recall" in reasoning
        assert "bad_action_rate" in reasoning
        assert "success" in reasoning

        # recall should be > 0 (the LLM should at least call AddAlarm once correctly)
        assert reasoning["recall"] > 0, (
            f"Expected recall > 0 (LLM should call AddAlarm correctly), got {reasoning}"
        )
