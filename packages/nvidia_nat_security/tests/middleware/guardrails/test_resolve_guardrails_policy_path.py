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

from __future__ import annotations

from pathlib import Path

import pytest

from nat.plugins.security.middleware.guardrails.nemo_guardrails_middleware_config import GuardrailsMiddlewareConfig


def _inline_verifier_guardrails() -> dict[str, object]:
    """Minimal inline verifier policy matching the retail example."""
    return {
        "models": [],
        "colang_version":
            "1.0",
        "rails": {
            "input": {
                "flows": ["self check input"]
            },
            "output": {
                "flows": ["self check output"]
            },
        },
        "prompts": [
            {
                "task": "self_check_input", "content": "User: {{ user_input }}\nAnswer:", "max_tokens": 3
            },
            {
                "task": "self_check_output",
                "content": "User: {{ user_input }}\nBot: {{ bot_response }}\nAnswer:",
                "max_tokens": 3
            },
        ],
    }


def test_verifier_policy_loads_self_check_rails() -> None:
    """Inline verifier policy enables self check input and output flows with retail prompts."""
    config = GuardrailsMiddlewareConfig(
        workflow_functions=["<workflow>", "retail_tools__get_product_info"],
        guardrails=_inline_verifier_guardrails(),
    )
    assert config.guardrails.rails.input.flows == ["self check input"]
    assert config.guardrails.rails.output.flows == ["self check output"]
    task_names: list[str] = [p.task for p in config.guardrails.prompts]
    assert "self_check_input" in task_names
    assert "self_check_output" in task_names
    assert config.guardrails_root is None


def test_guardrails_root_loads_policy_via_from_path(tmp_path: Path) -> None:
    """guardrails_root is passed through to RailsConfig.from_path unchanged."""
    (tmp_path / "config.yml").write_text(
        "models:\n"
        "  - type: content_safety\n    engine: nim\n    model: test-model\n"
        "colang_version: '1.0'\nrails:\n"
        "  input:\n    flows: [content safety check input $model=content_safety]\n"
        "  output:\n    flows: [content safety check output $model=content_safety]\n",
        encoding="utf-8",
    )
    (tmp_path / "prompts.yml").write_text(
        "prompts:\n"
        "  - task: content_safety_check_input $model=content_safety\n"
        "    content: 'user: {{ user_input }}'\n"
        "    max_tokens: 50\n"
        "  - task: content_safety_check_output $model=content_safety\n"
        "    content: 'agent: {{ bot_response }}'\n"
        "    max_tokens: 50\n",
        encoding="utf-8",
    )
    config = GuardrailsMiddlewareConfig(
        workflow_functions=["test_fn"],
        guardrails_root=str(tmp_path),
    )
    assert config.guardrails.rails.input.flows == ["content safety check input $model=content_safety"]
    assert len(config.guardrails.prompts) > 0


def test_guardrails_root_rejects_invalid_path() -> None:
    """Invalid guardrails_root surfaces the library Invalid config path error."""
    with pytest.raises(ValueError, match="Invalid config path"):
        GuardrailsMiddlewareConfig(
            workflow_functions=["test_fn"],
            guardrails_root="not_a_real_policy_directory",
        )


def test_retail_workflow_config_with_guardrails_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full retail guardrails workflow config validates when run from the repository root."""
    import os

    from nat.runtime.loader import load_config

    repo_root: Path = Path(__file__).resolve().parents[5]
    monkeypatch.chdir(repo_root)
    config_path: Path = (
        Path(os.getcwd()) /
        "examples/safety_and_security/retail_agent/src/nat_retail_agent/configs/config-with-guardrails.yml")
    try:
        nat_config = load_config(config_path)
    except ValueError as exc:
        if "retail_tools" in str(exc):
            pytest.skip("retail_agent example must be installed to validate the full workflow config")
        raise
    pii = nat_config.middleware["pii_guardrails"]
    selection = pii.workflow_functions["retail_tools__get_product_info"]
    assert selection.root == {"reviews": ["review"]}

    all_products = nat_config.middleware["all_products_guardrails"]
    all_products_selection = all_products.workflow_functions["retail_tools__get_all_products"]
    assert all_products_selection.root == {"description": [], "review_texts": []}
