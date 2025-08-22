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

# pyright: reportMissingTypeStubs=false, reportMissingImports=false

import importlib
import os
import urllib.error
import urllib.request
from pathlib import Path
import pytest


def _opensearch_reachable(url: str) -> bool:
    try:
        with urllib.request.urlopen(
            f"{url.rstrip('/')}/_cluster/health", timeout=1
        ) as resp:
            return 200 <= getattr(resp, "status", 0) < 300
    except Exception:
        return False


@pytest.mark.e2e
@pytest.mark.skipif(
    not os.environ.get("NVIDIA_API_KEY"),
    reason="NVIDIA_API_KEY not set; skipping live e2e test for haystack_deep_research_agent.",
)
@pytest.mark.skipif(
    not os.environ.get("SERPERDEV_API_KEY"),
    reason="SERPERDEV_API_KEY not set; skipping live e2e test for haystack_deep_research_agent.",
)
@pytest.mark.skipif(
    not _opensearch_reachable("http://localhost:9200"),
    reason="OpenSearch not reachable on http://localhost:9200; skipping e2e test.",
)
async def test_full_workflow_e2e() -> None:
    config_file = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "aiq_haystack_deep_research_agent"
        / "configs"
        / "config.yml"
    )

    loader_mod = importlib.import_module("aiq.runtime.loader")
    load_workflow = getattr(loader_mod, "load_workflow")

    async with load_workflow(config_file) as workflow:
        async with workflow.run("Give a short overview of this workflow.") as runner:
            result = await runner.result(to_type=str)

    assert isinstance(result, str)
    assert len(result) > 0


def test_config_yaml_loads_and_has_keys() -> None:
    config_file = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "aiq_haystack_deep_research_agent"
        / "configs"
        / "config.yml"
    )

    with open(config_file, "r", encoding="utf-8") as f:
        text = f.read()

    assert "workflow:" in text
    assert "_type: haystack_deep_research_agent" in text
    # key fields expected
    for key in [
        "llms:",
        "rag_llm:",
        "agent_llm:",
        "workflow:",
        "max_agent_steps:",
        "search_top_k:",
        "rag_top_k:",
        "opensearch_url:",
        "index_on_startup:",
        "data_dir:",
    ]:
        assert key in text, f"Missing key: {key}"
