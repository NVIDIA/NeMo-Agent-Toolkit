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

import os
import urllib.request
from pathlib import Path

import pytest


@pytest.fixture(name="opensearch_url", scope="session")
def opensearch_url_fixture(fail_missing: bool) -> str:
    url = os.getenv("NAT_CI_OPENSEARCH_URL", "http://localhost:9200")
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/_cluster/health", timeout=1) as resp:
            return 200 <= getattr(resp, "status", 0) < 300
    except Exception:
        failure_reason = f"Unable to connect to open search server at {url}"
        if fail_missing:
            raise RuntimeError(failure_reason)
        pytest.skip(reason=failure_reason)


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "serperdev")
async def test_full_workflow_e2e(opensearch_url: str) -> None:

    from nat.runtime.loader import load_config
    from nat.test.utils import run_workflow

    config_file = (Path(__file__).resolve().parents[1] / "src" / "nat_haystack_deep_research_agent" / "configs" /
                   "config.yml")

    config = load_config(config_file)
    config.workflow.opensearch_url = opensearch_url

    result = await run_workflow(question="Give a short overview of this workflow.",
                                expected_answer="workflow",
                                config=config)

    assert isinstance(result, str)
    assert len(result) > 0


def test_config_yaml_loads_and_has_keys() -> None:
    config_file = (Path(__file__).resolve().parents[1] / "configs" / "config.yml")

    with open(config_file, encoding="utf-8") as f:
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
