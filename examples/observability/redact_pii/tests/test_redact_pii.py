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

import time
import types
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest


@pytest.fixture(name="project_name", scope="module")
async def fixture_project_name(weave: types.ModuleType) -> AsyncGenerator[str]:
    # This currently has the following problems:
    # 1. Ideally we would create a new project for each test run to avoid conflicts, and then delete the project.
    #    However, W&B does not currently support deleting projects via the API.
    # 2. We don't have a way (that I know of) to identifiy traces from this specific test run, such that we only delete
    #    those traces.
    project_name = "redact_pii_e2e"
    yield project_name

    client = weave.init(project_name)
    call_ids = [c.id for c in client.get_calls()]
    if len(call_ids) > 0:
        client.delete_calls(call_ids)


@pytest.mark.xfail(reason="Not all PII is being redacted")
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "wandb_api_key")
async def test_agent_full_workflow(examples_dir: Path, project_name: str, weave: types.ModuleType):
    from nat.runtime.loader import load_config
    from nat.test.utils import run_workflow
    from nat_redact_pii.register import PiiTestConfig

    test_email = PiiTestConfig().test_email

    config_file = examples_dir / "observability/redact_pii/configs/weave_redact_pii_config.yml"
    config = load_config(config_file)
    config.general.telemetry.tracing["weave"].project = project_name

    await run_workflow(config=config,
                       question="What is John Doe's contact information?",
                       expected_answer="test@example.com")

    client = weave.init(project_name)
    client.flush()
    calls = client.get_calls()

    found_redacted_value = False
    for call in calls:
        call_str = str(call)
        # This test is currently failing
        assert test_email not in call_str, f"Found unredacted email address in call: {call_str}"
        if not found_redacted_value:
            found_redacted_value = "<EMAIL_ADDRESS>" in call_str

    assert found_redacted_value, "Did not find redacted email address in any call traces"
