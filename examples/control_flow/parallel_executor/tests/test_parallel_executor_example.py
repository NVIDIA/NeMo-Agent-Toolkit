# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "question, expected_elements",
    [
        ("Prepare a launch update for the new mobile feature next week.", [
            "parallel analysis report",
            "topic:",
            "urgency:",
            "risk:",
            "action:",
        ]),
        ("We have an urgent production incident and need an immediate response plan.", [
            "parallel analysis report",
            "topic:",
            "urgency:",
            "risk:",
            "action:",
        ]),
    ],
)
@pytest.mark.usefixtures("nvidia_api_key")
@pytest.mark.integration
async def test_parallel_executor_workflow(question: str, expected_elements: list[str]) -> None:
    from nat.test.utils import run_workflow

    config_file = Path(__file__).resolve().parents[1] / "configs" / "config.yml"
    assert config_file.exists(), f"Expected config file not found: {config_file}"
    result = await run_workflow(
        config_file=config_file,
        question=question,
        expected_answer="",
        assert_expected_answer=False,
    )
    result = result.lower()

    for element in expected_elements:
        assert element in result
