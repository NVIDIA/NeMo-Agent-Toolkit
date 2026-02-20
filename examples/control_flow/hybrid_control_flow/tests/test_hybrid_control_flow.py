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

import pytest


@pytest.mark.parametrize(
    "question, expected_answer",
    [
        # Test Pattern 1: Router → Sequential (text analysis pipeline)
        ("Process this text: Hello world from NeMo Agent Toolkit", "text analysis report"),
        ("Analyze the following: Testing sequential executor pipeline", "report generated successfully"),
        # Test Pattern 2: Sequential → Router (text formatting pipeline)
        ("Convert this to uppercase: hello world", "HELLO WORLD"),
        ("Make this lowercase: TESTING", "testing"),
        # Test Pattern 3: Router → Nested Router (fruit advisor)
        ("What yellow fruit would you recommend?", "banana"),
        ("I want a red fruit", "apple"),
        # Test Pattern 3: Router → Nested Router (city advisor)
        ("What city should I visit in Canada?", "toronto"),
        ("Recommend a city in the United Kingdom", "london"),
    ],
)
@pytest.mark.usefixtures("nvidia_api_key")
@pytest.mark.integration
async def test_full_workflow(question: str, expected_answer: str) -> None:
    from nat.test.utils import locate_example_config
    from nat.test.utils import run_workflow
    from nat_hybrid_control_flow.register import MockInputValidatorFunctionConfig

    config_file = locate_example_config(MockInputValidatorFunctionConfig)
    await run_workflow(config_file=config_file, question=question, expected_answer=expected_answer)
