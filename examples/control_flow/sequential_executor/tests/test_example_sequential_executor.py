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

import logging
import os

import pytest

from nat.runtime.loader import load_workflow

logger = logging.getLogger(__name__)


async def _test_workflow(config_file: str, input_text: str, expected_elements: list):
    async with load_workflow(config_file) as workflow:

        async with workflow.run(input_text) as runner:
            result = await runner.result(to_type=str)

        result = result.lower()
        for element in expected_elements:
            assert element in result


@pytest.mark.integration
async def test_full_workflow():

    current_dir = os.path.dirname(os.path.abspath(__file__))

    config_file = os.path.join(current_dir, "../configs", "config.yml")

    test_cases = [
        {
            "input_text": "The quick brown fox jumps over the lazy dog. This is a simple test sentence.",
            "expected_elements": ["text analysis report", "word count", "sentence count", "complexity"]
        },
        {
            "input_text":
                "Natural language processing is a fascinating field that combines computational linguistics with "
                "machine learning and artificial intelligence. It enables computers to understand, interpret, "
                "and generate human language in a valuable way.",
            "expected_elements": ["text analysis report", "word count", "sentence count", "complexity", "top words"]
        },
        {
            "input_text":
                "Hello world! This is a test.",
            "expected_elements": [
                "text analysis report", "word count", "sentence count", "report generated successfully"
            ]
        },
        {
            "input_text": "This text has special characters: @#$%^&*()! Let's see how the pipeline handles them.",
            "expected_elements": ["text analysis report", "word count", "sentence count", "complexity"]
        },
        {
            "input_text":
                "Short text.",
            "expected_elements": [
                "text analysis report", "word count", "sentence count", "report generated successfully"
            ]
        }
    ]

    for test_case in test_cases:
        await _test_workflow(config_file, test_case["input_text"], test_case["expected_elements"])
