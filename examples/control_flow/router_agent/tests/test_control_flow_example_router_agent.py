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


async def _test_workflow(config_file: str, question: str, answer: str):
    async with load_workflow(config_file) as workflow:

        async with workflow.run(question) as runner:
            result = await runner.result(to_type=str)

        result = result.lower()
        assert answer in result


@pytest.mark.usefixtures("nvidia_api_key")
@pytest.mark.integration
async def test_full_workflow():

    current_dir = os.path.dirname(os.path.abspath(__file__))

    config_file = os.path.join(current_dir, "../configs", "config.yml")

    test_cases = [{
        "question": "What yellow fruit would you recommend?", "answer": "banana"
    }, {
        "question": "I want a red fruit, what do you suggest?", "answer": "apple"
    }, {
        "question": "Can you recommend a green fruit?", "answer": "pear"
    }, {
        "question": "What city would you recommend in the United States?", "answer": "new york"
    }, {
        "question": "Which city should I visit in the United Kingdom?", "answer": "london"
    }, {
        "question": "What's a good city to visit in Canada?", "answer": "toronto"
    }, {
        "question": "Recommend a city in Australia", "answer": "sydney"
    }, {
        "question": "What city should I visit in India?", "answer": "mumbai"
    }, {
        "question": "What literature work by Shakespeare would you recommend?", "answer": "hamlet"
    }, {
        "question": "Can you suggest a work by Dante?", "answer": "the divine comedy"
    }, {
        "question": "What's a good literature piece by Milton?", "answer": "paradise lost"
    }]

    for test_case in test_cases:
        await _test_workflow(config_file, test_case["question"], test_case["answer"])
