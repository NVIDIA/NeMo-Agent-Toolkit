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

import pytest


@pytest.mark.parametrize(
    "question, expected_answer",
    [("What yellow fruit would you recommend?", "banana"), ("I want a red fruit, what do you suggest?", "apple"),
     ("Can you recommend a green fruit?", "pear"), ("What city would you recommend in the United States?", "new york"),
     ("Which city should I visit in the United Kingdom?", "london"),
     ("What's a good city to visit in Canada?", "toronto"), ("Recommend a city in Australia", "sydney"),
     ("What city should I visit in India?", "mumbai"),
     ("What literature work by Shakespeare would you recommend?", "hamlet"),
     ("Can you suggest a work by Dante?", "the divine comedy"),
     ("What's a good literature piece by Milton?", "paradise lost")])
@pytest.mark.usefixtures("nvidia_api_key")
@pytest.mark.integration
async def test_full_workflow(question: str, expected_answer: str) -> None:

    from nat.test.utils import locate_example_config
    from nat.test.utils import run_workflow
    from nat_router_agent.register import MockFruitAdvisorFunctionConfig

    config_file = locate_example_config(MockFruitAdvisorFunctionConfig)
    await run_workflow(config_file=config_file, question=question, expected_answer=expected_answer)
