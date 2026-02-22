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
    "input_text, expected_elements",
    [("The quick brown fox jumps over the lazy dog. This is a simple test sentence.",
      ["text analysis report", "word count", "sentence count", "complexity"]),
     (("Natural language processing is a fascinating field that combines computational linguistics with "
       "machine learning and artificial intelligence. It enables computers to understand, interpret, "
       "and generate human language in a valuable way."),
      ["text analysis report", "word count", "sentence count", "complexity", "top words"]),
     ("Hello world! This is a test.",
      ["text analysis report", "word count", "sentence count", "report generated successfully"]),
     ("This text has special characters: @#$%^&*()! Let's see how the pipeline handles them.",
      ["text analysis report", "word count", "sentence count", "complexity"]),
     ("Short text.", ["text analysis report", "word count", "sentence count", "report generated successfully"])])
@pytest.mark.integration
async def test_full_workflow(input_text: str, expected_elements: list) -> None:
    from nat.test.utils import locate_example_config
    from nat.test.utils import run_workflow
    from nat_sequential_executor.register import TextProcessorFunctionConfig

    config_file = locate_example_config(TextProcessorFunctionConfig)

    result = await run_workflow(config_file=config_file,
                                question=input_text,
                                expected_answer="",
                                assert_expected_answer=False)
    result = result.lower()
    for element in expected_elements:
        assert element in result
