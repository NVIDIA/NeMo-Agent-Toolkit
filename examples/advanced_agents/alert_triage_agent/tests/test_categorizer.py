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

from nat_alert_triage_agent.categorizer import _extract_markdown_heading_level


@pytest.mark.parametrize(
    "test_input,expected",
    [
        pytest.param("# Title", "#", id="single_hash"),
        pytest.param("### Title", "###", id="multiple_hashes"),
        pytest.param("No heading", "#", id="no_heading_default"),
        pytest.param("", "#", id="empty_string"),
        pytest.param("## My Title\n### Heading", "##", id="first_of_many"),
        pytest.param("Here is a title\n## Title Line", "##", id="first_after_text"),
        pytest.param("## Heading first\n# Title", "##", id="heading_precedence"),
        pytest.param("###No space between # and title", "###", id="no_space_after_hashes"),
    ],
)
def test_extract_markdown_heading_level(test_input, expected):
    assert _extract_markdown_heading_level(test_input) == expected
