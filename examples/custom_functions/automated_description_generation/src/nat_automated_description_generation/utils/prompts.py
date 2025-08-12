# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from langchain_core.prompts import ChatPromptTemplate

# pylint: disable=invalid-name
map_prompt_system = """Analyze these snippets from a data collection:
                            --------------
                            {documents}
                            --------------
                            Extract and summarize:
                            1. The main subject matter or topic.
                            2. Key technical terms or concepts.
                            3. The type of information being stored.      # noqa             
                            Provide a brief, factual summary focusing on these elements."""

direct_summary_template = """Below are retrieved content samples from a data collection:
                                --------------
                                {documents}
                                --------------
                                Create a single, comprehensive sentence that:
                                    1. Describes the general nature of the collection.
                                    2. Captures the primary type of data stored.
                                    3. Indicates the collection's apparent purpose.
                                """

reduce_template = """Below are summaries describing content samples from a data collection:
                            --------------
                            {documents}
                            --------------
                            Create a single, comprehensive sentence that:
                                1. Describes the general nature of the collection.
                                2. Captures the primary type of data stored.
                                3. Indicates the collection's apparent purpose.
                            """
reduce_prompt = ChatPromptTemplate([("human", reduce_template)])

direct_summary_prompt = ChatPromptTemplate([("human", direct_summary_template)])

map_prompt = ChatPromptTemplate.from_messages([("system", map_prompt_system)])
