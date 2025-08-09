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

import typing

import pytest

from aiq.tool.retriever import RetrieverConfig


@pytest.mark.parametrize("config_values",
                         [
                             {
                                 "retriever": "test_retriever",
                                 "raise_errors": False,
                                 "topic": "test_topic",
                                 "description": "test_description"
                             },
                             {
                                 "retriever": "test_retriever",
                             },
                         ],
                         ids=[
                             "all_fields_provided",
                             "only_required_fields",
                         ])
def test_retriever_config(config_values: dict[str, typing.Any]):
    """
    Test the AIQRetrieverConfig class.
    """

    RetrieverConfig.model_validate(config_values, strict=True)
    config = RetrieverConfig(**config_values)

    model_dump = config.model_dump()
    model_dump.pop('type')

    RetrieverConfig.model_validate(model_dump, strict=True)
