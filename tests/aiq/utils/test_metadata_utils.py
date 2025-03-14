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
from pydantic import Field

from aiq.data_models.common import TypedBaseModel
from aiq.data_models.common import TypedBaseModelT
from aiq.data_models.embedder import EmbedderBaseConfig
from aiq.data_models.evaluate import EvaluatorBaseConfig
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.llm import LLMBaseConfig
from aiq.data_models.memory import MemoryBaseConfig
from aiq.data_models.registry_handler import RegistryHandlerBaseConfig
from aiq.data_models.retriever import RetrieverBaseConfig
from aiq.utils.metadata_utils import generate_config_type_docs


@pytest.fixture(name="base_configs", scope="function", autouse=True)
def base_configs_fixture():

    base_configs = [
        TypedBaseModel,
        FunctionBaseConfig,
        LLMBaseConfig,
        EmbedderBaseConfig,
        RegistryHandlerBaseConfig,
        RetrieverBaseConfig,
        MemoryBaseConfig,
        EvaluatorBaseConfig
    ]

    return base_configs


def test_generate_config_type_docs_no_docstring(base_configs: list[TypedBaseModelT]):

    expected = ("Description unavailable.\n"
                "\n"
                "  Args:\n"
                "    _type (str): The type of the object.\n"
                "    field0 (str): description0.\n"
                "    field1 (str): description1. Defaults to \"value1\".\n"
                "    field2 (str | None): description2.\n"
                "    field3 (str | None): description3. Defaults to None.\n"
                "    field4 (str | dict[str, str]): description4.\n"
                "    field5 (str | dict[str, int]): description5. Defaults to {'key5': 0}.")

    for base_config in base_configs:

        class TestConfig(base_config, name="test"):  # type: ignore
            field0: str = Field(description="description0")
            field1: str = Field(default="value1", description="description1")
            field2: str | None = Field(description="description2")
            field3: str | None = Field(default=None, description="description3")
            field4: str | dict[str, str] = Field(description="description4")
            field5: str | dict[str, int] = Field(default={"key5": 0}, description="description5")

        assert generate_config_type_docs(TestConfig) == expected


def test_generate_config_type_docs_no_args(base_configs: list[TypedBaseModelT]):

    expected = ("Notional Docstring.\n"
                "\n"
                "  Args:\n"
                "    _type (str): The type of the object.\n"
                "    field0 (str): Description unavailable.\n"
                "    field1 (str): Description unavailable. Defaults to \"value1\".\n"
                "    field2 (str | None): Description unavailable.\n"
                "    field3 (str | None): Description unavailable. Defaults to None.\n"
                "    field4 (str | dict[str, str]): Description unavailable.\n"
                "    field5 (str | dict[str, int]): Description unavailable. Defaults to {'key5': 0}.")

    for base_config in base_configs:

        class TestConfig(base_config, name="test"):  # type: ignore
            """Notional Docstring."""

            field0: str
            field1: str = "value1"
            field2: str | None
            field3: str | None = None
            field4: str | dict[str, str]
            field5: str | dict[str, int] = {"key5": 0}

        assert generate_config_type_docs(TestConfig) == expected


def test_generate_config_type_docs_no_docstring_and_no_args(base_configs: list[TypedBaseModelT]):

    expected = ("Description unavailable.\n"
                "\n"
                "  Args:\n"
                "    _type (str): The type of the object.\n"
                "    field0 (str): Description unavailable.\n"
                "    field1 (str): Description unavailable. Defaults to \"value1\".\n"
                "    field2 (str | None): Description unavailable.\n"
                "    field3 (str | None): Description unavailable. Defaults to None.\n"
                "    field4 (str | dict[str, str]): Description unavailable.\n"
                "    field5 (str | dict[str, int]): Description unavailable. Defaults to {'key5': 0}.")

    for base_config in base_configs:

        class TestConfig(base_config, name="test"):  # type: ignore

            field0: str
            field1: str = "value1"
            field2: str | None
            field3: str | None = None
            field4: str | dict[str, str]
            field5: str | dict[str, int] = {"key5": 0}

        assert generate_config_type_docs(TestConfig) == expected
