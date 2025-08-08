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

from nat.data_models.component import ComponentGroup
from nat.data_models.component_ref import ComponentRef
from nat.data_models.component_ref import EmbedderRef
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.component_ref import MemoryRef
from nat.data_models.component_ref import ObjectStoreRef
from nat.data_models.component_ref import RetrieverRef
from nat.data_models.component_ref import generate_instance_id
from nat.data_models.embedder import EmbedderBaseConfig
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.memory import MemoryBaseConfig
from nat.data_models.object_store import ObjectStoreBaseConfig
from nat.data_models.retriever import RetrieverBaseConfig


def test_generate_instance_id():

    test_base_configs = [
        FunctionBaseConfig,
        LLMBaseConfig,
        EmbedderBaseConfig,
        MemoryBaseConfig,
        ObjectStoreBaseConfig,
        RetrieverBaseConfig
    ]

    # Validate instance id generation for each component type that maps to a ComponentGroup
    for name, config_base in enumerate(test_base_configs):

        class TestConfig(config_base, name=str(name)):  # type: ignore
            pass

        test_config = TestConfig()

        assert str(id(test_config)) == generate_instance_id(test_config)


def test_component_ref_type_checks():

    test_component_ref_group_map = {
        FunctionRef: ComponentGroup.FUNCTIONS,
        LLMRef: ComponentGroup.LLMS,
        EmbedderRef: ComponentGroup.EMBEDDERS,
        MemoryRef: ComponentGroup.MEMORY,
        ObjectStoreRef: ComponentGroup.OBJECT_STORES,
        RetrieverRef: ComponentGroup.RETRIEVERS
    }

    # Validate ComponentRef type instantation and properties
    for RefType, component_group in test_component_ref_group_map.items():
        function_ref = RefType("function_name")

        assert isinstance(function_ref, RefType)
        assert function_ref.component_group == component_group
        assert issubclass(type(function_ref), ComponentRef)
        assert issubclass(type(function_ref), str)


def test_component_ref_pydantic_validation():

    test_config_map = {
        FunctionBaseConfig: FunctionRef,
        LLMBaseConfig: LLMRef,
        EmbedderBaseConfig: EmbedderRef,
        MemoryBaseConfig: MemoryRef,
        ObjectStoreBaseConfig: ObjectStoreRef,
        RetrieverBaseConfig: RetrieverRef
    }

    # Validate configuration object instantiation with ComponentRef types
    for test_base_config, test_ref_type in test_config_map.items():

        class TestConfig(test_base_config, name="test"):  # type: ignore # pylint: disable=too-many-ancestors
            ref_field: test_ref_type  # type: ignore

        config_dict = {"ref_field": "ref_value"}

        validated_model = TestConfig.model_validate(config_dict)

        assert isinstance(validated_model, TestConfig)


def test_component_ref_interface():

    class TestRefType(ComponentRef):

        @property
        def component_group(self) -> ComponentGroup:
            return ComponentGroup.FUNCTIONS

    test_ref = TestRefType("")

    # Validate ComponentRef inheritance
    assert issubclass(TestRefType, ComponentRef)
    assert isinstance(test_ref.component_group, ComponentGroup)

    # Validate abstactmethod enforcement for component_group property
    class BadRefType(ComponentRef):
        pass

    # Should fail
    with pytest.raises(TypeError):
        _ = BadRefType("")  # type: ignore # pylint: disable=abstract-class-instantiated
