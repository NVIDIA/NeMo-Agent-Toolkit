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

import os
import typing
from unittest import mock

import networkx as nx
import pytest
from pydantic import BaseModel

from aiq.builder.builder import Builder
from aiq.builder.component_utils import ComponentInstanceData
from aiq.builder.component_utils import _component_group_order
from aiq.builder.component_utils import build_dependency_sequence
from aiq.builder.component_utils import config_to_dependency_objects
from aiq.builder.component_utils import group_from_component
from aiq.builder.component_utils import iterate_leaf_to_root
from aiq.builder.component_utils import recursive_componentref_discovery
from aiq.builder.component_utils import update_dependency_graph
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.cli.register_workflow import register_function
from aiq.data_models.component import ComponentGroup
from aiq.data_models.component_ref import ComponentRefNode
from aiq.data_models.component_ref import EmbedderRef
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.component_ref import MemoryRef
from aiq.data_models.component_ref import RetrieverRef
from aiq.data_models.component_ref import generate_instance_id
from aiq.data_models.config import AIQConfig
from aiq.data_models.embedder import EmbedderBaseConfig
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.llm import LLMBaseConfig
from aiq.data_models.memory import MemoryBaseConfig
from aiq.data_models.retriever import RetrieverBaseConfig
from aiq.embedder.nim_embedder import NIMEmbedderModelConfig
from aiq.llm.nim_llm import NIMModelConfig
from aiq.retriever.nemo_retriever.register import NemoRetrieverConfig
from aiq.runtime.session import AIQSessionManager
from aiq.test.memory import DummyMemoryConfig


@pytest.fixture(name="nested_aiq_config", scope="function")
def nested_aiq_config_fixture():

    # Setup nested AIQ Toolkit config
    class FnConfig(FunctionBaseConfig, name="test_fn"):
        llm_name: LLMRef
        embedder_name: EmbedderRef
        retriever_name: RetrieverRef | None = None
        memory_name: MemoryRef | None = None
        fn_names: list[FunctionRef] = []

    @register_function(FnConfig)
    async def outer_fn(config: FnConfig, builder: Builder):

        if config.llm_name is not None:
            await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        if config.embedder_name is not None:
            await builder.get_embedder(embedder_name=config.embedder_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        if config.retriever_name is not None:
            await builder.get_retriever(retriever_name=config.retriever_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

        for fn_name in config.fn_names:
            builder.get_function(name=fn_name)

        async def _inner_func(fn_input: str) -> str:
            return ""

        yield _inner_func

    class NnrefConfig(FunctionBaseConfig, name="noref"):
        pass

    @register_function(NnrefConfig)
    async def noref_outer_fn(config: NnrefConfig, builder: Builder):

        async def _inner_func(fn_input: str) -> str:
            return ""

        yield _inner_func

    nested_fns_config = {
        "leaf_fn0":
            FnConfig(llm_name="llm0", embedder_name="embedder0", retriever_name="retriever0"),  # type: ignore
        "leaf_fn1":
            FnConfig(llm_name="llm0", embedder_name="embedder0", retriever_name="retriever0"),  # type: ignore
        "leaf_fn2":
            NnrefConfig(),
        "nested_fn0":
            FnConfig(
                llm_name="llm0",  # type: ignore
                embedder_name="embedder0",  # type: ignore
                fn_names=[
                    "leaf_fn0",  # type: ignore
                    "nested_fn1"
                ]),  # type: ignore
        "leaf_fn3":
            NnrefConfig(),
        "nested_fn1":
            FnConfig(llm_name="llm0", embedder_name="embedder0", fn_names=["leaf_fn0"]),  # type: ignore
        "leaf_fn4":
            NnrefConfig()
    }

    nested_embedders_config = {"embedder0": NIMEmbedderModelConfig(model_name="")}
    nested_llms_config = {"llm0": NIMModelConfig(model_name="")}
    nested_retrievers_config = {"retriever0": NemoRetrieverConfig(uri="http://retriever.com")}  # type: ignore
    nested_memorys_config = {"memory0": DummyMemoryConfig()}
    nested_workflow_config = FnConfig(
        llm_name=LLMRef("llm0"),
        embedder_name="embedder0",  # type: ignore
        fn_names=["leaf_fn0", "nested_fn1"])  # type: ignore

    config = {
        "functions": nested_fns_config,
        "embedders": nested_embedders_config,
        "llms": nested_llms_config,
        "retrievers": nested_retrievers_config,
        "memory": nested_memorys_config,
        "workflow": nested_workflow_config
    }

    aiq_config = AIQConfig.model_validate(config)

    return aiq_config


@pytest.fixture(name="mock_env_vars", scope="module", autouse=True)
def mock_env_vars_fixture():
    with mock.patch.dict(os.environ, {"MEM0_API_KEY": "test-api-key"}):
        yield


def test_iterate_to_root():

    expected = ['D', 'E', 'B', 'C', 'A']
    graph = nx.DiGraph()
    graph.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D'), ('C', 'E')])

    result = []
    for node in iterate_leaf_to_root(graph.copy()):  # type: ignore
        result.append(node)

    # Checking for the correct leaf to root tree traversal
    assert result == expected


def test_group_from_component():

    test_component_config_group_map = {
        EmbedderBaseConfig: ComponentGroup.EMBEDDERS,
        FunctionBaseConfig: ComponentGroup.FUNCTIONS,
        LLMBaseConfig: ComponentGroup.LLMS,
        MemoryBaseConfig: ComponentGroup.MEMORY,
        RetrieverBaseConfig: ComponentGroup.RETRIEVERS
    }

    for TestBaseConfig, test_component_group in test_component_config_group_map.items():

        class ComponentConfig(TestBaseConfig, name="test"):  # type: ignore # pylint: disable=too-many-ancestors
            pass

        component_instance = ComponentConfig()

        # Check for the appropriate component group
        assert group_from_component(component_instance) == test_component_group

    class BadComponentConfig:  # type: ignore
        pass

    bad_component_instance = BadComponentConfig()

    # Not affiliated with a ComponentGroup so should return None
    assert group_from_component(bad_component_instance) is None  # type: ignore


def test_component_group_order():

    component_group_order_set = set(_component_group_order)
    component_groups_set = set(member for member in ComponentGroup)

    # Validate _component_group_order has fully coverage of the ComponentGroup enum
    assert len(component_group_order_set.difference(component_groups_set)) == 0


def test_recursive_componentref_discovery():

    # Setup testing objects
    expected_result = set((
        ComponentRefNode(ref_name="llm0", component_group=ComponentGroup.LLMS),  # type: ignore
        ComponentRefNode(ref_name="function0", component_group=ComponentGroup.FUNCTIONS),  # type: ignore
        ComponentRefNode(ref_name="function1", component_group=ComponentGroup.FUNCTIONS),  # type: ignore
        ComponentRefNode(ref_name="embedder0", component_group=ComponentGroup.EMBEDDERS),  # type: ignore
        ComponentRefNode(ref_name="retriever0", component_group=ComponentGroup.RETRIEVERS)))  # type: ignore

    # Validate across each base component type class
    base_config_types = [FunctionBaseConfig, LLMBaseConfig, EmbedderBaseConfig, MemoryBaseConfig, RetrieverBaseConfig]

    for base_config_type in base_config_types:

        class NestedFns(BaseModel):
            tool_names: list[FunctionRef]

        class MemoryTypedDict(typing.TypedDict):
            memory: MemoryRef

        # Not testing tuple or set based types due to limited Pydantic support
        class TestConfig(base_config_type):  # type: ignore # pylint: disable=too-many-ancestors
            llm: LLMRef
            function_from_model: NestedFns
            embedders_dict: dict[str, EmbedderRef]
            retrievers_list: list[RetrieverRef]
            memory_typed_dict: MemoryTypedDict
            function_union: FunctionRef | None = None

        instance_config = TestConfig(
            llm="llm0",
            function_from_model=NestedFns(tool_names=["function0", "function1"]),  # type: ignore
            embedders_dict={"embeder_key": "embedder0"},
            retrievers_list=["retriever0"],
            memory_typed_dict=MemoryTypedDict(memory="memory0"))  # type: ignore

        expected_instance_id = generate_instance_id(instance_config)

        result_set = set()
        for field_name, field_info in instance_config.model_fields.items():

            for instance_id, value_node in recursive_componentref_discovery(
                    instance_config,
                    getattr(instance_config, field_name),
                    field_info.annotation):  # type: ignore

                # Instance ID should match deep within recursion
                assert instance_id == expected_instance_id

                result_set.add(value_node)

        # Validate discovery of the expected ComponentRef types
        assert len(result_set.difference(expected_result)) == 0


def test_update_dependency_graph(nested_aiq_config: AIQConfig):

    dependency_graph = nx.DiGraph()

    assert len(dependency_graph.nodes) == 0

    # Test adding an unused leaf
    dependency_graph = update_dependency_graph(nested_aiq_config, nested_aiq_config.llms["llm0"], dependency_graph)

    assert len(dependency_graph.nodes) == 0

    # Add a function that depends on leaf nodes (llm/embedder/retriever)
    dependency_graph = update_dependency_graph(nested_aiq_config,
                                               nested_aiq_config.functions["leaf_fn0"],
                                               dependency_graph)

    assert len(dependency_graph.nodes) == 7
    assert dependency_graph.out_degree(generate_instance_id(nested_aiq_config.functions["leaf_fn0"])) == 3
    assert dependency_graph.out_degree(generate_instance_id(nested_aiq_config.llms["llm0"])) == 0
    assert dependency_graph.out_degree(generate_instance_id(nested_aiq_config.embedders["embedder0"])) == 0
    assert dependency_graph.out_degree(generate_instance_id(nested_aiq_config.retrievers["retriever0"])) == 0

    # Add a function that depends on other components (leaf and non-leaf nodes)
    dependency_graph = update_dependency_graph(nested_aiq_config,
                                               nested_aiq_config.functions["nested_fn0"],
                                               dependency_graph)

    assert dependency_graph.out_degree(generate_instance_id(nested_aiq_config.functions["leaf_fn0"])) == 3
    assert dependency_graph.out_degree(generate_instance_id(nested_aiq_config.llms["llm0"])) == 0
    assert dependency_graph.out_degree(generate_instance_id(nested_aiq_config.embedders["embedder0"])) == 0
    assert dependency_graph.out_degree(generate_instance_id(nested_aiq_config.retrievers["retriever0"])) == 0
    assert dependency_graph.out_degree(generate_instance_id(nested_aiq_config.functions["nested_fn0"])) == 4


def test_config_to_dependency_objects(nested_aiq_config: AIQConfig):

    # Setup some expected output
    functions_set = set(str(id(value)) for value in nested_aiq_config.functions.values())
    embedders_set = set(str(id(value)) for value in nested_aiq_config.embedders.values())
    llms_set = set(str(id(value)) for value in nested_aiq_config.llms.values())
    retrievers_set = set(str(id(value)) for value in nested_aiq_config.retrievers.values())
    memory_set = set(str(id(value)) for value in nested_aiq_config.memory.values())
    expected_instance_ids = functions_set | embedders_set | llms_set | retrievers_set | memory_set
    expected_instance_ids.add(str(id(nested_aiq_config.workflow)))

    dependency_map, dependency_graph = config_to_dependency_objects(nested_aiq_config)

    # Validate dependency object types
    assert isinstance(dependency_map, dict)
    assert isinstance(dependency_graph, nx.DiGraph)
    assert len(dependency_map) == 12

    # Check for valid dependency map entries
    for instance_id, component_instance_data in dependency_map.items():
        assert isinstance(instance_id, str)
        assert isinstance(component_instance_data, ComponentInstanceData)
        assert instance_id == component_instance_data.instance_id
        assert instance_id in expected_instance_ids

    # Check for valid graph nodes
    for node in dependency_graph.nodes:
        if isinstance(node, str):
            assert node in expected_instance_ids
        else:
            assert node.ref_name in getattr(nested_aiq_config, node.component_group.value)


def test_build_dependency_sequence(nested_aiq_config: AIQConfig):

    # Setup expected outputs
    expected_dependency_sequence = [
        {
            "component_group": ComponentGroup.MEMORY, "name": "memory0", "is_root": False
        },
        {
            "component_group": ComponentGroup.FUNCTIONS, "name": "leaf_fn2", "is_root": False
        },
        {
            "component_group": ComponentGroup.FUNCTIONS, "name": "leaf_fn3", "is_root": False
        },
        {
            "component_group": ComponentGroup.FUNCTIONS, "name": "leaf_fn4", "is_root": False
        },
        {
            "component_group": ComponentGroup.LLMS, "name": "llm0", "is_root": False
        },
        {
            "component_group": ComponentGroup.EMBEDDERS, "name": "embedder0", "is_root": False
        },
        {
            "component_group": ComponentGroup.RETRIEVERS, "name": "retriever0", "is_root": False
        },
        {
            "component_group": ComponentGroup.FUNCTIONS, "name": "leaf_fn0", "is_root": False
        },
        {
            "component_group": ComponentGroup.FUNCTIONS, "name": "leaf_fn1", "is_root": False
        },
        {
            "component_group": ComponentGroup.FUNCTIONS, "name": "nested_fn1", "is_root": False
        },
        {
            "component_group": ComponentGroup.FUNCTIONS, "name": "nested_fn0", "is_root": False
        },
        {
            "component_group": ComponentGroup.FUNCTIONS, "name": "<workflow>", "is_root": True
        },
    ]

    noref_order = {
        generate_instance_id(nested_aiq_config.memory["memory0"]): -1,
        generate_instance_id(nested_aiq_config.functions["leaf_fn2"]): -1,
        generate_instance_id(nested_aiq_config.functions["leaf_fn3"]): -1,
        generate_instance_id(nested_aiq_config.functions["leaf_fn4"]): -1,
    }

    dependency_sequence = build_dependency_sequence(nested_aiq_config)

    # Validate correct length of dependency sequence
    assert len(dependency_sequence) == len(expected_dependency_sequence)

    for idx, (component_instance_data,
              expected_instance_data) in enumerate(zip(dependency_sequence, expected_dependency_sequence)):

        # Each element in sequence must be a ComponentInstanceData
        assert isinstance(component_instance_data, ComponentInstanceData)
        # Validate attributes and position
        assert component_instance_data.component_group == expected_instance_data["component_group"]
        assert component_instance_data.name == expected_instance_data["name"]
        assert component_instance_data.is_root == expected_instance_data["is_root"]

        if component_instance_data.instance_id in noref_order:
            noref_order[component_instance_data.instance_id] = idx

    # Check all norefs included in sequence
    assert min(noref_order.values()) >= 0

    # Check order of norefs in sequence
    noref_order_index_list = list(noref_order.values())

    assert (all(noref_order_index_list[i] <= noref_order_index_list[i + 1]
                for i in range(len(noref_order_index_list) - 1)))

    # Check exact order of norefs in sequence
    noref_instance_ids = [
        component_instance_data.instance_id for component_instance_data in dependency_sequence[:len(noref_order)]
    ]

    assert noref_instance_ids == list(noref_order.keys())


async def test_load_hierarchial_workflow(nested_aiq_config: AIQConfig):

    # Validate nested workflow instantiation
    async with WorkflowBuilder.from_config(config=nested_aiq_config) as workflow:
        assert AIQSessionManager(workflow.build(), max_concurrency=1)
