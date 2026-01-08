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

import os
import sys
from unittest import mock

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

import networkx as nx
import pytest
from pydantic import BaseModel

from nat.builder.builder import Builder
from nat.builder.component_utils import ComponentInstanceData
from nat.builder.component_utils import _component_group_order
from nat.builder.component_utils import build_dependency_sequence
from nat.builder.component_utils import config_to_dependency_objects
from nat.builder.component_utils import group_from_component
from nat.builder.component_utils import iterate_leaf_to_root
from nat.builder.component_utils import recursive_componentref_discovery
from nat.builder.component_utils import update_dependency_graph
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.component import ComponentGroup
from nat.data_models.component_ref import ComponentRefNode
from nat.data_models.component_ref import EmbedderRef
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.component_ref import MemoryRef
from nat.data_models.component_ref import ObjectStoreRef
from nat.data_models.component_ref import RetrieverRef
from nat.data_models.component_ref import TrainerAdapterRef
from nat.data_models.component_ref import TrainerRef
from nat.data_models.component_ref import TrajectoryBuilderRef
from nat.data_models.component_ref import generate_instance_id
from nat.data_models.config import Config
from nat.data_models.embedder import EmbedderBaseConfig
from nat.data_models.finetuning import TrainerAdapterConfig
from nat.data_models.finetuning import TrainerConfig
from nat.data_models.finetuning import TrajectoryBuilderConfig
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.function import FunctionGroupBaseConfig
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.memory import MemoryBaseConfig
from nat.data_models.middleware import MiddlewareBaseConfig
from nat.data_models.object_store import ObjectStoreBaseConfig
from nat.data_models.retriever import RetrieverBaseConfig
from nat.data_models.ttc_strategy import TTCStrategyBaseConfig
from nat.embedder.nim_embedder import NIMEmbedderModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.object_store.in_memory_object_store import InMemoryObjectStoreConfig
from nat.retriever.nemo_retriever.register import NemoRetrieverConfig
from nat.runtime.session import SessionManager
from nat.test.memory import DummyMemoryConfig


@pytest.fixture(name="nested_nat_config", scope="function")
def nested_nat_config_fixture():

    # Setup nested NAT config
    class FnConfig(FunctionBaseConfig, name="test_fn"):
        llm_name: LLMRef
        embedder_name: EmbedderRef
        retriever_name: RetrieverRef | None = None
        memory_name: MemoryRef | None = None
        object_store_name: ObjectStoreRef | None = None
        fn_names: list[FunctionRef] = []

    @register_function(FnConfig)
    async def outer_fn(config: FnConfig, builder: Builder):

        if config.llm_name is not None:
            await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        if config.embedder_name is not None:
            await builder.get_embedder(embedder_name=config.embedder_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        if config.object_store_name is not None:
            await builder.get_object_store_client(object_store_name=config.object_store_name)
        if config.retriever_name is not None:
            await builder.get_retriever(retriever_name=config.retriever_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

        for fn_name in config.fn_names:
            await builder.get_function(name=fn_name)

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
    nested_object_stores_config = {"object_store0": InMemoryObjectStoreConfig()}
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
        "object_stores": nested_object_stores_config,
        "workflow": nested_workflow_config
    }

    nat_config = Config.model_validate(config)

    return nat_config


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
        FunctionGroupBaseConfig: ComponentGroup.FUNCTION_GROUPS,
        LLMBaseConfig: ComponentGroup.LLMS,
        MemoryBaseConfig: ComponentGroup.MEMORY,
        MiddlewareBaseConfig: ComponentGroup.MIDDLEWARE,
        ObjectStoreBaseConfig: ComponentGroup.OBJECT_STORES,
        RetrieverBaseConfig: ComponentGroup.RETRIEVERS,
        TTCStrategyBaseConfig: ComponentGroup.TTC_STRATEGIES,
        AuthProviderBaseConfig: ComponentGroup.AUTHENTICATION,
        TrainerConfig: ComponentGroup.TRAINERS,
        TrainerAdapterConfig: ComponentGroup.TRAINER_ADAPTERS,
        TrajectoryBuilderConfig: ComponentGroup.TRAJECTORY_BUILDERS
    }

    for TestBaseConfig, test_component_group in test_component_config_group_map.items():

        class ComponentConfig(TestBaseConfig, name="test"):  # type: ignore
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
        ComponentRefNode(ref_name="object_store0", component_group=ComponentGroup.OBJECT_STORES),  # type: ignore
        ComponentRefNode(ref_name="retriever0", component_group=ComponentGroup.RETRIEVERS),  # type: ignore
        ComponentRefNode(ref_name="trainer0", component_group=ComponentGroup.TRAINERS),  # type: ignore
        ComponentRefNode(ref_name="trainer_adapter0", component_group=ComponentGroup.TRAINER_ADAPTERS),  # type: ignore
        ComponentRefNode(ref_name="trajectory_builder0",
                         component_group=ComponentGroup.TRAJECTORY_BUILDERS)))  # type: ignore

    # Validate across each base component type class
    base_config_types = [
        FunctionBaseConfig,
        LLMBaseConfig,
        EmbedderBaseConfig,
        MemoryBaseConfig,
        RetrieverBaseConfig,
        TrainerConfig,
        TrainerAdapterConfig,
        TrajectoryBuilderConfig
    ]

    for base_config_type in base_config_types:

        class NestedFns(BaseModel):
            tool_names: list[FunctionRef]

        class MemoryTypedDict(TypedDict):
            memory: MemoryRef

        # Not testing tuple or set based types due to limited Pydantic support
        class TestConfig(base_config_type):  # type: ignore
            llm: LLMRef
            function_from_model: NestedFns
            embedders_dict: dict[str, EmbedderRef]
            retrievers_list: list[RetrieverRef]
            memory_typed_dict: MemoryTypedDict
            object_store_name: list[ObjectStoreRef]
            function_union: FunctionRef | None = None
            trainer: TrainerRef
            trainer_adapter: TrainerAdapterRef
            trajectory_builder: TrajectoryBuilderRef

        instance_config = TestConfig(
            llm="llm0",
            function_from_model=NestedFns(tool_names=["function0", "function1"]),  # type: ignore
            embedders_dict={"embeder_key": "embedder0"},
            retrievers_list=["retriever0"],
            memory_typed_dict=MemoryTypedDict(memory="memory0"),  # type: ignore
            object_store_name=["object_store0"],
            trainer="trainer0",  # type: ignore
            trainer_adapter="trainer_adapter0",  # type: ignore
            trajectory_builder="trajectory_builder0",  # type: ignore
        )

        expected_instance_id = generate_instance_id(instance_config)

        result_set = set()
        for field_name, field_info in TestConfig.model_fields.items():

            for instance_id, value_node in recursive_componentref_discovery(
                    instance_config,
                    getattr(instance_config, field_name),
                    field_info.annotation):  # type: ignore

                # Instance ID should match deep within recursion
                assert instance_id == expected_instance_id

                result_set.add(value_node)

        # Validate discovery of the expected ComponentRef types
        assert len(result_set.difference(expected_result)) == 0


def test_update_dependency_graph(nested_nat_config: Config):

    dependency_graph = nx.DiGraph()

    assert len(dependency_graph.nodes) == 0

    # Test adding an unused leaf
    dependency_graph = update_dependency_graph(nested_nat_config, nested_nat_config.llms["llm0"], dependency_graph)

    assert len(dependency_graph.nodes) == 0

    # Add a function that depends on leaf nodes (llm/embedder/retriever)
    dependency_graph = update_dependency_graph(nested_nat_config,
                                               nested_nat_config.functions["leaf_fn0"],
                                               dependency_graph)

    assert len(dependency_graph.nodes) == 7
    assert dependency_graph.out_degree(generate_instance_id(nested_nat_config.functions["leaf_fn0"])) == 3
    assert dependency_graph.out_degree(generate_instance_id(nested_nat_config.llms["llm0"])) == 0
    assert dependency_graph.out_degree(generate_instance_id(nested_nat_config.embedders["embedder0"])) == 0
    assert dependency_graph.out_degree(generate_instance_id(nested_nat_config.retrievers["retriever0"])) == 0

    # Add a function that depends on other components (leaf and non-leaf nodes)
    dependency_graph = update_dependency_graph(nested_nat_config,
                                               nested_nat_config.functions["nested_fn0"],
                                               dependency_graph)

    assert dependency_graph.out_degree(generate_instance_id(nested_nat_config.functions["leaf_fn0"])) == 3
    assert dependency_graph.out_degree(generate_instance_id(nested_nat_config.llms["llm0"])) == 0
    assert dependency_graph.out_degree(generate_instance_id(nested_nat_config.embedders["embedder0"])) == 0
    assert dependency_graph.out_degree(generate_instance_id(nested_nat_config.retrievers["retriever0"])) == 0
    assert dependency_graph.out_degree(generate_instance_id(nested_nat_config.functions["nested_fn0"])) == 4


def test_config_to_dependency_objects(nested_nat_config: Config):

    # Setup some expected output
    functions_set = set(str(id(value)) for value in nested_nat_config.functions.values())
    embedders_set = set(str(id(value)) for value in nested_nat_config.embedders.values())
    llms_set = set(str(id(value)) for value in nested_nat_config.llms.values())
    retrievers_set = set(str(id(value)) for value in nested_nat_config.retrievers.values())
    memory_set = set(str(id(value)) for value in nested_nat_config.memory.values())
    object_stores_set = set(str(id(value)) for value in nested_nat_config.object_stores.values())
    expected_instance_ids = functions_set | embedders_set | llms_set | retrievers_set | memory_set | object_stores_set
    expected_instance_ids.add(str(id(nested_nat_config.workflow)))

    dependency_map, dependency_graph = config_to_dependency_objects(nested_nat_config)

    # Validate dependency object types
    assert isinstance(dependency_map, dict)
    assert isinstance(dependency_graph, nx.DiGraph)
    assert len(dependency_map) == 13

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
            assert node.ref_name in getattr(nested_nat_config, node.component_group.value)


def test_build_dependency_sequence(nested_nat_config: Config):

    # Setup expected outputs
    expected_dependency_sequence = [
        {
            "component_group": ComponentGroup.MEMORY, "name": "memory0", "is_root": False
        },
        {
            "component_group": ComponentGroup.OBJECT_STORES, "name": "object_store0", "is_root": False
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
        generate_instance_id(nested_nat_config.memory["memory0"]): -1,
        generate_instance_id(nested_nat_config.object_stores["object_store0"]): -1,
        generate_instance_id(nested_nat_config.functions["leaf_fn2"]): -1,
        generate_instance_id(nested_nat_config.functions["leaf_fn3"]): -1,
        generate_instance_id(nested_nat_config.functions["leaf_fn4"]): -1,
    }

    dependency_sequence = build_dependency_sequence(nested_nat_config)

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


@pytest.mark.usefixtures("set_test_api_keys")
async def test_load_hierarchial_workflow(nested_nat_config: Config):

    # Validate nested workflow instantiation
    async with WorkflowBuilder.from_config(config=nested_nat_config) as builder:
        workflow = await builder.build()
        assert SessionManager(config=nested_nat_config,
                              shared_builder=builder,
                              shared_workflow=workflow,
                              max_concurrency=1)


def test_finetuning_component_dependencies():
    """Test that finetuning components can have dependencies and are properly tracked"""
    from nat.cli.register_workflow import register_trainer
    from nat.cli.register_workflow import register_trainer_adapter
    from nat.cli.register_workflow import register_trajectory_builder

    # Create finetuning configs with dependencies on other components
    class TrainerWithDepsConfig(TrainerConfig, name="trainer_with_deps"):
        llm: LLMRef
        function_ref: FunctionRef

    class TrainerAdapterWithDepsConfig(TrainerAdapterConfig, name="trainer_adapter_with_deps"):
        embedder: EmbedderRef
        memory: MemoryRef

    class TrajectoryBuilderWithDepsConfig(TrajectoryBuilderConfig, name="trajectory_builder_with_deps"):
        retriever: RetrieverRef
        object_store: ObjectStoreRef

    # Register the custom finetuning components
    @register_trainer(TrainerWithDepsConfig)
    async def build_trainer(config: TrainerWithDepsConfig, builder: Builder):
        from nat.plugins.openpipe.trainer import ARTTrainer
        yield ARTTrainer(trainer_config=config)

    @register_trainer_adapter(TrainerAdapterWithDepsConfig)
    async def build_trainer_adapter(config: TrainerAdapterWithDepsConfig, builder: Builder):
        from nat.plugins.openpipe.trainer_adapter import ARTTrainerAdapter
        yield ARTTrainerAdapter(adapter_config=config)

    @register_trajectory_builder(TrajectoryBuilderWithDepsConfig)
    async def build_trajectory_builder(config: TrajectoryBuilderWithDepsConfig, builder: Builder):
        from nat.plugins.openpipe.trajectory_builder import ARTTrajectoryBuilder
        yield ARTTrajectoryBuilder(trajectory_builder_config=config)

    # Setup a minimal function config
    class SimpleFnConfig(FunctionBaseConfig, name="simple_fn"):
        pass

    @register_function(SimpleFnConfig)
    async def simple_fn(config: SimpleFnConfig, builder: Builder):

        async def _inner_func(fn_input: str) -> str:
            return ""

        yield _inner_func

    # Create test config with finetuning components that have dependencies
    config_dict = {
        "functions": {
            "fn0": SimpleFnConfig()
        },
        "llms": {
            "llm0": NIMModelConfig(model_name="test")
        },
        "embedders": {
            "embedder0": NIMEmbedderModelConfig(model_name="test")
        },
        "memory": {
            "memory0": DummyMemoryConfig()
        },
        "retrievers": {
            "retriever0": NemoRetrieverConfig(uri="http://test.com")
        },  # type: ignore
        "object_stores": {
            "object_store0": InMemoryObjectStoreConfig()
        },
        "trainers": {
            "trainer0":
                TrainerWithDepsConfig(
                    llm="llm0",  # type: ignore
                    function_ref="fn0"  # type: ignore
                )
        },
        "trainer_adapters": {
            "trainer_adapter0":
                TrainerAdapterWithDepsConfig(
                    embedder="embedder0",  # type: ignore
                    memory="memory0"  # type: ignore
                )
        },
        "trajectory_builders": {
            "trajectory_builder0":
                TrajectoryBuilderWithDepsConfig(
                    retriever="retriever0",  # type: ignore
                    object_store="object_store0"  # type: ignore
                )
        },
        "workflow": SimpleFnConfig()
    }

    test_config = Config.model_validate(config_dict)

    # Test that dependencies are discovered
    dependency_map, dependency_graph = config_to_dependency_objects(test_config)

    # Verify that finetuning components are in the dependency map
    assert any(inst.component_group == ComponentGroup.TRAINERS for inst in dependency_map.values())
    assert any(inst.component_group == ComponentGroup.TRAINER_ADAPTERS for inst in dependency_map.values())
    assert any(inst.component_group == ComponentGroup.TRAJECTORY_BUILDERS for inst in dependency_map.values())

    # Verify that trainer has dependencies on llm and function
    trainer_instance = next(inst for inst in dependency_map.values() if inst.name == "trainer0")
    assert dependency_graph.out_degree(trainer_instance.instance_id) == 2  # llm + function

    # Verify that trainer_adapter has dependencies on embedder and memory
    trainer_adapter_instance = next(inst for inst in dependency_map.values() if inst.name == "trainer_adapter0")
    assert dependency_graph.out_degree(trainer_adapter_instance.instance_id) == 2  # embedder + memory

    # Verify that trajectory_builder has dependencies on retriever and object_store
    trajectory_builder_instance = next(inst for inst in dependency_map.values() if inst.name == "trajectory_builder0")
    assert dependency_graph.out_degree(trajectory_builder_instance.instance_id) == 2  # retriever + object_store

    # Test that the dependency sequence is correct
    dependency_sequence = build_dependency_sequence(test_config)

    # Find positions of components in sequence
    positions = {}
    for idx, inst in enumerate(dependency_sequence):
        positions[inst.name] = idx

    # Verify that dependencies come before the components that depend on them
    assert positions["llm0"] < positions["trainer0"]
    assert positions["fn0"] < positions["trainer0"]
    assert positions["embedder0"] < positions["trainer_adapter0"]
    assert positions["memory0"] < positions["trainer_adapter0"]
    assert positions["retriever0"] < positions["trajectory_builder0"]
    assert positions["object_store0"] < positions["trajectory_builder0"]

    # Verify that finetuning components come after functions but before workflow
    assert positions["fn0"] < positions["trainer0"]
    assert positions["trainer0"] < positions["<workflow>"]
    assert positions["trainer_adapter0"] < positions["<workflow>"]
    assert positions["trajectory_builder0"] < positions["<workflow>"]
