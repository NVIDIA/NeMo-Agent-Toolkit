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
import typing
from collections.abc import Generator
from collections.abc import Iterable

import networkx as nx
from pydantic import BaseModel

from aiq.data_models.authentication import AuthProviderBaseConfig
from aiq.data_models.common import TypedBaseModel
from aiq.data_models.component import ComponentGroup
from aiq.data_models.component_ref import ComponentRef
from aiq.data_models.component_ref import ComponentRefNode
from aiq.data_models.component_ref import generate_instance_id
from aiq.data_models.config import AIQConfig
from aiq.data_models.embedder import EmbedderBaseConfig
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.llm import LLMBaseConfig
from aiq.data_models.memory import MemoryBaseConfig
from aiq.data_models.object_store import ObjectStoreBaseConfig
from aiq.data_models.retriever import RetrieverBaseConfig
from aiq.data_models.ttc_strategy import TTCStrategyBaseConfig
from aiq.utils.type_utils import DecomposedType

logger = logging.getLogger(__name__)

# Order in which we want to process the component groups
_component_group_order = [
    ComponentGroup.AUTHENTICATION,
    ComponentGroup.EMBEDDERS,
    ComponentGroup.LLMS,
    ComponentGroup.MEMORY,
    ComponentGroup.OBJECT_STORES,
    ComponentGroup.RETRIEVERS,
    ComponentGroup.TTC_STRATEGIES,
    ComponentGroup.FUNCTIONS,
]


class ComponentInstanceData(BaseModel):
    """A data model to hold component runtime instance metadata to support generating build sequences.

    Args:
        component_group (ComponentGroup): The component group in an AIQ Toolkit configuration object.
        name (ComponentRef): The name of the component runtime instance.
        config (TypedBaseModel): The runtime instance's configuration object.
        instance_id (str): Unique identifier for each runtime instance.
        is_root (bool): A flag to indicate if the runtime instance is the root of the workflow.
    """

    component_group: ComponentGroup
    name: ComponentRef
    config: TypedBaseModel
    instance_id: str
    is_root: bool = False


def iterate_leaf_to_root(graph: nx.DiGraph) -> Generator[ComponentRefNode]:
    """A recursive generator that yields leaf nodes from the bottom to the root of a directed graph.

    Args:
        graph (nx.DiGraph): A networkx directed graph object.

    Yields:
        ComponentRefNode: An object contain a ComponentRef and its component group.
    """

    leaf_nodes = [node for node, degree in graph.out_degree() if degree == 0]

    if len(leaf_nodes) > 0:
        for leaf_node in leaf_nodes:
            yield leaf_node
            graph.remove_node(leaf_node)

        yield from iterate_leaf_to_root(graph)


def group_from_component(component: TypedBaseModel) -> ComponentGroup | None:
    """Determines the component group from a runtime instance configuration object.

    Args:
        component (TypedBaseModel): A runtime instance configuration object.

    Returns:
        ComponentGroup | None: The component group of the runtime instance configuration object. If the
            component is not a valid runtime instance, None is returned.
    """

    if (isinstance(component, AuthProviderBaseConfig)):
        return ComponentGroup.AUTHENTICATION
    if (isinstance(component, EmbedderBaseConfig)):
        return ComponentGroup.EMBEDDERS
    if (isinstance(component, FunctionBaseConfig)):
        return ComponentGroup.FUNCTIONS
    if (isinstance(component, LLMBaseConfig)):
        return ComponentGroup.LLMS
    if (isinstance(component, MemoryBaseConfig)):
        return ComponentGroup.MEMORY
    if (isinstance(component, ObjectStoreBaseConfig)):
        return ComponentGroup.OBJECT_STORES
    if (isinstance(component, RetrieverBaseConfig)):
        return ComponentGroup.RETRIEVERS
    if (isinstance(component, TTCStrategyBaseConfig)):
        return ComponentGroup.TTC_STRATEGIES

    return None


def recursive_componentref_discovery(cls: TypedBaseModel, value: typing.Any,
                                     type_hint: type[typing.Any]) -> Generator[tuple[str, ComponentRefNode]]:
    """Discovers instances of ComponentRefs in a configuration object and updates the dependency graph.

    Args:
        cls (TypedBaseModel): A configuration object for a runtime instance.
        value (typing.Any): The current traversed value from the configuration object.
        type_hint (type[typing.Any]): The type of the current traversed value from the configuration object.
    """

    decomposed_type = DecomposedType(type_hint)

    if (value is None):
        return

    if ((decomposed_type.origin is None) and (not issubclass(type(value), BaseModel))):
        if issubclass(type(value), ComponentRef):
            instance_id = generate_instance_id(cls)
            value_node = ComponentRefNode(ref_name=value, component_group=value.component_group)
            yield instance_id, value_node

    elif ((decomposed_type.origin in (tuple, list, set)) and (isinstance(value, Iterable))):
        for v in value:
            yield from recursive_componentref_discovery(cls, v, decomposed_type.args[0])
    elif ((decomposed_type.origin in (dict, type(typing.TypedDict))) and (isinstance(value, dict))):
        for v in value.values():
            yield from recursive_componentref_discovery(cls, v, decomposed_type.args[1])
    elif (issubclass(type(value), BaseModel)):
        for field, field_info in value.model_fields.items():
            field_data = getattr(value, field)
            yield from recursive_componentref_discovery(cls, field_data, field_info.annotation)
    if (decomposed_type.is_union):
        for arg in decomposed_type.args:
            if arg is typing.Any or (isinstance(value, DecomposedType(arg).root)):
                yield from recursive_componentref_discovery(cls, value, arg)
    else:
        for arg in decomposed_type.args:
            yield from recursive_componentref_discovery(cls, value, arg)


def update_dependency_graph(config: "AIQConfig", instance_config: TypedBaseModel,
                            dependency_graph: nx.DiGraph) -> nx.DiGraph:
    """Updates the hierarchical component instance dependency graph from a configuration runtime instance.

    Args:
        config (AIQConfig): An AIQ Toolkit configuration object with runtime instance details.
        instance_config (TypedBaseModel): A component's runtime instance configuration object.
        dependency_graph (nx.DiGraph): A graph tracking runtime instance component dependencies.

    Returns:
        nx.DiGraph: An dependency graph that has been updated with the provided runtime instance.
    """

    for field_name, field_info in instance_config.model_fields.items():

        for instance_id, value_node in recursive_componentref_discovery(
                instance_config,
                getattr(instance_config, field_name),
                field_info.annotation):  # type: ignore

            # add immediate edge
            dependency_graph.add_edge(instance_id, value_node)
            # add dependency edge to ensure connections to leaf nodes exist
            dependency_component_dict = getattr(config, value_node.component_group)
            dependency_component_instance_config = dependency_component_dict.get(value_node.ref_name)
            dependency_component_instance_id = generate_instance_id(dependency_component_instance_config)
            dependency_graph.add_edge(value_node, dependency_component_instance_id)

    return dependency_graph


def config_to_dependency_objects(config: "AIQConfig") -> tuple[dict[str, ComponentInstanceData], nx.DiGraph]:
    """Generates a map of component runtime instance IDs to use when generating a build sequence.

    Args:
        config (AIQConfig): The AIQ Toolkit workflow configuration object.

    Returns:
        tuple[dict[str, ComponentInstanceData], nx.DiGraph]: A tuple containing a map of component runtime instance
            IDs to a component object containing its metadata and a dependency graph of nested components.
    """

    # Build map of every runtime instances
    dependency_map: dict[str, ComponentInstanceData] = {}
    dependency_graph: nx.DiGraph = nx.DiGraph()

    # Create the dependency map preserving as much order as we can
    for group in _component_group_order:

        component_dict = getattr(config, group.value)

        assert isinstance(component_dict, dict), "Config components must be a dictionary"

        for component_instance_name, component_instance_config in component_dict.items():

            instance_id = generate_instance_id(component_instance_config)
            dependency_map[instance_id] = ComponentInstanceData(component_group=group,
                                                                instance_id=instance_id,
                                                                name=component_instance_name,
                                                                config=component_instance_config)

            dependency_graph = update_dependency_graph(config=config,
                                                       instance_config=component_instance_config,
                                                       dependency_graph=dependency_graph)

    # Set the workflow flag on the workflow instance (must be last)
    workflow_instance_id = generate_instance_id(config.workflow)

    dependency_map[workflow_instance_id] = ComponentInstanceData(
        component_group=ComponentGroup.FUNCTIONS,
        instance_id=workflow_instance_id,
        name="<workflow>",  # type: ignore
        config=config.workflow,
        is_root=True)

    dependency_graph = update_dependency_graph(config=config,
                                               instance_config=config.workflow,
                                               dependency_graph=dependency_graph)

    return dependency_map, dependency_graph


def build_dependency_sequence(config: "AIQConfig") -> list[ComponentInstanceData]:
    """Generates the depencency sequence from an AIQ Toolkit configuration object

    Args:
        config (AIQConfig): An AIQ Toolkit configuration object.

    Returns:
        list[ComponentInstanceData]: A list representing the instatiation sequence to ensure all valid
            runtime instance references.
    """

    total_node_count = len(config.embedders) + len(config.functions) + len(config.llms) + len(config.memory) + len(
        config.object_stores) + len(config.retrievers) + len(config.ttc_strategies) + len(
            config.authentication) + 1  # +1 for the workflow

    dependency_map: dict
    dependency_graph: nx.DiGraph
    dependency_map, dependency_graph = config_to_dependency_objects(config=config)

    dependency_sequence: list[ComponentInstanceData] = []
    instance_ids = set()
    for node in iterate_leaf_to_root(dependency_graph.copy()):  # type: ignore

        if (node not in dependency_sequence):

            # Convert node to id
            if (isinstance(node, ComponentRefNode) and issubclass(type(node.ref_name), ComponentRef)):

                component_group_configs = getattr(config, node.component_group.value)
                node_config = component_group_configs.get(node.ref_name, None)

                # Only add nodes that are valid in the current instance configuration
                if (node_config is None):
                    continue

                component_instance = ComponentInstanceData(
                    name=node.ref_name,
                    component_group=node.component_group.value,  # type: ignore
                    config=node_config,
                    instance_id=generate_instance_id(node_config))

            else:

                component_instance = dependency_map.get(node, None)

                # Only add nodes that are valid in the current instance configuration
                if (component_instance is None):
                    continue

            if (component_instance.instance_id not in instance_ids):

                dependency_sequence.append(component_instance)
                instance_ids.add(component_instance.instance_id)

    remaining_dependency_sequence: list[ComponentInstanceData] = []

    # Find the remaining nodes that are not in the sequence preserving order
    for instance_id, instance in dependency_map.items():
        if (instance_id not in instance_ids):
            remaining_dependency_sequence.append(instance)

    # Add the remaining at the front of the sequence
    dependency_sequence = remaining_dependency_sequence + dependency_sequence

    # Find the root node and make sure it is the last node in the sequence
    dependency_sequence = [x for x in dependency_sequence if not x.is_root
                           ] + [x for x in dependency_sequence if x.is_root]

    assert len(dependency_sequence) == total_node_count, "Dependency sequence generation failed. Report as bug."

    return dependency_sequence
