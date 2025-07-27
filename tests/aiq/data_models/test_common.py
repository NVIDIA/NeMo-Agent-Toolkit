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

import json
import typing
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiq.data_models import common


class ԊashableTĕstModel(common.HashableBaseModel):  # pylint: disable=non-ascii-name
    """
    Intentionally using non-ascci characters to test the encoding for the hash
    """
    apples: int
    pair: tuple[int, int]


def test_hashable_base_model_is_hashable():
    h1 = ԊashableTĕstModel(apples=2, pair=(4, 5))
    h2 = ԊashableTĕstModel(apples=3, pair=(4, 5))
    h3 = ԊashableTĕstModel(apples=2, pair=(4, 5))  # same as h1

    configs = {h1, h2, h3}
    assert len(configs) == 2
    assert h1 in configs
    assert h2 in configs
    assert h3 in configs


def test_hashable_base_model_write_json_schema(tmp_path: Path):
    schema_path = tmp_path / "test_schema.json"
    ԊashableTĕstModel.write_json_schema(schema_path)

    assert schema_path.exists()
    assert schema_path.is_file()

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
        assert schema == ԊashableTĕstModel.generate_json_schema()


def test_subclass_depth():

    class Parent:
        pass

    class Child(Parent):
        pass

    class GrandChild(Child):
        pass

    assert common.subclass_depth(GrandChild) == 3

    # We know that ԊashableTĕstModel has at least three levels of inheritance:
    # ԊashableTĕstModel -> HashableBaseModel -> BaseModel -> ... -> object
    # we don't want to make any assumptions about the number of levels of inheritance between BaseModel and object
    assert common.subclass_depth(ԊashableTĕstModel) >= 3


@pytest.mark.parametrize("v, expected_value",
                         [({
                             "_type": "_type_test"
                         }, "_type_test"), ({
                             "type": "type_test"
                         }, "type_test"), ({
                             "_type": "correct", "type": "incorrect"
                         }, "correct"), ({}, None), (MagicMock(spec=["type"], type="apples"), "apples")],
                         ids=["dict-with-_type", "dict-with-type", "dict with both", "no_type", "object"])
def test_type_discriminator(v: typing.Any, expected_value: str | None):
    assert common.TypedBaseModel.discriminator(v) == expected_value


class TestTypedBaseModelInheritance:
    """Test suite for TypedBaseModel inheritance and type handling."""

    def test_simple_inheritance_static_type(self):
        """Test that simple inheritance classes have correct static_type."""

        class ComponentA(common.TypedBaseModel, name="component_a"):
            pass

        class ComponentB(common.TypedBaseModel, name="component_b"):
            pass

        class ComponentC(common.TypedBaseModel, name="component_c"):
            pass

        # Each class should return its own name, not the last loaded one
        assert ComponentA.static_type() == "component_a"
        assert ComponentB.static_type() == "component_b"
        assert ComponentC.static_type() == "component_c"

    def test_instance_type_field_correct(self):
        """Test that instances get the correct type field value."""

        class ComponentA(common.TypedBaseModel, name="component_a"):
            pass

        class ComponentB(common.TypedBaseModel, name="component_b"):
            pass

        # Create instances
        instance_a = ComponentA()
        instance_b = ComponentB()

        # Each instance should have the correct type
        assert instance_a.type == "component_a"
        assert instance_b.type == "component_b"

    def test_no_cross_contamination(self):
        """Test that there's no cross-contamination between classes (regression test)."""

        # Simulate the original bug scenario with multiple classes loaded in sequence
        class FirstComponent(common.TypedBaseModel, name="first"):
            pass

        class SecondComponent(common.TypedBaseModel, name="second"):
            pass

        class ThirdComponent(common.TypedBaseModel, name="third"):
            pass

        # Verify no class shows the wrong name (original bug was all showing "third")
        assert FirstComponent.static_type() == "first"
        assert SecondComponent.static_type() == "second"
        assert ThirdComponent.static_type() == "third"

        # Also test instances
        first_instance = FirstComponent()
        second_instance = SecondComponent()
        third_instance = ThirdComponent()

        assert first_instance.type == "first"
        assert second_instance.type == "second"
        assert third_instance.type == "third"

    def test_mixin_inheritance_patterns(self):
        """Test that mixin inheritance patterns work correctly."""

        # Simulate the mixin patterns used in telemetry exporters
        class BatchConfigMixin:
            batch_size: int = 100

        class CollectorConfigMixin:
            endpoint = "http://localhost"

        class TelemetryExporterBase(common.TypedBaseModel):
            pass

        class WeaveExporter(TelemetryExporterBase, name="weave"):
            pass

        class PhoenixExporter(BatchConfigMixin, CollectorConfigMixin, TelemetryExporterBase, name="phoenix"):
            pass

        class CatalystExporter(BatchConfigMixin, TelemetryExporterBase, name="catalyst"):
            pass

        # Test static types (this was the main visible bug)
        assert WeaveExporter.static_type() == "weave"
        assert PhoenixExporter.static_type() == "phoenix"
        assert CatalystExporter.static_type() == "catalyst"

        # Test instances
        weave = WeaveExporter()
        phoenix = PhoenixExporter()
        catalyst = CatalystExporter()

        assert weave.type == "weave"
        assert phoenix.type == "phoenix"
        assert catalyst.type == "catalyst"

    def test_deep_inheritance_chains(self):
        """Test that deep inheritance chains work correctly."""

        class BaseComponent(common.TypedBaseModel, name="base"):
            pass

        class MiddleComponent(BaseComponent, name="middle"):
            pass

        class LeafComponent(MiddleComponent, name="leaf"):
            pass

        # Each level should have correct type
        assert BaseComponent.static_type() == "base"
        assert MiddleComponent.static_type() == "middle"
        assert LeafComponent.static_type() == "leaf"

        # Test instances
        base_instance = BaseComponent()
        middle_instance = MiddleComponent()
        leaf_instance = LeafComponent()

        assert base_instance.type == "base"
        assert middle_instance.type == "middle"
        assert leaf_instance.type == "leaf"

    def test_type_field_assignment(self):
        """Test that type field assignment works (needed for YAML loading)."""

        class TestComponent(common.TypedBaseModel, name="test_component"):
            pass

        instance = TestComponent()

        # Initial type should be correct
        assert instance.type == "test_component"

        # Should be able to assign new value (YAML loading scenario)
        instance.type = "custom_type"
        assert instance.type == "custom_type"

        # Static type should remain unchanged
        assert TestComponent.static_type() == "test_component"

    def test_unnamed_class_handling(self):
        """Test that classes without names are handled gracefully."""

        class UnnamedComponent(common.TypedBaseModel):
            pass

        # Should return None for static_type
        assert UnnamedComponent.static_type() is None

        # Instance should get default value
        instance = UnnamedComponent()
        assert instance.type == "unknown"

    def test_model_post_init_behavior(self):
        """Test that model_post_init correctly sets the type field."""

        class PostInitComponent(common.TypedBaseModel, name="post_init_test"):
            field1: str = "value1"

        instance = PostInitComponent()

        # Type should be set correctly after post-init
        assert instance.type == "post_init_test"

        # Other fields should work normally
        assert instance.field1 == "value1"

    def test_json_schema_generation_basic(self):
        """Test that JSON schema generation shows correct defaults for named components."""
        from pydantic import Field

        class SchemaTestComponent(common.TypedBaseModel, name="schema_test"):
            field1: str = Field(description="A test field")
            field2: int = Field(default=42, description="A number field")

        schema = SchemaTestComponent.model_json_schema()

        # Check that schema has correct structure
        assert "properties" in schema
        assert "type" in schema["properties"]

        # Check type field has correct default (not "unknown")
        type_field = schema["properties"]["type"]
        assert type_field["default"] == "schema_test"
        assert type_field["description"] == "The type of the object"
        assert type_field["type"] == "string"

        # Check other fields are preserved
        assert "field1" in schema["properties"]
        assert "field2" in schema["properties"]
        assert schema["properties"]["field2"]["default"] == 42

    def test_json_schema_generation_multiple_components(self):
        """Test that different components get different schema defaults."""

        class ComponentX(common.TypedBaseModel, name="component_x"):
            pass

        class ComponentY(common.TypedBaseModel, name="component_y"):
            pass

        schema_x = ComponentX.model_json_schema()
        schema_y = ComponentY.model_json_schema()

        # Each should have its own correct default
        assert schema_x["properties"]["type"]["default"] == "component_x"
        assert schema_y["properties"]["type"]["default"] == "component_y"

        # Schemas should be different
        assert schema_x["properties"]["type"]["default"] != schema_y["properties"]["type"]["default"]

    def test_json_schema_generation_unnamed_component(self):
        """Test that unnamed components show 'unknown' in schema."""

        class UnnamedSchemaComponent(common.TypedBaseModel):
            pass

        schema = UnnamedSchemaComponent.model_json_schema()

        # Unnamed component should have "unknown" default
        assert schema["properties"]["type"]["default"] == "unknown"

    def test_json_schema_generation_mixin_inheritance(self):
        """Test that mixin inheritance components have correct schema defaults."""

        class SchemaBatchMixin:
            batch_size: int = 100

        class SchemaCollectorMixin:
            endpoint: str = "http://localhost"

        class SchemaTelemetryBase(common.TypedBaseModel):
            pass

        class SchemaWeaveExporter(SchemaTelemetryBase, name="weave_schema"):
            pass

        class SchemaPhoenixExporter(SchemaBatchMixin, SchemaCollectorMixin, SchemaTelemetryBase, name="phoenix_schema"):
            pass

        weave_schema = SchemaWeaveExporter.model_json_schema()
        phoenix_schema = SchemaPhoenixExporter.model_json_schema()

        # Each should have correct schema default despite complex inheritance
        assert weave_schema["properties"]["type"]["default"] == "weave_schema"
        assert phoenix_schema["properties"]["type"]["default"] == "phoenix_schema"

    def test_json_schema_consistency_with_runtime(self):
        """Test that schema defaults match actual runtime behavior."""

        class ConsistencyTestA(common.TypedBaseModel, name="consistency_a"):
            pass

        class ConsistencyTestB(common.TypedBaseModel, name="consistency_b"):
            pass

        # Get schema defaults
        schema_a = ConsistencyTestA.model_json_schema()
        schema_b = ConsistencyTestB.model_json_schema()
        schema_default_a = schema_a["properties"]["type"]["default"]
        schema_default_b = schema_b["properties"]["type"]["default"]

        # Get runtime values
        instance_a = ConsistencyTestA()
        instance_b = ConsistencyTestB()
        static_a = ConsistencyTestA.static_type()
        static_b = ConsistencyTestB.static_type()

        # All should match
        assert schema_default_a == instance_a.type == static_a == "consistency_a"
        assert schema_default_b == instance_b.type == static_b == "consistency_b"

    def test_json_schema_field_metadata_preserved(self):
        """Test that other field metadata is preserved in schema generation."""
        from pydantic import Field

        class MetadataTestComponent(common.TypedBaseModel, name="metadata_test"):
            required_field: str = Field(description="This field is required")
            optional_field: str = Field(default="default_value",
                                        description="This field is optional",
                                        title="Optional Field")
            number_field: int = Field(default=100, ge=0, le=1000, description="A constrained number field")

        schema = MetadataTestComponent.model_json_schema()

        # Check that type field metadata is correct
        type_field = schema["properties"]["type"]
        assert type_field["default"] == "metadata_test"
        assert type_field["description"] == "The type of the object"
        assert type_field["title"] == "Type"

        # Check that other field metadata is preserved
        required_field = schema["properties"]["required_field"]
        assert required_field["description"] == "This field is required"
        assert "default" not in required_field  # Required field should not have default

        optional_field = schema["properties"]["optional_field"]
        assert optional_field["default"] == "default_value"
        assert optional_field["description"] == "This field is optional"
        assert optional_field["title"] == "Optional Field"

        number_field = schema["properties"]["number_field"]
        assert number_field["default"] == 100
        assert number_field["minimum"] == 0
        assert number_field["maximum"] == 1000

        # Check required fields
        assert "required_field" in schema["required"]
        assert "optional_field" not in schema["required"]
        assert "type" not in schema["required"]  # type field should not be required

    def test_json_schema_deep_inheritance(self):
        """Test that deep inheritance chains have correct schema defaults."""

        class SchemaBaseComponent(common.TypedBaseModel, name="schema_base"):
            pass

        class SchemaMiddleComponent(SchemaBaseComponent, name="schema_middle"):
            pass

        class SchemaLeafComponent(SchemaMiddleComponent, name="schema_leaf"):
            pass

        base_schema = SchemaBaseComponent.model_json_schema()
        middle_schema = SchemaMiddleComponent.model_json_schema()
        leaf_schema = SchemaLeafComponent.model_json_schema()

        # Each level should have its own correct default
        assert base_schema["properties"]["type"]["default"] == "schema_base"
        assert middle_schema["properties"]["type"]["default"] == "schema_middle"
        assert leaf_schema["properties"]["type"]["default"] == "schema_leaf"
