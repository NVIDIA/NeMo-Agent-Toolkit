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

import re

import pytest
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError

from nat.data_models.gated_field_mixin import GatedFieldMixin


# Test data fixtures
@pytest.fixture
def gpt_pattern():
    return re.compile(r"gpt")


@pytest.fixture
def claude_pattern():
    return re.compile(r"claude")


@pytest.fixture
def llama_pattern():
    return re.compile(r"llama")


@pytest.fixture
def gpt4_pattern():
    return re.compile(r"gpt-4")


@pytest.fixture
def gpt3_pattern():
    return re.compile(r"gpt-3")


class TestGatedFieldMixin:
    """Comprehensive tests for GatedFieldMixin"""

    @pytest.mark.parametrize("test_case",
                             [{
                                 "name": "both_selectors",
                                 "unsupported": (re.compile(r"alpha"), ),
                                 "supported": (re.compile(r"beta"), ),
                                 "error_msg": r"Only one of unsupported or supported must be provided"
                             },
                              {
                                  "name": "no_selectors",
                                  "unsupported": None,
                                  "supported": None,
                                  "error_msg": r"Either unsupported or supported must be provided"
                              }])
    def test_selector_validation(self, test_case):
        """Test selector validation scenarios."""
        with pytest.raises(ValueError, match=test_case["error_msg"]):

            class BadConfig(BaseModel,
                            GatedFieldMixin,
                            field_name="dummy",
                            default_if_supported=1,
                            keys=("model_name", ),
                            unsupported=test_case["unsupported"],
                            supported=test_case["supported"]):
                dummy: int | None = Field(default=None)
                model_name: str = "alpha"

            _ = BadConfig()

    def test_empty_model_keys_raises_error(self):
        """Test that empty model_keys raises an error."""
        with pytest.raises(ValueError, match=r"keys must be provided and non-empty"):

            class EmptyKeys(BaseModel,
                            GatedFieldMixin,
                            field_name="test",
                            default_if_supported=1,
                            supported=(re.compile(r"test"), ),
                            keys=()):
                test: int | None = Field(default=None)
                model_name: str = "test"

            _ = EmptyKeys()

    def test_supported_model_default(self):
        """Test supported model with default value."""

        class SupportedModelTest(BaseModel,
                                 GatedFieldMixin,
                                 field_name="dummy",
                                 default_if_supported=5,
                                 supported=(re.compile(r"gpt"), ),
                                 keys=("model_name", )):
            dummy: int | None = Field(default=None)
            model_name: str

        m = SupportedModelTest(model_name="gpt-4")
        assert m.dummy == 5

    def test_custom_model_keys_supported(self):
        """Test custom model keys with supported models."""

        class CustomKeysTest(BaseModel,
                             GatedFieldMixin,
                             field_name="dummy",
                             default_if_supported=9,
                             supported=(re.compile(r"valid"), ),
                             keys=("custom_key", )):
            dummy: int | None = Field(default=None)
            custom_key: str

        m = CustomKeysTest(custom_key="valid")
        assert m.dummy == 9

    def test_unsupported_model_validation_error(self):
        """Test validation error for unsupported models."""

        class UnsupportedModelTest(BaseModel,
                                   GatedFieldMixin,
                                   field_name="dummy",
                                   default_if_supported=5,
                                   unsupported=(re.compile(r"claude"), ),
                                   keys=("model_name", )):
            dummy: int | None = Field(default=None)
            model_name: str

        with pytest.raises(ValidationError, match=r"dummy is not supported for model_name: claude"):
            _ = UnsupportedModelTest(model_name="claude", dummy=3)

    def test_unsupported_model_none_value(self):
        """Test unsupported model with None value."""

        class UnsupportedModelNoneTest(BaseModel,
                                       GatedFieldMixin,
                                       field_name="dummy",
                                       default_if_supported=5,
                                       unsupported=(re.compile(r"claude"), ),
                                       keys=("model_name", )):
            dummy: int | None = Field(default=None)
            model_name: str

        m = UnsupportedModelNoneTest(model_name="claude")
        assert m.dummy is None

    def test_first_key_supported(self):
        """Test first key being supported in multiple keys scenario."""

        class MultiKeyModel(BaseModel,
                            GatedFieldMixin,
                            field_name="feature",
                            default_if_supported=42,
                            supported=(re.compile(r"gpt"), ),
                            keys=("primary_model", "fallback_model", "deployment")):
            feature: int | None = Field(default=None)
            primary_model: str
            fallback_model: str
            deployment: str

        m = MultiKeyModel(primary_model="gpt-4", fallback_model="claude", deployment="llama")
        assert m.feature == 42

    def test_first_key_unsupported(self):
        """Test first key being unsupported in multiple keys scenario."""

        class MultiKeyModel(BaseModel,
                            GatedFieldMixin,
                            field_name="feature",
                            default_if_supported=42,
                            supported=(re.compile(r"gpt"), ),
                            keys=("primary_model", "fallback_model", "deployment")):
            feature: int | None = Field(default=None)
            primary_model: str
            fallback_model: str
            deployment: str

        m = MultiKeyModel(primary_model="claude", fallback_model="gpt-3.5", deployment="llama")
        assert m.feature == 42

    def test_numeric_model_values(self):
        """Test numeric model values."""

        class NumericModelTest(BaseModel,
                               GatedFieldMixin,
                               field_name="numeric_feature",
                               default_if_supported=100,
                               supported=(re.compile(r"42"), re.compile(r"99")),
                               keys=("model_id", "version_num")):
            numeric_feature: int | None = Field(default=None)
            model_id: int
            version_num: int

        m = NumericModelTest(model_id=42, version_num=123)
        assert m.numeric_feature == 100

    def test_no_model_keys_fallback(self):
        """Test fallback behavior when no model keys are found."""

        class NoKeysModel(BaseModel,
                          GatedFieldMixin,
                          field_name="fallback_feature",
                          default_if_supported=42,
                          supported=(re.compile(r"gpt"), ),
                          keys=("nonexistent_key", )):
            fallback_feature: int | None = Field(default=None)

        m = NoKeysModel()
        assert m.fallback_feature == 42

    def test_find_blocking_key_edge_cases(self):
        """Test edge cases in finding blocking keys."""

        class BlockingKeyTest(BaseModel,
                              GatedFieldMixin,
                              field_name="test_feature",
                              default_if_supported=100,
                              unsupported=(re.compile(r"blocked"), ),
                              keys=("key1", "key2")):
            test_feature: int | None = Field(default=None)
            key1: str = "blocked"  # First key should be blocked
            key2: str = "allowed"

        with pytest.raises(ValidationError, match=r"test_feature is not supported for key1: blocked"):
            _ = BlockingKeyTest(test_feature=999)

    def test_deep_inheritance_chain(self, gpt_pattern, claude_pattern):
        """Test that deep inheritance chains work correctly."""

        class BaseMixinGPT(BaseModel,
                           GatedFieldMixin,
                           field_name="deep_feature",
                           default_if_supported=100,
                           supported=(gpt_pattern, ),
                           keys=("model_name", )):
            deep_feature: int | None = Field(default=None)

        class BaseMixinClaude(BaseModel,
                              GatedFieldMixin,
                              field_name="deep_feature_2",
                              default_if_supported=200,
                              supported=(claude_pattern, ),
                              keys=("model_name", )):
            deep_feature_2: int | None = Field(default=None)

        class MiddleMixinGPT(BaseMixinGPT):
            """This class inherits from BaseMixin but not directly from BaseModel, GatedFieldMixin."""
            pass

        class MiddleMixinClaude(BaseMixinClaude):
            """This class inherits from BaseMixin2 but not directly from BaseModel, GatedFieldMixin."""
            pass

        class FinalModelGPT(MiddleMixinGPT):
            """This class inherits from MiddleMixin, creating a deep inheritance chain."""
            model_name: str

        class FinalModelBoth(MiddleMixinGPT, MiddleMixinClaude):
            """This class inherits from MiddleMixin2, creating a deep inheritance chain."""
            model_name: str

        m = FinalModelGPT(model_name="gpt-4")
        assert m.deep_feature == 100

        with pytest.raises(ValidationError, match=r"deep_feature is not supported for model_name: claude"):
            _ = FinalModelGPT(model_name="claude", deep_feature=999)

        m = FinalModelGPT(model_name="gpt-4", deep_feature=50)
        assert m.deep_feature == 50

        m2_claude = FinalModelBoth(model_name="claude")
        assert m2_claude.deep_feature_2 == 200
        assert m2_claude.deep_feature is None

        with pytest.raises(ValidationError, match=r"deep_feature_2 is not supported for model_name: gpt-4"):
            _ = FinalModelBoth(model_name="gpt-4", deep_feature_2=999)

        m2_gpt = FinalModelBoth(model_name="gpt-4")
        assert m2_gpt.deep_feature == 100
        assert m2_gpt.deep_feature_2 is None

        with pytest.raises(ValidationError, match=r"deep_feature is not supported for model_name: claude"):
            _ = FinalModelBoth(model_name="claude", deep_feature=999)
