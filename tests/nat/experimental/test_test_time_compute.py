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

from nat.data_models.ttc_strategy import TTCStrategyBaseConfig
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.experimental.test_time_compute.models.strategy_base import StrategyBase
from nat.experimental.test_time_compute.models.ttc_item import TTCItem


# ──────────────────────────────────────────────────────────────────────────────
# Minimal concrete classes to exercise StrategyBase
# ──────────────────────────────────────────────────────────────────────────────
class DummyConfig(TTCStrategyBaseConfig, name="dummy_ttc_config"):
    """Bare-bones config so we can instantiate a StrategyBase subclass."""


class DummyStrategy(StrategyBase):
    """
    Tiny concrete Strategy used only for testing.

    * Supports PLANNING and AGENT_EXECUTION pipelines.
    * Declares itself as a SEARCH-stage strategy.
    * `build_components` flips a flag so we can assert it ran.
    * `ainvoke` returns shallow copies with extra metadata.
    """

    def __init__(self, config: DummyConfig):
        super().__init__(config)
        self._built = False  # toggled by build_components

    # ---- abstract-method implementations -----------------------------------
    async def build_components(self, builder):
        # Real code would wire things up with `builder`.
        self._built = True

    async def ainvoke(self,
                      items: list[TTCItem],
                      original_prompt: str | None = None,
                      agent_context: str | None = None,
                      **kwargs) -> [TTCItem]:
        if items is None:
            items = []
        out = []
        for itm in items:
            data = itm.model_dump()
            # Overwrite or add the metadata field explicitly to avoid duplication
            data["metadata"] = {"invoked": True}
            out.append(TTCItem(**data))
        return out

    def supported_pipeline_types(self):
        return [PipelineTypeEnum.PLANNING, PipelineTypeEnum.AGENT_EXECUTION]

    def stage_type(self):
        return StageTypeEnum.SEARCH


# ──────────────────────────────────────────────────────────────────────────────
# Tests for stage_enums.py
# ──────────────────────────────────────────────────────────────────────────────
def test_pipeline_and_stage_enum_strings():
    """`__str__` should return the raw enum value for readability / logging."""
    assert str(PipelineTypeEnum.PLANNING) == "planning"
    assert str(PipelineTypeEnum.AGENT_EXECUTION) == "agent_execution"
    assert str(StageTypeEnum.SEARCH) == "search"
    assert str(StageTypeEnum.SCORING) == "scoring"


# ──────────────────────────────────────────────────────────────────────────────
# Tests for ttc_item.py
# ──────────────────────────────────────────────────────────────────────────────
def test_ttc_item_accepts_extra_fields_and_preserves_data():
    """
    • Unknown keys should be accepted (model_config.extra == 'allow').
    • Standard fields retain their values.
    """
    item = TTCItem(
        input="in-val",
        output="out-val",
        score=0.75,
        some_extra="hello world",
    )
    assert item.input == "in-val"
    assert item.output == "out-val"
    assert item.score == 0.75
    # Pydantic stores extras in .model_extra / .__pydantic_extra__
    assert item.model_extra["some_extra"] == "hello world"  # type: ignore[attr-defined]


# ──────────────────────────────────────────────────────────────────────────────
# Tests for strategy_base.py via DummyStrategy
# ──────────────────────────────────────────────────────────────────────────────
async def test_set_pipeline_type_validation():
    """Supported pipeline types pass; unsupported ones raise ValueError."""
    strat = DummyStrategy(DummyConfig())

    # Valid
    strat.set_pipeline_type(PipelineTypeEnum.PLANNING)
    assert strat.pipeline_type == PipelineTypeEnum.PLANNING

    # Invalid
    with pytest.raises(ValueError):
        strat.set_pipeline_type(PipelineTypeEnum.TOOL_USE)


async def test_build_components_and_ainvoke_roundtrip():
    """Smoke-test the full lifecycle: build → invoke."""
    strat = DummyStrategy(DummyConfig())

    # build_components should toggle _built
    assert not strat._built
    await strat.build_components(builder=None)
    assert strat._built

    # ainvoke should pass items through and attach metadata
    original_items = [TTCItem(input="foo"), TTCItem(input="bar")]
    new_items = await strat.ainvoke(original_items)

    assert len(new_items) == len(original_items)
    for new, old in zip(new_items, original_items):
        assert new.input == old.input
        assert new.metadata == {"invoked": True}
