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

import pytest
from pydantic import BaseModel

from nat.plugins.config_optimizer.optimizer_runtime import optimize_config
from nat.data_models.optimizer import GAPromptOptimizationConfig
from nat.data_models.optimizer import OptunaParameterOptimizationConfig
from nat.data_models.optimizer import OptimizerConfig
from nat.data_models.optimizer import OptimizerRunConfig


class _DummyConfig(BaseModel):
    """Minimal config for tests that need walk_optimizables to return empty."""

    optimizer: OptimizerConfig = OptimizerConfig()


async def test_optimize_config_returns_input_when_no_space(monkeypatch):
    cfg = _DummyConfig()
    # Ensure no optimizer phases are enabled
    cfg.optimizer.numeric.enabled = False
    cfg.optimizer.prompt.enabled = False

    # Force walk_optimizables to empty mapping
    from nat.plugins.config_optimizer import optimizer_runtime as rt

    monkeypatch.setattr(rt, "walk_optimizables", lambda _cfg: {}, raising=True)
    # Also bypass load_config by passing BaseModel directly
    run = OptimizerRunConfig(config_file=cfg, dataset=None, result_json_path="$", endpoint=None)

    out = await optimize_config(run)
    assert out is cfg


async def test_optimize_config_calls_numeric_and_prompt(monkeypatch):
    cfg = _DummyConfig()
    # Enable both phases
    cfg.optimizer.numeric.enabled = True
    cfg.optimizer.prompt.enabled = True

    from contextlib import asynccontextmanager

    from nat.plugins.config_optimizer import optimizer_runtime as rt

    # Provide a small non-empty space
    monkeypatch.setattr(rt, "walk_optimizables", lambda _cfg: {"x": object()}, raising=True)

    calls = {"numeric": 0, "prompt": 0}

    class _FakeNumericRunner:

        async def run(self, **kwargs):  # noqa: ANN001, ARG002
            calls["numeric"] += 1
            return cfg

    class _FakePromptRunner:

        async def run(self, **kwargs):  # noqa: ANN001, ARG002
            calls["prompt"] += 1

    def _fake_build_numeric(_config):

        @asynccontextmanager
        async def _cm():
            yield _FakeNumericRunner()

        return _cm()

    def _fake_build_prompt(_config):

        @asynccontextmanager
        async def _cm():
            yield _FakePromptRunner()

        return _cm()

    from nat.cli.type_registry import GlobalTypeRegistry
    from nat.cli.type_registry import RegisteredOptimizerInfo
    from nat.data_models.discovery_metadata import DiscoveryMetadata

    registry = GlobalTypeRegistry.get()
    numeric_info = RegisteredOptimizerInfo(
        full_type="test/numeric",
        config_type=OptunaParameterOptimizationConfig,
        build_fn=lambda c: _fake_build_numeric(c),
        discovery_metadata=DiscoveryMetadata(),
    )
    prompt_info = RegisteredOptimizerInfo(
        full_type="test/ga",
        config_type=GAPromptOptimizationConfig,
        build_fn=lambda c: _fake_build_prompt(c),
        discovery_metadata=DiscoveryMetadata(),
    )
    monkeypatch.setattr(
        registry,
        "_registered_optimizer_infos",
        {
            OptunaParameterOptimizationConfig: numeric_info,
            GAPromptOptimizationConfig: prompt_info,
        },
    )

    run = OptimizerRunConfig(config_file=cfg, dataset=None, result_json_path="$", endpoint=None)
    out = await optimize_config(run)
    assert out is cfg
    assert calls["numeric"] == 1
    assert calls["prompt"] == 1
