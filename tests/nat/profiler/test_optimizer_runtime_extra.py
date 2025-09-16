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
from pydantic import BaseModel

from nat.data_models.optimizer import OptimizerRunConfig
from nat.profiler.parameter_optimization.optimizer_runtime import optimize_config


class _DummyInner(BaseModel):
    enabled: bool = False


class _DummyPrompt(BaseModel):
    enabled: bool = False


class _DummyOptimizer(BaseModel):
    numeric: _DummyInner = _DummyInner()
    prompt: _DummyPrompt = _DummyPrompt()


class _DummyConfig(BaseModel):
    optimizer: _DummyOptimizer = _DummyOptimizer()


@pytest.mark.asyncio
async def test_optimize_config_returns_input_when_no_space(monkeypatch):
    cfg = _DummyConfig()

    # Force walk_optimizables to empty mapping
    from nat.profiler.parameter_optimization import optimizer_runtime as rt

    monkeypatch.setattr(rt, "walk_optimizables", lambda _cfg: {}, raising=True)
    # Also bypass load_config by passing BaseModel directly
    run = OptimizerRunConfig(config_file=cfg, dataset=None, result_json_path="$", endpoint=None)

    out = await optimize_config(run)
    assert out is cfg


@pytest.mark.asyncio
async def test_optimize_config_calls_numeric_and_prompt(monkeypatch):
    cfg = _DummyConfig()
    # Enable both phases
    cfg.optimizer.numeric.enabled = True
    cfg.optimizer.prompt.enabled = True

    from nat.profiler.parameter_optimization import optimizer_runtime as rt

    # Provide a small non-empty space
    monkeypatch.setattr(rt, "walk_optimizables", lambda _cfg: {"x": object()}, raising=True)

    calls = {"numeric": 0, "prompt": 0}

    def _fake_optimize_parameters(**kwargs):  # noqa: ANN001, ARG001
        del kwargs
        calls["numeric"] += 1
        return cfg

    async def _fake_optimize_prompts(**kwargs):  # noqa: ANN001, ARG001
        del kwargs
        calls["prompt"] += 1

    monkeypatch.setattr(rt, "optimize_parameters", _fake_optimize_parameters, raising=True)
    monkeypatch.setattr(rt, "optimize_prompts", _fake_optimize_prompts, raising=True)

    run = OptimizerRunConfig(config_file=cfg, dataset=None, result_json_path="$", endpoint=None)
    out = await optimize_config(run)
    assert out is cfg
    assert calls["numeric"] == 1
    assert calls["prompt"] == 1
