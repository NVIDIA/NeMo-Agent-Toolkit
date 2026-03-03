# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

from nat.data_models.evaluate_runtime import EvaluationRunConfig
from nat.plugins.eval.cli.evaluate import _build_eval_callback_manager


class WeaveTelemetryExporter:
    """Test stub matching class-name based install hints."""


def test_callback_builder_warns_and_continues_when_callback_missing(caplog):
    caplog.set_level("WARNING")
    config = EvaluationRunConfig(config_file=Path("config.yml"))
    exporter = WeaveTelemetryExporter()
    loaded_cfg = SimpleNamespace()

    mock_registry = MagicMock()
    mock_registry.get_eval_callback.side_effect = KeyError("missing callback")

    with patch("nat.runtime.loader.load_config", return_value=loaded_cfg), \
         patch("nat.observability.utils.tracing_utils.get_tracing_configs", return_value={"weave": exporter}), \
         patch("nat.cli.type_registry.GlobalTypeRegistry.get", return_value=mock_registry):
        manager = _build_eval_callback_manager(config)

    assert manager is None
    assert "nvidia-nat-weave" in caplog.text
    assert "Continuing without eval metric export" in caplog.text


def test_callback_builder_registers_available_callback():
    config = EvaluationRunConfig(config_file=Path("config.yml"))
    loaded_cfg = SimpleNamespace()
    exporter = WeaveTelemetryExporter()

    registered = SimpleNamespace(factory_fn=lambda _cfg: object())
    mock_registry = MagicMock()
    mock_registry.get_eval_callback.return_value = registered

    with patch("nat.runtime.loader.load_config", return_value=loaded_cfg), \
         patch("nat.observability.utils.tracing_utils.get_tracing_configs", return_value={"weave": exporter}), \
         patch("nat.cli.type_registry.GlobalTypeRegistry.get", return_value=mock_registry):
        manager = _build_eval_callback_manager(config)

    assert manager is not None
    assert manager.has_callbacks
