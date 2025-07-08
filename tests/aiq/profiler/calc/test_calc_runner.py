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

from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from aiq.profiler.calc.calc_runner import CalcRunner
from aiq.profiler.calc.data_models import CalcRunnerConfig
from aiq.profiler.calc.data_models import CalcRunnerOutput
from aiq.profiler.calc.data_models import SizingMetricPerItem
from aiq.profiler.calc.data_models import SizingMetrics
from aiq.profiler.calc.data_models import SizingMetricsAlerts


def make_sizing_metrics(latency, runtime, interrupted=False):
    return SizingMetrics(
        llm_latency_p95=latency,
        workflow_runtime_p95=runtime,
        total_runtime=latency + runtime,
        per_item_metrics={0: SizingMetricPerItem(llm_latency=latency, workflow_runtime=runtime)},
        alerts=SizingMetricsAlerts(workflow_interrupted=interrupted),
    )


def make_config(
    offline_mode=False,
    target_latency=1.0,
    target_runtime=2.0,
    target_users=10,
    test_gpu_count=1,
    concurrencies=None,
):
    if concurrencies is None:
        concurrencies = [1, 2]
    return CalcRunnerConfig(
        config_file="config.yml",
        offline_mode=offline_mode,
        target_llm_latency_p95=target_latency,
        target_workflow_runtime_p95=target_runtime,
        target_users=target_users,
        test_gpu_count=test_gpu_count,
        concurrencies=concurrencies,
        output_dir=None,
    )


@pytest.fixture(autouse=True)
def patch_write_output():
    with patch("aiq.profiler.calc.calc_runner.CalcRunner.write_output", return_value=None):
        yield


@pytest.mark.parametrize("latencies,runtimes", [
    ([10, 20], [100, 200]),
    ([5, 50], [80, 300]),
])
async def test_calc_runner(latencies, runtimes):
    config = make_config(offline_mode=False, concurrencies=[1, 2, 3])
    runner = CalcRunner(config)

    with patch("aiq.profiler.calc.calc_runner.MultiEvaluationRunner") as mock_runner:
        mock_instance = mock_runner.return_value

        mock_instance.evaluation_run_outputs = {
            1:
                SimpleNamespace(profiler_results=SimpleNamespace(
                    llm_latency_ci=SimpleNamespace(p95=latencies[0]),
                    workflow_runtime_metrics=SimpleNamespace(p95=runtimes[0])),
                                usage_stats=SimpleNamespace(total_runtime=latencies[0] + runtimes[0]),
                                workflow_interrupted=False),
            2:
                SimpleNamespace(profiler_results=SimpleNamespace(
                    llm_latency_ci=SimpleNamespace(p95=latencies[1]),
                    workflow_runtime_metrics=SimpleNamespace(p95=runtimes[1])),
                                usage_stats=SimpleNamespace(total_runtime=latencies[1] + runtimes[1]),
                                workflow_interrupted=False),
            3:
                SimpleNamespace(profiler_results=SimpleNamespace(llm_latency_ci=SimpleNamespace(p95=30),
                                                                 workflow_runtime_metrics=SimpleNamespace(p95=300)),
                                usage_stats=SimpleNamespace(total_runtime=330),
                                workflow_interrupted=True)
        }

        with patch.object(runner, "output_dir", new_callable=MagicMock):
            output = await runner.run_online()

    # Validate structure
    assert isinstance(output, CalcRunnerOutput)
    assert output.gpu_estimates.gpu_estimate_by_llm_latency > 0
    assert output.gpu_estimates.gpu_estimate_by_wf_runtime > 0
    assert set(output.calc_data.keys()) == {1, 2, 3}

    # Validate metrics match inputs
    assert output.calc_data[1].llm_latency_p95 == latencies[0]
    assert output.calc_data[2].workflow_runtime_p95 == runtimes[1]
    assert output.calc_data[3].alerts.workflow_interrupted is True
