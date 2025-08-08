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
from unittest.mock import AsyncMock
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
    target_latency=20.0,
    target_runtime=200.0,
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
    target_latency = 20.0
    target_runtime = 200.0
    config = make_config(offline_mode=False,
                         concurrencies=[1, 2, 3],
                         target_latency=target_latency,
                         target_runtime=target_runtime)
    runner = CalcRunner(config)
    evaluation_run_outputs = {
        1:
            SimpleNamespace(profiler_results=SimpleNamespace(llm_latency_ci=SimpleNamespace(p95=latencies[0]),
                                                             workflow_runtime_metrics=SimpleNamespace(p95=runtimes[0])),
                            usage_stats=SimpleNamespace(total_runtime=runtimes[0] + 10, usage_stats_items={}),
                            workflow_interrupted=False),
        2:
            SimpleNamespace(profiler_results=SimpleNamespace(llm_latency_ci=SimpleNamespace(p95=latencies[1]),
                                                             workflow_runtime_metrics=SimpleNamespace(p95=runtimes[1])),
                            usage_stats=SimpleNamespace(total_runtime=runtimes[1] + 10, usage_stats_items={}),
                            workflow_interrupted=False),
        3:
            SimpleNamespace(profiler_results=SimpleNamespace(llm_latency_ci=SimpleNamespace(p95=30),
                                                             workflow_runtime_metrics=SimpleNamespace(p95=300)),
                            usage_stats=SimpleNamespace(total_runtime=330, usage_stats_items={}),
                            workflow_interrupted=True)
    }

    with patch("aiq.profiler.calc.calc_runner.MultiEvaluationRunner") as mock_runner:
        mock_instance = mock_runner.return_value
        mock_instance.run_all = AsyncMock(return_value=evaluation_run_outputs)
        output = await runner.run_online()

    concurrency_list = evaluation_run_outputs.keys()

    assert isinstance(output, CalcRunnerOutput)

    # Validate gpu estimates across concurrencies
    assert output.gpu_estimates.gpu_estimate_by_llm_latency is not None
    assert output.gpu_estimates.gpu_estimate_by_wf_runtime is not None

    # Check all concurrencies are present
    assert set(output.calc_data.keys()) == set(concurrency_list)

    # Check the inputs are copied correctly
    assert output.calc_data[1].sizing_metrics.llm_latency_p95 == latencies[0]
    assert output.calc_data[2].sizing_metrics.workflow_runtime_p95 == runtimes[1]
    assert output.calc_data[3].sizing_metrics.alerts.workflow_interrupted is True

    # check the gpu estimates are present per concurrency
    for concurrency in concurrency_list:
        workflow_interrupted = output.calc_data[concurrency].sizing_metrics.alerts.workflow_interrupted
        if output.calc_data[concurrency].sizing_metrics.llm_latency_p95 > target_latency or workflow_interrupted:
            assert output.calc_data[concurrency].gpu_estimates.gpu_estimate_by_llm_latency is None
        else:
            assert output.calc_data[concurrency].gpu_estimates.gpu_estimate_by_llm_latency is not None
        if output.calc_data[concurrency].sizing_metrics.workflow_runtime_p95 > target_runtime:
            assert output.calc_data[concurrency].gpu_estimates.gpu_estimate_by_wf_runtime is None
        else:
            assert output.calc_data[concurrency].gpu_estimates.gpu_estimate_by_wf_runtime is not None
