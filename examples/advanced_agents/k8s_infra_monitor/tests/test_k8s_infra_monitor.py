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

import json
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_k8s_infra_monitor_node_not_ready(root_repo_dir: Path):
    """Test the agent's ability to diagnose a NotReady node scenario."""
    from nat.test.utils import locate_example_config
    from nat.test.utils import run_workflow
    from nat_k8s_infra_monitor.register import K8sInfraMonitorWorkflowConfig

    config_file: Path = locate_example_config(K8sInfraMonitorWorkflowConfig, "config_offline_mode.yml")

    query = json.dumps({
        "scenario_id": "node-not-ready",
        "query": "Worker node worker-2 appears to be down. Investigate the cluster health.",
    })

    result = await run_workflow(config_file=config_file, question=query, expected_answer="warning")

    output = result.lower()
    assert "worker-2" in output
    assert "notready" in output or "not ready" in output
    assert len(result) > 200, f"Result too short ({len(result)} chars). Got: {result[:500]}"


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_k8s_infra_monitor_memory_pressure(root_repo_dir: Path):
    """Test the agent's ability to diagnose memory pressure with OOMKilled pods."""
    from nat.test.utils import locate_example_config
    from nat.test.utils import run_workflow
    from nat_k8s_infra_monitor.register import K8sInfraMonitorWorkflowConfig

    config_file: Path = locate_example_config(K8sInfraMonitorWorkflowConfig, "config_offline_mode.yml")

    query = json.dumps({
        "scenario_id": "memory-pressure",
        "query": "Multiple pods are crashing in the ml-serving namespace. Check what is happening.",
    })

    result = await run_workflow(config_file=config_file, question=query, expected_answer="critical")

    output = result.lower()
    assert "oom" in output or "memory" in output
    assert "worker-1" in output
    assert len(result) > 200, f"Result too short ({len(result)} chars). Got: {result[:500]}"


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_k8s_infra_monitor_healthy_cluster(root_repo_dir: Path):
    """Test the agent's ability to confirm a healthy cluster state."""
    from nat.test.utils import locate_example_config
    from nat.test.utils import run_workflow
    from nat_k8s_infra_monitor.register import K8sInfraMonitorWorkflowConfig

    config_file: Path = locate_example_config(K8sInfraMonitorWorkflowConfig, "config_offline_mode.yml")

    query = json.dumps({
        "scenario_id": "healthy-cluster",
        "query": "Run a routine health check on the Kubernetes cluster.",
    })

    result = await run_workflow(config_file=config_file, question=query, expected_answer="informational")

    output = result.lower()
    assert "healthy" in output or "normal" in output or "no issues" in output or "informational" in output
    assert len(result) > 200, f"Result too short ({len(result)} chars). Got: {result[:500]}"
