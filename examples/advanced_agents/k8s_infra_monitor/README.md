<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Kubernetes Infrastructure Monitor using NeMo Agent Toolkit

**Complexity:** 🟨 Intermediate

This example demonstrates how to build an intelligent Kubernetes cluster monitoring agent using NeMo Agent Toolkit and LangGraph. The agent analyzes cluster health queries by gathering diagnostics from multiple tools — node status, pod health, cluster events, and resource utilization — then correlates the findings to produce structured incident reports with severity classification.

## Table of Contents

- [Key Features](#key-features)
- [Installation and Setup](#installation-and-setup)
- [Use Case Description](#use-case-description)
  - [Why Use an Agentic Design?](#why-use-an-agentic-design)
- [How It Works](#how-it-works)
  - [Understanding the Configuration](#understanding-the-configuration)
- [Example Usage](#example-usage)
  - [Running in Offline Mode](#running-in-offline-mode)
  - [Running in Live Mode](#running-in-live-mode)

## Key Features

- **Automated Cluster Health Analysis:** An agent that autonomously investigates Kubernetes cluster health queries using multiple diagnostic tools and generates structured reports.
- **Multi-Tool Diagnostic Framework:** Integrates node status checks, pod health scanning, event collection, and resource pressure analysis for comprehensive cluster diagnosis.
- **Dynamic Tool Selection:** The agent selects appropriate diagnostic tools based on the query context — a question about crashing pods triggers pod health and event checks, while a node issue triggers node status and resource analysis.
- **Severity Classification:** Automatically classifies incidents as critical, warning, or informational based on the collected evidence.
- **Offline and Live Modes:** Run with synthetic scenarios for development and testing, or connect to a real Kubernetes cluster via `kubectl` for production monitoring.

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/get-started/installation.md#install-from-source) to create the development environment and install NeMo Agent Toolkit.

### Install This Workflow

From the root directory of the NeMo Agent Toolkit library:

```bash
uv pip install -e examples/advanced_agents/k8s_infra_monitor
```

### Set Up API Keys

Export your NVIDIA API key:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

## Use Case Description

Kubernetes clusters generate a constant stream of operational signals — node conditions, pod status changes, events, and resource metrics. Triaging these signals manually is time-consuming, especially in clusters running dozens of workloads across multiple namespaces.

This example provides an agentic system that:

1. **Gathers node diagnostics**: Checks node readiness, conditions (MemoryPressure, DiskPressure, PIDPressure), and resource utilization via `kubectl top`.
2. **Scans pod health**: Identifies unhealthy pods (CrashLoopBackOff, OOMKilled, Pending, Evicted) and flags containers with high restart counts.
3. **Collects cluster events**: Retrieves recent Warning events and correlates them with affected resources.
4. **Analyzes resource pressure**: Detects nodes approaching CPU or memory thresholds and flags active pressure conditions.
5. **Classifies severity**: Uses an LLM to classify the overall incident severity based on collected evidence.
6. **Generates structured reports**: Produces markdown reports with findings, root cause analysis, and recommended remediation steps.

### Why Use an Agentic Design?

An agentic approach provides significant advantages over static dashboards or rule-based alerting:

- **Contextual investigation**: The agent decides which tools to call based on the query, rather than running every check every time.
- **Cross-signal correlation**: Unlike siloed monitoring tools, the agent correlates data from nodes, pods, events, and resources to identify root causes (e.g., OOMKilled pods + MemoryPressure condition = memory exhaustion on a specific node).
- **Natural language reports**: Produces human-readable incident summaries that can be directly shared with team members or fed into ticketing systems.

## How It Works

### Diagnostic Tools

| Tool | Description | Live Mode |
|------|-------------|-----------|
| `node_status_check` | Retrieves node readiness, conditions, and resource utilization (`kubectl get nodes`, `kubectl top nodes`) | Uses `kubectl` |
| `pod_health_check` | Scans for unhealthy pods and high restart counts across namespaces | Uses `kubectl` |
| `event_collector` | Collects recent Warning events and correlates them with affected resources | Uses `kubectl` |
| `resource_pressure_check` | Analyzes CPU/memory utilization against configurable thresholds, checks for pressure conditions | Uses `kubectl` |
| `severity_classifier` | Classifies the final report's severity as critical, warning, or informational | LLM-based |

### Workflow

1. A cluster health query is received (natural language or JSON with scenario context).
2. The monitor agent selects relevant diagnostic tools based on the query.
3. Tools gather data (from `kubectl` in live mode, or from offline scenarios).
4. The agent correlates findings across all tool outputs.
5. A structured diagnostic report is generated.
6. The severity classifier appends an incident severity classification.

### Understanding the Configuration

#### Functions

Each tool is configured in the `functions` section:

```yaml
functions:
  node_status_check:
    _type: node_status_check
    offline_mode: true
  resource_pressure_check:
    _type: resource_pressure_check
    offline_mode: true
    cpu_threshold_percent: 80
    memory_threshold_percent: 85
```

- `offline_mode`: When `true`, tools return pre-defined responses from the offline scenario dataset.
- `cpu_threshold_percent` / `memory_threshold_percent`: Configurable thresholds for resource pressure alerts.
- `kubeconfig_path`: Optional path to a kubeconfig file for live mode. Defaults to the standard `kubectl` config.

#### Workflow

```yaml
workflow:
  _type: k8s_infra_monitor
  tool_names:
    - node_status_check
    - pod_health_check
    - event_collector
    - resource_pressure_check
  llm_name: monitor_agent_llm
  offline_mode: true
  offline_data_path: examples/advanced_agents/k8s_infra_monitor/data/offline_scenarios.json
```

#### LLMs

All tools and the main agent use NVIDIA NIM with `nvidia/nemotron-3-nano-30b-a3b` by default. You can swap this for any supported model.

## Example Usage

### Running in Offline Mode

Offline mode uses predefined scenarios to simulate cluster issues without requiring a real Kubernetes cluster.

Three scenarios are included:
- **`node-not-ready`**: A worker node becomes unreachable, causing pod evictions.
- **`memory-pressure`**: Multiple pods are OOMKilled due to memory exhaustion on a worker node.
- **`healthy-cluster`**: Normal cluster operations with no issues.

```bash
# Investigate a node failure
nat run \
  --config_file=examples/advanced_agents/k8s_infra_monitor/configs/config_offline_mode.yml \
  --input '{"scenario_id": "node-not-ready", "query": "Worker node worker-2 appears to be down. Investigate the cluster health."}'
```

```bash
# Investigate OOMKilled pods
nat run \
  --config_file=examples/advanced_agents/k8s_infra_monitor/configs/config_offline_mode.yml \
  --input '{"scenario_id": "memory-pressure", "query": "Multiple pods are crashing in the ml-serving namespace. Check what is happening."}'
```

```bash
# Routine health check
nat run \
  --config_file=examples/advanced_agents/k8s_infra_monitor/configs/config_offline_mode.yml \
  --input '{"scenario_id": "healthy-cluster", "query": "Run a routine health check on the Kubernetes cluster."}'
```

To evaluate the agent across all scenarios:

```bash
nat eval --config_file=examples/advanced_agents/k8s_infra_monitor/configs/config_offline_mode.yml
```

### Running in Live Mode

Live mode connects to a real Kubernetes cluster using `kubectl`. Ensure your `KUBECONFIG` is set or specify `kubeconfig_path` in each tool's configuration.

```bash
# Run a live cluster health check
nat run \
  --config_file=examples/advanced_agents/k8s_infra_monitor/configs/config_live_mode.yml \
  --input "Check the overall health of the Kubernetes cluster. Are there any unhealthy pods or nodes under resource pressure?"
```

You can customize the live mode configuration to:
- Target specific namespaces with the `namespaces` list in `pod_health_check`.
- Adjust resource thresholds with `cpu_threshold_percent` and `memory_threshold_percent`.
- Point to a specific kubeconfig file with `kubeconfig_path`.
