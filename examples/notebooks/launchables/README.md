<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# NVIDIA Brev Launchables for NeMo Agent Toolkit

Brev Launchables are an easy way to bundle a hardware and software environment into an easily shareable link, allowing for simple demos of GPU-powered software. Click **Deploy Now** to get started!

NeMo Agent Toolkit offers the following notebooks in Brev Launchable format:

1. [GPU Cluster Sizing with NeMo-Agent-Toolkit](./GPU_Cluster_Sizing_with_NeMo_Agent_Toolkit.ipynb) [![ Click here to deploy.](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://brev.nvidia.com/launchable/deploy?launchableID=env-31yFF6xbCKdp94CBlxHfspTjOn8)

    * This notebook demonstrates how to use the `NVIDIA NeMo Agent toolkit` sizing calculator to estimate the GPU cluster size required to accommodate a target number of users with a target response time. The estimation is based on the performance of the workflow at different concurrency levels.
    * The sizing calculator uses the [evaluation](https://docs.nvidia.com/nemo/agent-toolkit/latest/workflows/evaluate.html) and [profiling](https://docs.nvidia.com/nemo/agent-toolkit/latest/workflows/profiler.html) systems in the NeMo Agent toolkit.
