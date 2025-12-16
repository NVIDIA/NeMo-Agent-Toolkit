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

![NVIDIA NeMo Agent Toolkit](https://media.githubusercontent.com/media/NVIDIA/NeMo-Agent-Toolkit/refs/heads/main/docs/source/_static/banner.png "NeMo Agent toolkit banner image")


# NVIDIA NeMo Agent Toolkit A2A Subpackage
Subpackage for A2A Protocol integration in NeMo Agent toolkit.

This package provides A2A (Agent-to-Agent) Protocol functionality, allowing NeMo Agent toolkit workflows to connect to remote A2A agents and invoke their skills as functions. This package includes both the client and server components of the A2A protocol.

## Features
### Client
- Connect to remote A2A agents via HTTP with JSON-RPC transport
- Discover agent capabilities through Agent Cards
- Submit tasks to remote agents with async execution

### Server
- Serve A2A agents via HTTP with JSON-RPC transport
- Support for A2A agent executor pattern

For more information about the NVIDIA NeMo Agent Toolkit, please visit the [NeMo Agent Toolkit GitHub Repo](https://github.com/NVIDIA/NeMo-Agent-Toolkit).
