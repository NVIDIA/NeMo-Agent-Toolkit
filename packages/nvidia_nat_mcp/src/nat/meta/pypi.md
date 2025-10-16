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


# NVIDIA NeMo Agent Toolkit MCP Subpackage
Subpackage for MCP integration in NeMo Agent toolkit.

This package provides MCP (Model Context Protocol) functionality, allowing NeMo Agent toolkit workflows to connect to external MCP servers and use their tools as functions.

## Features

- Connect to MCP servers via streamable-http, SSE, or stdio transports
- Wrap individual MCP tools as NeMo Agent toolkit functions
- Connect to MCP servers and dynamically discover available tools

For more information about the NVIDIA NeMo Agent toolkit, please visit the [NeMo Agent toolkit GitHub Repo](https://github.com/NVIDIA/NeMo-Agent-Toolkit).
