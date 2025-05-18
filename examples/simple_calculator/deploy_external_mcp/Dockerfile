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

FROM ubuntu:22.04

# Install Python and pip
RUN apt-get update && apt-get upgrade -y && apt install -y python3 python3-pip

# Install MCP proxy and tools
RUN pip3 install uv uvx

RUN pip3 install "mcp==1.5.0"
RUN pip3 install "mcp-proxy==0.5.1"

# Create directory for scripts
RUN mkdir /scripts

# Set the entrypoint to run the MCP proxy
ENTRYPOINT [ "mcp-proxy", "--pass-environment"]
