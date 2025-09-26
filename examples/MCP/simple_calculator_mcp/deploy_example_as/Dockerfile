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

# ---- Base OS ----
   FROM ubuntu:22.04

   # ---- System deps ----
   RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
         python3 python3-pip git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

   # Make python/pip the default names
   RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
   ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

   # ---- Install uv (to match your steps) ----
   RUN pip install --no-cache-dir uv

   # ---- Clone the MCP SDK repo ----
   WORKDIR /opt
   RUN git clone https://github.com/modelcontextprotocol/python-sdk.git mcp-sdk
   WORKDIR /opt/mcp-sdk
   RUN git checkout v1.14.0

   # ---- Install the sample auth server (editable) ----
   WORKDIR /opt/mcp-sdk/examples/servers/simple-auth
   RUN uv pip install --system -e .

   # ---- Network ----
   EXPOSE 9000

   # ---- Run the sample auth server ----
   ENTRYPOINT ["mcp-simple-auth-as", "--port=9000"]
