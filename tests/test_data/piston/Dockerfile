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

ARG PISTON_RUN_TIMEOUT=30000
ARG PISTON_RUN_CPU_TIME=30000

FROM ghcr.io/engineer-man/piston:latest

# Work-around for https://github.com/engineer-man/piston/issues/730
RUN mkdir /piston

ARG PISTON_RUN_TIMEOUT
ARG PISTON_RUN_CPU_TIME
ENV PISTON_RUN_TIMEOUT=${PISTON_RUN_TIMEOUT}
ENV PISTON_RUN_CPU_TIME=${PISTON_RUN_CPU_TIME}
