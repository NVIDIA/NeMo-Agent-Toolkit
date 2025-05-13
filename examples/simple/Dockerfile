# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG BASE_IMAGE_URL=nvcr.io/nvidia/base/ubuntu
ARG BASE_IMAGE_TAG=22.04_20240212
ARG PYTHON_VERSION=3.12

# Specified on the command line with --build-arg AIQ_VERSION=$(python -m setuptools_scm)
ARG AIQ_VERSION=0.0.1

FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG}
COPY --from=ghcr.io/astral-sh/uv:0.6.17 /uv /uvx /bin/
ARG AIQ_VERSION
ARG PYTHON_VERSION

ENV PYTHONDONTWRITEBYTECODE=1

# Install certificates
RUN apt-get update && \
    apt-get install -y ca-certificates curl && \
    update-ca-certificates

# Set SSL environment variables
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Set working directory
WORKDIR /workspace

# Copy the project into the container
COPY ./ /workspace

# Install the AIQ Toolkit package and the example package
RUN --mount=type=cache,id=uv_cache,target=/root/.cache/uv,sharing=locked \
    export SETUPTOOLS_SCM_PRETEND_VERSION=${AIQ_VERSION} && \
    export SETUPTOOLS_SCM_PRETEND_VERSION_AIQTOOLKIT=${AIQ_VERSION} && \
    export SETUPTOOLS_SCM_PRETEND_VERSION_AIQTOOLKIT_LANGCHAIN=${AIQ_VERSION} && \
    export SETUPTOOLS_SCM_PRETEND_VERSION_AIQTOOLKIT_TEST=${AIQ_VERSION} && \
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AIQ_SIMPLE=${AIQ_VERSION} && \
    uv venv --python ${PYTHON_VERSION} /workspace/.venv && \
    uv sync --link-mode=copy --compile-bytecode --python ${PYTHON_VERSION} && \
    uv pip install --link-mode=copy ./examples/simple

# Set the config file environment variable
ENV AIQ_CONFIG_FILE=/workspace/examples/simple/configs/eval_config.yml

# Enivronment variables for the venv
ENV PATH="/workspace/.venv/bin:$PATH"

# Define the entry point to start the server
ENTRYPOINT ["aiq", "serve", "--config_file=/workspace/examples/simple/configs/eval_config.yml", "--host", "0.0.0.0"]
