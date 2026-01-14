# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
ARG PYTHON_VERSION=3.13
# Specified on the command line with --build-arg NAT_VERSION=$(python -m setuptools_scm)
ARG NAT_VERSION

FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG}

ARG PYTHON_VERSION
ARG NAT_VERSION

COPY --from=ghcr.io/astral-sh/uv:0.9.15 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1

# Install compiler [g++, gcc] (currently only needed for thinc indirect dependency)
RUN apt-get update && \
    apt-get install -y --no-install-recommends g++ gcc curl unzip jq ca-certificates && \
    rm -rf /var/lib/apt/lists/*


# Install AWS CLI v2 (architecture-aware)
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
      curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"; \
    else \
      curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"; \
    fi && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws

# Verify installation
CMD ["aws", "--version"]

# Set working directory
WORKDIR /workspace

# Copy the project into the container
COPY ./ /workspace

# Install the nvidia-nat package and the example package
RUN --mount=type=cache,id=uv_cache,target=/root/.cache/uv,sharing=locked \
    test -n "${NAT_VERSION}" || { echo "NAT_VERSION build-arg is required" >&2; exit 1; } && \
    export SETUPTOOLS_SCM_PRETEND_VERSION=${NAT_VERSION} && \
    export SETUPTOOLS_SCM_PRETEND_VERSION_NVIDIA_NAT=${NAT_VERSION} && \
    export SETUPTOOLS_SCM_PRETEND_VERSION_NVIDIA_NAT_LANGCHAIN=${NAT_VERSION} && \
    export SETUPTOOLS_SCM_PRETEND_VERSION_NVIDIA_NAT_TEST=${NAT_VERSION} && \
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NAT_SIMPLE_CALCULATOR=${NAT_VERSION} && \
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NAT_STRANDS_DEMO=${NAT_VERSION} && \
    uv venv --python ${PYTHON_VERSION} /workspace/.venv && \
    uv sync --link-mode=copy --compile-bytecode --python ${PYTHON_VERSION} && \
    uv pip install -e '.[telemetry]' --link-mode=copy --compile-bytecode --python ${PYTHON_VERSION} && \
    uv pip install -e ./examples/frameworks/strands_demo --link-mode=copy && \
    uv pip install boto3 aws-opentelemetry-distro && \
    find /workspace/.venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /workspace/.venv -type f -name "*.pyc" -delete && \
    find /workspace/.venv -type f -name "*.pyo" -delete && \
    find /workspace/.venv -name "*.dist-info" -type d -exec rm -rf {}/RECORD {} + 2>/dev/null || true && \
    rm -rf /workspace/.venv/lib/python*/site-packages/pip /workspace/.venv/lib/python*/site-packages/setuptools

# AWS OpenTelemetry Distribution
ENV OTEL_PYTHON_DISTRO=aws_distro
#OTEL_PYTHON_CONFIGURATOR=aws_configurator

# Export Protocol
ENV OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
ENV OTEL_TRACES_EXPORTER=otlp

# Enable Agent Observability
ENV AGENT_OBSERVABILITY_ENABLED=true

# Service Identification attributed (gets added to all span logs)
# Example:
# OTEL_RESOURCE_ATTRIBUTES=service.version=1.0,service.name=mcp-calculator,aws.log.group.names=mcp/mcp-calculator-logs
ENV OTEL_RESOURCE_ATTRIBUTES=service.name=nat_test_agent,aws.log.group.names=/aws/bedrock-agentcore/runtimes/<AGENTCORE_RUNTIME_ID>

# CloudWatch Integration (ensure the log group and log stream are pre-created and exists)
# Example:
# OTEL_EXPORTER_OTLP_LOGS_HEADERS=x-aws-log-group=mcp/mcp-calculator-logs,x-aws-log-stream=default,x-aws-metric-namespace=mcp-calculator
ENV OTEL_EXPORTER_OTLP_LOGS_HEADERS=x-aws-log-group=/aws/bedrock-agentcore/runtimes/<AGENTCORE_RUNTIME_ID>,x-aws-log-stream=otel-rt-logs,x-aws-metric-namespace=strands_demo

# Remove build dependencies and cleanup (keep ca-certificates, curl, jq, unzip)
RUN apt-mark manual ca-certificates curl jq unzip && \
    apt-get purge -y --auto-remove g++ gcc && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /workspace/.git /workspace/.github /workspace/tests /workspace/docs && \
    find /workspace -type f -name "*.md" -not -path "*/site-packages/*" -delete && \
    find /workspace -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true && \
    find /workspace -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    
# Environment variables for the venv
ENV PATH="/workspace/.venv/bin:$PATH"

# Set the config file environment variable
ENV NAT_CONFIG_FILE=/workspace/examples/frameworks/strands_demo/configs/agentcore_config.yml

# Define the entry point to start the server
#ENTRYPOINT ["sh", "-c", "exec /workspace/examples/frameworks/strands_demo/bedrock_agentcore/scripts/run_nat_with_OTEL.sh"]
ENTRYPOINT ["sh", "-c", "exec /workspace/examples/frameworks/strands_demo/bedrock_agentcore/scripts/run_nat_no_OTEL.sh"]
