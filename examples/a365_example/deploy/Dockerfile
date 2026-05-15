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
#
# A365 Bot (NAT front-end) — image built from this monorepo (A365 is not published to PyPI alone).
#
# Build from NeMo-Agent-Toolkit repository ROOT (linux/amd64 required for Azure Container Apps):
#   docker build --platform linux/amd64 -f examples/a365_example/deploy/Dockerfile -t nat-a365-bot:latest .
#
# Run (inject secrets at runtime; never bake into the image):
#   Worker + A365 telemetry (default CMD):
#     docker run --rm -e A365_APP_ID=... -e A365_APP_PASSWORD=... -e A365_BEARER_TOKEN=... -p 3978:3978 nat-a365-bot:latest
#   Front-end only: override CMD to use a365_frontend.yml — no A365_BEARER_TOKEN required.

FROM python:3.11-slim-bookworm

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /usr/local/bin/uv

ENV UV_NO_CACHE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

# Minimal tree for path-based `uv sync` from examples/a365_example (see that pyproject `[tool.uv.sources]`).
COPY pyproject.toml uv.lock README.md /workspace/
COPY packages /workspace/packages
COPY examples/a365_example/pyproject.toml /workspace/examples/a365_example/pyproject.toml
COPY examples/a365_example/README.md /workspace/examples/a365_example/README.md
COPY examples/a365_example/configs /workspace/examples/a365_example/configs

WORKDIR /workspace/examples/a365_example

# Root `nvidia-nat` uses setuptools-scm; `.git` is excluded from many Docker contexts, so pretend a version.
ARG SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+docker
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION}

# Passed by deploy/build_and_push.sh (`git rev-parse --short HEAD`) so runtime logs prove which sources were built.
ARG NAT_BUILD_STAMP=unknown
ENV NAT_BUILD_STAMP=${NAT_BUILD_STAMP}

RUN uv sync --no-dev

ENV PATH="/workspace/examples/a365_example/.venv/bin:$PATH"

WORKDIR /app
RUN cp -a /workspace/examples/a365_example/configs /app/configs

ENV ENVIRONMENT=Development

EXPOSE 3978

ENTRYPOINT ["nat"]
CMD ["start", "a365", "--config_file", "/app/configs/a365_worker.yml"]
