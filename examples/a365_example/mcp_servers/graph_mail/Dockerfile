# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Graph Mail MCP Server — wraps Microsoft Graph API mail endpoints as MCP tools.
#
# Build from this directory:
#   docker build --platform linux/amd64 -t graph-mail-mcp:latest .
#
# Run:
#   docker run --rm -e GRAPH_MAIL_TOKEN=<bearer> -p 8100:8100 graph-mail-mcp:latest

FROM python:3.11-slim-bookworm

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /usr/local/bin/uv

ENV UV_NO_CACHE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY pyproject.toml server.py ./

RUN uv pip install --system -e .

EXPOSE 8100

CMD ["python", "server.py"]
