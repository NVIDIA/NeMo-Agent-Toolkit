# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Sandbox Agent Docker Image
# This image provides a pre-configured environment for sandbox execution
# with all necessary tools and libraries pre-installed.

FROM python:3.12-slim

# Metadata
LABEL maintainer="NVIDIA Corporation"
LABEL description="Sandbox Agent execution environment with Python, browser automation, and document generation tools"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Basic utilities
    curl \
    wget \
    git \
    jq \
    vim \
    unzip \
    # Network tools
    ca-certificates \
    # Browser dependencies (for Playwright)
    libnss3 \
    libnspr4 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    # Fonts for document generation and browser
    fonts-liberation \
    fonts-noto \
    fonts-noto-cjk \
    fonts-dejavu-core \
    # Document generation dependencies
    pandoc \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    # Data processing
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    # Network requests
    requests>=2.31.0 \
    httpx>=0.24.0 \
    beautifulsoup4>=4.12.0 \
    lxml>=4.9.0 \
    # Browser automation
    playwright>=1.40.0 \
    # Web search
    tavily-python>=0.5.0 \
    # YouTube transcript
    youtube-transcript-api>=0.6.0 \
    # Document generation
    python-pptx>=0.6.21 \
    python-docx>=1.1.0 \
    reportlab>=4.0.0 \
    markdown>=3.5.0 \
    # Excel support
    openpyxl>=3.1.0 \
    xlrd>=2.0.0 \
    # JSON/YAML
    pyyaml>=6.0.0 \
    # Image processing
    pillow>=10.0.0 \
    # Utilities
    aiofiles>=23.0.0 \
    python-dateutil>=2.8.0

# Create shared directory for Playwright browsers (accessible by all users)
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright-browsers
RUN mkdir -p ${PLAYWRIGHT_BROWSERS_PATH} && chmod 755 ${PLAYWRIGHT_BROWSERS_PATH}

# Install Playwright browser (Chromium only to reduce size)
RUN playwright install chromium --with-deps

# Create non-root user for security (created early so we can chown)
RUN useradd -m -s /bin/bash sandbox

# Change ownership of Playwright browsers to sandbox user
RUN chown -R sandbox:sandbox ${PLAYWRIGHT_BROWSERS_PATH}

# Create workspace directories
RUN mkdir -p /workspace/input \
             /workspace/output \
             /workspace/temp \
             /workspace/downloads

# Set working directory
WORKDIR /workspace

# Set workspace ownership to sandbox user
RUN chown -R sandbox:sandbox /workspace

# Switch to non-root user
USER sandbox

# Default command - keep container running
CMD ["/bin/bash"]
