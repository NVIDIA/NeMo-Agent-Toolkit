<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A365 Deployment Guide

This folder contains the deployment assets for the primary A365 worker example.

Use [../docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md) as the main deployment
guide. It covers:

- the canonical deployed config
- runtime secrets
- build and hosting targets
- Azure Bot endpoint alignment
- deployment validation

This folder then provides the assets referenced by that guide:

- [Dockerfile](./Dockerfile)
- [build_and_push.sh](./build_and_push.sh)
- [aca_rollout_mcp.sh](./aca_rollout_mcp.sh)
- [deploy_phoenix.sh](./deploy_phoenix.sh)
