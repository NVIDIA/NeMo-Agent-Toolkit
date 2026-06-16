<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

- [`Dockerfile`](./Dockerfile)
- [build_and_push.sh](./build_and_push.sh)
- [deploy_phoenix.sh](./deploy_phoenix.sh)
