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

# NVIDIA NeMo Agent Toolkit Release Notes
This section contains the release notes for [NeMo Agent Toolkit](./index.md).

## Release v1.9.0
### Summary

* Migrated Redis memory and object store support out of the NeMo Agent Toolkit repository and into the Redis-maintained [`nemo-agent-toolkit-redis`](https://pypi.org/project/nemo-agent-toolkit-redis/) plugin. The `nvidia-nat[redis]` extra, historical `nvidia-nat-redis` distribution, Python imports, and Redis component configuration names remain compatible through the no-code shim. New projects should install the external package directly. The external plugin requires `redis>=5.0.0,<6.0.0`; environments constrained to an earlier Redis Python client must update that constraint. See the [migration guide](./resources/migration-guide.md#redis-package-migration) for details.

## Release v1.8.0
### Summary

* Added Guardrails support
* Added Experimental coding-agent adapters with NeMo-Relay telemetry
* Added Microsoft Agent 365 integration plugin
* Added Windows WSL2 setup instructions

Refer to the [changelog](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.8/CHANGELOG.md) for the complete list of changes.

## Known Issues
- Refer to [https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues](https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues) for an up to date list of current issues.
