---
name: nat-telemetry
description: Use when adding, configuring, or troubleshooting NeMo Agent toolkit logging, tracing, telemetry exporters, OpenTelemetry, Langfuse, LangSmith, Weave, Phoenix, profiling, or observability provider integrations.
metadata:
  version: "0.1.0"
  status: initial
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# NeMo Agent toolkit Telemetry

Use this skill for workflow observability, tracing, logging, and profiling.

## Workflow

1. Discover registered logging and tracing exporters:

```bash
uv run nat info components -t logging
uv run nat info components -t tracing
```

2. Prefer built-in provider integrations when available.
3. Use file or console exporters for deterministic local debugging.
4. For custom exporters, adapt `references/otel_file_exporter.py` and nearby toolkit exporter code.

## References

- `references/telemetry.md`
- `references/otel_file_exporter.py`
