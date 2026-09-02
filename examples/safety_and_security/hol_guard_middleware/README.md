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

# HOL Guard Security Middleware Example

**Complexity:** 🟡 Intermediate

This example demonstrates how to integrate [HOL Guard](https://github.com/hashgraph-online/hol-guard) as a security middleware in NeMo Agent Toolkit workflows. HOL Guard is an open-source, local-first runtime security layer for AI agents that can allow safe actions, block risky ones, or pause ambiguous actions for review.

## Key Features

- **Pre-execution security checks**: Validate actions before they reach the downstream function
- **Three decision types**: `allow`, `deny`, and `review`
- **Fail-closed design**: Guard errors or timeouts prevent execution
- **No cloud required**: HOL Guard runs locally by default

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Running the Example](#running-the-example)
- [Testing](#testing)

---

## Prerequisites

1. **NeMo Agent Toolkit** installed (see [Installation Guide](../../../docs/source/get-started/installation.md))
2. **HOL Guard CLI** installed:
   ```bash
   pip install hol-guard
   ```

## Installation

From the root directory of the NeMo Agent Toolkit repository:

```bash
pip install -e examples/safety_and_security/hol_guard_middleware
```

## How It Works

### Security Middleware Pattern

HOL Guard middleware wraps functions with a pre-execution security check:

```
User Request → [HOL Guard Check] → [Decision] → Function Execution
                                ↓
                         allow → Execute function
                         deny → Return blocked result (function NOT called)
                         review → Pause for approval
                         error → Fail closed (function NOT called)
```

### Configuration

The middleware is configured in YAML:

```yaml
middleware:
  hol_guard:
    _type: hol_guard
    # HOL Guard configuration options
    enabled: true

functions:
  my_function:
    _type: some_function_type
    middleware: ["hol_guard"]  # Apply security check
```

### Decision Semantics

| Decision | Behavior |
|----------|----------|
| `allow` | Execute the wrapped function exactly once |
| `deny` | Return/raise the blocked result without invoking the function |
| `review` | Pause before execution; requires explicit approval to continue |
| `error` | Fail closed with zero downstream execution |

## Running the Example

```bash
nat run --config_file examples/safety_and_security/hol_guard_middleware/configs/config.yml --input "Search for products"
```

## Testing

Run the tests to verify the middleware behavior:

```bash
uv run pytest examples/safety_and_security/hol_guard_middleware/tests/ -v
```

## Further Reading

- [HOL Guard Documentation](https://github.com/hashgraph-online/hol-guard)
- [NAT Middleware Documentation](../../../docs/source/build-workflows/advanced/middleware.md)
- [Third-Party Plugin Guide](../../../docs/source/extend/third-party-plugins.md)
