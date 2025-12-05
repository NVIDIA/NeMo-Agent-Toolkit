<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Defense Middleware Example

This example demonstrates how to use defense middleware to protect workflows from attacks and incorrect outputs.

## Available Defense Middleware

### Output Verifier (`output_verifier`)
Uses an LLM to verify function outputs for correctness and auto-correct errors.

### Content Safety Guard (`content_safety_guard`)
Uses safety guard models to classify and block harmful content.

## Configuration Example

```yaml
function_groups:
  my_calculator:
    _type: calculator
    middleware:
      - guard_defense
      - llm_defense

middleware:
  llm_defense:
    _type: output_verifier
    llm_name: verifier_llm
    target_function_or_group: my_calculator.multiply  # Target specific function
    action: block                                     # log, block, or sanitize
    threshold: 0.7                                    # Min confidence (0.0-1.0) to trigger action
    check_output: true

  guard_defense:
    _type: content_safety_guard
    llm_name: guard_llm
    target_function_or_group: my_calculator.get_random_string
    action: block
    severity_threshold: Unsafe
    check_output: true
```

## Running the Example

### Prerequisites

```bash
# Install simple calculator
cd examples/getting_started/simple_calculator
uv pip install -e .

# Install NAT from source
cd /path/to/AIQToolkit
uv pip install -e .

# Set API key
export NVIDIA_API_KEY="your-api-key"
```

### Test with Red Teaming

```bash
cd examples/red_teaming/redteaming_simple_calculator
nat red-team --red_team_config configs/red_teaming_multi_defense.yml
```

This will inject attacks and test if defenses successfully block them.

## Additional Resources

- [Defense Middleware Base Class](../../../src/nat/middleware/defense_middleware.py)
- [Output Verifier](../../../src/nat/middleware/defense_middleware_output_verifier.py)
- [Content Safety Guard](../../../src/nat/middleware/defense_middleware_content_guard.py)
- [Red Teaming Guide](./README.md)

