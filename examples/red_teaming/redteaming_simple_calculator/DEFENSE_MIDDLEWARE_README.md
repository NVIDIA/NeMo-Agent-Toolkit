<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Defense Middleware Example

This example demonstrates how to use defense middleware to protect workflows from attacks and incorrect outputs.

## Defense Middleware in Deployed Agents

Defense middleware is part of your **deployed agent configuration** (base config), not the red teaming evaluation config. This is because:

- **Defenses are permanent protections**: They protect your agent in production
- **Red teaming is temporary testing**: Attacks are only injected during security evaluation
- **Defenses run first**: Defense middleware wraps your functions before red teaming attacks are applied

In this example:
- `base_multi_defense.yml` contains the agent configuration with defense middleware (what you deploy)
- `red_teaming_multi_defense.yml` contains attack scenarios for testing (evaluation only)

## Available Defense Middleware

### Output Verifier (`output_verifier`)
Uses an LLM to verify function outputs for correctness. Can log warnings, block execution, or auto-correct with the right answer.

### Content Safety Guard (`content_safety_guard`)
Uses safety guard models to detect harmful content. Can log warnings, block execution, or return a user-friendly refusal message.

### PII Defense (`pii_defense`)
Uses Microsoft Presidio to detect and anonymize Personally Identifiable Information (PII).

## Defense Actions

All defense middleware support three actions:

- **`log`**: Detect and log threats, but allow content to pass through (monitoring mode)
- **`block`**: Detect and block threats, returning an error response
- **`sanitize`**: Detect and replace threats with safe content
  - Output Verifier: Replaces with the correct calculated answer
  - Content Safety Guard: Replaces with a polite refusal message
  - PII Defense: Replaces PII with anonymized placeholders

## Configuration Example

```yaml
function_groups:
  my_calculator:
    _type: calculator
    middleware:
      - llm_defense
      - guard_defense
      - pii_defense

middleware:
  llm_defense:
    _type: output_verifier
    llm_name: nim_llm
    target_function_or_group: my_calculator.multiply  # Target specific function
    action: sanitize                                  # log, block, or sanitize
    threshold: 0.7                                    # Min confidence (0.0-1.0) to trigger action
    system_prompt: |
      Validate calculator output correctness. Return JSON:
      {
        "threat_detected": true/false,
        "confidence": 0.0-1.0,
        "reason": "brief explanation",
        "correct_answer": "correct numeric value if wrong, null if correct"
      }

  guard_defense:
    _type: content_safety_guard
    llm_name: guard_llm
    target_function_or_group: my_calculator.get_random_string
    action: block                                     # log, block, or sanitize

  pii_defense:
    _type: pii_defense
    target_function_or_group: my_calculator
    action: sanitize                                  # log, block, or sanitize
    score_threshold: 0.01                             # Min confidence (0.0-1.0) for PII detection
    entities: [PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, US_SSN, LOCATION, IP_ADDRESS]
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

# Install Presidio for PII defense (optional)
uv pip install "nvidia-nat[pii-defense]"

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
- [PII Defense](../../../src/nat/middleware/defense_middleware_pii.py)
- [Red Teaming Guide](./README.md)


