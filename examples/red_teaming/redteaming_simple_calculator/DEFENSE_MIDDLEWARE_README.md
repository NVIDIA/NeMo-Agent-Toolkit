<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Defense Middleware Example

This example demonstrates how to use defense middleware to protect workflows from attacks and incorrect outputs.

## Defense Middleware in Deployed Agents

Defense middleware is part of your **deployed agent configuration** (base config), not the red teaming evaluation config. This is because:

- **Defenses are permanent protections**: They protect your agent in production
- **Red teaming is temporary testing**: Attacks are only injected during security evaluation
- **Defenses wrap other middleware**: Since defense middleware is configured in the base workflow, it appears first in the middleware list for each function, which means it wraps other middleware (like red teaming middleware). This ensures defenses check outputs after any attacks are applied, providing the final security layer.

**Middleware Execution Order**: When multiple middleware are applied to a function, they execute in the order they appear in the `middleware` list. Since defense middleware is configured in the base workflow, it runs first (outermost layer), wrapping other middleware. This means:
1. Defense middleware wraps the function
2. Red teaming middleware (if present) wraps inside the defense middleware
3. Function executes
4. Red teaming middleware processes output (applies attacks)
5. Defense middleware processes output (detects and handles threats)

In this example:
- `base_multi_defense.yml` contains the agent configuration with defense middleware (what you deploy)
- `red_teaming_multi_defense.yml` contains attack scenarios for testing (evaluation only)

## Available Defense Middleware

### Output Verifier (`output_verifier`)
Uses an LLM to verify function outputs for:
- Correctness and reasonableness
- Security validation (detecting malicious content and manipulated values)
- Providing automatic corrections when errors are detected

Can log warnings, block execution, or auto-correct with the correct answer. The prompt structure is built-in and only requires a `tool_description` in the configuration.

### Content Safety Guard (`content_safety_guard`)
Uses safety guard models (e.g., NVIDIA Nemoguard, Qwen Guard) to detect harmful or unsafe content in function outputs. The middleware analyzes content independently and extracts safety categories when unsafe content is detected. Can log warnings, block execution, or return a user-friendly refusal message. Supports various guard models via NAT's LLM system (NIM, Hugging Face, etc.).

### PII Defense (`pii_defense`)
Uses Microsoft Presidio (https://github.com/microsoft/presidio) to detect and anonymize Personally Identifiable Information (PII) in function outputs. Configurable with specific entity types and score thresholds. Uses rule-based entity recognition (no LLM required).

## Defense Actions

All defense middleware support actions using safety domain terminology:

- **`partial_compliance`**: Detect and log threats, but allow content to pass through (monitoring mode)
- **`refusal`**: Detect and block threats, returning an error response (hard stop)
- **`redirection`**: Detect and replace threats with safe content
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
    target_function_or_group: my_calculator.multiply
    action: redirection
    threshold: 0.7
    tool_description: "Performs basic arithmetic operations (add, subtract, multiply, divide) on numbers"

  guard_defense:
    _type: content_safety_guard
    llm_name: guard_llm
    target_function_or_group: my_calculator
    action: refusal
    #
    # Examples for different guard models:
    #
    # NVIDIA Nemoguard (via NIM):
    # llms:
    #   guard_llm:
    #     _type: nim
    #     model_name: nvidia/llama-3.1-nemoguard-8b-content-safety
    #     temperature: 0.0
    #     max_tokens: 256
    #
    # Qwen Guard (via Hugging Face):
    # llms:
    #   guard_llm:
    #     _type: huggingface
    #     model_name: Qwen/Qwen3Guard-Gen-0.6B
    #     temperature: 0.0
    #     max_new_tokens: 128
    #     device: auto

  pii_defense:
    _type: pii_defense
    target_function_or_group: my_calculator
    action: redirection
    score_threshold: 0.5
    entities: [PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, US_SSN, LOCATION, IP_ADDRESS]
```

### Configuration Options

All defense middleware share these common options:

- **`action`**: Required. What to do when a threat is detected: `partial_compliance`, `refusal`, or `redirection`.
- **`target_function_or_group`**: Optional. Which functions to protect (for example, `my_calculator` or `my_calculator.multiply`). If not specified, defends all functions that have the middleware attached via the `middleware` list.

**Note**: Defense middleware only checks function **outputs** (not inputs). This ensures that any malicious or incorrect data produced by functions is detected and handled before it reaches the agent or end user.

**Targeting**: 
- Middleware is attached to functions through the `middleware` list in `function_groups` or individual `functions`
- The `target_function_or_group` field provides runtime filtering to target specific functions or groups
- If `target_function_or_group` is not specified, the defense applies to all functions that have the middleware in their middleware list

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


