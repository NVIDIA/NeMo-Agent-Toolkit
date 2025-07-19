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

<!--
  SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Testing Weave PII Redaction in AIQ using W&B Weave

This example demonstrates how to use Weights & Biases (W&B) Weave with PII redaction in your AIQ workflows.

## Key Features

- **PII Redaction Integration:** Demonstrates automatic redaction of Personally Identifiable Information (PII) using Microsoft Presidio within AIQ toolkit workflows for data privacy compliance.
- **Weave Observability Platform:** Shows integration with Weights & Biases Weave for detailed workflow tracking and visualization with privacy-preserving telemetry.
- **Configurable Entity Detection:** Supports redaction of multiple PII types including email addresses, phone numbers, credit cards, Social Security Numbers, and person names through configurable entity type selection.
- **Custom Key Redaction:** Demonstrates redaction of custom sensitive keys like API keys, auth tokens, and other application-specific secrets beyond standard PII entities.
- **Privacy-Preserving Monitoring:** Shows how to maintain comprehensive observability while ensuring sensitive data is automatically redacted from traces and logs.

## Prerequisites

1. Install the required dependencies:

```bash
pip install -e '.[weave]'
```

It will install the `presidio-analyzer` and `presidio-anonymizer` packages along side the `aiqtoolkit-weave` package.

2. Have a Weights & Biases account (sign up at https://wandb.ai if you don't have one)

## Example Files

- `weave_redact_pii_config.yaml`: Workflow configuration that enables Weave telemetry with PII redaction
- `examples/observability/redact_pii/src/redact_pii/register.py`: Contains the `pii_redaction_test` function that generates sample PII data

## Running the Example

1. An example weave config is provided in the `weave_redact_pii_config.yaml` file.

```yaml
telemetry:
  tracing:
    weave:
      _type: weave
      project: "aiqtoolkit-pii"
      redact_pii: true
      redact_pii_fields:
        - EMAIL_ADDRESS
        - PHONE_NUMBER
        - CREDIT_CARD
        - US_SSN
        - PERSON
      redact_keys:
        - custom_secret
        - api_key
        - auth_token
```

2. Run the workflow:

```bash
aiq run --config_file examples/observability/redact_pii/configs/weave_redact_pii_config.yml --input "Test query"
```

3. Go to your Weights & Biases dashboard (https://wandb.ai) and navigate to the "aiqtoolkit-pii" project.

4. Open the Weave trace viewer to see the redacted PII data. Look for:
   - Redacted email addresses (`EMAIL_ADDRESS`)
   - Redacted phone numbers (`PHONE_NUMBER`)
   - Redacted credit card information (`CREDIT_CARD`)
   - Redacted social security numbers (`US_SSN`)
   - Redacted person names (`PERSON`)
   - Redacted custom keys (`custom_secret`)

![Weave PII Redaction](images/redact_weave_trace.png)

## Customizing PII Redaction

You can customize what gets redacted by modifying these fields in the `weave_pii_test.yaml` file:

### Entity Types

The `redact_pii_fields` array specifies which PII entity types to redact. For a full list of the entities that can be detected and redacted, see PII entities supported by [Presidio](https://microsoft.github.io/presidio/supported_entities/).

```yaml
redact_pii_fields:
  - EMAIL_ADDRESS
  - PHONE_NUMBER
  - CREDIT_CARD
  - US_SSN
  - PERSON
  # Add other entity types as needed
```

### Custom Keys

The `redact_keys` array specifies additional keys to redact beyond the default ones:

```yaml
redact_keys:
  - custom_secret
  - api_key
  - auth_token
  # Add your own custom keys here
```
