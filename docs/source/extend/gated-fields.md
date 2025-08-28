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

# Gated Fields

Use {py:class}`~nat.data_models.gated_field_mixin.GatedFieldMixin` to gate configuration fields based on whether an analyzed field supports them. This enables provider-agnostic, model-aware validation with sensible defaults and clear errors.

## How It Works

- **Detection keys**: The mixin scans `keys` specified on the instance to identify values used to determine if the field is supported.
- **Selection modes**: Provide exactly one of the following when subclassing:
  - `unsupported`: A sequence of compiled regex patterns that mark the detector field where the mixin's field is not supported.
  - `supported`: A sequence of compiled regex patterns that mark the detector field where the mixin's field is supported.
- **Behavior**:
  - Supported and value not provided → sets `default_if_supported`.
  - Supported and value provided → keeps the provided value (and performs all other validations if defined).
  - Unsupported and value provided → raises a validation error.
  - Unsupported and value not provided → leaves the field as `None`.
  - No detection keys present → applies `default_if_supported`.

## Implementing a Gated Field

```python
import re
from pydantic import BaseModel, Field
from nat.data_models.gated_field_mixin import GatedFieldMixin

class FrequencyPenaltyMixin(
    BaseModel,
    GatedFieldMixin,
    field_name="frequency_penalty",
    default_if_supported=0.0,
    keys=("model_name", "model", "azure_deployment"),
    supported=(re.compile(r"^gpt-4.*$", re.IGNORECASE),),
):
    frequency_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
```

### Overriding Detection Keys

```python
class AzureOnlyMixin(
    BaseModel,
    GatedFieldMixin,
    field_name="some_param",
    default_if_supported=1,
    keys=("azure_deployment",),
    unsupported=(re.compile(r"gpt-?5", re.IGNORECASE),),
):
    some_param: int | None = Field(default=None)
    azure_deployment: str
```

## Built-in Gated Mixins

- {py:class}`~nat.data_models.temperature_mixin.TemperatureMixin`
  - Field: `temperature` in [0, 1]
  - Default when supported: `0.0`
  - Not supported on GPT-5 models

- {py:class}`~nat.data_models.top_p_mixin.TopPMixin`
  - Field: `top_p` in [0, 1]
  - Default when supported: `1.0`
  - Not supported on GPT-5 models

- {py:class}`~nat.data_models.thinking_mixin.ThinkingMixin`
  - Field: `thinking: bool | None`
  - Default when supported: `None` (use model default)
  - Only currently supported on Nemotron models

### Example: Integrating into a Provider Configuration

```python
from pydantic import BaseModel, Field
from nat.data_models.temperature_mixin import TemperatureMixin
from nat.data_models.top_p_mixin import TopPMixin

class MyProviderConfig(BaseModel, TemperatureMixin, TopPMixin):
    model_name: str = Field(...)
    # temperature and top_p are now validated and defaulted based on model support
```

## Best Practices

- Use `supported` for allowlist and `unsupported` for denylist; do not set both.
- Keep regex patterns specific (anchor with `^` and `$` when appropriate).
- If your config uses a non-standard model identifier field, set `keys` accordingly.
