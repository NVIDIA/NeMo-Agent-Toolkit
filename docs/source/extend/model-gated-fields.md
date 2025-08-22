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

# Model-Gated Configuration Fields

Use {py:class}`~nat.data_models.model_gated_field_mixin.ModelGatedFieldMixin` to gate configuration fields based on whether a model supports them. This enables provider-agnostic, model-aware validation with sensible defaults and clear errors.

## How It Works

- **Detection keys**: The mixin scans these keys on the model instance to identify the model by default: `model_name`, `model`, `azure_deployment`. You can override them with `model_keys`.
- **Selection modes**: Provide exactly one of the following when subclassing:
  - `unsupported_models`: A sequence of compiled regex patterns that mark models where the field is not supported.
  - `supported_models`: A sequence of compiled regex patterns that mark models where the field is supported.
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
from nat.data_models.model_gated_field_mixin import ModelGatedFieldMixin

_SUPPORTED = (re.compile(r"^gpt-4.*$", re.IGNORECASE),)

class FrequencyPenaltyMixin(
    BaseModel,
    ModelGatedFieldMixin[float],
    field_name="frequency_penalty",
    default_if_supported=0.0,
    supported_models=_SUPPORTED,
):
    frequency_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
```

### Overriding Detection Keys

```python
class AzureOnlyMixin(
    BaseModel,
    ModelGatedFieldMixin[int],
    field_name="some_param",
    default_if_supported=1,
    unsupported_models=(re.compile(r"gpt-?5", re.IGNORECASE),),
    model_keys=("azure_deployment",),
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

- Use `supported_models` for allowlist and `unsupported_models` for denylist; do not set both.
- Keep regex patterns specific (anchor with `^` and `$` when appropriate).
- If your config uses a non-standard model identifier field, set `model_keys` accordingly.


