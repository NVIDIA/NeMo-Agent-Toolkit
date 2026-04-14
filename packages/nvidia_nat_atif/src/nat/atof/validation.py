# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON Schema validator wrapper for ATOF profile contracts.

Thin shim over the ``jsonschema`` library, used by ``ProfileContract`` to run
producer-side validation at profile construction (spec §4, D-15). Exposes a
single function ``validate_profile(payload, schema)`` that raises on
validation failure. Validators are cached per schema ``$id`` so that repeated
profile constructions reuse the same compiled validator.

Producers MAY swap validator implementations by re-assigning the module-level
``validate_profile`` callable; see spec §4 (Validation Contract, D-15).

See ATOF spec Section 4.
"""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

# Cache validators by schema $id (fallback: id() of the dict for inline schemas without $id).
# Per RESEARCH.md Open Question 3, unbounded growth is acceptable — schemas are small,
# author-controlled class attributes (no runtime fetch).
_VALIDATOR_CACHE: dict[str, Draft202012Validator] = {}


def _get_validator(schema: dict[str, Any]) -> Draft202012Validator:
    """Return a cached Draft 2020-12 validator for the given schema."""
    schema_id = schema.get("$id") or f"__anonymous__{id(schema)}"
    cached = _VALIDATOR_CACHE.get(schema_id)
    if cached is not None:
        return cached
    validator = Draft202012Validator(schema)
    _VALIDATOR_CACHE[schema_id] = validator
    return validator


def validate_profile(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    """Validate a profile payload against a JSON Schema (spec §4, D-15).

    Raises ``jsonschema.exceptions.ValidationError`` on failure. Producers that
    wrap this in a Pydantic ``model_validator`` will see the error re-raised
    as a ``pydantic.ValidationError`` (Pydantic wraps ``ValueError`` raised
    inside an after-validator).
    """
    validator = _get_validator(schema)
    validator.validate(payload)
