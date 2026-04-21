# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Permissive pass-through containers for the ATOF annotation layer.

ATOF core does not define the shape of ``annotated_request`` /
``annotated_response``. Each schema ID referenced by ``ScopeStart.schema`` /
``ScopeEnd.schema`` defines its own shape independently (see
``atof-schema-profiles.md``). These Pydantic classes are deliberately empty
containers with ``extra="allow"`` — they provide a typed slot on the event
model without constraining the payload shape. Consumers that want to validate
``annotated_*`` against a specific schema do so via the four-priority
resolution chain (``atof-schema-profiles.md`` §7.1), not through these
classes.

Used only as type hints on ``ScopeStartEvent.annotated_request`` and
``ScopeEndEvent.annotated_response`` in ``nat.atof.events``.
"""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict


class Request(BaseModel):
    """Permissive container for ``ScopeStart.annotated_request``.

    No required fields; any keys the producer emits pass through verbatim.
    Shape is declared externally by the event's ``schema`` reference, not
    by this class.
    """

    model_config = ConfigDict(extra="allow")


class Response(BaseModel):
    """Permissive container for ``ScopeEnd.annotated_response``.

    No required fields; any keys the producer emits pass through verbatim.
    Shape is declared externally by the event's ``schema`` reference, not
    by this class.
    """

    model_config = ConfigDict(extra="allow")
