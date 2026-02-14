# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Health route registration."""

import logging

from fastapi import FastAPI
from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger(__name__)


async def add_health_route(app: FastAPI) -> None:
    """Add a health check endpoint to the FastAPI app."""

    class HealthResponse(BaseModel):
        status: str = Field(description="Health status of the server")

    async def health_check() -> HealthResponse:
        """Health check endpoint for liveness/readiness probes."""
        return HealthResponse(status="healthy")

    app.add_api_route(path="/health",
                      endpoint=health_check,
                      methods=["GET"],
                      response_model=HealthResponse,
                      description="Health check endpoint for liveness/readiness probes",
                      tags=["Health"],
                      responses={
                          200: {
                              "description": "Server is healthy",
                              "content": {
                                  "application/json": {
                                      "example": {
                                          "status": "healthy"
                                      }
                                  }
                              }
                          }
                      })

    logger.info("Added health check endpoint at /health")
