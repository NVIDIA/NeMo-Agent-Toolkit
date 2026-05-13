# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
from collections.abc import AsyncIterator
from collections.abc import Iterable
from typing import Any

import httpx
from art import dev
from art.backend import AnyModel
from art.backend import AnyTrainableModel
from art.trajectories import Trajectory
from art.trajectories import TrajectoryGroup
from art.types import TrainConfig
from art.types import TrainResult
from art.types import TrainSFTConfig

logger = logging.getLogger(__name__)

# `art run` serves a FastAPI app whose endpoints proxy LocalBackend methods.
# In openpipe-art 0.5.x `art.Backend` is a typing.Protocol and no longer
# instantiable; this client class fills that gap so a remote `art run` server
# can be used as a Backend implementation.
_DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=None, pool=None)


class RemoteBackend:
    """HTTP client that satisfies the art.Backend Protocol against `art run`."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=_DEFAULT_TIMEOUT)
        # Cached step from the most recent _prepare_backend_for_training response so
        # `_model_inference_name` can produce `name@step` like LocalBackend does.
        self._latest_step: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def _model_inference_name(self, model: AnyModel, step: int | None = None) -> str:
        if step is None:
            step = self._latest_step.get(model.name, 0)
        return f"{model.name}@{step}"

    async def close(self) -> None:
        try:
            await self._client.post("/close")
        finally:
            await self._client.aclose()

    async def register(self, model: AnyModel) -> None:
        body = model.model_dump(mode="json")
        # The server-side `register` only accepts JSON-serializable fields; drop
        # config to match TrainableModel.safe_model_dump behavior.
        body["config"] = None
        r = await self._client.post("/register", json=body)
        r.raise_for_status()

    async def _get_step(self, model: AnyTrainableModel) -> int:
        body = model.model_dump(mode="json")
        body["config"] = None
        r = await self._client.post("/_get_step", json=body)
        r.raise_for_status()
        return int(r.json())

    async def _delete_checkpoint_files(self, model: AnyTrainableModel, steps_to_keep: list[int]) -> None:
        body = {
            "model": _dump_model(model),
            "steps_to_keep": steps_to_keep,
        }
        r = await self._client.post("/_delete_checkpoint_files", json=body)
        r.raise_for_status()

    async def _prepare_backend_for_training(
        self,
        model: AnyTrainableModel,
        config: "dev.OpenAIServerConfig | None",
    ) -> tuple[str, str]:
        body = {
            "model": _dump_model(model),
            "config": dict(config) if config is not None else None,
        }
        r = await self._client.post("/_prepare_backend_for_training", json=body)
        r.raise_for_status()
        base_url, api_key = r.json()
        return base_url, api_key

    async def _train_model(
        self,
        model: AnyTrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        config: TrainConfig,
        dev_config: "dev.TrainConfig",
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        body = {
            "model": _dump_model(model),
            "trajectory_groups": [g.model_dump(mode="json") for g in trajectory_groups],
            "config": config.model_dump(mode="json"),
            "dev_config": dict(dev_config) if dev_config is not None else {},
            "verbose": verbose,
        }
        async with self._client.stream("POST", "/_train_model", json=body, timeout=None) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                yield json.loads(line)

    async def train(
        self,
        model: AnyTrainableModel,
        trajectory_groups: Iterable[TrajectoryGroup],
        **kwargs: Any,
    ) -> TrainResult:
        # `Backend.train` is the new high-level entry point; the NAT adapter
        # uses the legacy `Model.train` path that goes through `_train_model`,
        # so this is a thin shim to satisfy the Protocol.
        groups = list(trajectory_groups)
        last_metrics: dict[str, float] = {}
        async for metrics in self._train_model(model, groups, TrainConfig(**kwargs), {}, verbose=False):
            last_metrics = metrics
        return TrainResult(step=await self._get_step(model), metrics=last_metrics)

    def _train_sft(
        self,
        model: AnyTrainableModel,
        trajectories: Iterable[Trajectory],
        config: TrainSFTConfig,
        dev_config: "dev.TrainSFTConfig",
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        raise NotImplementedError("SFT over the remote ART HTTP API is not implemented.")


def _dump_model(model: AnyModel) -> dict:
    data = model.model_dump(mode="json")
    data["config"] = None
    return data
