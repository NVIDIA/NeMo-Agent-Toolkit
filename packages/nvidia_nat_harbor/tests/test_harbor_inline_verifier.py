# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for Harbor-facing ATIF inline verifier class."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from nat_harbor.verifier.inline_verifier import ATIFInlineVerifier
from nat_harbor.verifier.inline_verifier import InlineVerifierResult


class StubInlineDriver:
    """Capture request and return a deterministic verifier result."""

    def __init__(self) -> None:
        self.request = None

    async def verify(self, request):
        self.request = request
        output_dir = request.verifier_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return InlineVerifierResult(
            reward=1.0,
            rewards={"reward": 1.0},
            details={"status": "ok"},
            reward_json_path=output_dir / "reward.json",
            reward_txt_path=output_dir / "reward.txt",
            details_json_path=output_dir / "details.json",
        )


def _build_task(verifier_env: dict[str, str]):
    return SimpleNamespace(config=SimpleNamespace(verifier=SimpleNamespace(env=verifier_env)))


def _build_trial_paths(tmp_path: Path):
    return SimpleNamespace(verifier_dir=tmp_path / "verifier")


def test_atif_library_mode_verifier_builds_request_from_env_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATIF_KIND", "trajectory")
    driver = StubInlineDriver()
    verifier = ATIFInlineVerifier(
        task=_build_task({
            "NAT_HARBOR_ATIF_EVALUATOR_KIND": "${ATIF_KIND}",
            "NAT_HARBOR_ATIF_CONFIG_FILE": "config-from-task.yml",
        }),
        trial_paths=_build_trial_paths(tmp_path),
        environment=SimpleNamespace(),
        verifier_env={"NAT_HARBOR_ATIF_EVALUATOR_NAME": "eval-from-step"},
        override_env={
            "NAT_HARBOR_ATIF_ARTIFACT_PATH": "atif-output.json",
            "NAT_HARBOR_ATIF_FALLBACK_MODE": "raw_output",
            "NAT_HARBOR_ATIF_EVALUATOR_TIMEOUT_SEC": "12.5",
        },
        driver=driver,
    )

    result = asyncio.run(verifier.verify())

    assert result.rewards == {"reward": pytest.approx(1.0)}
    assert driver.request is not None
    assert driver.request.evaluator_kind == "trajectory"
    assert driver.request.config_file == "config-from-task.yml"
    assert driver.request.evaluator_name == "eval-from-step"
    assert driver.request.fallback_mode == "raw_output"
    assert driver.request.trajectory_path == Path("atif-output.json")
    assert driver.request.evaluator_timeout_sec == pytest.approx(12.5)


def test_atif_library_mode_verifier_can_disable_evaluator_timeout(tmp_path: Path) -> None:
    driver = StubInlineDriver()
    verifier = ATIFInlineVerifier(
        task=_build_task({}),
        trial_paths=_build_trial_paths(tmp_path),
        environment=SimpleNamespace(),
        override_env={"NAT_HARBOR_ATIF_EVALUATOR_TIMEOUT_SEC": "0"},
        driver=driver,
    )

    asyncio.run(verifier.verify())

    assert driver.request is not None
    assert driver.request.evaluator_timeout_sec is None


def test_atif_library_mode_verifier_supports_custom_evaluator_ref(tmp_path: Path, ) -> None:
    driver = StubInlineDriver()
    verifier = ATIFInlineVerifier(
        task=_build_task({}),
        trial_paths=_build_trial_paths(tmp_path),
        environment=SimpleNamespace(),
        override_env={
            "NAT_HARBOR_ATIF_EVALUATOR_KIND": "custom",
            "NAT_HARBOR_ATIF_EVALUATOR_REF": "pkg.module:evaluator_fn",
        },
        driver=driver,
    )

    asyncio.run(verifier.verify())

    assert driver.request is not None
    assert driver.request.evaluator_kind == "custom"
    assert driver.request.evaluator_ref == "pkg.module:evaluator_fn"
