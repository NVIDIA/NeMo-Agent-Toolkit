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
"""Inline verifier contracts and drivers for ATIF evaluation."""

from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path
from typing import Any
from typing import Protocol

from harbor.models.verifier.result import VerifierResult
from harbor.utils.env import resolve_env_vars
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from nat_harbor.verifier.evaluator_adapter import evaluate_artifact

DEFAULT_EVALUATOR_TIMEOUT_SEC = 600.0


class InlineVerifierRequest(BaseModel):
    """Request contract for phase-1 inline verifier execution."""
    model_config = ConfigDict(frozen=True)

    trajectory_path: Path
    evaluator_kind: str
    evaluator_ref: str | None
    config_file: str | None
    evaluator_name: str | None
    verifier_output_dir: Path
    fallback_mode: str = "fail"
    raw_output_path: Path = Field(default_factory=lambda: Path("/logs/agent/nemo-agent-output.txt"))
    evaluator_timeout_sec: float | None = DEFAULT_EVALUATOR_TIMEOUT_SEC


class InlineVerifierResult(BaseModel):
    """Result contract for phase-1 inline verifier execution."""
    model_config = ConfigDict(frozen=True)

    reward: float
    rewards: dict[str, float]
    details: dict[str, Any]
    reward_json_path: Path
    reward_txt_path: Path
    details_json_path: Path


class InlineVerifierDriver(Protocol):
    """Protocol for inline verifier implementations."""

    async def verify(self, request: InlineVerifierRequest) -> InlineVerifierResult:
        """Run inline verifier logic and return Harbor-compatible reward outputs."""
        ...


class InlineVerifierError(RuntimeError):
    """Raised when inline verifier execution fails and fallback is disabled."""


class DefaultInlineVerifierDriver:
    """Default inline verifier implementation with Harbor-compatible artifacts."""

    def _resolve_trajectory_path(self, trajectory_path: Path, verifier_output_dir: Path) -> Path:
        """Resolve artifact path across local and container verifier layouts."""
        if trajectory_path.is_absolute():
            return trajectory_path

        candidates = [
            verifier_output_dir.parent / "agent" / trajectory_path,
            Path("/logs/agent") / trajectory_path,
            Path("/workspace/agent") / trajectory_path,
            Path.cwd() / trajectory_path,
            trajectory_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _write_reward(self, output_dir: Path, reward: float) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        reward_json_path = output_dir / "reward.json"
        reward_txt_path = output_dir / "reward.txt"
        payload = {"reward": float(reward)}
        reward_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        reward_txt_path.write_text(f"{float(reward)}\n", encoding="utf-8")
        return reward_json_path, reward_txt_path

    def _details_path(self, output_dir: Path) -> Path:
        return output_dir / "details.json"

    def _write_details(self, output_dir: Path, details: dict[str, Any]) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        details_path = self._details_path(output_dir)
        details_path.write_text(json.dumps(details, indent=2, default=str), encoding="utf-8")
        return details_path

    def _raw_output_details(self, raw_output_path: Path) -> dict[str, Any]:
        raw_text = raw_output_path.read_text(encoding="utf-8").strip() if raw_output_path.exists() else ""
        return {
            "raw_output_path": str(raw_output_path),
            "raw_output_exists": raw_output_path.exists(),
            "raw_output_prefix": raw_text[:120] if raw_text else "",
        }

    def _metadata(
        self,
        *,
        evaluator_mode: str,
        request: InlineVerifierRequest,
        reward_json_path: Path,
        details_json_path: Path,
    ) -> dict[str, Any]:
        return build_inline_verifier_metadata(
            evaluator_mode=evaluator_mode,
            trajectory_path=request.trajectory_path,
            reward_json_path=reward_json_path,
            details_json_path=details_json_path,
        )

    async def _evaluate_artifact_with_timeout(
        self,
        *,
        request: InlineVerifierRequest,
        resolved_trajectory_path: Path,
    ) -> tuple[float, dict[str, Any]]:
        evaluator_task = evaluate_artifact(
            artifact_path=resolved_trajectory_path,
            evaluator_kind=request.evaluator_kind,
            evaluator_ref=request.evaluator_ref,
            config_file=request.config_file,
            evaluator_name=request.evaluator_name,
        )
        if request.evaluator_timeout_sec is None:
            return await evaluator_task
        return await asyncio.wait_for(evaluator_task, timeout=request.evaluator_timeout_sec)

    def _raw_output_fallback_result(
        self,
        *,
        request: InlineVerifierRequest,
        details: dict[str, Any],
        result: str,
    ) -> InlineVerifierResult:
        details["result"] = result
        details.update(self._raw_output_details(request.raw_output_path))
        reward_json_path, reward_txt_path = self._write_reward(request.verifier_output_dir, 0.0)
        details_json_path = self._details_path(request.verifier_output_dir)
        details["inline_metadata"] = self._metadata(
            evaluator_mode="raw_output_fallback",
            request=request,
            reward_json_path=reward_json_path,
            details_json_path=details_json_path,
        )
        details_json_path = self._write_details(request.verifier_output_dir, details)
        return InlineVerifierResult(
            reward=0.0,
            rewards={"reward": 0.0},
            details=details,
            reward_json_path=reward_json_path,
            reward_txt_path=reward_txt_path,
            details_json_path=details_json_path,
        )

    async def verify(self, request: InlineVerifierRequest) -> InlineVerifierResult:
        """Run evaluator dispatch inline and emit Harbor-compatible reward artifacts."""
        resolved_trajectory_path = self._resolve_trajectory_path(request.trajectory_path, request.verifier_output_dir)
        details: dict[str, Any] = {
            "artifact_path_input": str(request.trajectory_path),
            "artifact_path": str(resolved_trajectory_path),
            "artifact_exists": resolved_trajectory_path.exists(),
            "evaluator_kind": request.evaluator_kind,
            "evaluator_ref": request.evaluator_ref,
            "fallback_mode": request.fallback_mode,
            "evaluator_timeout_sec": request.evaluator_timeout_sec,
        }

        if request.fallback_mode not in {"fail", "raw_output"}:
            raise InlineVerifierError(
                f"Unsupported fallback mode '{request.fallback_mode}'. Supported values: fail, raw_output.")

        if not resolved_trajectory_path.exists():
            if request.fallback_mode == "raw_output":
                return self._raw_output_fallback_result(
                    request=request,
                    details=details,
                    result="raw_fallback_missing_artifact",
                )
            details["result"] = "missing_artifact_fail"
            self._write_details(request.verifier_output_dir, details)
            raise InlineVerifierError("ATIF artifact file is missing for inline verifier execution.")

        try:
            reward, evaluator_details = await self._evaluate_artifact_with_timeout(
                request=request,
                resolved_trajectory_path=resolved_trajectory_path,
            )
        except TimeoutError as exc:
            details["error"] = f"Inline verifier evaluator timed out after {request.evaluator_timeout_sec} seconds."
            details["error_type"] = type(exc).__name__
            details["traceback"] = traceback.format_exc()
            if request.fallback_mode == "raw_output":
                return self._raw_output_fallback_result(
                    request=request,
                    details=details,
                    result="raw_fallback_evaluator_timeout",
                )
            details["result"] = "evaluator_timeout_fail"
            self._write_details(request.verifier_output_dir, details)
            raise InlineVerifierError("Inline verifier evaluator timed out.") from exc
        except Exception as exc:
            details["error"] = str(exc)
            details["error_type"] = type(exc).__name__
            details["traceback"] = traceback.format_exc()
            if request.fallback_mode == "raw_output":
                return self._raw_output_fallback_result(
                    request=request,
                    details=details,
                    result="raw_fallback_evaluator_error",
                )
            details["result"] = "evaluator_error_fail"
            self._write_details(request.verifier_output_dir, details)
            raise InlineVerifierError("Inline verifier evaluator dispatch failed.") from exc

        details["result"] = "evaluated"
        details["reward"] = float(reward)
        details["evaluator_details"] = evaluator_details
        reward_json_path, reward_txt_path = self._write_reward(request.verifier_output_dir, reward)
        details_json_path = self._details_path(request.verifier_output_dir)
        details["inline_metadata"] = self._metadata(
            evaluator_mode=str(evaluator_details.get("evaluator_mode", "unknown")),
            request=request,
            reward_json_path=reward_json_path,
            details_json_path=details_json_path,
        )
        details_json_path = self._write_details(request.verifier_output_dir, details)
        return InlineVerifierResult(
            reward=float(reward),
            rewards={"reward": float(reward)},
            details=details,
            reward_json_path=reward_json_path,
            reward_txt_path=reward_txt_path,
            details_json_path=details_json_path,
        )


class ATIFInlineVerifier:
    """Harbor verifier class that executes NAT ATIF evaluation inline."""

    def __init__(
        self,
        task: Any,
        trial_paths: Any,
        environment: Any,
        override_env: dict[str, str] | None = None,
        logger: Any | None = None,
        verifier_env: dict[str, str] | None = None,
        step_name: str | None = None,
        driver: InlineVerifierDriver | None = None,
        **_: Any,
    ) -> None:
        del environment
        del step_name
        self._task = task
        self._trial_paths = trial_paths
        self._override_env = override_env or {}
        self._verifier_env = verifier_env or {}
        self._logger = logger
        self._driver: InlineVerifierDriver = driver or DefaultInlineVerifierDriver()

    def _resolve_runtime_env(self) -> dict[str, str]:
        task_verifier_env = getattr(self._task.config.verifier, "env", {}) or {}
        merged_env = {
            **task_verifier_env,
            **self._verifier_env,
            **self._override_env,
        }
        if not merged_env:
            return {}
        return resolve_env_vars(merged_env)

    @staticmethod
    def _none_if_empty(value: str | None) -> str | None:
        if value is None or value == "":
            return None
        return value

    @staticmethod
    def _evaluator_timeout_sec(value: str | None) -> float | None:
        if value is None:
            return DEFAULT_EVALUATOR_TIMEOUT_SEC
        if value == "":
            return None
        timeout_sec = float(value)
        if timeout_sec <= 0:
            return None
        return timeout_sec

    async def verify(self) -> VerifierResult:
        runtime_env = self._resolve_runtime_env()
        request = InlineVerifierRequest(
            trajectory_path=Path(runtime_env.get("NAT_HARBOR_ATIF_ARTIFACT_PATH", "trajectory.json")),
            evaluator_kind=runtime_env.get("NAT_HARBOR_ATIF_EVALUATOR_KIND", "custom"),
            evaluator_ref=self._none_if_empty(runtime_env.get("NAT_HARBOR_ATIF_EVALUATOR_REF")),
            config_file=self._none_if_empty(runtime_env.get("NAT_HARBOR_ATIF_CONFIG_FILE")),
            evaluator_name=self._none_if_empty(runtime_env.get("NAT_HARBOR_ATIF_EVALUATOR_NAME")),
            verifier_output_dir=self._trial_paths.verifier_dir,
            fallback_mode=runtime_env.get("NAT_HARBOR_ATIF_FALLBACK_MODE", "fail"),
            raw_output_path=Path(runtime_env.get("NAT_HARBOR_ATIF_RAW_OUTPUT_PATH",
                                                 "/logs/agent/nemo-agent-output.txt")),
            evaluator_timeout_sec=self._evaluator_timeout_sec(runtime_env.get("NAT_HARBOR_ATIF_EVALUATOR_TIMEOUT_SEC")),
        )
        result = await self._driver.verify(request)
        if self._logger:
            self._logger.debug("ATIF inline verifier completed with reward=%s", result.reward)
        return VerifierResult(rewards=result.rewards)


def verify_inline_sync(request: InlineVerifierRequest,
                       driver: InlineVerifierDriver | None = None) -> InlineVerifierResult:
    """Synchronously run an inline verifier request for CLI callers."""
    active_driver: InlineVerifierDriver = driver or DefaultInlineVerifierDriver()
    return asyncio.run(active_driver.verify(request))


def build_inline_verifier_metadata(
    *,
    evaluator_mode: str,
    trajectory_path: Path,
    reward_json_path: Path,
    details_json_path: Path,
) -> dict[str, Any]:
    """Build baseline metadata for phase-1 inline verifier traceability."""
    return {
        "inline_mode": True,
        "evaluator_mode": evaluator_mode,
        "trajectory_path": str(trajectory_path),
        "reward_json_path": str(reward_json_path),
        "details_json_path": str(details_json_path),
    }
