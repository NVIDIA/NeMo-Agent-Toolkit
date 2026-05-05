# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental Harbor wrapper for OpenCode with NeMo-Flow enabled."""

from __future__ import annotations

import copy
import os
import shlex
import shutil
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class OpenCodeNeMoFlow(OpenCode):
    """Run a local NeMo-Flow-patched OpenCode checkout inside Harbor.

    This is intentionally opt-in and experimental. It preserves Harbor's normal
    OpenCode JSONL capture while enabling NeMo-Flow sidecar artifacts.
    """

    _DEFAULT_CONTAINER_NEMO_FLOW_DIR = "/opt/nemo-flow"
    _DEFAULT_ATOF_DIR = "/logs/agent/nemo-flow-atof"
    _DEFAULT_ATIF_DIR = "/logs/agent/nemo-flow-atif"
    _CONVERTED_ATIF_DIR_NAME = "nemo-flow-atof-atif"
    _CONVERTED_ATIF_FILENAME = "trajectory.json"
    _DEFAULT_TASK_DIR = "/testbed"

    def __init__(
        self,
        *args: Any,
        nemo_flow_repo: str | None = None,
        container_nemo_flow_dir: str = _DEFAULT_CONTAINER_NEMO_FLOW_DIR,
        fail_missing_nemoflow_atof: bool = True,
        fail_missing_nemoflow_atif: bool = False,
        convert_nemoflow_atof: bool = True,
        fail_nemoflow_atof_conversion: bool = True,
        opencode_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        merged_config = self._deep_merge(
            {"experimental": {
                "nemo_flow": True
            }},
            copy.deepcopy(opencode_config or {}),
        )
        super().__init__(*args, opencode_config=merged_config, **kwargs)
        self._nemo_flow_repo = Path(nemo_flow_repo or os.environ.get("NEMO_FLOW_REPO", "external/nemo-flow")).resolve()
        self._container_nemo_flow_dir = container_nemo_flow_dir.rstrip("/")
        self._fail_missing_nemoflow_atof = fail_missing_nemoflow_atof
        self._fail_missing_nemoflow_atif = fail_missing_nemoflow_atif
        self._convert_nemoflow_atof = convert_nemoflow_atof
        self._fail_nemoflow_atof_conversion = fail_nemoflow_atof_conversion

    @staticmethod
    def name() -> str:
        return "opencode-nemoflow"

    @property
    def _container_opencode_root(self) -> str:
        return f"{self._container_nemo_flow_dir}/third_party/opencode"

    @property
    def _container_opencode_package(self) -> str:
        return f"{self._container_opencode_root}/packages/opencode"

    def _validate_local_nemo_flow_checkout(self) -> None:
        opencode_root = self._nemo_flow_repo / "third_party" / "opencode"
        node_root = self._nemo_flow_repo / "crates" / "node"
        required = [
            opencode_root / "package.json",
            opencode_root / "packages" / "opencode" / "src" / "nemo_flow" / "index.ts",
            opencode_root / "packages" / "opencode" / "src" / "plugin" / "nemo_flow.ts",
            node_root / "package.json",
            node_root / "index.js",
            node_root / "typed.js",
            node_root / "nemo-flow.linux-x64-gnu.node",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("NeMo-Flow OpenCode checkout is not ready. Missing:\n" +
                                    "\n".join(f"- {path}" for path in missing))

    def _prepare_upload_tree(self) -> Path:
        """Create a trimmed upload tree under the Harbor trial setup dir."""
        self._validate_local_nemo_flow_checkout()

        setup_dir = self.logs_dir / "setup" / "nemo-flow-upload"
        if setup_dir.exists():
            shutil.rmtree(setup_dir)
        setup_dir.mkdir(parents=True, exist_ok=True)

        ignore = shutil.ignore_patterns(
            ".git",
            "node_modules",
            "target",
            ".turbo",
            "coverage",
            "dist",
            "__pycache__",
        )
        shutil.copytree(
            self._nemo_flow_repo / "third_party" / "opencode",
            setup_dir / "third_party" / "opencode",
            ignore=ignore,
            dirs_exist_ok=True,
        )
        shutil.copytree(
            self._nemo_flow_repo / "crates" / "node",
            setup_dir / "crates" / "node",
            ignore=ignore,
            dirs_exist_ok=True,
        )
        return setup_dir

    async def install(self, environment: BaseEnvironment) -> None:
        if self.model_name and "/" in self.model_name:
            provider, _ = self.model_name.split("/", 1)
            self._build_provider_env(provider)

        upload_tree = self._prepare_upload_tree()

        await self.exec_as_root(
            environment,
            command=("command -v curl >/dev/null 2>&1 || "
                     "(apt-get update && apt-get install -y curl)"),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        await self.exec_as_root(
            environment,
            command=(f"mkdir -p {shlex.quote(self._container_nemo_flow_dir)} && "
                     f"chmod 777 {shlex.quote(self._container_nemo_flow_dir)}"),
        )
        await environment.upload_dir(
            source_dir=upload_tree,
            target_dir=self._container_nemo_flow_dir,
        )

        await self.exec_as_agent(
            environment,
            command=("set -euo pipefail; "
                     "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash && "
                     'export NVM_DIR="$HOME/.nvm" && '
                     '\\. "$NVM_DIR/nvm.sh" || true && '
                     "command -v nvm >/dev/null || { echo 'Error: NVM failed to load' >&2; exit 1; } && "
                     "nvm install 22 && "
                     "npm install -g bun@1.3.10 && "
                     "bun --version"),
            timeout_sec=600,
        )
        await self.exec_as_agent(
            environment,
            command=(". ~/.nvm/nvm.sh; "
                     f"cd {shlex.quote(self._container_opencode_root)} && "
                     "bun install --frozen-lockfile"),
            timeout_sec=1200,
        )

    def get_version_command(self) -> str | None:
        return ("set -euo pipefail; "
                ". ~/.nvm/nvm.sh; "
                f"cd {shlex.quote(self._container_opencode_package)}; "
                "bun run --conditions=browser ./src/index.ts --version")

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        escaped_instruction = shlex.quote(instruction)

        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")

        provider, _ = self.model_name.split("/", 1)
        env = self._build_provider_env(provider)
        env.update({
            "OPENCODE_FAKE_VCS": "git",
            "NEMO_FLOW_ENABLED": "1",
            "NEMO_FLOW_ATOF_DIR": self._DEFAULT_ATOF_DIR,
            "NEMO_FLOW_ATIF_DIR": self._DEFAULT_ATIF_DIR,
        })

        await self.exec_as_agent(
            environment,
            command=f"mkdir -p {self._DEFAULT_ATOF_DIR} {self._DEFAULT_ATIF_DIR}",
            env=env,
        )

        skills_command = self._build_register_skills_command()
        if skills_command:
            await self.exec_as_agent(environment, command=skills_command, env=env)

        mcp_command = self._build_register_config_command()
        if mcp_command:
            await self.exec_as_agent(environment, command=mcp_command, env=env)

        cli_flags = self.build_cli_flags()
        cli_flags_arg = (cli_flags + " ") if cli_flags else ""

        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                ". ~/.nvm/nvm.sh; "
                f"cd {shlex.quote(self._container_opencode_package)}; "
                f"bun run --conditions=browser ./src/index.ts run --model={shlex.quote(self.model_name)} --format=json "
                f"{cli_flags_arg}--thinking --dir {shlex.quote(self._DEFAULT_TASK_DIR)} -- {escaped_instruction} "
                "2>&1 </dev/null | stdbuf -oL tee /logs/agent/opencode.txt"),
            env=env,
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        events = self._parse_stdout()
        super().populate_context_post_run(context)

        if not events:
            return

        atof_path = self.logs_dir / "nemo-flow-atof" / "events.jsonl"
        atif_dir = self.logs_dir / "nemo-flow-atif"
        atif_files = sorted(atif_dir.glob("*.json")) if atif_dir.exists() else []
        converted_atif_path = self.logs_dir / self._CONVERTED_ATIF_DIR_NAME / self._CONVERTED_ATIF_FILENAME

        if self._fail_missing_nemoflow_atof and not atof_path.exists():
            raise FileNotFoundError(f"Missing NeMo-Flow ATOF JSONL artifact: {atof_path}")
        if self._fail_missing_nemoflow_atif and not atif_files:
            raise FileNotFoundError(f"Missing NeMo-Flow ATIF artifact under: {atif_dir}")

        if self._convert_nemoflow_atof and atof_path.exists():
            try:
                self._convert_atof_to_atif(atof_path, converted_atif_path)
            except Exception as exc:
                self.logger.exception("Failed to convert NeMo-Flow ATOF artifact to ATIF")
                if self._fail_nemoflow_atof_conversion:
                    raise RuntimeError(f"Failed to convert NeMo-Flow ATOF artifact to ATIF: {atof_path}") from exc

        self._record_nemoflow_artifacts(
            context,
            atof_path=atof_path,
            native_atif_files=atif_files,
            converted_atif_path=converted_atif_path,
        )

    def _convert_atof_to_atif(self, atof_path: Path, output_path: Path) -> Path:
        try:
            from nat.atof.scripts.atof_to_atif_converter import convert_file
        except ImportError as exc:
            raise RuntimeError("The ATOF-to-ATIF converter is unavailable. Install nvidia-nat-atif[full].") from exc

        trajectory = convert_file(atof_path, output_path)
        steps_count = len(getattr(trajectory, "steps", []) or [])
        self.logger.debug("Wrote ATOF-derived ATIF trajectory to %s with %s steps", output_path, steps_count)
        return output_path

    def _record_nemoflow_artifacts(
        self,
        context: AgentContext,
        *,
        atof_path: Path,
        native_atif_files: list[Path],
        converted_atif_path: Path,
    ) -> None:
        metadata = dict(context.metadata or {})
        metadata["nemo_flow_atof_path"] = str(atof_path)
        metadata["nemo_flow_atof_exists"] = atof_path.exists()
        metadata["nemo_flow_native_atif_paths"] = [str(path) for path in native_atif_files]
        metadata["nemo_flow_converted_atif_path"] = str(converted_atif_path)
        metadata["nemo_flow_converted_atif_exists"] = converted_atif_path.exists()
        context.metadata = metadata

    def _build_provider_env(self, provider: str) -> dict[str, str]:
        keys: list[str]
        if provider == "amazon-bedrock":
            keys = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"]
        elif provider == "anthropic":
            keys = ["ANTHROPIC_API_KEY"]
        elif provider == "azure":
            keys = ["AZURE_RESOURCE_NAME", "AZURE_API_KEY"]
        elif provider == "deepseek":
            keys = ["DEEPSEEK_API_KEY"]
        elif provider == "github-copilot":
            keys = ["GITHUB_TOKEN"]
        elif provider == "google":
            keys = [
                "GEMINI_API_KEY",
                "GOOGLE_GENERATIVE_AI_API_KEY",
                "GOOGLE_APPLICATION_CREDENTIALS",
                "GOOGLE_CLOUD_PROJECT",
                "GOOGLE_CLOUD_LOCATION",
                "GOOGLE_GENAI_USE_VERTEXAI",
                "GOOGLE_API_KEY",
            ]
        elif provider == "groq":
            keys = ["GROQ_API_KEY"]
        elif provider == "huggingface":
            keys = ["HF_TOKEN"]
        elif provider == "llama":
            keys = ["LLAMA_API_KEY"]
        elif provider == "mistral":
            keys = ["MISTRAL_API_KEY"]
        elif provider == "nvidia-frontier":
            keys = ["NVIDIA_FRONTIER_API_KEY", "NVIDIA_FRONTIER_BASE_URL"]
        elif provider == "nvidia":
            keys = ["NVIDIA_API_KEY", "NVIDIA_BASE_URL"]
        elif provider == "openai":
            keys = ["OPENAI_API_KEY", "OPENAI_BASE_URL"]
        elif provider == "opencode":
            keys = ["OPENCODE_API_KEY"]
        elif provider == "xai":
            keys = ["XAI_API_KEY"]
        elif provider == "openrouter":
            keys = ["OPENROUTER_API_KEY"]
        else:
            raise ValueError(f"Unknown provider {provider}. If you believe this provider "
                             "should be supported, please contact the maintainers.")

        env = {}
        for key in keys:
            value = self._get_env(key)
            if value is not None:
                env[key] = value

        required_key_by_provider = {
            "nvidia-frontier": "NVIDIA_FRONTIER_API_KEY",
            "nvidia": "NVIDIA_API_KEY",
        }
        required_key = required_key_by_provider.get(provider)
        if required_key and required_key not in env:
            raise ValueError(f"Provider {provider!r} requires {required_key} in the host environment or extra_env")

        return env
