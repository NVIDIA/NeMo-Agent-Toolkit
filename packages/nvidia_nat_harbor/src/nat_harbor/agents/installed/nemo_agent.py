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
"""NAT bridge for Harbor's NemoAgent behavior gaps.

Upstreaming note:
    This class subclasses Harbor's upstream `NemoAgent` and intentionally keeps
    only minimal local deltas (local install policy, python_bin override, and
    workflow package setup behavior) so changes can be upstreamed cleanly.
"""

from __future__ import annotations

import json
import shlex
import traceback
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.agents.installed.base import CliFlag
from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.nemo_agent import NemoAgent as HarborNemoAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from nat_harbor.agents.installed.inline_runner import DefaultNemoInlineRunner
from nat_harbor.agents.installed.library_mode import NemoInlineRunner
from nat_harbor.agents.installed.library_mode import NemoInlineRunnerInput
from nat_harbor.agents.installed.policy import is_local_install_allowed
from nat_harbor.agents.installed.policy import resolve_local_install_policy


class NemoAgent(HarborNemoAgent):
    """Bridge agent that layers local-mode and setup parity over Harbor 0.5.0."""

    CLI_FLAGS = [
        *HarborNemoAgent.CLI_FLAGS,
        # Comma-separated extra workflow packages to install before execution.
        CliFlag(
            "workflow_packages",
            cli="--workflow-packages",
            type="str",
            env_fallback="NVIDIA_NAT_WORKFLOW_PACKAGES",
        ),
        # Policy for whether host-local install is allowed in local mode.
        CliFlag(
            "local_install_policy",
            cli="--local-install-policy",
            type="enum",
            choices=["skip", "allow"],
            default="skip",
            env_fallback="HARBOR_LOCAL_INSTALL_POLICY",
        ),
        # Python executable used to invoke the NAT wrapper in shell mode.
        CliFlag(
            "python_bin",
            cli="--python-bin",
            type="str",
            default="python3",
            env_fallback="NVIDIA_NAT_PYTHON_BIN",
        ),
        # Enables inline (in-process) NAT execution instead of shell wrapper.
        CliFlag(
            "library_mode",
            cli="--library-mode",
            type="bool",
            default=False,
            env_fallback="NAT_HARBOR_LIBRARY_MODE",
        ),
    ]

    def __init__(self, *args, inline_runner: NemoInlineRunner | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._inline_runner = inline_runner or DefaultNemoInlineRunner()

    def _resolve_workflow_packages(self) -> list[str]:
        """Resolve workflow package install list from single + multi-package flags."""
        packages: list[str] = []

        single_package = self._resolved_flags.get("workflow_package")
        if single_package:
            packages.append(str(single_package).strip())

        multi_packages = self._resolved_flags.get("workflow_packages")
        if multi_packages:
            for package in str(multi_packages).split(","):
                candidate = package.strip()
                if candidate:
                    packages.append(candidate)

        deduped_packages: list[str] = []
        seen: set[str] = set()
        for package in packages:
            if package not in seen:
                seen.add(package)
                deduped_packages.append(package)

        return deduped_packages

    async def install(self, environment: BaseEnvironment) -> None:
        """Apply local install policy before delegating to Harbor install logic."""
        env_type_obj = environment.type()
        env_type = getattr(env_type_obj, "value", str(env_type_obj))
        local_install_policy = resolve_local_install_policy(self._resolved_flags.get("local_install_policy", "skip"))

        should_run_install = True
        if env_type == "local":
            should_run_install = is_local_install_allowed(local_install_policy)

        install_metadata = {
            "environment_type": env_type,
            "local_install_policy": local_install_policy if env_type == "local" else None,
            "local_install_allowed": should_run_install if env_type == "local" else None,
            "install_executed": should_run_install,
        }
        setup_dir = self.logs_dir / "setup"
        setup_dir.mkdir(parents=True, exist_ok=True)
        (setup_dir / "install-policy.json").write_text(
            json.dumps(install_metadata, indent=2),
            encoding="utf-8",
        )

        if not should_run_install:
            self.logger.warning("Skipping agent install in local mode (safe default). "
                                "Assuming dependencies are pre-provisioned.")
            return

        if env_type == "local":
            self.logger.warning("Local mode host install is enabled for this run. "
                                "Agent setup may mutate host packages/files.")
        await super().install(environment)

    async def setup(self, environment: BaseEnvironment) -> None:
        """Install base agent + wrapper and support multiple workflow packages."""
        await BaseInstalledAgent.setup(self, environment)

        wrapper_path = Path(__file__).parent / "nemo_agent_run_wrapper.py"
        await environment.upload_file(
            source_path=wrapper_path,
            target_path=self._CONTAINER_WRAPPER_PATH,
        )

        for index, workflow_package in enumerate(self._resolve_workflow_packages()):
            local_path = Path(workflow_package)
            if local_path.exists():
                container_pkg_dir = f"/installed-agent/workflow-package-{index}"
                await environment.upload_dir(
                    source_dir=local_path,
                    target_dir=container_pkg_dir,
                )
                result = await environment.exec(
                    command=('export PATH="/opt/nvidia-nat-venv/bin:$HOME/.local/bin:$PATH"; '
                             "SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 "
                             f"pip install --no-deps {shlex.quote(container_pkg_dir)}"),
                    env={"DEBIAN_FRONTEND": "noninteractive"},
                )
            else:
                result = await environment.exec(
                    command=('export PATH="/opt/nvidia-nat-venv/bin:$HOME/.local/bin:$PATH"; '
                             "uv pip install --python /opt/nvidia-nat-venv/bin/python "
                             f"{shlex.quote(workflow_package)}"),
                    env={"DEBIAN_FRONTEND": "noninteractive"},
                )
            if result.return_code != 0:
                raise RuntimeError(f"Failed to install workflow package '{workflow_package}': "
                                   f"{result.stderr or result.stdout}")

        config_file = self._resolved_flags.get("config_file")
        if config_file:
            host_config_path = Path(config_file)
            if not host_config_path.exists():
                raise FileNotFoundError(f"NeMo Agent Toolkit config file not found: {config_file}")
            await environment.upload_file(
                source_path=host_config_path,
                target_path=self._CONTAINER_CONFIG_PATH,
            )

    def _build_run_command(self, instruction: str) -> str:
        """Honor python_bin override while keeping Harbor command behavior."""
        run_command = super()._build_run_command(instruction)
        python_bin = shlex.quote(self._resolved_flags.get("python_bin", "python3"))
        wrapper_invocation = f"python3 {self._CONTAINER_WRAPPER_PATH}"
        if wrapper_invocation not in run_command:
            raise RuntimeError("Unable to apply python_bin override because the Harbor NemoAgent command "
                               f"does not contain the expected wrapper invocation: {wrapper_invocation}")

        return run_command.replace(wrapper_invocation, f"{python_bin} {self._CONTAINER_WRAPPER_PATH}", 1)

    def _resolve_inline_config_path(self) -> str:
        """Resolve or generate the config path used by inline workflow execution."""
        config_file = self._resolved_flags.get("config_file")
        if config_file:
            return str(config_file)

        api_key = self._resolve_api_key()
        model_name = self._resolve_model_name()
        config_yaml = self._generate_config_yaml(model_name, api_key)
        config_path = self.logs_dir / "nemo-agent-inline-config.yml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config_yaml, encoding="utf-8")
        return str(config_path)

    async def _sync_inline_outputs_to_environment(self,
                                                  environment: BaseEnvironment,
                                                  output_path: Path,
                                                  trajectory_path: Path,
                                                  stderr_path: Path) -> None:
        """Copy inline outputs to canonical Harbor paths used by existing verifiers."""
        target_files = [
            (output_path, "/app/answer.txt"),
            (output_path, "/app/result.json"),
            (output_path, "/workspace/answer.txt"),
            (output_path, "/workspace/solution.txt"),
            (output_path, "/app/response.txt"),
            (output_path, "/logs/agent/nemo-agent-output.txt"),
            (stderr_path, "/logs/agent/nemo-agent-stderr.txt"),
        ]
        if trajectory_path.exists():
            target_files.append((trajectory_path, "/logs/agent/trajectory.json"))

        for source_path, target_path in target_files:
            translated_target = target_path
            if hasattr(environment, "_translate_path"):
                translated_target = environment._translate_path(target_path)  # type: ignore[attr-defined]
            source_resolved = source_path.resolve()
            target_resolved = Path(translated_target).resolve()
            if source_resolved == target_resolved:
                continue
            await environment.upload_file(source_path=source_path, target_path=target_path)

    async def _run_library_mode(self, instruction: str, environment: BaseEnvironment) -> None:
        """Execute the Nemo workflow inline and write compatibility artifacts."""
        env_type_obj = environment.type()
        env_type = getattr(env_type_obj, "value", str(env_type_obj))
        if env_type != "local":
            raise RuntimeError("library_mode is currently supported only with local environment execution.")

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        config_path = self._resolve_inline_config_path()
        runner_result = await self._inline_runner.run(
            NemoInlineRunnerInput(
                instruction=instruction,
                config_file=config_path,
                artifact_dir=self.logs_dir,
                env=self._build_env(),
            ))

        output_path = self.logs_dir / "nemo-agent-output.txt"
        output_path.write_text(runner_result.output_text, encoding="utf-8")
        stderr_path = self.logs_dir / "nemo-agent-stderr.txt"
        stderr_path.write_text("", encoding="utf-8")

        await self._sync_inline_outputs_to_environment(
            environment=environment,
            output_path=output_path,
            trajectory_path=runner_result.trajectory_path,
            stderr_path=stderr_path,
        )

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        """Run Nemo agent using shell mode (default) or inline library mode."""
        if self._resolved_flags.get("library_mode"):
            try:
                await self._run_library_mode(instruction, environment)
            except Exception as exc:
                stderr_path = self.logs_dir / "nemo-agent-stderr.txt"
                stderr_path.parent.mkdir(parents=True, exist_ok=True)
                stderr_path.write_text(traceback.format_exc(), encoding="utf-8")
                raise RuntimeError(f"Inline library mode execution failed: {exc}") from exc
            return

        await super().run(instruction, environment, context)
